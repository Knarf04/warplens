/* A stand-in for libcudart: the CUDA runtime calls a compiled program makes are sent to
 * `warplens run`, which executes them in the simulator. Only what programs have needed so
 * far is here. Built against the installed CUDA headers; nothing from them is copied. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "protocol.h"

#define MAX_FUNCTIONS 512
#define MAX_PARAMS 64
#define MAX_ALLOCATIONS 4096

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int channel = -2;
static cudaError_t last_error = cudaSuccess;

struct function { const void *host; uint32_t id, count, size[MAX_PARAMS]; };
static struct function functions[MAX_FUNCTIONS];
static int function_count;

struct allocation { uint64_t base, size; };
static struct allocation allocations[MAX_ALLOCATIONS];
static int allocation_count;

static __thread struct { dim3 grid, block; size_t smem; cudaStream_t stream; } pending[8];
static __thread int pending_depth;

static int tracing = -1;
#define TRACE(...) do { \
    if (tracing < 0) tracing = getenv("WARPLENS_SHIM_DEBUG") != NULL; \
    if (tracing) { fprintf(stderr, "warplens shim: "); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } \
  } while (0)

static void die(const char *what) {
  fprintf(stderr, "warplens shim: %s\n", what);
  _exit(70);
}

static void connect_once(void) {
  if (channel != -2) return;
  const char *fd = getenv("WARPLENS_SHIM_FD");
  if (!fd) die("this program was built against the CUDA runtime; run it with `warplens run`");
  channel = atoi(fd);
}

static void write_all(const void *data, size_t size) {
  const char *p = data;
  while (size) {
    ssize_t n = write(channel, p, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("lost the connection to warplens");
    p += n; size -= (size_t)n;
  }
}

static void read_all(void *data, size_t size) {
  char *p = data;
  while (size) {
    ssize_t n = read(channel, p, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("lost the connection to warplens");
    p += n; size -= (size_t)n;
  }
}

int warplens_call(uint32_t op, const struct wl_part *parts, int count,
                  void *reply, size_t capacity, size_t *length) {
  pthread_mutex_lock(&lock);
  connect_once();
  uint32_t header[2] = {op, 0};
  for (int i = 0; i < count; ++i) header[1] += (uint32_t)parts[i].size;
  write_all(header, sizeof header);
  for (int i = 0; i < count; ++i) write_all(parts[i].data, parts[i].size);

  int32_t status; uint32_t size;
  read_all(&status, sizeof status);
  read_all(&size, sizeof size);
  char *body = malloc((size_t)size + 1);
  if (!body) die("out of memory");
  read_all(body, size);
  body[size] = 0;
  if (status != 0) {
    fprintf(stderr, "warplens: %s\n", body);
    last_error = (cudaError_t)status;
  } else if (reply) {
    memcpy(reply, body, size < capacity ? size : capacity);
  }
  if (length) *length = size;
  free(body);
  pthread_mutex_unlock(&lock);
  return status;
}

static cudaError_t result(int status) { return status ? (cudaError_t)status : cudaSuccess; }

/* -- modules and kernels: called by code the compiler generates ------------------------- */

struct fat_wrapper { int magic, version; const unsigned char *data; void *unused; };
struct fat_header { uint32_t magic; uint16_t version, header_size; uint64_t size; };

void **__cudaRegisterFatBinary(void *fatCubin) {
  const struct fat_wrapper *wrapper = fatCubin;
  const struct fat_header *header = (const struct fat_header *)wrapper->data;
  if (header->magic != 0xBA55ED50u) die("unrecognized fat binary");
  struct wl_part part = {wrapper->data, header->header_size + header->size};
  uint32_t *module = malloc(sizeof *module);
  *module = 0;
  if (warplens_call(WL_REGISTER_FATBIN, &part, 1, module, sizeof *module, NULL)) _exit(70);
  return (void **)module;
}

void __cudaRegisterFatBinaryEnd(void **handle) { (void)handle; }
void __cudaUnregisterFatBinary(void **handle) { (void)handle; }
void __cudaRegisterVar(void **handle, char *hostVar, char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global) {
  (void)handle; (void)hostVar; (void)deviceAddress; (void)deviceName;
  (void)ext; (void)size; (void)constant; (void)global;
}

void __cudaRegisterFunction(void **handle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
  if (function_count == MAX_FUNCTIONS) die("too many kernels");
  struct wl_part parts[2] = {{handle, sizeof(uint32_t)}, {deviceName, strlen(deviceName)}};
  uint32_t reply[2 + MAX_PARAMS];
  size_t length = 0;
  if (warplens_call(WL_REGISTER_FUNCTION, parts, 2, reply, sizeof reply, &length)) _exit(70);
  if (length > sizeof reply) die("a kernel has too many parameters");
  struct function *f = &functions[function_count++];
  f->host = hostFun; f->id = reply[0]; f->count = reply[1];
  memcpy(f->size, reply + 2, f->count * sizeof(uint32_t));
}

static struct function *find_function(const void *host) {
  for (int i = 0; i < function_count; ++i)
    if (functions[i].host == host) return &functions[i];
  return NULL;
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *hostFun) {
  TRACE("__cudaGetKernel(%p)", hostFun);
  *kernel = (cudaKernel_t)hostFun;
  return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                     struct CUstream_st *stream) {
  TRACE("__cudaPushCallConfiguration grid %u block %u", gridDim.x, blockDim.x);
  if (pending_depth == 8) die("launch configurations nested too deeply");
  pending[pending_depth].grid = gridDim; pending[pending_depth].block = blockDim;
  pending[pending_depth].smem = sharedMem; pending[pending_depth].stream = stream;
  ++pending_depth;
  return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
                                       void *stream) {
  TRACE("__cudaPopCallConfiguration depth %d", pending_depth);
  if (!pending_depth) return cudaErrorInvalidConfiguration;
  --pending_depth;
  *gridDim = pending[pending_depth].grid; *blockDim = pending[pending_depth].block;
  *sharedMem = pending[pending_depth].smem; *(cudaStream_t *)stream = pending[pending_depth].stream;
  return cudaSuccess;
}

static cudaError_t launch(const void *host, dim3 grid, dim3 block, dim3 cluster, size_t smem,
                          void **args) {
  struct function *f = find_function(host);
  if (!f) {
    TRACE("launch(%p)", host);
    fprintf(stderr, "warplens: launch of a kernel that was never registered\n");
    return last_error = cudaErrorInvalidDeviceFunction;
  }
  uint32_t head[10] = {f->id, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                       cluster.x, cluster.y, cluster.z};
  uint64_t dynamic = smem;
  struct wl_part parts[2 + MAX_PARAMS] = {{head, sizeof head}, {&dynamic, sizeof dynamic}};
  for (uint32_t i = 0; i < f->count; ++i) {
    parts[2 + i].data = args[i]; parts[2 + i].size = f->size[i];
  }
  size_t length = 0;
  char *text = malloc(1 << 20);  /* the reply is whatever the kernel printed */
  if (!text) die("out of memory");
  int status = warplens_call(WL_LAUNCH, parts, 2 + (int)f->count, text, (1 << 20) - 1, &length);
  if (!status && length) {
    if (length > (1 << 20) - 1) length = (1 << 20) - 1;
    fwrite(text, 1, length, stdout);
    fflush(stdout);
  }
  free(text);
  return result(status);
}

static const dim3 no_cluster = {0, 0, 0};

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim, void **args,
                               size_t sharedMem, cudaStream_t stream) {
  (void)stream;
  return launch((const void *)kernel, gridDim, blockDim, no_cluster, sharedMem, args);
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                             size_t sharedMem, cudaStream_t stream) {
  (void)stream;
  return launch(func, gridDim, blockDim, no_cluster, sharedMem, args);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
  dim3 cluster = no_cluster;
  for (unsigned i = 0; i < config->numAttrs; ++i)
    if (config->attrs[i].id == cudaLaunchAttributeClusterDimension) {
      cluster.x = config->attrs[i].val.clusterDim.x;
      cluster.y = config->attrs[i].val.clusterDim.y;
      cluster.z = config->attrs[i].val.clusterDim.z;
    }
  return launch(func, config->gridDim, config->blockDim, cluster, config->dynamicSmemBytes, args);
}

/* -- memory ---------------------------------------------------------------------------- */

static int is_device(const void *p) {
  uint64_t a = (uint64_t)p;
  for (int i = 0; i < allocation_count; ++i)
    if (a >= allocations[i].base && a < allocations[i].base + allocations[i].size) return 1;
  return 0;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  uint64_t request = size, address = 0;
  struct wl_part part = {&request, sizeof request};
  int status = warplens_call(WL_MALLOC, &part, 1, &address, sizeof address, NULL);
  if (!status) {
    if (allocation_count == MAX_ALLOCATIONS) die("too many allocations");
    allocations[allocation_count].base = address;
    allocations[allocation_count++].size = size ? size : 1;
    *devPtr = (void *)address;
  }
  return result(status);
}

cudaError_t cudaFree(void *devPtr) {
  if (!devPtr) return cudaSuccess;
  uint64_t address = (uint64_t)devPtr;
  struct wl_part part = {&address, sizeof address};
  for (int i = 0; i < allocation_count; ++i)
    if (allocations[i].base == address) { allocations[i] = allocations[--allocation_count]; break; }
  return result(warplens_call(WL_FREE, &part, 1, NULL, 0, NULL));
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  int to_device = is_device(dst), from_device = is_device(src);
  (void)kind;
  if (to_device && from_device) {
    uint64_t head[3] = {(uint64_t)dst, (uint64_t)src, count};
    struct wl_part part = {head, sizeof head};
    return result(warplens_call(WL_COPY, &part, 1, NULL, 0, NULL));
  }
  if (to_device) {
    uint64_t address = (uint64_t)dst;
    struct wl_part parts[2] = {{&address, sizeof address}, {src, count}};
    return result(warplens_call(WL_WRITE, parts, 2, NULL, 0, NULL));
  }
  if (from_device) {
    uint64_t head[2] = {(uint64_t)src, count};
    struct wl_part part = {head, sizeof head};
    return result(warplens_call(WL_READ, &part, 1, dst, count, NULL));
  }
  memmove(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
  (void)stream;
  TRACE("cudaMemcpyAsync(%zu bytes)", count);
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  uint64_t head[2] = {(uint64_t)devPtr, count};
  uint32_t byte = (uint32_t)value & 0xFF;
  struct wl_part parts[2] = {{head, sizeof head}, {&byte, sizeof byte}};
  return result(warplens_call(WL_FILL, parts, 2, NULL, 0, NULL));
}

/* -- everything is synchronous ----------------------------------------------------------- */

cudaError_t cudaDeviceSynchronize(void) { TRACE("cudaDeviceSynchronize -> %d", (int)last_error); return last_error; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; TRACE("cudaStreamSynchronize -> %d", (int)last_error); return last_error; }
cudaError_t cudaGetLastError(void) { cudaError_t e = last_error; TRACE("cudaGetLastError -> %d", (int)e); last_error = cudaSuccess; return e; }
cudaError_t cudaPeekAtLastError(void) { TRACE("cudaPeekAtLastError -> %d", (int)last_error); return last_error; }

const char *cudaGetErrorName(cudaError_t error) {
  TRACE("cudaGetErrorName(%d)", (int)error);
  switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    default: return "cudaErrorUnknown";
  }
}
const char *cudaGetErrorString(cudaError_t error) {
  static __thread char text[96];
  if (error == cudaSuccess) return "no error";
  snprintf(text, sizeof text, "error %d from the warplens CUDA runtime shim; see any warplens "
           "message above", (int)error);
  return text;
}

/* -- the device ------------------------------------------------------------------------ */

cudaError_t cudaGetDeviceCount(int *count) { *count = 1; return cudaSuccess; }
cudaError_t cudaGetDevice(int *device) { TRACE("cudaGetDevice"); *device = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int device) { return device == 0 ? cudaSuccess : cudaErrorInvalidDevice; }

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  TRACE("cudaGetDeviceProperties(%d)", device);
  if (device != 0) return cudaErrorInvalidDevice;
  memset(prop, 0, sizeof *prop);
  const char *model = getenv("WARPLENS_MODEL");
  snprintf(prop->name, sizeof prop->name, "warplens simulated %s", model ? model : "B200");
  prop->totalGlobalMem = (size_t)192 << 30;
  prop->sharedMemPerBlock = 48 * 1024;
  prop->sharedMemPerBlockOptin = 227 * 1024;
  prop->sharedMemPerMultiprocessor = 228 * 1024;
  prop->regsPerBlock = 65536;
  prop->regsPerMultiprocessor = 65536;
  prop->warpSize = 32;
  prop->maxThreadsPerBlock = 1024;
  prop->maxThreadsPerMultiProcessor = 2048;
  prop->maxThreadsDim[0] = 1024; prop->maxThreadsDim[1] = 1024; prop->maxThreadsDim[2] = 64;
  prop->maxGridSize[0] = 2147483647; prop->maxGridSize[1] = 65535; prop->maxGridSize[2] = 65535;
  prop->major = 10; prop->minor = 0;
  prop->multiProcessorCount = 148;
  prop->maxBlocksPerMultiProcessor = 32;
  return cudaSuccess;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  TRACE("cudaFuncGetAttributes(%p)", func);
  memset(attr, 0, sizeof *attr);
  attr->maxThreadsPerBlock = 1024;
  attr->maxDynamicSharedSizeBytes = 227 * 1024;
  attr->ptxVersion = 100;  /* the compute architecture (compute_100), which thrust dispatches on */
  attr->binaryVersion = 100;
  return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  (void)func; (void)attr; (void)value;  /* e.g. opting in to more shared memory: always allowed */
  return cudaSuccess;
}

/* -- driver API entry points ----------------------------------------------------------- */

/* Programs ask the runtime for driver functions by name (CUTLASS does, for
 * cuTensorMapEncodeTiled). The libcuda shim is preloaded next to this one, so whatever it
 * implements can be found among the loaded symbols. */
cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                             unsigned int cudaVersion, unsigned long long flags,
                                             enum cudaDriverEntryPointQueryResult *driverStatus) {
  (void)cudaVersion; (void)flags;
  void *found = dlsym(RTLD_DEFAULT, symbol);
  TRACE("cudaGetDriverEntryPointByVersion(%s) -> %p", symbol, found);
  *funcPtr = found;
  if (driverStatus)
    *driverStatus = found ? cudaDriverEntryPointSuccess : cudaDriverEntryPointSymbolNotFound;
  if (!found) fprintf(stderr, "warplens: the driver function %s is not implemented\n", symbol);
  return cudaSuccess;
}
