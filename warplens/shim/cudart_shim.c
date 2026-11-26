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
/* The last error a call returned, for cudaGetLastError and cudaPeekAtLastError; and the
 * error of a kernel that failed while running, which is what the synchronize calls report.
 * On a GPU an error a call returns (a bad argument, a refused launch) is not repeated by
 * cudaDeviceSynchronize; a kernel fault is. */
static __thread cudaError_t last_error = cudaSuccess;
static __thread cudaError_t kernel_error = cudaSuccess;

struct function { const void *host; uint32_t id, count, size[MAX_PARAMS]; };
static struct function functions[MAX_FUNCTIONS];
static int function_count;
static pthread_mutex_t function_lock = PTHREAD_MUTEX_INITIALIZER;

struct allocation { uint64_t base, size; };
static struct allocation allocations[MAX_ALLOCATIONS];
static int allocation_count;
static pthread_mutex_t allocation_lock = PTHREAD_MUTEX_INITIALIZER;

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
  for (int i = 0; i < count; ++i) {
    if (parts[i].size > UINT32_MAX - header[1]) {
      fprintf(stderr, "warplens: request exceeds the shim protocol's 32-bit payload limit\n");
      if (length) *length = 0;
      pthread_mutex_unlock(&lock);
      return cudaErrorInvalidValue;
    }
    header[1] += (uint32_t)parts[i].size;
  }
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
  } else if (reply) {
    memcpy(reply, body, size < capacity ? size : capacity);
  }
  if (length) *length = size;
  free(body);
  pthread_mutex_unlock(&lock);
  return status;
}

static cudaError_t result(int status) {
  if (status) last_error = (cudaError_t)status;
  return (cudaError_t)status;
}

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

static int register_function(uint32_t module, const char *name, const void *key) {
  struct wl_part parts[2] = {{&module, sizeof module}, {name, strlen(name)}};
  uint32_t reply[2 + MAX_PARAMS];
  size_t length = 0;
  int status = warplens_call(WL_REGISTER_FUNCTION, parts, 2, reply, sizeof reply, &length);
  if (status) return status;
  if (length > sizeof reply) die("a kernel has too many parameters");
  pthread_mutex_lock(&function_lock);
  if (function_count == MAX_FUNCTIONS) die("too many kernels");
  struct function *f = &functions[function_count];
  f->host = key; f->id = reply[0]; f->count = reply[1];
  memcpy(f->size, reply + 2, f->count * sizeof(uint32_t));
  ++function_count;
  pthread_mutex_unlock(&function_lock);
  return 0;
}

void __cudaRegisterFunction(void **handle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
  if (register_function(*(uint32_t *)handle, deviceName, hostFun)) _exit(70);
}

/* -- modules a program loads itself ------------------------------------------------------ */

struct kernel { struct kernel *next; char name[]; };
struct module { uint32_t id; struct kernel *kernels; };
static pthread_mutex_t module_lock = PTHREAD_MUTEX_INITIALIZER;

void *warplens_load_module(const void *image) {
  const unsigned char *bytes = image;
  struct wl_part part = {bytes, 0};
  if (memcmp(bytes, "\x7f" "ELF", 4) == 0) {
    part.size = 4;  /* machine code: the server only needs to know that it is */
  } else if (memcmp(bytes, "\x50\xed\x55\xba", 4) == 0) {
    const struct fat_header *header = image;
    part.size = header->header_size + header->size;
  } else {
    part.size = strlen((const char *)bytes);
  }
  struct module *loaded = calloc(1, sizeof *loaded);
  if (!loaded) die("out of memory");
  if (warplens_call(WL_REGISTER_MODULE, &part, 1, &loaded->id, sizeof loaded->id, NULL)) {
    free(loaded);
    return NULL;
  }
  return loaded;
}

void *warplens_module_kernel(void *module, const char *name) {
  struct module *loaded = module;
  pthread_mutex_lock(&module_lock);
  struct kernel *k = loaded->kernels;
  while (k && strcmp(k->name, name) != 0) k = k->next;
  if (!k) {
    k = malloc(sizeof *k + strlen(name) + 1);
    if (!k) die("out of memory");
    strcpy(k->name, name);
    if (register_function(loaded->id, name, k)) {
      free(k);
      k = NULL;
    } else {
      k->next = loaded->kernels;
      loaded->kernels = k;
    }
  }
  pthread_mutex_unlock(&module_lock);
  return k;
}

#if CUDART_VERSION >= 12080
/* The runtime's own module API (CUDA 12.8 on), which the CuTe DSL's host code uses. Options
 * steer the driver's compiler and loader; there is nothing for them to steer here. */
cudaError_t cudaLibraryLoadData(cudaLibrary_t *library, const void *code,
                                enum cudaJitOption *jitOptions, void **jitOptionsValues,
                                unsigned int numJitOptions, enum cudaLibraryOption *libraryOptions,
                                void **libraryOptionValues, unsigned int numLibraryOptions) {
  (void)jitOptions; (void)jitOptionsValues; (void)numJitOptions;
  (void)libraryOptions; (void)libraryOptionValues; (void)numLibraryOptions;
  if (!library || !code) return result(cudaErrorInvalidValue);
  *library = (cudaLibrary_t)warplens_load_module(code);
  return result(*library ? cudaSuccess : cudaErrorInvalidKernelImage);
}

cudaError_t cudaLibraryGetKernel(cudaKernel_t *kernel, cudaLibrary_t library, const char *name) {
  if (!kernel || !library || !name) return result(cudaErrorInvalidValue);
  *kernel = (cudaKernel_t)warplens_module_kernel(library, name);
  return result(*kernel ? cudaSuccess : cudaErrorSymbolNotFound);
}

/* The server keeps the parsed module, and its kernels stay launchable. */
cudaError_t cudaLibraryUnload(cudaLibrary_t library) { (void)library; return cudaSuccess; }
#endif

static struct function *find_function(const void *host) {
  struct function *found = NULL;
  pthread_mutex_lock(&function_lock);
  for (int i = 0; i < function_count && !found; ++i)
    if (functions[i].host == host) found = &functions[i];
  pthread_mutex_unlock(&function_lock);
  return found;
}

/* -- the shared memory a kernel may be launched with -------------------------------------
 * A kernel may use WL_DEFAULT_SMEM_BYTES of shared memory; a launch with more dynamic shared
 * memory than that leaves is refused (cudaErrorInvalidValue from the launch call) unless the
 * program raised the kernel's maximum first. The server keeps the maximum of each kernel and
 * knows its static shared memory: see `default_max_dynamic` in warplens/units/smem.py for the
 * rule, which was measured on an RTX 3080 Ti (validation/README.md, "Launch errors"). */

int warplens_set_function_attribute(const void *key, int attribute, int value) {
  struct function *f = find_function(key);
  if (!f) return cudaErrorInvalidDeviceFunction;
  if (!wl_function_attribute_is_valid(attribute, value)) return cudaErrorInvalidValue;
  if (attribute != WL_ATTRIBUTE_MAX_DYNAMIC_SMEM && attribute != WL_ATTRIBUTE_NONPORTABLE_CLUSTER) return cudaSuccess;
  int32_t body[3] = {(int32_t)f->id, attribute, value};
  struct wl_part part = {body, sizeof body};
  return warplens_call(WL_SET_ATTRIBUTE, &part, 1, NULL, 0, NULL);
}

int warplens_function_shared_memory(const void *key, int *fixed, int *max_dynamic) {
  struct function *f = find_function(key);
  if (!f) return cudaErrorInvalidDeviceFunction;
  uint32_t reply[2] = {0, 0};
  struct wl_part part = {&f->id, sizeof f->id};
  int status = warplens_call(WL_GET_ATTRIBUTES, &part, 1, reply, sizeof reply, NULL);
  *fixed = (int)reply[0]; *max_dynamic = (int)reply[1];
  return status;
}

cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel, enum cudaFuncAttribute attr,
                                            int value, int device) {
  TRACE("cudaKernelSetAttributeForDevice(%d, %d, device %d)", (int)attr, value, device);
  if (device != 0) return result(cudaErrorInvalidDevice);
  return result(warplens_set_function_attribute((const void *)kernel, (int)attr, value));
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
  if (!pending_depth) return result(cudaErrorInvalidConfiguration);
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
  if (status && status != WL_INVALID_VALUE && status != WL_INVALID_CONFIGURATION &&
      status != WL_INVALID_CLUSTER_SIZE)
    kernel_error = (cudaError_t)status;
  /* A block or grid the launch refuses: the runtime of CUDA 13 on a B200 passes on the
   * driver's "invalid value" (validation/blackwell/measured/b200/e2e/launch_errors.txt). The
   * runtime of CUDA 12.4 on an RTX 3080 Ti said cudaErrorInvalidConfiguration for the same six
   * launches (validation/ampere/launch_errors_3080ti.txt); this shim stands in for the first. */
  return result(status == WL_INVALID_CONFIGURATION ? cudaErrorInvalidValue : status);
}

static const dim3 no_cluster = {0, 0, 0};

int warplens_launch(const void *key, const uint32_t grid[3], const uint32_t block[3],
                    const uint32_t cluster[3], uint64_t shared_memory, void **arguments) {
  dim3 g = {grid[0], grid[1], grid[2]}, b = {block[0], block[1], block[2]};
  dim3 c = {cluster[0], cluster[1], cluster[2]};
  return launch(key, g, b, c, shared_memory, arguments);
}

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
  int found = 0;
  pthread_mutex_lock(&allocation_lock);
  for (int i = 0; i < allocation_count; ++i)
    if (a >= allocations[i].base && a < allocations[i].base + allocations[i].size) {
      found = 1;
      break;
    }
  pthread_mutex_unlock(&allocation_lock);
  return found;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  uint64_t request = size, address = 0;
  struct wl_part part = {&request, sizeof request};
  int status = warplens_call(WL_MALLOC, &part, 1, &address, sizeof address, NULL);
  if (!status) {
    pthread_mutex_lock(&allocation_lock);
    if (allocation_count == MAX_ALLOCATIONS) die("too many allocations");
    allocations[allocation_count].base = address;
    allocations[allocation_count++].size = size ? size : 1;
    pthread_mutex_unlock(&allocation_lock);
    *devPtr = (void *)address;
  }
  return result(status);
}

cudaError_t cudaFree(void *devPtr) {
  if (!devPtr) return cudaSuccess;
  uint64_t address = (uint64_t)devPtr;
  struct wl_part part = {&address, sizeof address};
  int status = warplens_call(WL_FREE, &part, 1, NULL, 0, NULL);
  if (!status) {
    pthread_mutex_lock(&allocation_lock);
    for (int i = 0; i < allocation_count; ++i)
      if (allocations[i].base == address) { allocations[i] = allocations[--allocation_count]; break; }
    pthread_mutex_unlock(&allocation_lock);
  }
  return result(status);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  /* Only cudaMemcpyDefault infers the direction. Explicit device pointers must
   * reach the server even when stale or out of bounds, never host memmove. */
  if (kind == cudaMemcpyDefault) {
    int to_device = is_device(dst), from_device = is_device(src);
    kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                     : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
  }
  if (kind == cudaMemcpyDeviceToDevice) {
    uint64_t head[3] = {(uint64_t)dst, (uint64_t)src, count};
    struct wl_part part = {head, sizeof head};
    return result(warplens_call(WL_COPY, &part, 1, NULL, 0, NULL));
  }
  if (kind == cudaMemcpyHostToDevice) {
    uint64_t address = (uint64_t)dst;
    struct wl_part parts[2] = {{&address, sizeof address}, {src, count}};
    return result(warplens_call(WL_WRITE, parts, 2, NULL, 0, NULL));
  }
  if (kind == cudaMemcpyDeviceToHost) {
    uint64_t head[2] = {(uint64_t)src, count};
    struct wl_part part = {head, sizeof head};
    return result(warplens_call(WL_READ, &part, 1, dst, count, NULL));
  }
  if (kind != cudaMemcpyHostToHost) return result(cudaErrorInvalidMemcpyDirection);
  memmove(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
  (void)stream;
  TRACE("cudaMemcpyAsync(%zu bytes)", count);
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                                size_t count, cudaStream_t stream) {
  (void)stream;
  if (dstDevice != 0 || srcDevice != 0) return result(cudaErrorInvalidDevice);
  return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

/* Host memory needs no pinning here: every transfer goes through the server anyway. */
cudaError_t cudaMallocHost(void **ptr, size_t size) {
  *ptr = malloc(size ? size : 1);
  return result(*ptr ? cudaSuccess : cudaErrorMemoryAllocation);
}
cudaError_t cudaFreeHost(void *ptr) { free(ptr); return cudaSuccess; }

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  uint64_t head[2] = {(uint64_t)devPtr, count};
  uint32_t byte = (uint32_t)value & 0xFF;
  struct wl_part parts[2] = {{head, sizeof head}, {&byte, sizeof byte}};
  return result(warplens_call(WL_FILL, parts, 2, NULL, 0, NULL));
}

/* -- everything is synchronous ----------------------------------------------------------- */

cudaError_t cudaDeviceSynchronize(void) { TRACE("cudaDeviceSynchronize -> %d", (int)kernel_error); return kernel_error; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; TRACE("cudaStreamSynchronize -> %d", (int)kernel_error); return kernel_error; }
cudaError_t cudaGetLastError(void) {
  cudaError_t e = last_error;
  TRACE("cudaGetLastError -> %d", (int)e);
  /* The simulated context survives a failed kernel, so reading the error clears that too. */
  last_error = kernel_error = cudaSuccess;
  return e;
}
cudaError_t cudaPeekAtLastError(void) { TRACE("cudaPeekAtLastError -> %d", (int)last_error); return last_error; }

/* Streams and events are tokens: work finishes before the call that started it returns, and
 * the simulator's clock is not wall time, so an elapsed time is always zero. */
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  (void)flags;
  *stream = malloc(1);
  return result(*stream ? cudaSuccess : cudaErrorMemoryAllocation);
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) { free(stream); return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  (void)stream; (void)event; (void)flags;
  return cudaSuccess;
}
cudaError_t cudaEventCreate(cudaEvent_t *event) {
  *event = malloc(1);
  return result(*event ? cudaSuccess : cudaErrorMemoryAllocation);
}
cudaError_t cudaEventDestroy(cudaEvent_t event) { free(event); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { (void)event; (void)stream; return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t event) { (void)event; return kernel_error; }
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  (void)start; (void)end;
  *ms = 0.0f;
  return cudaSuccess;
}

const char *cudaGetErrorName(cudaError_t error) {
  TRACE("cudaGetErrorName(%d)", (int)error);
  switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
    case cudaErrorSymbolNotFound: return "cudaErrorSymbolNotFound";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    case cudaErrorInvalidClusterSize: return "cudaErrorInvalidClusterSize";
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
cudaError_t cudaSetDevice(int device) { return result(device == 0 ? cudaSuccess : cudaErrorInvalidDevice); }

/* NVIDIA's CUDA GPU list gives B200 CC 10.0 and B300 CC 10.3.
 * https://developer.nvidia.com/cuda/gpus */
static int is_b300(void) {
  const char *model = getenv("WARPLENS_MODEL");
  return model && strcmp(model, "B300") == 0;
}
static size_t total_memory(void) { return (size_t)(is_b300() ? 288 : 192) << 30; }

int warplens_device_attribute(int attribute) {
  switch (attribute) {
    case cudaDevAttrMaxThreadsPerBlock: return 1024;
    case cudaDevAttrMaxBlockDimX: case cudaDevAttrMaxBlockDimY: return 1024;
    case cudaDevAttrMaxBlockDimZ: return 64;
    case cudaDevAttrMaxGridDimX: return 2147483647;
    case cudaDevAttrMaxGridDimY: case cudaDevAttrMaxGridDimZ: return 65535;
    case cudaDevAttrMaxSharedMemoryPerBlock: return WL_DEFAULT_SMEM_BYTES;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: return WL_MAX_SMEM_BYTES;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: return 228 * 1024;
    case cudaDevAttrMaxRegistersPerBlock: case cudaDevAttrMaxRegistersPerMultiprocessor: return 65536;
    case cudaDevAttrWarpSize: return 32;
    case cudaDevAttrMultiProcessorCount: return is_b300() ? 160 : 148;  /* sm_count of configs/b200.yaml, b300.yaml */
    case cudaDevAttrMaxBlocksPerMultiprocessor: return 32;
    case cudaDevAttrComputeCapabilityMajor: return 10;
    case cudaDevAttrComputeCapabilityMinor: return is_b300() ? 3 : 0;
    /* Also CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH (same number). The simulator launches clusters
     * (cudaLaunchKernelEx with cudaLaunchAttributeClusterDimension), so programs that ask
     * before using them must be told so. */
    case cudaDevAttrClusterLaunch: return 1;
    default: TRACE("device attribute %d is not described; answering 0", attribute); return 0;
  }
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
  if (device != 0) return result(cudaErrorInvalidDevice);
  *value = warplens_device_attribute((int)attr);
  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  if (free_bytes) *free_bytes = total_memory();
  if (total_bytes) *total_bytes = total_memory();
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  TRACE("cudaGetDeviceProperties(%d)", device);
  if (device != 0) return result(cudaErrorInvalidDevice);
  memset(prop, 0, sizeof *prop);
  const char *model = getenv("WARPLENS_MODEL");
  snprintf(prop->name, sizeof prop->name, "warplens simulated %s", model ? model : "B200");
  prop->totalGlobalMem = total_memory();
  prop->sharedMemPerBlock = (size_t)warplens_device_attribute(cudaDevAttrMaxSharedMemoryPerBlock);
  prop->sharedMemPerBlockOptin = (size_t)warplens_device_attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin);
  prop->sharedMemPerMultiprocessor = (size_t)warplens_device_attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  prop->regsPerBlock = warplens_device_attribute(cudaDevAttrMaxRegistersPerBlock);
  prop->regsPerMultiprocessor = warplens_device_attribute(cudaDevAttrMaxRegistersPerMultiprocessor);
  prop->warpSize = warplens_device_attribute(cudaDevAttrWarpSize);
  prop->maxThreadsPerBlock = warplens_device_attribute(cudaDevAttrMaxThreadsPerBlock);
  prop->maxThreadsPerMultiProcessor = 2048;
  for (int i = 0; i < 3; ++i) {
    prop->maxThreadsDim[i] = warplens_device_attribute(cudaDevAttrMaxBlockDimX + i);
    prop->maxGridSize[i] = warplens_device_attribute(cudaDevAttrMaxGridDimX + i);
  }
  prop->major = warplens_device_attribute(cudaDevAttrComputeCapabilityMajor);
  prop->minor = warplens_device_attribute(cudaDevAttrComputeCapabilityMinor);
  prop->multiProcessorCount = warplens_device_attribute(cudaDevAttrMultiProcessorCount);
  prop->maxBlocksPerMultiProcessor = warplens_device_attribute(cudaDevAttrMaxBlocksPerMultiprocessor);
  prop->clusterLaunch = warplens_device_attribute(cudaDevAttrClusterLaunch);
  return cudaSuccess;
}

/* A function that is no kernel of the program has no attributes (cudaErrorInvalidDevice-
 * Function, measured). */
/* An attribute asked of a function that is not a kernel: cudaErrorInvalidResourceHandle from
 * the runtime of CUDA 13 on a B200 (measured/b200/e2e/launch_errors.txt), where that of CUDA
 * 12.4 said cudaErrorInvalidDeviceFunction (validation/ampere/launch_errors_3080ti.txt). */
static int not_a_kernel(int status) {
  return status == cudaErrorInvalidDeviceFunction ? cudaErrorInvalidResourceHandle : status;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  TRACE("cudaFuncGetAttributes(%p)", func);
  int fixed = 0, max_dynamic = 0;
  memset(attr, 0, sizeof *attr);
  int status = warplens_function_shared_memory(func, &fixed, &max_dynamic);
  if (status) return result(not_a_kernel(status));
  attr->sharedSizeBytes = (size_t)fixed;
  attr->maxThreadsPerBlock = 1024;
  attr->maxDynamicSharedSizeBytes = max_dynamic;
  attr->ptxVersion = 100;  /* the compute architecture (compute_100), which thrust dispatches on */
  attr->binaryVersion = 100;
  return cudaSuccess;
}

/* Opting in to more dynamic shared memory than the default leaves, which later launches of
 * the kernel are held to; or any other maximum, a smaller one included. */
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  TRACE("cudaFuncSetAttribute(%p, %d, %d)", func, (int)attr, value);
  return result(not_a_kernel(warplens_set_function_attribute(func, (int)attr, value)));
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
