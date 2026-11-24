/* A stand-in for libcuda (the driver API): only the calls programs make directly. A runtime
 * that loads its own kernels (TVM's, for TIRx) does so through the module and launch calls. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "protocol.h"

/* Export each context-creation ABI explicitly, independent of cuda.h's default alias. */
#undef cuCtxCreate
#undef cuGetProcAddress

CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
                                cuuint32_t tensorRank, void *globalAddress,
                                const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
                                const cuuint32_t *boxDim, const cuuint32_t *elementStrides,
                                CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                                CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
  if (!tensorMap || (uintptr_t)tensorMap % 64 || !globalDim || !boxDim || !elementStrides ||
      tensorRank < 1 || tensorRank > 5 || (tensorRank > 1 && !globalStrides) ||
      l2Promotion < CU_TENSOR_MAP_L2_PROMOTION_NONE || l2Promotion > CU_TENSOR_MAP_L2_PROMOTION_L2_256B ||
      oobFill < CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE || oobFill > CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)
    return CUDA_ERROR_INVALID_VALUE;
  uint32_t head[5] = {(uint32_t)tensorDataType, tensorRank, (uint32_t)swizzle, (uint32_t)oobFill,
                      (uint32_t)interleave};
  uint64_t address = (uint64_t)globalAddress, dims[5] = {0}, strides[4] = {0};
  uint32_t box[5] = {0}, steps[5] = {0};
  for (cuuint32_t i = 0; i < tensorRank; ++i) {
    dims[i] = globalDim[i]; box[i] = boxDim[i]; steps[i] = elementStrides[i];
    if (i + 1 < tensorRank) strides[i] = globalStrides[i];
  }
  struct wl_part parts[6] = {{head, sizeof head}, {&address, sizeof address}, {dims, sizeof dims},
                             {strides, sizeof strides}, {box, sizeof box}, {steps, sizeof steps}};
  memset(tensorMap, 0, sizeof *tensorMap);
  return warplens_call(WL_TENSOR_MAP, parts, 6, tensorMap, sizeof *tensorMap, NULL)
             ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult cuInit(unsigned int flags) { return flags ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS; }

CUresult cuDriverGetVersion(int *version) {
  if (!version) return CUDA_ERROR_INVALID_VALUE;
  *version = CUDA_VERSION;  /* that of the headers this was built against */
  return CUDA_SUCCESS;
}

/* How cuda-python's driver module finds every function it wraps: by name, here. One that is
 * not implemented is reported as not found, and fails in the caller only if it is used. */
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  if (!symbol || !pfn || cudaVersion < 0 || cudaVersion > CUDA_VERSION || flags > 2)
    return CUDA_ERROR_INVALID_VALUE;
  static void *self;
  if (!self) {
    Dl_info info;
    if (dladdr((void *)cuInit, &info)) self = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
  }
  static const char *const suffixes[] = {"", "_v2", "_v3", "_v4"};  /* the name is asked for unversioned */
  char name[128];
  void *found = NULL;
  /* These APIs changed their argument lists. Returning whichever exported suffix is
   * found first calls through an incompatible function pointer (notably cuda-python's
   * older cuCtxCreate signature against a CUDA 13 build). See cudaTypedefs.h. */
  const char *exact = NULL;
  if (!strcmp(symbol, "cuCtxCreate"))
    exact = cudaVersion >= 12050 ? "cuCtxCreate_v4" : cudaVersion >= 11040 ? "cuCtxCreate_v3"
            : cudaVersion >= 3020 ? "cuCtxCreate_v2" : "cuCtxCreate";
  else if (!strcmp(symbol, "cuGetProcAddress"))
    exact = cudaVersion >= 12000 ? "cuGetProcAddress_v2" : "cuGetProcAddress";
  else if (!strcmp(symbol, "cuCtxGetDevice"))
    exact = cudaVersion >= 13000 ? "cuCtxGetDevice_v2" : "cuCtxGetDevice";
  else if (!strcmp(symbol, "cuCtxSynchronize"))
    exact = cudaVersion >= 13000 ? "cuCtxSynchronize_v2" : "cuCtxSynchronize";
  if (exact) {
    if (self) found = dlsym(self, exact);
  } else for (int i = 0; self && i < 4 && !found; ++i) {
    snprintf(name, sizeof name, "%s%s", symbol, suffixes[i]);
    found = dlsym(self, name);
  }
  *pfn = found;
  if (symbolStatus)
    *symbolStatus = found ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  return CUDA_SUCCESS;  /* v2 reports lookup failure through pfn and symbolStatus */
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
  CUresult status = cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, NULL);
  return status == CUDA_SUCCESS && !*pfn ? CUDA_ERROR_NOT_FOUND : status;
}

/* -- the device ------------------------------------------------------------------------ */

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  if (!pi || dev != 0) return pi ? CUDA_ERROR_INVALID_DEVICE : CUDA_ERROR_INVALID_VALUE;
  *pi = warplens_device_attribute((int)attrib);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
  *device = 0;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  const char *model = getenv("WARPLENS_MODEL");
  snprintf(name, (size_t)len, "warplens simulated %s", model ? model : "B200");
  return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
  switch (error) {
    case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_INVALID_DEVICE: *pStr = "CUDA_ERROR_INVALID_DEVICE"; break;
    case CUDA_ERROR_INVALID_IMAGE: *pStr = "CUDA_ERROR_INVALID_IMAGE"; break;
    case CUDA_ERROR_INVALID_HANDLE: *pStr = "CUDA_ERROR_INVALID_HANDLE"; break;
    case CUDA_ERROR_NOT_FOUND: *pStr = "CUDA_ERROR_NOT_FOUND"; break;
    case CUDA_ERROR_LAUNCH_FAILED: *pStr = "CUDA_ERROR_LAUNCH_FAILED"; break;
    case CUDA_ERROR_INVALID_CLUSTER_SIZE: *pStr = "CUDA_ERROR_INVALID_CLUSTER_SIZE"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **pStr) {
  *pStr = error == CUDA_SUCCESS ? "no error"
                                : "an error from the warplens CUDA driver shim; see any warplens message above";
  return CUDA_SUCCESS;
}

/* -- modules and kernels --------------------------------------------------------------- */

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
  *module = (CUmodule)warplens_load_module(image);
  return *module ? CUDA_SUCCESS : CUDA_ERROR_INVALID_IMAGE;
}

CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name) {
  if (!function || !name) return CUDA_ERROR_INVALID_VALUE;
  if (!module) return CUDA_ERROR_INVALID_HANDLE;
  *function = (CUfunction)warplens_module_kernel(module, name);
  return *function ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

/* The server keeps the parsed module, and kernels stay launchable, as handles to them may
 * outlive the module in a program that unloads eagerly. */
CUresult cuModuleUnload(CUmodule module) { (void)module; return CUDA_SUCCESS; }

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction function) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  if (!function) return CUDA_ERROR_INVALID_HANDLE;
  switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 1024; break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *pi = WL_MAX_SMEM_BYTES; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *pi = 100; break;
    default: *pi = 0; break;  /* sizes of the compiled kernel, which PTX does not fix */
  }
  return CUDA_SUCCESS;
}

/* A kernel from the runtime's cudaLibrary calls is the same handle. */
CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  return cuFuncGetAttribute(pi, attrib, (CUfunction)kernel);
}

CUresult cuFuncSetAttribute(CUfunction function, CUfunction_attribute attrib, int value) {
  if (!function) return CUDA_ERROR_INVALID_HANDLE;
  return wl_function_attribute_is_valid((int)attrib, value) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

static CUresult launch(CUfunction function, const uint32_t grid[3], const uint32_t block[3],
                       const uint32_t cluster[3], uint64_t shared, void **params, void **extra) {
  if (!function) return CUDA_ERROR_INVALID_HANDLE;
  if (extra) {
    fprintf(stderr, "warplens: kernel arguments packed into `extra` are not supported; pass kernelParams\n");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  int status = warplens_launch(function, grid, block, cluster, shared, params);
  /* The driver has no "invalid configuration": a block or grid it refuses is an invalid value
   * (measured on an RTX 3080 Ti; the runtime turns it into cudaErrorInvalidConfiguration). */
  return status == WL_INVALID_CONFIGURATION ? CUDA_ERROR_INVALID_VALUE : (CUresult)status;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
  (void)hStream;
  uint32_t grid[3] = {gridDimX, gridDimY, gridDimZ}, block[3] = {blockDimX, blockDimY, blockDimZ};
  uint32_t cluster[3] = {0, 0, 0};
  return launch(f, grid, block, cluster, sharedMemBytes, kernelParams, extra);
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams,
                          void **extra) {
  if (!config) return CUDA_ERROR_INVALID_VALUE;
  uint32_t grid[3] = {config->gridDimX, config->gridDimY, config->gridDimZ};
  uint32_t block[3] = {config->blockDimX, config->blockDimY, config->blockDimZ};
  uint32_t cluster[3] = {0, 0, 0};
  for (unsigned i = 0; i < config->numAttrs; ++i)
    if (config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      cluster[0] = config->attrs[i].value.clusterDim.x;
      cluster[1] = config->attrs[i].value.clusterDim.y;
      cluster[2] = config->attrs[i].value.clusterDim.z;
    }
  return launch(f, grid, block, cluster, config->sharedMemBytes, kernelParams, extra);
}

/* -- contexts, memory and streams: what a program needs to hold arrays through the driver ---
 * (cuda-python's driver module, and the CuTe DSL's helpers built on it). There is one
 * device and one context; a thread has it current once it creates, retains, sets or pushes
 * it. Memory goes through the runtime shim, whose error numbers the driver shares. */

static int the_context;
/* All supported handles identify this one context, so its per-thread stack needs only
 * a depth. Push/create add an entry, set replaces the top, and pop restores the old top. */
static __thread size_t context_depth;

static CUresult create_context(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  *pctx = (CUcontext)&the_context;
  ++context_depth;
  return CUDA_SUCCESS;
}
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  return create_context(pctx, flags, dev);
}
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  return create_context(pctx, flags, dev);
}
#if CUDA_VERSION >= 11040
CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *params, int count,
                        unsigned int flags, CUdevice dev) {
  (void)params;
  if (count) return CUDA_ERROR_NOT_SUPPORTED;
  return create_context(pctx, flags, dev);
}
#endif
#if CUDA_VERSION >= 12050
CUresult cuCtxCreate_v4(CUcontext *pctx, CUctxCreateParams *params, unsigned int flags, CUdevice dev) {
  if (params) return CUDA_ERROR_NOT_SUPPORTED;
  return create_context(pctx, flags, dev);
}
#endif
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  *pctx = (CUcontext)&the_context;
  return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return dev == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE; }
CUresult cuCtxDestroy(CUcontext ctx) {
  if (ctx != (CUcontext)&the_context) return CUDA_ERROR_INVALID_CONTEXT;
  if (context_depth) --context_depth;
  return CUDA_SUCCESS;
}
CUresult cuCtxSetCurrent(CUcontext ctx) {
  if (ctx && ctx != (CUcontext)&the_context) return CUDA_ERROR_INVALID_CONTEXT;
  if (!ctx) { if (context_depth) --context_depth; }
  else if (!context_depth) context_depth = 1;
  return CUDA_SUCCESS;
}
CUresult cuCtxPushCurrent(CUcontext ctx) {
  if (ctx != (CUcontext)&the_context) return CUDA_ERROR_INVALID_CONTEXT;
  ++context_depth;
  return CUDA_SUCCESS;
}
CUresult cuCtxPopCurrent(CUcontext *pctx) {
  if (!context_depth) return CUDA_ERROR_INVALID_CONTEXT;
  if (pctx) *pctx = (CUcontext)&the_context;
  --context_depth;
  return CUDA_SUCCESS;
}
CUresult cuCtxGetCurrent(CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = context_depth ? (CUcontext)&the_context : NULL;
  return CUDA_SUCCESS;
}
CUresult cuCtxGetDevice(CUdevice *device) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (!context_depth) return CUDA_ERROR_INVALID_CONTEXT;
  *device = 0;
  return CUDA_SUCCESS;
}
CUresult cuCtxSynchronize(void) { return (CUresult)cudaDeviceSynchronize(); }
#if CUDA_VERSION >= 13000
CUresult cuCtxGetDevice_v2(CUdevice *device, CUcontext ctx) {
  if (!ctx) return cuCtxGetDevice(device);
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (ctx != (CUcontext)&the_context) return CUDA_ERROR_INVALID_CONTEXT;
  *device = 0;
  return CUDA_SUCCESS;
}
CUresult cuCtxSynchronize_v2(CUcontext ctx) {
  if (ctx && ctx != (CUcontext)&the_context) return CUDA_ERROR_INVALID_CONTEXT;
  return cuCtxSynchronize();
}
#endif

CUresult cuDeviceGetCount(int *count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  *count = 1;
  return CUDA_SUCCESS;
}
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  size_t unused;
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
  return (CUresult)cudaMemGetInfo(&unused, bytes);
}
CUresult cuMemGetInfo(size_t *free_bytes, size_t *total) { return (CUresult)cudaMemGetInfo(free_bytes, total); }

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  void *address = NULL;
  if (!dptr || !bytesize) return CUDA_ERROR_INVALID_VALUE;
  CUresult status = (CUresult)cudaMalloc(&address, bytesize);
  *dptr = (CUdeviceptr)address;
  return status;
}
CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream stream) {
  (void)stream;
  return cuMemAlloc(dptr, bytesize);
}
CUresult cuMemFree(CUdeviceptr dptr) { return (CUresult)cudaFree((void *)dptr); }
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream stream) { (void)stream; return cuMemFree(dptr); }

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
  return (CUresult)cudaMemcpy((void *)dst, src, bytes, cudaMemcpyHostToDevice);
}
CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) {
  return (CUresult)cudaMemcpy(dst, (const void *)src, bytes, cudaMemcpyDeviceToHost);
}
CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
  return (CUresult)cudaMemcpy((void *)dst, (const void *)src, bytes, cudaMemcpyDeviceToDevice);
}
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream) {
  (void)stream;
  return cuMemcpyHtoD(dst, src, bytes);
}
CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t bytes, CUstream stream) {
  (void)stream;
  return cuMemcpyDtoH(dst, src, bytes);
}
CUresult cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
  return (CUresult)cudaMemset((void *)dst, value, count);
}

CUresult cuStreamCreate(CUstream *stream, unsigned int flags) {
  return (CUresult)cudaStreamCreateWithFlags((cudaStream_t *)stream, flags);
}
CUresult cuStreamDestroy(CUstream stream) { return (CUresult)cudaStreamDestroy((cudaStream_t)stream); }
CUresult cuStreamSynchronize(CUstream stream) { return (CUresult)cudaStreamSynchronize((cudaStream_t)stream); }
