/* A stand-in for libcuda (the driver API): only the calls programs make directly. A runtime
 * that loads its own kernels (TVM's, for TIRx) does so through the module and launch calls. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#include "protocol.h"

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
  (void)cudaVersion; (void)flags;
  if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
  static void *self;
  if (!self) {
    Dl_info info;
    if (dladdr((void *)cuInit, &info)) self = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
  }
  static const char *const suffixes[] = {"", "_v2", "_v3"};  /* the name is asked for unversioned */
  char name[128];
  void *found = NULL;
  for (int i = 0; self && i < 3 && !found; ++i) {
    snprintf(name, sizeof name, "%s%s", symbol, suffixes[i]);
    found = dlsym(self, name);
  }
  *pfn = found;
  if (symbolStatus)
    *symbolStatus = found ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  return found ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
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
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *pi = 227 * 1024; break;
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
  (void)attrib; (void)value;  /* e.g. opting in to more shared memory: always allowed */
  return function ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

static CUresult launch(CUfunction function, const uint32_t grid[3], const uint32_t block[3],
                       const uint32_t cluster[3], uint64_t shared, void **params, void **extra) {
  if (!function) return CUDA_ERROR_INVALID_HANDLE;
  if (extra) {
    fprintf(stderr, "warplens: kernel arguments packed into `extra` are not supported; pass kernelParams\n");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return (CUresult)warplens_launch(function, grid, block, cluster, shared, params);
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
