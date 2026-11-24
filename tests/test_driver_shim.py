"""The driver API side of the shim: a runtime that loads PTX and launches kernels itself, as
TVM's does for TIRx, instead of going through code that nvcc generated. A native client
makes the calls against the real shim libraries and a ShimServer."""

import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import HEADER, cuda_include

from warplens.host.shim_server import REGISTER_FUNCTION, REGISTER_MODULE, ShimServer

ROOT = Path(__file__).resolve().parent.parent
PTX = HEADER + """
.visible .entry scale(.param .u64 src, .param .u64 dst, .param .u32 factor)
{
    .reg .b64 %ps, %pd, %off; .reg .b32 %i, %x, %y, %v, %f;
    ld.param.u64 %ps, [src]; ld.param.u64 %pd, [dst]; ld.param.u32 %f, [factor];
    mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
    mad.lo.u32 %i, %x, 32, %y;
    mul.wide.u32 %off, %i, 4;
    add.u64 %ps, %ps, %off; add.u64 %pd, %pd, %off;
    ld.global.u32 %v, [%ps];
    mul.lo.u32 %v, %v, %f;
    st.global.u32 [%pd], %v;
    ret;
}
"""
CLIENT = r"""
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "check failed on line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)
#define N 64

static int run(CUfunction kernel, int extended, unsigned factor) {
  unsigned input[N], output[N], *source, *target;
  for (unsigned i = 0; i < N; ++i) input[i] = i + 1;
  CHECK(cudaMalloc((void **)&source, sizeof input) == cudaSuccess);
  CHECK(cudaMalloc((void **)&target, sizeof output) == cudaSuccess);
  CHECK(cudaMemcpy(source, input, sizeof input, cudaMemcpyHostToDevice) == cudaSuccess);
  void *arguments[3] = {&source, &target, &factor};
  if (extended) {
    CUlaunchConfig config;
    memset(&config, 0, sizeof config);
    config.gridDimX = N / 32; config.gridDimY = 1; config.gridDimZ = 1;
    config.blockDimX = 32; config.blockDimY = 1; config.blockDimZ = 1;
    CHECK(cuLaunchKernelEx(&config, kernel, arguments, NULL) == CUDA_SUCCESS);
    CHECK(cuLaunchKernelEx(&config, kernel, NULL, arguments) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(cuLaunchKernelEx(NULL, kernel, arguments, NULL) == CUDA_ERROR_INVALID_VALUE);
  } else {
    CHECK(cuLaunchKernel(kernel, N / 32, 1, 1, 32, 1, 1, 0, NULL, arguments, NULL) == CUDA_SUCCESS);
  }
  CHECK(cudaMemcpy(output, target, sizeof output, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (unsigned i = 0; i < N; ++i) CHECK(output[i] == (i + 1) * factor);
  CHECK(cudaFree(source) == cudaSuccess && cudaFree(target) == cudaSuccess);
  return 0;
}

int main(int argc, char **argv) {
  CHECK(argc == 3);
  FILE *file = fopen(argv[2], "rb");
  CHECK(file);
  static char ptx[1 << 16];
  size_t size = fread(ptx, 1, sizeof ptx - 1, file);
  ptx[size] = 0;
  fclose(file);

  if (!strcmp(argv[1], "launch")) {
    CUmodule module; CUfunction kernel, again, missing;
    CHECK(cuInit(0) == CUDA_SUCCESS);
    CHECK(cuModuleLoadData(&module, ptx) == CUDA_SUCCESS);
    CHECK(cuModuleGetFunction(&kernel, module, "scale") == CUDA_SUCCESS);
    CHECK(cuModuleGetFunction(&again, module, "scale") == CUDA_SUCCESS && again == kernel);
    CHECK(cuModuleGetFunction(&missing, module, "absent") == CUDA_ERROR_NOT_FOUND);
    CHECK(run(kernel, 1, 3) == 0);
    CHECK(run(kernel, 0, 5) == 0);
    CHECK(cuModuleUnload(module) == CUDA_SUCCESS);
    return 0;
  }
  if (!strcmp(argv[1], "images")) {
    CUmodule module;
    CHECK(cuModuleLoadData(&module, "\x7f" "ELF\x02\x01\x01") == CUDA_ERROR_INVALID_IMAGE);
    CHECK(cuModuleLoadData(&module, "not a module") == CUDA_ERROR_INVALID_IMAGE);
    CHECK(cuModuleLoadData(&module, NULL) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuModuleLoadData(&module, ptx) == CUDA_SUCCESS);
    return 0;
  }
  if (!strcmp(argv[1], "device")) {
    int value = 0, minor = -1;
    char name[64];
    const char *text = NULL;
    CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS && value == 10);
    CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0) == CUDA_SUCCESS);
    CHECK(minor == atoi(getenv("EXPECT_MINOR")));
    CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 0) == CUDA_SUCCESS);
    CHECK(value == 227 * 1024);
    CHECK(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 1) == CUDA_ERROR_INVALID_DEVICE);
    CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, 0) == cudaSuccess && value == minor);
    CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0) == cudaSuccess && value == 32);
    CHECK(cuDeviceGetName(name, sizeof name, 0) == CUDA_SUCCESS && strstr(name, "warplens simulated"));
    CHECK(cuGetErrorName(CUDA_ERROR_NOT_FOUND, &text) == CUDA_SUCCESS && !strcmp(text, "CUDA_ERROR_NOT_FOUND"));
    CHECK(cuGetErrorString(CUDA_SUCCESS, &text) == CUDA_SUCCESS && !strcmp(text, "no error"));
    size_t free_bytes = 0, total = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total) == cudaSuccess && total >= ((size_t)192 << 30) && free_bytes == total);
    return 0;
  }
  if (!strcmp(argv[1], "library")) {  /* the runtime's module API, handed machine code */
#if CUDART_VERSION >= 12080
    static const unsigned char cubin[64] = {0x7f, 'E', 'L', 'F', 2, 1, 1};
    cudaLibrary_t library; cudaKernel_t kernel, missing;
    int value = -1;
    CHECK(cudaLibraryLoadData(&library, cubin, NULL, NULL, 0, NULL, NULL, 0) == cudaSuccess);
    CHECK(cudaLibraryGetKernel(&kernel, library, "scale") == cudaSuccess);
    CHECK(cudaLibraryGetKernel(&missing, library, "absent") == cudaErrorSymbolNotFound);
    CHECK(cudaGetLastError() == cudaErrorSymbolNotFound);
    CHECK(cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, (CUkernel)kernel, 0) == CUDA_SUCCESS);
    CHECK(value == 0);
    CHECK(cudaKernelSetAttributeForDevice(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096, 0) == cudaSuccess);
    unsigned input[N], output[N], factor = 7, *source, *target;
    for (unsigned i = 0; i < N; ++i) input[i] = i + 1;
    CHECK(cudaMalloc((void **)&source, sizeof input) == cudaSuccess);
    CHECK(cudaMalloc((void **)&target, sizeof output) == cudaSuccess);
    CHECK(cudaMemcpy(source, input, sizeof input, cudaMemcpyHostToDevice) == cudaSuccess);
    void *arguments[3] = {&source, &target, &factor};
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof config);
    config.gridDim.x = N / 32; config.gridDim.y = 1; config.gridDim.z = 1;
    config.blockDim.x = 32; config.blockDim.y = 1; config.blockDim.z = 1;
    CHECK(cudaLaunchKernelExC(&config, kernel, arguments) == cudaSuccess);
    CHECK(cudaMemcpy(output, target, sizeof output, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (unsigned i = 0; i < N; ++i) CHECK(output[i] == (i + 1) * factor);
    CHECK(cudaLibraryUnload(library) == cudaSuccess);
    return 0;
#else
    return 77;
#endif
  }
  if (!strcmp(argv[1], "lookup")) {  /* how cuda-python's driver module finds functions */
    void *found = NULL;
    CUdriverProcAddressQueryResult status;
    int version = 0;
    CUdevice device = -1;
    CHECK(cuGetProcAddress("cuInit", &found, CUDA_VERSION, 0, &status) == CUDA_SUCCESS);
    CHECK(found == (void *)cuInit && status == CU_GET_PROC_ADDRESS_SUCCESS);
    CHECK(cuGetProcAddress("cuGetProcAddress", &found, CUDA_VERSION, 0, NULL) == CUDA_SUCCESS && found);
#if CUDA_VERSION >= 13000
    CHECK(cuGetProcAddress("cuCtxCreate", &found, 12050, 0, NULL) == CUDA_SUCCESS && found == (void *)cuCtxCreate);
#else
    CHECK(cuGetProcAddress("cuCtxCreate", &found, 3020, 0, NULL) == CUDA_SUCCESS && found == (void *)cuCtxCreate);
#endif
    CHECK(cuGetProcAddress("cuMemAllocManaged", &found, CUDA_VERSION, 0, &status) == CUDA_SUCCESS);
    CHECK(found == NULL && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    CHECK(cuDriverGetVersion(&version) == CUDA_SUCCESS && version == CUDA_VERSION);
    CHECK(cuDeviceGet(&device, 0) == CUDA_SUCCESS && device == 0);
    CHECK(cuDeviceGet(&device, 1) == CUDA_ERROR_INVALID_DEVICE);
    return 0;
  }
  if (!strcmp(argv[1], "abi")) {
    void *found = NULL, *other = NULL;
    CUcontext context = NULL;
    CUdriverProcAddressQueryResult query_status;
    CHECK(cuGetProcAddress("cuCtxCreate", &found, 3020, 0, NULL) == CUDA_SUCCESS && found);
    PFN_cuCtxCreate_v3020 legacy_create = (PFN_cuCtxCreate_v3020)found;
    CHECK(legacy_create(&context, 0, 0) == CUDA_SUCCESS && context);
    CHECK(cuCtxPopCurrent(NULL) == CUDA_SUCCESS);
    CHECK(legacy_create(&context, 0, 1) == CUDA_ERROR_INVALID_DEVICE);
    CHECK(cuGetProcAddress("cuCtxCreate", &other, 11040, 0, NULL) == CUDA_SUCCESS && other && other != found);
    PFN_cuCtxCreate_v11040 affinity_create = (PFN_cuCtxCreate_v11040)other;
    CHECK(affinity_create(&context, NULL, 0, 0, 0) == CUDA_SUCCESS && context);
    CHECK(cuCtxPopCurrent(NULL) == CUDA_SUCCESS);
#if CUDA_VERSION >= 12050
    CHECK(cuGetProcAddress("cuCtxCreate", &found, 12050, 0, NULL) == CUDA_SUCCESS && found && found != other);
    CHECK(((PFN_cuCtxCreate_v12050)found)(&context, NULL, 0, 0) == CUDA_SUCCESS && context);
    CHECK(cuCtxPopCurrent(NULL) == CUDA_SUCCESS);
#endif
#if CUDA_VERSION >= 13000
    CUdevice device = -1;
    CHECK(cuGetProcAddress("cuCtxGetDevice", &found, 13000, 0, NULL) == CUDA_SUCCESS && found);
    CHECK(((PFN_cuCtxGetDevice_v13000)found)(&device, context) == CUDA_SUCCESS && device == 0);
    CHECK(((PFN_cuCtxGetDevice_v13000)found)(&device, (CUcontext)1) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cuGetProcAddress("cuCtxGetDevice", &other, 2000, 0, NULL) == CUDA_SUCCESS && other != found);
    CHECK(((PFN_cuCtxGetDevice_v2000)other)(&device) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cuGetProcAddress("cuCtxSynchronize", &found, 13000, 0, NULL) == CUDA_SUCCESS && found);
    CHECK(((PFN_cuCtxSynchronize_v13000)found)(context) == CUDA_SUCCESS);
    CHECK(((PFN_cuCtxSynchronize_v13000)found)((CUcontext)1) == CUDA_ERROR_INVALID_CONTEXT);
#endif
    CHECK(cuGetProcAddress("cuGetProcAddress", &found, 11030, 0, NULL) == CUDA_SUCCESS && found);
    CHECK(found != (void *)cuGetProcAddress);
    PFN_cuGetProcAddress_v11030 legacy_lookup = (PFN_cuGetProcAddress_v11030)found;
    CHECK(legacy_lookup("cuInit", &other, 2000, 0) == CUDA_SUCCESS && other == (void *)cuInit);
    CHECK(legacy_lookup("not_a_cuda_function", &other, 11030, 0) == CUDA_ERROR_NOT_FOUND && !other);
    CHECK(cuGetProcAddress("not_a_cuda_function", &other, CUDA_VERSION, 0, &query_status) == CUDA_SUCCESS);
    CHECK(!other && query_status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    CHECK(cuGetProcAddress("cuInit", &other, CUDA_VERSION + 1000, 0, NULL) == CUDA_ERROR_INVALID_VALUE);
    return 0;
  }
  if (!strcmp(argv[1], "nested-context")) {
    CUcontext context = NULL, after = NULL;
    CUdevice device = -1;
    CHECK(cuDevicePrimaryCtxRetain(&context, 0) == CUDA_SUCCESS && context);
    CHECK(cuCtxSetCurrent(context) == CUDA_SUCCESS);
    CHECK(cuCtxPushCurrent(context) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(context) == CUDA_SUCCESS);  /* replace, not another push */
    CHECK(cuCtxPopCurrent(&after) == CUDA_SUCCESS && after == context);
    CHECK(cuCtxGetCurrent(&after) == CUDA_SUCCESS && after == context);
    CHECK(cuCtxGetDevice(&device) == CUDA_SUCCESS && device == 0);
    CHECK(cuCtxPushCurrent(context) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);  /* removes only the top */
    CHECK(cuCtxGetCurrent(&after) == CUDA_SUCCESS && after == context);
    CHECK(cuCtxPopCurrent(&after) == CUDA_SUCCESS && after == context);
    CHECK(cuCtxGetCurrent(&after) == CUDA_SUCCESS && after == NULL);
    CHECK(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);  /* empty stack is a no-op */
    return 0;
  }
  if (!strcmp(argv[1], "memory")) {  /* arrays held through the driver, as cuda-python does */
    CUcontext context = NULL, other = (CUcontext)1;
    CUdevice device = -1;
    CUdeviceptr first, second;
    CUstream stream;
    unsigned char host[32], back[32];
    size_t total = 0;
    CHECK(cuCtxGetCurrent(&other) == CUDA_SUCCESS && other == NULL);
    CHECK(cuCtxGetDevice(&device) == CUDA_ERROR_INVALID_CONTEXT);
#if CUDA_VERSION >= 13000
    CHECK(cuCtxCreate(&context, NULL, 0, 0) == CUDA_SUCCESS && context);
#else
    CHECK(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS && context);
#endif
    CHECK(cuCtxGetCurrent(&other) == CUDA_SUCCESS && other == context);
    CHECK(cuCtxGetDevice(&device) == CUDA_SUCCESS && device == 0);
    CHECK(cuCtxPopCurrent(&other) == CUDA_SUCCESS && other == context);
    CHECK(cuCtxGetCurrent(&other) == CUDA_SUCCESS && other == NULL);
    CHECK(cuDevicePrimaryCtxRetain(&other, 0) == CUDA_SUCCESS && other == context);
    CHECK(cuCtxPushCurrent(context) == CUDA_SUCCESS && cuCtxSetCurrent(context) == CUDA_SUCCESS);
    CHECK(cuDeviceTotalMem(&total, 0) == CUDA_SUCCESS && total >= ((size_t)192 << 30));
    CHECK(cuStreamCreate(&stream, 0) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&first, 0) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuMemAlloc(&first, sizeof host) == CUDA_SUCCESS);
    CHECK(cuMemAllocAsync(&second, sizeof host, stream) == CUDA_SUCCESS && second != first);
    for (unsigned i = 0; i < sizeof host; ++i) host[i] = (unsigned char)(3 * i + 1);
    CHECK(cuMemcpyHtoD(first, host, sizeof host) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoD(second, first, sizeof host) == CUDA_SUCCESS);
    CHECK(cuMemsetD8(second, 0xee, 4) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoHAsync(back, second, sizeof back, stream) == CUDA_SUCCESS);
    CHECK(cuStreamSynchronize(stream) == CUDA_SUCCESS && cuCtxSynchronize() == CUDA_SUCCESS);
    for (unsigned i = 0; i < sizeof host; ++i) CHECK(back[i] == (i < 4 ? 0xee : host[i]));
    CHECK(cuMemcpyHtoDAsync(first + 30, host, 4, stream) != CUDA_SUCCESS);  /* past the end */
    CHECK(cuMemFree(first) == CUDA_SUCCESS && cuMemFreeAsync(second, stream) == CUDA_SUCCESS);
    CHECK(cuMemcpyDtoH(back, first, 1) != CUDA_SUCCESS);  /* freed */
    CHECK(cuStreamDestroy(stream) == CUDA_SUCCESS && cuCtxDestroy(context) == CUDA_SUCCESS);
    return 0;
  }
  if (!strcmp(argv[1], "errors")) {  /* refusals, with the codes an RTX 3080 Ti gave for them */
    CUmodule module; CUfunction kernel;
    const char *text = NULL;
    unsigned *source, *target, factor = 2;
    int limit = 0;
    CHECK(cuInit(0) == CUDA_SUCCESS && cuModuleLoadData(&module, ptx) == CUDA_SUCCESS);
    CHECK(cuModuleGetFunction(&kernel, module, "scale") == CUDA_SUCCESS);
    CHECK(cudaMalloc((void **)&source, 4096) == cudaSuccess && cudaMalloc((void **)&target, 4096) == cudaSuccess);
    void *arguments[3] = {&source, &target, &factor};
    dim3 grid = {2, 1, 1}, block = {32, 1, 1}, big = {1025, 1, 1}, none = {0, 1, 1};
    CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) == cudaSuccess);
    CHECK(limit == 227 * 1024);

    /* cudaFuncSetAttribute: up to the opt-in maximum the device reports, and no further */
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0) == cudaSuccess);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit) == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit + 1) == cudaErrorInvalidValue);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue && cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit + 16) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, -1) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100) == cudaSuccess);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, -1) == cudaSuccess);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 101) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, -2) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, (enum cudaFuncAttribute)99, 0) == cudaErrorInvalidValue);
    CHECK(cudaFuncSetAttribute(kernel, (enum cudaFuncAttribute)14, 1) == cudaSuccess);  /* non-portable cluster size */
    CHECK(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, limit) == CUDA_SUCCESS);
    CHECK(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, limit + 1) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 101) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 5) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuFuncSetAttribute(NULL, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0) == CUDA_ERROR_INVALID_HANDLE);
    (void)cudaGetLastError();

    /* a refused launch: returned by the call, kept for cudaGetLastError, not a kernel error */
    CHECK(cudaLaunchKernel(kernel, grid, block, arguments, (size_t)limit, NULL) == cudaSuccess);
    CHECK(cudaLaunchKernel(kernel, grid, block, arguments, (size_t)limit + 16, NULL) == cudaErrorInvalidValue);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue && cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess);
    CHECK(cudaLaunchKernel(kernel, grid, big, arguments, 0, NULL) == cudaErrorInvalidConfiguration);
    CHECK(cudaLaunchKernel(kernel, grid, none, arguments, 0, NULL) == cudaErrorInvalidConfiguration);
    CHECK(cudaLaunchKernel(kernel, none, block, arguments, 0, NULL) == cudaErrorInvalidConfiguration);
    CHECK(cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaErrorInvalidConfiguration);
    CHECK(cuLaunchKernel(kernel, 2, 1, 1, 1025, 1, 1, 0, NULL, arguments, NULL) == CUDA_ERROR_INVALID_VALUE);
    CHECK(cuLaunchKernel(kernel, 2, 1, 1, 32, 1, 1, (unsigned)limit + 16, NULL, arguments, NULL) == CUDA_ERROR_INVALID_VALUE);
    (void)cudaGetLastError();  /* the two libraries share one last error here */
    for (unsigned clusters = 16; clusters <= 32; clusters += 16) {  /* 16 CTAs is the largest cluster */
      cudaLaunchConfig_t config; cudaLaunchAttribute attribute; CUlaunchConfig driver; CUlaunchAttribute shape;
      memset(&config, 0, sizeof config); memset(&attribute, 0, sizeof attribute);
      memset(&driver, 0, sizeof driver); memset(&shape, 0, sizeof shape);
      config.gridDim.x = clusters; config.gridDim.y = config.gridDim.z = 1; config.blockDim.x = 1;
      config.blockDim.y = config.blockDim.z = 1;
      attribute.id = cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x = clusters; attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
      config.attrs = &attribute; config.numAttrs = 1;
      driver.gridDimX = clusters; driver.gridDimY = driver.gridDimZ = 1;
      driver.blockDimX = driver.blockDimY = driver.blockDimZ = 1;
      shape.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      shape.value.clusterDim.x = clusters; shape.value.clusterDim.y = shape.value.clusterDim.z = 1;
      driver.attrs = &shape; driver.numAttrs = 1;
      cudaError_t expected = clusters == 16 ? cudaSuccess : cudaErrorInvalidClusterSize;
      CHECK(cudaLaunchKernelExC(&config, kernel, arguments) == expected);
      CHECK(cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == expected);
      CHECK(cuLaunchKernelEx(&driver, kernel, arguments, NULL) ==
            (clusters == 16 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CLUSTER_SIZE));
      (void)cudaGetLastError();
    }
    CHECK(!strcmp(cudaGetErrorName(cudaErrorInvalidClusterSize), "cudaErrorInvalidClusterSize"));
    CHECK(!strcmp(cudaGetErrorName(cudaErrorInvalidConfiguration), "cudaErrorInvalidConfiguration"));
    CHECK(!strcmp(cudaGetErrorName(cudaErrorMemoryAllocation), "cudaErrorMemoryAllocation"));
    CHECK(cuGetErrorName(CUDA_ERROR_INVALID_CLUSTER_SIZE, &text) == CUDA_SUCCESS);
    CHECK(!strcmp(text, "CUDA_ERROR_INVALID_CLUSTER_SIZE"));

    /* a kernel that faults is still a launch failure, and the synchronize reports that one */
    grid.x = 1024;
    CHECK(cudaLaunchKernel(kernel, grid, block, arguments, 0, NULL) == cudaErrorLaunchFailure);
    CHECK(cudaDeviceSynchronize() == cudaErrorLaunchFailure && cudaGetLastError() == cudaErrorLaunchFailure);
    CHECK(cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess);
    return 0;
  }
  if (!strcmp(argv[1], "tokens")) {  /* streams, events and host memory, as TVM's runtime uses them */
    cudaStream_t stream; cudaEvent_t start, end;
    float elapsed = -1.0f;
    unsigned char *host, *device, *copy, back = 0;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess);
    CHECK(cudaEventRecord(start, stream) == cudaSuccess);
    CHECK(cudaMallocHost((void **)&host, 16) == cudaSuccess);
    memset(host, 0x5a, 16);
    CHECK(cudaMalloc((void **)&device, 16) == cudaSuccess && cudaMalloc((void **)&copy, 16) == cudaSuccess);
    CHECK(cudaMemcpyAsync(device, host, 16, cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(cudaMemcpyPeerAsync(copy, 0, device, 0, 16, stream) == cudaSuccess);
    CHECK(cudaMemcpyPeerAsync(copy, 1, device, 0, 16, stream) == cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    CHECK(cudaMemcpyAsync(&back, copy + 15, 1, cudaMemcpyDeviceToHost, stream) == cudaSuccess && back == 0x5a);
    CHECK(cudaEventRecord(end, stream) == cudaSuccess && cudaStreamWaitEvent(stream, end, 0) == cudaSuccess);
    CHECK(cudaEventSynchronize(end) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaEventElapsedTime(&elapsed, start, end) == cudaSuccess && elapsed == 0.0f);
    CHECK(cudaFreeHost(host) == cudaSuccess && cudaEventDestroy(start) == cudaSuccess);
    CHECK(cudaEventDestroy(end) == cudaSuccess && cudaStreamDestroy(stream) == cudaSuccess);
    return 0;
  }
  return 2;
}
"""


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    include = cuda_include("cuda.h")
    if compiler is None or include is None or not (include / "cuda_runtime_api.h").exists():
        pytest.skip("the driver shim tests need a C compiler and CUDA toolkit headers")
    build = tmp_path_factory.mktemp("driver-shim")
    subprocess.run([str(ROOT / "warplens/shim/build.sh"), str(include), str(build)],
                   check=True, capture_output=True, text=True)
    (runtime,) = build.glob("libcudart.so.*")
    (build / "client.c").write_text(CLIENT)
    (build / "module.ptx").write_text(PTX)
    subprocess.run([compiler, "-Wall", "-Wextra", "-Werror", f"-I{include}", str(build / "client.c"),
                    str(build / "libcuda.so.1"), str(runtime), f"-Wl,-rpath,{build}", "-o", str(build / "client")],
                   check=True, capture_output=True, text=True)
    return build


def run_client(build, mode, model="B200", server=None):
    server = server or ShimServer(model)
    ours, theirs = socket.socketpair()
    thread = threading.Thread(target=server.serve, args=(ours,), daemon=True)
    thread.start()
    try:
        done = subprocess.run([str(build / "client"), mode, str(build / "module.ptx")], pass_fds=[theirs.fileno()],
                              env={**os.environ, "WARPLENS_SHIM_FD": str(theirs.fileno()), "WARPLENS_MODEL": model,
                                   "EXPECT_MINOR": "3" if model == "B300" else "0"},
                              capture_output=True, text=True, timeout=60)
    finally:
        theirs.close()
        thread.join(timeout=5)
        ours.close()
    if done.returncode == 77:
        pytest.skip("the runtime library API needs CUDA 12.8 or newer headers")
    assert done.returncode == 0, f"native client exited {done.returncode}:\n{done.stderr}"
    return server, done


def test_driver_module_is_loaded_from_ptx_and_launched(client):
    server, _ = run_client(client, "launch")
    assert [name for name, _ in server.launches] == ["scale", "scale"]
    assert len(server.functions) == 1  # the second cuModuleGetFunction reused the first handle


def test_driver_rejects_machine_code_and_text_that_is_not_ptx(client):
    server, done = run_client(client, "images")
    assert "machine code for a real GPU" in done.stderr and "target_format" in done.stderr
    assert "not PTX text" in done.stderr
    assert len(server.modules) == 1 and server.machine_code


@pytest.mark.parametrize("model", ["B200", "B300"])
def test_driver_and_runtime_describe_the_same_device(client, model):
    run_client(client, "device", model)


def test_refusals_carry_the_codes_a_gpu_gives(client):
    """`cudaFuncSetAttribute` above the opt-in maximum, and launches refused for their shared
    memory, block, grid or cluster: the code the call returns, that `cudaGetLastError` has it
    once, and that `cudaDeviceSynchronize` does not repeat it (measured on an RTX 3080 Ti with
    its own limits; the cluster cases are documented behaviour, Ampere having no clusters)."""
    server, done = run_client(client, "errors")
    assert "more than 16 CTAs" in done.stderr and "the limit is 232448" in done.stderr
    assert [stats.ctas for _, stats in server.launches] == [2, 16, 16]  # no refused launch ran


def test_streams_events_and_host_memory_are_accepted(client):
    run_client(client, "tokens")


def test_runtime_library_of_machine_code_takes_its_kernels_from_kept_ptx(client, tmp_path):
    """What the CuTe DSL does: the module it loads is machine code, and the PTX is on disk."""
    (tmp_path / "older.ptx").write_text(PTX.replace("mul.lo.u32 %v, %v, %f;", "mov.u32 %v, 0;"))
    newest = tmp_path / "newest.ptx"
    newest.write_text(PTX)
    os.utime(tmp_path / "older.ptx", (1, 1))
    (tmp_path / "other.ptx").write_text(PTX.replace("scale", "rescale"))
    server, _ = run_client(client, "library", server=ShimServer(kept_ptx=tmp_path))
    assert [name for name, _ in server.launches] == ["scale"]
    assert server.kept == {0} and server.machine_code  # the lookup of `absent` failed


def test_runtime_library_of_machine_code_is_refused_without_kept_ptx(client):
    with pytest.raises(AssertionError, match="machine code for a real GPU"):
        run_client(client, "library")


def test_arrays_are_held_through_the_driver(client):
    _, done = run_client(client, "memory")
    assert done.stderr.count("warplens:") == 2  # the copy past the end and the one from freed memory


def test_driver_functions_are_found_by_name(client):
    run_client(client, "lookup")


def test_server_registers_ptx_text_as_a_module():
    import struct
    server = ShimServer()
    (module,) = struct.unpack("<I", server.handle(REGISTER_MODULE, PTX.encode()))
    function, count, *sizes = struct.unpack("<5I", server.handle(REGISTER_FUNCTION, struct.pack("<I", module) + b"scale"))
    assert (function, count, sizes) == (0, 3, [8, 8, 4])


def test_driver_lookup_preserves_each_requested_function_abi(client):
    run_client(client, "abi")


def test_driver_nested_context_push_pop_restores_previous_context(client):
    run_client(client, "nested-context")


@pytest.mark.parametrize("overwrite", [False, True])
def test_kept_ptx_is_bound_when_each_module_loads_not_when_its_kernel_is_looked_up(tmp_path, overwrite):
    import struct
    import numpy as np

    server = ShimServer(kept_ptx=tmp_path)
    first = tmp_path / "first.ptx"
    first.write_text(PTX)
    os.utime(first, ns=(1, 1))
    first_module = server.handle(REGISTER_MODULE, b"\x7fELF")
    second = first if overwrite else tmp_path / "second.ptx"
    second.write_text(PTX.replace("mul.lo.u32 %v, %v, %f;", "add.u32 %v, %v, %f;"))
    second_module = server.handle(REGISTER_MODULE, b"\x7fELF")
    # Lookup is deliberately deferred until after the second module compiles/loads.
    values = np.arange(32, dtype=np.uint32)
    source = server.device.from_numpy(values)
    for module_id, expected in ((first_module, values * 3), (second_module, values + 3)):
        function = struct.unpack_from("<I", server.handle(REGISTER_FUNCTION, module_id + b"scale"))[0]
        module, index, name = server.functions[function]
        out = server.device.empty(32, "u32")
        server.device.launch(server.modules[module][index], name, grid=1, block=32, args=[source, out, 3])
        np.testing.assert_array_equal(out.numpy(), expected)


def test_direct_ptx_module_does_not_snapshot_kept_sources(tmp_path):
    server = ShimServer(kept_ptx=tmp_path)
    server.handle(REGISTER_MODULE, PTX.encode())
    assert not server.kept_sources
