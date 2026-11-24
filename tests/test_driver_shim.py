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

from warplens.host.shim_server import REGISTER_FUNCTION, REGISTER_PTX, ShimServer

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


def test_streams_events_and_host_memory_are_accepted(client):
    run_client(client, "tokens")


def test_server_registers_ptx_text_as_a_module():
    import struct
    server = ShimServer()
    (module,) = struct.unpack("<I", server.handle(REGISTER_PTX, PTX.encode()))
    function, count, *sizes = struct.unpack("<5I", server.handle(REGISTER_FUNCTION, struct.pack("<I", module) + b"scale"))
    assert (function, count, sizes) == (0, 3, [8, 8, 4])
