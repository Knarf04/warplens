"""Native libcudart behavior, tested without loading a GPU driver."""

import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import cuda_include

from warplens.errors import WarplensError
from warplens.host.shim_server import FREE, MALLOC, ShimServer

ROOT = Path(__file__).resolve().parent.parent
CLIENT = r'''
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "protocol.h"

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "check failed on line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

static cudaError_t worker_error;
static void *fail_in_worker(void *unused) {
  (void)unused;
  worker_error = cudaMemset((void *)(uintptr_t)0xdeadbeef, 0, 1);
  if (cudaPeekAtLastError() != worker_error) worker_error = cudaSuccess;
  return NULL;
}


static void *allocate_in_worker(void *argument) {
  unsigned char input = (unsigned char)(uintptr_t)argument;
  for (int i = 0; i < 128; ++i) {
    unsigned char *device, output = 0;
    if (cudaMalloc((void **)&device, 4) != cudaSuccess ||
        cudaMemcpy(device, &input, 1, cudaMemcpyDefault) != cudaSuccess ||
        cudaMemcpy(&output, device, 1, cudaMemcpyDefault) != cudaSuccess ||
        output != input || cudaFree(device) != cudaSuccess)
      return (void *)(uintptr_t)1;
  }
  return NULL;
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  if (!strcmp(argv[1], "oversized_part") || !strcmp(argv[1], "oversized_sum")) {
    unsigned char tiny = 0;
    size_t length = 123;
    struct wl_part parts[2] = {{&tiny, (size_t)UINT32_MAX}, {&tiny, 1}};
    int count = 2;
    if (!strcmp(argv[1], "oversized_part")) {
      parts[0].size = (size_t)UINT32_MAX + 1;
      count = 1;
    }
    CHECK(warplens_call(WL_WRITE, parts, count, NULL, 0, &length) == cudaErrorInvalidValue);
    CHECK(length == 0);
    /* The rejected request must not write bytes or retain the transport lock. */
    void *device;
    CHECK(cudaMalloc(&device, 4) == cudaSuccess);
    CHECK(cudaFree(device) == cudaSuccess);
    return 0;
  }
  if (!strcmp(argv[1], "properties")) {
    struct cudaDeviceProp prop;
    const char *model = getenv("WARPLENS_MODEL");
    int b300 = model && !strcmp(model, "B300");
    CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    CHECK(prop.major == 10 && prop.minor == (b300 ? 3 : 0));
    CHECK(prop.totalGlobalMem == (size_t)(b300 ? 288 : 192) << 30);
    CHECK(strstr(prop.name, b300 ? "B300" : "B200"));
    int cluster_launch = -1;
    CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0) == cudaSuccess);
    CHECK(cluster_launch == 1 && prop.clusterLaunch == 1);
    return 0;
  }
  if (!strcmp(argv[1], "thread_allocations")) {
    pthread_t workers[16];
    for (int i = 0; i < 16; ++i)
      CHECK(pthread_create(&workers[i], NULL, allocate_in_worker, (void *)(uintptr_t)(i + 1)) == 0);
    for (int i = 0; i < 16; ++i) {
      void *failed;
      CHECK(pthread_join(workers[i], &failed) == 0);
      CHECK(!failed);
    }
    return 0;
  }
  if (!strcmp(argv[1], "thread_error")) {
    pthread_t worker;
    CHECK(pthread_create(&worker, NULL, fail_in_worker, NULL) == 0);
    CHECK(pthread_join(worker, NULL) == 0);
    CHECK(worker_error == cudaErrorInvalidValue);
    CHECK(cudaPeekAtLastError() == cudaSuccess);
    return 0;
  }
  if (!strcmp(argv[1], "local_error")) {
    CHECK(cudaSetDevice(42) == cudaErrorInvalidDevice);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidDevice);
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    CHECK(cudaPeekAtLastError() == cudaSuccess);
    return 0;
  }
  unsigned char *device;
  unsigned char value = 37, output = 0;
  CHECK(cudaMalloc((void **)&device, 4) == cudaSuccess);
  CHECK(cudaMemcpy(device, &value, 1, cudaMemcpyHostToDevice) == cudaSuccess);
  if (!strcmp(argv[1], "zero_copy")) {
    unsigned char *second;
    CHECK(cudaMalloc((void **)&second, 4) == cudaSuccess);
    CHECK(cudaMemcpy(second, device, 0, cudaMemcpyDeviceToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(&output, device, 0, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(&output, &value, 0, cudaMemcpyHostToHost) == cudaSuccess);
    CHECK(cudaMemcpy(device, &output, 0, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(&output, device, 0, cudaMemcpyDefault) == cudaSuccess);
    CHECK(output == 0);
    CHECK(cudaMemcpy(&output, device, 1, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(output == value);
    CHECK(cudaFree(device) == cudaSuccess);
    CHECK(cudaFree(second) == cudaSuccess);
    return 0;
  }
  if (!strcmp(argv[1], "out_of_bounds")) {
    CHECK(cudaMemcpy(&output, device + 4, 1, cudaMemcpyDeviceToHost) == cudaErrorInvalidValue);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue);
    return 0;
  }
  if (!strcmp(argv[1], "after_free")) {
    CHECK(cudaFree(device) == cudaSuccess);
    CHECK(cudaMemcpy(&output, device, 1, cudaMemcpyDeviceToHost) == cudaErrorInvalidValue);
    return 0;
  }
  if (!strcmp(argv[1], "failed_free")) {
    CHECK(cudaFree(device) == cudaErrorInvalidValue);
    CHECK(cudaMemcpy(&output, device, 1, cudaMemcpyDefault) == cudaSuccess);
    CHECK(output == value);
    CHECK(cudaFree(device) == cudaSuccess);
    return 0;
  }
  if (!strcmp(argv[1], "copy_kinds")) {
    unsigned char *second;
    CHECK(cudaMalloc((void **)&second, 4) == cudaSuccess);
    CHECK(cudaMemcpy(second, device, 1, cudaMemcpyDeviceToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(&output, second, 1, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(output == value);
    output = 0;
    CHECK(cudaMemcpy(&output, &value, 1, cudaMemcpyHostToHost) == cudaSuccess);
    CHECK(output == value);
    value = 83;
    CHECK(cudaMemcpy(second, &value, 1, cudaMemcpyDefault) == cudaSuccess);
    CHECK(cudaMemcpy(device, second, 1, cudaMemcpyDefault) == cudaSuccess);
    CHECK(cudaMemcpy(&output, device, 1, cudaMemcpyDefault) == cudaSuccess);
    CHECK(output == value);
    CHECK(cudaMemcpy(&output, &value, 1, (enum cudaMemcpyKind)99) == cudaErrorInvalidMemcpyDirection);
    CHECK(cudaFree(device) == cudaSuccess);
    CHECK(cudaFree(second) == cudaSuccess);
    return 0;
  }
  return 2;
}
'''


@pytest.fixture(scope="module")
def runtime_client(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    include = cuda_include("cuda_runtime_api.h")
    if compiler is None or include is None:
        pytest.skip("native runtime regressions need a C compiler and CUDA toolkit headers")
    build = tmp_path_factory.mktemp("runtime-shim")
    library = build / "libwarplens_runtime.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", f"-I{include}",
                    str(ROOT / "warplens/shim/cudart_shim.c"), "-lpthread", "-ldl", "-o", str(library)],
                   check=True, capture_output=True, text=True)
    source = build / "client.c"
    source.write_text(CLIENT)
    executable = build / "client"
    subprocess.run([compiler, f"-I{include}", f"-I{ROOT / 'warplens/shim'}", str(source), str(library), "-lpthread",
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


def run_client(executable, mode, server=None, model="B200"):
    server = server or ShimServer()
    ours, theirs = socket.socketpair()
    thread = threading.Thread(target=server.serve, args=(ours,), daemon=True)
    thread.start()
    try:
        done = subprocess.run([str(executable), mode], pass_fds=[theirs.fileno()],
                              env={**os.environ, "WARPLENS_SHIM_FD": str(theirs.fileno()), "WARPLENS_MODEL": model},
                              capture_output=True, text=True, timeout=15)
    finally:
        theirs.close()
        thread.join(timeout=5)
        ours.close()
    assert done.returncode == 0, f"native client exited {done.returncode}:\n{done.stderr}"


@pytest.mark.parametrize("mode", ["out_of_bounds", "after_free", "copy_kinds", "zero_copy"])
def test_memcpy_uses_the_requested_direction(runtime_client, mode):
    run_client(runtime_client, mode)


def test_last_error_is_local_to_the_calling_host_thread(runtime_client):
    run_client(runtime_client, "thread_error")


def test_local_runtime_failures_update_last_error(runtime_client):
    run_client(runtime_client, "local_error")


def test_failed_free_preserves_pointer_classification(runtime_client):
    class RejectFirstFree(ShimServer):
        failed = False

        def handle(self, op, data):
            if op == FREE and not self.failed:
                self.failed = True
                raise WarplensError("free failed before releasing the allocation")
            return super().handle(op, data)

    run_client(runtime_client, "failed_free", RejectFirstFree())


def test_allocations_and_default_copies_from_concurrent_host_threads(runtime_client):
    run_client(runtime_client, "thread_allocations")


@pytest.mark.parametrize("model", ["B200", "B300"])
def test_device_properties_match_the_selected_model(runtime_client, model):
    run_client(runtime_client, "properties", model=model)


@pytest.mark.parametrize("mode", ["oversized_part", "oversized_sum"])
def test_oversized_request_is_rejected_before_accessing_payload(runtime_client, mode):
    class RecordRequests(ShimServer):
        def __init__(self):
            super().__init__()
            self.requests = []

        def handle(self, op, data):
            self.requests.append(op)
            return super().handle(op, data)

    server = RecordRequests()
    run_client(runtime_client, mode, server)
    assert server.requests == [MALLOC, FREE]
