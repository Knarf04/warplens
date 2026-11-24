// Host program: run the kernels of addrwrap_probe.cu on the GPU, one CTA of 64 threads each,
// and print one line per kernel: its name and out[0..63].
#include <cstdio>
#include <cuda_runtime.h>
#include "host_check.h"
#include "addrwrap_probe.cu"

int main() {
    const int n = ADDRWRAP_N;
    int fill[ADDRWRAP_N], out[ADDRWRAP_N];
    for (int i = 0; i < n; ++i) fill[i] = -99;  // what a thread that stored nothing leaves
    int* device;
    CHECK(cudaMalloc(&device, sizeof out));
#define ADDRWRAP_RUN(kernel)                                                      \
    do {                                                                          \
        CHECK(cudaMemcpy(device, fill, sizeof fill, cudaMemcpyHostToDevice));     \
        kernel<<<1, n>>>(device);                                                 \
        CHECK_LAUNCH();                                                           \
        CHECK(cudaMemcpy(out, device, sizeof out, cudaMemcpyDeviceToHost));       \
        printf(#kernel);                                                          \
        for (int i = 0; i < n; ++i) printf(" %d", out[i]);                        \
        printf("\n");                                                             \
    } while (0)
    ADDRWRAP_RUN(addrwrap_cxx_load);
    ADDRWRAP_RUN(addrwrap_cxx_store);
    ADDRWRAP_RUN(addrwrap_ptx_load);
    ADDRWRAP_RUN(addrwrap_ptx_store);
    ADDRWRAP_RUN(addrwrap_ptx_far);
    ADDRWRAP_RUN(addrwrap_ptx_back);
    return 0;
}
