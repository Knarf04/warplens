// Host program: run call_probe on the GPU and print "a b c d e" for each of 256 threads.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "call_probe.cu"

int main() {
    const int n = 256;
    std::vector<int> out(5 * n);
    int* device;
    CHECK(cudaMalloc(&device, 5 * n * sizeof(int)));
    call_probe<<<n / 64, 64>>>(device);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), device, 5 * n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        printf("%d %d %d %d %d\n", out[5 * i], out[5 * i + 1], out[5 * i + 2], out[5 * i + 3], out[5 * i + 4]);
    return 0;
}
