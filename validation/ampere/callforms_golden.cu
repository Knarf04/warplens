// Host program: run callforms_probe on the GPU and print its twelve values for each of 256 threads.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "callforms_probe.cu"

int main() {
    const int n = 256, width = 12;
    std::vector<int> out(width * n);
    int* device;
    cudaMalloc(&device, width * n * sizeof(int));
    callforms_probe<<<n / 64, 64>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
    cudaMemcpy(out.data(), device, width * n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < width; ++k) printf("%d%c", out[width * i + k], k + 1 < width ? ' ' : '\n');
    return 0;
}
