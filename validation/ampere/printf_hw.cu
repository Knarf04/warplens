// Host program: run printf_probe on the GPU. Prints the kernel's own output,
// then the result array.
#include <cstdio>
#include <cuda_runtime.h>
#include "host_check.h"
#include "printf_probe.cu"

int main() {
    int h_table[4] = {3, 1, 4, 1}, h_out[8];
    int *d_table, *d_out;
    CHECK(cudaMalloc(&d_table, sizeof h_table));
    CHECK(cudaMalloc(&d_out, sizeof h_out));
    CHECK(cudaMemcpy(d_table, h_table, sizeof h_table, cudaMemcpyHostToDevice));
    Config cfg = {100, 2.5f, d_table, (short)-7};
    printf_probe<<<1, 8>>>(cfg, d_out);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));
    printf("out:");
    for (int v : h_out) printf(" %d", v);
    printf("\n");
    return 0;
}
