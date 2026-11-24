// Host program: record cpasync_probe from the GPU, one line of 16 hex bytes per lane,
// preceded by the lane's source size.
#include <cstdio>
#include <cuda_runtime.h>
#include "host_check.h"
#include "cpasync_probe.cu"

int main() {
    unsigned char h_in[512], h_out[512];
    unsigned h_size[32];
    for (int i = 0; i < 512; ++i) h_in[i] = (unsigned char)(i * 7 + 1);
    for (int l = 0; l < 32; ++l) h_size[l] = l <= 16 ? l : (l % 5) * 4;  // 0..16, then 0,4,8,12,16
    unsigned char *d_in, *d_out; unsigned* d_size;
    CHECK(cudaMalloc(&d_in, 512)); CHECK(cudaMalloc(&d_out, 512)); CHECK(cudaMalloc(&d_size, sizeof h_size));
    CHECK(cudaMemcpy(d_in, h_in, 512, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_size, h_size, sizeof h_size, cudaMemcpyHostToDevice));
    cpasync_probe<<<1, 32>>>(d_in, d_size, d_out);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(h_out, d_out, 512, cudaMemcpyDeviceToHost));
    for (int l = 0; l < 32; ++l) {
        printf("%u", h_size[l]);
        for (int k = 0; k < 16; ++k) printf(" %02x", h_out[l * 16 + k]);
        printf("\n");
    }
    return 0;
}
