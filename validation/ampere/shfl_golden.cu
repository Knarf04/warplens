// Host program: run shfl_probe on the GPU for a grid of parameters and print
// one line per case: "mode b width r0 r1 ... r31".
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "host_check.h"
#include "shfl_probe.cu"

int main() {
    int* d_out;
    int h_out[32];
    CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    const int widths[] = {2, 4, 8, 16, 32};
    const int extra_b[] = {32, 33, 47, 63};
    for (int mode = 0; mode < 4; ++mode)
        for (int width : widths)
            for (int k = 0; k < 32 + 4; ++k) {
                int b = k < 32 ? k : extra_b[k - 32];
                shfl_probe<<<1, 32>>>(d_out, mode, b, width);
                CHECK_LAUNCH();
                CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
                printf("%d %d %d", mode, b, width);
                for (int v : h_out) printf(" %d", v);
                printf("\n");
            }
    return 0;
}
