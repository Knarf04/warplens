// Host program: run the three transpose kernels from examples/01_smem_transpose
// on the GPU and check their output. Profile it with ncu to read the
// shared-memory counters (see run_ncu.sh).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "../../examples/01_smem_transpose/transpose.cu"

typedef void (*Kernel)(const float*, float*);

int main(int argc, char** argv) {
    int ctas = argc > 1 ? atoi(argv[1]) : 4;
    size_t n = (size_t)ctas * TILE * TILE;
    std::vector<float> in(n), out(n);
    for (size_t i = 0; i < n; ++i) in[i] = (float)i;
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const char* names[] = {"transpose_linear", "transpose_padded", "transpose_swizzled"};
    Kernel kernels[] = {transpose_linear, transpose_padded, transpose_swizzled};
    int bad = 0;
    for (int k = 0; k < 3; ++k) {
        CHECK(cudaMemset(d_out, 0, n * sizeof(float)));
        kernels[k]<<<ctas, dim3(TILE, TILE)>>>(d_in, d_out);
        CHECK_LAUNCH();
        CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t wrong = 0;
        for (int c = 0; c < ctas; ++c)
            for (int y = 0; y < TILE; ++y)
                for (int x = 0; x < TILE; ++x)
                    wrong += out[(c * TILE + y) * TILE + x] != in[(c * TILE + x) * TILE + y];
        printf("%s: %s\n", names[k], wrong ? "WRONG" : "ok");
        bad += wrong != 0;
    }
    return bad;
}
