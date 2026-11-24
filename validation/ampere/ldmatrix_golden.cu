// Host program: record ldmatrix and bfe results from the GPU.
//   "L variant lane r0 r1 r2 r3"   and   "B a pos len unsigned signed"   (hex)
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "ldmatrix_probe.cu"

int main() {
    unsigned* d_out;
    std::vector<unsigned> h(6 * 32 * 4);
    CHECK(cudaMalloc(&d_out, h.size() * 4));
    ldmatrix_probe<<<1, 32>>>(d_out);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(h.data(), d_out, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int v = 0; v < 6; ++v)
        for (int l = 0; l < 32; ++l)
            printf("L %d %d %x %x %x %x\n", v, l, h[(v * 32 + l) * 4], h[(v * 32 + l) * 4 + 1],
                   h[(v * 32 + l) * 4 + 2], h[(v * 32 + l) * 4 + 3]);

    std::vector<unsigned> a, pos, len;
    const unsigned values[] = {0x00000000u, 0xFFFFFFFFu, 0x80000001u, 0x12345678u, 0xDEADBEEFu, 0x7FFFFFFFu};
    const unsigned positions[] = {0, 1, 6, 15, 31, 32, 33, 40, 255, 256, 261};
    const unsigned lengths[] = {0, 1, 2, 8, 31, 32, 33, 64, 255, 256, 258};
    for (unsigned v : values) for (unsigned p : positions) for (unsigned l : lengths) {
        a.push_back(v); pos.push_back(p); len.push_back(l);
    }
    int n = (int)a.size();
    std::vector<unsigned> out(2 * n);
    unsigned *da, *dp, *dl, *dout;
    CHECK(cudaMalloc(&da, n * 4)); CHECK(cudaMalloc(&dp, n * 4)); CHECK(cudaMalloc(&dl, n * 4)); CHECK(cudaMalloc(&dout, 2 * n * 4));
    CHECK(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dp, pos.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dl, len.data(), n * 4, cudaMemcpyHostToDevice));
    bfe_probe<<<(n + 255) / 256, 256>>>(da, dp, dl, dout, n);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), dout, 2 * n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) printf("B %x %x %x %x %x\n", a[i], pos[i], len[i], out[i], out[n + i]);
    return 0;
}
