// Host program: record bfi and shf results from the GPU.
//   "a b pos len bfi32 bfi64lo bfi64hi shf.l.wrap shf.r.wrap shf.l.clamp shf.r.clamp"   (hex)
// `pos` is also the shift amount of the four shf forms.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "bitfield_probe.cu"

int main() {
    const u32 values[] = {0x00000000u, 0xFFFFFFFFu, 0x80000001u, 0x12345678u, 0xDEADBEEFu};
    const u32 others[] = {0x00000000u, 0xFFFFFFFFu, 0x0F0F1234u};
    const u32 positions[] = {0, 1, 7, 16, 31, 32, 33, 63, 64, 65, 255, 256, 257, 0x120, 0xFFFFFFFFu};
    const u32 lengths[] = {0, 1, 8, 31, 32, 33, 63, 64, 65, 255, 256, 0x108, 0xFFFFFFFFu};
    std::vector<u32> a, b, pos, len;
    for (u32 x : values) for (u32 y : others) for (u32 p : positions) for (u32 l : lengths) {
        a.push_back(x); b.push_back(y); pos.push_back(p); len.push_back(l);
    }
    int n = (int)a.size();
    std::vector<u32> out(7 * n);
    u32 *da, *db, *dp, *dl, *dout;
    CHECK(cudaMalloc(&da, n * 4)); CHECK(cudaMalloc(&db, n * 4)); CHECK(cudaMalloc(&dp, n * 4));
    CHECK(cudaMalloc(&dl, n * 4)); CHECK(cudaMalloc(&dout, 7 * n * 4));
    CHECK(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dp, pos.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dl, len.data(), n * 4, cudaMemcpyHostToDevice));
    bitfield_probe<<<(n + 255) / 256, 256>>>(da, db, dp, dl, dout, n);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), dout, 7 * n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        printf("%x %x %x %x", a[i], b[i], pos[i], len[i]);
        for (int k = 0; k < 7; ++k) printf(" %x", out[7 * i + k]);
        printf("\n");
    }
    return 0;
}
