// Host program: record integer `div` and `rem` from the GPU as "instruction a b result",
// 64-bit hex patterns with a narrower value in the low bits.
//
//   div rem   .s16 .u16 .s32 .u32 .s64 .u64
//
// over all pairs of: 0, 1, 2, 3, 7, 10, 100, the same negated, the type's minimum and maximum
// and their neighbours, and a few patterns. So the file has every dividend over zero and the
// minimum over -1, the two cases the PTX manual gives no result for, among ordinary divisions.
#include <cstdio>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"

typedef unsigned long long u64;

#define OP16(id, text) \
    case id: { unsigned short x = (unsigned short)a[i], y = (unsigned short)b[i], r = 0; \
               asm volatile(text " %0, %1, %2;" : "=h"(r) : "h"(x), "h"(y)); out[i] = r; break; }
#define OP32(id, text) \
    case id: { unsigned x = (unsigned)a[i], y = (unsigned)b[i], r = 0; \
               asm volatile(text " %0, %1, %2;" : "=r"(r) : "r"(x), "r"(y)); out[i] = r; break; }
#define OP64(id, text) \
    case id: { u64 x = a[i], y = b[i], r = 0; \
               asm volatile(text " %0, %1, %2;" : "=l"(r) : "l"(x), "l"(y)); out[i] = r; break; }

__global__ void intdiv_probe(const int* op, const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (op[i]) {
        OP16(0, "div.s16") OP16(1, "div.u16") OP32(2, "div.s32") OP32(3, "div.u32") OP64(4, "div.s64") OP64(5, "div.u64")
        OP16(6, "rem.s16") OP16(7, "rem.u16") OP32(8, "rem.s32") OP32(9, "rem.u32") OP64(10, "rem.s64") OP64(11, "rem.u64")
    }
}

int main() {
    const char* names[] = {"div.s16", "div.u16", "div.s32", "div.u32", "div.s64", "div.u64",
                           "rem.s16", "rem.u16", "rem.s32", "rem.u32", "rem.s64", "rem.u64"};
    const int bits[] = {16, 16, 32, 32, 64, 64};
    std::vector<int> op;
    std::vector<u64> a, b;
    for (int o = 0; o < 12; ++o) {
        int w = bits[o % 6];
        u64 mask = w == 64 ? ~0ull : (1ull << w) - 1, top = 1ull << (w - 1);
        std::vector<u64> v;
        for (long long x : {0, 1, 2, 3, 7, 10, 100}) { v.push_back((u64)x & mask); if (x) v.push_back((u64)-x & mask); }
        for (u64 x : {top, top + 1, top + 2, top - 1, top - 2, mask - 1,
                      0x123456789abcdef0ull & mask, 0xfedcba9876543210ull & mask, 0x00000001ffffffffull & mask})
            v.push_back(x);
        for (u64 x : v) for (u64 y : v) { op.push_back(o); a.push_back(x); b.push_back(y); }
    }
    int n = (int)op.size();
    std::vector<u64> out(n);
    int* dop; u64 *da, *db, *dout;
    CHECK(cudaMalloc(&dop, n * 4)); CHECK(cudaMalloc(&da, n * 8)); CHECK(cudaMalloc(&db, n * 8)); CHECK(cudaMalloc(&dout, n * 8));
    CHECK(cudaMemcpy(dop, op.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dout, 0, n * 8));
    intdiv_probe<<<(n + 255) / 256, 256>>>(dop, da, db, dout, n);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), dout, n * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) printf("%s %016llx %016llx %016llx\n", names[op[i]], a[i], b[i], out[i]);
    return 0;
}
