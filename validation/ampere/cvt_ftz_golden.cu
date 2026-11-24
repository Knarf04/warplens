// Host program: record `cvt` with `.ftz` from the GPU, around the subnormal range, as
// "instruction a result" with 64-bit hex patterns (a narrower value sits in the low bits).
// Each `.ftz` instruction is recorded next to the same instruction without it.
//
//   cvt[.ftz].f64.f32   cvt.rn[.ftz].f32.f64      what `(double)x` and `(float)x` compile to,
//                                                 with .ftz under --use_fast_math
//   cvt.rn[.ftz].f16.f32  cvt[.ftz].f32.f16
//   cvt.rn.f16.f64  cvt.f64.f16                   no .ftz: ptxas takes it only when the source
//                                                 or the result is f32 ("Modifier '.ftz' does
//                                                 not apply to this type of convert operation")
//   cvt[.ftz].f32.f32
//   cvt.{rni,rzi,rmi,rpi}[.ftz].f32.f32           rintf, truncf, floorf, ceilf
//   cvt.{rni,rzi,rmi,rpi}.f64.f64                 rint, trunc, floor, ceil
//   cvt.{rni,rzi,rmi,rpi}.ftz.{s32,u32,s64,u64}.f32   `(int)x` under --use_fast_math
//
// (validation/ampere/minmax_cvt_golden.cu has float to integer without .ftz, and
// nan_golden.cu the NaNs of the conversions between f16, f32 and f64.)
//
// The values of each source type: zeros, the smallest and largest subnormals and some
// between, the smallest normal numbers, ordinary numbers and halves (for the rounding
// modes), the largest finite numbers, infinities and NaNs; of either sign. And the values
// that become subnormal, or nearly so, in a narrower type: for f64 to f32, powers of two down
// to half the smallest f32 subnormal, ties between two subnormals, and the numbers between
// the largest subnormal and the smallest normal number 2^-126: its midpoint with the largest
// subnormal, 2^-126 (1 - 2^-25) (below which a result is "tiny after rounding", see
// docs/numerics.md) and their neighbours. Likewise for f32 and f64 to f16 around 2^-14.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"

typedef unsigned long long u64;

#define CVT_IN_H "h"(xh)
#define CVT_IN_F "f"(xf)
#define CVT_IN_D "d"(xd)
#define CVT_OUT_H "=h"(rh)
#define CVT_OUT_F "=f"(rf)
#define CVT_OUT_D "=d"(rd)
#define CVT_OUT_I "=r"(ri)
#define CVT_OUT_L "=l"(rl)
#define CVT_PUT_H out[i] = rh
#define CVT_PUT_F out[i] = __float_as_uint(rf)
#define CVT_PUT_D out[i] = (u64)__double_as_longlong(rd)
#define CVT_PUT_I out[i] = (unsigned)ri
#define CVT_PUT_L out[i] = (u64)rl

// X(text, source, result): H, F, D are f16, f32, f64; I and L are 32- and 64-bit integers.
#define CVT_MODES(X, tail, S, R) \
    X("cvt.rni" tail, S, R) X("cvt.rzi" tail, S, R) X("cvt.rmi" tail, S, R) X("cvt.rpi" tail, S, R)
#define CVT_OPS(X)                                                                          \
    X("cvt.f64.f32", F, D) X("cvt.ftz.f64.f32", F, D)                                       \
    X("cvt.rn.f32.f64", D, F) X("cvt.rn.ftz.f32.f64", D, F)                                 \
    X("cvt.rn.f16.f32", F, H) X("cvt.rn.ftz.f16.f32", F, H)                                 \
    X("cvt.f32.f16", H, F) X("cvt.ftz.f32.f16", H, F)                                       \
    X("cvt.rn.f16.f64", D, H) X("cvt.f64.f16", H, D)                                        \
    X("cvt.f32.f32", F, F) X("cvt.ftz.f32.f32", F, F)                                       \
    CVT_MODES(X, ".f32.f32", F, F) CVT_MODES(X, ".ftz.f32.f32", F, F)                       \
    CVT_MODES(X, ".f64.f64", D, D)                                                          \
    CVT_MODES(X, ".ftz.s32.f32", F, I) CVT_MODES(X, ".ftz.u32.f32", F, I)                   \
    CVT_MODES(X, ".ftz.s64.f32", F, L) CVT_MODES(X, ".ftz.u64.f32", F, L)

#define CVT_CASE(text, S, R) \
    case __COUNTER__ - cvt_first: asm volatile(text " %0, %1;" : CVT_OUT_##R : CVT_IN_##S); CVT_PUT_##R; break;

__global__ void cvt_ftz_probe(const int* op, const u64* a, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xf = __uint_as_float((unsigned)a[i]), rf = 0.0f;
    double xd = __longlong_as_double((long long)a[i]), rd = 0.0;
    unsigned short xh = (unsigned short)a[i], rh = 0;
    int ri = 0;
    long long rl = 0;
    enum { cvt_first = __COUNTER__ + 1 };
    switch (op[i]) {
        CVT_OPS(CVT_CASE)
    }
}

#define CVT_NAME(text, S, R) names.push_back(text); source.push_back(#S[0]);

int main() {
    std::vector<std::string> names;
    std::vector<char> source;
    CVT_OPS(CVT_NAME)

    std::vector<u64> v16, v32, v64;
    auto both = [](std::vector<u64>& v, std::initializer_list<u64> magnitudes, int bits) {
        for (u64 m : magnitudes) { v.push_back(m); v.push_back(m | 1ull << (bits - 1)); }
    };
    // f16: 0, subnormals, the smallest normal numbers, ordinary, the largest, infinity, NaNs
    both(v16, {0x0000, 0x0001, 0x0002, 0x0155, 0x0200, 0x03ff, 0x0400, 0x0401, 0x3400, 0x3800, 0x3a00,
               0x3c00, 0x3e00, 0x4100, 0x4500, 0x7bff, 0x7c00, 0x7c01, 0x7d00, 0x7e00, 0x7e12, 0x7fff}, 16);
    // f32
    both(v32, {0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00012345, 0x00400000, 0x00555555,
               0x007ffffe, 0x007fffff, 0x00800000, 0x00800001, 0x00ffffff, 0x01000000,
               // around the f16 subnormals: 2^-25 and its neighbours (half the smallest), 2^-24,
               // ties, the largest subnormal 2^-14 (1 - 2^-10) up to 2^-14
               0x32ffffff, 0x33000000, 0x33000001, 0x33800000, 0x33c00000, 0x34000000, 0x34400000,
               0x36000000, 0x38000000, 0x387fc000, 0x387fdfff, 0x387fe000, 0x387fe001, 0x387fefff,
               0x387ff000, 0x387ff001, 0x387fffff, 0x38800000, 0x38800001, 0x38802000,
               0x3e800000, 0x3f000000, 0x3f400000, 0x3f7fffff, 0x3f800000, 0x3fc00000, 0x40200000,
               0x40600000, 0x4b000001, 0x501502f9, 0x477fe000, 0x477ff000, 0x7f7fffff,
               0x7f800000, 0x7f800001, 0x7fa00000, 0x7fc00000, 0x7fc12345, 0x7fffffff}, 32);
    // f64
    both(v64, {0x0000000000000000ull, 0x0000000000000001ull, 0x0008000000000000ull, 0x000fffffffffffffull,
               0x0010000000000000ull, 0x0010000000000001ull,
               // around the f32 subnormals: 2^-150 and its neighbours (half the smallest), 2^-149,
               // ties, exact subnormals
               0x368fffffffffffffull, 0x3690000000000000ull, 0x3690000000000001ull, 0x36a0000000000000ull,
               0x36a8000000000000ull, 0x36a8000000000001ull, 0x36b0000000000000ull, 0x36b4000000000000ull,
               0x3725a2b3c0000000ull, 0x37f123456789abcdull, 0x3800000000000000ull, 0x3800000010000000ull,
               0x3800000030000000ull, 0x3808000000000000ull,
               // the largest f32 subnormal 2^-126 (1 - 2^-23) up to 2^-126
               0x380fffffc0000000ull, 0x380fffffc0000001ull, 0x380fffffdfffffffull, 0x380fffffe0000000ull,
               0x380fffffe0000001ull, 0x380fffffefffffffull, 0x380ffffff0000000ull, 0x380ffffff0000001ull,
               0x380fffffffffffffull, 0x3810000000000000ull, 0x3810000000000001ull, 0x3810000010000000ull,
               0x3810000030000000ull,
               // around the f16 subnormals
               0x3e5fffffffffffffull, 0x3e60000000000000ull, 0x3e60000000000001ull, 0x3e70000000000000ull,
               0x3e78000000000000ull, 0x3f0ff80000000000ull, 0x3f0ffbffffffffffull, 0x3f0ffc0000000000ull,
               0x3f0ffc0000000001ull, 0x3f0ffdffffffffffull, 0x3f0ffe0000000000ull, 0x3f0ffe0000000001ull,
               0x3f0fffffffffffffull, 0x3f10000000000000ull,
               0x3fd0000000000000ull, 0x3fe0000000000000ull, 0x3fe8000000000000ull, 0x3ff0000000000000ull,
               0x3ff8000000000000ull, 0x4004000000000000ull, 0x400c000000000000ull, 0x3fb999999999999aull,
               0x4202a05f20000000ull, 0x40effc0000000000ull, 0x40effe0000000000ull,
               0x47efffffe0000000ull, 0x47efffffefffffffull, 0x47effffff0000000ull, 0x47f0000000000000ull,
               0x7fefffffffffffffull, 0x7ff0000000000000ull, 0x7ff0000000000001ull, 0x7ff4000000000000ull,
               0x7ff8000000000000ull, 0x7ff8000012345678ull, 0x7ffd432187654321ull, 0x7fffffffffffffffull}, 64);
    unsigned state = 2026u;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return (u64)state; };
    for (int k = 0; k < 48; ++k) {
        v32.push_back(next() & 0x807fffffu);                             // subnormal
        v32.push_back((next() & 0x807fffffu) | (u64)(88 + k % 28) << 23);  // 2^-39 to 2^-12: around f16's
        u64 hi = next();                                                 // 2^-153 to 2^-122: around f32's
        v64.push_back(((hi & 0x800fffffu) | (u64)(870 + k % 32) << 20) << 32 | next());
        hi = next();                                                     // the same, with few bits set
        v64.push_back(((hi & 0x800fffffu) | (u64)(870 + k % 32) << 20) << 32 | (next() & 0xe0000000u));
        hi = next();                                                     // 2^-28 to 2^-12
        v64.push_back(((hi & 0x800fffffu) | (u64)(995 + k % 17) << 20) << 32 | next());
    }

    std::vector<int> op;
    std::vector<u64> a;
    for (int o = 0; o < (int)names.size(); ++o)
        for (u64 x : source[o] == 'H' ? v16 : source[o] == 'F' ? v32 : v64) { op.push_back(o); a.push_back(x); }

    int n = (int)op.size();
    std::vector<u64> out(n);
    int* dop; u64 *da, *dout;
    CHECK(cudaMalloc(&dop, n * 4)); CHECK(cudaMalloc(&da, n * 8)); CHECK(cudaMalloc(&dout, n * 8));
    CHECK(cudaMemcpy(dop, op.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dout, 0, n * 8));
    cvt_ftz_probe<<<(n + 255) / 256, 256>>>(dop, da, dout, n);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), dout, n * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        printf("%s %016llx %016llx\n", names[op[i]].c_str(), a[i], out[i]);
    return 0;
}
