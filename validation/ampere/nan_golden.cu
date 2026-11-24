// Host program: record what the GPU's floating-point instructions return when the result is
// a NaN, as "instruction a b c result" with 64-bit hex patterns (a narrower value sits in the
// low bits; operands an instruction does not have are 0).
//
//   add sub mul div      .f32, .rn.f32, .rn.ftz.f32, .f64, .rn.f64    all pairs of the values below
//   fma                  .rn.f32, .rn.ftz.f32, .rn.f64                all triples of a smaller set
//   sqrt                 .rn.f32, .rn.ftz.f32, .rn.f64                every value
//   neg abs mov          .f32, .f64                                   every value
//   cvt                  .f64.f32, .rn.f32.f64, .rn.ftz.f32.f64, .ftz.f64.f32,
//                        .rn.f16.f32, .f32.f16, .rn.f16.f64, .f64.f16  every value of the source type
//
// The values: zeros, ones, infinities, a subnormal, a few ordinary numbers, and NaNs: quiet
// and signalling, of either sign, with and without a payload, and the all-ones pattern. So
// the file holds every invalid operation (inf - inf, 0 * inf, 0 / 0, inf / inf, sqrt(-1),
// fma(0, inf, c), fma(inf, 1, -inf)) and a NaN in every operand position.
//
// The two conversions with .ftz (cvt.ftz.f64.f32, cvt.rn.ftz.f32.f64; 44 lines) are in a
// kernel of their own, nan_cvt_ftz_probe, because the simulator does not implement them and
// refuses a kernel that contains one. A GPU does not care, and the file is the same as when
// one kernel held everything. validation/blackwell/formats/ampere_replay.cu includes this
// file and runs the two parts as separate cases (nan_run below).
#include <cstdio>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"

typedef unsigned long long u64;

#define BIN32(id, text) \
    case id: asm volatile(text " %0, %1, %2;" : "=f"(rf) : "f"(xf), "f"(yf)); out[i] = __float_as_uint(rf); break;
#define BIN64(id, text) \
    case id: asm volatile(text " %0, %1, %2;" : "=d"(rd) : "d"(xd), "d"(yd)); out[i] = (u64)__double_as_longlong(rd); break;
#define TRI32(id, text) \
    case id: asm volatile(text " %0, %1, %2, %3;" : "=f"(rf) : "f"(xf), "f"(yf), "f"(zf)); out[i] = __float_as_uint(rf); break;
#define TRI64(id, text) \
    case id: asm volatile(text " %0, %1, %2, %3;" : "=d"(rd) : "d"(xd), "d"(yd), "d"(zd)); out[i] = (u64)__double_as_longlong(rd); break;
#define UN32(id, text) \
    case id: asm volatile(text " %0, %1;" : "=f"(rf) : "f"(xf)); out[i] = __float_as_uint(rf); break;
#define UN64(id, text) \
    case id: asm volatile(text " %0, %1;" : "=d"(rd) : "d"(xd)); out[i] = (u64)__double_as_longlong(rd); break;

#define OPS(X32, X64, T32, T64, U32, U64)                                                       \
    X32(0, "add.f32") X32(1, "add.rn.f32") X32(2, "add.rn.ftz.f32") X64(3, "add.f64") X64(4, "add.rn.f64")   \
    X32(5, "sub.f32") X32(6, "sub.rn.f32") X32(7, "sub.rn.ftz.f32") X64(8, "sub.f64") X64(9, "sub.rn.f64")   \
    X32(10, "mul.f32") X32(11, "mul.rn.f32") X32(12, "mul.rn.ftz.f32") X64(13, "mul.f64") X64(14, "mul.rn.f64") \
    X32(15, "div.rn.f32") X32(16, "div.rn.ftz.f32") X64(17, "div.rn.f64")                         \
    T32(18, "fma.rn.f32") T32(19, "fma.rn.ftz.f32") T64(20, "fma.rn.f64")                         \
    U32(21, "sqrt.rn.f32") U32(22, "sqrt.rn.ftz.f32") U64(23, "sqrt.rn.f64")                      \
    U32(24, "neg.f32") U64(25, "neg.f64") U32(26, "abs.f32") U64(27, "abs.f64")                   \
    U32(28, "mov.f32") U64(29, "mov.f64")
#define FIRST_CVT 30

__global__ void nan_probe(const int* op, const u64* a, const u64* b, const u64* c, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xf = __uint_as_float((unsigned)a[i]), yf = __uint_as_float((unsigned)b[i]),
          zf = __uint_as_float((unsigned)c[i]), rf = 0.0f;
    double xd = __longlong_as_double((long long)a[i]), yd = __longlong_as_double((long long)b[i]),
           zd = __longlong_as_double((long long)c[i]), rd = 0.0;
    unsigned short xh = (unsigned short)a[i], rh = 0;
    switch (op[i]) {
        OPS(BIN32, BIN64, TRI32, TRI64, UN32, UN64)
        case 30: asm volatile("cvt.f64.f32 %0, %1;" : "=d"(rd) : "f"(xf)); out[i] = (u64)__double_as_longlong(rd); break;
        case 32: asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(rf) : "d"(xd)); out[i] = __float_as_uint(rf); break;
        case 34: asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(rh) : "f"(xf)); out[i] = rh; break;
        case 35: asm volatile("cvt.rn.f16.f64 %0, %1;" : "=h"(rh) : "d"(xd)); out[i] = rh; break;
        case 36: asm volatile("cvt.f32.f16 %0, %1;" : "=f"(rf) : "h"(xh)); out[i] = __float_as_uint(rf); break;
        case 37: asm volatile("cvt.f64.f16 %0, %1;" : "=d"(rd) : "h"(xh)); out[i] = (u64)__double_as_longlong(rd); break;
    }
}

// Operations 31 and 33, which nan_probe leaves alone.
__global__ void nan_cvt_ftz_probe(const int* op, const u64* a, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xf = __uint_as_float((unsigned)a[i]), rf = 0.0f;
    double xd = __longlong_as_double((long long)a[i]), rd = 0.0;
    switch (op[i]) {
        case 31: asm volatile("cvt.ftz.f64.f32 %0, %1;" : "=d"(rd) : "f"(xf)); out[i] = (u64)__double_as_longlong(rd); break;
        case 33: asm volatile("cvt.rn.ftz.f32.f64 %0, %1;" : "=f"(rf) : "d"(xd)); out[i] = __float_as_uint(rf); break;
    }
}
static bool is_cvt_ftz(int op) { return op == 31 || op == 33; }

#define NAME(id, text) names[id] = text;
#define ARITY2(id, text) arity[id] = 2;
#define ARITY3(id, text) arity[id] = 3;
#define ARITY1(id, text) arity[id] = 1;
#define IS32(id, text) width[id] = 32;
#define IS64(id, text) width[id] = 64;

// Which lines to compute and print: everything (the golden file), everything but the two
// .ftz conversions, or only those.
enum { NAN_ALL, NAN_WITHOUT_CVT_FTZ, NAN_ONLY_CVT_FTZ };

static int nan_run(int part) {
    std::string names[38];
    int arity[38], width[38];
    OPS(NAME, NAME, NAME, NAME, NAME, NAME)
    OPS(ARITY2, ARITY2, ARITY3, ARITY3, ARITY1, ARITY1)
    OPS(IS32, IS64, IS32, IS64, IS32, IS64)
    const char* cvt[] = {"cvt.f64.f32", "cvt.ftz.f64.f32", "cvt.rn.f32.f64", "cvt.rn.ftz.f32.f64",
                         "cvt.rn.f16.f32", "cvt.rn.f16.f64", "cvt.f32.f16", "cvt.f64.f16"};
    const int cvt_width[] = {32, 32, 64, 64, 32, 64, 16, 16};
    for (int k = 0; k < 8; ++k) { names[FIRST_CVT + k] = cvt[k]; arity[FIRST_CVT + k] = 1; width[FIRST_CVT + k] = cvt_width[k]; }

    // NaNs first: quiet, signalling, either sign, with payloads, all ones.
    const std::vector<u64> v32 = {
        0x7fc00000, 0xffc00000, 0x7fc12345, 0xffd54321, 0x7f800001, 0xff800001, 0x7fa00000, 0xffbfffff,
        0x7fffffff, 0xffffffff,
        0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
        0x00000001, 0x80400000, 0x40200000, 0xc0200000, 0x7f7fffff, 0xff7fffff};
    const std::vector<u64> v64 = {
        0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff8000012345678ull, 0xfffd432187654321ull,
        0x7ff0000000000001ull, 0xfff0000000000001ull, 0x7ff4000000000000ull, 0xfff7ffffffffffffull,
        0x7fffffffffffffffull, 0xffffffffffffffffull,
        0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
        0x7ff0000000000000ull, 0xfff0000000000000ull,
        0x0000000000000001ull, 0x8008000000000000ull, 0x4004000000000000ull, 0xc004000000000000ull,
        0x7fefffffffffffffull, 0xffefffffffffffffull};
    const std::vector<u64> v16 = {0x7e00, 0xfe00, 0x7e12, 0xff45, 0x7c01, 0xfc01, 0x7d00, 0xfdff, 0x7fff, 0xffff,
                                  0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x0001, 0x8200, 0x4100, 0xc100,
                                  0x7bff, 0xfbff};
    // fma: three NaNs of different kinds, and what makes an invalid operation
    const std::vector<int> pick = {0, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 18};

    std::vector<int> op;
    std::vector<u64> a, b, c;
    auto push = [&](int o, u64 x, u64 y, u64 z) { op.push_back(o); a.push_back(x); b.push_back(y); c.push_back(z); };
    for (int o = 0; o < 38; ++o) {
        const std::vector<u64>& v = width[o] == 32 ? v32 : width[o] == 64 ? v64 : v16;
        if (arity[o] == 1) for (u64 x : v) push(o, x, 0, 0);
        if (arity[o] == 2) for (u64 x : v) for (u64 y : v) push(o, x, y, 0);
        if (arity[o] == 3) for (int x : pick) for (int y : pick) for (int z : pick) push(o, v[x], v[y], v[z]);
    }

    int n = (int)op.size();
    std::vector<u64> out(n);
    int* dop; u64 *da, *db, *dc, *dout;
    CHECK(cudaMalloc(&dop, n * 4)); CHECK(cudaMalloc(&da, n * 8)); CHECK(cudaMalloc(&db, n * 8));
    CHECK(cudaMalloc(&dc, n * 8)); CHECK(cudaMalloc(&dout, n * 8));
    CHECK(cudaMemcpy(dop, op.data(), n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dc, c.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dout, 0, n * 8));
    if (part != NAN_ONLY_CVT_FTZ) {
        nan_probe<<<(n + 255) / 256, 256>>>(dop, da, db, dc, dout, n);
        CHECK_LAUNCH();
    }
    if (part != NAN_WITHOUT_CVT_FTZ) {
        nan_cvt_ftz_probe<<<(n + 255) / 256, 256>>>(dop, da, dout, n);
        CHECK_LAUNCH();
    }
    CHECK(cudaMemcpy(out.data(), dout, n * 8, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        if (part == NAN_ALL || is_cvt_ftz(op[i]) == (part == NAN_ONLY_CVT_FTZ))
            printf("%s %016llx %016llx %016llx %016llx\n", names[op[i]].c_str(), a[i], b[i], c[i], out[i]);
    return 0;
}

#ifndef NAN_NO_MAIN
int main() { return nan_run(NAN_ALL); }
#endif
