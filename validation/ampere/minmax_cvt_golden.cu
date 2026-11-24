// Host program: record floating min/max and float-to-integer cvt from the GPU as
// "instruction a b result", the operands and the result as 64-bit hex patterns (a 32-bit
// value sits in the low half; b is 0 for cvt).
//
//   min.f32 max.f32 min.f64 max.f64     all pairs of zeros, ones, infinities, quiet and
//                                       signalling NaNs, a subnormal and a few ordinary values
//   cvt.{rni,rzi,rmi,rpi}.{s32,u32,s64,u64}.{f32,f64}
//                                       NaNs, infinities, halves (for the rounding modes),
//                                       values at and next to every limit of the four integer
//                                       types, and pseudo-random patterns
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

typedef unsigned long long u64;

#define C32(id, name, in) \
    case id: { int r; asm volatile("cvt." name " %0, %1;" : "=r"(r) : in); out[i] = (unsigned)r; break; }
#define C64(id, name, in) \
    case id: { long long r; asm volatile("cvt." name " %0, %1;" : "=l"(r) : in); out[i] = (u64)r; break; }
#define MODE(base, m) \
    C32(base + 0, m ".s32.f32", "f"(xf)) C32(base + 1, m ".s32.f64", "d"(xd)) \
    C32(base + 2, m ".u32.f32", "f"(xf)) C32(base + 3, m ".u32.f64", "d"(xd)) \
    C64(base + 4, m ".s64.f32", "f"(xf)) C64(base + 5, m ".s64.f64", "d"(xd)) \
    C64(base + 6, m ".u64.f32", "f"(xf)) C64(base + 7, m ".u64.f64", "d"(xd))

__global__ void minmax_cvt_probe(const int* op, const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xf = __uint_as_float((unsigned)a[i]), yf = __uint_as_float((unsigned)b[i]), rf = 0.0f;
    double xd = __longlong_as_double((long long)a[i]), yd = __longlong_as_double((long long)b[i]), rd = 0.0;
    switch (op[i]) {
        case 0: asm volatile("min.f32 %0, %1, %2;" : "=f"(rf) : "f"(xf), "f"(yf)); out[i] = __float_as_uint(rf); break;
        case 1: asm volatile("max.f32 %0, %1, %2;" : "=f"(rf) : "f"(xf), "f"(yf)); out[i] = __float_as_uint(rf); break;
        case 2: asm volatile("min.f64 %0, %1, %2;" : "=d"(rd) : "d"(xd), "d"(yd)); out[i] = (u64)__double_as_longlong(rd); break;
        case 3: asm volatile("max.f64 %0, %1, %2;" : "=d"(rd) : "d"(xd), "d"(yd)); out[i] = (u64)__double_as_longlong(rd); break;
        MODE(4, "rni") MODE(12, "rzi") MODE(20, "rmi") MODE(28, "rpi")
    }
}

static u64 bits32(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static u64 bits64(double d) { u64 u; memcpy(&u, &d, 8); return u; }

int main() {
    std::vector<std::string> names = {"min.f32", "max.f32", "min.f64", "max.f64"};
    for (const char* mode : {"rni", "rzi", "rmi", "rpi"})
        for (const char* dst : {"s32", "u32", "s64", "u64"})
            for (const char* src : {"f32", "f64"})
                names.push_back(std::string("cvt.") + mode + "." + dst + "." + src);

    std::vector<int> op;
    std::vector<u64> a, b;
    auto push = [&](int o, u64 x, u64 y) { op.push_back(o); a.push_back(x); b.push_back(y); };

    // min and max
    const u64 mm32[] = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                        0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fc12345,  // NaNs: quiet, signalling, payload
                        0x00000001, 0x40200000, 0xc0200000, 0x7f7fffff};
    const u64 mm64[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                        0x7ff0000000000000ull, 0xfff0000000000000ull,
                        0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull, 0xfff0000000000001ull,
                        0x7ff8000012345678ull,
                        0x0000000000000001ull, 0x4004000000000000ull, 0xc004000000000000ull, 0x7fefffffffffffffull};
    for (u64 x : mm32) for (u64 y : mm32) { push(0, x, y); push(1, x, y); }
    for (u64 x : mm64) for (u64 y : mm64) { push(2, x, y); push(3, x, y); }

    // float to integer
    std::vector<u64> in32 = {0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001, 0x7fc12345,  // NaNs
                             0x7f800000, 0xff800000, 0x00000000, 0x80000000, 0x00000001, 0x80000001,
                             0x7f7fffff, 0xff7fffff};
    std::vector<u64> in64 = {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull,
                             0xfff0000000000001ull, 0x7ff8000012345678ull,
                             0x7ff0000000000000ull, 0xfff0000000000000ull, 0x0000000000000000ull,
                             0x8000000000000000ull, 0x0000000000000001ull, 0x8000000000000001ull,
                             0x7fefffffffffffffull, 0xffefffffffffffffull};
    for (double v : {0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 3.5, 1e10, 1e30})
        for (double s : {1.0, -1.0}) { in32.push_back(bits32((float)(s * v))); in64.push_back(bits64(s * v)); }
    for (int e : {31, 32, 63, 64})
        for (double s : {1.0, -1.0}) {
            float f = (float)(s * ldexp(1.0, e));
            for (float v : {f, nextafterf(f, 0.0f), nextafterf(f, s * INFINITY)}) in32.push_back(bits32(v));
            double d = s * ldexp(1.0, e);
            for (double v : {d, nextafter(d, 0.0), nextafter(d, s * INFINITY)}) in64.push_back(bits64(v));
            if (e < 53)  // the integers and halves next to the limit are exact in f64
                for (double v : {d - 1.0, d - 0.5, d + 0.5, d + 1.0, d - 1.5, d + 1.5}) in64.push_back(bits64(v));
        }
    unsigned state = 2026u;
    auto next = [&]() { state = state * 1664525u + 1013904223u; return (u64)state; };
    for (int k = 0; k < 200; ++k) {
        in32.push_back(next());
        u64 hi = next(); in64.push_back(hi << 32 | next());
        // exponents near the integer ranges, where the rounding mode and the limits matter
        in32.push_back((next() & 0x807fffffu) | (u64)(118 + k % 76) << 23);
        hi = next(); in64.push_back(((hi & 0x800fffffu) | (u64)(1014 + k % 76) << 20) << 32 | next());
    }
    for (int o = 4; o < 36; ++o)
        for (u64 x : (o % 2 ? in64 : in32)) push(o, x, 0);

    int n = (int)op.size();
    std::vector<u64> out(n);
    int* dop; u64 *da, *db, *dout;
    cudaMalloc(&dop, n * 4); cudaMalloc(&da, n * 8); cudaMalloc(&db, n * 8); cudaMalloc(&dout, n * 8);
    cudaMemcpy(dop, op.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 8, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, n * 8);
    minmax_cvt_probe<<<(n + 255) / 256, 256>>>(dop, da, db, dout, n);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
    cudaMemcpy(out.data(), dout, n * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        printf("%s %016llx %016llx %016llx\n", names[op[i]].c_str(), a[i], b[i], out[i]);
    return 0;
}
