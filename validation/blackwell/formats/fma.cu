// fma and floating-point mad: one exact product and sum, rounded once.
//
// 10^4 seeded cases for f32 and 10^4 for f64, in five classes (case index mod 5):
//   0  random operands with moderate exponents
//   1  rounding midpoints: a and b have half-width significands, so a * b needs one or two
//      bits more than fit and often lies exactly between two floats; c is zero or a tiny
//      value of either sign that decides the tie only if the product is not rounded first
//   2  cancellation: c = -(a * b rounded), so the result is the rounding error of a * b,
//      which a separate multiply and add would return as zero
//   3  subnormal results: the product's exponent is at the bottom of the range
//   4  overflow edge: the product is near the largest finite value, c pulls it back
// The kernel computes fma.rn and mad.rn of every case.
//
//   fma          the first 1000 cases of each type in full ("type index a b c result"), then
//                one checksum line per 100 cases for all 10^4, and how often mad equals fma
//   fma all      every case in full (20000 lines), for locating a checksum that differs
//
// Claim tested (docs/numerics.md; matrix row NUM-10): fma and mad compute the exact product
// and sum and round once, to nearest even, for f32 and f64. A NaN result is printed as the
// word "nan" (its payload is not claimed).
//
// Nothing here is newer than Ampere: the Ampere build (build.sh --ampere) runs on the
// development machine's GPU.
#include "../common.h"

#define CASES 10000
#define FULL 1000

extern "C" __global__ void fma_f32(const float* a, const float* b, const float* c, float* fma_out, float* mad_out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i], r, s;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(x), "f"(y), "f"(z));
    asm volatile("mad.rn.f32 %0, %1, %2, %3;" : "=f"(s) : "f"(x), "f"(y), "f"(z));
    fma_out[i] = r;
    mad_out[i] = s;
}

extern "C" __global__ void fma_f64(const double* a, const double* b, const double* c, double* fma_out,
                                   double* mad_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = a[i], y = b[i], z = c[i], r, s;
    asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(r) : "d"(x), "d"(y), "d"(z));
    asm volatile("mad.rn.f64 %0, %1, %2, %3;" : "=d"(s) : "d"(x), "d"(y), "d"(z));
    fma_out[i] = r;
    mad_out[i] = s;
}

static uint64_t state = 0x9E3779B97F4A7C15ull;
static uint64_t next() {  // splitmix64
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t below(uint64_t n) { return next() % n; }

// A float of either width from sign, unbiased exponent and significand bits below the
// leading one. `bits` is 23 or 52.
static uint64_t make(int bits, uint64_t sign, int exponent, uint64_t fraction) {
    int bias = bits == 23 ? 127 : 1023;
    return (sign << (bits == 23 ? 31 : 63)) | ((uint64_t)(exponent + bias) << bits) | fraction;
}

// Operands of case `i` as bit patterns. `product` returns a * b rounded, as bits.
static void operands(int bits, int i, uint64_t* a, uint64_t* b, uint64_t* c, uint64_t (*product)(uint64_t, uint64_t)) {
    int emax = bits == 23 ? 127 : 1023, half = (bits + 1) / 2;
    uint64_t mask = (1ull << bits) - 1;
    switch (i % 5) {
        case 0:
            *a = make(bits, below(2), (int)below(40) - 20, next() & mask);
            *b = make(bits, below(2), (int)below(40) - 20, next() & mask);
            *c = make(bits, below(2), (int)below(60) - 30, next() & mask);
            break;
        case 1: {
            int ea = (int)below(20) - 10, eb = (int)below(20) - 10;
            *a = make(bits, below(2), ea, (next() & ((1ull << half) - 1)) << (bits - half));
            *b = make(bits, below(2), eb, (next() & ((1ull << half) - 1)) << (bits - half));
            uint64_t kind = below(4);
            if (kind == 0) *c = make(bits, below(2), -emax, 0) & (1ull << (bits == 23 ? 31 : 63));  // +-0
            else *c = make(bits, below(2), ea + eb - bits - (int)below(8) - 1, kind == 1 ? 0 : next() & mask);
            break;
        }
        case 2:
            *a = make(bits, below(2), (int)below(40) - 20, next() & mask);
            *b = make(bits, below(2), (int)below(40) - 20, next() & mask);
            *c = product(*a, *b) ^ (1ull << (bits == 23 ? 31 : 63));
            break;
        case 3: {
            int ea = -(emax / 2) - (int)below(12), eb = -(emax - 1) - ea - (int)below(bits + 6) + 2;
            *a = make(bits, below(2), ea, next() & mask);
            *b = make(bits, below(2), eb, next() & mask);
            uint64_t kind = below(3);
            *c = kind == 0 ? 0 : kind == 1 ? (next() & mask) | (below(2) << (bits == 23 ? 31 : 63))  // subnormal
                             : make(bits, below(2), -(emax - 1) + (int)below(4), next() & mask);
            break;
        }
        default: {
            int ea = emax / 2 + (int)below(8), eb = emax - ea - (int)below(3) + 1;
            uint64_t sa = below(2), sb = below(2);
            *a = make(bits, sa, ea, next() & mask);
            *b = make(bits, sb, eb, next() & mask);
            *c = below(3) == 0 ? 0 : make(bits, (sa ^ sb) ^ below(2), emax - (int)below(3), next() & mask);
            break;
        }
    }
}

static uint64_t product32(uint64_t a, uint64_t b) {
    uint32_t x = (uint32_t)a, y = (uint32_t)b, r;
    float fx, fy;
    memcpy(&fx, &x, 4);
    memcpy(&fy, &y, 4);
    volatile float p = fx * fy;
    float q = p;
    memcpy(&r, &q, 4);
    return r;
}
static uint64_t product64(uint64_t a, uint64_t b) {
    double fx, fy;
    memcpy(&fx, &a, 8);
    memcpy(&fy, &b, 8);
    volatile double p = fx * fy;
    double q = p;
    uint64_t r;
    memcpy(&r, &q, 8);
    return r;
}

static bool is_nan(int bits, uint64_t x) {
    return bits == 23 ? (x & 0x7fffffffu) > 0x7f800000u : (x & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}

static void report(const char* type, int bits, const uint64_t* a, const uint64_t* b, const uint64_t* c,
                   const uint64_t* r, const uint64_t* mad, bool all) {
    int width = bits == 23 ? 8 : 16;
    for (int i = 0; i < (all ? CASES : FULL); ++i) {
        printf("%s %4d %0*llx %0*llx %0*llx ", type, i, width, (unsigned long long)a[i], width,
               (unsigned long long)b[i], width, (unsigned long long)c[i]);
        if (is_nan(bits, r[i])) printf("nan\n");
        else printf("%0*llx\n", width, (unsigned long long)r[i]);
    }
    if (all) return;
    int same = 0, nans = 0;
    for (int block = 0; block < CASES / 100; ++block) {
        uint64_t sum = 0xcbf29ce484222325ull;  // FNV-1a over the result words, NaNs as one value
        for (int i = block * 100; i < block * 100 + 100; ++i) {
            uint64_t v = is_nan(bits, r[i]) ? 0x7ff8dead00000000ull : r[i];
            for (int k = 0; k < 8; ++k) sum = (sum ^ ((v >> (8 * k)) & 0xff)) * 0x100000001b3ull;
        }
        printf("%s cases %4d-%4d: checksum of the results %016llx\n", type, block * 100, block * 100 + 99,
               (unsigned long long)sum);
    }
    for (int i = 0; i < CASES; ++i) {
        nans += is_nan(bits, r[i]);
        same += r[i] == mad[i] || (is_nan(bits, r[i]) && is_nan(bits, mad[i]));
    }
    printf("%s: mad.rn equals fma.rn in %d of %d cases; NaN results: %d\n", type, same, CASES, nans);
}

int main(int argc, char** argv) {
    bool all = argc > 1 && !strcmp(argv[1], "all");
    static uint64_t a[CASES], b[CASES], c[CASES], r[CASES], mad[CASES];
    {
        static uint32_t a32[CASES], b32[CASES], c32[CASES], r32[CASES], m32[CASES];
        for (int i = 0; i < CASES; ++i) {
            operands(23, i, &a[i], &b[i], &c[i], product32);
            a32[i] = (uint32_t)a[i];
            b32[i] = (uint32_t)b[i];
            c32[i] = (uint32_t)c[i];
        }
        float* da = (float*)dev_copy(a32, CASES);
        float* db = (float*)dev_copy(b32, CASES);
        float* dc = (float*)dev_copy(c32, CASES);
        float* dr = (float*)dev_alloc<uint32_t>(CASES);
        float* dm = (float*)dev_alloc<uint32_t>(CASES);
        fma_f32<<<(CASES + 255) / 256, 256>>>(da, db, dc, dr, dm, CASES);
        probe_sync("fma_f32");
        dev_read(r32, (uint32_t*)dr, CASES);
        dev_read(m32, (uint32_t*)dm, CASES);
        for (int i = 0; i < CASES; ++i) {
            r[i] = r32[i];
            mad[i] = m32[i];
        }
        report("f32", 23, a, b, c, r, mad, all);
    }
    for (int i = 0; i < CASES; ++i) operands(52, i, &a[i], &b[i], &c[i], product64);
    double* da = (double*)dev_copy(a, CASES);
    double* db = (double*)dev_copy(b, CASES);
    double* dc = (double*)dev_copy(c, CASES);
    double* dr = (double*)dev_alloc<uint64_t>(CASES);
    double* dm = (double*)dev_alloc<uint64_t>(CASES);
    fma_f64<<<(CASES + 255) / 256, 256>>>(da, db, dc, dr, dm, CASES);
    probe_sync("fma_f64");
    dev_read(r, (uint64_t*)dr, CASES);
    dev_read(mad, (uint64_t*)dm, CASES);
    report("f64", 52, a, b, c, r, mad, all);
    return 0;
}
