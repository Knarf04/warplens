// Scalar edge cases: min/max with NaNs and zeros, float-to-integer conversion at the limits,
// and integer division, remainder, shifts and mul.hi at theirs.
//
// One kernel executes one instruction, chosen by `op`, over arrays of operand bit patterns.
// Every line is "instruction a [b] -> result" in hex.
//
//   min, max (f32, f64)    all ordered pairs of +-0, +-1, +-infinity, a quiet and a
//                          signalling NaN. A NaN result is printed as "nan".
//   cvt.{rni,rzi,rmi,rpi}.{s32,u32,s64,u64}.{f32,f64}
//                          NaN, +-infinity, +-0, halves, and the values at and next to
//                          +-2^31, 2^32, +-2^63 and 2^64
//   div, rem, mul.hi (s32, u32, s64, u64); shl (b32, b64); shr (s32, u32, s64, u64)
//                          the limits of each type; shift amounts 0, 1, 31, 32, 33, 63, 64,
//                          65, 2^31 and 2^32 - 1
// Division and remainder by zero are executed too; PTX leaves their results unspecified and machine-specific, so
// they go on "info:" lines and are never compared.
//
// Claims tested (docs/numerics.md; matrix rows NUM-11 and NUM-12):
//   * min/max with one NaN operand give the other operand; min(+0, -0) is -0, max is +0;
//   * float to integer saturates; NaN converts to 0, or to the sign-bit pattern when the
//     source is f64 or the result is 64 bits wide;
//   * integer arithmetic wraps, division truncates toward zero, shifts clamp at the
//     register width.
//
// Nothing here is newer than Ampere: the Ampere build (build.sh --ampere) runs on the
// development machine's GPU.
//
//   scalar_edges
#include "../common.h"

#include <vector>

enum {
    MIN_F32, MAX_F32, MIN_F64, MAX_F64,
    CVT_FIRST,  // + mode * 8 + source * 4 + destination; modes rni rzi rmi rpi; f32 f64; s32 u32 s64 u64
    INT_FIRST = CVT_FIRST + 32,  // + operation * 4 + type; div rem mul.hi shr; s32 u32 s64 u64
    SHL_B32 = INT_FIRST + 16, SHL_B64, OPS
};

#define CVT1(code, mode, dst, dreg, dtype, src, sreg, sval)                                      \
    case code: {                                                                                 \
        dtype r;                                                                                 \
        asm volatile("cvt." mode "." dst "." src " %0, %1;" : "=" dreg(r) : sreg(sval));         \
        result = (uint64_t)r;                                                                    \
    } break;
#define CVT_MODE(base, mode)                                     \
    CVT1(base + 0, mode, "s32", "r", uint32_t, "f32", "f", fa)   \
    CVT1(base + 1, mode, "u32", "r", uint32_t, "f32", "f", fa)   \
    CVT1(base + 2, mode, "s64", "l", uint64_t, "f32", "f", fa)   \
    CVT1(base + 3, mode, "u64", "l", uint64_t, "f32", "f", fa)   \
    CVT1(base + 4, mode, "s32", "r", uint32_t, "f64", "d", da)   \
    CVT1(base + 5, mode, "u32", "r", uint32_t, "f64", "d", da)   \
    CVT1(base + 6, mode, "s64", "l", uint64_t, "f64", "d", da)   \
    CVT1(base + 7, mode, "u64", "l", uint64_t, "f64", "d", da)
#define BIN(code, text, reg, type)                                                               \
    case code: {                                                                                 \
        type r;                                                                                  \
        asm volatile(text " %0, %1, %2;" : "=" reg(r) : reg((type)a), reg((type)b));             \
        result = (uint64_t)r;                                                                    \
    } break;
#define SHIFT(code, text, reg, type)                                                             \
    case code: {                                                                                 \
        type r;                                                                                  \
        asm volatile(text " %0, %1, %2;" : "=" reg(r) : reg((type)a), "r"((uint32_t)b));         \
        result = (uint64_t)r;                                                                    \
    } break;
#define INT_OP(base, name, MACRO)            \
    MACRO(base + 0, name ".s32", "r", uint32_t) \
    MACRO(base + 1, name ".u32", "r", uint32_t) \
    MACRO(base + 2, name ".s64", "l", uint64_t) \
    MACRO(base + 3, name ".u64", "l", uint64_t)

extern "C" __global__ void scalar_edges(int op, const uint64_t* as, const uint64_t* bs, uint64_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t a = as[i], b = bs[i], result = 0;
    float fa = __uint_as_float((uint32_t)a), fb = __uint_as_float((uint32_t)b), fr;
    double da = __longlong_as_double((long long)a), db = __longlong_as_double((long long)b), dr;
    switch (op) {
        case MIN_F32:
            asm volatile("min.f32 %0, %1, %2;" : "=f"(fr) : "f"(fa), "f"(fb));
            result = __float_as_uint(fr);
            break;
        case MAX_F32:
            asm volatile("max.f32 %0, %1, %2;" : "=f"(fr) : "f"(fa), "f"(fb));
            result = __float_as_uint(fr);
            break;
        case MIN_F64:
            asm volatile("min.f64 %0, %1, %2;" : "=d"(dr) : "d"(da), "d"(db));
            result = (uint64_t)__double_as_longlong(dr);
            break;
        case MAX_F64:
            asm volatile("max.f64 %0, %1, %2;" : "=d"(dr) : "d"(da), "d"(db));
            result = (uint64_t)__double_as_longlong(dr);
            break;
        CVT_MODE(CVT_FIRST + 0, "rni")
        CVT_MODE(CVT_FIRST + 8, "rzi")
        CVT_MODE(CVT_FIRST + 16, "rmi")
        CVT_MODE(CVT_FIRST + 24, "rpi")
        INT_OP(INT_FIRST + 0, "div", BIN)
        INT_OP(INT_FIRST + 4, "rem", BIN)
        INT_OP(INT_FIRST + 8, "mul.hi", BIN)
        INT_OP(INT_FIRST + 12, "shr", SHIFT)
        SHIFT(SHL_B32, "shl.b32", "r", uint32_t)
        SHIFT(SHL_B64, "shl.b64", "l", uint64_t)
    }
    out[i] = result;
}

static const char* TYPES[4] = {"s32", "u32", "s64", "u64"};
static const char* MODES[4] = {"rni", "rzi", "rmi", "rpi"};

static uint32_t bits32(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t bits64(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

static std::vector<uint64_t> run(int op, const std::vector<uint64_t>& a, const std::vector<uint64_t>& b,
                                 const char* what) {
    int n = (int)a.size();
    std::vector<uint64_t> out(n);
    uint64_t* da = dev_copy(a.data(), n);
    uint64_t* db = dev_copy(b.data(), n);
    uint64_t* d_out = dev_alloc<uint64_t>(n);
    dev_fill(d_out, n, 0xff);
    scalar_edges<<<(n + 255) / 256, 256>>>(op, da, db, d_out, n);
    probe_sync(what);
    dev_read(out.data(), d_out, n);
    HOST_CHECK(cudaFree(da));
    HOST_CHECK(cudaFree(db));
    HOST_CHECK(cudaFree(d_out));
    return out;
}

int main() {
    char name[64];
    // ---- min / max ----
    const uint32_t f32s[8] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
                              0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u};
    const uint64_t f64s[8] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull,
                              0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
                              0x7ff8000000000000ull, 0x7ff0000000000001ull};
    for (int op = MIN_F32; op <= MAX_F64; ++op) {
        bool wide = op >= MIN_F64;
        std::vector<uint64_t> a, b;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) {
                a.push_back(wide ? f64s[i] : f32s[i]);
                b.push_back(wide ? f64s[j] : f32s[j]);
            }
        snprintf(name, sizeof name, "%s.%s", (op & 1) ? "max" : "min", wide ? "f64" : "f32");
        std::vector<uint64_t> r = run(op, a, b, name);
        int width = wide ? 16 : 8;
        for (size_t i = 0; i < a.size(); ++i) {
            bool nan = wide ? (r[i] & 0x7fffffffffffffffull) > 0x7ff0000000000000ull
                            : (r[i] & 0x7fffffffu) > 0x7f800000u;
            printf("%s %0*llx %0*llx -> ", name, width, (unsigned long long)a[i], width, (unsigned long long)b[i]);
            if (nan) printf("nan\ninfo: %s case %zu NaN bits %0*llx\n", name, i, width, (unsigned long long)r[i]);
            else printf("%0*llx\n", width, (unsigned long long)r[i]);
        }
    }

    // ---- float to integer ----
    std::vector<uint64_t> in32, in64, none;
    const float limits32[4] = {2147483648.0f, 4294967296.0f, 9223372036854775808.0f, 18446744073709551616.0f};
    const double limits64[4] = {2147483648.0, 4294967296.0, 9223372036854775808.0, 18446744073709551616.0};
    for (uint32_t special : {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x7f800000u, 0xff800000u, 0x00000000u,
                             0x80000000u, 0x00000001u, 0x80000001u})
        in32.push_back(special);
    for (float v : {0.5f, 1.5f, 2.5f, 0.49999997f, 0.99999994f, 1.0f, 1e30f})
        for (float sign : {1.0f, -1.0f}) in32.push_back(bits32(sign * v));
    for (float limit : limits32)
        for (uint32_t sign : {0u, 0x80000000u})
            for (int step : {-1, 0, 1}) in32.push_back((bits32(limit) + (uint32_t)step) | sign);
    for (uint64_t special : {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull,
                             0x7ff0000000000000ull, 0xfff0000000000000ull, 0x0000000000000000ull,
                             0x8000000000000000ull, 0x0000000000000001ull, 0x8000000000000001ull})
        in64.push_back(special);
    for (double v : {0.5, 1.5, 2.5, 0.49999999999999994, 0.9999999999999999, 1.0, 1e30, 1e300})
        for (double sign : {1.0, -1.0}) in64.push_back(bits64(sign * v));
    for (double limit : limits64)
        for (uint64_t sign : {0ull, 0x8000000000000000ull}) {
            for (int step : {-1, 0, 1}) in64.push_back((bits64(limit) + (uint64_t)(int64_t)step) | sign);
            if (limit < 1e10)
                for (double offset : {-1.0, -0.5, 0.5, 1.0, 1.5})
                    in64.push_back(bits64(limit + offset) | sign);
        }
    for (int mode = 0; mode < 4; ++mode)
        for (int source = 0; source < 2; ++source)
            for (int destination = 0; destination < 4; ++destination) {
                const std::vector<uint64_t>& in = source ? in64 : in32;
                snprintf(name, sizeof name, "cvt.%s.%s.%s", MODES[mode], TYPES[destination], source ? "f64" : "f32");
                std::vector<uint64_t> r = run(CVT_FIRST + mode * 8 + source * 4 + destination, in, in, name);
                for (size_t i = 0; i < in.size(); ++i)
                    printf("%s %0*llx -> %0*llx\n", name, source ? 16 : 8, (unsigned long long)in[i],
                           destination < 2 ? 8 : 16, (unsigned long long)r[i]);
            }

    // ---- integers ----
    const uint64_t v32[11] = {0, 1, 2, 3, 7, 0x12345678u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu,
                              0xffffffffu};
    const uint64_t v64[11] = {0, 1, 2, 3, 7, 0x123456789abcdef0ull, 0x7fffffffffffffffull, 0x8000000000000000ull,
                              0x8000000000000001ull, 0xfffffffffffffffeull, 0xffffffffffffffffull};
    const uint64_t amounts[10] = {0, 1, 31, 32, 33, 63, 64, 65, 0x80000000u, 0xffffffffu};
    static const char* int_names[4] = {"div", "rem", "mul.hi", "shr"};
    for (int operation = 0; operation < 5; ++operation)
        for (int type = 0; type < (operation == 4 ? 2 : 4); ++type) {
            bool shift = operation >= 3;
            bool wide = operation == 4 ? type == 1 : type >= 2;
            int op = operation == 4 ? SHL_B32 + type : INT_FIRST + operation * 4 + type;
            std::vector<uint64_t> a, b;
            for (int i = 0; i < 11; ++i)
                for (int j = 0; j < (shift ? 10 : 11); ++j) {
                    a.push_back(wide ? v64[i] : v32[i]);
                    b.push_back(shift ? amounts[j] : wide ? v64[j] : v32[j]);
                }
            if (operation == 4) snprintf(name, sizeof name, "shl.%s", type ? "b64" : "b32");
            else snprintf(name, sizeof name, "%s.%s", int_names[operation], TYPES[type]);
            std::vector<uint64_t> r = run(op, a, b, name);
            int width = wide ? 16 : 8;
            for (size_t i = 0; i < a.size(); ++i)
                printf("%s%s %0*llx %0*llx -> %0*llx\n", operation < 2 && b[i] == 0 ? "info: " : "", name, width,
                       (unsigned long long)a[i], shift ? 8 : width, (unsigned long long)b[i], width,
                       (unsigned long long)r[i]);
        }
    return 0;
}
