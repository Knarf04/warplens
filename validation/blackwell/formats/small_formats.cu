// The GPU's own conversions to and from the small floating-point formats, for comparison
// with warplens.numerics.
//
// The kernel calls the CUDA math library's conversion functions (cuda_fp8.h, cuda_fp4.h),
// which on sm_100a compile to the conversion instructions (cvt.rn.satfinite.e4m3x2.f32 and
// relatives) wherever one exists:
//   E4M3, E5M2:  __nv_cvt_float_to_fp8(x, __NV_SATFINITE, kind), __nv_cvt_fp8_to_halfraw
//   E2M1:        __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest), __nv_cvt_fp4_to_halfraw
//   UE8M0:       __nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundNearest), __nv_cvt_e8m0_to_bf16raw
//
// The simulator does not execute these instructions. The prediction for this probe is
// written by formats/predict_small_formats.py from warplens.numerics, which generates the
// same inputs; keep the two generators in step.
//
// Claims tested (docs/numerics.md, "Small formats"):
//   * decoding of every code (fp8, fp4, e8m0);
//   * encoding rounds to nearest with ties to the even significand, saturating at the
//     largest finite value; UE8M0 ties go to the larger power of two;
//   * NaN, infinity, zero, negative zero and out-of-range inputs (special).
// Inputs warplens.numerics refuses (NaN to E2M1, negative values to UE8M0) are printed on
// "info:" lines, which are not compared.
//
//   small_formats fp8|fp4|e8m0|special
#include "../common.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <vector>

enum Mode { ENC_E4M3, ENC_E5M2, ENC_E2M1, ENC_UE8M0, DEC_E4M3, DEC_E5M2, DEC_E2M1, DEC_UE8M0 };

extern "C" __global__ void convert(int mode, const uint32_t* in, uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = in[i], y = 0;
    float f = __uint_as_float(x);
    switch (mode) {
        case ENC_E4M3: y = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); break;
        case ENC_E5M2: y = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2); break;
        case ENC_E2M1: y = __nv_cvt_float_to_fp4(f, __NV_E2M1, cudaRoundNearest); break;
        case ENC_UE8M0: y = __nv_cvt_float_to_e8m0(f, __NV_SATFINITE, cudaRoundNearest); break;
        case DEC_E4M3: y = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)x, __NV_E4M3).x; break;
        case DEC_E5M2: y = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)x, __NV_E5M2).x; break;
        case DEC_E2M1: y = __nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)x, __NV_E2M1).x; break;
        case DEC_UE8M0: y = __nv_cvt_e8m0_to_bf16raw((__nv_fp8_storage_t)x).x; break;
    }
    out[i] = y;
}

static std::vector<uint32_t> run(int mode, const std::vector<uint32_t>& in) {
    int n = (int)in.size();
    std::vector<uint32_t> out(n);
    uint32_t* d_in = dev_copy(in.data(), n);
    uint32_t* d_out = dev_alloc<uint32_t>(n);
    convert<<<(n + 255) / 256, 256>>>(mode, d_in, d_out, n);
    probe_sync("convert");
    dev_read(out.data(), d_out, n);
    return out;
}

// Float32 patterns around every value a format with up to 4 fraction bits can hold between
// 2^-18 and 2^18: the top five fraction bits take every value (which covers every code and
// every midpoint between codes), each with its two float32 neighbours.
static std::vector<uint32_t> sweep_inputs() {
    std::vector<uint32_t> in;
    for (uint32_t sign = 0; sign < 2; ++sign)
        for (int e = -18; e <= 17; ++e)
            for (uint32_t j = 0; j < 32; ++j) {
                uint32_t base = (sign << 31) | ((uint32_t)(e + 127) << 23) | (j << 18);
                in.push_back(base - 1);
                in.push_back(base);
                in.push_back(base + 1);
            }
    return in;
}

// Every binade of float32: the power of two, the midpoint 1.5 * 2^e (the tie), sqrt(2) * 2^e,
// and their neighbours.
static std::vector<uint32_t> e8m0_inputs() {
    std::vector<uint32_t> in;
    static const uint32_t fractions[] = {0x000000, 0x000001, 0x3504f3, 0x3fffff, 0x400000, 0x400001, 0x7fffff};
    for (uint32_t e = 1; e <= 254; ++e)
        for (uint32_t fraction : fractions) in.push_back((e << 23) | fraction);
    return in;
}

static std::vector<uint32_t> special_inputs() {
    std::vector<uint32_t> in = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x00800000u,
                                0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
                                0x7f800001u, 0xffa12345u};
    static const float values[] = {1e10f,   448.0f,   464.0f,   479.9f,  480.0f,  512.0f,   57344.0f, 61440.0f,
                                   65536.0f, 6.0f,    6.9f,     7.0f,    8.0f,    100.0f,   0.25f,    0.24f,
                                   1e-3f,    1e-10f,  1e-30f,   1e30f,   3.0e38f, 1.0f,     0.5f,     0.75f};
    for (float v : values) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        in.push_back(bits);
        in.push_back(bits | 0x80000000u);
    }
    return in;
}

static bool is_nan(uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }

// Eight "input:code" pairs per line. Pairs the reference model has no answer for go on
// "info:" lines instead.
static void print_encoding(const char* name, int mode, const std::vector<uint32_t>& in) {
    std::vector<uint32_t> out = run(mode, in);
    int column = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        bool unsupported = (mode == ENC_E2M1 && is_nan(in[i])) ||
                           (mode == ENC_UE8M0 && (in[i] >> 31) && (in[i] & 0x7fffffffu) != 0 && !is_nan(in[i]));
        if (unsupported) {
            if (column) printf("\n");
            column = 0;
            printf("info: %s %08x:%02x (warplens.numerics rejects this input)\n", name, in[i], out[i]);
            continue;
        }
        printf("%s%08x:%02x", column ? " " : name, in[i], out[i]);
        column = (column + 1) % 8;
        if (!column) printf("\n");
    }
    if (column) printf("\n");
}

// One line per code: the decoded value's bits (half precision; bfloat16 for UE8M0), or "nan".
static void print_decoding(const char* name, int mode, int codes, bool bfloat) {
    std::vector<uint32_t> in(codes);
    for (int c = 0; c < codes; ++c) in[c] = (uint32_t)c;
    std::vector<uint32_t> out = run(mode, in);
    for (int c = 0; c < codes; ++c) {
        uint32_t v = out[c] & 0xffff;
        bool nan = bfloat ? (v & 0x7fff) > 0x7f80 : (v & 0x7fff) > 0x7c00;
        if (nan) {
            printf("%s code %02x -> nan\n", name, c);
            printf("info: %s code %02x decodes to the NaN pattern %04x\n", name, c, v);
        } else {
            printf("%s code %02x -> %s %04x\n", name, c, bfloat ? "bfloat16" : "half", v);
        }
    }
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    if (!strcmp(which, "fp8")) {
        print_decoding("e4m3", DEC_E4M3, 256, false);
        print_decoding("e5m2", DEC_E5M2, 256, false);
        print_encoding("e4m3 ", ENC_E4M3, sweep_inputs());
        print_encoding("e5m2 ", ENC_E5M2, sweep_inputs());
    } else if (!strcmp(which, "fp4")) {
        print_decoding("e2m1", DEC_E2M1, 16, false);
        print_encoding("e2m1 ", ENC_E2M1, sweep_inputs());
    } else if (!strcmp(which, "e8m0")) {
        print_decoding("ue8m0", DEC_UE8M0, 256, true);
        print_encoding("ue8m0 ", ENC_UE8M0, e8m0_inputs());
    } else if (!strcmp(which, "special")) {
        print_encoding("e4m3 ", ENC_E4M3, special_inputs());
        print_encoding("e5m2 ", ENC_E5M2, special_inputs());
        print_encoding("e2m1 ", ENC_E2M1, special_inputs());
        print_encoding("ue8m0 ", ENC_UE8M0, special_inputs());
    } else {
        printf("usage: small_formats fp8|fp4|e8m0|special\n");
        return 1;
    }
    return 0;
}
