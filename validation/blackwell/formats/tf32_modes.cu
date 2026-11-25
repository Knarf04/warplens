// float32 -> TF32 in the three rounding modes: cvt.rna.tf32.f32, cvt.rn.tf32.f32,
// cvt.rz.tf32.f32.
//
// The inputs are the 715 patterns of validation/ampere/tf32_golden.cu: rounding boundaries
// at five exponents, subnormals, infinities, NaNs and pseudo-random patterns. Each line is
// "input rna rn rz" in hex.
//
// Claims tested (docs/numerics.md, "TF32"):
//   * .rna on Blackwell gives what it gave on the RTX 3080 Ti: the rna column must equal
//     validation/ampere/tf32_golden.txt (tests/validation/probes/test_blackwell_probes.py checks that the
//     committed prediction does);
//   * .rn (ties to even) and .rz (toward zero) act on the same magnitude bits and treat
//     subnormals, infinities and NaNs as .rna does. They need Hopper or newer and have
//     never been checked on hardware; the document calls this an assumption.
//
//   tf32_modes
#include "../common.h"

#include <vector>

extern "C" __global__ void tf32_modes(const uint32_t* in, uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = in[i], rna, rn, rz;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(rna) : "r"(x));
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(rn) : "r"(x));
    asm volatile("cvt.rz.tf32.f32 %0, %1;" : "=r"(rz) : "r"(x));
    out[3 * i] = rna;
    out[3 * i + 1] = rn;
    out[3 * i + 2] = rz;
}

int main() {
    std::vector<uint32_t> in;
    const uint32_t exponents[] = {0x00000000u, 0x00800000u, 0x3F800000u, 0x40000000u, 0x7F000000u};
    const uint32_t low[] = {0x0000, 0x0001, 0x0FFF, 0x1000, 0x1001, 0x1FFF};  // the 13 discarded bits
    const uint32_t kept[] = {0x000, 0x001, 0x155, 0x3FE, 0x3FF};              // the 10 kept bits
    for (uint32_t sign = 0; sign < 2; ++sign)
        for (uint32_t e : exponents)
            for (uint32_t k : kept)
                for (uint32_t l : low) in.push_back((sign << 31) | e | (k << 13) | l);
    const uint32_t special[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u,
                                0x7FA12345u, 0xFFFFFFFFu, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F7FF000u,
                                0x7F7FEFFFu, 0x00000FFFu, 0x00001000u, 0x007FFFFFu, 0x807FF000u};
    for (uint32_t s : special) in.push_back(s);
    uint32_t state = 12345u;
    for (int i = 0; i < 400; ++i) { state = state * 1664525u + 1013904223u; in.push_back(state); }

    int n = (int)in.size();
    std::vector<uint32_t> out(3 * n);
    uint32_t* d_in = dev_copy(in.data(), n);
    uint32_t* d_out = dev_alloc<uint32_t>(3 * n);
    tf32_modes<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    probe_sync("tf32_modes");
    dev_read(out.data(), d_out, 3 * n);
    printf("input    rna      rn       rz\n");
    for (int i = 0; i < n; ++i) printf("%08x %08x %08x %08x\n", in[i], out[3 * i], out[3 * i + 1], out[3 * i + 2]);
    return 0;
}
