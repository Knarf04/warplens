// What a kind::tf32 tcgen05.mma does with operand bits below TF32 precision.
//
// TF32 keeps 10 fraction bits of a float32; the low 13 are "below precision". Each result
// here is ONE product: A[m][0] * B[n][0], every other operand element zero, so there is no
// summation to confuse the picture. A's first column runs over float32 bit patterns with
// interesting low bits; B's over eight multipliers. The raw result bits are printed next to
// three candidate answers computed on the host:
//   full:   the float32 product of the operands as stored (what the simulator does)
//   rz:     the float32 product after clearing the low 13 bits of both operands
//   rna:    the float32 product after rounding both to TF32, nearest, ties away
// (rz and rna are computed on the magnitude bits as docs/numerics.md describes
// cvt.rz.tf32.f32 and cvt.rna.tf32.f32.) The result's own low bits also show whether the
// product is rounded to TF32 or kept as a full float32.
//
// Claim tested (docs/numerics.md, "tcgen05.mma", row kind::tf32; finding
// `mma-tf32-operand-not-converted`): the operands are used as the float32 values stored in
// shared memory, with no conversion. The document calls this an assumption.
//
//   tf32_low_bits
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

static uint32_t tf32_rz(uint32_t x) { return x & ~0x1fffu; }
static uint32_t tf32_rna(uint32_t x) {
    if ((x & 0x7f800000u) == 0x7f800000u) return x & ~0x1fffu;
    return (x + 0x1000u) & ~0x1fffu;
}
static uint32_t product(uint32_t x, uint32_t y) {
    volatile float fx = bits_float(x), fy = bits_float(y);
    volatile float p = fx * fy;
    return float_bits(p);
}

int main() {
    static const uint32_t a_bits[] = {
        0x3f800000u, 0x3f800001u, 0x3f800fffu, 0x3f801000u, 0x3f801001u, 0x3f801fffu, 0x3f802000u, 0x3f802001u,
        0x3f803000u, 0x3f803fffu, 0x3fffe000u, 0x3fffefffu, 0x3ffff000u, 0x3fffffffu, 0x40490fdbu, 0x3dcccccdu,
        0xbf801000u, 0xbf800fffu, 0xc0490fdbu, 0x00800000u, 0x00801000u, 0x007fffffu, 0x00001000u, 0x00000001u,
        0x7f7fffffu, 0x7f7ff000u, 0x7f7fe000u, 0x42f6e979u, 0x3eaaaaabu, 0x461c4000u, 0x461c4001u, 0x3a83126fu};
    static const uint32_t b_bits[8] = {0x3f800000u, 0x3f800001u, 0x3f801000u, 0x3fc00000u,
                                       0x40400000u, 0x3fffffffu, 0x3f000000u, 0x3dcccccdu};
    const int rows = (int)(sizeof a_bits / sizeof a_bits[0]);
    static uint32_t a[128 * 8], b[8 * 8];
    for (int m = 0; m < rows; ++m) a[m * 8] = a_bits[m];
    for (int n = 0; n < 8; ++n) b[n * 8] = b_bits[n];
    MmaJob job;
    job.m = 128;
    job.n = 8;
    job.fmt = 2;
    job.desc_a = place_k_major(image, 0, a, 128, 4, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 4, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump");
    printf("kind::tf32, one product per element: D[m][n] = A[m][0] * B[n][0]\n");
    printf("A bits     B bits     result     full       rz         rna        result's low 13 bits\n");
    for (int m = 0; m < rows; ++m)
        for (int n = 0; n < 8; ++n) {
            uint32_t got = out[m * 8 + n];
            printf("%08x   %08x   %08x   %08x   %08x   %08x   %04x\n", a_bits[m], b_bits[n], got,
                   product(a_bits[m], b_bits[n]), product(tf32_rz(a_bits[m]), tf32_rz(b_bits[n])),
                   product(tf32_rna(a_bits[m]), tf32_rna(b_bits[n])), got & 0x1fffu);
        }
    return 0;
}
