// tcgen05.mma (kind::f16) with half-precision subnormals among its operands, as raw result bits.
//
// A half-precision subnormal is a multiple of 2^-24 below 2^-14, and every product of two
// half-precision numbers is exact in float32 (the smallest, 2^-24 * 2^-24 = 2^-48, is far above
// float32's own subnormals). So a tensor core that treats subnormal operands as numbers
// prints the exact products below, and one that takes them for zero prints zeros.
//
// Part 1, one product per element. Row m of A has one non-zero entry, at k = 0, and column n
// of B is constant; the result is A[m][0] * B[n][0]. A and B each go through subnormals,
// the smallest normal, ordinary values and the largest finite one.
//
// Part 2, a subnormal next to a normal term. Row m of A is 1.0 at k = 0 and the value of
// part 1 at k = 1, and B is 1.0 everywhere, so the exact sum is 1 + A[m][1]. The model
// (docs/numerics.md, "tcgen05.mma") lines every term up under the largest and cuts it off
// 25 bits below that term's leading bit, so a term below 2^-25 of the largest is dropped
// and the results are 1.0 or the float32 next to it.
//
// Claim tested (docs/numerics.md; warplens/units/tcgen05.py:mma_sum): subnormal operands are
// multiplied like any other, bit for bit. Not measured before the third rental.
//
//   subnormals
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

static const uint16_t A_VALUES[16] = {0x0001, 0x0002, 0x0200, 0x03ff, 0x0400, 0x0401, 0x8001, 0x83ff,
                                      0x3c00, 0xbc00, 0x7bff, 0x0000, 0x8000, 0x0155, 0x82aa, 0x3555};
static const uint16_t B_VALUES[8] = {0x3c00, 0x0001, 0x03ff, 0x0400, 0x7bff, 0x8001, 0x4000, 0x3800};

static void print_rows(const char* what) {
    for (int m = 0; m < 16; ++m) {
        printf("  %s %04x:", what, A_VALUES[m]);
        for (int n = 0; n < 8; ++n) printf(" %08x", out[m * 8 + n]);
        printf("\n");
    }
}

int main() {
    static uint16_t a[128 * 16], b[8 * 16];
    MmaJob job;
    job.m = 128;
    job.n = 8;

    memset(a, 0, sizeof a);
    for (int m = 0; m < 16; ++m) a[m * 16] = A_VALUES[m];
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k) b[n * 16 + k] = B_VALUES[n];
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump, part 1");
    printf("part 1: one product per element, A[m][0] * B[n]; half-precision bits of B for n = 0..7 are");
    for (int n = 0; n < 8; ++n) printf(" %04x", B_VALUES[n]);
    printf("\n");
    print_rows("A =");

    memset(a, 0, sizeof a);
    for (int m = 0; m < 16; ++m) {
        a[m * 16] = 0x3c00;
        a[m * 16 + 1] = A_VALUES[m];
    }
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k) b[n * 16 + k] = 0x3c00;
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump, part 2");
    printf("part 2: 1.0 at k = 0 and the value at k = 1, B = 1.0; the exact sum is 1 + the value (3f800000 is 1.0)\n");
    print_rows("1.0 +");
    return 0;
}
