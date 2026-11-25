// How one tcgen05.mma (kind::f16) adds its sixteen products, as raw result bits.
//
// Half-precision products are exact in float32, so any difference between two correct
// implementations comes from the order and the precision of the additions.
//
// Part 1, cancellation rows. Row m of A has three non-zero entries: +2^14 at position i,
// 2^-10 at position j and -2^14 at position l, for every ordered triple of the positions
// {0, 1, 2, 7, 8, 15}. Every row of B is constant: B[n][k] = 1024 (n + 1). The three
// products for column n are +2^24 (n + 1), n + 1 and -2^24 (n + 1); the exact sum is n + 1.
// A float32 sum that adds the small term to a large one first loses it (or rounds it to the
// next multiple of the spacing), so each printed value shows where in the order the small
// term was added. An implementation that sums exactly and rounds once prints n + 1 everywhere.
//
// Part 2, 128 x 8 results of pseudo-random operands, and part 3, the same MMA issued three
// times (overwrite, accumulate, accumulate).
//
// Parts 1 to 3 were measured on a B200 on 2026-11-25 (validation/blackwell/measured/b200/mma/
// accumulation_order.txt) and gave the model the simulator has since (docs/numerics.md,
// "tcgen05.mma"): the products and the old accumulator are one sum, each term lined up under
// the largest and cut off toward zero 25 bits below its leading bit, the cut terms added
// exactly, the sum cut off toward zero to float32. Part 1 printed 1 2 3 4 4 6 6 8 in every
// row, whatever the positions. Parts 4 to 6 ask for what parts 1 to 3 could not show and
// the model takes from the book's kernels (captured on the same day, replayed bit for bit):
//
// Part 4, part 1 with the small entry negative (-2^-10): exact sums -1 .. -8. Cut toward
// zero gives -1 -2 -3 -4 -4 -6 -6 -8; cut toward minus infinity would give -6 and -8 for the
// fifth and seventh.
//
// Part 5, part 1 with operands whose significands multiply to 2 or more: +-12288 (1.5 * 2^13)
// in A and 1536 (n + 1) in B, so the large products are 2.25 * 2^23 (n + 1) and the small
// one is 1.5 (n + 1). The model lines terms up by the SUM of the operands' exponents, as if
// that product had not carried: for n = 0 the small term 1.5 is then kept whole, where
// lining up under the product's own leading bit (2^24) would cut it to 1.
//
// Part 6, the old accumulator as the small term: every accumulator is first set to 7.0
// (tcgen05.st), then one accumulating MMA adds +2^14 and -2^14 times 1024 (n + 1), which
// cancel. If the accumulator is lined up and cut like a product the results are
// 7 7 7 6 6 6 6 4; if it were added whole they would all be 7.
//
// Claim tested (docs/numerics.md, "tcgen05.mma"; warplens/units/tcgen05.py:mma_sum): the
// model above, bit for bit. compare.py reports a difference in units in the last place.
//
//   accumulation_order
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

static void print_rows(int rows, const char* (*label)(int)) {
    for (int m = 0; m < rows; ++m) {
        printf("  %s", label(m));
        for (int n = 0; n < 8; ++n) printf(" %08x", out[m * 8 + n]);
        printf("\n");
    }
}

static int triples[128][3], triple_count = 0;
static const char* triple_label(int m) {
    static char text[64];
    snprintf(text, sizeof text, "+2^14 at k=%2d, 2^-10 at k=%2d, -2^14 at k=%2d:", triples[m][0], triples[m][1],
             triples[m][2]);
    return text;
}
static const char* row_label(int m) {
    static char text[32];
    snprintf(text, sizeof text, "row %3d:", m);
    return text;
}

int main() {
    static uint16_t a[128 * 16], b[8 * 16];
    MmaJob job;
    job.m = 128;
    job.n = 8;

    // Part 1.
    const int positions[6] = {0, 1, 2, 7, 8, 15};
    for (int i : positions)
        for (int j : positions)
            for (int l : positions)
                if (i != j && j != l && i != l) {
                    triples[triple_count][0] = i;
                    triples[triple_count][1] = j;
                    triples[triple_count][2] = l;
                    ++triple_count;
                }
    memset(a, 0, sizeof a);
    for (int m = 0; m < triple_count; ++m) {
        a[m * 16 + triples[m][0]] = half_bits(16384.0f);
        a[m * 16 + triples[m][1]] = half_bits(1.0f / 1024.0f);
        a[m * 16 + triples[m][2]] = half_bits(-16384.0f);
    }
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k) b[n * 16 + k] = half_bits(1024.0f * (n + 1));
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump, part 1");
    printf("part 1: cancellation rows; columns n = 0..7, exact sums are 1..8 = 3f800000 40000000 40400000 "
           "40800000 40a00000 40c00000 40e00000 41000000\n");
    print_rows(triple_count, triple_label);

    // Part 4: the small entry negative.
    for (int m = 0; m < triple_count; ++m) a[m * 16 + triples[m][1]] = half_bits(-1.0f / 1024.0f);
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump, part 4");
    printf("part 4: part 1 with -2^-10 for the small entry; exact sums are -1..-8 = bf800000 c0000000 c0400000 "
           "c0800000 c0a00000 c0c00000 c0e00000 c1000000\n");
    print_rows(triple_count, triple_label);

    // Part 5: significands that multiply to 2 or more.
    memset(a, 0, sizeof a);
    for (int m = 0; m < triple_count; ++m) {
        a[m * 16 + triples[m][0]] = half_bits(12288.0f);
        a[m * 16 + triples[m][1]] = half_bits(1.0f / 1024.0f);
        a[m * 16 + triples[m][2]] = half_bits(-12288.0f);
    }
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k) b[n * 16 + k] = half_bits(1536.0f * (n + 1));
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    run_mma(image, job, out, "mma_dump, part 5");
    printf("part 5: +12288 and -12288 in place of +-2^14 (labels as in part 1) and B = 1536 (n + 1); exact sums "
           "are 1.5 (n + 1) = 3fc00000 40400000 40900000 40c00000 40f00000 41100000 41280000 41400000\n");
    print_rows(triple_count, triple_label);

    // Part 6: the old accumulator is the small term.
    memset(a, 0, sizeof a);
    for (int m = 0; m < triple_count; ++m) {
        a[m * 16 + triples[m][0]] = half_bits(16384.0f);
        a[m * 16 + triples[m][2]] = half_bits(-16384.0f);
    }
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k) b[n * 16 + k] = half_bits(1024.0f * (n + 1));
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    job.prefill = 2;
    job.prefill_bits = 0x40e00000u;  // 7.0f
    run_mma(image, job, out, "mma_dump, part 6");
    job.prefill = 0;
    job.prefill_bits = MMA_TMEM_SENTINEL;
    printf("part 6: accumulators set to 7.0 (40e00000), then one accumulating MMA of +2^14 and -2^14 times "
           "1024 (n + 1); the exact sum is 7 everywhere\n");
    print_rows(triple_count, triple_label);

    // Parts 2 and 3: multiples of 1/1024 in (-2, 2), exactly representable in half precision.
    uint32_t state = 7u;
    for (uint16_t& v : a) { state = state * 1664525u + 1013904223u; v = half_bits(((int)(state >> 20) - 2048) / 1024.0f); }
    for (uint16_t& v : b) { state = state * 1664525u + 1013904223u; v = half_bits(((int)(state >> 20) - 2048) / 1024.0f); }
    memset(image, 0, sizeof image);
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    printf("part 2: operands are multiples of 1/1024 in (-2, 2) from the generator x -> 1664525 x + 1013904223, "
           "seed 7 (A first, row-major, then B; value = ((x >> 20) - 2048) / 1024)\n");
    printf("  A[0][0..3] = %04x %04x %04x %04x, B[0][0..3] = %04x %04x %04x %04x (half-precision bits)\n", a[0], a[1],
           a[2], a[3], b[0], b[1], b[2], b[3]);
    run_mma(image, job, out, "mma_dump, part 2");
    print_rows(128, row_label);
    job.repeat = 3;
    run_mma(image, job, out, "mma_dump, part 3");
    printf("part 3: the same MMA three times into the same accumulators (D = P, then D += P twice)\n");
    print_rows(128, row_label);
    return 0;
}
