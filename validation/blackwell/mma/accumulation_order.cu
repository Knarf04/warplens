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
// Claim tested (docs/numerics.md, "tcgen05.mma"): NONE that is expected to hold exactly.
// The simulator adds in float32 in the order NumPy's matrix product uses and says its
// results "can differ in the last bits" from hardware. The prediction is committed so that
// compare.py can report whether the GPU agrees bit for bit and, if not, by how many units
// in the last place (this probe is compared with a tolerance; see the README).
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
