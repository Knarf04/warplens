// An MMA operand that does not start on a swizzle period, and the descriptor's base offset.
//
// The shared-memory descriptor has a 3-bit "matrix base offset" at bits 49-51. PTX gives it
// a meaning only for swizzled layouts whose start address is not a multiple of the swizzle
// period (1024, 512, 256 bytes for 128B, 64B, 32B): it is then to hold the 128-byte row of
// the start within the period, (start >> 7) & 7, & 3 or & 1. CUTLASS never sets the field
// (make_umma_desc leaves it 0), whatever the start address.
//
// The simulator (warplens/units/tcgen05.py `_operand_bytes`) refuses a non-zero field and,
// with the field 0, swizzles the ABSOLUTE shared address of every byte: an operand that
// starts j rows into a period is read through the swizzle of rows j, j + 1, ... That is a
// claim about hardware which the aligned operands of the other probes do not test.
//
// The readout trick of mma/swizzle_mismatch.cu: B is the 16 x 16 identity, so D[m][n] is
// the half-precision value the tensor core fetched as A[m][n]. The first 24576 bytes of the
// operand buffer hold a pattern that names its own position, in two runs: every half float
// holds the number of its 128-byte row (0-191), then its position in that row (0-63). So
// D names the shared bytes behind every (m, n), with no model assumed. The output compares
// them with three candidate rules for the byte of element (m, n) at offset `off` of the
// K-major formula (MMA-02), with mask = 7, 3, 1:
//   address    swizzle(start + off)                               the simulator's rule
//   relative   start + swizzle(off)                               the period starts at the operand
//   base       start + (off ^ ((((off >> 7) + field) & mask) << 4))    relative, rows counted from the field
// For a start that is a multiple of 128 with the field set as PTX says, `address` and
// `base` name the same bytes; with the field 0, `address` and `relative` differ.
//
//   base_offset          field 0 (in the v1 scope, predicted): starts 0, 128, 384, 896 and
//                        64 for 128B; 128 and 384 for 64B; 128 for 32B
//   base_offset field    NO PREDICTION (the simulator refuses the field): the same starts
//                        with the field PTX prescribes, plus the field set on an aligned
//                        start and a wrong field. MAY FAULT.
#include "../tcgen05_common.h"

#define N 16
#define B_AT 24576      // the identity, away from anything A's formula can reach
#define PATTERN_BYTES B_AT

struct Case {
    int layout;       // index into LAYOUTS: 1 = 32B, 2 = 64B, 3 = 128B
    uint32_t start;   // of operand A, relative to the 1024-aligned buffer
    uint32_t field;   // matrix base offset, bits 49-51
};
static const Case ZERO_FIELD[] = {{3, 0, 0}, {3, 128, 0}, {3, 384, 0}, {3, 896, 0}, {3, 64, 0},
                                  {2, 128, 0}, {2, 384, 0}, {1, 128, 0}};
static const Case WITH_FIELD[] = {{3, 128, 1}, {3, 384, 3}, {3, 896, 7}, {2, 128, 1}, {2, 384, 3}, {1, 128, 1},
                                  {3, 0, 1}, {3, 128, 7}};

static uint8_t image[OPERAND_BYTES];
static uint32_t rows_read[128 * N], positions_read[128 * N];

// Byte offset of element (m, n) before swizzling, relative to the operand's start.
static uint32_t k_major_offset(int m, int n, const Layout& layout) {
    return (m % 8) * layout.row_bytes + (m / 8) * 8 * layout.row_bytes + (n % 8) * 2 + (n / 8) * 16;
}

static void run_case(const Case& c, uint64_t desc_b) {
    const Layout& layout = LAYOUTS[c.layout];
    uint32_t mask = (1u << layout.bits) - 1;
    uint64_t desc_a = mma_smem_desc(c.start, 16, 8 * layout.row_bytes, layout.code) | ((uint64_t)c.field << 49);
    for (int which = 0; which < 2; ++which) {
        for (uint32_t o = 0; o < PATTERN_BYTES; o += 2) {
            uint16_t h = half_bits(which ? (float)((o & 127) >> 1) : (float)(o >> 7));
            memcpy(image + o, &h, 2);
        }
        MmaJob job;
        job.desc_a = desc_a;
        job.desc_b = desc_b;
        job.n = N;
        run_mma(image, job, which ? positions_read : rows_read, "mma_dump (base offset)");
    }
    int address = 0, relative = 0, base = 0;
    for (int m = 0; m < 128; ++m)
        for (int n = 0; n < N; ++n) {
            uint32_t off = k_major_offset(m, n, layout);
            uint32_t candidates[3] = {swizzle_offset(c.start + off, layout.bits),
                                      c.start + swizzle_offset(off, layout.bits),
                                      c.start + (off ^ ((((off >> 7) + c.field) & mask) << 4))};
            int* counts[3] = {&address, &relative, &base};
            for (int i = 0; i < 3; ++i)
                *counts[i] += rows_read[m * N + n] == float_bits((float)(candidates[i] >> 7))
                              && positions_read[m * N + n] == float_bits((float)((candidates[i] & 127) >> 1));
        }
    printf("layout %s, operand A starts at byte %u of a 1024-aligned buffer, base offset field %u, "
           "descriptor %016llx\n", layout.name, c.start, c.field, (unsigned long long)desc_a);
    printf("  elements read from where each rule says, of %d: address %d, relative %d, base %d\n", 128 * N, address,
           relative, base);
    printf("  shared bytes (128-byte row.half float in the row) behind D[m][0..15], for m = 0-8:\n");
    for (int m = 0; m < 9; ++m) {
        printf("  m %d:", m);
        for (int n = 0; n < N; ++n)
            printf(" %3d.%-2d", (int)bits_float(rows_read[m * N + n]), (int)bits_float(positions_read[m * N + n]));
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int with_field = argc > 1 && !strcmp(argv[1], "field");
    static uint16_t b[N * 16];
    for (int n = 0; n < N; ++n) b[n * 16 + n] = half_bits(1.0f);
    uint64_t desc_b = place_k_major(image, B_AT, b, N, 2, LAYOUTS[3]);
    if (with_field)
        for (const Case& c : WITH_FIELD) run_case(c, desc_b);
    else
        for (const Case& c : ZERO_FIELD) run_case(c, desc_b);
    return 0;
}
