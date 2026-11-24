// The negate bits of the tcgen05.mma instruction descriptor, and accumulation with them.
//
// D[m][n] = 256 m + n + 1 (A[m] = (m, 1, 0, ...), B[n] = (256, n + 1, 0, ...); exact, never
// zero). The MMA is run with neither, one, the other and both negate bits set, once
// overwriting and once issued twice (overwrite, then accumulate), and a few elements are
// printed as raw bits with the count of elements equal to the expected signed value.
//
// Claims tested (docs/numerics.md "Negation flags"; warplens/units/tcgen05.py: bits 13 and
// 14 of the instruction descriptor negate A and B):
//   * each flag flips the sign of the product; both together leave it unchanged;
//   * a second MMA with enable-input-d adds its (signed) product to the first.
//
//   negate
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

int main() {
    uint16_t a[128 * 16] = {0}, b[8 * 16] = {0};
    for (int m = 0; m < 128; ++m) {
        a[m * 16] = half_bits((float)m);
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < 8; ++n) {
        b[n * 16] = half_bits(256.0f);
        b[n * 16 + 1] = half_bits((float)(n + 1));
    }
    MmaJob job;
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    static const char* names[4] = {"neither", "negate A (bit 13)", "negate B (bit 14)", "negate A and B"};
    for (uint32_t repeat = 1; repeat <= 2; ++repeat)
        for (uint32_t flags = 0; flags < 4; ++flags) {
            job.idesc_extra = flags << 13;
            job.repeat = repeat;
            run_mma(image, job, out, "mma_dump");
            float sign = (flags == 1 || flags == 2) ? -1.0f : 1.0f;
            int right = 0;
            for (int m = 0; m < 128; ++m)
                for (int n = 0; n < 8; ++n)
                    right += out[m * 8 + n] == float_bits(sign * repeat * (256.0f * m + n + 1));
            printf("%-18s issued %u time(s): D[0][0] = %08x, D[1][2] = %08x, D[127][7] = %08x; "
                   "%d of 1024 elements equal %s%u (256 m + n + 1)\n",
                   names[flags], repeat, out[0], out[1 * 8 + 2], out[127 * 8 + 7], right, sign < 0 ? "-" : "+", repeat);
        }
    return 0;
}
