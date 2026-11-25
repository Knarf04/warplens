// Which tensor-memory lanes a warp reaches with tcgen05.ld.32x32b, for each lane field.
//
// An MMA with M = 128, N = 8 writes D[m][n] = 256 m + n (exact in float32), so every
// accumulator names its row and column; row m of such a result is on lane m, which the
// CUTLASS tutorials establish on hardware. All four warps then read with the SAME address,
// base + (L << 16), and the probe prints which rows each warp saw.
//
// Claim tested (docs/history/m3.md, "tcgen05.ld"; the rule that was wrong before): warp w of a
// warpgroup reaches lanes 32 (w % 4) .. + 31 and nothing else. The lane field is a local
// index: the physical lane is 32 (w % 4) + L % 32 + thread lane. So L = 0, 32, 64 and 96,
// which "name" the four windows, all read the issuing warp's own window, and no warp can
// read another's.
// Cases with L % 32 != 0 (own process each): the transfer would run past local row 31. The
// simulator refuses these as unsupported; what hardware does is not known.
//
//   lane_window            L = canonical (32 * warp), 0, 32, 64, 96
//   lane_window <L>        one other lane field, e.g. 16, 1, 31, 48
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

static void run(uint32_t read_lane, const MmaJob& base) {
    MmaJob job = base;
    job.read_lane = read_lane;
    if (read_lane == READ_CANONICAL) printf("lane field 32 * warp (each warp names its own window):\n");
    else printf("lane field %u in all four warps:\n", read_lane);
    fflush(stdout);
    run_mma(image, job, out, "mma_dump");
    for (int warp = 0; warp < 4; ++warp) {
        int rows[32];
        bool consecutive = true, columns_ok = true;
        for (int lane = 0; lane < 32; ++lane) {
            int tid = warp * 32 + lane;
            float v0 = bits_float(out[tid * 8]);
            rows[lane] = (v0 >= 0 && v0 < 32768 && v0 == floorf(v0)) ? (int)v0 / 256 : -1;
            for (int n = 0; n < 8; ++n) columns_ok = columns_ok && bits_float(out[tid * 8 + n]) == v0 + n;
            consecutive = consecutive && rows[lane] >= 0 && rows[lane] == rows[0] + lane;
        }
        if (consecutive && columns_ok) {
            printf("  warp %d: thread lanes 0-31 read rows %d-%d\n", warp, rows[0], rows[31]);
        } else {
            printf("  warp %d: rows by thread lane (-1: not a row value; columns %s):", warp,
                   columns_ok ? "consistent" : "INCONSISTENT");
            for (int lane = 0; lane < 32; ++lane) printf(" %d", rows[lane]);
            printf("\n    raw column 0:");
            for (int lane = 0; lane < 32; ++lane) printf(" %08x", out[(warp * 32 + lane) * 8]);
            printf("\n");
        }
    }
}

int main(int argc, char** argv) {
    uint16_t a[128 * 16] = {0}, b[8 * 16] = {0};
    for (int m = 0; m < 128; ++m) {
        a[m * 16] = half_bits((float)m);
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < 8; ++n) {
        b[n * 16] = half_bits(256.0f);
        b[n * 16 + 1] = half_bits((float)n);
    }
    MmaJob job;
    job.m = 128;
    job.n = 8;
    job.desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    printf("MMA: M = 128, N = 8, D[m][n] = 256 m + n; row m is expected on lane m\n");
    if (argc > 1) {
        run((uint32_t)atoi(argv[1]), job);
        return 0;
    }
    run(READ_CANONICAL, job);
    for (uint32_t lane : {0u, 32u, 64u, 96u}) run(lane, job);
    return 0;
}
