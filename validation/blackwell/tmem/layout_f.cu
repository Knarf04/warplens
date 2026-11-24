// Where the rows of a tcgen05.mma result land in tensor memory: M = 128, and M = 64
// ("Layout F") at both lane offsets.
//
// Tensor memory is first filled with a sentinel (lanes 0-127 of the result's columns). An
// MMA then writes D[m][n] = 256 m + n to the address base + (L << 16), and all 128 lanes are
// read back with every warp addressing its own 32 lanes. The probe prints, for every lane,
// the row found there, or "." where the sentinel survived.
//
// Claims tested (docs/m3.md, acceptance; cute/atom/mma_traits_sm100_frag.hpp):
//   * M = 128: row m is on lane m;
//   * M = 64 with L = 0: row m is on lane 32 (m / 16) + m % 16, the low 16 lanes of each
//     warp's window; the other 64 lanes are not written;
//   * M = 64 with L = 16: the same shifted by 16 lanes, the high 16 lanes of each window.
// Cases given on the command line (own process each) use another M and lane offset, for
// which the simulator simply adds L to the lane and the hardware behaviour is not known:
//
//   layout_f               M = 128 L = 0; M = 64 L = 0; M = 64 L = 16
//   layout_f <M> <L>       e.g. `64 8`, `64 32`
#include "../tcgen05_common.h"

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * 8];

static void run(int m_dim, uint32_t d_lane) {
    uint16_t a[128 * 16] = {0}, b[8 * 16] = {0};
    for (int m = 0; m < m_dim; ++m) {
        a[m * 16] = half_bits((float)m);
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < 8; ++n) {
        b[n * 16] = half_bits(256.0f);
        b[n * 16 + 1] = half_bits((float)n);
    }
    memset(image, 0, sizeof image);
    MmaJob job;
    job.m = m_dim;
    job.n = 8;
    job.d_lane = d_lane;
    job.prefill = 1;
    job.desc_a = place_k_major(image, 0, a, m_dim, 2, LAYOUTS[3]);
    job.desc_b = place_k_major(image, B_OFFSET, b, 8, 2, LAYOUTS[3]);
    printf("M = %d, N = 8, destination lane field %u: row found on each lane ('.': not written)\n", m_dim, d_lane);
    fflush(stdout);
    run_mma(image, job, out, "mma_dump");
    int strange = 0;
    for (int warp = 0; warp < 4; ++warp) {
        printf("  lanes %3d-%3d:", warp * 32, warp * 32 + 31);
        for (int lane = 0; lane < 32; ++lane) {
            const uint32_t* cells = out + (warp * 32 + lane) * 8;
            float v0 = bits_float(cells[0]);
            bool unwritten = true, row = v0 >= 0 && v0 < 256 * m_dim && v0 == floorf(v0) && (int)v0 % 256 == 0;
            for (int n = 0; n < 8; ++n) {
                unwritten = unwritten && cells[n] == MMA_TMEM_SENTINEL;
                row = row && bits_float(cells[n]) == v0 + n;
            }
            if (unwritten) printf("   .");
            else if (row) printf(" %3d", (int)v0 / 256);
            else { printf("   ?"); ++strange; }
        }
        printf("\n");
    }
    if (strange) {
        printf("  %d lane(s) hold neither a whole row nor the sentinel; raw column 0 of every lane:\n", strange);
        for (int lane = 0; lane < 128; ++lane) printf("%s%08x", lane % 8 ? " " : "\n   ", out[lane * 8]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc > 2) {
        run(atoi(argv[1]), (uint32_t)atoi(argv[2]));
        return 0;
    }
    run(128, 0);
    run(64, 0);
    run(64, 16);
    return 0;
}
