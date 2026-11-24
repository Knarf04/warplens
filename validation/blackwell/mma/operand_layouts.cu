// tcgen05.mma operand layouts: K-major and MN-major, in every swizzle mode.
//
// A (M x K) and B (N x K) hold small integers, so every product and every partial sum is
// exact in float32 and the result does not depend on the order of the additions: D must
// equal A B^T bit for bit. The host places the operands in a shared-memory image with the
// formulas in ../tcgen05_common.h (the claims under test); the kernel copies the image with
// ordinary stores and runs one MMA. A wrong address formula, descriptor field or swizzle
// scrambles the result, and the probe prints how many elements are right and the first
// wrong ones.
//
// Claims tested (docs/m3.md "Design"; warplens/units/tcgen05.py `_operand_bytes`):
//   * K-major addressing for no swizzle, 32B, 64B and 128B, for kind::f16 and kind::tf32
//     (only 128B has run in a real kernel, in the CUTLASS tutorials);
//   * MN-major addressing for kind::f16 in the same four modes, for A, for B and for both
//     (Triton emits this; nothing here has run on hardware);
//   * the descriptor fields: start, leading and stride byte offsets, layout type; and the
//     instruction descriptor's M, N, formats and major-ness bits.
// Case `tf32-mn` (own process): MN-major kind::tf32, which per PTX needs a 128B swizzle
// with 32-byte atomicity that the simulator does not implement; it refuses.
//
//   operand_layouts [tf32-mn]
#include "../tcgen05_common.h"

#define M 128
#define N 64

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * N];
static int a_values[M * 16], b_values[N * 16];

static void run(int m_dim, int fmt, int layout, int a_mn, int b_mn) {
    int size = fmt == 0 ? 2 : 4, k = 32 / size;
    static uint8_t a_raw[M * 16 * 4], b_raw[N * 16 * 4];
    for (int i = 0; i < m_dim * k; ++i) {
        if (size == 2) { uint16_t h = half_bits((float)a_values[i]); memcpy(a_raw + 2 * i, &h, 2); }
        else { float f = (float)a_values[i]; memcpy(a_raw + 4 * i, &f, 4); }
    }
    for (int i = 0; i < N * k; ++i) {
        if (size == 2) { uint16_t h = half_bits((float)b_values[i]); memcpy(b_raw + 2 * i, &h, 2); }
        else { float f = (float)b_values[i]; memcpy(b_raw + 4 * i, &f, 4); }
    }
    memset(image, 0, sizeof image);
    MmaJob job;
    job.m = m_dim;
    job.n = N;
    job.fmt = fmt;
    job.a_mn_major = a_mn;
    job.b_mn_major = b_mn;
    job.prefill = 1;
    job.desc_a = (a_mn ? place_mn_major : place_k_major)(image, 0, a_raw, m_dim, size, LAYOUTS[layout]);
    job.desc_b = (b_mn ? place_mn_major : place_k_major)(image, B_OFFSET, b_raw, N, size, LAYOUTS[layout]);
    printf("kind::%s, M = %d, N = %d, K = %d, swizzle %s, A %s-major, B %s-major\n", fmt == 0 ? "f16" : "tf32", m_dim,
           N, k, LAYOUTS[layout].name, a_mn ? "MN" : "K", b_mn ? "MN" : "K");
    printf("  descriptors (relative to the operand buffer): A %016llx, B %016llx, instruction %08x\n",
           (unsigned long long)job.desc_a, (unsigned long long)job.desc_b,
           mma_idesc(m_dim, N, fmt, a_mn, b_mn));
    fflush(stdout);
    run_mma(image, job, out, "mma_dump");
    int right = 0, shown = 0;
    for (int m = 0; m < m_dim; ++m) {
        int lane = m_dim == 128 ? m : 32 * (m / 16) + m % 16;
        for (int n = 0; n < N; ++n) {
            int expect = 0;
            for (int j = 0; j < k; ++j) expect += a_values[m * k + j] * b_values[n * k + j];
            uint32_t got = out[lane * N + n];
            if (got == float_bits((float)expect)) {
                ++right;
            } else if (shown < 6) {
                printf("  D[%d][%d] = %08x, expected %08x (%d)\n", m, n, got, float_bits((float)expect), expect);
                ++shown;
            }
        }
    }
    printf("  elements equal to A B^T bit for bit: %d of %d\n", right, m_dim * N);
}

int main(int argc, char** argv) {
    uint32_t state = 2026u;
    for (int& v : a_values) { state = state * 1664525u + 1013904223u; v = (int)((state >> 16) % 9) - 4; }
    for (int& v : b_values) { state = state * 1664525u + 1013904223u; v = (int)((state >> 16) % 9) - 4; }
    if (argc > 1 && !strcmp(argv[1], "tf32-mn")) {
        run(M, 2, 3, 1, 0);
        run(M, 2, 3, 0, 1);
        return 0;
    }
    for (int layout = 0; layout < 4; ++layout) run(M, 0, layout, 0, 0);
    for (int layout = 0; layout < 4; ++layout) run(M, 2, layout, 0, 0);
    for (int layout = 0; layout < 4; ++layout) run(64, 0, layout, 0, 0);
    for (int layout = 0; layout < 4; ++layout) {
        run(M, 0, layout, 1, 0);
        run(M, 0, layout, 0, 1);
        run(M, 0, layout, 1, 1);
    }
    return 0;
}
