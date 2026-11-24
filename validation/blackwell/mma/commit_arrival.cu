// tcgen05.commit: one arrival per commit, made once the issuing thread's MMAs are done.
//
// M = 128, N = 8, kind::f16, product P[m][n] = 256 m + n + 1 (the pattern of mma/negate.cu).
// The mbarrier is initialised with TWO arrivals per phase. Thread 0:
//   1. issues eight MMAs to the same accumulator (overwrite, then seven accumulating)
//   2. tcgen05.commit              one arrival: the phase must stay open
//   3. polls test_wait.parity 0 a bounded number of times (4000) and records the answer
//   4. tcgen05.commit again        the second arrival: the phase completes
// Every thread then waits on the mbarrier and loads the accumulator.
//
// Claims tested (docs/m3.md; warplens/units/tcgen05.py `_commit`):
//   * a commit makes exactly one arrival (phase 0 still open after the first, complete
//     after the second);
//   * what is loaded after waiting on the mbarrier is final: 8 P, bit for bit (exact:
//     every partial sum is an integer below 2^24).
//
//   commit_arrival
#include "../tcgen05_common.h"

#define N 8
#define CHAIN 8

extern "C" __global__ void mma_commit_arrival(const uint32_t* image, uint64_t desc_a, uint64_t desc_b,
                                              uint32_t idesc, uint32_t* out, uint32_t* state) {
    __shared__ alignas(1024) uint32_t operands[OPERAND_BYTES / 4];
    __shared__ uint64_t bar;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    for (uint32_t i = tid; i < OPERAND_BYTES / 4; i += blockDim.x) operands[i] = image[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&bar, 2);
        mbar_init_fence();
    }
    __syncthreads();  // see mma_dump in ../tcgen05_common.h
    if (warp == 0) tmem_alloc(&tmem_base, 32);
    __syncthreads();
    uint32_t d = tmem_base, own = d + ((warp * 32) << 16);
    if (tid == 0) {
        uint64_t where = (uint64_t)((smem_addr(operands) >> 4) & 0x3fff);
        for (uint32_t i = 0; i < CHAIN; ++i) mma_f16(d, desc_a + where, desc_b + where, idesc, i);
        mma_commit(&bar);
        state[0] = mbar_poll(&bar, 0, 4000);
        mma_commit(&bar);
    }
    mbar_wait(&bar, 0);
    if (tid == 0) state[1] = mbar_test(&bar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (uint32_t c = 0; c < N; ++c) {
        uint32_t v = tmem_ld_32x32b(own + c);
        tmem_wait_ld();
        out[tid * N + c] = v;
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(d, 32);
}

static uint8_t image[OPERAND_BYTES];
static uint32_t out[128 * N];

int main() {
    uint16_t a[128 * 16] = {0}, b[N * 16] = {0};
    for (int m = 0; m < 128; ++m) {
        a[m * 16] = half_bits((float)m);
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < N; ++n) {
        b[n * 16] = half_bits(256.0f);
        b[n * 16 + 1] = half_bits((float)(n + 1));
    }
    uint64_t desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    uint64_t desc_b = place_k_major(image, B_OFFSET, b, N, 2, LAYOUTS[3]);
    uint32_t* d_image = dev_copy((const uint32_t*)image, OPERAND_BYTES / 4);
    uint32_t* d_out = dev_alloc<uint32_t>(128 * N);
    uint32_t* d_state = dev_alloc<uint32_t>(2);
    uint32_t state[2];
    dev_fill(d_out, 128 * N, 0);
    dev_fill(d_state, 2, 0xff);
    mma_commit_arrival<<<1, 128>>>(d_image, desc_a, desc_b, mma_idesc(128, N, 0, 0, 0), d_out, d_state);
    probe_sync("mma_commit_arrival");
    dev_read(out, d_out, 128 * N);
    dev_read(state, d_state, 2);
    int right = 0;
    for (int m = 0; m < 128; ++m)
        for (int n = 0; n < N; ++n) right += out[m * N + n] == float_bits(CHAIN * (256.0f * m + n + 1));
    printf("mbarrier with 2 arrivals per phase, %d chained MMAs\n", CHAIN);
    printf("after the first commit:  phase 0 completed %u (polled)\n", state[0]);
    printf("after the second commit: phase 0 completed %u (after the wait)\n", state[1]);
    printf("loaded after the wait: D[0][0] = %08x, D[1][2] = %08x, D[127][7] = %08x; %d of %d elements equal "
           "%d (256 m + n + 1)\n", out[0], out[N + 2], out[127 * N + 7], right, 128 * N, CHAIN);
    return 0;
}
