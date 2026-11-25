// The disable-output-lane operand of tcgen05.mma.
//
// M = 128, N = 8, kind::f16, D[m][n] = 256 m + n + 1 (the pattern of mma/negate.cu; row m is
// on TMEM lane m). Every cell is first set to the sentinel 0xEEEEEEEE; then one overwriting
// MMA is issued with a mask of four 32-bit words, and every cell is read back.
//
// Claim tested (docs/history/m3.md; warplens/units/tcgen05.py `_mma`: `disabled`): bit l of word i
// leaves lane 32 i + l unwritten; all other lanes get the result. Masks:
//   {0, 0, 0, 0}                      nothing kept
//   {1, 0, 0, 0}                      lane 0
//   {0, 0x80000000, 0, 0}             lane 63
//   {0xFFFF, 0, 0, 0xFFFF0000}        lanes 0-15 and 112-127
//   {0xFFFFFFFF x 4}                  everything kept
// For each: the lanes that still hold the sentinel in all 8 columns, the lanes that hold the
// pattern in all 8 columns, and the lanes that hold anything else.
//
//   disable_lanes
#include "../tcgen05_common.h"

#define N 8

extern "C" __global__ void mma_disable(const uint32_t* image, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                       uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t* out) {
    __shared__ alignas(1024) uint32_t operands[OPERAND_BYTES / 4];
    __shared__ uint64_t bar;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    for (uint32_t i = tid; i < OPERAND_BYTES / 4; i += blockDim.x) operands[i] = image[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&bar, 1);
        mbar_init_fence();
    }
    __syncthreads();  // see mma_dump in ../tcgen05_common.h
    if (warp == 0) tmem_alloc(&tmem_base, 32);
    __syncthreads();
    uint32_t d = tmem_base, own = d + ((warp * 32) << 16);
    for (uint32_t c = 0; c < N; ++c) tmem_st_32x32b(own + c, MMA_TMEM_SENTINEL);
    tmem_wait_st();
    __syncthreads();
    if (tid == 0) {
        uint64_t where = (uint64_t)((smem_addr(operands) >> 4) & 0x3fff);
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, p;\n}"
                     :: "r"(d), "l"(desc_a + where), "l"(desc_b + where), "r"(idesc), "r"(m0), "r"(m1), "r"(m2),
                        "r"(m3) : "memory");
        mma_commit(&bar);
    }
    mbar_wait(&bar, 0);
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

// Print the lanes of one kind as ranges.
static void print_lanes(const char* title, const int* kind, int which) {
    printf("  %s:", title);
    int count = 0;
    for (int lane = 0; lane < 128; ++lane) {
        if (kind[lane] != which) continue;
        int end = lane;
        while (end + 1 < 128 && kind[end + 1] == which) ++end;
        if (end > lane) printf(" %d-%d", lane, end);
        else printf(" %d", lane);
        count += end - lane + 1;
        lane = end;
    }
    printf(count ? " (%d lanes)\n" : " none\n", count);
}

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
    static const uint32_t masks[5][4] = {{0, 0, 0, 0},
                                         {1, 0, 0, 0},
                                         {0, 0x80000000u, 0, 0},
                                         {0xFFFFu, 0, 0, 0xFFFF0000u},
                                         {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
    for (const uint32_t* mask : masks) {
        dev_fill(d_out, 128 * N, 0);
        mma_disable<<<1, 128>>>(d_image, desc_a, desc_b, mma_idesc(128, N, 0, 0, 0), mask[0], mask[1], mask[2],
                                mask[3], d_out);
        probe_sync("mma_disable");
        dev_read(out, d_out, 128 * N);
        int kind[128];  // 0 sentinel, 1 pattern, 2 something else
        for (int lane = 0; lane < 128; ++lane) {
            int sentinel = 0, pattern = 0;
            for (int n = 0; n < N; ++n) {
                sentinel += out[lane * N + n] == MMA_TMEM_SENTINEL;
                pattern += out[lane * N + n] == float_bits(256.0f * lane + n + 1);
            }
            kind[lane] = sentinel == N ? 0 : pattern == N ? 1 : 2;
        }
        printf("disable-output-lane {0x%08x, 0x%08x, 0x%08x, 0x%08x}\n", mask[0], mask[1], mask[2], mask[3]);
        print_lanes("lanes that kept the sentinel", kind, 0);
        print_lanes("lanes that hold the result  ", kind, 1);
        print_lanes("lanes that hold neither     ", kind, 2);
    }
    return 0;
}
