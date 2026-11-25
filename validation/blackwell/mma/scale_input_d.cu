// The scale-input-d operand of tcgen05.mma: the old accumulator is multiplied by 2^-s
// before the new product is added.
//
// M = 128, N = 8, kind::f16. A[m] = (m + 1, 1, 0, ...). Two B operands in the same buffer:
// B1[n] = (32768, 0, ...) and B2[n] = (0, n, 0, ...). Thread 0 issues
//   1. an overwriting MMA with B1:                  D[m][n] = 2^15 (m + 1)
//   2. an accumulating MMA with B2 and scale s:     D[m][n] = 2^(15 - s) (m + 1) + n
// for s = 0, 1, 7 and 15 (the operand is an immediate, so the kernel has one instruction per
// value), and once with the six-operand form without a scale as the control. Every value is
// an integer below 2^24, so the result is exact whatever the order of operations.
//
// Claim tested (docs/history/m3.md; warplens/units/tcgen05.py `_mma`: `scale`): scale-input-d = s
// multiplies the old accumulator by 2^-s before the addition, and nothing else.
//
//   scale_input_d
#include "../tcgen05_common.h"

#define B2_OFFSET (B_OFFSET + 8192)
#define N 8

#define MMA_SCALED(s)                                                                                   \
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"                                              \
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p, " #s ";\n}"                    \
                 :: "r"(d), "l"(desc_a), "l"(desc_b2), "r"(idesc) : "memory")

// `scale` is 0, 1, 7 or 15, or 0xff for the form without the operand.
extern "C" __global__ void mma_scale(const uint32_t* image, uint64_t desc_a, uint64_t desc_b1, uint64_t desc_b2,
                                     uint32_t idesc, uint32_t scale, uint32_t* out) {
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
    uint32_t d = tmem_base;
    if (tid == 0) {
        uint64_t where = (uint64_t)((smem_addr(operands) >> 4) & 0x3fff);
        desc_a += where;
        desc_b1 += where;
        desc_b2 += where;
        mma_f16(d, desc_a, desc_b1, idesc, 0);
        switch (scale) {
            case 0: MMA_SCALED(0); break;
            case 1: MMA_SCALED(1); break;
            case 7: MMA_SCALED(7); break;
            case 15: MMA_SCALED(15); break;
            default: mma_f16(d, desc_a, desc_b2, idesc, 1); break;
        }
        mma_commit(&bar);
    }
    mbar_wait(&bar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t own = d + ((warp * 32) << 16);
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
    uint16_t a[128 * 16] = {0}, b1[N * 16] = {0}, b2[N * 16] = {0};
    for (int m = 0; m < 128; ++m) {
        a[m * 16] = half_bits((float)(m + 1));
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < N; ++n) {
        b1[n * 16] = half_bits(32768.0f);
        b2[n * 16 + 1] = half_bits((float)n);
    }
    uint64_t desc_a = place_k_major(image, 0, a, 128, 2, LAYOUTS[3]);
    uint64_t desc_b1 = place_k_major(image, B_OFFSET, b1, N, 2, LAYOUTS[3]);
    uint64_t desc_b2 = place_k_major(image, B2_OFFSET, b2, N, 2, LAYOUTS[3]);
    uint32_t* d_image = dev_copy((const uint32_t*)image, OPERAND_BYTES / 4);
    uint32_t* d_out = dev_alloc<uint32_t>(128 * N);
    printf("old accumulator 2^15 (m + 1), new product n; expected 2^(15 - s) (m + 1) + n\n");
    for (uint32_t scale : {0xffu, 0u, 1u, 7u, 15u}) {
        dev_fill(d_out, 128 * N, 0);
        mma_scale<<<1, 128>>>(d_image, desc_a, desc_b1, desc_b2, mma_idesc(128, N, 0, 0, 0), scale, d_out);
        probe_sync("mma_scale");
        dev_read(out, d_out, 128 * N);
        uint32_t s = scale == 0xff ? 0 : scale;
        int right = 0;
        for (int m = 0; m < 128; ++m)
            for (int n = 0; n < N; ++n)
                right += out[m * N + n] == float_bits((float)(1u << (15 - s)) * (m + 1) + n);
        if (scale == 0xff) printf("no scale operand:  ");
        else printf("scale-input-d %2u:  ", scale);
        printf("D[0][0] = %08x, D[0][7] = %08x, D[2][3] = %08x, D[127][7] = %08x; %d of %d elements as expected\n",
               out[0], out[7], out[2 * N + 3], out[127 * N + 7], right, 128 * N);
    }
    return 0;
}
