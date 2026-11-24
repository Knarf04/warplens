// tcgen05.cp: what it copies from shared memory to tensor memory. A RECORDING, not a test.
//
// tcgen05.cp is outside the v1 scope (PLAN.md, "v1 instruction scope"; it serves the
// block-scaled MMAs, which are v2) and the simulator refuses it, so this run has NO
// PREDICTION. It exists so that the rental brings back what an implementation will need:
// which shared bytes each shape puts in which (lane, column) of tensor memory.
//
// 16384 bytes of shared memory hold a pattern that names its own position: the 32-bit word
// at byte offset o holds o. The first 16 columns of tensor memory hold a sentinel
// (EEEEEEEE). Thread 0 issues ONE tcgen05.cp to (lane 0, column 0) with the descriptor an
// unswizzled 128-row K-major MMA operand has in the other probes (start 0, leading offset
// 2048, stride 128, layout "none"), then tcgen05.commit to an mbarrier on which every
// thread waits; that commit covers a cp is itself an assumption (CUTLASS's block-scaled
// kernels rely on it). Every warp then loads the 16 columns of its 32 lanes (32x32b).
//
// Printed per shape: for lanes 0-5, 32, 64, 96 and 127 the 16 columns as byte offsets (hex)
// or `....` for a cell that still holds the sentinel; and how many of the 128 x 16 cells
// were written. The forms are the four CUTLASS uses for one CTA (cute/arch/copy_sm100.hpp):
//
//   cp                all four, one launch each
//   cp <shape>        one of: 128x256b  128x128b  4x256b  32x128b.warpx4
// MAY FAULT: nothing here has run on hardware.
#include "../tcgen05_common.h"

#define COLS 16
#define PATTERN_WORDS 4096

extern "C" __global__ void cp_dump(const uint32_t* image, uint64_t desc, uint32_t shape, uint32_t* out) {
    __shared__ alignas(1024) uint32_t source[PATTERN_WORDS];
    __shared__ uint64_t bar;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    for (uint32_t i = tid; i < PATTERN_WORDS; i += blockDim.x) source[i] = image[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&bar, 1);
        mbar_init_fence();
    }
    __syncthreads();  // see mma_dump in ../tcgen05_common.h
    if (warp == 0) tmem_alloc(&tmem_base, 32);
    __syncthreads();
    uint32_t base = tmem_base, own = base + ((warp * 32) << 16);
    for (uint32_t c = 0; c < COLS; ++c) tmem_st_32x32b(own + c, MMA_TMEM_SENTINEL);
    tmem_wait_st();
    __syncthreads();
    if (tid == 0) {
        uint64_t d = desc + (uint64_t)((smem_addr(source) >> 4) & 0x3fff);
        if (shape == 0) asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(base), "l"(d) : "memory");
        if (shape == 1) asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(base), "l"(d) : "memory");
        if (shape == 2) asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;" :: "r"(base), "l"(d) : "memory");
        if (shape == 3)
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(base), "l"(d) : "memory");
        mma_commit(&bar);
    }
    mbar_wait(&bar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (uint32_t c = 0; c < COLS; ++c) {
        uint32_t v = tmem_ld_32x32b(own + c);
        tmem_wait_ld();
        out[tid * COLS + c] = v;
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, 32);
}

static const char* SHAPES[4] = {"128x256b", "128x128b", "4x256b", "32x128b.warpx4"};

int main(int argc, char** argv) {
    static uint32_t image[PATTERN_WORDS], out[128 * COLS];
    for (uint32_t i = 0; i < PATTERN_WORDS; ++i) image[i] = 4 * i;
    uint32_t* d_image = dev_copy(image, PATTERN_WORDS);
    uint32_t* d_out = dev_alloc<uint32_t>(128 * COLS);
    uint64_t desc = mma_smem_desc(0, 2048, 128, 0);
    const int lanes[10] = {0, 1, 2, 3, 4, 5, 32, 64, 96, 127};
    int known = 0;
    for (int shape = 0; shape < 4; ++shape) {
        if (argc > 1 && strcmp(argv[1], SHAPES[shape])) continue;
        ++known;
        dev_fill(d_out, 128 * COLS, 0xff);
        cp_dump<<<1, 128>>>(d_image, desc, (uint32_t)shape, d_out);
        probe_sync("cp_dump");
        dev_read(out, d_out, 128 * COLS);
        int written = 0;
        for (int i = 0; i < 128 * COLS; ++i) written += out[i] != MMA_TMEM_SENTINEL;
        printf("tcgen05.cp.cta_group::1.%s, descriptor %016llx: %d of %d cells written\n", SHAPES[shape],
               (unsigned long long)desc, written, 128 * COLS);
        printf("  source byte offset (hex) in columns 0-15; .... = not written\n");
        for (int lane : lanes) {
            printf("  lane %3d:", lane);
            for (int c = 0; c < COLS; ++c) {
                if (out[lane * COLS + c] == MMA_TMEM_SENTINEL) printf(" ....");
                else printf(" %04x", out[lane * COLS + c]);
            }
            printf("\n");
        }
    }
    if (!known) {
        fprintf(stderr, "usage: cp [128x256b|128x128b|4x256b|32x128b.warpx4]\n");
        return 2;
    }
    return 0;
}
