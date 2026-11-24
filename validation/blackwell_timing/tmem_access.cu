// tmem_access: cost of moving data between tensor memory and registers (tcgen05.ld / tcgen05.st).
//
// Calibrates: `tmem_access_clk`, the fixed cost of one tcgen05.ld or tcgen05.st:
//   tmem_access_clk = config value
//        + measured ns/rep * clock_ghz - predicted ns/rep * clock_ghz(config)     [ld/x1]
// (one ld and one tcgen05.wait::ld per repetition; the simulator counts the other
// instructions of the loop). fit.py reports the same quantity from st/x1.
//
// Structure check: the model charges the same for .x1 and .x8 (1 or 8 registers per
// thread). ld/x8 and st/x8 show whether the cost grows with the amount moved.
//
// Modes:
//   ld  v = tcgen05.ld [base + v]; tcgen05.wait::ld   the address depends on the value
//                                                     loaded before (tensor memory holds
//                                                     zeros), so loads cannot overlap
//   st  tcgen05.st [base], v; tcgen05.wait::st
//   ld/x1/nowait, st/x1/nowait   the same loops with one tcgen05.wait after the loop instead
//                                of one per access. The difference to ld/x1 and st/x1 is the
//                                cost of tcgen05.wait, which the model charges one clock
//                                (validation matrix TIM-19).
// One warp, 32x32b shape: every lane reads or writes its own TMEM lane.
// Blackwell only: simulator and sm_100a here.
#include "blackwell.h"

__device__ __forceinline__ void tmem_st1(uint32_t address, uint32_t v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(address), "r"(v) : "memory");
}
__device__ __forceinline__ void tmem_st8(uint32_t address, uint32_t v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %1, %1, %1, %1, %1, %1, %1};"
                 :: "r"(address), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t tmem_ld8(uint32_t address) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
                 : "r"(address));
    return r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7;
}

// block = 32 threads. mode: 0 ld x1, 1 ld x8, 2 st x1, 3 st x8, 4 ld x1 no wait, 5 st x1 no wait.
__global__ void tmem_kernel(uint64_t* out, int reps, int mode) {
    __shared__ uint32_t tmem_base;
    tmem_alloc(&tmem_base, 32);
    __syncthreads();
    uint32_t base = tmem_base, v = 0, zero = reps < 0;
    tmem_st8(base, zero);
    tmem_wait_st();
    uint64_t t0 = BENCH_NOW();
    if (mode == 0) {
        for (int i = 0; i < reps; ++i) { v = tmem_ld1(base + v); tmem_wait_ld(); }
    } else if (mode == 1) {
        for (int i = 0; i < reps; ++i) { v = tmem_ld8(base + v); tmem_wait_ld(); }
    } else if (mode == 2) {
        for (int i = 0; i < reps; ++i) { tmem_st1(base, zero); tmem_wait_st(); }
    } else if (mode == 3) {
        for (int i = 0; i < reps; ++i) { tmem_st8(base, zero); tmem_wait_st(); }
    } else if (mode == 4) {
        for (int i = 0; i < reps; ++i) v = tmem_ld1(base + v);
        tmem_wait_ld();
    } else {
        for (int i = 0; i < reps; ++i) tmem_st1(base, zero);
        tmem_wait_st();
    }
    uint64_t t1 = BENCH_NOW();
    if (threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = v + reps; }
    __syncthreads();
    tmem_dealloc(base, 32);
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "tmem_access");
    const char* modes[] = {"ld/x1", "ld/x8", "st/x1", "st/x8", "ld/x1/nowait", "st/x1/nowait"};
    for (int mode = 0; mode < 6; ++mode)
        measure(o, std::string("tmem_access/") + modes[mode], "", {1000000, 2000000}, {16, 32},
                [&](long reps) { tmem_kernel<<<1, 32>>>(o.d_out, (int)reps, mode); });
    return 0;
}
