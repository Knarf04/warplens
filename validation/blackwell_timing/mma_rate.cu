// mma_rate: rate and latency of tcgen05.mma, against M, N, the operand kind, and the batch size.
//
// One thread issues J MMAs back to back into the same accumulator, then tcgen05.commit to
// an mbarrier, and waits on it; `reps` times. The slope of time against reps is the round
// trip T(J) of one batch. A straight line through T against J (J = 1, 2, 4, 8) gives
//   slope      ns per MMA when MMAs queue behind each other (throughput)
//   intercept  what a batch costs beyond its MMAs: pipeline latency, commit, wake-up
//
// Calibrates:
//   `tensor_core_tflops_f16`, `tensor_core_tflops_tf32` (whole-GPU rates in the config):
//        TFLOPS = 2 * M * N * K / (ns per MMA) * sm_count / 1000
//     from M=128, N=256. fit.py uses the ctas=allsm variant (one CTA per SM, all issuing)
//     when present, else ctas=1, which assumes the rate scales with the SM count.
//   `mma_latency_clk`:
//        mma_latency_clk = config value + intercept(measured) * clock_ghz
//                          - intercept(predicted) * clock_ghz(config)
//     from kind f16, M=128, N=256, ctas=1. The model has one latency for every shape and kind.
// Structure check: the model says ns per MMA is proportional to M * N * K whatever the
// shape. The per-shape table (M = 64, 128; N = 32, 128, 256) tests that.
//
// K per instruction is fixed by the kind: f16 -> 16, tf32 -> 8, i8 -> 32 (32 bytes per
// operand row in each case). kind=i8 is hardware-only: warplens implements f16 and tf32,
// and the timing model has no rate for other kinds. It runs only with `--only i8`
// (run_all.sh does that in a separate process, so a failure cannot affect the rest).
//
// Operands: A and B are the same 32 KiB of shared memory holding ones, K-major with the
// 128-byte swizzle layout (as examples/04_pipelined_gemm). Values do not matter for timing;
// `check` is the accumulator element D[0][0] after the last batch, which must be K * J.
//
// Blackwell only: simulator and sm_100a here.
#include "blackwell.h"

#define OPERAND_BYTES 32768  // 256 rows of 128 bytes

// K-major operand, 128-byte swizzle: start / 16, 1024 bytes between 8-row groups.
__device__ __forceinline__ uint64_t smem_desc(uint32_t start) {
    return (uint64_t)(start >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46)
           | (2ull << 61);
}

#define DEFINE_MMA(name, kind) \
    __device__ __forceinline__ void name(uint32_t tmem, uint64_t desc_a, uint64_t desc_b, \
                                         uint32_t idesc, uint32_t accumulate) { \
        asm volatile("{\n" \
                     ".reg .pred p;\n" \
                     "setp.ne.b32 p, %4, 0;\n" \
                     "tcgen05.mma.cta_group::1.kind::" kind " [%0], %1, %2, %3, {%5, %5, %5, %5}, p;\n" \
                     "}" \
                     :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate), "r"(0u) \
                     : "memory"); \
    }
DEFINE_MMA(mma_f16, "f16")
DEFINE_MMA(mma_tf32, "tf32")
DEFINE_MMA(mma_i8, "i8")

#define BATCH_LOOP(mma) \
    for (int i = 0; i < reps; ++i) { \
        for (int j = 0; j < batch; ++j) mma(tmem, desc, desc, idesc, j); \
        mma_commit(&bar); \
        mbarrier_wait(&bar, i & 1); \
    }

// block = 32 threads (one warp). kind: 0 f16, 1 tf32, 2 i8. fill: the 32-bit pattern of ones.
__global__ void mma_kernel(uint64_t* out, int reps, int batch, uint32_t idesc, int kind, uint32_t fill) {
    __shared__ alignas(1024) uint32_t operand[OPERAND_BYTES / 4];
    __shared__ uint64_t bar;
    __shared__ uint32_t tmem_base;
    for (int i = threadIdx.x; i < OPERAND_BYTES / 4; i += blockDim.x) operand[i] = fill;
    if (threadIdx.x == 0) mbarrier_init(&bar, 1);
    tmem_alloc(&tmem_base, 256);
    __syncthreads();
    uint32_t tmem = tmem_base;
    if (threadIdx.x == 0) {
        uint64_t desc = smem_desc(smem_addr(operand));
        uint64_t t0 = BENCH_NOW();
        if (kind == 0) { BATCH_LOOP(mma_f16) }
        else if (kind == 1) { BATCH_LOOP(mma_tf32) }
        else { BATCH_LOOP(mma_i8) }
        uint64_t t1 = BENCH_NOW();
        if (blockIdx.x == 0) out[OUT_CLK] = t1 - t0;
    }
    __syncthreads();
    uint32_t d00 = tmem_ld1(tmem);  // lane t of warp 0 reads row t, column 0
    tmem_wait_ld();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        out[OUT_CHECK] = kind == 2 ? (uint64_t)d00 : (uint64_t)__uint_as_float(d00);
    __syncthreads();
    tmem_dealloc(tmem, 256);
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "mma_rate");
    // Instruction descriptor: D format at bit 4 (1 = f32, 2 = s32), A and B formats at
    // bits 7 and 10 (f16 = 0, tf32 = 2, s8 = 0), N / 8 at bit 17, M / 16 at bit 24.
    struct Kind { const char* name; int k; uint32_t formats, fill; } kinds[] = {
        {"f16", 16, (1u << 4), 0x3C003C00u},
        {"tf32", 8, (1u << 4) | (2u << 7) | (2u << 10), 0x3F800000u},
        {"i8", 32, (2u << 4), 0x01010101u},
    };
    bool want_i8 = o.only && strstr(o.only, "i8");
    for (int allsm = 0; allsm < 2; ++allsm)
        for (int kind = 0; kind < 3; ++kind)
            for (int m : {64, 128})
                for (int n : {32, 128, 256})
                    for (int batch : {1, 2, 4, 8}) {
                        if (kind == 2 && (!want_i8 || o.sim)) continue;
                        if ((allsm || kind == 2) && (m != 128 || n != 256)) continue;
                        const Kind& kd = kinds[kind];
                        int ctas = allsm ? o.sm_count : 1;
                        std::string id = std::string("mma_rate/kind=") + kd.name + "/m=" + std::to_string(m) +
                                         "/n=" + std::to_string(n) + "/batch=" + std::to_string(batch) +
                                         "/ctas=" + (allsm ? "allsm" : "1");
                        uint32_t idesc = kd.formats | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24);
                        std::string extra = "kind=" + std::string(kd.name) + " m=" + std::to_string(m) +
                                            " n=" + std::to_string(n) + " k=" + std::to_string(kd.k) +
                                            " batch=" + std::to_string(batch) + " nctas=" + std::to_string(ctas);
                        measure(o, id, extra, {50000, 100000},
                                allsm ? std::vector<long>{2, 4} : std::vector<long>{4, 8}, [&](long reps) {
                                    mma_kernel<<<ctas, 32>>>(o.d_out, (int)reps, batch, idesc, kind, kd.fill);
                                });
                    }
    return 0;
}
