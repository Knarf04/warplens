// smem_wavefronts: cost of a shared-memory access, and of each extra bank-conflict wavefront.
//
// Calibrates: `smem_wavefront_clk`. The 32 lanes of a warp access word `lane * stride`.
// With stride = 1, 2, 4, ..., 32 words the lanes fall on 32/stride banks at different
// addresses, a `stride`-way conflict: `conflict` wavefronts per access. The model charges
// each wavefront `smem_wavefront_clk` clocks, so
//   smem_wavefront_clk = d(ns per access) / d(conflict) * clock_ghz        [load_chain]
// (least-squares slope over conflict = 1..32).
//
// Modes:
//   load_chain  v = s[lane*stride + v]   each load's address depends on the previous load,
//                                         so this measures latency, which is what the
//                                         model charges (a warp waits for its access)
//   load_indep  v += s[lane*stride + r]   independent loads: throughput. If this is much
//                                         cheaper than load_chain, loads overlap on
//                                         hardware and the model's serial charge is wrong
//   store       s[lane*stride + i%2] = i
// warps=4 repeats two points with four warps in the CTA: the model has ONE shared-memory
// server per SM, so it predicts four warps take four times as long per access.
//
// Method: `reps` accesses per thread in one loop. Valid on Ampere.
#include "common.h"

#define SMEM_WORDS 1056  // 31 * 32 words is the highest index, plus slack

__global__ void smem_kernel(uint64_t* out, int reps, int stride, int mode) {
    __shared__ uint32_t s[SMEM_WORDS];
    for (int i = threadIdx.x; i < SMEM_WORDS; i += blockDim.x) s[i] = 0;
    __syncthreads();
    // Loads and stores are written in PTX so the compiler cannot hoist, merge or drop them.
    // The independent ones move between two adjacent words (all lanes shift by one
    // bank together, so the conflict degree is unchanged): with a fixed or alternating
    // address the assembler removed the repeated loads on the 3080 Ti.
    uint64_t p = (uint64_t)__cvta_generic_to_shared(s + (threadIdx.x & 31) * stride);
    uint32_t v = 0, w;
    uint64_t t0 = BENCH_NOW();
    if (mode == 0) {
        for (int i = 0; i < reps; ++i)
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"(p + v) : "memory");
    } else if (mode == 1) {
        // The word offset (0 or 1) comes from a generator the assembler cannot predict,
        // the same in every lane; otherwise it hoists the loads out of the loop.
        uint32_t x = 12345u + reps;
        for (int i = 0; i < reps; ++i) {
            x = x * 1664525u + 1013904223u;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w) : "l"(p + ((x >> 14) & 4)) : "memory");
            v += w;
        }
    } else {
        for (int i = 0; i < reps; ++i)
            asm volatile("st.shared.b32 [%0], %1;" :: "l"(p + 4 * (i & 1)), "r"(i) : "memory");
    }
    uint64_t t1 = BENCH_NOW();
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = v; }
    else if (v == 0x12345u && reps < 0) out[2] = v;
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "smem_wavefronts");
    const char* modes[] = {"load_chain", "load_indep", "store"};
    for (int mode = 0; mode < 3; ++mode)
        for (int warps : {1, 4})
            for (int conflict : {1, 2, 4, 8, 16, 32}) {
                if (warps == 4 && (mode != 0 || (conflict != 1 && conflict != 32))) continue;
                std::string id = std::string("smem_wavefronts/") + modes[mode] + "/conflict=" +
                                 std::to_string(conflict) + "/warps=" + std::to_string(warps);
                measure(o, id, "", {2000000, 4000000}, {32, 64}, [&](long reps) {
                    smem_kernel<<<1, 32 * warps>>>(o.d_out, (int)reps, conflict, mode);
                });
            }
    return 0;
}
