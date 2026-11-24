// alu_chain: how fast one warp issues register-only instructions, and how warps share an SM.
//
// Calibrates: `clock_ghz`. The model issues one PTX instruction per clock per warp, so
//   clock_ghz = (PTX instructions per repetition) / (measured ns per repetition)
//             = clock_ghz(config) * predicted ns/rep / measured ns/rep       [warps=1]
// The instruction count comes from the simulator run of this same program. This is an
// EFFECTIVE clock: the frequency that makes the model's issue rate right for this loop.
// It is not a measurement of the core clock; SASS need not map 1:1 to PTX.
// Second quantity: %clock64 ticks per ns (clk slope / event-time slope), reported as is.
//
// Structure check (no parameter): time against the number of warps in one CTA. The model
// has 4 issue slots per SM (warp w uses slot w % 4), so it predicts flat time up to 4
// warps and time proportional to warps/4 beyond. The 4 is hard-coded, not a config entry.
//
// Method: every thread runs a dependent shift-xor-multiply-add chain `reps` times. Valid on Ampere.
#include "common.h"

__global__ void issue_kernel(uint64_t* out, int reps) {
    uint32_t x = threadIdx.x + 1;
    uint64_t t0 = BENCH_NOW();
    for (int i = 0; i < reps; ++i) x = (x ^ (x >> 13)) * 1664525u + 1013904223u;
    uint64_t t1 = BENCH_NOW();
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = x; }
    else if (x == 0x12345u && reps < 0) out[2] = x;  // keeps the chain live in every thread
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "alu_chain");
    for (int warps : {1, 2, 4, 5, 8, 16, 32}) {
        std::string id = "alu_chain/warps=" + std::to_string(warps);
        measure(o, id, "", {4000000, 8000000}, {64, 128},
                [&](long reps) { issue_kernel<<<1, 32 * warps>>>(o.d_out, (int)reps); });
    }
    return 0;
}
