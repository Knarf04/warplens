// mbar: cost of an mbarrier arrive + wait round trip with nothing asynchronous behind it.
//
// Calibrates: no model parameter. The model charges mbarrier instructions one clock each,
// like any instruction, and a wait that is already satisfied costs nothing more. This is
// the overhead floor contained in the intercepts of the tma and mma benchmarks: if it is
// far above the prediction, the fitted `tma_latency_ns` and `mma_latency_clk` absorb
// mbarrier cost that the model has no separate place for.
//
// Method: the barrier expects `threads` arrivals. Each repetition, every participating
// thread arrives (completing the phase) and waits for the phase it just completed, with
// the usual try_wait loop. threads=1: thread 0 alone; threads=32: the whole warp.
// Blackwell (needs sm_90 or newer); simulator and sm_100a only here.
#include "blackwell.h"

__global__ void mbar_kernel(uint64_t* out, int reps, int threads) {
    __shared__ uint64_t bar;
    if (threadIdx.x == 0) mbarrier_init(&bar, threads);
    __syncthreads();
    if ((int)threadIdx.x >= threads) return;
    uint64_t t0 = BENCH_NOW();
    for (int i = 0; i < reps; ++i) {
        mbarrier_arrive(&bar);
        mbarrier_wait(&bar, i & 1);
    }
    uint64_t t1 = BENCH_NOW();
    if (threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = reps; }
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "mbar");
    for (int threads : {1, 32}) {
        std::string id = "mbar/arrive_wait/threads=" + std::to_string(threads);
        measure(o, id, "", {1000000, 2000000}, {16, 32},
                [&](long reps) { mbar_kernel<<<1, 32>>>(o.d_out, (int)reps, threads); });
    }
    return 0;
}
