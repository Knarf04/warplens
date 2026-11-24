// global_load: cost of an ordinary (non-TMA) global load and store.
//
// Calibrates: `global_access_latency_clk`, the fixed time a warp waits for a global load:
//   global_access_latency_clk = config value
//        + measured ns/rep * clock_ghz - predicted ns/rep * clock_ghz(config)   [load_chain]
// i.e. the value that makes the model reproduce the measured time of this loop, whose
// other instructions the simulator counted. fit.py uses the `l2` footprint for the config
// entry and reports the other two; the model has one number for all three.
//
// Modes:
//   load_chain  v = data[i + v]; i = next   each address depends on the previous load
//   store       data[i] = 0;     i = next   the model charges a store nothing beyond its
//                                           instruction ("stores are not waited for")
// Footprints: `l1` 16 KiB (stays in the SM's L1), `l2` 2 MiB (stays in L2), `dram` 1 GiB
// (mostly misses). In the simulator the footprints are smaller (16/64/256 KiB): the model
// ignores where the data is, so the prediction is the same for all three.
//
// Method: one warp; each lane walks its own sequence of 64-byte lines through the buffer
// with a large odd stride, `reps` accesses. Valid on Ampere.
#include "common.h"

#define STEP_WORDS (40503u * 16u)

__global__ void gmem_kernel(uint64_t* out, uint32_t* data, uint32_t mask, int reps, int mode) {
    uint32_t i = (threadIdx.x * 1031u * 16u) & mask;
    uint32_t v = 0, zero = reps < 0;
    uint64_t t0 = BENCH_NOW();
    if (mode == 0) {
        for (int r = 0; r < reps; ++r) { v = data[i + v]; i = (i + STEP_WORDS) & mask; }
    } else {
        for (int r = 0; r < reps; ++r) { data[i] = zero; i = (i + STEP_WORDS) & mask; }
    }
    uint64_t t1 = BENCH_NOW();
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = v + i; }
    else if (v == 0x12345u && reps < 0) out[2] = v;
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "global_load");
    struct { const char* name; size_t hw_bytes, sim_bytes; long hw_reps; } footprints[] = {
        {"l1", 16u << 10, 16u << 10, 2000000},
        {"l2", 2u << 20, 64u << 10, 1000000},
        {"dram", 1u << 30, 256u << 10, 100000},
    };
    const char* modes[] = {"load_chain", "store"};
    for (auto& f : footprints) {
        size_t bytes = o.sim ? f.sim_bytes : f.hw_bytes;
        while (!o.sim && bytes > o.free_mem / 2) bytes >>= 1;
        uint32_t* data;
        CHECK(cudaMalloc(&data, bytes));
        CHECK(cudaMemset(data, 0, bytes));
        for (int mode = 0; mode < 2; ++mode) {
            std::string id = std::string("global_load/") + modes[mode] + "/footprint=" + f.name;
            measure(o, id, "footprint_bytes=" + std::to_string(bytes), {f.hw_reps, 2 * f.hw_reps},
                    {32, 64}, [&](long reps) {
                        gmem_kernel<<<1, 32>>>(o.d_out, data, (uint32_t)(bytes / 4 - 1), (int)reps, mode);
                    });
        }
        CHECK(cudaFree(data));
    }
    return 0;
}
