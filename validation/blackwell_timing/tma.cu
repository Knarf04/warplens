// tma: latency and throughput of TMA loads and stores, against tile size and L2 state.
//
// One thread issues a copy and waits for it, `reps` times:
//   load   mbarrier.arrive.expect_tx; cp.async.bulk.tensor (global -> shared); wait
//   store  cp.async.bulk.tensor (shared -> global); commit_group; wait_group 0
// For each tile size the slope of time against reps is the round-trip time T(bytes) of one
// copy. A straight line through T against bytes gives an intercept and a slope.
//
// Calibrates:
//   `tma_latency_ns`     intercept of T(bytes), warm loads, minus the instruction overhead
//                        the simulator counts for the same loop:
//                          tma_latency_ns = config value + intercept(measured)
//                                           - intercept(predicted) rescaled to the fitted clock
//   `tma_bytes_per_clk`  1 / slope of T(bytes), in bytes/ns, divided by clock_ghz. The model
//                        takes max(TMA engine time, memory time) per copy, so one slope
//                        gives min(engine rate, memory rate). fit.py takes the faster of the
//                        warm-load and the store slope as the engine rate.
//   `l2_bandwidth_per_sm_gb_s`  1 / slope of T(bytes) for WARM loads, bytes/ns = GB/s. Only a
//                        lower bound when it equals the engine rate (see README).
//   `hbm_bandwidth_tb_s` cold loads with one CTA per SM (ctas=allsm), each walking its own
//                        fresh memory: sm_count * bytes / T / 1000. Cross-check of gstream.
//
// Variants (ids):
//   tma/load/warm/bytes=B/ctas=1   the same tile every time: in L2 after the first copy
//   tma/load/cold/bytes=B/ctas=1   a fresh tile every time, from a buffer far larger than L2;
//                                  the host never reuses a region until the whole buffer
//                                  (hardware: up to 16 GiB) has been walked. One SM's cold
//                                  fetch rate. The model gives one SM 1/active_sms of the HBM
//                                  bandwidth; `warplens run` fixes active_sms at sm_count.
//   tma/load/cold/bytes=16384/ctas=8, 37, allsm   that many CTAs, one per SM, each walking
//                                  its own fresh memory (validation matrix TIM-07)
//   tma/store/bytes=B/ctas=1       the same destination tile every time
// Tile sizes: 64 B to 32 KiB (box widths 64, 128, 256 bytes; no swizzle; uint8 elements;
// L2 promotion off).
//
// Blackwell only (TMA needs sm_90 or newer): simulator and sm_100a here.
#include "blackwell.h"

#define TILE_BYTES 32768

__global__ void tma_kernel(const __grid_constant__ CUtensorMap map, uint64_t* out, int reps,
                           uint32_t bytes, int row0, int rows_per_cta, int row_step, int store) {
    __shared__ alignas(128) uint32_t tile[TILE_BYTES / 4];
    __shared__ uint64_t bar;
    for (int i = threadIdx.x; i < TILE_BYTES / 4; i += blockDim.x) tile[i] = 0;
    if (threadIdx.x == 0) mbarrier_init(&bar, 1);
    fence_proxy_async();
    __syncthreads();
    if (threadIdx.x != 0) return;
    int row = row0 + blockIdx.x * rows_per_cta;
    uint64_t t0 = BENCH_NOW();
    if (!store) {
        for (int i = 0; i < reps; ++i) {
            mbarrier_arrive_expect_tx(&bar, bytes);
            tma_load_2d(tile, &map, &bar, 0, row);
            mbarrier_wait(&bar, i & 1);
            row += row_step;
        }
    } else {
        for (int i = 0; i < reps; ++i) {
            tma_store_2d(&map, tile, 0, row);
            tma_store_commit();
            tma_store_wait();
            row += row_step;
        }
    }
    uint64_t t1 = BENCH_NOW();
    if (blockIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = tile[0] + reps; }
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "tma");
    // The buffer cold loads walk through. Hardware: 16 GiB or a quarter of free memory.
    size_t buffer = o.sim ? (size_t)32 << 20 : (size_t)16 << 30;
    while (!o.sim && buffer > o.free_mem / 4) buffer >>= 1;
    char* data;
    CHECK(cudaMalloc(&data, buffer));
    if (!o.sim) CHECK(cudaMemset(data, 1, buffer));  // the simulator's memory starts zeroed
    size_t cursor = 0;  // next unused byte for cold loads

    struct Box { uint32_t w, h; } boxes[] = {{64, 1}, {64, 4}, {64, 16}, {64, 64}, {128, 128}, {256, 128}};
    struct Variant { const char* name; int store, cold, ctas; long hw_reps; };  // ctas 0: one per SM
    Variant variants[] = {
        {"load/warm", 0, 0, 1, 20000},
        {"store", 1, 0, 1, 20000},
        {"load/cold", 0, 1, 1, 2048},
        {"load/cold", 0, 1, 8, 2048},
        {"load/cold", 0, 1, 37, 2048},
        {"load/cold", 0, 1, 0, 1024},
    };
    for (auto& v : variants)
        for (auto& box : boxes) {
            uint32_t bytes = box.w * box.h;
            if (v.ctas != 1 && bytes != 16384) continue;
            if (v.ctas > o.sm_count) continue;
            int ctas = v.ctas ? v.ctas : o.sm_count;
            std::string ctas_name = v.ctas ? std::to_string(v.ctas) : "allsm";
            std::string id = std::string("tma/") + v.name + "/bytes=" + std::to_string(bytes);
            id += "/ctas=" + ctas_name;
            CUtensorMap map = make_byte_map(data, box.w, buffer / box.w, box.w, box.h);
            // A cold launch needs ctas * reps fresh tiles; shrink reps to fit half the buffer.
            long hw_reps = v.hw_reps;
            while (v.cold && (size_t)ctas * 2 * scaled(o, hw_reps) * bytes > buffer / 2) hw_reps /= 2;
            if (hw_reps < 1) hw_reps = 1;
            std::string extra = "bytes=" + std::to_string(bytes) + " nctas=" + std::to_string(ctas) +
                                " buffer_bytes=" + std::to_string(buffer);
            measure(o, id, extra, {hw_reps, 2 * hw_reps}, v.ctas != 1 ? std::vector<long>{2, 4} : std::vector<long>{4, 8},
                    [&](long reps) {
                        int row0 = 0, rows_per_cta = 0;
                        if (v.cold) {
                            size_t need = (size_t)ctas * reps * bytes;
                            if (cursor + need > buffer) cursor = 0;
                            row0 = (int)(cursor / box.w);
                            rows_per_cta = (int)(reps * box.h);
                            cursor += need;
                        }
                        tma_kernel<<<ctas, 32>>>(map, o.d_out, (int)reps, bytes, row0, rows_per_cta,
                                                 v.cold ? box.h : 0, v.store);
                    });
        }
    return 0;
}
