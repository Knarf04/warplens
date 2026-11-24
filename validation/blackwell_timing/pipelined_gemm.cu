// pipelined_gemm: the warp-specialized GEMM of examples/04_pipelined_gemm, timed per K tile
// with 1, 2 and 4 stages (validation matrix TIM-18).
//
// Calibrates: nothing. It is the end-to-end check of the parameters the other benchmarks
// calibrate: TMA latency and rate, HBM and L2 bandwidth, MMA rate and latency together decide
// what one K tile costs and which resource limits it. The one thing a calibrated model must
// keep is the ordering: 2 stages no slower than 1, 4 no slower than 2.
//
// The kernel is examples/04_pipelined_gemm/gemm.cu, included unchanged (in a namespace: its
// helpers have the names of blackwell.h's). One CTA computes a 128 x 64 tile of D = A * B^T
// and walks K in tiles of 64; `reps` is the number of K tiles, so the slope of launch time
// against reps is the time per K tile. The kernel has no %clock64 reads, so `clk` is 0 and
// only the event time is measured.
//
//   pipelined_gemm/stages=S/ctas=8       8 CTAs, each with its own 128 rows of A
//   pipelined_gemm/stages=S/ctas=allsm   one CTA per SM
// Each CTA reads its own rows of A (16 KiB per K tile) and all share B (8 KiB per K tile),
// from memory far larger than L2 on hardware. A and B hold the half-precision value
// 0x3C3C (about 1.0586) everywhere; after each configuration D[0][0] is checked against
// K * 0x3C3C^2 and a wrong result is reported as a BENCH_ERROR line.
//
// Prediction caveat: `warplens run --timing` always assumes all 148 SMs share HBM, so the
// simulator's ctas=8 numbers are the "whole device busy" row of docs/m4.md, not its "8 SMs
// share HBM" row; that needs `active_sms=8`, which only the Python API has
// (examples/04_pipelined_gemm/run.py).
// Not verified on a B200: the kernel declares 98 KiB of STATIC shared memory and never calls
// cudaFuncSetAttribute. ptxas accepts that for sm_100a (up to 227 KiB; 48 KiB for sm_100), and
// an RTX 3080 Ti launches static shared memory above its 48 KiB default without opt-in
// (validation/README.md, "The shared-memory opt-in"), which is the rule the simulator follows.
// Whether a B200 does is RT-13. A refused launch shows as a BENCH_ERROR line.
//
// Blackwell only (simulator and sm_100a here).
#include <cmath>
#include <cuda.h>
#include "common.h"

namespace gemm {
#include "../../examples/04_pipelined_gemm/gemm.cu"
}

// A rank-2 map of half-precision elements over `base`: dims (width, rows), box (64, box_rows),
// 128-byte swizzle, as examples/04_pipelined_gemm/run.py builds them. The driver function is
// fetched by name, so the program does not link libcuda and `warplens run` can stand in.
static CUtensorMap* make_half_map(void* base, uint64_t width, uint64_t rows, uint32_t box_rows) {
    typedef CUresult (*Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                               const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                               CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                               CUtensorMapFloatOOBfill);
    static Encode encode = nullptr;
    if (!encode) {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult status;
        CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
        if (!fn || status != cudaDriverEntryPointSuccess) {
            printf("BENCH_ERROR where=cuTensorMapEncodeTiled error=entry_point_not_found\n");
            exit(1);
        }
        encode = (Encode)fn;
    }
    CUtensorMap map;
    cuuint64_t dims[2] = {width, rows}, strides[1] = {width * 2};
    cuuint32_t box[2] = {TILE_K, box_rows}, steps[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, base, dims, strides, box, steps,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        printf("BENCH_ERROR where=cuTensorMapEncodeTiled error=%d\n", (int)r);
        exit(1);
    }
    // The kernel takes the maps by pointer, so they live in device memory.
    CUtensorMap* device;
    CHECK(cudaMalloc(&device, sizeof map));
    CHECK(cudaMemcpy(device, &map, sizeof map, cudaMemcpyHostToDevice));
    return device;
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "pipelined_gemm");
    for (int allsm = 0; allsm < 2; ++allsm) {
        int ctas = allsm ? o.sm_count : 8;
        if (ctas > o.sm_count) continue;
        // K tiles per launch. Hardware: shrink until A (the larger operand) fits a quarter of
        // free memory.
        long hw = 512;
        size_t rows_a = (size_t)ctas * TILE_M;
        while (!o.sim && hw > 8 && rows_a * (size_t)(2 * scaled(o, hw)) * TILE_K * 2 > o.free_mem / 4) hw /= 2;
        std::vector<long> reps_hw = {hw, 2 * hw}, reps_sim = {8, 16};
        long most = o.sim ? reps_sim[1] : scaled(o, reps_hw[1]);
        size_t width = (size_t)most * TILE_K;  // elements of K
        uint16_t *a, *b;
        float* d;
        CHECK(cudaMalloc(&a, rows_a * width * 2));
        CHECK(cudaMalloc(&b, (size_t)TILE_N * width * 2));
        CHECK(cudaMalloc(&d, rows_a * TILE_N * sizeof(float)));
        CHECK(cudaMemset(a, 0x3C, rows_a * width * 2));
        CHECK(cudaMemset(b, 0x3C, (size_t)TILE_N * width * 2));
        CUtensorMap* map_a = make_half_map(a, width, rows_a, TILE_M);
        CUtensorMap* map_b = make_half_map(b, width, TILE_N, TILE_N);
        for (int stages : {1, 2, 4}) {
            std::string id = "pipelined_gemm/stages=" + std::to_string(stages) + "/ctas=" +
                             (allsm ? "allsm" : std::to_string(ctas));
            if (!selected(o, id)) continue;
            std::string extra = "m=" + std::to_string(TILE_M) + " n=" + std::to_string(TILE_N) + " k=" +
                                std::to_string(TILE_K) + " nctas=" + std::to_string(ctas);
            long last = 0;
            CHECK(cudaMemset(d, 0, rows_a * TILE_N * sizeof(float)));
            measure(o, id, extra, reps_hw, reps_sim, [&](long reps) {
                last = reps;
                gemm::pipelined_gemm<<<dim3(ctas, 1), 128>>>(map_a, map_b, d, TILE_N, (int)reps, stages);
            });
            float d00 = 0, d_last = 0;
            CHECK(cudaMemcpy(&d00, d, sizeof d00, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&d_last, d + rows_a * TILE_N - 1, sizeof d_last, cudaMemcpyDeviceToHost));
            const float one = 1.05859375f;  // half precision 0x3C3C
            float expected = (float)(last * TILE_K) * one * one;
            if (std::fabs(d00 - expected) > 1e-3f * expected || std::fabs(d_last - expected) > 1e-3f * expected) {
                printf("BENCH_ERROR where=\"%s\" error=wrong_result d00=%g d_last=%g expected=%g\n", id.c_str(), d00,
                       d_last, expected);
                fflush(stdout);
            }
        }
        CHECK(cudaFree(a));
        CHECK(cudaFree(b));
        CHECK(cudaFree(d));
        CHECK(cudaFree(map_a));
        CHECK(cudaFree(map_b));
    }
    return 0;
}
