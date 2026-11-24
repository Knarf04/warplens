// cp_async: round trip of a group of per-thread cp.async copies (validation matrix TIM-14).
//
// Calibrates: no parameter of its own. The model charges cp.async against the TMA engine's
// rate and latency (`tma_bytes_per_clk`, `tma_latency_ns`) and, having no line addresses
// for it, always against HBM rather than L2. This benchmark shows how far that borrowed
// charge is from the separate hardware path; fit.py reports measured against predicted.
//
// Method: every lane of one warp issues one 16-byte cp.async (512 bytes per group), then
// cp.async.commit_group and cp.async.wait_group 0; `reps` groups.
//   cp_async/warm  the same 512 source bytes every time
//   cp_async/cold  the source advances 512 bytes per group through a large buffer
//                  (hardware 1 GiB, wrapping; simulator 256 KiB)
// Valid on Ampere.
#include "common.h"

__global__ void cp_async_kernel(uint64_t* out, const char* src, uint32_t mask, int reps, uint32_t step) {
    __shared__ alignas(16) char buf[512];
    uint32_t lane = threadIdx.x & 31;
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(buf + lane * 16);
    uint32_t off = lane * 16;
    uint64_t t0 = BENCH_NOW();
    for (int i = 0; i < reps; ++i) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src + off) : "memory");
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        off = (off + step) & mask;
    }
    uint64_t t1 = BENCH_NOW();
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = buf[0] + reps; }
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "cp_async");
    size_t bytes = o.sim ? (size_t)256 << 10 : (size_t)1 << 30;
    while (!o.sim && bytes > o.free_mem / 2) bytes >>= 1;
    char* src;
    CHECK(cudaMalloc(&src, bytes));
    CHECK(cudaMemset(src, 1, bytes));
    struct { const char* name; uint32_t step; long hw_reps; } variants[] = {{"warm", 0, 200000}, {"cold", 512, 100000}};
    for (auto& v : variants)
        measure(o, std::string("cp_async/") + v.name, "buffer_bytes=" + std::to_string(bytes),
                {v.hw_reps, 2 * v.hw_reps}, {8, 16}, [&](long reps) {
                    // mask keeps a multiple of 16 below the buffer size
                    cp_async_kernel<<<1, 32>>>(o.d_out, src, (uint32_t)(bytes - 1) & ~15u, (int)reps, v.step);
                });
    return 0;
}
