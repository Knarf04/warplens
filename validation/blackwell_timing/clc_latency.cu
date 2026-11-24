// clc_latency: how long the launch takes to answer a clusterlaunchcontrol.try_cancel
// (validation matrix TIM-20).
//
// Calibrates: no parameter yet. The model lands the response at the time the request is
// issued (warplens/timing/model.py, kind "clc": "not known, so it is not modeled"), so the
// prediction is the loop's instructions and nothing else. The measured round trip minus
// that is the latency a config entry would hold.
//
// Method: thread 0 arms its mbarrier (arrive.expect_tx 16), issues try_cancel and waits for
// the phase, over and over. Nothing decodes a response in the failing variants, so no
// thread ever "observes" a failure (after which PTX leaves a further request undefined).
//
//   clc_latency/fail/plain/ctas=1       the launch is one cluster of one CTA: nothing can be
//                                       canceled and every request fails. reps = requests
//   clc_latency/fail/plain/ctas=2       the same in a cluster of two CTAs, with a cluster
//                                       barrier per request: the baseline for the next one
//   clc_latency/fail/multicast/ctas=2   rank 0 requests with .multicast::cluster::all; both
//                                       CTAs arm and wait on their own mbarrier
//   clc_latency/succeed/plain/ctas=1    a grid of reps + 1 one-CTA clusters; a cluster that
//                                       starts requests until a request fails
//   clc_latency/succeed/multicast/ctas=2  the same with clusters of two CTAs and multicast
//
// Reading the succeed variants. In the simulator the first cluster takes over all `reps`
// others, so the slope of time against reps is the round trip of one successful request.
// On a GPU many clusters start at once (each CTA asks for 100 KB of shared memory it never
// touches, so an SM holds two) and each takes over some: the launch time is NOT reps round
// trips. Use the kernel's own numbers instead: `clk` is the %clock64 ticks the cluster at
// (0, 0) spent in its request loop and `check` the number of requests it made that
// succeeded; it made check + 1 requests in clk ticks. fit.py's ns-per-repetition for these
// two ids is therefore meaningful for the prediction only.
//
// Blackwell only (simulator and sm_100a here).
#include "blackwell.h"

#define OCCUPANCY_BYTES (100 * 1024)

__device__ __forceinline__ uint32_t ctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ void try_cancel(const void* response, uint64_t* bar, int multicast) {
    if (multicast)
        asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"
                     ".multicast::cluster::all.b128 [%0], [%1];"
                     :: "r"(smem_addr(response)), "r"(smem_addr(bar)) : "memory");
    else
        asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
                     " [%0], [%1];" :: "r"(smem_addr(response)), "r"(smem_addr(bar)) : "memory");
}
// 1 if the response says a cluster was canceled; its x coordinate is read as kernels do.
__device__ __forceinline__ uint32_t canceled(const void* response, uint32_t& x) {
    uint32_t c, cx = 0;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 ".reg .b128 r;\n"
                 "ld.shared.b128 r, [%2];\n"
                 "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, r;\n"
                 "}" : "=r"(c), "+r"(cx) : "r"(smem_addr(response)) : "memory");
    x = cx;
    return c;
}

// One cluster in the launch: `reps` requests, all failing. block = 32 threads.
// sync: a cluster barrier per request (needed with multicast, so that every CTA has armed
// its mbarrier before the response can land).
__global__ void clc_fail_kernel(uint64_t* out, int reps, int multicast, int sync) {
    __shared__ alignas(16) uint32_t response[4];
    __shared__ uint64_t bar;
    bool lead = threadIdx.x == 0;
    uint32_t rank = ctarank();
    if (lead) mbarrier_init(&bar, 1);
    cluster_sync();
    uint64_t t0 = BENCH_NOW();
    for (int i = 0; i < reps; ++i) {
        bool waits = lead && (rank == 0 || multicast);
        if (waits) mbarrier_arrive_expect_tx(&bar, 16);
        if (sync) cluster_sync();
        if (lead && rank == 0) try_cancel(response, &bar, multicast);
        if (waits) mbarrier_wait(&bar, i & 1);
    }
    uint64_t t1 = BENCH_NOW();
    cluster_sync();
    if (lead && blockIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = reps; }
}

// A grid of clusters: every cluster that starts requests until a request fails.
__global__ void clc_succeed_kernel(uint64_t* out, int multicast) {
    __shared__ alignas(16) uint32_t response[4];
    __shared__ uint64_t bar;
    __shared__ uint32_t go_on;
    bool lead = threadIdx.x == 0;
    uint32_t rank = ctarank();
    if (lead) mbarrier_init(&bar, 1);
    cluster_sync();
    uint64_t t0 = BENCH_NOW();
    uint64_t taken = 0;
    for (uint32_t phase = 0;; phase ^= 1) {
        bool waits = lead && (rank == 0 || multicast);
        if (waits) mbarrier_arrive_expect_tx(&bar, 16);
        if (multicast) cluster_sync();
        if (lead && rank == 0) try_cancel(response, &bar, multicast);
        if (waits) {
            mbarrier_wait(&bar, phase);
            uint32_t x;
            go_on = canceled(response, x);
            fence_proxy_async();
        }
        // Without multicast only rank 0 has the response; those clusters are one CTA here.
        __syncthreads();
        if (!go_on) break;
        ++taken;
    }
    uint64_t t1 = BENCH_NOW();
    if (lead && blockIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = taken; }
}

template <class... Args>
static void launch_clusters(void (*kernel)(Args...), int clusters, int cluster_size, size_t smem, Args... args) {
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof config);
    config.gridDim = dim3(clusters * cluster_size, 1, 1);
    config.blockDim = dim3(32, 1, 1);
    config.dynamicSmemBytes = smem;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = cluster_size;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel, args...);  // timed_launch checks cudaGetLastError
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "clc_latency");
    CHECK(cudaFuncSetAttribute(clc_succeed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, OCCUPANCY_BYTES));
    struct Fail { const char* id; int ctas, multicast, sync; } fails[] = {
        {"clc_latency/fail/plain/ctas=1", 1, 0, 0},
        {"clc_latency/fail/plain/ctas=2", 2, 0, 1},
        {"clc_latency/fail/multicast/ctas=2", 2, 1, 1},
    };
    for (auto& f : fails)
        measure(o, f.id, "nctas=" + std::to_string(f.ctas), {10000, 20000}, {16, 32}, [&](long reps) {
            launch_clusters(clc_fail_kernel, 1, f.ctas, (size_t)0, o.d_out, (int)reps, f.multicast, f.sync);
        });
    struct Succeed { const char* id; int ctas, multicast; } succeeds[] = {
        {"clc_latency/succeed/plain/ctas=1", 1, 0},
        {"clc_latency/succeed/multicast/ctas=2", 2, 1},
    };
    for (auto& s : succeeds)
        measure(o, s.id, "nctas=" + std::to_string(s.ctas), {10000, 20000}, {16, 32}, [&](long reps) {
            launch_clusters(clc_succeed_kernel, (int)reps + 1, s.ctas, (size_t)OCCUPANCY_BYTES, o.d_out,
                            s.multicast);
        });
    return 0;
}
