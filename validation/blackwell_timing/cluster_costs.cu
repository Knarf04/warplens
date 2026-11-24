// cluster_costs: what a multicast TMA load and a two-CTA MMA cost, against the same work
// done without the cluster feature (validation matrix TIM-15).
//
// Calibrates: no parameter. The model charges a multicast load to each receiving SM as a
// copy of its own (warplens/units/tma.py:_load) and a two-CTA MMA to the issuing SM only
// (warplens/core/cta.py:Cta.schedule). Both are assumptions; this measures them.
//
// Loads. A cluster of N CTAs (N = 1, 2, 4), each with a 16 KiB tile. Per repetition every
// CTA arms its own mbarrier (arrive.expect_tx), the cluster passes a cluster barrier (so
// every mbarrier is armed before a copy can land), then
//   cluster_costs/load/multicast/ctas=N   rank 0 issues ONE load with .multicast::cluster
//                                         to all N CTAs
//   cluster_costs/load/separate/ctas=N    every CTA issues its own load of the same box
// and every CTA waits for its phase. The same source box every time, so it is in L2 after
// the first repetition. The model predicts the same time for both.
//
// MMAs. One thread issues `batch` MMAs, commits and waits, as mma_rate does; f16, N = 256.
//   cluster_costs/mma/pair/m=256/n=256/batch=B            a cluster of two CTAs, rank 0
//                                         issues tcgen05.mma.cta_group::2 with M = 256
//   cluster_costs/mma/single/m=128/n=256/batch=B/ctas=2   two CTAs on their own, each
//                                         issuing cta_group::1 MMAs with M = 128
// Both do the same arithmetic per repetition (2 x 128 x 256 x 16 x batch products). The
// model predicts a pair MMA takes the time of ONE M = 256 MMA on one SM's Tensor Core,
// i.e. twice the single-CTA time per repetition, where two SMs' Tensor Cores working in
// parallel would give the same time as the single variant.
//
// Blackwell only (simulator and sm_100a here).
#include "blackwell.h"

#define TILE_BYTES 16384
#define OPERAND_BYTES 32768  // 256 rows of 128 bytes

__device__ __forceinline__ uint32_t ctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t nctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// block = 32 threads; a cluster of `ctas` CTAs.
__global__ void load_kernel(const __grid_constant__ CUtensorMap map, uint64_t* out, int reps, int multicast) {
    __shared__ alignas(128) uint32_t tile[TILE_BYTES / 4];
    __shared__ uint64_t bar;
    for (int i = threadIdx.x; i < TILE_BYTES / 4; i += blockDim.x) tile[i] = 0;
    bool lead = threadIdx.x == 0;
    uint32_t rank = ctarank();
    uint16_t everyone = (uint16_t)((1u << nctarank()) - 1);
    if (lead) mbarrier_init(&bar, 1);
    fence_proxy_async();
    cluster_sync();
    uint64_t t0 = BENCH_NOW();
    for (int i = 0; i < reps; ++i) {
        if (lead) mbarrier_arrive_expect_tx(&bar, TILE_BYTES);
        cluster_sync();
        if (lead && multicast && rank == 0)
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                         ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
                         :: "r"(smem_addr(tile)), "l"(&map), "r"(smem_addr(&bar)), "r"(0), "r"(0), "h"(everyone)
                         : "memory");
        if (lead && !multicast) tma_load_2d(tile, &map, &bar, 0, 0);
        if (lead) mbarrier_wait(&bar, i & 1);
    }
    uint64_t t1 = BENCH_NOW();
    cluster_sync();
    if (lead && blockIdx.x == 0) { out[OUT_CLK] = t1 - t0; out[OUT_CHECK] = tile[0] + reps; }
}

// K-major operand, 128-byte swizzle: start / 16, 1024 bytes between 8-row groups.
__device__ __forceinline__ uint64_t smem_desc(uint32_t start) {
    return (uint64_t)(start >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46)
           | (2ull << 61);
}

// block = 32 threads. One kernel per granularity, because ptxas does not allow .cta_group::1
// and .cta_group::2 in one function:
//   mma_pair_kernel    a cluster of two CTAs; rank 0 issues cta_group::2 MMAs
//   mma_single_kernel  every CTA on its own, issuing cta_group::1 MMAs
#define DEFINE_MMA_KERNEL(name, group, pair)                                                              \
    __global__ void name(uint64_t* out, int reps, int batch, uint32_t idesc) {                            \
        __shared__ alignas(1024) uint32_t operand[OPERAND_BYTES / 4];                                     \
        __shared__ uint64_t bar;                                                                          \
        __shared__ uint32_t tmem_base;                                                                    \
        for (int i = threadIdx.x; i < OPERAND_BYTES / 4; i += blockDim.x) operand[i] = 0x3C003C00u;       \
        fence_proxy_async();                                                                              \
        uint32_t rank = ctarank();                                                                        \
        if (threadIdx.x == 0) mbarrier_init(&bar, 1);                                                     \
        asm volatile("tcgen05.alloc.cta_group::" group ".sync.aligned.shared::cta.b32 [%0], %1;"          \
                     :: "r"(smem_addr(&tmem_base)), "r"(256) : "memory");                                 \
        __syncthreads();                                                                                  \
        cluster_sync(); /* a pair: both CTAs have their operands and their allocation */                  \
        uint32_t tmem = tmem_base;                                                                        \
        if (threadIdx.x == 0 && (!pair || rank == 0)) {                                                   \
            uint64_t desc = smem_desc(smem_addr(operand));                                                \
            uint64_t t0 = BENCH_NOW();                                                                    \
            for (int i = 0; i < reps; ++i) {                                                              \
                for (int j = 0; j < batch; ++j)                                                           \
                    asm volatile("{\n"                                                                    \
                                 ".reg .pred p;\n"                                                        \
                                 "setp.ne.b32 p, %4, 0;\n"                                                \
                                 "tcgen05.mma.cta_group::" group ".kind::f16 [%0], %1, %2, %3, p;\n"      \
                                 "}" :: "r"(tmem), "l"(desc), "l"(desc), "r"(idesc), "r"(j) : "memory");  \
                asm volatile("tcgen05.commit.cta_group::" group                                           \
                             ".mbarrier::arrive::one.shared::cluster.b64 [%0];"                           \
                             :: "r"(smem_addr(&bar)) : "memory");                                         \
                mbarrier_wait(&bar, i & 1);                                                               \
            }                                                                                             \
            uint64_t t1 = BENCH_NOW();                                                                    \
            if (blockIdx.x == 0) out[OUT_CLK] = t1 - t0;                                                  \
        }                                                                                                 \
        __syncthreads();                                                                                  \
        cluster_sync(); /* a pair: rank 1 keeps its operands and tensor memory until rank 0 is done */    \
        uint32_t d00 = tmem_ld1(tmem); /* lane t of warp 0 reads row t, column 0 */                       \
        tmem_wait_ld();                                                                                   \
        if (blockIdx.x == 0 && threadIdx.x == 0) out[OUT_CHECK] = (uint64_t)__uint_as_float(d00);         \
        __syncthreads();                                                                                  \
        asm volatile("tcgen05.dealloc.cta_group::" group ".sync.aligned.b32 %0, %1;"                      \
                     :: "r"(tmem), "r"(256) : "memory");                                                  \
    }
DEFINE_MMA_KERNEL(mma_pair_kernel, "2", 1)
DEFINE_MMA_KERNEL(mma_single_kernel, "1", 0)

template <class... Args>
static void launch_clustered(void (*kernel)(Args...), int ctas, int cluster_size, Args... args) {
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof config);
    config.gridDim = dim3(ctas, 1, 1);
    config.blockDim = dim3(32, 1, 1);
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
    Opts o = bench_init(argc, argv, "cluster_costs");
    char* data;
    CHECK(cudaMalloc(&data, TILE_BYTES));
    CHECK(cudaMemset(data, 1, TILE_BYTES));
    CUtensorMap map = make_byte_map(data, 128, TILE_BYTES / 128, 128, TILE_BYTES / 128);
    for (int multicast = 1; multicast >= 0; --multicast)
        for (int ctas : {1, 2, 4}) {
            std::string id = std::string("cluster_costs/load/") + (multicast ? "multicast" : "separate") +
                             "/ctas=" + std::to_string(ctas);
            std::string extra = "bytes=" + std::to_string(TILE_BYTES) + " nctas=" + std::to_string(ctas);
            measure(o, id, extra, {20000, 40000}, {4, 8}, [&](long reps) {
                launch_clustered(load_kernel, ctas, ctas, map, o.d_out, (int)reps, multicast);
            });
        }
    for (int pair = 1; pair >= 0; --pair)
        for (int batch : {1, 4}) {
            int m = pair ? 256 : 128, n = 256;
            std::string id = std::string("cluster_costs/mma/") + (pair ? "pair" : "single") + "/m=" +
                             std::to_string(m) + "/n=" + std::to_string(n) + "/batch=" + std::to_string(batch) +
                             (pair ? "" : "/ctas=2");
            uint32_t idesc = (1u << 4) | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24);
            std::string extra = "kind=f16 m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=16 batch=" +
                                std::to_string(batch) + " nctas=2";
            measure(o, id, extra, {50000, 100000}, {4, 8}, [&](long reps) {
                if (pair) launch_clustered(mma_pair_kernel, 2, 2, o.d_out, (int)reps, batch, idesc);
                else launch_clustered(mma_single_kernel, 2, 1, o.d_out, (int)reps, batch, idesc);
            });
        }
    return 0;
}
