// A warp-specialized, pipelined GEMM for Blackwell:  D = A * B^T
//
//   A is (M x K) and B is (N x K), half precision; D is (M x N), float32.
//   One CTA computes one 128 x 64 tile of D, walking K in tiles of 64.
//
// Three roles share the CTA:
//
//   producer  (thread 0, in warp 0)   TMA-loads the A and B tiles for step k
//                                     into shared-memory stage k % stages
//   consumer  (thread 32, in warp 1)  issues tcgen05.mma on each stage once it
//                                     is full, accumulating into tensor memory
//   epilogue  (all four warps)        at the end, moves the accumulators from
//                                     tensor memory to registers to global memory
//
// Each stage has two mbarriers. `full[s]` completes when TMA has delivered the
// stage's bytes; the consumer waits on it. `empty[s]` completes when the MMAs
// that read the stage are done (tcgen05.commit); the producer waits on it
// before loading into the stage again. With one stage, loading and multiplying
// take turns. With more stages the producer runs ahead, and the loads of the
// next tiles overlap the multiply of the current one.

#include <stdint.h>

#define TILE_M 128
#define TILE_N 64
#define TILE_K 64                                   // elements; 128 bytes per row
#define A_BYTES (TILE_M * TILE_K * 2)               // 16384
#define B_BYTES (TILE_N * TILE_K * 2)               //  8192
#define STAGE_BYTES (A_BYTES + B_BYTES)             // 24576, a multiple of 1024
#define MAX_STAGES 4

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(arrivals));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase_parity) {
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra DONE;\n"
                 "bra LAB_WAIT;\n"
                 "DONE:\n"
                 "}" :: "r"(smem_addr(bar)), "r"(phase_parity) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem, const void* tensor_map, uint64_t* bar,
                                            int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(tensor_map), "r"(smem_addr(bar)), "r"(c0), "r"(c1)
                 : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* where, uint32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr(where)), "r"(columns) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(address), "r"(columns)
                 : "memory");
}
// D (+)= A * B^T for one 128 x 64 x 16 block. `accumulate` = 0 overwrites D.
__device__ __forceinline__ void mma_f16(uint32_t tmem, uint64_t desc_a, uint64_t desc_b,
                                        uint32_t idesc, uint32_t accumulate) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %5, %5, %5}, p;\n"
                 "}"
                 :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate), "r"(0u)
                 : "memory");
}
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ uint32_t tmem_load(uint32_t address) {
    uint32_t value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(value) : "r"(address));
    return value;
}

// Shared-memory matrix descriptor for a K-major operand with 128-byte swizzle:
// start address / 16, stride between 8-row groups (1024 bytes) / 16, version 1, layout 2.
__device__ __forceinline__ uint64_t smem_desc(uint32_t start) {
    return (uint64_t)(start >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46)
           | (2ull << 61);
}

// grid = (M / 128, N / 64), block = 128 threads.
extern "C" __global__ void pipelined_gemm(const void* map_a, const void* map_b, float* d,
                                          int n_total, int k_tiles, int stages) {
    __shared__ alignas(1024) char tiles[MAX_STAGES * STAGE_BYTES];
    __shared__ uint64_t full[MAX_STAGES], empty[MAX_STAGES], done;
    __shared__ uint32_t tmem_base;

    int tid = threadIdx.x, warp = tid / 32;
    int row0 = blockIdx.x * TILE_M, col0 = blockIdx.y * TILE_N;

    if (tid == 0) {
        for (int s = 0; s < stages; ++s) {
            mbarrier_init(&full[s], 1);
            mbarrier_init(&empty[s], 1);
        }
        mbarrier_init(&done, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) tmem_alloc(&tmem_base, TILE_N);
    __syncthreads();
    uint32_t tmem = tmem_base;

    if (tid == 0) {  // producer
        for (int k = 0; k < k_tiles; ++k) {
            int s = k % stages, use = k / stages;
            if (use > 0) mbarrier_wait(&empty[s], (use - 1) & 1);  // the stage's last reader is done
            mbarrier_arrive_expect_tx(&full[s], STAGE_BYTES);
            char* stage = tiles + s * STAGE_BYTES;
            tma_load_2d(stage, map_a, &full[s], k * TILE_K, row0);
            tma_load_2d(stage + A_BYTES, map_b, &full[s], k * TILE_K, col0);
        }
    } else if (tid == 32) {  // consumer
        uint32_t idesc = (1u << 4) | ((TILE_N >> 3) << 17) | ((TILE_M >> 4) << 24);
        for (int k = 0; k < k_tiles; ++k) {
            int s = k % stages, use = k / stages;
            mbarrier_wait(&full[s], use & 1);  // TMA has filled the stage
            uint32_t a = smem_addr(tiles + s * STAGE_BYTES);
            for (int j = 0; j < TILE_K / 16; ++j)  // 16 elements of K (32 bytes) per MMA
                mma_f16(tmem, smem_desc(a + 32 * j), smem_desc(a + A_BYTES + 32 * j), idesc, k | j);
            mma_commit(&empty[s]);  // arrives once these MMAs no longer need the stage
        }
        mma_commit(&done);
    }

    mbarrier_wait(&done, 0);  // every warp: the accumulators are final
    // Epilogue: thread (warp, lane) owns tensor-memory lane 32 * warp + lane = result row tid.
    float* out = d + (size_t)(row0 + tid) * n_total + col0;
    uint32_t address = tmem + ((uint32_t)(warp * 32) << 16);
    for (int n = 0; n < TILE_N; ++n) out[n] = __uint_as_float(tmem_load(address + n));

    __syncthreads();
    if (warp == 0) tmem_dealloc(tmem, TILE_N);
}
