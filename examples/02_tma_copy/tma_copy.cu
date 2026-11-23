// Copy a 2D tensor tile by tile through shared memory with TMA, doubling it on the way:
//
//   GMEM --TMA load--> SMEM (swizzled) --threads: x2--> SMEM --TMA store--> GMEM
//
// Each CTA handles one tile. The steps, and what each one is waiting for:
//
//   1. thread 0 initializes an mbarrier that expects one arrival
//   2. thread 0 arrives on it, tells it how many bytes TMA will deliver
//      (expect_tx), and issues the asynchronous TMA load
//   3. every thread waits for the mbarrier's first phase to complete: that
//      happens once the arrival is in and all expected bytes have landed
//   4. threads double the tile in place (an elementwise op, so the swizzled
//      order of the bytes in SMEM does not matter)
//   5. fence.proxy.async makes those ordinary writes visible to TMA
//   6. thread 0 issues the TMA store, commits it as a group, and waits until
//      TMA has finished reading the tile
//
// `bug` deliberately breaks one step, to show what warplens reports.

#include <stdint.h>

enum Bug {
    NONE = 0,
    EXPECT_TOO_FEW = 1,   // step 2 announces half the bytes: the phase completes early
    EXPECT_TOO_MANY = 2,  // step 2 announces twice the bytes: the phase never completes
    NO_WAIT = 3,          // step 3 skipped: threads read the tile while TMA is writing it
    NO_FENCE = 4,         // step 5 skipped: TMA may not see the doubled values
    EARLY_REUSE = 5,      // step 6: the tile is overwritten before TMA has read it
};

#define MAX_TILE_BYTES 8192

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(arrivals));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
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

__device__ __forceinline__ void tma_store_2d(const void* tensor_map, const void* smem, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"(tensor_map), "r"(smem_addr(smem)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// box_w x box_h float32 elements per tile; the tensor maps must use the same box.
extern "C" __global__ void tma_copy(const void* src_map, const void* dst_map,
                                    int box_w, int box_h, int bug) {
    __shared__ alignas(1024) float tile[MAX_TILE_BYTES / sizeof(float)];
    __shared__ uint64_t bar;

    int t = threadIdx.x;
    int elems = box_w * box_h;
    uint32_t bytes = elems * sizeof(float);
    int c0 = blockIdx.x * box_w;  // TMA coordinates: dimension 0 is the innermost
    int c1 = blockIdx.y * box_h;

    if (t == 0) mbarrier_init(&bar, 1);
    __syncthreads();

    if (t == 0) {
        uint32_t expect = bug == EXPECT_TOO_FEW ? bytes / 2 : bug == EXPECT_TOO_MANY ? bytes * 2 : bytes;
        mbarrier_arrive_expect_tx(&bar, expect);
        tma_load_2d(tile, src_map, &bar, c0, c1);
    }
    if (bug != NO_WAIT) mbarrier_wait(&bar, 0);

    for (int i = t; i < elems; i += blockDim.x) tile[i] *= 2.0f;

    if (bug != NO_FENCE) fence_proxy_async();
    __syncthreads();

    if (t == 0) {
        tma_store_2d(dst_map, tile, c0, c1);
        tma_store_commit();
        if (bug == EARLY_REUSE) tile[0] = -1.0f;
        tma_store_wait();
    }
}
