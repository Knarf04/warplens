// Cluster launch control: clusters that keep working instead of exiting.
//
// The launch has one cluster of two CTAs per pair of tiles, and a CTA scales the tile named by
// its blockIdx. Instead of exiting when that is done, the cluster asks the launch to cancel a
// cluster that has not started yet, and does that cluster's tiles too. It stops when a request
// fails. Each iteration:
//
//   1. thread 0 of every CTA arrives on its CTA's mbarrier and announces 16 bytes (expect_tx)
//   2. thread 0 of the cluster's first CTA issues clusterlaunchcontrol.try_cancel. With
//      .multicast::cluster::all the 16-byte response is written to both CTAs' shared memory
//      and completes both mbarriers
//   3. every thread computes its element of the current tile while the request is in flight
//   4. every thread waits for its CTA's mbarrier phase, then loads the response
//   5. query_cancel.is_canceled says whether a cluster was canceled, and get_first_ctaid gives
//      the blockIdx of its first CTA: the next tiles
//   6. fence.proxy.async orders the read of the response before the next request overwrites
//      it, and the cluster barrier keeps the two CTAs in step
//
// The sequence follows the example in the PTX ISA manual (section 9.7.15.18).

#include <stdint.h>

#define TILE 64  // elements per tile, and threads per CTA

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

extern "C" __global__ void scale_tiles(const float* in, float* out, uint32_t* worker, float factor) {
    __shared__ __align__(16) uint32_t response[4];  // opaque: only query_cancel can read it
    __shared__ __align__(8) uint64_t barrier;
    uint32_t response_addr = smem_addr(response), barrier_addr = smem_addr(&barrier);

    uint32_t rank;  // this CTA's position in its cluster
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(rank));
    uint32_t first = blockIdx.x - rank;  // blockIdx.x of the first CTA of the cluster being computed

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier_addr) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    cluster_sync();

    for (uint32_t phase = 0;; phase ^= 1) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;"
                         :: "r"(barrier_addr) : "memory");  // step 1
            if (rank == 0) {
                asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta"
                             ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
                             :: "r"(response_addr), "r"(barrier_addr) : "memory");  // step 2
            }
        }

        uint32_t tile = first + rank;  // step 3
        uint32_t i = tile * TILE + threadIdx.x;
        out[i] = factor * in[i];
        if (threadIdx.x == 0) worker[tile] = blockIdx.x;

        asm volatile("{\n\t.reg .pred done;\n"
                     "WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
                     "\t@!done bra WAIT;\n\t}"
                     :: "r"(barrier_addr), "r"(phase) : "memory");  // step 4

        uint32_t canceled;
        asm volatile("{\n\t.reg .pred p;\n\t.reg .b128 r;\n"
                     "\tld.shared.b128 r, [%2];\n"
                     "\tclusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
                     "\tselp.u32 %1, 1, 0, p;\n"
                     "\t@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, r;\n\t}"
                     : "+r"(first), "=r"(canceled) : "r"(response_addr) : "memory");  // step 5
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // step 6
        if (!canceled) return;  // nothing left to take over; both CTAs see the same response
        cluster_sync();
    }
}
