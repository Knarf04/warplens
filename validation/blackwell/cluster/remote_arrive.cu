// mbarrier.arrive on another CTA's mbarrier, through its cluster-wide address.
//
// A cluster of two CTAs; thread 0 of each has an mbarrier at the same shared offset,
// initialised with two arrivals per phase.
//   1. rank 0 arrives once on its own mbarrier (ordinary arrive): its phase stays open
//   2. cluster barrier
//   3. rank 1 executes mbarrier.arrive.shared::cluster.b64 _, [mapa(bar, rank 0)]
//   4. cluster barrier
//   5. both poll their own mbarrier (bounded) for the completion of phase 0
//
// Claim tested (docs/history/m5.md; warplens/units/mbarrier.py `_mbarrier`): the remote arrive
// counts on the TARGET CTA's mbarrier. So rank 0's phase completes (one local and one
// remote arrival), and rank 1's own mbarrier, which nobody arrived on, stays open.
// Nothing here waits without a bound; a GPU on which the remote arrival did not count
// prints "phase completed 0" for rank 0.
//
//   remote_arrive
#include "../common.h"

extern "C" __global__ void remote_arrive(uint32_t* out) {
    __shared__ uint64_t bar;
    uint32_t rank = cluster_ctarank();
    bool lead = threadIdx.x == 0;
    uint32_t before = 9;
    if (lead) {
        mbar_init(&bar, 2);
        mbar_init_fence();
        if (rank == 0) mbar_arrive(&bar);
        before = mbar_test(&bar, 0);
    }
    cluster_sync();
    if (lead && rank == 1) {
        uint32_t target = mapa(&bar, 0);
        asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(target) : "memory");
    }
    cluster_sync();
    if (lead) {
        out[rank * 2] = before;
        out[rank * 2 + 1] = mbar_poll(&bar, 0, 4000);
    }
}

int main() {
    uint32_t out[4];
    uint32_t* d_out = dev_alloc<uint32_t>(4);
    dev_fill(d_out, 4, 0xff);
    launch_cluster(remote_arrive, 2, 2, 32, d_out);
    probe_sync("remote_arrive");
    dev_read(out, d_out, 4);
    printf("two arrivals per phase; rank 0 arrives once locally, rank 1 arrives once on rank 0's mbarrier\n");
    printf("rank 0 (the target):   phase completed before the remote arrival %u, after it %u\n", out[0], out[1]);
    printf("rank 1 (the arriver):  its own mbarrier: phase completed before %u, after %u\n", out[2], out[3]);
    return 0;
}
