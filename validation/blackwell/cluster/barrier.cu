// The cluster barrier and distributed shared memory.
//
// A cluster of four CTAs with two warps each.
//   1. every CTA clears a shared cell; cluster barrier (so every cell is cleared)
//   2. thread 0 of the CTA of rank r spins for (r + 1) * `spin` iterations, then writes
//      0x1000 + r to its cell; the other threads go straight on
//   3. cluster barrier
//   4. thread 0 of each CTA reads the cell of rank (r + 1) % 4 through its cluster-wide
//      address (mapa.shared::cluster, ld.shared::cluster)
// If the barrier really holds every thread until all threads of all four CTAs have arrived,
// each CTA reads its neighbour's token, however long the neighbour spun. If it did not,
// CTAs with a slower neighbour would read 0.
//
// Claims tested (docs/m5.md, "Cluster barrier", "Distributed shared memory"):
//   * barrier.cluster.arrive + barrier.cluster.wait hold a thread until every running
//     thread of the cluster has arrived;
//   * mapa.shared::cluster gives the address of the same shared offset in the CTA of a
//     given rank, and ld.shared::cluster reads it;
//   * %cluster_ctarank is 0..3 along x, %cluster_nctarank is 4.
//
//   barrier
#include "../common.h"

extern "C" __global__ void cluster_barrier(uint32_t spin, uint32_t* out) {
    __shared__ uint32_t cell;
    uint32_t rank = cluster_ctarank(), n = cluster_nctarank();
    if (threadIdx.x == 0) cell = 0;
    cluster_sync();
    if (threadIdx.x == 0) {
        uint32_t sink = 0;
        // An instruction per iteration that the compiler has to keep. (`volatile` shared
        // memory would do, but the simulator has no st.volatile.)
        for (uint32_t i = 0; i < (rank + 1) * spin; ++i) asm volatile("add.u32 %0, %0, %1;" : "+r"(sink) : "r"(i));
        cell = 0x1000 + rank;
    }
    cluster_sync();
    if (threadIdx.x == 0) {
        uint32_t neighbour = (rank + 1) % n, seen;
        uint32_t address = mapa(&cell, neighbour);
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(seen) : "r"(address) : "memory");
        out[rank * 4] = rank;
        out[rank * 4 + 1] = n;
        out[rank * 4 + 2] = neighbour;
        out[rank * 4 + 3] = seen;
    }
}

int main() {
    uint32_t out[16];
    uint32_t* d_out = dev_alloc<uint32_t>(16);
    for (uint32_t spin : {0u, 500u}) {
        dev_fill(d_out, 16, 0xff);
        launch_cluster(cluster_barrier, 4, 4, 64, spin, d_out);
        probe_sync("cluster_barrier");
        dev_read(out, d_out, 16);
        printf("thread 0 of rank r spins %u (r + 1) iterations before writing its token:\n", spin);
        for (int c = 0; c < 4; ++c)
            printf("  grid CTA %d: rank %u of %u read the cell of rank %u: %08x (expected %08x)\n", c, out[c * 4],
                   out[c * 4 + 1], out[c * 4 + 2], out[c * 4 + 3], 0x1000 + (c + 1) % 4);
    }
    return 0;
}
