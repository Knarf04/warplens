// What cluster-wide shared addresses look like.
//
// A cluster of four CTAs. Thread 0 of every CTA records, for one shared symbol:
//   * its shared-window address (cvta.shared, 64-bit) and its generic address;
//   * mapa.shared::cluster.u32 and .u64 of it for every rank 0-3.
// Each CTA also writes a token to the symbol; after a cluster barrier, rank 0 reads the
// symbol of every rank through the u32 and the u64 mapa address (ld.shared::cluster).
//
// Claims (docs/history/m5.md, "Distributed shared memory"; warplens/core/cta.py): the simulator's
// cluster-wide address of offset o in rank r is r * 2^24 + o. That encoding is the
// simulator's own and the raw addresses go on info: lines. Printed for comparison are the
// facts kernels use:
//   * mapa for a CTA's own rank addresses its own symbol, and every rank's address reads
//     that rank's token, through the u32 and the u64 form alike;
//   * the four addresses are distinct, and the u64 form equals the u32 form;
//   * clearing bit 24 of the address for rank 1 gives the address for rank 0, and likewise
//     3 -> 2 (CUTLASS's Sm100MmaPeerBitMask arithmetic for a pair). This last one is a
//     property of the simulator's encoding that hardware need not share: informational.
//
//   address_dump
#include "../common.h"

#define RANKS 4
#define PER_CTA 16

extern "C" __global__ void address_dump(uint64_t* out) {
    __shared__ uint32_t cell;
    uint32_t rank = cluster_ctarank();
    if (threadIdx.x == 0) cell = 0x7000 + rank;
    cluster_sync();
    if (threadIdx.x == 0) {
        uint64_t* mine = out + rank * PER_CTA;
        uint64_t shared64 = (uint64_t)__cvta_generic_to_shared(&cell);
        mine[0] = shared64;
        mine[1] = (uint64_t)&cell;
        for (uint32_t r = 0; r < RANKS; ++r) {
            uint64_t wide;
            asm volatile("mapa.shared::cluster.u64 %0, %1, %2;" : "=l"(wide) : "l"(shared64), "r"(r));
            uint32_t narrow = mapa(&cell, r), seen32, seen64;
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(seen32) : "r"(narrow) : "memory");
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(seen64) : "l"(wide) : "memory");
            mine[2 + r * 3] = narrow;
            mine[3 + r * 3] = wide;
            mine[4 + r * 3] = ((uint64_t)seen64 << 32) | seen32;
        }
    }
    cluster_sync();
}

int main() {
    uint64_t out[RANKS * PER_CTA];
    uint64_t* d_out = dev_alloc<uint64_t>(RANKS * PER_CTA);
    dev_fill(d_out, RANKS * PER_CTA, 0);
    launch_cluster(address_dump, RANKS, RANKS, 32, d_out);
    probe_sync("address_dump");
    dev_read(out, d_out, RANKS * PER_CTA);
    for (int c = 0; c < RANKS; ++c) {
        const uint64_t* o = out + c * PER_CTA;
        printf("info: rank %d: shared address %016llx, generic address %016llx\n", c, (unsigned long long)o[0],
               (unsigned long long)o[1]);
        bool own = (uint32_t)o[2 + c * 3] == (uint32_t)o[0], reads = true, same_width = true, distinct = true;
        for (int r = 0; r < RANKS; ++r) {
            uint32_t narrow = (uint32_t)o[2 + r * 3];
            uint64_t wide = o[3 + r * 3];
            printf("info: rank %d: mapa to rank %d: u32 %08x, u64 %016llx\n", c, r, narrow, (unsigned long long)wide);
            reads = reads && (uint32_t)o[4 + r * 3] == 0x7000u + r && (uint32_t)(o[4 + r * 3] >> 32) == 0x7000u + r;
            same_width = same_width && wide == narrow;
            for (int q = 0; q < r; ++q) distinct = distinct && narrow != (uint32_t)o[2 + q * 3];
        }
        bool pair = ((uint32_t)o[2 + 1 * 3] & ~(1u << 24)) == (uint32_t)o[2 + 0 * 3]
                    && ((uint32_t)o[2 + 3 * 3] & ~(1u << 24)) == (uint32_t)o[2 + 2 * 3];
        printf("rank %d: mapa to its own rank is the symbol's shared address: %s; each rank's address reads that "
               "rank's token (u32 and u64 forms): %s; u64 form equals u32 form: %s; four distinct addresses: %s; "
               "clearing bit 24 maps rank 1 to 0 and 3 to 2: %s\n",
               c, own ? "yes" : "no", reads ? "yes" : "no", same_width ? "yes" : "no", distinct ? "yes" : "no",
               pair ? "yes" : "no");
    }
    return 0;
}
