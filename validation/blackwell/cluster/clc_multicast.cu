// Cluster launch control: who receives the response (matrix CL-19).
//
// One cluster of two CTAs and nothing else in the launch, so the request fails; a failed
// request still writes a response and completes the mbarrier, which is all this needs, and
// it makes the outcome the same on a GPU and in the simulator. Both CTAs put a sentinel in
// their response slot and arm their own mbarrier (one arrival, expect_tx 16). Rank 0 issues
// one try_cancel. Each CTA polls its own mbarrier (bounded) and reports its slot.
// Two launches: the plain form, then .multicast::cluster::all.
//
// Claims: the plain form writes only the requesting CTA's slot and completes only its
// mbarrier; the multicast form writes the same response at the same offset in every CTA of
// the cluster and completes each one's mbarrier.
//
// Compared lines: per rank, whether the phase completed and whether the slot still holds the
// sentinel; for the multicast form, whether both CTAs hold the same 16 bytes.
// "info:" lines: the response words (opaque, and the simulator's own encoding).
//
//   clc_multicast
#include "clc_common.h"

int main() {
    for (uint32_t multicast = 0; multicast < 2; ++multicast) {
        uint32_t r[2 * CLC_ONE_WORDS];
        printf("cluster of two CTAs, rank 0 requests %s:\n",
               multicast ? "with .multicast::cluster::all" : "without multicast");
        clc_run_one(2, 16, multicast, 0, 0, r, multicast ? "clc_one_request (multicast)" : "clc_one_request (plain)");
        for (int rank = 0; rank < 2; ++rank) {
            const uint32_t* m = r + rank * CLC_ONE_WORDS;
            bool sentinel = m[1] == CLC_SENTINEL && m[2] == CLC_SENTINEL && m[3] == CLC_SENTINEL
                            && m[4] == CLC_SENTINEL;
            printf("  rank %d: phase completed %u; response slot %s\n", rank, m[0],
                   sentinel ? "still holds the sentinel" : "was written");
            if (m[0]) printf("  rank %d: is_canceled %u\n", rank, m[5]);
            printf("info: rank %d response words %08x %08x %08x %08x\n", rank, m[1], m[2], m[3], m[4]);
        }
        printf("  both CTAs hold the same 16 bytes: %s\n", memcmp(r + 1, r + CLC_ONE_WORDS + 1, 16) ? "no" : "yes");
    }
    return 0;
}
