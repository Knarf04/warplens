// Cluster launch control: how many bytes a request reports to its mbarrier (matrix CL-20).
//
// One cluster and nothing else in the launch (the request fails; it completes all the
// same). Thread 0 arms its mbarrier with one arrival and expect_tx N, issues ONE
// try_cancel, and polls the phase a bounded number of times. One launch per case:
//   N = 16, one CTA                      the size every CLC kernel announces
//   N = 16 in each of two CTAs, multicast
//   N = 8, one CTA                       the request reports more than was expected
//   N = 32, one CTA                      it reports less
//
// Claim: the completion is a complete_tx of 16 bytes on the mbarrier of each receiving CTA,
// so only the N = 16 phases complete.
//
// Compared lines: "phase completed 0|1" per case and rank. A 1 for N = 8 or N = 32 means the
// byte count is not 16 (or that a transaction count that overshoots completes a phase, see
// MB-07); a 0 for N = 16 means the request took longer than CLC_POLL_BUDGET polls or reports
// another size. The simulator reports the two mismatches as mbarrier-tx-mismatch findings.
//
//   clc_tx_bytes
#include "clc_common.h"

int main() {
    struct Case { uint32_t ctas, expect, multicast; const char* what; };
    const Case cases[] = {
        {1, 16, 0, "one CTA, expect_tx 16"},
        {2, 16, 1, "two CTAs, multicast, expect_tx 16 in each"},
        {1, 8, 0, "one CTA, expect_tx 8"},
        {1, 32, 0, "one CTA, expect_tx 32"},
    };
    for (const Case& c : cases) {
        uint32_t r[2 * CLC_ONE_WORDS];
        printf("%s:\n", c.what);
        clc_run_one(c.ctas, c.expect, c.multicast, 0, 0, r, c.what);
        for (uint32_t rank = 0; rank < c.ctas; ++rank)
            printf("  rank %u: phase completed %u\n", rank, r[rank * CLC_ONE_WORDS]);
    }
    return 0;
}
