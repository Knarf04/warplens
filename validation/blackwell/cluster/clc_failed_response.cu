// Cluster launch control: what a failed response decodes to (matrix CL-17), and a request
// issued after a failure (CL-18).
//
// Default case: a grid of ONE cluster of one CTA, so there is nothing to cancel and the one
// request must fail. Thread 0 arms its mbarrier with expect_tx 16, issues try_cancel, polls
// (bounded), then decodes the response with every query form.
//   Compared lines: the phase completes, and is_canceled is false. Both are promised.
//   "info:" lines: what get_first_ctaid gives for the failed response (undefined in PTX; the
//   simulator gives 0, 0, 0 and its own tag as the fourth .v4 element, with a
//   clc-ctaid-of-failed-cancel finding), and the 16 raw response bytes (opaque).
//
// Case `again` (own process; PTX leaves it undefined, so it may hang or fault): the same
// kernel then issues three more requests, each announced and polled for.
//   Compared line: none beyond the first request's. "info:" lines: which of the three
//   completed and whether any reported a canceled cluster. The simulator: all complete, all
//   fail, with clc-try-cancel-after-failure findings.
//
// Case `again-grid` (own process, same reason): the 64 x 32 grid of clc_order, where every
// worker issues one more request after the first failure it observes and goes on working if
// that one succeeds.
//   Compared lines: the invariants of clc_order, which must hold whatever the repeats do.
//   "info:" lines: how many repeats were issued and how many succeeded. The simulator: one
//   repeat (by the cluster that took over everything), which fails.
//
//   clc_failed_response [again|again-grid]
#include "clc_common.h"

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    if (!strcmp(which, "again-grid")) {
        ClcGrid g = {64, 32, 1, 1, 1, 1};
        printf("grid 64 x 32 x 1, clusters of one CTA; one more request after an observed failure:\n");
        g.run(1, "clc_worker (again)");
        g.print_invariants();
        g.print_info();
        printf("info: requests issued after an observed failure: %u, of which succeeded: %u\n", g.repeats_issued,
               g.repeats_granted);
        g.release();
        return 0;
    }
    uint32_t repeats = !strcmp(which, "again") ? 3 : 0;
    uint32_t r[CLC_ONE_WORDS];
    printf("one cluster of one CTA, one request:\n");
    clc_run_one(1, 16, 0, repeats, 1, r, "clc_one_request");
    printf("  phase completed %u\n", r[0]);
    if (r[0]) printf("  is_canceled %u\n", r[5]);
    else printf("  is_canceled not queried: the request did not complete\n");
    printf("info: get_first_ctaid.v4 of the failed response: %u %u %u, fourth element %08x\n", r[6], r[7], r[8],
           r[9]);
    printf("info: get_first_ctaid::x, ::y, ::z of the failed response: %u %u %u\n", r[10], r[11], r[12]);
    printf("info: response words %08x %08x %08x %08x\n", r[1], r[2], r[3], r[4]);
    if (repeats) {
        printf("three more requests after the failure was observed: see the info lines\n");
        for (uint32_t i = 0; i < repeats; ++i)
            printf("info: repeat %u: %s%s\n", i,
                   r[13] >> i & 1 ? "completed" : "did not complete within the poll budget",
                   r[13] >> i & 1 ? (r[14] >> i & 1 ? ", canceled a cluster" : ", failed") : "");
        if (r[14]) printf("info: last coordinate returned by a repeat: %u %u %u\n", r[15], r[16], r[17]);
    }
    return 0;
}
