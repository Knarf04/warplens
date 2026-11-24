// Cluster launch control: which cluster a request cancels (matrix CL-14), and whether a
// request fails while clusters are still to start (CL-15).
//
// A 64 x 32 grid of one-CTA clusters, one thread each (clc_worker in clc_common.h). Each
// CTA asks for 100 KB of shared memory it never touches, so an SM holds two of them and the
// 2048 CTAs are about seven times what a B200 starts at once. Every cluster that starts
// works on its own coordinate, then requests cancellations and works on each coordinate it
// is handed, until a request fails.
//
// Compared lines: what PTX promises whatever the mix of started and canceled clusters is.
// Every cluster either starts or is handed to a running cluster, never both; no
// coordinate is returned twice; none is returned that is not a cluster's first CTA.
// "info:" lines: how many clusters started; whether the first cluster was handed the
// coordinates in launch order (x fastest, then y), which is the simulator's rule and is not
// promised by the manual; whether any cluster started after a request had already failed (the
// simulator never fails a request while clusters are pending).
//
// In the simulator (`warplens run` grants every request) cluster (0, 0) takes over all the
// other 2047 clusters, in launch order.
//
//   clc_order
#include "clc_common.h"

int main() {
    ClcGrid g = {64, 32, 1, 1, 1, 1};
    printf("grid 64 x 32 x 1, clusters of one CTA:\n");
    g.run(0, "clc_worker");
    g.print_invariants();
    g.print_info();
    // The coordinates the first cluster was handed, in the order it got them.
    uint32_t mine = 0, in_order = 0;
    static uint32_t by_seq[64 * 32 + 1];
    for (uint32_t c = 0; c < g.total; ++c)
        if (g.returned_by[c] == 1 && g.seq[c] > 0 && g.seq[c] <= g.total) {
            ++mine;
            by_seq[g.seq[c]] = c;
        }
    while (in_order < mine && by_seq[in_order + 1] == in_order + 1) ++in_order;
    printf("info: the cluster at (0, 0) took over %u cluster(s); the first %u of them came in launch order\n", mine,
           in_order);
    printf("info: its first coordinates, as linear indices:");
    for (uint32_t k = 1; k <= mine && k <= 16; ++k) printf(" %u", by_seq[k]);
    printf("\n");
    uint32_t shown = 0;
    printf("info: first clusters that started, as linear indices:");
    for (uint32_t c = 0; c < g.total && shown < 16; ++c)
        if (g.started[c]) {
            printf(" %u", c);
            ++shown;
        }
    printf("\n");
    g.release();
    return 0;
}
