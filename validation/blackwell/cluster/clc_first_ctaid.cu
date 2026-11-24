// Cluster launch control: the coordinate a successful request returns (matrix CL-21).
//
// 16 x 16 x 8 grids (2048 CTAs, one thread each, two per SM: see clc_common.h) with cluster
// shapes (2, 1, 1), (2, 2, 1), (1, 2, 2), and one launch with no cluster shape at all. The
// first CTA of every cluster that starts works as a CLC kernel does and counts each
// coordinate that query_cancel.get_first_ctaid returns (the .v4 form; x, y and z).
//
// Claim: the coordinate is that of the canceled cluster's first CTA in GRID coordinates,
// clusterid * cluster shape in each dimension; with no cluster shape, the canceled CTA's own
// %ctaid.
//
// Compared lines: the started first-CTA coordinates and the returned coordinates together
// are every cluster origin exactly once; nothing else is returned; no CTA of a canceled
// cluster started. These hold for any number of clusters the device starts at once.
// "info:" lines: how many clusters started and how many requests were granted. If every
// cluster of a launch started, that launch tested nothing; the info line shows it.
//
// In the simulator the first cluster takes over all the others.
//
//   clc_first_ctaid
#include "clc_common.h"

int main() {
    const uint32_t shapes[4][3] = {{2, 1, 1}, {2, 2, 1}, {1, 2, 2}, {0, 0, 0}};
    for (const uint32_t* s : shapes) {
        ClcGrid g = {16, 16, 8, s[0], s[1], s[2]};
        if (s[0]) printf("grid 16 x 16 x 8, cluster shape (%u, %u, %u):\n", s[0], s[1], s[2]);
        else printf("grid 16 x 16 x 8, launched with no cluster shape:\n");
        g.run(0, "clc_worker");
        g.print_invariants();
        g.print_info();
        g.release();
    }
    return 0;
}
