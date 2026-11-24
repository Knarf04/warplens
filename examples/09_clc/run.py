"""Cluster launch control: a cluster that finishes its tiles takes over one that has not started.

    python examples/09_clc/run.py

The simulator starts clusters one at a time, so a kernel that asks for more work gets all of
it: the first cluster ends up doing every tile. On a GPU many clusters start at once and far
fewer requests succeed. `clc_limit` shows the kernel both ways, and in between: it is how
many cancellations each cluster that starts is granted. The result must not depend on it.
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
TILE, TILES, CLUSTER = 64, 12, 2

for limit in (None, 2, 0):
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal(TILES * TILE).astype(np.float32)
    out, worker = sim.zeros(TILES * TILE, "f32"), sim.zeros(TILES, "u32")
    stats = sim.launch(here / "clc.ptx", "scale_tiles", grid=TILES, block=TILE, cluster=CLUSTER,
                       args=[sim.from_numpy(data), out, worker, 3.0], clc_limit=limit)
    correct = np.array_equal(out.numpy(), np.float32(3.0) * data)
    granted = "every request while clusters are left" if limit is None else f"{limit} per cluster"
    print(f"cancellations granted: {granted}")
    print(f"  {stats.ctas - stats.canceled_ctas} of {stats.ctas} CTAs started, {stats.canceled_ctas} were canceled; "
          f"output {'correct' if correct else 'WRONG'}, {len(stats.findings)} finding(s)")
    print(f"  tile -> CTA that computed it: {worker.numpy().tolist()}")
