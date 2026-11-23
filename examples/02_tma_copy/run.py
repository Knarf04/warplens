"""Copy a tensor through shared memory with TMA, then break the kernel on purpose.

    python examples/02_tma_copy/run.py
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
BUGS = ["none", "expect_tx too small", "expect_tx too large", "no mbarrier wait",
        "no fence.proxy.async", "tile overwritten before the store finished"]


def run(swizzle, box, bug=0):
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal((48, 64)).astype(np.float32)
    src = sim.from_numpy(data)
    dst = sim.from_numpy(np.full(data.shape, np.nan, dtype=np.float32))
    # One tensor map per tensor. `box` is the tile each TMA instruction moves, (rows, columns).
    src_map = sim.tensor_map_tiled(src, box, swizzle=swizzle)
    dst_map = sim.tensor_map_tiled(dst, box, swizzle=swizzle)
    grid = (data.shape[1] // box[1], data.shape[0] // box[0])
    stats = sim.launch(here / "tma_copy.ptx", "tma_copy", grid=grid, block=128,
                       args=[src_map, dst_map, box[1], box[0], bug])
    return np.array_equal(dst.numpy(), 2 * data), stats


print("Round trip GMEM -> SMEM -> GMEM for each swizzle mode:")
for swizzle, box in [(None, (16, 32)), ("32B", (16, 8)), ("64B", (16, 16)), ("128B", (16, 32))]:
    ok, stats = run(swizzle, box)
    print(f"  swizzle {str(swizzle):>5}, box {box}: {'correct' if ok else 'WRONG'}, "
          f"{len(stats.findings)} finding(s)")

for bug in range(1, len(BUGS)):
    print(f"\n== bug {bug}: {BUGS[bug]}")
    try:
        ok, stats = run("128B", (16, 32), bug)
    except warplens.DeadlockError as e:
        print(e)
        continue
    print(f"output {'correct' if ok else 'WRONG'}")
    for finding in stats.findings:
        print(finding)
