"""Sum each block with warp shuffles and dynamically sized shared memory.

    python examples/01_warp_reduce/run.py
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
blocks, threads = 4, 256
x = np.random.default_rng(0).integers(-100, 100, (blocks, threads)).astype(np.int32)

sim = warplens.Device("B200")
out = sim.empty(blocks, "s32")
# `smem` sizes the kernel's `extern __shared__ int partial[]`: one int per warp.
stats = sim.launch(here / "reduce.ptx", "block_sum", grid=blocks, block=threads,
                   smem=(threads // 32) * 4, args=[sim.from_numpy(x), out])

print("sums:   ", out.numpy().tolist())
print("NumPy:  ", x.sum(axis=1).tolist())
print(stats.report())
