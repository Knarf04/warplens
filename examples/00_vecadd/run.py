"""Run the nvcc-compiled vecadd kernel on the CPU and compare with NumPy.

    python examples/00_vecadd/run.py
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
n = 1000
rng = np.random.default_rng(0)
a = rng.standard_normal(n).astype(np.float32)
b = rng.standard_normal(n).astype(np.float32)

sim = warplens.Device("B200")
A, B, C = sim.from_numpy(a), sim.from_numpy(b), sim.empty(n, "f32")
stats = sim.launch(here / "vecadd.ptx", "vecadd", grid=4, block=256, args=[A, B, C, n])

print("matches NumPy:", np.array_equal(C.numpy(), a + b))
print(stats)
