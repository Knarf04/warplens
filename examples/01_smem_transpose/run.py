"""Transpose through shared memory in three layouts and compare their bank conflicts.

    python examples/01_smem_transpose/run.py
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
ctas = 2
data = np.random.default_rng(0).standard_normal((ctas, 32, 32)).astype(np.float32)

sim = warplens.Device("B200")
src = sim.from_numpy(data)
for layout in ("linear", "padded", "swizzled"):
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(here / "transpose.ptx", f"transpose_{layout}",
                       grid=ctas, block=(32, 32), args=[src, out])
    ok = np.array_equal(out.numpy(), data.transpose(0, 2, 1))
    print(f"== {layout}: output {'matches NumPy' if ok else 'WRONG'}, "
          f"{stats.smem_bytes} bytes of shared memory per CTA")
    print(stats.report(), end="\n\n")
