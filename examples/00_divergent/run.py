"""Watch a warp diverge: print the active-lane mask at every instruction of one warp.

    python examples/00_divergent/run.py
"""

from pathlib import Path

import numpy as np

import warplens

here = Path(__file__).parent
x = np.array([1, 2, 3, 6, 7, 27, 97, 871] * 4, dtype=np.uint32)

sim = warplens.Device("B200")
X, steps = sim.from_numpy(x), sim.empty(len(x), "u32")


shown = 0


def trace(warp, pc, ins, mask):
    # The loop runs for hundreds of iterations; show the first 40 instructions only.
    global shown
    if shown < 40:
        shown += 1
        lanes = "".join("#" if m else "." for m in mask)
        print(f"{pc:3d} {lanes}  {' '.join(ins.text.split())}")


stats = sim.launch(here / "divergent.ptx", "collatz", grid=1, block=32,
                   args=[X, steps, len(x)], trace=trace)
print("...")
print("x:    ", x[:8].tolist())
print("steps:", steps.numpy()[:8].tolist())
print(stats)
