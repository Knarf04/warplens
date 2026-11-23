"""A pipelined GEMM under the approximate timing model: what limits it with 1, 2 and 4 stages?

    python examples/04_pipelined_gemm/run.py [output directory]

Writes, for the 4-stage run, a Perfetto trace (open it at ui.perfetto.dev) and
an HTML report. The timings come from an uncalibrated model with several
assumed parameters; read them as an explanation of the mechanism, not as a
prediction for a real B200.
"""

import sys
import operator
from pathlib import Path

import numpy as np

import warplens
from warplens.timing.config import GpuConfig

here = Path(__file__).parent
M, N, K = 128, 64, 1024  # one CTA, 16 tiles along K


def run(stages, **timing):
    try:
        if isinstance(stages, (bool, np.bool_)):
            raise TypeError
        stages = operator.index(stages)
    except TypeError:
        raise ValueError("stages must be an integer from 1 to 4") from None
    if not 1 <= stages <= 4:
        raise ValueError("stages must be an integer from 1 to 4")
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (N, K)).astype(np.float16)
    sim = warplens.Device("B200")
    map_a = sim.tensor_map_tiled(sim.from_numpy(A), (128, 64), swizzle="128B")
    map_b = sim.tensor_map_tiled(sim.from_numpy(B), (64, 64), swizzle="128B")
    D = sim.empty((M, N), "f32")
    stats = sim.launch(here / "gemm.ptx", "pipelined_gemm", grid=(M // 128, N // 64), block=128,
                       args=[map_a, map_b, D, N, K // 64, stages], **timing)
    reference = A.astype(np.float32) @ B.astype(np.float32).T
    assert np.allclose(D.numpy(), reference, rtol=1e-4, atol=1e-4) and not stats.findings
    return stats


SCENARIOS = [
    ("whole device busy: 148 SMs share HBM", dict(timing=True)),
    ("8 SMs share HBM", dict(timing=True, active_sms=8)),
    ("8 SMs share HBM, and a TMA engine 4x faster than assumed",
     dict(timing=GpuConfig.load("B200").replace(tma_bytes_per_clk=512), active_sms=8)),
]

if __name__ == "__main__":
    last = None
    for title, timing in SCENARIOS:
        print(f"\n{title}")
        print("  stages   time (ns)   per K tile   limited by      TMA   HBM   Tensor Core")
        for stages in (1, 2, 4):
            t = last = run(stages, **timing).timing
            u = t.utilization
            print(f"  {stages:>6}  {t.time_ns:>10,.0f}  {t.pipeline_ns / (K // 64):>10,.0f}   "
                  f"{t.bottleneck:<12}  {u['TMA engine']:>5.0%} {u['HBM']:>5.0%}  {u['Tensor Core']:>5.0%}")
    print()
    print(last.summary())
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        last.write_trace(out / "pipelined_gemm.trace.json")
        last.write_html(out / "pipelined_gemm.html", "Pipelined GEMM, 4 stages")
        print(f"\nwrote {out / 'pipelined_gemm.trace.json'} and {out / 'pipelined_gemm.html'}")
