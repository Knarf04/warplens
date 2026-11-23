"""Named barriers: `bar.sync`, `bar.arrive` (also spelled `barrier` and `.cta`)."""

from __future__ import annotations

import numpy as np

from ..core.cta import NUM_BARRIERS
from ..errors import ExecutionError
from .alu import op, unsupported


def _uniform(w, ins, operand, mask, what: str) -> int:
    values = w.read(operand, "u32")[mask]
    if len(values) and (values != values[0]).any():
        raise ExecutionError(f"{what} differs between the lanes of a warp")
    return int(values[0]) if len(values) else 0


@op("bar", "barrier")
def _bar(w, ins, mask):
    mods = tuple(m for m in ins.mods if m != "cta")
    if mods not in (("sync",), ("arrive",)):
        raise unsupported(ins)
    wait = mods == ("sync",)
    if not 1 <= len(ins.operands) <= 2 or (not wait and len(ins.operands) != 2):
        raise unsupported(ins, "expected `bar.sync a[, count]` or `bar.arrive a, count`")
    if not mask.any():
        return
    barrier = _uniform(w, ins, ins.operands[0], mask, "barrier number")
    if barrier >= NUM_BARRIERS:
        raise ExecutionError(f"barrier number {barrier} is out of range (0-{NUM_BARRIERS - 1})")
    count = _uniform(w, ins, ins.operands[1], mask, "barrier thread count") if len(ins.operands) == 2 else None

    live = w.live_lanes()
    if not np.array_equal(live, mask):
        w.cta.findings.add(
            "barrier-in-divergent-code",
            f"{ins.name} reached by only {int(mask.sum())} of the warp's {int(live.sum())} running "
            "lanes. CUDA only defines a barrier in conditional code when every thread takes the "
            "same path; here the whole warp is counted as arrived",
            w, ins, np.flatnonzero(mask))
    w.cta.arrive(barrier, w, ins, count, wait)
