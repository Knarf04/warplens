"""Named barriers: `bar.sync`, `bar.arrive` (also spelled `barrier` and `.cta`)."""

from __future__ import annotations

import numpy as np

from ..core.cta import NUM_BARRIERS, WARP_SIZE
from ..errors import ExecutionError
from .alu import op, unsupported


def _uniform(w, ins, operand, mask, what: str) -> int:
    values = w.read(operand, "u32")[mask]
    if len(values) and (values != values[0]).any():
        raise ExecutionError(f"{what} differs between the lanes of a warp")
    return int(values[0]) if len(values) else 0


def _cluster_barrier(w, ins, mask):
    """barrier.cluster.arrive / barrier.cluster.wait: a barrier over every thread of the cluster."""
    mods = [m for m in ins.mods[1:] if m not in ("aligned", "release", "acquire", "relaxed")]
    if mods not in (["arrive"], ["wait"]) or ins.operands:
        raise unsupported(ins)
    if not mask.any():
        return
    if mods == ["arrive"]:
        w.cta.cluster.barrier_arrive(w)
    else:
        w.cta.cluster.barrier_wait(w, ins)


@op("bar", "barrier")
def _bar(w, ins, mask):
    if ins.opcode == "barrier" and ins.mods[:1] == ("cluster",):
        return _cluster_barrier(w, ins, mask)
    mods = tuple(m for m in ins.mods if m != "cta")
    if ins.opcode == "barrier" and mods in (("sync", "aligned"), ("arrive", "aligned")):
        mods = mods[:1]  # explicit spelling of the alignment implicit in `bar`
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
    if count is not None and count % WARP_SIZE:
        raise ExecutionError(f"barrier thread count {count} must be a multiple of {WARP_SIZE}")
    if count == 0:
        if not wait:
            raise ExecutionError("bar.arrive requires a non-zero barrier thread count")
        count = None  # zero encodes the all-CTA form, just like an omitted count

    live = w.live_lanes()
    if not np.array_equal(live, mask):
        w.cta.findings.add(
            "barrier-in-divergent-code",
            f"{ins.name} reached by only {int(mask.sum())} of the warp's {int(live.sum())} running "
            "lanes. CUDA only defines a barrier in conditional code when every thread takes the "
            "same path; here the whole warp is counted as arrived",
            w, ins, np.flatnonzero(mask))
    w.cta.arrive(barrier, w, ins, count, wait)
