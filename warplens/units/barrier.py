"""Named barriers: `bar.sync`, `bar.arrive` (also spelled `barrier` and `.cta`)."""

from __future__ import annotations

import numpy as np

from ..core.cta import NUM_BARRIERS, WARP_SIZE
from ..errors import ExecutionError
from .alu import membermask_check
from .dispatch import op, unsupported


def _uniform(w, ins, operand, mask, what: str) -> int:
    values = w.read(operand, "u32")[mask]
    if len(values) and (values != values[0]).any():
        raise ExecutionError(f"{what} differs between the lanes of a warp")
    return int(values[0]) if len(values) else 0


def _cluster_barrier(w, ins, mask):
    """barrier.cluster.arrive / barrier.cluster.wait: a barrier over every thread of the cluster."""
    mods = list(ins.mods[1:])
    if mods and mods[-1] == "aligned":
        mods.pop()
    if tuple(mods) not in (("arrive",), ("arrive", "release"), ("arrive", "relaxed"),
                           ("wait",), ("wait", "acquire")) or ins.operands:
        raise unsupported(ins)
    if not mask.any():
        return
    live = w.live_lanes()
    if not np.array_equal(live, mask):
        if "aligned" not in ins.mods:
            raise unsupported(ins, "divergent cluster barrier rendezvous is not modeled")
        w.cta.findings.add("barrier-in-divergent-code",
                           f"{ins.name} reached by only {int(mask.sum())} of the warp's "
                           f"{int(live.sum())} running lanes; .aligned requires uniform "
                           "participation; here the whole warp is counted as arrived",
                           w, ins, np.flatnonzero(mask))
    if mods[0] == "arrive":
        w.cta.cluster.barrier_arrive(w, ins, release="relaxed" not in mods)
    else:
        w.cta.cluster.barrier_wait(w, ins)


def _warp_sync(w, ins, mask):
    """bar.warp.sync membermask (`__syncwarp()`): the threads of the warp named by the mask
    wait for each other, which orders their memory accesses before and after it.

    The lanes of a warp that execute an instruction here execute it together, and everything
    they did earlier has already happened, so there is nothing to wait for. What is not
    modeled is a rendezvous between lanes on different divergent paths: those run one path
    after the other, as for `shfl.sync`, and that case is reported."""
    if len(ins.operands) != 1:
        raise unsupported(ins, "expected `bar.warp.sync membermask`")
    if not mask.any():
        return
    _, _, mismatch = membermask_check(w, ins.operands[0], mask)
    if mismatch.any():
        w.cta.findings.add(
            "warp-sync-membermask-mismatch",
            "bar.warp.sync caller is absent from its member mask, or a non-exited member is "
            "not executing it with the same member mask; lanes on separate divergent paths "
            "do not wait for each other here", w, ins, np.flatnonzero(mismatch))


@op("bar", "barrier")
def _bar(w, ins, mask):
    if ins.opcode == "barrier" and ins.mods[:1] == ("cluster",):
        return _cluster_barrier(w, ins, mask)
    if ins.opcode == "bar" and ins.mods == ("warp", "sync"):
        return _warp_sync(w, ins, mask)
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

    # Warp.step has collected every non-exited lane before counting this warp.
    # Reaching this instruction through different paths is not itself a violation.
    w.cta.arrive(barrier, w, ins, count, wait)
