"""State shared by the warps of one CTA: shared memory and named barriers."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ExecutionError
from ..units.smem import SharedMemory

NUM_BARRIERS = 16
WARP_SIZE = 32


@dataclass(slots=True)
class Barrier:
    """A named barrier (`bar.sync` / `bar.arrive`).

    Arrivals are counted per warp, as 32 threads each. A barrier with no
    explicit thread count waits for every warp of the CTA that is still running.
    """
    arrived: list = field(default_factory=list)  # warps that arrived in this round
    waiters: list = field(default_factory=list)  # (warp, instr) blocked in bar.sync
    count: int | None = None  # explicit thread count, if any arrival gave one

    def satisfied(self, warps) -> bool:
        if self.count is not None:
            return len(self.arrived) * WARP_SIZE >= self.count
        here = {id(w) for w in self.arrived}
        return all(w.done or id(w) in here for w in warps)


class Cta:
    def __init__(self, cta_id: tuple[int, int, int], smem_bytes: int, smem_report, findings):
        self.id = cta_id
        self.smem = SharedMemory(smem_bytes)
        self.smem_report = smem_report
        self.findings = findings
        self.barriers: dict[int, Barrier] = {}
        self.warps: list = []

    def arrive(self, barrier_id: int, warp, ins, count: int | None, wait: bool) -> None:
        bar = self.barriers.setdefault(barrier_id, Barrier())
        if any(w is warp for w in bar.arrived):
            raise ExecutionError(
                f"warp {warp.warp_id} arrived at barrier {barrier_id} more than once "
                "before the barrier completed")
        if count is not None and bar.count is not None and count != bar.count:
            raise ExecutionError(
                f"barrier {barrier_id} thread count {count} differs from the current "
                f"phase's thread count {bar.count}")
        bar.arrived.append(warp)
        if count is not None:
            bar.count = count
        if wait:
            bar.waiters.append((warp, ins))
            warp.blocked = (barrier_id, ins)

    def release_barriers(self) -> bool:
        """Complete every barrier whose arrivals are all in. True if any warp was released."""
        released = False
        for bar in self.barriers.values():
            if bar.arrived and bar.satisfied(self.warps):
                for warp, _ in bar.waiters:
                    warp.blocked = None
                released = released or bool(bar.waiters)
                self.smem.epoch += 1
                bar.arrived, bar.waiters, bar.count = [], [], None
        return released

    def describe_blocked(self) -> str:
        lines = []
        for barrier_id, bar in sorted(self.barriers.items()):
            if not bar.waiters:
                continue
            running = sum(not w.done for w in self.warps)
            need = (f"{bar.count} threads" if bar.count is not None
                    else f"all {running} running warp(s)")
            lines.append(f"  barrier {barrier_id}: {len(bar.arrived)} warp(s) arrived, needs {need}")
            for warp, ins in bar.waiters:
                lines.append(f"    warp {warp.warp_id} waits at line {ins.line}: {' '.join(ins.text.split())}")
        return "\n".join(lines)
