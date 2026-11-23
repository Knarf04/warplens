"""State shared by the warps of one CTA: shared memory and named barriers."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ExecutionError
from ..units.smem import SharedMemory

NUM_BARRIERS = 16
WARP_SIZE = 32


class Wait:
    """Why a warp is blocked. With `retry`, the warp re-executes the blocking
    instruction when woken instead of continuing after it."""
    __slots__ = ("kind", "ins", "retry", "detail")

    def __init__(self, kind: str, ins, retry: bool = False, detail=None):
        self.kind, self.ins, self.retry, self.detail = kind, ins, retry, detail


class AsyncOp:
    """One TMA copy in flight (or finished)."""
    __slots__ = ("id", "kind", "warp", "ins", "lane", "done", "swizzle")

    def __init__(self, op_id: int, kind: str, warp, ins, lane: int):
        self.id, self.kind, self.warp, self.ins, self.lane = op_id, kind, warp, ins, lane
        self.done = False
        self.swizzle = None  # swizzle bits of a TMA tensor load (None: not known)


@dataclass(slots=True)
class Barrier:
    """A named barrier (`bar.sync` / `bar.arrive`).

    Arrivals are counted per warp, as 32 threads each. A barrier with no
    explicit thread count waits for every warp of the CTA that is still running.
    """
    arrived: list = field(default_factory=list)  # warps that arrived in this round
    waiters: list = field(default_factory=list)  # (warp, instr) blocked in bar.sync
    count: int | None = None  # explicit thread count, if any arrival gave one
    knowledge: set[int] = field(default_factory=set)  # copies known at arrival, not later

    def satisfied(self, warps) -> bool:
        if self.count is not None:
            return len(self.arrived) * WARP_SIZE >= self.count
        here = {id(w) for w in self.arrived}
        return all(w.done or id(w) in here for w in warps)


class Cta:
    def __init__(self, cta_id: tuple[int, int, int], smem_bytes: int, smem_report, findings,
                 async_delay: int = 4):
        self.id = cta_id
        self.async_delay = async_delay
        self.async_ops: dict[int, AsyncOp] = {}
        self.async_queue: list[list] = []  # [rounds left, AsyncOp, completion callback]
        self.mbarriers: dict[int, object] = {}  # shared offset -> Mbarrier
        self.output: list[str] = []  # text printed by the kernel
        self.tmem = None  # units.tcgen05.Tmem, created on first use
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
        bar.knowledge |= warp.seen
        if count is not None:
            bar.count = count
        if wait:
            bar.waiters.append((warp, ins))
            warp.blocked = Wait("barrier", ins)

    def release_barriers(self) -> bool:
        """Complete every barrier whose arrivals are all in. True if any warp was released."""
        released = False
        for bar in self.barriers.values():
            if bar.arrived and bar.satisfied(self.warps):
                # Arrivals release prior knowledge; only waiting warps acquire
                # it. A bar.arrive caller may have run ahead since its arrival.
                for warp, _ in bar.waiters:
                    warp.seen |= bar.knowledge
                    warp.blocked = None
                released = released or bool(bar.waiters)
                self.smem.epoch += 1
                bar.arrived, bar.waiters, bar.count = [], [], None
                bar.knowledge = set()
        return released

    # -- mbarriers ------------------------------------------------------------
    def mbarrier_init(self, offset: int, count: int, warp, ins) -> None:
        from ..units.mbarrier import mbarrier_init
        mbarrier_init(self, offset, count, warp, ins)

    def mbarrier(self, offset: int, warp, ins, lanes):
        bar = self.mbarriers.get(offset)
        if bar is None:
            from ..errors import ExecutionError
            raise ExecutionError(f"mbarrier at shared offset {offset} is used before mbarrier.init")
        if bar.init_warp != warp.warp_id and bar.init_epoch == self.smem.epoch:
            self.findings.add(
                "mbarrier-init-race",
                f"mbarrier at shared offset {offset} was initialized by warp {bar.init_warp} and "
                "is used by another warp with no barrier in between", warp, ins, lanes)
        return bar

    def phase_completed(self) -> None:
        """An mbarrier phase completed: wake its waiters; it also orders shared-memory accesses."""
        self.smem.epoch += 1
        self._wake("mbarrier")

    def _wake(self, kind: str) -> None:
        for w in self.warps:
            if w.blocked is not None and w.blocked.kind == kind:
                w.blocked = None

    # -- asynchronous operations (TMA) ----------------------------------------
    def new_async(self, kind: str, warp, ins, lane: int) -> AsyncOp:
        operation = AsyncOp(len(self.async_ops) + 1, kind, warp, ins, lane)
        self.async_ops[operation.id] = operation
        return operation

    def schedule(self, operation: AsyncOp, complete) -> None:
        self.async_queue.append([self.async_delay, operation, complete])

    def _complete(self, entry) -> None:
        _, operation, complete = entry
        complete()
        operation.done = True
        if operation.kind == "store":
            self._wake("bulk group")

    def tick_async(self) -> None:
        """One scheduling round has passed."""
        if not self.async_queue:
            return
        for entry in self.async_queue:
            entry[0] -= 1
        while self.async_queue and self.async_queue[0][0] <= 0:
            self._complete(self.async_queue.pop(0))

    def complete_next_async(self) -> bool:
        """Nothing else can run: let the oldest copy in flight finish now."""
        if not self.async_queue:
            return False
        self._complete(self.async_queue.pop(0))
        return True

    def finish(self) -> None:
        """All warps returned. Copies still in flight are reported, then completed."""
        while self.async_queue:
            entry = self.async_queue.pop(0)
            operation = entry[1]
            kind, what, how = {
                "store": ("tma-store-not-waited", "TMA store", "cp.async.bulk.wait_group"),
                "load": ("tma-load-not-waited", "TMA load", "its mbarrier"),
            }.get(operation.kind, ("tcgen05-not-waited", f"tcgen05.{operation.kind}",
                                   "a tcgen05.commit and its mbarrier"))
            self.findings.add(kind, f"the kernel returned while this {what} was still in flight; "
                              f"nothing waited on {how}",
                              operation.warp, operation.ins, [operation.lane])
            self._complete(entry)
        for bar in self.mbarriers.values():
            bar.finish()
        if self.tmem is not None:
            for column, (ncols, (warp, ins, lane)) in self.tmem.allocations.items():
                self.findings.add("tmem-leak", f"{ncols} tensor-memory columns allocated here were "
                                  "never released with tcgen05.dealloc", warp, ins, [lane])

    def describe_blocked(self) -> str:
        from ..check.report import _ranges
        lines = []
        groups: dict[tuple, list[int]] = {}  # warps waiting at the same place for the same thing
        for w in self.warps:
            if w.blocked is not None and w.blocked.kind != "barrier":
                detail = w.blocked.detail() if w.blocked.detail is not None else ""
                where = f"line {w.blocked.ins.line}: {' '.join(w.blocked.ins.text.split())}"
                groups.setdefault((where, detail), []).append(w.warp_id)
        for (where, detail), ids in groups.items():
            who = f"warp {ids[0]} waits" if len(ids) == 1 else f"warps {_ranges(ids)} wait"
            lines.append(f"  {who} at {where}")
            if detail:
                lines.append(f"    {detail}")
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
