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
    __slots__ = ("id", "kind", "warp", "ins", "lane", "done", "swizzle", "work")

    def __init__(self, op_id: int, kind: str, warp, ins, lane: int):
        self.id, self.kind, self.warp, self.ins, self.lane = op_id, kind, warp, ins, lane
        self.done = False
        self.swizzle = None  # swizzle bits of a TMA tensor load (None: not known)
        self.work = None  # what the timing model needs: bytes moved, FLOPs, ...


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
        self.timeline = None  # timing.model.Timeline when the launch is timed
        self.cluster = None  # core.cluster.Cluster this CTA belongs to
        self.rank = 0  # position in the cluster
        self.retired = False  # its warps exited; peers may still be running
        self.clc_failed = False  # a thread saw a clusterlaunchcontrol.try_cancel fail
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
                self.smem_report.barrier_completions += 1
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
        if warp.cta is self and bar.init_warp != warp.warp_id and bar.init_epoch == self.smem.epoch:
            self.findings.add(
                "mbarrier-init-race",
                f"mbarrier at shared offset {offset} was initialized by warp {bar.init_warp} and "
                "is used by another warp with no barrier in between", warp, ins, lanes)
        return bar

    def peer(self, rank: int, ins=None):
        """The CTA with this rank in the cluster."""
        ctas = self.cluster.ctas
        if not 0 <= rank < len(ctas):
            from ..errors import ExecutionError
            raise ExecutionError(f"CTA rank {rank} does not exist in a cluster of {len(ctas)} CTA(s)")
        return ctas[rank]

    def phase_completed(self) -> None:
        """An mbarrier phase completed: wake its waiters; it also orders shared-memory accesses."""
        self.smem.epoch += 1
        if self.timeline is not None:
            self.timeline.timing.instants.append(("mbarrier phase completes", self.timeline.now, self.id))
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
        if self.timeline is None:
            self.async_queue.append([self.async_delay, operation, complete])
            return
        # Timed run: the queue holds completion times and stays sorted by them.
        when = self.timeline.schedule(operation)
        at = len(self.async_queue)
        while at and self.async_queue[at - 1][0] > when:
            at -= 1
        self.async_queue.insert(at, [when, operation, complete])

    def _complete(self, entry) -> None:
        _, operation, complete = entry
        complete()
        operation.done = True
        if operation.kind in ("store", "copy"):
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

    def retire(self, *, after: float | None = None) -> None:
        """Diagnose operations outstanding at CTA exit without completing them early.

        In timed mode a queued completion at or before the last instruction's
        finish time has already completed by retirement, even if not dispatched.
        """
        if self.retired:
            return
        self.retired = True
        for when, operation, _ in self.async_queue:
            if after is not None and when <= after:
                continue
            kind, what, how = {
                "store": ("tma-store-not-waited", "TMA store", "cp.async.bulk.wait_group"),
                "load": ("tma-load-not-waited", "TMA load", "its mbarrier"),
                "copy": ("cp-async-not-waited", "cp.async copy", "cp.async.wait_group"),
                "clc": ("clc-not-waited", "clusterlaunchcontrol.try_cancel request", "its mbarrier"),
            }.get(operation.kind, ("tcgen05-not-waited", f"tcgen05.{operation.kind}",
                                   "a tcgen05.commit and its mbarrier"))
            self.findings.add(kind, f"CTA {self.id} returned while this {what} was still in flight; "
                              f"nothing waited on {how}",
                              operation.warp, operation.ins, [operation.lane])

    def finish(self) -> None:
        """Finalize a retired CTA after outstanding operations have been diagnosed."""
        self.retire()
        while self.async_queue:
            entry = self.async_queue.pop(0)
            if self.timeline is not None:
                self.timeline.now = entry[0]
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
