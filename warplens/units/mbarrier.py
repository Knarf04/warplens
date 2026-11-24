"""mbarrier: the barrier TMA reports to.

An mbarrier lives in 8 bytes of shared memory and counts two things for its
current phase: pending arrivals, and bytes of asynchronous data still expected
(the tx-count). The phase completes when both reach zero; then the phase
number advances and the arrival count is reloaded.

    init [bar], n             expect n arrivals per phase
    arrive [bar]              one arrival
    expect_tx [bar], bytes    announce bytes that TMA will deliver
    arrive.expect_tx          both, the usual form before a TMA load
    complete_tx               what a TMA load does to the barrier when it lands
    test_wait / try_wait      has a given phase completed?

`try_wait` may suspend the thread on real hardware; here it blocks the warp
until the phase completes, which lets a wait that can never succeed be
reported as a deadlock instead of spinning forever.
"""

from __future__ import annotations

import numpy as np

from ..core.cta import Wait
from ..errors import ExecutionError
from ..frontend.ir import Sym
from .alu import _address, op, unsupported
from .smem import CLUSTER_WINDOW, GENERIC_WINDOW, local_shared_offset, resolve_shared

_NOISE = frozenset({"shared", "shared::cta", "shared::cluster", "b64",
                    "release", "relaxed", "acquire", "cta", "cluster"})
_WAITS = {("test_wait",): (False, False), ("try_wait",): (True, False),
          ("test_wait", "parity"): (False, True), ("try_wait", "parity"): (True, True)}


@op("mbarrier")
def _mbarrier(w, ins, mask):
    kind = tuple(m for m in ins.mods if m not in _NOISE)
    lanes = np.flatnonzero(mask)
    if not len(lanes):
        return
    cta, ops = w.cta, ins.operands
    local = any(m in ins.mods for m in ("shared", "shared::cta"))

    def offsets(operand):
        addr = _address(w, ins, operand)
        if not any(m in ins.mods for m in ("shared", "shared::cta", "shared::cluster")):
            window = np.uint64(GENERIC_WINDOW)
            shared = (addr >= window) & (addr < window + np.uint64(CLUSTER_WINDOW))
            if not shared[mask].all():
                raise ExecutionError("generic mbarrier address does not point to shared memory")
            addr = addr - window
        return local_shared_offset(addr) if local else addr

    def bars(operand):
        addr = offsets(operand)
        if local:
            return [(int(l), cta.mbarrier(int(addr[l]), w, ins, [int(l)])) for l in lanes]
        # A cluster-wide address (from mapa) may name another CTA's mbarrier.
        found = []
        for l in lanes:
            target, offset = resolve_shared(cta, int(addr[l]))
            if target is not cta and kind[:1] != ("arrive",):
                raise ExecutionError(f"mbarrier.{'.'.join(kind)} on another CTA's mbarrier; only "
                                     "arrive may be remote")
            found.append((int(l), target.mbarrier(offset, w, ins, [int(l)])))
        return found

    def count(operand):
        return w.read(operand, "u32")

    if kind == ("init",) and len(ops) == 2:
        addr, n = offsets(ops[0]), count(ops[1])
        for l in lanes:
            target, offset = (cta, int(addr[l])) if local else resolve_shared(cta, int(addr[l]))
            if target is not cta:
                raise ExecutionError("mbarrier.init must address this CTA's shared memory")
            cta.mbarrier_init(offset, int(n[l]), w, ins)
    elif kind in (("arrive",), ("arrive", "expect_tx")) and 2 <= len(ops) <= 3:
        expect = kind[-1] == "expect_tx"
        if expect and len(ops) != 3:
            raise unsupported(ins, "arrive.expect_tx needs a byte count")
        n = count(ops[2]) if len(ops) == 3 else None
        token = np.zeros(len(mask), dtype=np.uint64)
        for l, bar in bars(ops[1]):
            token[l] = bar.phase
            if expect:
                bar.expect_tx(int(n[l]))
                bar.arrive(1, w, ins, [l])
            else:
                bar.arrive(int(n[l]) if n is not None else 1, w, ins, [l])
        if ops[0] != Sym("_"):
            w.write(ops[0], token, mask)
    elif kind == ("expect_tx",) and len(ops) == 2:
        n = count(ops[1])
        for l, bar in bars(ops[0]):
            bar.expect_tx(int(n[l]))
    elif kind == ("complete_tx",) and len(ops) == 2:
        n = count(ops[1])
        for l, bar in bars(ops[0]):
            bar.complete_tx(int(n[l]), w, ins, [l])
    elif kind in _WAITS and 3 <= len(ops) <= 4:
        blocking, parity = _WAITS[kind]
        arg = w.read(ops[2], "u32" if parity else "u64")
        result = np.zeros(len(mask), dtype=np.bool_)
        waiting = []
        for l, bar in bars(ops[1]):
            done = (bar.phase & 1) != (int(arg[l]) & 1) if parity else bar.phase != int(arg[l])
            result[l] = done
            if not parity and int(arg[l]) not in (bar.phase, bar.phase - 1):
                # PTX state tokens are valid only in their own phase and the next.
                # Keep the model's deterministic predicate for exploratory kernels,
                # but never acquire a different phase's completion knowledge.
                cta.findings.add(
                    "mbarrier-invalid-token",
                    f"mbarrier token for phase {int(arg[l])} used in phase {bar.phase}; "
                    "only the current or immediately preceding phase is valid. The "
                    "predicate is a nonportable model fallback and grants no synchronization",
                    w, ins, [l])
                continue
            if done:
                if "relaxed" not in ins.mods:
                    w.seen |= bar.completed_knowledge
                else:
                    # A successful relaxed wait can order tcgen05 operations
                    # when followed by fence::after_thread_sync. It does not
                    # acquire TMA/generic-memory visibility on its own.
                    w.tcgen_observed.update(
                        op_id for op_id in bar.completed_knowledge
                        if op_id in cta.async_ops and cta.async_ops[op_id].kind == "mma")
            else:
                waiting.append(bar)
        if blocking and waiting:
            w.blocked = Wait("mbarrier", ins, retry=True,
                             detail=lambda: "; ".join(b.describe() for b in dict.fromkeys(waiting)))
            return
        w.write(ops[0], result, mask)
    elif kind == ("inval",) and len(ops) == 1:
        for _, bar in bars(ops[0]):
            del bar.cta.mbarriers[bar.offset]
    else:
        raise unsupported(ins)


@op("fence")
def _fence(w, ins, mask):
    """Fences order accesses made through different proxies or by different threads.

    An async proxy fence covering shared memory makes this warp's earlier
    ordinary writes visible to TMA. A global-only fence does not cover shared
    memory. Other explicitly supported fences have no effect in this model.
    """
    mods = ins.mods
    if not mask.any():
        return
    if ins.operands:
        raise unsupported(ins)
    if mods in (("proxy", "async"), ("proxy", "async", "shared::cta"),
                ("proxy", "async", "shared::cluster"), ("proxy", "async", "global")):
        if mods[-1] != "global":
            w.cta.smem.fence_async(w.warp_id)
    elif mods in (("mbarrier_init", "release", "cluster"), ("proxy", "alias")):
        pass
    elif len(mods) == 2 and mods[0] in ("acq_rel", "sc") and mods[1] in (
            "cta", "cluster", "gpu", "sys"):
        pass
    else:
        raise unsupported(ins)


class Mbarrier:
    __slots__ = ("cta", "offset", "expected", "pending", "tx", "phase", "init_warp",
                 "init_epoch", "knowledge", "completed_knowledge", "in_flight", "tx_source")

    def __init__(self, cta, offset: int, expected: int, warp_id: int):
        self.cta = cta
        self.offset = offset
        self.expected = expected  # arrivals per phase
        self.pending = expected  # arrivals still needed in this phase
        self.tx = 0  # bytes still expected in this phase
        self.phase = 0
        self.init_warp = warp_id
        self.init_epoch = cta.smem.epoch
        self.knowledge: set[int] = set()  # async operations contributed to the current phase
        self.completed_knowledge: set[int] = set()  # the immediately preceding completed phase
        self.in_flight = 0  # TMA loads that will still report to this barrier
        self.tx_source = None  # completion responsible for an unresolved negative tx-count

    def describe(self) -> str:
        text = (f"mbarrier at shared offset {self.offset} is in phase {self.phase}, still needs "
                f"{self.pending} arrival(s) and {self.tx} byte(s) of TMA data")
        if self.tx < 0:
            text += (f"; {-self.tx} completion byte(s) have not been accounted for by expect_tx "
                     "(expect_tx too small or missing?)")
        elif self.tx > 0 and not self.in_flight:
            text += "; no TMA load in flight will deliver them (expect_tx too large?)"
        return text

    def expect_tx(self, n: int) -> None:
        self.tx += n
        self._check()

    def arrive(self, n: int, warp, ins, lanes, *, knowledge=None) -> None:
        if "relaxed" not in ins.mods:
            self.knowledge |= warp.seen if knowledge is None else knowledge
        if n > self.pending:
            # The PTX manual gives no meaning to this. warplens carries the surplus into the
            # following phases (docs/m2.md, an assumption) and reports it.
            self.cta.findings.add(
                "mbarrier-over-arrival",
                f"{n} arrival(s) on an mbarrier that needed only {max(self.pending, 0)} more in "
                "this phase; the extra ones count toward the following phases", warp, ins, lanes)
        self.pending -= n
        self._check()

    def complete_tx(self, n: int, warp, ins, lanes) -> None:
        self.tx -= n
        if self.tx < 0:
            self.tx_source = (warp, ins, lanes)
        self._check()

    def finish(self, when: str = "the kernel returned") -> None:
        # Completions may precede expect_tx, so a temporary negative tx-count is
        # valid. Only diagnose unbalanced credits after the kernel has finished.
        if self.tx < 0 and self.tx_source is not None:
            warp, ins, lanes = self.tx_source
            self.cta.findings.add(
                "mbarrier-tx-mismatch",
                f"{when} with {-self.tx} completion byte(s) not accounted for by "
                f"expect_tx on the mbarrier at shared offset {self.offset}; expect_tx is missing "
                "or too small", warp, ins, lanes)

    def _check(self) -> None:
        if self.pending > 0 or self.tx != 0:
            return
        self.completed_knowledge = self.knowledge
        self.knowledge = set()
        self.tx_source = None
        # More arrivals than one phase needs complete as many phases as they cover, so the
        # pending count is positive again afterwards (it is negative only while a surplus
        # waits for the phase's bytes).
        while self.pending <= 0:
            self.phase += 1
            self.pending += self.expected
            self.cta.phase_completed()


def mbarrier_init(cta, offset: int, count: int, warp, ins) -> None:
    if offset % 8:
        raise ExecutionError(f"mbarrier at shared offset {offset} is not 8-byte aligned")
    if offset + 8 > cta.smem.size:
        raise ExecutionError(f"mbarrier at shared offset {offset} is outside the CTA's "
                             f"{cta.smem.size} bytes of shared memory")
    if not 1 <= count < 1 << 20:
        raise ExecutionError(f"mbarrier.init with an arrival count of {count}; it must be 1 to 2^20-1")
    cta.mbarriers[offset] = Mbarrier(cta, offset, count, warp.warp_id)
