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

    def bars(operand):
        offsets = _address(w, ins, operand)
        return [(int(l), cta.mbarrier(int(offsets[l]), w, ins, [int(l)])) for l in lanes]

    def count(operand):
        return w.read(operand, "u32")

    if kind == ("init",) and len(ops) == 2:
        offsets, n = _address(w, ins, ops[0]), count(ops[1])
        for l in lanes:
            cta.mbarrier_init(int(offsets[l]), int(n[l]), w, ins)
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
            if done:
                w.seen |= bar.knowledge
            else:
                waiting.append(bar)
        if blocking and waiting:
            w.blocked = Wait("mbarrier", ins, retry=True,
                             detail=lambda: "; ".join(b.describe() for b in dict.fromkeys(waiting)))
            return
        w.write(ops[0], result, mask)
    elif kind == ("inval",) and len(ops) == 1:
        offsets = _address(w, ins, ops[0])
        for l in lanes:
            cta.mbarrier(int(offsets[l]), w, ins, [int(l)])
            del cta.mbarriers[int(offsets[l])]
    else:
        raise unsupported(ins)


@op("fence")
def _fence(w, ins, mask):
    """Fences order accesses made through different proxies or by different threads.

    Only `fence.proxy.async` changes checker state: it makes this warp's earlier
    ordinary writes to shared memory visible to TMA. The others are accepted
    and have no effect in a simulator that executes one instruction at a time.
    """
    mods = ins.mods
    if not mask.any():
        return
    if mods[:2] == ("proxy", "async"):
        w.cta.smem.fence_async(w.warp_id)
    elif mods[:1] in (("mbarrier_init",), ("acq_rel",), ("sc",)) or mods[:1] == ("proxy",):
        pass
    else:
        raise unsupported(ins)


class Mbarrier:
    __slots__ = ("cta", "offset", "expected", "pending", "tx", "phase", "init_warp",
                 "init_epoch", "knowledge", "in_flight")

    def __init__(self, cta, offset: int, expected: int, warp_id: int):
        self.cta = cta
        self.offset = offset
        self.expected = expected  # arrivals per phase
        self.pending = expected  # arrivals still needed in this phase
        self.tx = 0  # bytes still expected in this phase
        self.phase = 0
        self.init_warp = warp_id
        self.init_epoch = cta.smem.epoch
        self.knowledge: set[int] = set()  # async operations a successful waiter may rely on
        self.in_flight = 0  # TMA loads that will still report to this barrier

    def describe(self) -> str:
        text = (f"mbarrier at shared offset {self.offset} is in phase {self.phase}, still needs "
                f"{self.pending} arrival(s) and {self.tx} byte(s) of TMA data")
        if self.tx > 0 and not self.in_flight:
            text += "; no TMA load in flight will deliver them (expect_tx too large?)"
        return text

    def expect_tx(self, n: int) -> None:
        self.tx += n

    def arrive(self, n: int, warp, ins, lanes) -> None:
        self.knowledge |= warp.seen
        if n > self.pending:
            self.cta.findings.add(
                "mbarrier-over-arrival",
                f"{n} arrival(s) on an mbarrier that needed only {self.pending} more in this phase; "
                "the extra ones count toward the next phase", warp, ins, lanes)
        self.pending -= n
        self._check()

    def complete_tx(self, n: int, warp, ins, lanes) -> None:
        if n > self.tx:
            self.cta.findings.add(
                "mbarrier-tx-mismatch",
                f"{n} byte(s) delivered to an mbarrier phase that expected only {max(self.tx, 0)} "
                "more. expect_tx is missing, too small, or came after the phase completed, so "
                "the phase can complete before the data has arrived", warp, ins, lanes)
        self.tx -= n
        self._check()

    def _check(self) -> None:
        if self.pending <= 0 and self.tx <= 0:
            self.phase += 1
            self.pending += self.expected
            self.tx = 0
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
