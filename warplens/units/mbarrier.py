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

The model's arrival token carries phase parity in bit 63 and the phase number
in its low half. A wait compares the parity bit, reproducing the predicates
recorded on an RTX 3080 Ti (validation/ampere/mbarrier_token_golden.txt).
Hardware token bits are opaque. State tokens are valid only in their current
or immediately preceding phase: the model validates its full encoding and
reports other tokens without granting synchronization credit. For invalid
inputs it keeps the measured Ampere parity predicate only as a nonportable
fallback; those observations do not establish Blackwell behavior.

`try_wait` may suspend the thread on real hardware and, per PTX, return false
after a system-dependent time limit. Here it blocks the warp until the phase
completes and never returns false, which lets a wait that can never succeed
be reported as a deadlock instead of spinning forever. That is exact for the
loop every target kernel uses (`while (!try_wait)`), and it is the v1 scope:
a kernel that does other work when `try_wait` fails (a non-blocking fallback)
is not modeled and shows up as a deadlock; `test_wait` is the non-blocking
form. validation/blackwell/mbarrier/try_wait.cu records what one `try_wait`
on an open phase returns on a B200, and after how long (matrix row MB-11).
"""

from __future__ import annotations

import numpy as np

from ..core.clocks import join
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
            token[l] = token_of(bar.phase)
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
            done = (bar.phase & 1) != (int(arg[l]) & 1 if parity else int(arg[l]) >> 63)
            result[l] = done
            valid_tokens = {token_of(bar.phase)}
            if bar.phase > 0:
                valid_tokens.add(token_of(bar.phase - 1))
            if not parity and int(arg[l]) not in valid_tokens:
                # PTX state tokens are valid only in their own phase and the next.
                # Keep the model's deterministic predicate for exploratory kernels,
                # but never acquire a different phase's completion knowledge.
                cta.findings.add(
                    "mbarrier-invalid-token",
                    f"mbarrier token 0x{int(arg[l]):016x} (phase {int(arg[l]) & 0xFFFFFFFF}) "
                    f"used in phase {bar.phase}; "
                    "only the current or immediately preceding phase is valid. The "
                    "predicate is a nonportable model fallback and grants no synchronization",
                    w, ins, [l])
                continue
            if done:
                if "relaxed" not in ins.mods:
                    w.seen |= bar.completed_knowledge
                    cta.clocks.acquire(w.sync_id, bar.completed_released)
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
                             detail=lambda: "; ".join(b.describe() for b in dict.fromkeys(waiting)),
                             on=dict.fromkeys((b.offset, b.phase) for b in waiting))
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
            w.cta.smem.fence_async(w.sync_id)
    elif mods in (("mbarrier_init", "release", "cluster"), ("proxy", "alias")):
        pass
    elif len(mods) == 2 and mods[0] in ("acq_rel", "sc") and mods[1] in (
            "cta", "cluster", "gpu", "sys"):
        pass
    else:
        raise unsupported(ins)


def token_of(phase: int) -> int:
    """Model token: parity in bit 63 for the predicate, phase below for validation.

    Hardware token bits are opaque; the low half is the simulator's own encoding.
    """
    return (phase & 1) << 63 | phase & 0xFFFFFFFF


# The size of the pieces in which a TMA copy reports its bytes to an mbarrier (Mbarrier.deliver).
# What was measured on a B200 fixes only that it divides 64 (a 256-byte load against
# expect_tx 128, and a 128-byte box against expect_tx 64, both complete the phase): 64 is
# an ASSUMPTION, the largest size the measurements allow, and so the one that completes
# the fewest phases early. validation/blackwell/mbarrier/expect_tx `pieces` asks a GPU.
TX_PIECE_BYTES = 64


class Mbarrier:
    __slots__ = ("cta", "offset", "expected", "pending", "tx", "phase", "init_warp",
                 "init_stamp", "knowledge", "completed_knowledge", "released",
                 "completed_released", "in_flight", "tx_source")

    def __init__(self, cta, offset: int, expected: int, warp_id: int):
        self.cta = cta
        self.offset = offset
        self.expected = expected  # arrivals per phase
        self.pending = expected  # arrivals still needed in this phase
        self.tx = 0  # bytes still expected in this phase
        self.phase = 0
        self.init_warp = warp_id
        self.init_stamp = None  # core.clocks stamp of the mbarrier.init, for mbarrier-init-race
        self.knowledge: set[int] = set()  # async operations contributed to the current phase
        self.completed_knowledge: set[int] = set()  # the immediately preceding completed phase
        # What the arriving warps had done (core.clocks), which a successful wait acquires.
        # It only grows: a phase completes after the phases before it.
        self.released = None
        self.completed_released = None
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

    def arrive(self, n: int, warp, ins, lanes, *, knowledge=None, released=None) -> None:
        """`knowledge` and `released` are given for an arrival made on a warp's behalf some
        time after the instruction (tcgen05.commit): what the warp had then, not now."""
        if "relaxed" not in ins.mods:
            self.knowledge |= warp.seen if knowledge is None else knowledge
            if released is None:
                released = self.cta.clocks.release(warp.sync_id)
            self.released = join(self.released, released)
        if n > self.pending:
            # The PTX manual gives no meaning to this, and a B200 ends the kernel: three
            # arrivals at once on an mbarrier that expected two gave cudaErrorLaunchFailure
            # (Xid 13; validation/blackwell/measured/b200/mbarrier/over_arrival.txt). The
            # first model carried the surplus into the following phases and reported it.
            raise ExecutionError(
                f"{n} arrival(s) on the mbarrier at shared offset {self.offset}, which needed only "
                f"{max(self.pending, 0)} more in this phase: a GPU ends the kernel here with "
                "cudaErrorLaunchFailure (mbarrier-over-arrival)")
        self.pending -= n
        self._check()

    def complete_tx(self, n: int, warp, ins, lanes) -> None:
        self.tx -= n
        if self.tx < 0:
            self.tx_source = (warp, ins, lanes)
        self._check()

    def deliver(self, n: int, warp, ins, lanes) -> None:
        """A TMA copy of `n` bytes has landed. Unlike the instruction `mbarrier.complete_tx`,
        which subtracts its whole count at once, the copy engine reports its bytes in pieces,
        so the tx-count passes through every multiple of TX_PIECE_BYTES on its way down. A
        copy larger than what was expected therefore completes the phase when the count
        reaches zero part of the way through, and the rest of its bytes count against the
        next phase. Measured on a B200 (validation/blackwell/measured/b200/mbarrier/
        expect_tx_small.txt, tma/oob_complete_tx.txt; docs/architecture.md, "TMA and mbarriers"):
        a waiter is released before all the data is there, which is reported."""
        before = self.tx
        if not (0 < before < n and before % TX_PIECE_BYTES == 0 and self.pending <= 0):
            self.complete_tx(n, warp, ins, lanes)
            return
        self.cta.findings.add(
            "mbarrier-tx-mismatch",
            f"a copy of {n} byte(s) landed on the mbarrier at shared offset {self.offset}, which expected "
            f"only {before} more: the phase completed after {before} byte(s), so a wait on it returns "
            f"before the other {n - before} are there; expect_tx is too small", warp, ins, lanes)
        self.tx = 0
        self._check()
        self.complete_tx(n - before, warp, ins, lanes)

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
        self.completed_released = self.released
        self.knowledge = set()
        self.tx_source = None
        # More arrivals than one phase needs complete as many phases as they cover, so the
        # pending count is positive again afterwards (it is negative only while a surplus
        # waits for the phase's bytes).
        while self.pending <= 0:
            self.phase += 1
            self.pending += self.expected
            self.cta.phase_completed(self.offset, self.phase - 1)


def mbarrier_init(cta, offset: int, count: int, warp, ins) -> None:
    if offset % 8:
        raise ExecutionError(f"mbarrier at shared offset {offset} is not 8-byte aligned")
    if offset + 8 > cta.smem.size:
        raise ExecutionError(f"mbarrier at shared offset {offset} is outside the CTA's "
                             f"{cta.smem.size} bytes of shared memory")
    if not 1 <= count < 1 << 20:
        raise ExecutionError(f"mbarrier.init with an arrival count of {count}; it must be 1 to 2^20-1")
    bar = cta.mbarriers[offset] = Mbarrier(cta, offset, count, warp.warp_id)
    bar.init_stamp = cta.clocks.stamp(warp.sync_id)
