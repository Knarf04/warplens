"""The SIMT reconvergence stack of one warp.

Each entry is a group of lanes running together: where they are (`pc`), which
lanes they are (`mask`), and where they rejoin the entry below (`rpc`). Only
the top entry executes. When it reaches its `rpc` it is popped, and its lanes
continue as part of the entry below, which has been waiting at that pc.
"""

from __future__ import annotations

import numpy as np


class Entry:
    __slots__ = ("pc", "mask", "rpc", "frame", "rendezvous")

    def __init__(self, pc: int, mask: np.ndarray, rpc: int, frame=None):
        self.pc, self.mask, self.rpc = pc, mask, rpc
        # Set on the entry a recursive call pushes (units/call.py): what to restore when its
        # lanes return. Such an entry has no reconvergence pc; the return pops it.
        self.frame = frame
        self.rendezvous = False


class SimtStack:
    def __init__(self, mask: np.ndarray, exit_pc: int):
        self.entries: list[Entry] = []
        self.max_depth = 0
        if mask.any():
            self.entries.append(Entry(0, mask.copy(), exit_pc))
            self.max_depth = 1

    @property
    def done(self) -> bool:
        return not self.entries

    @property
    def top(self) -> Entry:
        return self.entries[-1]

    def call(self, pc: int, lanes: np.ndarray, frame) -> None:
        """Lanes of the top entry enter a function; the entry waits below until they return."""
        self.entries.append(Entry(pc, lanes.copy(), -1, frame))
        self.max_depth = max(self.max_depth, len(self.entries))

    def diverge(self, fallthrough: int, target: int, taken: np.ndarray,
                not_taken: np.ndarray, reconv: int) -> None:
        """Split the top entry at a branch where both `taken` and `not_taken` are non-empty."""
        top = self.entries[-1]
        if top.rpc == reconv:
            # The entry below already waits at `reconv` with these lanes in its mask.
            # Reusing it keeps the stack flat across loop iterations.
            self.entries.pop()
        else:
            top.pc = reconv
        if fallthrough != reconv:
            self.entries.append(Entry(fallthrough, not_taken, reconv))
        if target != reconv:
            self.entries.append(Entry(target, taken, reconv))
        self.max_depth = max(self.max_depth, len(self.entries))

    def rendezvous_at(self, pc: int, active: np.ndarray) -> None:
        """Run the other paths before issuing a warp-wide synchronizing instruction.

        Static postdominators need not be the first instruction every *actual* path
        reaches. Park this path at the synchronization point and give the other lanes
        a chance to reach it or exit. Parent entries include their children's lanes;
        only their exclusive masks describe lanes currently waiting at their pc.
        """
        from ..errors import UnsupportedError

        # Recursive activation boundaries carry saved register/local state. Never
        # move lanes across one of those boundaries to manufacture convergence.
        start = next((i for i in range(len(self.entries) - 1, -1, -1)
                      if self.entries[i].frame is not None), 0)
        region = self.entries[start:]
        live = np.zeros_like(active)
        for entry in self.entries:
            live |= entry.mask
        members = np.zeros_like(active)
        for entry in region:
            members |= entry.mask
        if not np.array_equal(members, live):
            raise UnsupportedError("warp rendezvous across divergent recursive calls is not modeled")
        if any(entry.rendezvous and entry.pc != pc for entry in region):
            raise UnsupportedError("warp lanes waiting at different synchronizing instructions "
                                   "cannot rendezvous here")

        covered = np.zeros_like(active)
        pending = []
        for entry in reversed(region):
            lanes = entry.mask & ~covered
            covered |= entry.mask
            if entry is self.top:
                # Predicate-false lanes have not executed this instruction. They
                # continue independently; they may exit or reach it on a later path.
                inactive = lanes & ~active
                if inactive.any():
                    pending.append(Entry(pc + 1, inactive, pc))
            elif entry.pc != pc and lanes.any():
                pending.append(Entry(entry.pc, lanes, pc))
        root = region[0]
        join = Entry(pc, members, root.rpc, root.frame)
        join.rendezvous = True
        # Retain the usual top-first path order after replacing duplicate parent
        # masks with disjoint runnable paths.
        self.entries[start:] = [join, *reversed(pending)]
        self.max_depth = max(self.max_depth, len(self.entries))

    def retire(self, lanes: np.ndarray) -> None:
        """Lanes that returned leave every entry."""
        keep = ~lanes
        for e in self.entries:
            e.mask = e.mask & keep

    def settle(self) -> None:
        """Pop entries that reached their reconvergence point or have no lanes left."""
        entries = self.entries
        while entries:
            top = entries[-1]
            if top.pc == top.rpc or not top.mask.any():
                entries.pop()
            else:
                break
