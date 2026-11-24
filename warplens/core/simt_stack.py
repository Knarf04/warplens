"""The SIMT reconvergence stack of one warp.

Each entry is a group of lanes running together: where they are (`pc`), which
lanes they are (`mask`), and where they rejoin the entry below (`rpc`). Only
the top entry executes. When it reaches its `rpc` it is popped, and its lanes
continue as part of the entry below, which has been waiting at that pc.
"""

from __future__ import annotations

import numpy as np


class Entry:
    __slots__ = ("pc", "mask", "rpc", "frame")

    def __init__(self, pc: int, mask: np.ndarray, rpc: int, frame=None):
        self.pc, self.mask, self.rpc = pc, mask, rpc
        # Set on the entry a recursive call pushes (units/call.py): what to restore when its
        # lanes return. Such an entry has no reconvergence pc; the return pops it.
        self.frame = frame


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
