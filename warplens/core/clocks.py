"""Vector clocks: which accesses a synchronization has ordered, per warp.

Every warp of a cluster has a clock, a counter that starts at 1, and a row of
the matrix `vc`: `vc[w, u]` is the latest clock value of warp `u` whose work
warp `w` is ordered after. An access that warp `u` made at clock `c` is ordered
before whatever `w` does next exactly when `c <= vc[w, u]`.

Only the warps that take part in a synchronization are ordered by it:

  * a warp that arrives at a barrier (`bar.sync`, `bar.arrive`, an mbarrier
    arrival, `barrier.cluster.arrive`) *releases*: the barrier takes a copy of
    its row, and its own clock advances, so that what it does afterwards is
    not covered by that copy;
  * a warp that waits for the barrier and passes it *acquires*: its row
    becomes the maximum of itself and what the arrivals released.

A warp that does neither learns nothing, and what it did is unknown to the
others: a barrier among warps 0 and 1 does not order warp 2's accesses.

The units are warps because the shared-memory checker's are: lanes of one warp
execute an instruction together and are not checked against each other.
"""

from __future__ import annotations

import numpy as np


class Clocks:
    def __init__(self, warps: int = 0):
        self.n = 0
        self.vc = np.zeros((0, 0), dtype=np.int32)
        if warps:
            self.grow(warps)

    def grow(self, warps: int) -> None:
        """Make room for warp ids below `warps`."""
        if warps <= self.n:
            return
        vc = np.zeros((warps, warps), dtype=np.int32)
        vc[:self.n, :self.n] = self.vc
        fresh = np.arange(self.n, warps)
        vc[fresh, fresh] = 1
        self.vc, self.n = vc, warps

    def row(self, warp: int) -> np.ndarray:
        """Warp `warp`'s row; `row[warp]` is its own clock. A view: do not keep it."""
        if warp >= self.n:
            self.grow(max(warp + 1, 2 * self.n))
        return self.vc[warp]

    def release(self, warp: int, into: np.ndarray | None = None) -> np.ndarray:
        """What `warp` has done and knows up to now, joined with `into`; its clock advances."""
        row = self.row(warp)
        out = row.copy() if into is None else join(into, row)
        row[warp] += 1
        return out

    def acquire(self, warp: int, released: np.ndarray | None) -> None:
        """Order `warp` after everything in `released` (from `release` or `join`)."""
        if released is None:
            return
        self.grow(max(warp + 1, len(released)))
        row = self.vc[warp]
        own = row[warp]
        np.maximum(row[:len(released)], released, out=row[:len(released)])
        row[warp] = own  # nothing released can be ahead of the warp itself

    def stamp(self, warp: int) -> tuple[int, int]:
        """(warp, its clock now): names what the warp is doing at this moment."""
        return warp, int(self.row(warp)[warp])

    def ordered(self, stamp: tuple[int, int], warp: int) -> bool:
        """Was the work named by `stamp` ordered before what `warp` does now?"""
        who, clock = stamp
        return who == warp or (who < self.n and clock <= int(self.row(warp)[who]))


def join(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    """Elementwise maximum of two released rows; they may differ in length. A new array."""
    if a is None or b is None:
        return None if a is None and b is None else (b if a is None else a).copy()
    if len(a) < len(b):
        a, b = b, a
    out = a.copy()
    np.maximum(out[:len(b)], b, out=out[:len(b)])
    return out
