"""Schedules warps. For now: round-robin, one instruction per warp per turn.

Stepping warps one instruction at a time (rather than running each to
completion) is what later lets asynchronous units and blocking waits
interleave with them.
"""

from __future__ import annotations

from ..errors import ExecutionError


class Engine:
    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.steps = 0

    def run(self, warps) -> None:
        active = [w for w in warps if not w.done]
        limit = self.max_steps
        while active:
            for w in active:
                w.step()
            self.steps += len(active)
            if limit is not None and self.steps > limit:
                raise ExecutionError(
                    f"kernel did not finish within max_steps={limit} warp instructions "
                    "(infinite loop?)")
            active = [w for w in active if not w.done]
