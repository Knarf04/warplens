"""Schedules the warps of one CTA: round-robin, one instruction per warp per turn.

Stepping warps one instruction at a time (rather than running each to
completion) is what lets blocking waits, and later asynchronous units,
interleave with them.
"""

from __future__ import annotations

from ..errors import DeadlockError, ExecutionError


class Engine:
    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.steps = 0

    def run(self, cta) -> None:
        warps = cta.warps
        limit = self.max_steps
        while True:
            runnable = [w for w in warps if w.blocked is None and not w.done]
            if not runnable:
                if all(w.done for w in warps):
                    cta.finish()
                    return
                if cta.release_barriers() or cta.complete_next_async():
                    continue
                raise DeadlockError(
                    f"deadlock in CTA {cta.id}: every unfinished warp is blocked and "
                    f"nothing can release them\n{cta.describe_blocked()}")
            for w in runnable:
                w.step()
                # Reset a completed phase before another warp can arrive. A
                # counted barrier can complete more than once in one round.
                cta.release_barriers()
            cta.tick_async()
            self.steps += len(runnable)
            if limit is not None and self.steps > limit:
                raise ExecutionError(
                    f"kernel did not finish within max_steps={limit} warp instructions "
                    "(infinite loop?)")
