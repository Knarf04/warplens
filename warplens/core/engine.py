"""Schedules the warps of one cluster: round-robin, one instruction per warp per turn.

Stepping warps one instruction at a time (rather than running each to
completion) is what lets blocking waits and asynchronous units interleave with
them, within a CTA and across the CTAs of a cluster.
"""

from __future__ import annotations

from ..errors import DeadlockError, ExecutionError


def deadlock(group) -> DeadlockError:
    for bar in group.mbarriers.values():  # record unbalanced byte counts as findings
        bar.finish(when="the kernel deadlocked")
    return DeadlockError(
        f"deadlock in {group.label}: every unfinished warp is blocked and "
        f"nothing can release them\n{group.describe_blocked()}", group.findings.items)


class Engine:
    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.steps = 0

    def run(self, group) -> None:
        """`group` is a core.cluster.Cluster."""
        warps = group.warps
        limit = self.max_steps
        while True:
            group.retire_ctas()
            runnable = [w for w in warps if w.blocked is None and not w.done]
            if not runnable:
                if all(w.done for w in warps):
                    group.finish()
                    return
                if group.release_barriers() or group.complete_next_async():
                    continue
                raise deadlock(group)
            for w in runnable:
                w.step()
                # Reset a completed phase before another warp can arrive. A
                # counted barrier can complete more than once in one round.
                group.release_barriers()
            group.tick_async()
            self.steps += len(runnable)
            if limit is not None and self.steps > limit:
                raise ExecutionError(
                    f"kernel did not finish within max_steps={limit} warp instructions "
                    "(infinite loop?)")
