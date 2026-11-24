"""Schedules the warps of one cluster: one instruction per warp per round, by default in
warp order (round-robin).

Stepping warps one instruction at a time (rather than running each to
completion) is what lets blocking waits and asynchronous units interleave with
them, within a CTA and across the CTAs of a cluster.

Real hardware promises no order between warps that nothing synchronizes, so
every order in which runnable warps are stepped is one a correct kernel must
tolerate. A `Schedule` selects another order than round-robin:

  * a seed: the warps that can run are stepped in a pseudo-random order, drawn
    anew every round. The same seed gives the same run.
  * adversarial: the warps are taken last to first, and a warp keeps running,
    up to `ADVERSARIAL_QUANTUM` instructions, until it blocks, exits or polls
    an mbarrier (`test_wait`). The others meanwhile stand still, and an
    asynchronous operation, which completes `async_delay` rounds after issue,
    completes that many times later in instructions. With a seed as well, the
    order and the length of each turn are drawn from it.

The findings of the checkers do not depend on the schedule. The schedule is for
the kernel's results: a kernel whose output changes with it relies on an order
that nothing guarantees.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from ..errors import DeadlockError, ExecutionError, LaunchError

ADVERSARIAL_QUANTUM = 64  # instructions a warp may run ahead in one adversarial turn
SCHEDULE_VARIABLE = "WARPLENS_SCHEDULE"  # the schedule of launches that do not name one


@dataclass(frozen=True, slots=True)
class Schedule:
    """In what order the warps that can run are stepped. The default is round-robin."""
    seed: int | None = None
    adversarial: bool = False

    @classmethod
    def parse(cls, spec) -> "Schedule":
        """From what `Device.launch(schedule=...)`, `warplens run --schedule` and
        WARPLENS_SCHEDULE accept: None or "round-robin"; a seed (an integer, or its
        decimal text, or "seed:N"); "adversarial"; "adversarial:N" for both."""
        if isinstance(spec, cls):
            return spec
        if spec is None:
            return cls()
        if isinstance(spec, int) and not isinstance(spec, bool):
            if spec < 0:
                raise LaunchError(f"schedule seed must not be negative, got {spec}")
            return cls(seed=spec)
        if isinstance(spec, str):
            head, colon, tail = spec.strip().lower().partition(":")
            if not colon and head.isdigit():
                return cls(seed=int(head))
            if head in ("seed", "adversarial") and colon and tail.isdigit():
                return cls(seed=int(tail), adversarial=head == "adversarial")
            if head == "adversarial" and not colon:
                return cls(adversarial=True)
            if head in ("round-robin", "") and not colon:
                return cls()
        raise LaunchError(f"schedule must be None, 'round-robin', a seed (N or 'seed:N'), "
                          f"'adversarial' or 'adversarial:N', got {spec!r}")

    @classmethod
    def from_environment(cls) -> "Schedule":
        try:
            return cls.parse(os.environ.get(SCHEDULE_VARIABLE))
        except LaunchError as error:
            raise LaunchError(f"{SCHEDULE_VARIABLE}: {error}") from None

    def __str__(self) -> str:
        if self.adversarial:
            return "adversarial" if self.seed is None else f"adversarial:{self.seed}"
        return "round-robin" if self.seed is None else f"seed:{self.seed}"


def deadlock(group) -> DeadlockError:
    for bar in group.mbarriers.values():  # record unbalanced byte counts as findings
        bar.finish(when="the kernel deadlocked")
    return DeadlockError(
        f"deadlock in {group.label}: every unfinished warp is blocked and "
        f"nothing can release them\n{group.describe_blocked()}", group.findings.items)


class Engine:
    def __init__(self, max_steps: int | None = None, schedule: Schedule | None = None):
        self.max_steps = max_steps
        self.steps = 0
        self.clock = 0  # scheduling rounds so far in this launch; what %clock64 reads
        self.schedule = schedule or Schedule()
        # One stream for the whole launch, so that its clusters do not repeat each other.
        # Only Random.random() is used: its sequence for a seed is the same on every Python.
        self.random = None if self.schedule.seed is None else random.Random(self.schedule.seed)

    def _below(self, n: int) -> int:
        """A pseudo-random integer in 0..n-1."""
        return min(int(self.random.random() * n), n - 1)

    def _order(self, runnable: list) -> list:
        """The order in which the warps that can run take this round's turns."""
        if self.random is None:
            return runnable[::-1] if self.schedule.adversarial else runnable
        order = list(runnable)
        for i in range(len(order) - 1, 0, -1):  # Fisher-Yates
            j = self._below(i + 1)
            order[i], order[j] = order[j], order[i]
        return order

    def _run_ahead(self, group, w) -> int:
        """An adversarial turn of warp `w`: how many instructions it executed."""
        quantum = ADVERSARIAL_QUANTUM if self.random is None else 1 + self._below(ADVERSARIAL_QUANTUM)
        steps = 0
        while True:
            ins = w.step()
            group.release_barriers()
            steps += 1
            if (w.blocked is not None or w.done or steps >= quantum
                    # A poll asks whether others have got somewhere: let them.
                    or (ins.opcode == "mbarrier" and "test_wait" in ins.mods)):
                return steps
            # %clock64 counts a warp's own instructions too (Warp.clock).
            self.clock += 1
            group.clock = self.clock

    def run(self, group) -> None:
        """`group` is a core.cluster.Cluster."""
        warps = group.warps
        limit = self.max_steps
        adversarial = self.schedule.adversarial
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
            group.clock = self.clock
            if adversarial:
                self.steps += sum(self._run_ahead(group, w) for w in self._order(runnable))
            else:
                for w in self._order(runnable):
                    w.step()
                    # Reset a completed phase before another warp can arrive. A
                    # counted barrier can complete more than once in one round.
                    group.release_barriers()
                self.steps += len(runnable)
            group.tick_async()
            self.clock += 1
            if limit is not None and self.steps > limit:
                raise ExecutionError(
                    f"kernel did not finish within max_steps={limit} warp instructions "
                    "(infinite loop?)")
