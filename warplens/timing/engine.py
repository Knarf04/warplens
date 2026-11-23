"""A time-ordered engine: always advances whichever warp or completion is earliest."""

from __future__ import annotations

from ..core.engine import Engine
from ..errors import DeadlockError, ExecutionError


class TimedEngine(Engine):
    def run(self, cta) -> None:
        warps = cta.warps
        line = cta.timeline
        queue = cta.async_queue  # entries [completion time, operation, callback], sorted by time
        for w in warps:
            w.time = 0.0
        while True:
            runnable = [w for w in warps if w.blocked is None and not w.done]
            if not runnable and not queue:
                if all(w.done for w in warps):
                    cta.finish()
                    return
                if cta.release_barriers():
                    continue
                for bar in cta.mbarriers.values():
                    bar.finish(when="the kernel deadlocked")
                raise DeadlockError(
                    f"deadlock in CTA {cta.id}: every unfinished warp is blocked and "
                    f"nothing can release them\n{cta.describe_blocked()}", cta.findings.items)
            waiting = [w for w in warps if w.blocked is not None]
            warp = min(runnable, key=lambda w: w.time) if runnable else None
            if queue and (warp is None or queue[0][0] <= warp.time):
                line.now = queue[0][0]
                cta._complete(queue.pop(0))
            else:
                ins = warp.program.instrs[warp.stack.top.pc]
                line.now = warp.time
                warp.step()
                line.charge(warp, ins)
                line.now = warp.time
                if warp.blocked is not None:
                    line.blocked(warp, warp.time)
                elif warp.done:
                    line.finished(warp)
                self.steps += 1
                if self.max_steps is not None and self.steps > self.max_steps:
                    raise ExecutionError(
                        f"kernel did not finish within max_steps={self.max_steps} warp "
                        "instructions (infinite loop?)")
            cta.release_barriers()
            for w in waiting:
                if w.blocked is None:
                    line.resumed(w, line.now)
            if warp is not None and warp.blocked is None and warp.warp_id in line.wait_start:
                line.resumed(warp, line.now)  # blocked and released within the same step
