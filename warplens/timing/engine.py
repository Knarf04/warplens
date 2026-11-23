"""A time-ordered engine: always advances whichever warp or completion is earliest."""

from __future__ import annotations

from ..core.engine import Engine, deadlock
from ..errors import ExecutionError


class TimedEngine(Engine):
    def run(self, group) -> None:
        warps = group.warps
        lines = [cta.timeline for cta in group.ctas]
        for w in warps:
            w.time = 0.0
        while True:
            runnable = [w for w in warps if w.blocked is None and not w.done]
            source = group.next_async()  # each CTA's queue is sorted by completion time
            if not runnable and source is None:
                if all(w.done for w in warps):
                    group.finish()
                    return
                if group.release_barriers():
                    continue
                raise deadlock(group)
            waiting = [w for w in warps if w.blocked is not None]
            warp = min(runnable, key=lambda w: w.time) if runnable else None
            if source is not None and (warp is None or source.async_queue[0][0] <= warp.time):
                now = source.async_queue[0][0]
                for line in lines:
                    line.now = now
                source._complete(source.async_queue.pop(0))
                warp = None
            else:
                line = warp.cta.timeline
                ins = warp.program.instrs[warp.stack.top.pc]
                for other in lines:
                    other.now = warp.time
                warp.step()
                line.charge(warp, ins)
                now = warp.time
                for other in lines:
                    other.now = now
                if warp.blocked is not None:
                    line.blocked(warp, now)
                elif warp.done:
                    line.finished(warp)
                self.steps += 1
                if self.max_steps is not None and self.steps > self.max_steps:
                    raise ExecutionError(
                        f"kernel did not finish within max_steps={self.max_steps} warp "
                        "instructions (infinite loop?)")
            group.release_barriers()
            for w in waiting:
                if w.blocked is None:
                    w.cta.timeline.resumed(w, now)
            if warp is not None and warp.blocked is None and warp.warp_id in warp.cta.timeline.wait_start:
                warp.cta.timeline.resumed(warp, now)  # blocked and released within the same step
