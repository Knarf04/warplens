"""A cluster: CTAs that run together and can reach each other's shared memory and mbarriers.

Every launch is made of clusters; without an explicit cluster shape each
cluster holds one CTA. The engine schedules a cluster as a whole, so warps of
different CTAs interleave.
"""

from __future__ import annotations

from .cta import Wait
from ..errors import ExecutionError


class Cluster:
    def __init__(self, cluster_id: tuple[int, int, int], shape: tuple[int, int, int]):
        self.id = cluster_id
        self.shape = shape
        self.ctas: list = []  # in rank order: x fastest, then y, then z
        self.async_ops: dict = {}  # shared by the CTAs, so operation ids are unique in the cluster
        self.warps: list = []
        self.arrived: list = []  # warps that arrived at the cluster barrier in this round
        self.waiting: list = []
        self.knowledge: set[int] = set()  # only knowledge released at arrival
        self.owed: dict[int, set] = {}  # knowledge for warps whose round completed before they waited
        self.launch = None  # core.launch_queue.LaunchQueue of the launch this cluster belongs to
        self.cancel_requests: list = []  # its clusterlaunchcontrol.try_cancel requests

    def add(self, cta) -> None:
        cta.cluster, cta.rank = self, len(self.ctas)
        cta.async_ops = self.async_ops
        self.ctas.append(cta)

    @property
    def label(self) -> str:
        return f"CTA {self.ctas[0].id}" if len(self.ctas) == 1 else f"cluster {self.id}"

    @property
    def findings(self):
        return self.ctas[0].findings

    @property
    def mbarriers(self) -> dict:
        return {(c.rank, offset): bar for c in self.ctas for offset, bar in c.mbarriers.items()}

    # -- the cluster barrier: barrier.cluster.arrive / barrier.cluster.wait ---------------
    def barrier_arrive(self, warp, ins=None, *, release: bool = True) -> None:
        if any(w is warp for w in self.arrived):
            raise ExecutionError("arrived at the cluster barrier twice before its phase completed")
        # A completed older phase need not be waited on, but its knowledge
        # cannot satisfy the wait corresponding to this new arrival.
        self.owed.pop(id(warp), None)
        self.arrived.append(warp)
        if release:
            self.knowledge |= warp.seen

    def barrier_wait(self, warp, ins) -> None:
        """Acquire the preceding arrival's phase, or wait until it completes."""
        if id(warp) in self.owed:
            warp.seen |= self.owed.pop(id(warp))
            return
        # A wait with no arrival cannot satisfy the all-thread arrival count.
        # It blocks rather than silently allowing this warp to pass the barrier.
        self.waiting.append(warp)
        warp.blocked = Wait("cluster barrier", ins)

    def _release_cluster_barrier(self) -> bool:
        if not self.arrived:
            return False
        here = {id(w) for w in self.arrived}
        if not all(w.done or id(w) in here for w in self.warps):
            return False
        known = self.knowledge
        waiting = {id(w) for w in self.waiting}
        for warp in self.arrived:  # warps that met at the barrier share what they know,
            if id(warp) in waiting:  # each one when it waits
                warp.blocked = None
                warp.seen |= known
            else:
                self.owed[id(warp)] = known
        released = bool(self.waiting)
        self.arrived, self.waiting = [], []
        self.knowledge = set()
        for cta in self.ctas:
            cta.smem.epoch += 1
        return released

    # -- what the engine needs -----------------------------------------------------------
    def release_barriers(self) -> bool:
        released = False
        for cta in self.ctas:
            released = cta.release_barriers() or released
        return self._release_cluster_barrier() or released

    def tick_async(self) -> None:
        for cta in self.ctas:
            cta.tick_async()

    def next_async(self):
        """The CTA whose oldest asynchronous operation completes first, or None."""
        pending = [c for c in self.ctas if c.async_queue]
        return min(pending, key=lambda c: c.async_queue[0][0]) if pending else None

    def complete_next_async(self) -> bool:
        cta = self.next_async()
        return cta.complete_next_async() if cta is not None else False

    def retire_ctas(self) -> None:
        """Record in-flight work at each CTA's exit, even while its peers run."""
        for cta in self.ctas:
            if not cta.retired and all(w.done for w in cta.warps):
                after = max(w.time for w in cta.warps) if cta.timeline is not None else None
                cta.retire(after=after)

    def finish(self) -> None:
        self.retire_ctas()
        # Complete pending work in cluster order before inspecting any remote
        # mbarrier's final counts. A different CTA can still complete that bar.
        while (source := self.next_async()) is not None:
            if source.timeline is not None:
                when = source.async_queue[0][0]
                for cta in self.ctas:
                    cta.timeline.now = when
            source.complete_next_async()
        for cta in self.ctas:
            cta.finish()

    def describe_blocked(self) -> str:
        lines = []
        for cta in self.ctas:
            text = cta.describe_blocked()
            if text and len(self.ctas) > 1:
                lines.append(f"  in CTA {cta.id} (rank {cta.rank} in the cluster):")
                text = "\n".join("  " + line for line in text.splitlines())
            if text:
                lines.append(text)
        stuck = [w for w in self.waiting if w.blocked is not None]
        if stuck:
            lines.append(f"  cluster barrier: {len(self.arrived)} warp(s) arrived, needs all "
                         f"{sum(not w.done for w in self.warps)} running warp(s) of the cluster")
            for w in stuck:
                lines.append(f"    CTA {w.cta.id} warp {w.warp_id} waits at line {w.blocked.ins.line}: "
                             f"{' '.join(w.blocked.ins.text.split())}")
        return "\n".join(lines)
