"""A cluster: CTAs that run together and can reach each other's shared memory and mbarriers.

Every launch is made of clusters; without an explicit cluster shape each
cluster holds one CTA. The engine schedules a cluster as a whole, so warps of
different CTAs interleave.
"""

from __future__ import annotations

from .cta import Wait


class Cluster:
    def __init__(self, cluster_id: tuple[int, int, int], shape: tuple[int, int, int]):
        self.id = cluster_id
        self.shape = shape
        self.ctas: list = []  # in rank order: x fastest, then y, then z
        self.async_ops: dict = {}  # shared by the CTAs, so operation ids are unique in the cluster
        self.warps: list = []
        self.arrived: list = []  # warps that arrived at the cluster barrier in this round
        self.waiting: list = []
        self.owed: dict[int, set] = {}  # knowledge for warps whose round completed before they waited

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
    def barrier_arrive(self, warp) -> None:
        if not any(w is warp for w in self.arrived):
            self.arrived.append(warp)

    def barrier_wait(self, warp, ins) -> None:
        """Block until every running warp of the cluster has arrived in this round.
        A warp that did not arrive, or whose round already completed, does not wait."""
        if any(w is warp for w in self.arrived):
            self.waiting.append(warp)
            warp.blocked = Wait("cluster barrier", ins)
        else:
            warp.seen |= self.owed.pop(id(warp), set())

    def _release_cluster_barrier(self) -> bool:
        if not self.arrived:
            return False
        here = {id(w) for w in self.arrived}
        if not all(w.done or id(w) in here for w in self.warps):
            return False
        known = set().union(*(w.seen for w in self.arrived))
        waiting = {id(w) for w in self.waiting}
        for warp in self.arrived:  # warps that met at the barrier share what they know,
            if id(warp) in waiting:  # each one when it waits
                warp.blocked = None
                warp.seen |= known
            else:
                self.owed[id(warp)] = known
        released = bool(self.waiting)
        self.arrived, self.waiting = [], []
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

    def finish(self) -> None:
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
