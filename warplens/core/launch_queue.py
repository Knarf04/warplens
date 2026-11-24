"""The clusters of one launch that have not started, and the requests to cancel them.

A launch starts its clusters one at a time, in order (x fastest, then y, then z), each when
the one before has finished. So while a cluster runs, the clusters after it in that order are
exactly those "not launched yet", which is what `clusterlaunchcontrol.try_cancel` may cancel.
A canceled cluster never starts: the cluster that canceled it is handed the coordinate of its
first CTA and is expected to do its work.

See docs/clc.md for what this model can and cannot show.
"""

from __future__ import annotations

from collections import deque


class CancelRequest:
    """One `clusterlaunchcontrol.try_cancel` by one thread."""
    __slots__ = ("serial", "first_ctaid", "warp", "ins", "lane", "decoded")

    def __init__(self, serial: int, first_ctaid, warp, ins, lane: int):
        self.serial = serial
        self.first_ctaid = first_ctaid  # of the canceled cluster; None if the request failed
        self.warp, self.ins, self.lane = warp, ins, lane
        self.decoded = False  # some thread read the coordinate with query_cancel.get_first_ctaid

    @property
    def canceled(self) -> bool:
        return self.first_ctaid is not None


class LaunchQueue:
    def __init__(self, cluster_ids, cluster_shape: tuple[int, int, int], limit: int | None = None):
        self.pending = deque(cluster_ids)  # in launch order
        self.shape = cluster_shape
        self.limit = limit  # cancellations granted to each running cluster; None: no limit
        self.requests: dict[int, CancelRequest] = {}  # by serial, for query_cancel
        self.canceled = 0  # clusters

    def start_next(self):
        """The id of the next cluster to start, or None when none is left."""
        return self.pending.popleft() if self.pending else None

    def try_cancel(self, cluster, warp, ins, lane: int) -> CancelRequest:
        """Cancel the next pending cluster on behalf of `cluster`, if one is left and the
        cluster has not used up its grants."""
        granted = sum(r.canceled for r in cluster.cancel_requests)
        first = None
        if self.pending and (self.limit is None or granted < self.limit):
            qx, qy, qz = self.pending.popleft()
            first = (qx * self.shape[0], qy * self.shape[1], qz * self.shape[2])
            self.canceled += 1
        request = CancelRequest(len(self.requests) + 1, first, warp, ins, lane)
        self.requests[request.serial] = request
        cluster.cancel_requests.append(request)
        return request

    def close(self, cluster) -> None:
        """`cluster` has finished: every cluster it canceled should have been taken over."""
        for request in cluster.cancel_requests:
            if request.canceled and not request.decoded:
                x, y, z = request.first_ctaid
                cluster.findings.add(
                    "clc-canceled-cluster-lost",
                    f"this request canceled the launch of the cluster whose first CTA is "
                    f"({x}, {y}, {z}), but {cluster.label} finished without reading that coordinate "
                    "with clusterlaunchcontrol.query_cancel.get_first_ctaid; nothing ran in its place",
                    request.warp, request.ins, [request.lane])
