"""Cluster launch control: a running cluster takes over one that has not started.

    clusterlaunchcontrol.try_cancel.async{.shared::cta}.mbarrier::complete_tx::bytes
                                         {.multicast::cluster::all}.b128 [addr], [mbar];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128            p, response;
    clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128      {x, y, z, _}, response;
    clusterlaunchcontrol.query_cancel.get_first_ctaid::{x,y,z}.b32.b128   d, response;

`try_cancel` asks the launch to cancel a cluster that has not started (PTX ISA 9.5, section
9.7.15.18). The answer is a 16-byte response written to shared memory some scheduling rounds
later, on the async proxy like a TMA load, and reported to an mbarrier as 16 bytes. With
`.multicast::cluster::all` every CTA of the cluster receives the response at the same offset
and the completion on its own mbarrier. `query_cancel` (section 9.7.15.19) decodes a response
that was loaded into a `.b128` register.

Which cluster is canceled, and when a request fails, is decided by `core.launch_queue`.

The response is opaque in PTX. The encoding here is warplens's own, not the hardware's:

    bytes 0-3, 4-7, 8-11   x, y, z of the first CTA of the canceled cluster (zero on failure)
    bytes 12-15            0xC1 in the top byte, the request's number, and bit 0 = canceled
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError
from ..frontend.ir import B128, Sym, Vec
from .alu import _address, op, unsupported
from .smem import CLUSTER_WINDOW, GENERIC_WINDOW, local_shared_offset, resolve_shared

RESPONSE_BYTES = 16
_MAGIC = 0xC1
_MAX_SERIAL = (1 << 23) - 1

_TRY_CANCEL = {("try_cancel", "async", *space, "mbarrier::complete_tx::bytes", *multicast, "b128")
               for space in ((), ("shared::cta",)) for multicast in ((), ("multicast::cluster::all",))}
_DIMENSIONS = {"get_first_ctaid::x": 0, "get_first_ctaid::y": 1, "get_first_ctaid::z": 2}


def encode(request) -> np.ndarray:
    """The 16 response bytes of a request."""
    if request.serial > _MAX_SERIAL:
        raise ExecutionError(f"more than {_MAX_SERIAL} cancellation requests in one launch")
    x, y, z = request.first_ctaid or (0, 0, 0)
    tag = _MAGIC << 24 | request.serial << 1 | int(request.canceled)
    return np.array([x, y, z, tag], dtype="<u4").view(np.uint8)


def _decode(response: np.ndarray):
    """(x, y, z, tag, is a response, canceled) of `.b128` values, per lane."""
    lo, hi = response["lo"], response["hi"]
    x, y = (lo & np.uint64(0xFFFF_FFFF)).astype(np.uint32), (lo >> np.uint64(32)).astype(np.uint32)
    z, tag = (hi & np.uint64(0xFFFF_FFFF)).astype(np.uint32), (hi >> np.uint64(32)).astype(np.uint32)
    genuine = (tag >> np.uint32(24)) == _MAGIC
    return x, y, z, tag, genuine, genuine & ((tag & np.uint32(1)) == 1)


def _local_offsets(w, ins, operand, mask, what: str) -> np.ndarray:
    """Offsets in this CTA's shared memory that an address operand names."""
    addr = _address(w, ins, operand)
    if "shared::cta" in ins.mods:
        return local_shared_offset(addr)
    window = np.uint64(GENERIC_WINDOW)  # no state space: generic addresses
    inside = (addr >= window) & (addr < window + np.uint64(CLUSTER_WINDOW))
    if not inside[mask].all():
        raise ExecutionError(f"{what} is a generic address outside shared memory")
    out = np.zeros(len(mask), dtype=np.uint64)
    for lane in np.flatnonzero(mask):
        target, out[lane] = resolve_shared(w.cta, int(addr[lane] - window))
        if target is not w.cta:
            raise ExecutionError(f"{what} is in another CTA's shared memory; "
                                 "clusterlaunchcontrol.try_cancel takes .shared::cta addresses")
    return out


def _try_cancel(w, ins, mask) -> None:
    if ins.mods not in _TRY_CANCEL or len(ins.operands) != 2:
        raise unsupported(ins, "expected clusterlaunchcontrol.try_cancel.async[.shared::cta]"
                               ".mbarrier::complete_tx::bytes[.multicast::cluster::all].b128 [addr], [mbar]")
    if not mask.any():
        return
    cta = w.cta
    cluster = cta.cluster
    multicast = "multicast::cluster::all" in ins.mods
    responses = _local_offsets(w, ins, ins.operands[0], mask, "the response address")
    barriers = _local_offsets(w, ins, ins.operands[1], mask, "the mbarrier address")
    for lane in (int(l) for l in np.flatnonzero(mask)):  # each thread makes its own request
        start, bar_offset = int(responses[lane]), int(barriers[lane])
        if start % RESPONSE_BYTES:
            raise ExecutionError(f"response address at shared offset {start} is not aligned to "
                                 f"{RESPONSE_BYTES} bytes")
        if start + RESPONSE_BYTES > cta.smem.size:
            raise ExecutionError(f"the {RESPONSE_BYTES}-byte response at shared offset {start} runs past "
                                 f"the CTA's {cta.smem.size} bytes of shared memory")
        targets = cluster.ctas if multicast else [cta]
        bars = [target.mbarrier(bar_offset, w, ins, [lane]) for target in targets]
        if cta.clc_failed:
            cta.findings.add(
                "clc-try-cancel-after-failure",
                "this CTA already saw a cancellation request fail (query_cancel.is_canceled was "
                "false); PTX leaves another try_cancel after that undefined. Here it fails again",
                w, ins, [lane])
        if multicast and any(target.retired for target in targets):
            cta.findings.add(
                "clc-multicast-to-exited-cta",
                "try_cancel with .multicast::cluster::all after a CTA of the cluster has exited; "
                "PTX leaves that undefined. Here the exited CTA's shared memory is still written",
                w, ins, [lane])
        idx = np.arange(start, start + RESPONSE_BYTES, dtype=np.intp)
        for target in targets:
            for op_id in np.unique(target.smem.async_writer[idx]):
                if op_id > 0 and int(op_id) not in w.seen:
                    previous = target.async_ops[int(op_id)]
                    state = "is still in flight" if not previous.done else \
                        "has completed, but this warp never waited on its mbarrier"
                    if previous.kind == "clc":
                        kind, what = "clc-response-unsynchronized", \
                            f"the response of the try_cancel at line {previous.ins.line}; that request"
                    else:
                        kind, what = "tma-load-unsynchronized", \
                            f"shared memory that the TMA load at line {previous.ins.line} writes; that load"
                    cta.findings.add(kind, f"try_cancel overwrites {what} {state}", w, ins, [lane])

        # The launch answers now, so requests are served in the order they are issued; the
        # answer only becomes readable when the response lands.
        request = cluster.launch.try_cancel(cluster, w, ins, lane)
        data = encode(request)
        operation = cta.new_async("clc", w, ins, lane)
        for target, bar in zip(targets, bars):
            target.smem.begin_async_write(idx, operation.id)
            bar.in_flight += 1

        def land(targets=targets, bars=bars, idx=idx, data=data, operation=operation, lane=lane):
            for target, bar in zip(targets, bars):
                target.smem.finish_async_write(idx, data)
                bar.in_flight -= 1
                bar.knowledge.add(operation.id)
                bar.complete_tx(RESPONSE_BYTES, w, ins, [lane])

        cta.schedule(operation, land)


def _query_cancel(w, ins, mask) -> None:
    mods = ins.mods[1:]
    if len(ins.operands) != 2:
        raise unsupported(ins, "expected a destination and a response")
    dest, source = ins.operands
    whole = mods == ("get_first_ctaid", "v4", "b32", "b128")
    single = len(mods) == 3 and mods[0] in _DIMENSIONS and mods[1:] == ("b32", "b128")
    if mods != ("is_canceled", "pred", "b128") and not whole and not single:
        raise unsupported(ins, "expected .is_canceled.pred.b128, .get_first_ctaid.v4.b32.b128 or "
                               ".get_first_ctaid::{x,y,z}.b32.b128")
    if whole and not (isinstance(dest, Vec) and len(dest.elems) == 4):
        raise unsupported(ins, "get_first_ctaid.v4 needs a destination of four elements, {x, y, z, _}")
    if not mask.any():
        return
    response = w.read(source, "b128")
    if response.dtype != B128:
        raise ExecutionError("the response operand is not a .b128 value")
    x, y, z, tag, genuine, canceled = _decode(response)
    cta = w.cta
    forged = mask & ~genuine
    if forged.any():
        cta.findings.add(
            "clc-query-not-a-response",
            "the value queried was not written by clusterlaunchcontrol.try_cancel (wrong address, "
            "or loaded before the request's mbarrier phase completed?); it is treated as a "
            "failed request", w, ins, np.flatnonzero(forged))
    if mods[0] == "is_canceled":
        if (mask & genuine & ~canceled).any():
            cta.clc_failed = True  # the CTA has now observed a failed request
        w.write(dest, canceled, mask)
        return
    failed = mask & genuine & ~canceled
    if failed.any():
        cta.findings.add(
            "clc-ctaid-of-failed-cancel",
            "get_first_ctaid of a request that did not cancel anything; PTX leaves the result "
            "undefined. Here it is zero. Predicate the instruction on is_canceled",
            w, ins, np.flatnonzero(failed))
    requests = cta.cluster.launch.requests
    serials = (tag >> np.uint32(1)) & np.uint32(_MAX_SERIAL)
    for serial in np.unique(serials[mask & canceled]):
        if (request := requests.get(int(serial))) is not None:
            request.decoded = True  # the coordinate reached a thread that can do the work
    zero = np.zeros(len(mask), dtype=np.uint32)
    coords = [np.where(canceled, c, zero) for c in (x, y, z)]
    if single:
        w.write(dest, coords[_DIMENSIONS[mods[0]]], mask)
        return
    for element, value in zip(dest.elems, (*coords, tag)):  # the fourth element is unspecified
        if element != Sym("_"):
            w.write(element, value, mask)


@op("clusterlaunchcontrol")
def _clusterlaunchcontrol(w, ins, mask):
    if ins.mods[:1] == ("try_cancel",):
        return _try_cancel(w, ins, mask)
    if ins.mods[:1] == ("query_cancel",):
        return _query_cancel(w, ins, mask)
    raise unsupported(ins)
