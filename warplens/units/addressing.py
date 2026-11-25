"""Address operands, the generic address space, and the loads and stores of a warp's lanes.

What every instruction that touches memory shares: the address a `[reg+offset]` operand
names for each lane (`effective_address`), which memory a generic address belongs to
(`route`, and the windows it is made of), and the access itself, lane by lane, to global,
shared, local or parameter memory and to the shared memory of another CTA of the cluster
(`load_lanes`, `store_lanes`), with the findings and the bank accounting that go with it
(`async_check`, `record_memory_banks`). The handlers of `ld` and `st` are in alu.py; those of
the atomics, TMA, mbarriers, tcgen05 and cluster launch control use the same pieces.
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError, MemoryFault
from ..frontend.ir import Mem, Reg
from .dispatch import unsupported
from .smem import CLUSTER_STRIDE, CLUSTER_WINDOW, GENERIC_WINDOW, local_shared_offset

__all__ = ["LOCAL_WINDOW", "LOCAL_WINDOW_BYTES", "PARAM_WINDOW", "async_check", "effective_address",
           "fault_lanes", "load_lanes", "record_memory_banks", "route", "store_lanes"]


def effective_address(w, ins, mem) -> np.ndarray:
    if not isinstance(mem, Mem):
        raise unsupported(ins, "expected a [address] operand")
    base = w.read_address(mem.base)
    if not mem.offset:
        return base
    address = base + np.uint64(mem.offset % (1 << 64))
    # [reg+offset] is summed in the register's width. nvcc 13.5 relies on it for shared
    # memory: `fixed[N - 1 - tid]` becomes `sub.s32 %r, fixed, 4 * tid` and `[%r+4*(N-1)]`,
    # where %r has wrapped below zero. An RTX 3080 Ti reads the wrapped address
    # (tests/execution/test_sim_bug_regressions.py).
    if type(mem.base) is Reg and w.regs[mem.base.name].itemsize == 4:
        address &= np.uint64(0xFFFFFFFF)
    return address


LOCAL_WINDOW = 0x1_4000_0000  # where per-thread local memory appears in the generic space
LOCAL_WINDOW_BYTES = 1 << 24


PARAM_WINDOW = 0x1_8000_0000  # where the kernel's parameters appear in the generic space


def route(address: int) -> tuple[str, int]:
    """Which memory a generic address refers to, and the address within it."""
    if PARAM_WINDOW <= address < PARAM_WINDOW + LOCAL_WINDOW_BYTES:
        return "param", address - PARAM_WINDOW
    if GENERIC_WINDOW <= address < GENERIC_WINDOW + CLUSTER_WINDOW:
        return "shared", address - GENERIC_WINDOW
    if LOCAL_WINDOW <= address < LOCAL_WINDOW + LOCAL_WINDOW_BYTES:
        return "local", address - LOCAL_WINDOW
    return "global", address


def _cluster_groups(w, addrs: np.ndarray, mask: np.ndarray):
    """Split a `shared::cluster` access by the CTA it addresses:
    yields (CTA, lanes, offsets within that CTA's shared memory)."""
    stride = np.uint64(CLUSTER_STRIDE)
    which = addrs // stride
    for rank in np.unique(which[mask]):
        lanes = mask & (which == rank)
        target = w.cta.peer(int(rank))
        yield target, lanes, addrs % stride


def _split_generic(space: str, addrs: np.ndarray, mask: np.ndarray):
    """(lanes addressing global memory, lanes addressing shared memory, shared offsets,
    lanes addressing local memory, local offsets)."""
    if space in ("global", "const"):
        return mask, None, None, None, None
    if space == "shared":
        return None, mask, local_shared_offset(addrs), None, None
    if space == "local":
        return None, None, None, mask, addrs
    window = np.uint64(GENERIC_WINDOW)
    in_smem = (addrs >= window) & (addrs < window + np.uint64(CLUSTER_WINDOW))
    local = np.uint64(LOCAL_WINDOW)
    in_local = (addrs >= local) & (addrs < local + np.uint64(LOCAL_WINDOW_BYTES))
    to_smem, to_local = mask & in_smem, mask & in_local
    return (mask & ~in_smem & ~in_local, (to_smem if to_smem.any() else None),
            local_shared_offset(addrs - window),
            (to_local if to_local.any() else None), addrs - local)


def _local_index(w, ins, lanes: np.ndarray, offsets: np.ndarray, size: int, verb: str):
    total = w.local.shape[1]
    off = offsets[lanes]
    bad = off > np.uint64(max(total - size, 0)) if total >= size else np.ones(len(off), bool)
    if bad.any():
        e = MemoryFault(f"{verb} of {size} byte(s) at local offset {int(off[bad][0])} is outside "
                        f"the thread's {total} bytes of local memory", np.flatnonzero(bad), off[bad])
        e.lanes = [int(x) for x in lanes[bad]]
        raise e
    return lanes[:, None], off.astype(np.intp)[:, None] + np.arange(size, dtype=np.intp)


def load_lanes(w, ins, space, addrs, dt, mask) -> np.ndarray:
    value = np.zeros(len(mask), dtype=dt)
    if space == "generic":
        shared = mask & (addrs >= GENERIC_WINDOW) & (addrs < GENERIC_WINDOW + CLUSTER_WINDOW)
        if shared.any():
            value[shared] = load_lanes(w, ins, "shared::cluster", addrs - np.uint64(GENERIC_WINDOW),
                                  dt, shared)[shared]
            mask = mask & ~shared
    if space == "shared::cluster":  # distributed shared memory: maybe another CTA's
        for target, lanes, offsets in _cluster_groups(w, addrs, mask):
            if target is w.cta:
                value[lanes] = load_lanes(w, ins, "shared", offsets, dt, lanes)[lanes]
                continue
            try:
                value[lanes], unwritten, _ = target.smem.read(offsets[lanes], dt, w.sync_id)
            except MemoryFault as e:
                fault_lanes(e, lanes)
                raise
            if len(unwritten):
                w.cta.findings.add("smem-uninitialized-read",
                                   f"read of shared memory that CTA {target.id} never wrote",
                                   w, ins, np.flatnonzero(lanes)[unwritten])
        return value
    # Kernel parameters are read-only and their generic window is contained
    # in global space (PTX 5.1.6); ld.global can therefore also read them.
    if space in ("generic", "global", "param"):
        if space == "param":
            param, offsets = mask, addrs
        else:
            window = np.uint64(PARAM_WINDOW)
            param = mask & (addrs >= window) & (addrs < window + np.uint64(LOCAL_WINDOW_BYTES))
            offsets = addrs - window
        if param.any():
            lanes = np.flatnonzero(param)
            raw = np.frombuffer(w.program.param_blob(w.params), dtype=np.uint8)
            off = offsets[param]
            bad = off > np.uint64(max(len(raw) - dt.itemsize, 0)) if len(raw) >= dt.itemsize \
                else np.ones(len(off), bool)
            if bad.any():
                e = MemoryFault(f"read of {dt.itemsize} byte(s) at parameter offset "
                                f"{int(off[bad][0])} is outside the kernel's {len(raw)} parameter bytes",
                                np.flatnonzero(bad), off[bad])
                e.lanes = [int(x) for x in lanes[bad]]
                raise e
            idx = off.astype(np.intp)[:, None] + np.arange(dt.itemsize)
            value[param] = raw[idx].view(dt).ravel()
            mask = mask & ~param
    g, s, offsets, loc, local_offsets = _split_generic(space, addrs, mask)
    try:
        if g is not None and g.any():
            value[g] = w.mem.read(addrs[g], dt)
    except MemoryFault as e:
        fault_lanes(e, g)
        raise
    if loc is not None and loc.any():
        lanes = np.flatnonzero(loc)
        value[loc] = w.local[_local_index(w, ins, lanes, local_offsets, dt.itemsize, "read")] \
            .view(dt).ravel()
    if s is not None and s.any():
        cta = w.cta
        try:
            value[s], unwritten, racy = cta.smem.read(offsets[s], dt, w.sync_id)
        except MemoryFault as e:
            fault_lanes(e, s)
            raise
        if cta.smem.async_active:
            async_check(w, ins, s, offsets, dt.itemsize, "read of", writing=False)
        if len(unwritten):
            cta.findings.add("smem-uninitialized-read",
                             "read of shared memory that this CTA never wrote",
                             w, ins, np.flatnonzero(s)[unwritten])
        if len(racy):
            cta.findings.add("smem-race",
                             "read of shared memory that another warp wrote with no barrier "
                             "in between", w, ins, np.flatnonzero(s)[racy])
    return value


def async_check(w, ins, s, offsets, size: int, what: str, writing: bool) -> None:
    """Report ordinary accesses to bytes that TMA is using and this warp has not
    synchronized with: a load it never waited for, or a store still reading."""
    cta = w.cta
    smem = cta.smem
    idx = smem.index(offsets[s], size, "access")
    lanes = np.flatnonzero(s)

    def unseen(shadow):
        ids = shadow[idx]
        return [(int(i), lanes[(ids == i).any(axis=1)]) for i in np.unique(ids[ids > 0])
                if int(i) not in w.seen]

    for op_id, hit in unseen(smem.async_writer):
        operation = cta.async_ops[op_id]
        state = "is still in flight" if not operation.done else \
            "has completed, but this warp never waited on its mbarrier"
        if operation.kind == "clc":
            cta.findings.add("clc-response-unsynchronized",
                             f"{what} the response that the clusterlaunchcontrol.try_cancel at line "
                             f"{operation.ins.line} writes; that request {state}", w, ins, hit)
            continue
        cta.findings.add("tma-load-unsynchronized",
                         f"{what} shared memory that the TMA load at line {operation.ins.line} "
                         f"writes; that load {state}", w, ins, hit)
    if writing:
        for op_id, hit in unseen(smem.async_reader):
            operation = cta.async_ops[op_id]
            if operation.kind == "mma":
                state = "is still reading it" if not operation.done else \
                    "has completed, but this warp never waited on the mbarrier of its tcgen05.commit"
                cta.findings.add("mma-operand-overwritten",
                                 f"write to shared memory that the tcgen05.mma at line "
                                 f"{operation.ins.line} reads; that MMA {state}", w, ins, hit)
                continue
            state = "is still reading it" if not operation.done else \
                "has completed, but this warp never waited for it (cp.async.bulk.wait_group.read)"
            cta.findings.add("tma-store-source-overwritten",
                             f"write to shared memory that the TMA store at line "
                             f"{operation.ins.line} reads; that store {state}", w, ins, hit)


def store_lanes(w, ins, space, addrs, value, mask) -> None:
    if space == "generic":
        shared = mask & (addrs >= GENERIC_WINDOW) & (addrs < GENERIC_WINDOW + CLUSTER_WINDOW)
        if shared.any():
            store_lanes(w, ins, "shared::cluster", addrs - np.uint64(GENERIC_WINDOW), value, shared)
            mask = mask & ~shared
    if space == "shared::cluster":
        for target, lanes, offsets in _cluster_groups(w, addrs, mask):
            if target is w.cta:
                store_lanes(w, ins, "shared", offsets, value, lanes)
                continue
            try:
                target.smem.write(offsets[lanes], value[lanes], w.sync_id)
            except MemoryFault as e:
                fault_lanes(e, lanes)
                raise
        return
    if space in ("generic", "global"):
        window = np.uint64(PARAM_WINDOW)
        param = mask & (addrs >= window) & (addrs < window + np.uint64(LOCAL_WINDOW_BYTES))
        if param.any():
            raise ExecutionError("kernel parameter memory is read-only")
    g, s, offsets, loc, local_offsets = _split_generic(space, addrs, mask)
    try:
        if g is not None and g.any():
            w.mem.write(addrs[g], value[g])
    except MemoryFault as e:
        fault_lanes(e, g)
        raise
    if loc is not None and loc.any():
        lanes = np.flatnonzero(loc)
        size = value.dtype.itemsize
        w.local[_local_index(w, ins, lanes, local_offsets, size, "write")] = \
            np.ascontiguousarray(value[loc]).view(np.uint8).reshape(len(lanes), size)
    if s is not None and s.any():
        try:
            if w.cta.smem.async_active:
                idx = w.cta.smem.index(offsets[s], value.dtype.itemsize, "write")
                async_check(w, ins, s, offsets, value.dtype.itemsize, "write to", writing=True)
                w.cta.smem.async_writer[idx] = 0  # overwritten by an ordinary store
            racy = w.cta.smem.write(offsets[s], value[s], w.sync_id)
        except MemoryFault as e:
            fault_lanes(e, s)
            raise
        if len(racy):
            w.cta.findings.add("smem-race",
                               "write to shared memory that another warp read or wrote with no "
                               "barrier in between", w, ins, np.flatnonzero(s)[racy])


def record_memory_banks(w, ins, space, addrs, mask, size: int) -> None:
    """Account for one complete scalar or vector instruction, not each element."""
    if space == "shared::cluster":
        return  # bank accounting across CTAs is not modeled
    if space == "generic":
        shared = (addrs >= GENERIC_WINDOW) & (addrs < GENERIC_WINDOW + CLUSTER_WINDOW)
        rank = (addrs - np.uint64(GENERIC_WINDOW)) // np.uint64(CLUSTER_STRIDE)
        mask = mask & ~(shared & (rank != w.cta.rank))
    in_global, shared, offsets, _, _ = _split_generic(space, addrs, mask)
    if shared is not None and shared.any():
        w.smem_waves += w.cta.smem_report.record(ins, w.program.fn.name, offsets[shared], size,
                                                 pc=w.stack.top.pc)
    if in_global is not None and in_global.any() and ins.opcode in ("ld", "atom", "red"):
        # For the timing model: a load makes the warp wait, a store does not; an atom loads.
        w.global_accesses += 1


def fault_lanes(e: MemoryFault, mask: np.ndarray) -> None:
    e.lanes = [int(x) for x in np.flatnonzero(mask)[e.which]]
