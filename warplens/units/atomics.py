"""Scalar atomic addition for ordinary CUDA atomicAdd kernels.

PTX atom/red: https://docs.nvidia.com/cuda/parallel-thread-execution/
#parallel-synchronization-and-communication-instructions-atom

The scheduler executes an entire instruction before another warp runs. Serializing
its active lanes therefore makes each read/modify/write indivisible. Lane order is
one deterministic permitted execution, not a prediction of GPU arbitration.
"""

import numpy as np

from ..errors import ExecutionError, MemoryFault
from ..frontend.ir import Imm, NP_TYPES, Reg, Sym
from .alu import _address, _async_check, _fault_lanes, _flush, _operands, op, route, unsupported
from .smem import CLUSTER_STRIDE, local_shared_offset

_TYPES = {"u32", "s32", "u64", "f32", "f64"}
_SPACES = {"global", "shared", "shared::cta"}
_SCOPES = {"cta", "gpu", "sys"}


def _form(ins):
    # nvcc and PTX examples use several orders for space/sem/scope; each field
    # must be unique. The operation and type still have their defined positions.
    mods = list(ins.mods)
    if not mods or mods[-1] not in _TYPES:
        raise unsupported(ins, "atomic add supports u32, s32, u64, f32 and f64")
    ty = mods.pop()
    noftz = bool(mods and mods[-1] == "noftz")
    if noftz:
        mods.pop()
        if ty != "f32":
            raise unsupported(ins, ".noftz is supported only with atomic add.f32")
    if not mods or mods.pop() != "add":
        raise unsupported(ins, "only scalar atomic .add is implemented")
    if len(mods) != len(set(mods)) or set(mods) - (_SPACES | _SCOPES | {"relaxed"}):
        raise unsupported(ins, "only relaxed atomic add with cta/gpu/sys scope and local shared/global space is implemented")
    spaces, scopes = set(mods) & _SPACES, set(mods) & _SCOPES
    if len(spaces) > 1 or len(scopes) > 1:
        raise unsupported(ins, "expected at most one state space and one scope")
    return ty, next(iter(spaces), "generic"), noftz


@op("atom", "red")
def _add(w, ins, mask):
    ty, space, noftz = _form(ins)
    if ins.opcode == "atom":
        dest, address, source = _operands(ins, 3)
        if not isinstance(dest, Reg) and not (isinstance(dest, Sym) and dest.name == "_"):
            raise unsupported(ins, "atom.add requires a scalar register or _ destination")
    else:
        address, source = _operands(ins, 2)
        dest = None
    if not isinstance(source, (Reg, Imm)):
        raise unsupported(ins, "atomic add requires a scalar register or immediate addend")
    values = w.read(source, ty)
    if isinstance(dest, Reg):
        w.read(dest, ty)  # validate destination width/type before changing memory
    addrs = _address(w, ins, address)
    dt = NP_TYPES[ty]
    lanes = np.flatnonzero(mask)
    if np.any(addrs[mask] % dt.itemsize):
        raise ExecutionError(f"`{ins.name}` requires addresses aligned to {dt.itemsize} bytes")
    shared = np.zeros(len(mask), dtype=bool)
    offsets = addrs.copy()
    for lane in lanes:
        if space in ("shared", "shared::cta"):
            shared[lane] = True
            offsets[lane] = local_shared_offset(addrs[lane])
        elif space == "generic":
            target, offset = route(int(addrs[lane]))
            if target == "shared":
                rank, offset = divmod(offset, CLUSTER_STRIDE)
                if rank != w.cta.rank:
                    raise unsupported(ins, "atomic access to another CTA's shared memory is not implemented")
                shared[lane] = True
                offsets[lane] = offset
            elif target != "global":
                raise unsupported(ins, "atomic addresses must refer to global or shared memory")
    global_lanes = mask & ~shared
    # Validate all active addresses before changing any memory. Re-read in the
    # loop: a later lane must see an earlier lane's update to the same address.
    try:
        if global_lanes.any():
            w.mem.read(addrs[global_lanes], dt)
    except MemoryFault as error:
        _fault_lanes(error, global_lanes)
        raise
    smem = w.cta.smem
    try:
        idx = smem.index(offsets[shared], dt.itemsize, "atomic access") if shared.any() else None
    except MemoryFault as error:
        _fault_lanes(error, shared)
        raise
    if shared.any() and smem.async_active:
        _async_check(w, ins, shared, offsets, dt.itemsize, "atomic access to", writing=True)
    old_values = np.zeros(len(mask), dtype=dt)
    uninitialized, races = [], []
    for lane in lanes:
        at = offsets[lane:lane + 1] if shared[lane] else addrs[lane:lane + 1]
        if shared[lane]:
            old, unwritten, racy = smem.read(at, dt, w.warp_id, atomic=True)
            if len(unwritten):
                uninitialized.append(lane)
            if len(racy):
                races.append(lane)
        else:
            old = w.mem.read(at, dt)
        old_values[lane] = old[0]
        value = values[lane:lane + 1]
        ftz = ty == "f32" and not shared[lane] and not noftz
        if ftz:
            old, value = _flush(old), _flush(value)
        with np.errstate(all="ignore"):
            updated = np.add(old, value, dtype=dt)
        if ftz:
            updated = _flush(updated)
        if shared[lane]:
            if len(smem.write(at, updated, w.warp_id, atomic=True)):
                races.append(lane)
        else:
            w.mem.write(at, updated)
    if shared.any() and smem.async_active:
        smem.async_writer[idx] = 0
    if uninitialized:
        w.cta.findings.add("smem-uninitialized-read", "atomic add reads shared memory that this CTA never wrote",
                          w, ins, uninitialized)
    if races:
        w.cta.findings.add("smem-race", "atomic add and another warp's ordinary access have no barrier in between",
                          w, ins, sorted(set(races)))
    if dest is not None:
        w.write(dest, old_values, mask)
    # Reuse ordinary global-load latency as a coarse estimate. Atomic contention,
    # shared atomic banks, throughput and host/system concurrency are not modeled.
    if global_lanes.any():
        w.global_accesses += 1
