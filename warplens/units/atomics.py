"""Scalar atomic read/modify/write operations, and scalar red.add.

PTX rules: https://docs.nvidia.com/cuda/parallel-thread-execution/
#parallel-synchronization-and-communication-instructions-atom

The scheduler executes an entire instruction before another warp runs. Serializing
its active lanes therefore makes each read/modify/write indivisible. Lane order is
one deterministic permitted execution, not a prediction of GPU arbitration.
"""

import numpy as np

from ..errors import ExecutionError, MemoryFault
from ..frontend.ir import Imm, NP_TYPES, Reg, Sym
from .alu import (_address, _async_check, _fault_lanes, _flush, _operands,
                  _record_memory_banks, _remote_id, op, route, unsupported)
from .smem import CLUSTER_STRIDE, local_shared_offset

_TYPES = {
    "add": {"u32", "s32", "u64", "f32", "f64"},
    "and": {"b32", "b64"}, "or": {"b32", "b64"}, "xor": {"b32", "b64"},
    "exch": {"b32", "b64"}, "cas": {"b32", "b64"},
    "inc": {"u32"}, "dec": {"u32"},
    "min": {"s32", "u32", "s64", "u64"}, "max": {"s32", "u32", "s64", "u64"},
}
_SPACES = {"global", "shared", "shared::cta"}
_SCOPES = {"cta", "gpu", "sys"}


def _form(ins):
    # nvcc and PTX examples use several orders for space/sem/scope; each field
    # must be unique. Operation, optional noftz and type follow these qualifiers.
    mods = list(ins.mods)
    if len(mods) < 2:
        raise unsupported(ins, "expected a scalar atomic operation and type")
    ty = mods.pop()
    noftz = bool(mods and mods[-1] == "noftz")
    if noftz:
        mods.pop()
    operation = mods.pop() if mods else ""
    if ty not in _TYPES.get(operation, ()) or ins.opcode == "red" and operation != "add":
        raise unsupported(ins, "unsupported scalar atomic operation/type; red supports add only")
    if noftz and (operation, ty) != ("add", "f32"):
        raise unsupported(ins, ".noftz is supported only with atomic add.f32")
    if len(mods) != len(set(mods)) or set(mods) - (_SPACES | _SCOPES | {"relaxed"}):
        raise unsupported(ins, "only relaxed atomics with cta/gpu/sys scope and shared/global space are implemented")
    spaces, scopes = set(mods) & _SPACES, set(mods) & _SCOPES
    if len(spaces) > 1 or len(scopes) > 1:
        raise unsupported(ins, "expected at most one state space and one scope")
    return operation, ty, next(iter(spaces), "generic"), noftz


def _result(operation, old, value, replacement):
    if operation == "add":
        with np.errstate(all="ignore"):
            return np.add(old, value, dtype=old.dtype)
    if operation == "exch":
        return value
    if operation == "cas":
        return np.where(old == value, replacement, old)
    if operation == "inc":
        return np.where(old >= value, 0, old + old.dtype.type(1)).astype(old.dtype)
    if operation == "dec":
        return np.where((old == 0) | (old > value), value, old - old.dtype.type(1)).astype(old.dtype)
    return {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
            "min": np.minimum, "max": np.maximum}[operation](old, value)


@op("atom", "red")
def _atomic(w, ins, mask):
    operation, ty, space, noftz = _form(ins)
    if ins.opcode == "atom":
        dest, address, *sources = _operands(ins, 4 if operation == "cas" else 3)
        if not isinstance(dest, Reg) and not (isinstance(dest, Sym) and dest.name == "_"):
            raise unsupported(ins, "atom requires a scalar register or _ destination")
    else:
        address, source = _operands(ins, 2)
        dest, sources = None, [source]
    if any(not isinstance(source, (Reg, Imm)) for source in sources):
        raise unsupported(ins, "atomics require scalar register or immediate operands")
    values = w.read(sources[0], ty)
    replacement = w.read(sources[1], ty) if operation == "cas" else None
    if isinstance(dest, Reg):
        w.read(dest, ty)  # validate destination width/type before changing memory
    addrs = _address(w, ins, address)
    dt = NP_TYPES[ty]
    lanes = np.flatnonzero(mask)
    if np.any(addrs[mask] % dt.itemsize):
        raise ExecutionError(f"`{ins.name}` requires addresses aligned to {dt.itemsize} bytes")
    targets = [None] * len(mask)  # None means global, otherwise the shared-memory CTA
    offsets = addrs.copy()
    for lane in lanes:
        if space in ("shared", "shared::cta"):
            targets[lane] = w.cta
            offsets[lane] = local_shared_offset(addrs[lane])
        elif space == "generic":
            target, offset = route(int(addrs[lane]))
            if target == "shared":
                rank, offset = divmod(offset, CLUSTER_STRIDE)
                targets[lane] = w.cta.peer(rank)
                offsets[lane] = offset
            elif target != "global":
                what = "local memory" if target == "local" else "kernel parameters"
                raise ExecutionError(f"`{ins.name}` on {what}: PTX defines atomic operations on global "
                                     "and shared memory only")
    global_lanes = mask & np.asarray([target is None for target in targets])
    # Validate all active addresses before changing any memory. Re-read in the
    # loop: a later lane must see an earlier lane's update to the same address.
    try:
        if global_lanes.any():
            w.mem.read(addrs[global_lanes], dt)
    except MemoryFault as error:
        _fault_lanes(error, global_lanes)
        raise
    for target in {target for target in targets if target is not None}:
        selected = mask & np.asarray([item is target for item in targets])
        try:
            target.smem.index(offsets[selected], dt.itemsize, "atomic access")
        except MemoryFault as error:
            _fault_lanes(error, selected)
            raise
        if target is w.cta and target.smem.async_active:
            _async_check(w, ins, selected, offsets, dt.itemsize, "atomic access to", writing=True)
    old_values = np.zeros(len(mask), dtype=dt)
    uninitialized, races = [], []
    for lane in lanes:
        target = targets[lane]
        at = offsets[lane:lane + 1] if target is not None else addrs[lane:lane + 1]
        value = values[lane:lane + 1]
        change = replacement[lane:lane + 1] if replacement is not None else None
        ftz = ty == "f32" and target is None and not noftz

        def update(old):
            before, operand = (_flush(old), _flush(value)) if ftz else (old, value)
            updated = _result(operation, before, operand, change)
            return _flush(updated) if ftz else updated

        if target is not None:
            writer = w.warp_id if target is w.cta else _remote_id(w)
            old, unwritten, racy = target.smem.atomic(at, dt, writer, update)
            if len(unwritten):
                uninitialized.append(lane)
            if len(racy):
                races.append(lane)
            if target is w.cta and target.smem.async_active:
                target.smem.async_writer[target.smem.index(at, dt.itemsize, "atomic write")] = 0
        else:
            old = w.mem.read(at, dt)
            w.mem.write(at, update(old))
        old_values[lane] = old[0]
    if uninitialized:
        w.cta.findings.add("smem-uninitialized-read", "atomic update reads shared memory never written by its CTA",
                          w, ins, uninitialized)
    if races:
        w.cta.findings.add("smem-race", "atomic update and another warp's ordinary load or store have no barrier in between",
                          w, ins, sorted(set(races)))
    if dest is not None:
        w.write(dest, old_values, mask)
    # Preserve the assumed load+store timing model and ordinary-bank accounting.
    # It does not predict atomic throughput or contention; see docs/atomics.md.
    waves = w.smem_waves
    _record_memory_banks(w, ins, space, addrs, mask, dt.itemsize)
    w.smem_waves += w.smem_waves - waves
