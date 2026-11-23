"""TMA: asynchronous bulk copies between global and shared memory.

    cp.async.bulk.tensor.Nd  shared <- global   load a box; reports to an mbarrier
    cp.async.bulk.tensor.Nd  global <- shared   store a box; tracked by bulk groups
    cp.async.bulk            the same without a tensor map (raw byte ranges)
    cp.async.bulk.commit_group / wait_group[.read]

A TMA instruction only *issues* the copy. The copy completes a few scheduling
rounds later (`async_delay`), on the async proxy. Shadow state in shared
memory tracks which bytes an operation is writing or reading, and each warp
tracks which operations it has synchronized with, so unsynchronized use is
reported regardless of how the timing happened to work out.
"""

from __future__ import annotations

import numpy as np

from ..core.cta import Wait
from ..errors import ExecutionError, MemoryFault
from ..frontend.ir import Imm, TensorRef
from ..host import tensor_map as tm
from .alu import _address, op, unsupported

_LOAD_FLAGS = frozenset({"mbarrier::complete_tx::bytes", "L2::cache_hint", "multicast::cluster"})
_STORE_FLAGS = frozenset({"bulk_group", "L2::cache_hint"})
_UINT = {1: np.dtype(np.uint8), 2: np.dtype(np.uint16), 4: np.dtype(np.uint32), 8: np.dtype(np.uint64)}


def _scalar(w, ins, operand, lane: int, ty: str = "u32") -> int:
    return int(w.read(operand, ty)[lane])


def _smem_bytes(w, ins, start: int, nbytes: int, bits: int, verb: str) -> np.ndarray:
    """Shared-memory byte indices of a (possibly swizzled) buffer."""
    size = w.cta.smem.size
    # Reject impossible transfers before allocating an index for every byte.
    # Swizzling is a permutation, so a transfer larger than SMEM cannot fit.
    if nbytes > size or (nbytes and start >= size):
        raise ExecutionError(f"TMA {verb} of {nbytes} byte(s) at shared offset {start} runs past "
                             f"the CTA's {size} bytes of shared memory")
    idx = tm.swizzle_offsets(np.uint64(start) + np.arange(nbytes, dtype=np.uint64), bits)
    if len(idx) and int(idx.max()) >= size:
        raise ExecutionError(f"TMA {verb} of {nbytes} byte(s) at shared offset {start} runs past "
                             f"the CTA's {size} bytes of shared memory")
    return idx.astype(np.intp)


def _tensor_map(w, ins, operand: TensorRef, lane: int, nd: int):
    if not isinstance(operand, TensorRef):
        raise unsupported(ins, "expected a [tensor_map, {coordinates}] operand")
    if len(operand.coords) != nd:
        raise ExecutionError(f"{nd}d TMA instruction with {len(operand.coords)} coordinate(s)")
    addr = int(w.read_address(operand.base)[lane])
    raw = w.read_bytes(addr, tm.SIZE, lane)  # global memory, or a by-value kernel parameter
    info = tm.decode(raw)
    if info.rank != nd:
        raise ExecutionError(f"{nd}d TMA instruction used with a {info.rank}-dimensional tensor map")
    coords = [int(w.read(c, "s32")[lane]) for c in operand.coords]
    if coords[0] * info.elem_size % 16:
        raise ExecutionError("TMA innermost coordinate must give a 16-byte aligned bounding box "
                             f"(coordinate {coords[0]}, element size {info.elem_size})")
    return info, coords


def _check_alignment(w, ins, info, start: int, lane: int) -> None:
    if start % 16:
        raise ExecutionError(f"TMA needs a 16-byte aligned shared-memory address, got offset {start}")
    if info.swizzle and start % info.swizzle_period:
        name = tm.SWIZZLE_MODES[info.swizzle][0]
        w.cta.findings.add(
            "tma-swizzle-misaligned",
            f"shared buffer at offset {start} is not aligned to {info.swizzle_period} bytes, the "
            f"period of swizzle {name}. The swizzle is a function of the address, so code that "
            "indexes this buffer by offset will not find the data where it expects", w, ins, [lane])


def _load(w, ins, lane: int, start: int, idx: np.ndarray, data: np.ndarray, bar,
          swizzle: int | None = None, source=None, global_bytes: int | None = None,
          target=None) -> None:
    """Issue an asynchronous write of `data` to shared bytes `idx`, reporting to `bar`.
    `target` is the CTA whose shared memory is written, when it is not the issuer's."""
    cta = target or w.cta
    _require_seen(w, ins, [lane], cta.smem.async_writer[idx], "TMA load overwrites")
    for op_id in np.unique(cta.smem.async_reader[idx]):
        if op_id > 0 and int(op_id) not in w.seen:
            previous = cta.async_ops[int(op_id)]
            if previous.kind == "mma":
                cta.findings.add(
                    "mma-operand-overwritten",
                    f"TMA load overwrites shared memory that the tcgen05.mma at line "
                    f"{previous.ins.line} reads, without waiting on its tcgen05.commit mbarrier",
                    w, ins, [lane])
                continue
            cta.findings.add(
                "tma-store-source-overwritten",
                f"TMA load overwrites shared memory that the TMA store at line "
                f"{previous.ins.line} reads, without waiting for it "
                "(cp.async.bulk.wait_group.read)", w, ins, [lane])
    operation = cta.new_async("load", w, ins, lane)
    operation.swizzle = swizzle
    operation.work = {"bytes": len(idx),
                      "global_bytes": len(idx) if global_bytes is None else global_bytes,
                      "lines": None if source is None else np.unique(source >> np.uint64(7))}
    cta.smem.begin_async_write(idx, operation.id)
    bar.in_flight += 1

    def land():
        cta.smem.finish_async_write(idx, data)
        bar.in_flight -= 1
        bar.knowledge.add(operation.id)
        bar.complete_tx(len(idx), w, ins, [lane])

    cta.schedule(operation, land)


def _store(w, ins, lane: int, idx: np.ndarray, write_global) -> None:
    """Issue an asynchronous read of shared bytes `idx`; `write_global(bytes)` delivers them."""
    cta = w.cta
    smem = cta.smem
    if smem.has_uninitialized_bytes(idx):
        cta.findings.add(
            "smem-uninitialized-read", "TMA store reads shared memory that this CTA never wrote",
            w, ins, [lane])
    if smem.unfenced[idx].any():
        cta.findings.add(
            "tma-store-missing-fence",
            "TMA store of shared memory written by ordinary stores with no fence.proxy.async "
            "in between; TMA may read stale data", w, ins, [lane])
    _require_seen(w, ins, [lane], smem.async_writer[idx], "TMA store reads")
    operation = cta.new_async("store", w, ins, lane)
    operation.work = {"bytes": len(idx)}
    smem.async_reader[idx] = operation.id
    smem.async_active = True
    w.open_stores.append(operation)
    where = w._where(ins, np.arange(32) == lane, ExecutionError(""))

    def land():
        try:
            write_global(smem.data[idx])
        except MemoryFault as e:
            e.where = where
            raise

    cta.schedule(operation, land)


def _require_seen(w, ins, lanes, ids: np.ndarray, what: str) -> None:
    """Report use of bytes whose TMA load this warp has not synchronized with."""
    for op_id in np.unique(ids[ids > 0]):
        if int(op_id) not in w.seen:
            operation = w.cta.async_ops[int(op_id)]
            state = "is still in flight" if not operation.done else \
                "has completed, but this warp never waited on its mbarrier"
            w.cta.findings.add(
                "tma-load-unsynchronized",
                f"{what} shared memory that the TMA load at line {operation.ins.line} writes; "
                f"that load {state}", w, ins, lanes)


def _cp_async(w, ins, mask) -> None:
    """The per-thread asynchronous copy that predates TMA (Ampere and later):

        cp.async.{ca,cg}.shared.global [dst], [src], size[, src_size]
        cp.async.commit_group
        cp.async.wait_group N

    Each executing thread copies `size` bytes (4, 8 or 16) from global to shared memory.
    With `src_size`, only that many bytes are read and the rest are zero. Copies issued
    since the last commit form a group; `wait_group N` waits until at most N of the warp's
    groups are still pending, and only then may the data be read.
    """
    mods = ins.mods[1:]
    cta, ops = w.cta, ins.operands
    lanes = np.flatnonzero(mask)
    if not len(lanes):
        return
    if mods == ("commit_group",) and not ops:
        if w.cp_open:
            w.cp_groups.append(w.cp_open)
            w.cp_open = []
        return
    if mods == ("wait_group",) and len(ops) == 1:
        # Every group except the N most recently committed must be complete. Which groups
        # those are is fixed by commit order, not by which copies happen to finish first.
        newest = _scalar(w, ins, ops[0], int(lanes[0]))
        older = w.cp_groups[:len(w.cp_groups) - newest] if newest else w.cp_groups
        pending = [g for g in older if not all(o.done for o in g)]
        if pending:
            w.blocked = Wait("bulk group", ins, retry=True,
                             detail=lambda: f"{len(pending)} cp.async group(s) still in flight")
            return
        for group in older:
            w.seen.update(o.id for o in group)
        w.cp_groups = w.cp_groups[len(older):]
        return
    if len(mods) != 3 or mods[0] not in ("ca", "cg") or mods[1:] != ("shared", "global") \
            or len(ops) not in (3, 4):
        raise unsupported(ins, "expected cp.async.{ca,cg}.shared.global [dst], [src], size[, src_size]")
    size = _scalar(w, ins, ops[2], int(lanes[0]))
    if size not in (4, 8, 16):
        raise ExecutionError(f"cp.async copies 4, 8 or 16 bytes, not {size}")
    wanted = w.read(ops[3], "u32")[lanes].astype(np.int64) if len(ops) == 4 else np.full(len(lanes), size)
    if (wanted > size).any():
        raise ExecutionError("cp.async source size is larger than the copy size")
    span = np.arange(size, dtype=np.uint64)
    source = _address(w, ins, ops[1])[lanes]
    data = np.zeros((len(lanes), size), dtype=np.uint8)
    reads = span[None, :] < wanted[:, None].astype(np.uint64)  # bytes actually read; the rest are zero
    try:
        data[reads] = w.mem.read((source[:, None] + span[None, :])[reads], _UINT[1])
    except MemoryFault as e:
        e.lanes = sorted({int(lanes[k]) for k in np.flatnonzero(reads.any(axis=1))})
        raise
    offsets = _address(w, ins, ops[0])[lanes]
    try:
        idx = cta.smem.index(offsets, size, "write")
    except MemoryFault as e:
        e.lanes = [int(lanes[k]) for k in e.which]
        raise
    operation = cta.new_async("copy", w, ins, int(lanes[0]))
    operation.work = {"bytes": int(reads.sum()), "lines": None}
    cta.smem.begin_async_write(idx.reshape(-1), operation.id)
    w.cp_open.append(operation)

    def land():
        cta.smem.finish_async_write(idx.reshape(-1), data.reshape(-1))

    cta.schedule(operation, land)


@op("cp")
def _cp(w, ins, mask):
    mods = ins.mods
    if mods[:1] == ("async",) and mods[1:2] != ("bulk",):
        return _cp_async(w, ins, mask)
    if mods[:2] != ("async", "bulk"):
        raise unsupported(ins)
    rest = list(mods[2:])
    lanes = [int(l) for l in np.flatnonzero(mask)]
    if not lanes:
        return
    cta, ops = w.cta, ins.operands

    if rest == ["commit_group"]:
        if ops:
            raise unsupported(ins, "commit_group takes no operands")
        for lane in lanes:
            group = [o for o in w.open_stores if o.lane == lane]
            # An empty commit still occupies a position among the newest groups.
            w.groups.setdefault(lane, []).append(group)
        w.open_stores = [o for o in w.open_stores if not mask[o.lane]]
        return
    if rest in (["wait_group"], ["wait_group", "read"]):
        if len(ops) != 1:
            raise unsupported(ins, "expected a group count")
        count = ops[0]
        if not isinstance(count, Imm) or count.kind != "int" or count.value < 0:
            raise ExecutionError("cp.async.bulk.wait_group needs a nonnegative integer constant")
        allowed = count.value
        older = {lane: w.groups.get(lane, [])[:max(0, len(w.groups.get(lane, [])) - allowed)]
                 for lane in lanes}
        pending = sum(not all(o.done for o in group)
                      for groups in older.values() for group in groups)
        if pending:
            w.blocked = Wait("bulk group", ins, retry=True,
                             detail=lambda: f"{pending} TMA store group(s) still in flight")
            return
        for lane, groups in older.items():
            for group in groups:
                w.seen.update(o.id for o in group)
            w.groups[lane] = w.groups.get(lane, [])[len(groups):]
        return
    if rest and rest[0] == "prefetch":
        return  # a cache hint; no functional effect

    nd = 0
    if rest and rest[0] == "tensor":
        if len(rest) < 2 or rest[1] not in ("1d", "2d", "3d", "4d", "5d"):
            raise unsupported(ins)
        nd = int(rest[1][0])
        rest = rest[2:]
    cta_group = next((m for m in rest if m.startswith("cta_group::")), None)
    if cta_group is not None:  # a load whose completion is reported for a pair of CTAs
        rest.remove(cta_group)
        if cta_group not in ("cta_group::1", "cta_group::2"):
            raise unsupported(ins, f"TMA with .{cta_group}")
    if len(rest) < 2:
        raise unsupported(ins)
    dst, src, *flags = rest
    to_shared = dst.startswith("shared") and src == "global"
    to_global = dst == "global" and src.startswith("shared")
    if not (to_shared or to_global):
        raise unsupported(ins)
    if nd and "tile" in flags:
        flags.remove("tile")  # explicitly naming the default tensor copy mode
    extra = set(flags) - (_LOAD_FLAGS if to_shared else _STORE_FLAGS)
    if extra:
        raise unsupported(ins, f"modifier .{sorted(extra)[0]}")
    needed = "mbarrier::complete_tx::bytes" if to_shared else "bulk_group"
    if needed not in flags:
        raise unsupported(ins, f"expected .{needed}")
    hint = 1 if "L2::cache_hint" in flags else 0

    for lane in lanes:
        if to_shared and nd:  # [smem], [tensor_map, {coords}], [mbarrier]
            multicast = "multicast::cluster" in flags
            if len(ops) != 3 + hint + multicast:
                raise unsupported(ins, "expected [dst], [tensor_map, {coords}], [mbarrier][, mask]")
            start = int(_address(w, ins, ops[0])[lane])
            info, coords = _tensor_map(w, ins, ops[1], lane, nd)
            bar_offset = int(_address(w, ins, ops[2])[lane])
            # With multicast, the same box lands at the same offset in every CTA of the
            # cluster whose rank is set in the mask, and reports to each one's mbarrier.
            targets = [cta]
            if multicast:
                ranks = int(w.read_address(ops[3])[lane]) & 0xFFFF  # a 16-bit operand
                targets = [cta.peer(r) for r in range(16) if ranks >> r & 1]
            _check_alignment(w, ins, info, start, lane)
            idx = _smem_bytes(w, ins, start, info.box_bytes, info.swizzle, "load")
            addrs, inside = tm.element_addresses(info, coords)
            dt = _UINT[info.elem_size]
            values = np.zeros(len(addrs), dtype=dt)
            if info.nan_fill:
                values[:] = np.array(np.nan, dtype=f"f{info.elem_size}").view(dt)
            values[inside] = w.mem.read(addrs[inside], dt)
            for target in targets:
                # A .cta_group::2 load reports to the mbarrier of the even-ranked CTA of the
                # receiving pair, which is the one that waits and issues the MMA.
                reports_to = target.peer(target.rank & ~1) if cta_group == "cta_group::2" else target
                _load(w, ins, lane, start, idx, values.view(np.uint8),
                      reports_to.mbarrier(bar_offset, w, ins, [lane]), swizzle=info.swizzle,
                      source=addrs[inside], global_bytes=int(inside.sum()) * info.elem_size,
                      target=target)
        elif to_shared:  # [smem], [global], size, [mbarrier]
            if len(ops) != 4 + hint:
                raise unsupported(ins, "expected [dst], [src], size, [mbarrier]")
            start = int(_address(w, ins, ops[0])[lane])
            source = int(_address(w, ins, ops[1])[lane])
            nbytes = _raw_size(w, ins, ops[2], lane, start, source)
            bar = cta.mbarrier(int(_address(w, ins, ops[3])[lane]), w, ins, [lane])
            idx = _smem_bytes(w, ins, start, nbytes, 0, "load")
            addrs = np.uint64(source) + np.arange(nbytes, dtype=np.uint64)
            data = w.mem.read(addrs, _UINT[1])
            _load(w, ins, lane, start, idx, data, bar, source=addrs)
        elif nd:  # [tensor_map, {coords}], [smem]
            if len(ops) != 2 + hint:
                raise unsupported(ins, "expected [tensor_map, {coords}], [src]")
            info, coords = _tensor_map(w, ins, ops[0], lane, nd)
            start = int(_address(w, ins, ops[1])[lane])
            _check_alignment(w, ins, info, start, lane)
            idx = _smem_bytes(w, ins, start, info.box_bytes, info.swizzle, "store")
            addrs, inside = tm.element_addresses(info, coords)
            dt = _UINT[info.elem_size]

            def write_global(data, addrs=addrs, inside=inside, dt=dt):
                w.mem.write(addrs[inside], np.ascontiguousarray(data).view(dt)[inside])

            _store(w, ins, lane, idx, write_global)
        else:  # [global], [smem], size
            if len(ops) != 3 + hint:
                raise unsupported(ins, "expected [dst], [src], size")
            target = int(_address(w, ins, ops[0])[lane])
            start = int(_address(w, ins, ops[1])[lane])
            nbytes = _raw_size(w, ins, ops[2], lane, start, target)
            idx = _smem_bytes(w, ins, start, nbytes, 0, "store")

            def write_global(data, target=target, nbytes=nbytes):
                w.mem.write(np.uint64(target) + np.arange(nbytes, dtype=np.uint64), data)

            _store(w, ins, lane, idx, write_global)


def _raw_size(w, ins, operand, lane: int, *addresses: int) -> int:
    nbytes = _scalar(w, ins, operand, lane)
    if nbytes % 16 or any(a % 16 for a in addresses):
        raise ExecutionError("cp.async.bulk needs 16-byte aligned addresses and a size that is a "
                             f"multiple of 16 (size {nbytes}, addresses {[hex(a) for a in addresses]})")
    return nbytes


@op("prefetch")
def _prefetch(w, ins, mask):
    if ins.mods != ("tensormap",):
        raise unsupported(ins)
