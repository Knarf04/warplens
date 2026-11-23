"""Functional semantics of the core PTX instructions.

Every handler runs one instruction on all 32 lanes of a warp at once and
writes its result only to the lanes in `mask`. Handlers reject modifiers they
do not implement instead of ignoring them.
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError, MemoryFault, UnsupportedError
from ..frontend.ir import Instr, Mem, NP_TYPES, Pair, Sym, UINT_BY_SIZE, Vec
from .floating import fused_multiply_add
from .smem import GENERIC_WINDOW, MAX_SMEM_BYTES

HANDLERS: dict = {}

_ROUND_NEAREST = frozenset({"rn"})


def op(*names):
    def register(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return register


def unsupported(ins: Instr, why: str = "") -> UnsupportedError:
    detail = f" ({why})" if why else ""
    return UnsupportedError(f"unsupported instruction `{ins.name}`{detail}\n  line {ins.line}: {ins.text}")


def _type(ins: Instr, allowed=frozenset(), kinds: str = "iuf"):
    """Split modifiers into (type, extras). `kinds` lists accepted NumPy dtype kinds."""
    if not ins.mods or ins.mods[-1] not in NP_TYPES:
        raise unsupported(ins, "missing or unknown type")
    *extra, ty = ins.mods
    if NP_TYPES[ty].kind not in kinds:
        raise unsupported(ins, f"type .{ty} is not valid here")
    for e in extra:
        if e not in allowed:
            raise unsupported(ins, f"modifier .{e}")
    return ty, extra


def _operands(ins: Instr, n: int):
    if len(ins.operands) != n:
        raise unsupported(ins, f"expected {n} operands, found {len(ins.operands)}")
    return ins.operands


def _is_float(ty: str) -> bool:
    return NP_TYPES[ty].kind == "f"


def _wide(ty: str) -> str:
    if ty not in ("u16", "s16", "u32", "s32"):
        raise ValueError(ty)
    return f"{ty[0]}{int(ty[1:]) * 2}"


def _mul_hi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dt = a.dtype
    bits = dt.itemsize * 8
    if bits <= 32:
        big = np.int64 if dt.kind == "i" else np.uint64
        return ((a.astype(big) * b.astype(big)) >> bits).astype(dt)
    hi = [(int(x) * int(y)) >> 64 for x, y in zip(a, b)]
    return np.array([h & 0xFFFF_FFFF_FFFF_FFFF for h in hi], dtype=np.uint64).view(dt)


# -- moves ---------------------------------------------------------------------

@op("mov")
def _mov(w, ins, mask):
    ty, _ = _type(ins, kinds="iufb")
    d, a = _operands(ins, 2)
    if isinstance(a, Vec) or isinstance(d, Vec):
        # Pack {lo, ..., hi} into one register, or unpack one register into several.
        # The first element is the least significant (checked on real hardware).
        elems = (a if isinstance(a, Vec) else d).elems
        total = NP_TYPES[ty].itemsize
        if isinstance(a, Vec) and isinstance(d, Vec) or total % max(len(elems), 1) \
                or total // len(elems) not in UINT_BY_SIZE or NP_TYPES[ty].kind == "b":
            raise unsupported(ins, "vector mov must pack or unpack equal power-of-two parts")
        part = UINT_BY_SIZE[total // len(elems)]
        whole = UINT_BY_SIZE[total]
        bits = np.uint64(8 * part.itemsize)
        if isinstance(a, Vec):
            value = np.zeros(len(mask), dtype=np.uint64)
            for i, e in enumerate(elems):
                value |= w.read(e, f"b{8 * part.itemsize}").astype(np.uint64) << (bits * np.uint64(i))
            w.write(d, value.astype(whole), mask)
        else:
            value = w.read(a, ty).view(whole).astype(np.uint64)
            for i, e in enumerate(elems):
                w.write(e, (value >> (bits * np.uint64(i))).astype(part), mask)
        return
    w.write(d, w.read(a, ty), mask)


@op("cvta")
def _cvta(w, ins, mask):
    # Address-space conversion. Global addresses are already generic addresses;
    # shared addresses are offsets that sit in a fixed window of the generic space.
    ty, extra = _type(ins, allowed={"to", "global", "shared", "local"}, kinds="u")
    spaces = [e for e in extra if e != "to"]
    if len(spaces) != 1 or (ty != "u64" and spaces[0] != "global"):
        raise unsupported(ins)
    d, a = _operands(ins, 2)
    value = w.read(a, ty)
    if spaces[0] != "global":
        window = np.uint64(GENERIC_WINDOW if spaces[0] == "shared" else LOCAL_WINDOW)
        value = value - window if "to" in extra else value + window
    w.write(d, value, mask)


# -- integer and floating-point arithmetic ---------------------------------------

@op("add", "sub")
def _add_sub(w, ins, mask):
    ty, _ = _type(ins, allowed=_ROUND_NEAREST if _is_float(ins.mods[-1]) else frozenset())
    d, a, b = _operands(ins, 3)
    x, y = w.read(a, ty), w.read(b, ty)
    w.write(d, x + y if ins.opcode == "add" else x - y, mask)


def _mul_parts(w, ins, a, b):
    """Shared by mul and mad: returns (product, type the product has)."""
    last = ins.mods[-1] if ins.mods else ""
    if last in NP_TYPES and _is_float(last):
        ty, _ = _type(ins, allowed=_ROUND_NEAREST)
        return w.read(a, ty) * w.read(b, ty), ty
    ty, extra = _type(ins, allowed={"lo", "hi", "wide"}, kinds="iu")
    if len(extra) != 1:
        raise unsupported(ins, "integer multiply needs exactly one of .lo/.hi/.wide")
    x, y = w.read(a, ty), w.read(b, ty)
    if extra[0] == "lo":
        return x * y, ty
    if extra[0] == "hi":
        return _mul_hi(x, y), ty
    try:
        wide = _wide(ty)
    except ValueError:
        raise unsupported(ins, ".wide needs a 16- or 32-bit type") from None
    dt = NP_TYPES[wide]
    return x.astype(dt) * y.astype(dt), wide


@op("mul")
def _mul(w, ins, mask):
    d, a, b = _operands(ins, 3)
    prod, _ = _mul_parts(w, ins, a, b)
    w.write(d, prod, mask)


@op("mad")
def _mad(w, ins, mask):
    if ins.mods and ins.mods[-1] in NP_TYPES and _is_float(ins.mods[-1]):
        # On the supported sm_20+ targets, floating mad is an alias for fma.
        return _fma(w, ins, mask)
    d, a, b, c = _operands(ins, 4)
    prod, ty = _mul_parts(w, ins, a, b)
    w.write(d, prod + w.read(c, ty), mask)


@op("fma")
def _fma(w, ins, mask):
    ty, _ = _type(ins, allowed=_ROUND_NEAREST, kinds="f")
    d, a, b, c = _operands(ins, 4)
    x, y, z = w.read(a, ty), w.read(b, ty), w.read(c, ty)
    w.write(d, fused_multiply_add(x, y, z), mask)


@op("div", "rem")
def _div_rem(w, ins, mask):
    last = ins.mods[-1] if ins.mods else ""
    if last in NP_TYPES and _is_float(last):
        if ins.opcode == "rem":
            raise unsupported(ins, "rem is integer-only")
        ty, _ = _type(ins, allowed={"rn", "full", "approx"})
        d, a, b = _operands(ins, 3)
        w.write(d, w.read(a, ty) / w.read(b, ty), mask)
        return
    ty, _ = _type(ins, kinds="iu")
    d, a, b = _operands(ins, 3)
    x, y = w.read(a, ty), w.read(b, ty)
    y = np.where(y == 0, 1, y).astype(x.dtype)  # division by zero is undefined in PTX
    q = x // y
    r = x - q * y
    if x.dtype.kind == "i":
        # NumPy floors; PTX truncates toward zero like C.
        fix = (r != 0) & ((x < 0) != (y < 0))
        q = q + fix.astype(x.dtype)
        r = x - q * y
    w.write(d, q if ins.opcode == "div" else r, mask)


@op("min", "max")
def _min_max(w, ins, mask):
    ty, _ = _type(ins)
    d, a, b = _operands(ins, 3)
    x, y = w.read(a, ty), w.read(b, ty)
    if _is_float(ty):
        fn = np.fmin if ins.opcode == "min" else np.fmax  # a NaN operand yields the other one
        result = fn(x, y)
        both_zero = (x == 0) & (y == 0)
        negative = (np.signbit(x) | np.signbit(y)) if ins.opcode == "min" else \
            (np.signbit(x) & np.signbit(y))
        result[both_zero] = np.copysign(np.zeros_like(x), np.where(negative, -1, 1))[both_zero]
    else:
        fn = np.minimum if ins.opcode == "min" else np.maximum
        result = fn(x, y)
    w.write(d, result, mask)


@op("neg", "abs")
def _neg_abs(w, ins, mask):
    ty, _ = _type(ins, kinds="if")
    d, a = _operands(ins, 2)
    x = w.read(a, ty)
    w.write(d, -x if ins.opcode == "neg" else np.abs(x), mask)


@op("sqrt")
def _sqrt(w, ins, mask):
    ty, _ = _type(ins, allowed={"rn", "approx"}, kinds="f")
    d, a = _operands(ins, 2)
    w.write(d, np.sqrt(w.read(a, ty)), mask)


# -- bitwise ---------------------------------------------------------------------

@op("and", "or", "xor")
def _bitwise(w, ins, mask):
    ty, _ = _type(ins, kinds="ub")
    d, a, b = _operands(ins, 3)
    x, y = w.read(a, ty), w.read(b, ty)
    w.write(d, x & y if ins.opcode == "and" else x | y if ins.opcode == "or" else x ^ y, mask)


@op("not")
def _not(w, ins, mask):
    ty, _ = _type(ins, kinds="ub")
    d, a = _operands(ins, 2)
    w.write(d, ~w.read(a, ty), mask)


@op("shl", "shr")
def _shift(w, ins, mask):
    ty, _ = _type(ins, kinds="iu")
    d, a, b = _operands(ins, 3)
    x = w.read(a, ty)
    width = x.dtype.itemsize * 8
    amount = w.read(b, "u32")
    over = amount >= width  # PTX clamps shift amounts to the register width
    sh = np.where(over, 0, amount).astype(x.dtype)
    if ins.opcode == "shl":
        r = np.where(over, 0, x << sh)
    elif x.dtype.kind == "i":
        r = np.where(over, x >> (width - 1), x >> sh)
    else:
        r = np.where(over, 0, x >> sh)
    w.write(d, r.astype(x.dtype), mask)


# -- comparison and selection ------------------------------------------------------

def _compare(cmp: str, x: np.ndarray, y: np.ndarray):
    kind = x.dtype.kind
    if cmp in ("lo", "ls", "hi", "hs"):
        if kind == "f":
            return None
        u = UINT_BY_SIZE[x.dtype.itemsize]
        x, y = x.view(u), y.view(u)
        return {"lo": x < y, "ls": x <= y, "hi": x > y, "hs": x >= y}[cmp]
    ordered = {"eq": np.equal, "ne": np.not_equal, "lt": np.less,
               "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal}
    if cmp in ordered:
        r = ordered[cmp](x, y)
        if kind == "f" and cmp == "ne":
            r &= ~(np.isnan(x) | np.isnan(y))  # ordered: false if either is NaN
        return r
    if kind != "f":
        return None
    nan = np.isnan(x) | np.isnan(y)
    if cmp == "num":
        return ~nan
    if cmp == "nan":
        return nan
    if cmp.endswith("u") and cmp[:-1] in ordered:
        return ordered[cmp[:-1]](x, y) | nan
    return None


@op("setp")
def _setp(w, ins, mask):
    if len(ins.mods) != 2:
        raise unsupported(ins, "only the plain `setp.cmp.type` form is implemented")
    cmp = ins.mods[0]
    ty, _ = _type(ins, allowed={cmp})
    d, a, b = _operands(ins, 3)
    r = _compare(cmp, w.read(a, ty), w.read(b, ty))
    if r is None:
        raise unsupported(ins, f"comparison .{cmp} on .{ty}")
    w.write(d, r, mask)


@op("selp")
def _selp(w, ins, mask):
    ty, _ = _type(ins)
    d, a, b, c = _operands(ins, 4)
    w.write(d, np.where(w.read(c, "pred"), w.read(a, ty), w.read(b, ty)), mask)


# -- conversion --------------------------------------------------------------------

_FLOAT_TO_INT = {"rni": np.rint, "rzi": np.trunc, "rmi": np.floor, "rpi": np.ceil}


@op("cvt")
def _cvt(w, ins, mask):
    if len(ins.mods) < 2 or ins.mods[-1] not in NP_TYPES or ins.mods[-2] not in NP_TYPES:
        raise unsupported(ins, "expected cvt[.round].dtype.atype")
    *extra, dty, aty = ins.mods
    dd, da = NP_TYPES[dty], NP_TYPES[aty]
    if "b" in (dd.kind, da.kind):
        raise unsupported(ins, "predicates cannot be converted")
    d, a = _operands(ins, 2)
    x = w.read(a, aty)
    if da.kind == "f" and dd.kind in "iu":
        if len(extra) != 1 or extra[0] not in _FLOAT_TO_INT:
            raise unsupported(ins, "float-to-int needs one of .rni/.rzi/.rmi/.rpi")
        r = _FLOAT_TO_INT[extra[0]](x.astype(np.float64))
        info = np.iinfo(dd)
        nan = np.isnan(r)
        high = r >= float(info.max)
        low = r <= float(info.min)
        mid = ~(nan | high | low)
        out = np.zeros(len(r), dtype=dd)
        if aty == "f64" or dty in ("s64", "u64"):
            # PTX uses the sign-bit sentinel for these NaN conversions.
            out[nan] = info.min if dd.kind == "i" else 1 << (dd.itemsize * 8 - 1)
        out[mid] = r[mid].astype(dd)
        out[high] = info.max  # float-to-int saturates
        out[low] = info.min
        w.write(d, out, mask)
        return
    if extra and not (dd.kind == "f" and set(extra) <= _ROUND_NEAREST):
        raise unsupported(ins, f"modifier .{extra[0]}")
    w.write(d, x.astype(dd), mask)


# -- memory ------------------------------------------------------------------------

def _address(w, ins, mem) -> np.ndarray:
    if not isinstance(mem, Mem):
        raise unsupported(ins, "expected a [address] operand")
    base = w.read_address(mem.base)
    return base + np.uint64(mem.offset % (1 << 64)) if mem.offset else base


LOCAL_WINDOW = 0x1_4000_0000  # where per-thread local memory appears in the generic space
_LOCAL_WINDOW_BYTES = 1 << 24


def route(address: int) -> tuple[str, int]:
    """Which memory a generic address refers to, and the address within it."""
    if GENERIC_WINDOW <= address < GENERIC_WINDOW + MAX_SMEM_BYTES:
        return "shared", address - GENERIC_WINDOW
    if LOCAL_WINDOW <= address < LOCAL_WINDOW + _LOCAL_WINDOW_BYTES:
        return "local", address - LOCAL_WINDOW
    return "global", address


def _space(ins: Instr):
    """(type, address space, vector length) of a load or store."""
    ty, extra = _type(ins, allowed={"global", "shared", "param", "local", "v2", "v4"})
    width = 1
    for v, n in (("v2", 2), ("v4", 4)):
        if v in extra:
            extra.remove(v)
            width = n
    if len(extra) > 1:
        raise unsupported(ins)
    return ty, (extra[0] if extra else "generic"), width


def _split_generic(space: str, addrs: np.ndarray, mask: np.ndarray):
    """(lanes addressing global memory, lanes addressing shared memory, shared offsets,
    lanes addressing local memory, local offsets)."""
    if space == "global":
        return mask, None, None, None, None
    if space == "shared":
        return None, mask, addrs, None, None
    if space == "local":
        return None, None, None, mask, addrs
    window = np.uint64(GENERIC_WINDOW)
    in_smem = (addrs >= window) & (addrs < window + np.uint64(MAX_SMEM_BYTES))
    local = np.uint64(LOCAL_WINDOW)
    in_local = (addrs >= local) & (addrs < local + np.uint64(_LOCAL_WINDOW_BYTES))
    to_smem, to_local = mask & in_smem, mask & in_local
    return (mask & ~in_smem & ~in_local, (to_smem if to_smem.any() else None), addrs - window,
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


def _load(w, ins, space, addrs, dt, mask) -> np.ndarray:
    value = np.zeros(len(mask), dtype=dt)
    g, s, offsets, loc, local_offsets = _split_generic(space, addrs, mask)
    try:
        if g is not None and g.any():
            value[g] = w.mem.read(addrs[g], dt)
    except MemoryFault as e:
        _fault_lanes(e, g)
        raise
    if loc is not None and loc.any():
        lanes = np.flatnonzero(loc)
        value[loc] = w.local[_local_index(w, ins, lanes, local_offsets, dt.itemsize, "read")] \
            .view(dt).ravel()
    if s is not None and s.any():
        cta = w.cta
        try:
            value[s], unwritten, racy = cta.smem.read(offsets[s], dt, w.warp_id)
        except MemoryFault as e:
            _fault_lanes(e, s)
            raise
        if cta.smem.async_active:
            _async_check(w, ins, s, offsets, dt.itemsize, "read of", writing=False)
        cta.smem_report.record(ins, w.program.fn.name, offsets[s], dt.itemsize,
                               pc=w.stack.top.pc)
        if len(unwritten):
            cta.findings.add("smem-uninitialized-read",
                             "read of shared memory that this CTA never wrote",
                             w, ins, np.flatnonzero(s)[unwritten])
        if len(racy):
            cta.findings.add("smem-race",
                             "read of shared memory that another warp wrote with no barrier "
                             "in between", w, ins, np.flatnonzero(s)[racy])
    return value


def _async_check(w, ins, s, offsets, size: int, what: str, writing: bool) -> None:
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


def _store(w, ins, space, addrs, value, mask) -> None:
    g, s, offsets, loc, local_offsets = _split_generic(space, addrs, mask)
    try:
        if g is not None and g.any():
            w.mem.write(addrs[g], value[g])
    except MemoryFault as e:
        _fault_lanes(e, g)
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
                _async_check(w, ins, s, offsets, value.dtype.itemsize, "write to", writing=True)
                w.cta.smem.async_writer[idx] = 0  # overwritten by an ordinary store
            racy = w.cta.smem.write(offsets[s], value[s], w.warp_id)
        except MemoryFault as e:
            _fault_lanes(e, s)
            raise
        if len(racy):
            w.cta.findings.add("smem-race",
                               "write to shared memory that another warp read or wrote with no "
                               "barrier in between", w, ins, np.flatnonzero(s)[racy])
        w.cta.smem_report.record(ins, w.program.fn.name, offsets[s], value.dtype.itemsize,
                                 pc=w.stack.top.pc)


def _fault_lanes(e: MemoryFault, mask: np.ndarray) -> None:
    e.lanes = [int(x) for x in np.flatnonzero(mask)[e.which]]


def _elements(ins, operand, width: int):
    if width == 1:
        if isinstance(operand, Vec):
            raise unsupported(ins, "a {vector} operand needs .v2 or .v4")
        return [operand]
    if not isinstance(operand, Vec) or len(operand.elems) != width:
        raise unsupported(ins, f"expected a vector of {width} operands")
    return list(operand.elems)


@op("ld")
def _ld(w, ins, mask):
    ty, space, width = _space(ins)
    d, src = _operands(ins, 2)
    dests = _elements(ins, d, width)
    dt = NP_TYPES[ty]
    if space == "param":
        if not (isinstance(src, Mem) and isinstance(src.base, Sym)):
            raise unsupported(ins, "expected ld.param from [parameter_name+offset]")
        name = src.base.name
        if name in w.params:
            raw = w.params[name]
            if src.offset < 0 or src.offset + width * dt.itemsize > len(raw):
                raise ExecutionError(f"ld.param.{ty} at offset {src.offset} reads past the "
                                     f"{len(raw)}-byte parameter `{name}`")
            values = np.frombuffer(raw, dtype=dt, count=width, offset=src.offset)
            for dest, v in zip(dests, values):
                w.write(dest, np.full(len(mask), v, dtype=dt), mask)
        elif name in w.call_params and width == 1 and src.offset == 0:  # e.g. a call's return value
            w.write(dests[0], w.call_params[name].astype(UINT_BY_SIZE[dt.itemsize]).view(dt), mask)
        else:
            raise ExecutionError(f"`{name}` is not a parameter this instruction can read")
        return
    addrs = _address(w, ins, src)
    for k, dest in enumerate(dests):
        w.write(dest, _load(w, ins, space, addrs + np.uint64(k * dt.itemsize), dt, mask), mask)


@op("st")
def _st(w, ins, mask):
    ty, space, width = _space(ins)
    dst, a = _operands(ins, 2)
    values = _elements(ins, a, width)
    dt = NP_TYPES[ty]
    if space == "param":  # an argument for an upcoming call
        if not (isinstance(dst, Mem) and isinstance(dst.base, Sym) and dst.offset == 0
                and dst.base.name in w.program.fn.call_params and width == 1):
            raise unsupported(ins, "st.param is only supported for scalar call arguments")
        slot = w.call_params.setdefault(dst.base.name, np.zeros(len(mask), dtype=np.uint64))
        bits = w.read(values[0], ty).view(UINT_BY_SIZE[dt.itemsize]).astype(np.uint64)
        np.copyto(slot, bits, where=mask)
        return
    addrs = _address(w, ins, dst)
    for k, value in enumerate(values):
        _store(w, ins, space, addrs + np.uint64(k * dt.itemsize), w.read(value, ty), mask)


@op("elect")
def _elect(w, ins, mask):
    """elect.sync d|p, membermask: choose one leader among the lanes of the mask.
    Every lane gets the leader's lane id; the predicate is true only for the leader."""
    if ins.mods != ("sync",):
        raise unsupported(ins)
    d, member = _operands(ins, 2)
    members = ((w.read(member, "u32")[:, None] >> _LANES.astype(np.uint32)) & 1).astype(bool)
    leader = np.zeros(len(mask), dtype=np.uint32)
    for lane in np.flatnonzero(mask):
        candidates = np.flatnonzero(members[lane] & mask)
        leader[lane] = candidates[0] if len(candidates) else lane
    if isinstance(d, Pair):
        w.write(d.first, leader, mask)
        w.write(d.second, leader == _LANES, mask)
    else:
        w.write(d, leader, mask)


_LANES = np.arange(32, dtype=np.int64)


# -- warp shuffle ------------------------------------------------------------------

@op("shfl")
def _shfl(w, ins, mask):
    """shfl.sync.{up,down,bfly,idx}.b32 d[|p], a, b, c, membermask

    Each lane reads register `a` of another lane `j` of the same warp. `c`
    packs a clamp value (bits 0-4) and a segment mask (bits 8-12) that confine
    `j` to the lane's segment; if `j` falls outside, the lane keeps its own value
    and the optional predicate `p` is false.
    """
    if len(ins.mods) != 3 or ins.mods[0] != "sync" or ins.mods[2] != "b32":
        raise unsupported(ins, "expected shfl.sync.mode.b32")
    mode = ins.mods[1]
    d, a, b, c, member = _operands(ins, 5)
    value = w.read(a, "b32")
    bval = (w.read(b, "u32") & 31).astype(np.int64)
    cbits = w.read(c, "u32")
    cval = (cbits & 31).astype(np.int64)
    seg = ((cbits >> 8) & 31).astype(np.int64)
    max_lane = (_LANES & seg) | (cval & ~seg & 31)
    min_lane = _LANES & seg
    if mode == "up":
        j = _LANES - bval
        valid = j >= max_lane
    elif mode == "down":
        j = _LANES + bval
        valid = j <= max_lane
    elif mode == "bfly":
        j = _LANES ^ bval
        valid = j <= max_lane
    elif mode == "idx":
        j = min_lane | (bval & ~seg & 31)
        valid = j <= max_lane
    else:
        raise unsupported(ins, f"shuffle mode .{mode}")
    j = np.where(valid, j, _LANES)

    member_values = w.read(member, "u32")
    members = ((member_values[:, None] >> _LANES.astype(np.uint32)) & 1).astype(bool)
    # Each caller must name itself, and its non-exited members must participate
    # with the same membermask. Exited/unlaunched lanes may remain in the mask;
    # distinct groups can execute the same instruction together on sm_70+.
    named = members[_LANES, _LANES]
    same_group = mask[None, :] & (member_values[:, None] == member_values[None, :])
    missing = (members & w.live_lanes()[None, :] & ~same_group).any(axis=1)
    mismatch = mask & (~named | missing)
    if mismatch.any():
        w.cta.findings.add(
            "shfl-membermask-mismatch",
            "shfl.sync caller is absent from its member mask, or a non-exited member "
            "is not executing with the same member mask", w, ins, np.flatnonzero(mismatch))
    # Callers outside their own mask were already reported above. For valid
    # callers the source must both execute the instruction and be a named member.
    stale = mask & named & (~mask[j] | ~members[_LANES, j])
    if stale.any():
        w.cta.findings.add(
            "shfl-inactive-source",
            "shfl.sync reads a lane that is not executing this instruction or is outside "
            "the caller's member mask; its value is undefined",
            w, ins, np.flatnonzero(stale))

    if isinstance(d, Pair):
        w.write(d.first, value[j], mask)
        w.write(d.second, valid, mask)
    else:
        w.write(d, value[j], mask)


# Control flow is executed by the warp itself; listed here so that the
# "is this opcode implemented" check has one source of truth.
CONTROL_FLOW = frozenset({"bra", "ret", "exit"})
SUPPORTED_OPCODES = frozenset(HANDLERS) | CONTROL_FLOW
