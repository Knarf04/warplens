"""Functional semantics of the core PTX instructions.

Every handler runs one instruction on all 32 lanes of a warp at once and
writes its result only to the lanes in `mask`. Handlers reject modifiers they
do not implement instead of ignoring them.
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError, MemoryFault, UnsupportedError
from ..frontend.ir import Instr, Mem, NP_TYPES, Sym, UINT_BY_SIZE, Vec

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
    w.write(d, w.read(a, ty), mask)


@op("cvta")
def _cvta(w, ins, mask):
    # Address-space conversion. Global addresses are already generic addresses.
    ty, extra = _type(ins, allowed={"to", "global"}, kinds="u")
    if "global" not in extra:
        raise unsupported(ins, "only the global address space exists so far")
    d, a = _operands(ins, 2)
    w.write(d, w.read(a, ty), mask)


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
    d, a, b, c = _operands(ins, 4)
    prod, ty = _mul_parts(w, ins, a, b)
    w.write(d, prod + w.read(c, ty), mask)


@op("fma")
def _fma(w, ins, mask):
    ty, _ = _type(ins, allowed=_ROUND_NEAREST, kinds="f")
    d, a, b, c = _operands(ins, 4)
    x, y, z = w.read(a, ty), w.read(b, ty), w.read(c, ty)
    if ty == "f64":
        # No fused multiply-add in NumPy; double rounding is possible here.
        w.write(d, x * y + z, mask)
    else:
        wide = np.float64
        w.write(d, (x.astype(wide) * y.astype(wide) + z.astype(wide)).astype(x.dtype), mask)


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
    else:
        fn = np.minimum if ins.opcode == "min" else np.maximum
    w.write(d, fn(x, y), mask)


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
        out = np.zeros(len(r), dtype=dd)  # NaN converts to 0
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
    if isinstance(mem.base, Sym):
        raise unsupported(ins, f"address of symbol `{mem.base.name}`")
    base = w.read(mem.base, "u64")
    return base + np.uint64(mem.offset % (1 << 64)) if mem.offset else base


def _space(ins: Instr):
    ty, extra = _type(ins, allowed={"global", "param"})
    if len(extra) > 1:
        raise unsupported(ins)
    return ty, (extra[0] if extra else "global")


def _fault_lanes(e: MemoryFault, mask: np.ndarray) -> None:
    e.lanes = [int(x) for x in np.flatnonzero(mask)[e.which]]


@op("ld")
def _ld(w, ins, mask):
    ty, space = _space(ins)
    d, src = _operands(ins, 2)
    if isinstance(d, Vec):
        raise unsupported(ins, "vector loads")
    dt = NP_TYPES[ty]
    if space == "param":
        if not (isinstance(src, Mem) and isinstance(src.base, Sym) and src.offset == 0):
            raise unsupported(ins, "expected ld.param from [parameter_name]")
        if src.base.name not in w.params:
            raise ExecutionError(f"`{src.base.name}` is not a parameter of this kernel")
        bits, size = w.params[src.base.name]
        if size < dt.itemsize:
            raise ExecutionError(f"ld.param.{ty} reads past the {size}-byte parameter `{src.base.name}`")
        value = np.full(len(mask), bits & ((1 << (8 * dt.itemsize)) - 1),
                        dtype=UINT_BY_SIZE[dt.itemsize]).view(dt)
        w.write(d, value, mask)
        return
    addrs = _address(w, ins, src)
    try:
        loaded = w.mem.read(addrs[mask], dt)
    except MemoryFault as e:
        _fault_lanes(e, mask)
        raise
    value = np.zeros(len(mask), dtype=dt)
    value[mask] = loaded
    w.write(d, value, mask)


@op("st")
def _st(w, ins, mask):
    ty, space = _space(ins)
    if space != "global":
        raise unsupported(ins, "stores go to global memory only so far")
    dst, a = _operands(ins, 2)
    if isinstance(a, Vec):
        raise unsupported(ins, "vector stores")
    addrs = _address(w, ins, dst)
    value = w.read(a, ty)
    try:
        w.mem.write(addrs[mask], value[mask])
    except MemoryFault as e:
        _fault_lanes(e, mask)
        raise


# Control flow is executed by the warp itself; listed here so that the
# "is this opcode implemented" check has one source of truth.
CONTROL_FLOW = frozenset({"bra", "ret", "exit"})
SUPPORTED_OPCODES = frozenset(HANDLERS) | CONTROL_FLOW
