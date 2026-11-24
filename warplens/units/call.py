"""Calls. Calls to the module's own functions are inlined before a kernel runs
(frontend/inline.py); what is left for run time is recursion, at the end of this file, and
the device-side `vprintf`, which is what `printf` in CUDA C++ compiles to:

    call.uni (retval0), vprintf, (format, args);

`args` points at a buffer in the thread's local memory where the arguments
are packed like a C struct: 4 bytes for integers, 8 for long integers, doubles
and pointers, each at its natural alignment.
"""

from __future__ import annotations

import re
import struct

import numpy as np

from ..errors import ExecutionError, UnsupportedError
from ..frontend.ir import Reg, Sym, Vec
from .alu import op, unsupported

_SPEC = re.compile(r"%([-+ #0]*)(\d+|\*)?(?:\.(\d+|\*))?(hh|h|ll|l|j|z|t|L)?([diouxXeEfFgGaAcspn%])")


def format_printf(fmt: str, read, read_cstring) -> tuple[str, int]:
    """Return text and the number of parsed arguments (CUDA printf's result).

    `read(offset, size)` returns argument bytes.
    """
    out, pos, offset, argc = [], 0, 0, 0

    def take(size: int, code: str):
        nonlocal offset, argc
        offset = -(-offset // size) * size
        (value,) = struct.unpack("<" + code, read(offset, size))
        offset += size
        argc += 1
        return value

    for m in _SPEC.finditer(fmt):
        out.append(fmt[pos:m.start()])
        pos = m.end()
        flags, width, precision, length, conv = m.groups()
        if conv == "%":
            out.append("%")
            continue
        if width == "*":
            width = str(take(4, "i"))
        if precision == "*":
            precision = str(take(4, "i"))
        spec = "%" + flags + (width or "") + ("." + precision if precision is not None else "")
        wide = length in ("l", "ll", "j", "z", "t")
        if conv in "di":
            out.append((spec + "d") % take(8, "q") if wide else (spec + "d") % take(4, "i"))
        elif conv in "ouxX":
            out.append((spec + conv.replace("u", "d")) % (take(8, "Q") if wide else take(4, "I")))
        elif conv in "eEfFgGaA":
            value = take(8, "d")
            out.append(float.hex(value) if conv in "aA" else (spec + conv) % value)
        elif conv == "c":
            out.append((spec + "c") % chr(take(4, "I") & 0xFF))
        elif conv == "s":
            out.append((spec + "s") % read_cstring(take(8, "Q")))
        elif conv == "p":
            out.append((spec + "s") % hex(take(8, "Q")))
        else:
            raise ExecutionError(f"printf conversion %{conv} is not supported")
    out.append(fmt[pos:])
    return "".join(out), argc


@op("call")
def _call(w, ins, mask):
    ops = list(ins.operands)
    ret = ops.pop(0) if ops and isinstance(ops[0], Vec) else Vec(())
    if ops and isinstance(ops[0], Reg):
        # Lanes get here when the register holds the address of none of the functions the call
        # was expanded to (frontend/inline.py).
        lane = int(np.flatnonzero(mask)[0]) if mask.any() else 0
        address = int(w.read(ops[0], "u64")[lane])
        err = ExecutionError(
            f"indirect call through {ops[0].name} to {address:#x}, which is not the address of a "
            "function of this module that takes these arguments")
        raise err
    if not ops or not isinstance(ops[0], Sym):
        raise unsupported(ins, "expected call (results), function, (arguments)")
    name = ops[0].name
    args = ops[1].elems if len(ops) > 1 and isinstance(ops[1], Vec) else ()
    if name != "vprintf":
        raise unsupported(ins, f"call to `{name}`; only printf is implemented")
    if len(args) != 2 or len(ret.elems) > 1:
        raise unsupported(ins, "unexpected vprintf signature")
    try:
        fmt_ptr, arg_ptr = (w.call_params[a.name] for a in args)
    except (KeyError, AttributeError):
        raise ExecutionError("vprintf called before its arguments were stored") from None
    written = np.zeros(len(mask), dtype=np.uint64)
    for lane in np.flatnonzero(mask):
        lane = int(lane)
        base = int(arg_ptr[lane])
        if not fmt_ptr[lane]:
            written[lane] = 0xFFFF_FFFF  # CUDA printf returns -1 for a null format.
            continue
        text, argc = format_printf(w.read_cstring(int(fmt_ptr[lane]), lane),
                                   lambda off, size: w.read_bytes(base + off, size, lane),
                                   lambda address: w.read_cstring(address, lane))
        w.cta.output.append(text)
        written[lane] = argc
    if ret.elems:
        slot = w.call_params.setdefault(ret.elems[0].name, np.zeros(len(mask), dtype=np.uint64))
        np.copyto(slot, written, where=mask)


# -- recursion -----------------------------------------------------------------------

# How many recursive calls of one thread may be active at once. Hardware gives each thread a
# stack of a fixed number of bytes instead (1024 by default, CU_LIMIT_STACK_SIZE), and how many
# bytes a call takes there is the compiler's business, so this is a simulator limit and not a
# prediction of where a GPU overflows.
MAX_CALL_DEPTH = 1024


class Activation:
    """One recursive call in progress: what its lanes get back when they return."""
    __slots__ = ("layout", "delta", "top", "regs", "params", "memory", "rets")


def _align_up(x: int, align: int) -> int:
    return -(-x // align) * align


def _stored(w, name: str):
    """What call-parameter storage `name` holds on every lane: the bytes of a struct, or a
    scalar per lane. None for a scalar that nothing has stored yet."""
    if name in w.program.local_symbols:
        at = w.local_offset(name)
        return w.local[:, at:at + w.program.local_sizes[name]].copy()
    value = w.call_params.get(name)
    return None if value is None else value.copy()


def _store(w, name: str, value: np.ndarray, lanes: np.ndarray) -> None:
    if name in w.program.local_symbols:
        at = w.local_offset(name)
        w.local[lanes, at:at + value.shape[1]] = value[lanes]
    else:
        np.copyto(w.call_params.setdefault(name, np.zeros(len(lanes), dtype=np.uint64)), value, where=lanes)


@op("callrec")
def _callrec(w, ins, mask):
    """callrec.bound (results), copy, (arguments): call the function whose inlined body is
    `copy`, from inside that body. The lanes in `mask` enter it again as a new entry on the
    SIMT stack; the others wait, as they would at a branch."""
    if not mask.any():
        return
    rets, target, args = ins.operands
    prog, local = w.program, w.program.local_symbols
    layout = prog.frames[target.name]
    active = [e.frame for e in w.stack.entries if e.frame is not None]
    if len(active) >= MAX_CALL_DEPTH:
        raise ExecutionError(
            f"call stack overflow: `{layout.name}` recursed past {MAX_CALL_DEPTH} active calls "
            "(warplens.units.call.MAX_CALL_DEPTH); recursion without an end?")

    def spans(names):
        return [(w.local_offset(n), prog.local_sizes[n]) for n in names if n in local]

    # The arguments as the caller stored them, before the callee's copy of that storage is saved.
    passed = [_stored(w, a.name) for a in args.elems]
    if any(value is None for value in passed):
        raise ExecutionError(f"`{layout.name}` called before its arguments were stored")

    act = Activation()
    act.layout, act.rets = layout, tuple(r.name for r in rets.elems)
    act.regs = {r: w.regs[r].copy() for r in layout.regs}
    scalars = [n for n in (*layout.params, *layout.args, *layout.rets) if n not in local]
    act.params = {n: w.call_params[n].copy() for n in scalars if n in w.call_params}
    act.memory = [(at, w.local[:, at:at + size].copy()) for at, size in spans((*layout.args, *layout.rets))]

    # A new stack frame for the callee's local variables, above every frame in use.
    top = max([prog.local_bytes, *(a.top for a in active)])
    act.delta = _align_up(top - layout.start, max(layout.align, 16)) if layout.locals else 0
    act.top = layout.end + act.delta if layout.locals else top
    if act.top > w.local.shape[1]:
        from .alu import _LOCAL_WINDOW_BYTES
        if act.top > _LOCAL_WINDOW_BYTES:
            raise ExecutionError(f"call stack overflow: the local variables of `{layout.name}` and its "
                                 f"callers need more than {_LOCAL_WINDOW_BYTES} bytes per thread")
        grown = np.zeros((w.local.shape[0], max(act.top, 2 * w.local.shape[1])), dtype=np.uint8)
        grown[:, :w.local.shape[1]] = w.local
        w.local = grown
    if layout.locals:
        w.local[:, layout.start + act.delta:act.top] = 0

    for name, value in zip(layout.args, passed):
        _store(w, name, value, mask)
    w.stack.call(layout.entry_pc, mask, act)


@op("callret")
def _callret(w, ins, mask):
    """callret copy: the end of a recursive function's inlined body. Lanes that came by
    `callrec` go back to it; lanes of the first, inlined, call just continue."""
    entries = w.stack.entries
    owner = next((e for e in reversed(entries) if e.frame is not None), None)
    if owner is None or owner.frame.layout.prefix != ins.operands[0].name:
        return
    act, layout, lanes = owner.frame, owner.frame.layout, owner.mask
    if owner is not entries[-1]:
        raise UnsupportedError(
            f"recursive call to `{layout.name}`: lanes return separately, which happens when the "
            "function contains `exit` or a loop without an end; that is not supported in recursion")
    results = [_stored(w, name) for name in layout.rets]
    entries.pop()
    for name, saved in act.regs.items():
        np.copyto(w.regs[name], saved, where=lanes)
    for name, saved in act.params.items():
        np.copyto(w.call_params[name], saved, where=lanes)
    for at, saved in act.memory:
        w.local[lanes, at:at + saved.shape[1]] = saved[lanes]
    for name, value in zip(act.rets, results):
        if value is not None:  # None: the callee returned without storing a result
            _store(w, name, value, lanes)
