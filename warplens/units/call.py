"""Calls. Only the device-side `vprintf` is implemented, which is what `printf`
in CUDA C++ compiles to:

    call.uni (retval0), vprintf, (format, args);

`args` points at a buffer in the thread's local memory where the arguments
are packed like a C struct: 4 bytes for integers, 8 for long integers, doubles
and pointers, each at its natural alignment.
"""

from __future__ import annotations

import re
import struct

import numpy as np

from ..errors import ExecutionError
from ..frontend.ir import Sym, Vec
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
    if not ops or not isinstance(ops[0], Sym):
        raise unsupported(ins, "indirect calls")
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
