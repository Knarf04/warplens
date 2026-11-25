"""How an instruction finds the code that executes it.

Every module of this package that implements instructions registers its handlers here with
`@op`: a handler runs one instruction on all 32 lanes of a warp at once and writes its
result only to the lanes in `mask`. `warplens.core.program` binds the handlers to the
instructions of a function; importing a module is what registers its handlers.
"""

from __future__ import annotations

from ..errors import UnsupportedError
from ..frontend.ir import Instr

__all__ = ["CONTROL_FLOW", "HANDLERS", "expect_operands", "op", "unsupported"]

HANDLERS: dict = {}

# Control flow is executed by the warp itself; listed here so that the
# "is this opcode implemented" check has one source of truth.
CONTROL_FLOW = frozenset({"bra", "ret", "exit"})


def op(*names):
    def register(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return register


def unsupported(ins: Instr, why: str = "") -> UnsupportedError:
    detail = f" ({why})" if why else ""
    return UnsupportedError(f"unsupported instruction `{ins.name}`{detail}\n  line {ins.line}: {ins.text}")


def expect_operands(ins: Instr, n: int):
    if len(ins.operands) != n:
        raise unsupported(ins, f"expected {n} operands, found {len(ins.operands)}")
    return ins.operands
