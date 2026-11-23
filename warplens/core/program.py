"""A function prepared for execution: handlers bound, branches resolved."""

from __future__ import annotations

from ..errors import UnsupportedError
from ..frontend.cfg import reconvergence_points
from ..frontend.ir import Function, Module
from ..units.alu import CONTROL_FLOW, HANDLERS


class Program:
    def __init__(self, module: Module, fn: Function):
        self.module = module
        self.fn = fn
        self.instrs = fn.instrs
        self.exit_pc = len(fn.instrs)

        missing = sorted({i.opcode for i in fn.instrs} - set(HANDLERS) - CONTROL_FLOW)
        if missing:
            first = next(i for i in fn.instrs if i.opcode in missing)
            raise UnsupportedError(
                f"kernel `{fn.name}` uses instructions warplens does not implement yet: "
                f"{', '.join(missing)}\n  first at {module.name}:{first.line}: {first.text}")

        self.handlers = [HANDLERS.get(i.opcode) for i in fn.instrs]
        self.targets = [fn.labels[i.operands[0].name] if i.opcode == "bra" else -1
                        for i in fn.instrs]
        self.reconv = reconvergence_points(fn)
