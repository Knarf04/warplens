"""A function prepared for execution: handlers bound, branches resolved."""

from __future__ import annotations

from ..errors import UnsupportedError
from ..frontend.cfg import reconvergence_points
from ..frontend.ir import Function, Module
from ..units import barrier as _barrier  # noqa: F401  (registers its handlers)
from ..units.alu import CONTROL_FLOW, HANDLERS


def _align_up(x: int, align: int) -> int:
    return -(-x // align) * align


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

        # Shared-memory layout: sized variables in declaration order, then the
        # dynamically sized `.extern .shared name[]` region requested at launch.
        self.symbols: dict[str, int] = {}
        offset = 0
        variables = [*module.shared, *fn.shared]
        for v in variables:
            if v.size is not None:
                offset = _align_up(offset, v.align)
                self.symbols[v.name] = offset
                offset += v.size
        self.smem_static = offset
        self.smem_dynamic_base = _align_up(offset, 16)
        for v in variables:
            if v.size is None:
                self.symbols[v.name] = self.smem_dynamic_base
        self.uses_dynamic_smem = any(v.size is None for v in variables)

    def smem_bytes(self, dynamic: int) -> int:
        return self.smem_dynamic_base + dynamic if dynamic else self.smem_static
