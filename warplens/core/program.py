"""A function prepared for execution: handlers bound, branches resolved."""

from __future__ import annotations

from ..errors import UnsupportedError
from ..frontend.cfg import reconvergence_points
from ..frontend.inline import inline_calls
from ..frontend.ir import Function, Module
from ..units import barrier as _barrier  # noqa: F401  (these imports register handlers)
from ..units import call as _call  # noqa: F401
from ..units import mbarrier as _mbarrier  # noqa: F401
from ..units import tcgen05 as _tcgen05  # noqa: F401
from ..units import tma as _tma  # noqa: F401
from ..units.alu import CONTROL_FLOW, HANDLERS


def _align_up(x: int, align: int) -> int:
    return -(-x // align) * align


class Program:
    def __init__(self, module: Module, fn: Function, global_symbols: dict[str, int] | None = None):
        self.module = module
        fn = inline_calls(module, fn)  # calls to the module's own device functions disappear here
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

        # Per-thread local memory: variables in declaration order.
        self.local_symbols: dict[str, int] = {}
        offset = 0
        for v in fn.locals:
            offset = _align_up(offset, v.align)
            self.local_symbols[v.name] = offset
            offset += v.size or 0
        self.local_bytes = offset
        self.global_symbols = global_symbols or {}

        # Parameter space: where each parameter sits when its address is taken.
        self.param_offsets: dict[str, int] = {}
        offset = 0
        for p in fn.params:
            offset = _align_up(offset, p.align or min(p.size, 8) or 1)
            self.param_offsets[p.name] = offset
            offset += p.size
        self._param_bytes = offset
        self._blob: tuple[dict[str, bytes], bytes] | None = None

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
        dynamic_align = max([16, *(v.align for v in variables if v.size is None)])
        self.smem_dynamic_base = _align_up(offset, dynamic_align)
        for v in variables:
            if v.size is None:
                self.symbols[v.name] = self.smem_dynamic_base
        self.uses_dynamic_smem = any(v.size is None for v in variables)

    def param_blob(self, params: dict[str, bytes]) -> bytes:
        """All parameters of one launch laid out in parameter space."""
        # Keep the dictionary alive while its blob is cached: a bare id can
        # be reused by a later launch after the previous warps are collected.
        if self._blob is None or self._blob[0] is not params:
            blob = bytearray(self._param_bytes)
            for name, offset in self.param_offsets.items():
                blob[offset:offset + len(params[name])] = params[name]
            self._blob = (params, bytes(blob))
        return self._blob[1]

    def smem_bytes(self, dynamic: int) -> int:
        return self.smem_dynamic_base + dynamic if dynamic else self.smem_static
