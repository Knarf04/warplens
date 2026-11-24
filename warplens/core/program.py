"""A function prepared for execution: handlers bound, branches resolved."""

from __future__ import annotations

from ..errors import UnsupportedError
from ..frontend.cfg import block_reachability, build_cfg, reconvergence_points
from ..frontend.inline import inline_calls
from ..frontend.ir import Function, Module
from ..units import barrier as _barrier  # noqa: F401  (these imports register handlers)
from ..units import call as _call  # noqa: F401
from ..units import clc as _clc  # noqa: F401
from ..units import mbarrier as _mbarrier  # noqa: F401
from ..units import tcgen05 as _tcgen05  # noqa: F401
from ..units import tma as _tma  # noqa: F401
from ..units.alu import CONTROL_FLOW, HANDLERS


def _align_up(x: int, align: int) -> int:
    return -(-x // align) * align


class FrameLayout:
    """A recursive function's copy in a kernel, with its `.local` variables placed: the block
    they occupy is the stack frame that every active call needs a copy of."""

    def __init__(self, prefix: str, frame, entry_pc: int, offsets: dict[str, int], variables: dict):
        self.prefix, self.name, self.entry_pc = prefix, frame.name, entry_pc
        self.regs, self.params, self.args, self.rets = frame.regs, frame.params, frame.args, frame.rets
        self.locals = frozenset(frame.locals)
        self.start = min((offsets[v] for v in frame.locals), default=0)
        self.end = max((offsets[v] + (variables[v].size or 0) for v in frame.locals), default=0)
        self.align = max((variables[v].align for v in frame.locals), default=1)


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
        self._reach: tuple[list[int], list[int]] | None = None  # see `behind`

        # Per-thread local memory: variables in declaration order.
        self.local_symbols: dict[str, int] = {}
        offset = 0
        for v in fn.locals:
            offset = _align_up(offset, v.align)
            self.local_symbols[v.name] = offset
            offset += v.size or 0
        self.local_bytes = offset
        self.global_symbols = global_symbols or {}
        self.function_symbols = {name: module.function_address(name) for name in module.device_functions}
        # Copies of recursive functions (frontend/inline.py), by the prefix of their names.
        variables = {v.name: v for v in fn.locals}
        self.local_sizes = {name: v.size or 0 for name, v in variables.items()}
        self.frames = {prefix: FrameLayout(prefix, frame, fn.labels[frame.entry], self.local_symbols, variables)
                       for prefix, frame in fn.frames.items()}
        self.frame_locals = frozenset().union(*(f.locals for f in self.frames.values()))

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

    def behind(self, a: int, b: int) -> bool:
        """Whether lanes at pc `a` are strictly behind lanes at pc `b`: control can flow from
        `a` to `b` and not from `b` back to `a`. Lanes that are behind can still arrive where
        the others are, and the others cannot come to them, so the warp runs them first
        (core/warp.py, `_regroup`). Two pcs on one cycle are not behind each other."""
        if self._reach is None:
            cfg = build_cfg(self.fn)
            self._reach = ([*cfg.block_of, cfg.exit], block_reachability(cfg))
        block_of, reach = self._reach
        block_a, block_b = block_of[a], block_of[b]
        forward = (block_a == block_b and a < b) or bool(reach[block_a] >> block_b & 1)
        back = (block_a == block_b and b < a) or bool(reach[block_b] >> block_a & 1)
        return forward and not back

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
