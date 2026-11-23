"""One warp: 32 lanes that execute the same instruction under an active mask."""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError, MemoryFault, UnsupportedError
from ..frontend.ir import (Imm, Reg, Special, Sym, UINT_BY_SIZE, INT_BY_SIZE, NP_TYPES,
                            storage_dtype)
from .program import Program
from .simt_stack import SimtStack

WARP_SIZE = 32


class Warp:
    def __init__(self, program: Program, *, cta, warp_id: int, tid, mask: np.ndarray,
                 specials: dict[str, np.ndarray], params: dict, mem, trace=None):
        self.program = program
        self.cta = cta  # the Cta this warp belongs to
        self.blocked = None  # a core.cta.Wait while the warp cannot run
        self.seen: set[int] = set()  # TMA copies this warp has synchronized with
        self.open_stores: list = []  # TMA stores issued but not yet in a bulk group
        self.groups: list[list] = []  # committed bulk groups not yet waited for
        self.warp_id = warp_id
        self.tid = tid  # (x, y, z) arrays, one entry per lane
        self.specials = specials
        self.params = params  # name -> raw little-endian bytes
        self.call_params: dict[str, np.ndarray] = {}  # arguments being set up for a call, per lane
        self.local = np.zeros((WARP_SIZE, program.local_bytes), dtype=np.uint8)  # per-thread stack
        self.mem = mem
        self.trace = trace
        self.regs = {name: np.zeros(WARP_SIZE, dtype=storage_dtype(ty))
                     for name, ty in program.fn.regs.items()}
        self.stack = SimtStack(mask, program.exit_pc)
        self.instructions = 0
        self.lane_instructions = 0
        self.divergences = 0

    @property
    def done(self) -> bool:
        return self.stack.done

    def live_lanes(self) -> np.ndarray:
        """Lanes that have not returned, wherever they are in the SIMT stack."""
        live = np.zeros(WARP_SIZE, dtype=np.bool_)
        for e in self.stack.entries:
            live |= e.mask
        return live

    # -- operand access ------------------------------------------------------
    def read_address(self, operand) -> np.ndarray:
        """An address operand as 32 unsigned 64-bit values. Shared-memory
        addresses are often kept in 32-bit registers."""
        kind = type(operand)
        if kind is Reg:
            raw = self.regs[operand.name]
            if raw.dtype.kind == "b":
                raise ExecutionError(f"predicate register {operand.name} used as an address")
            return raw.astype(np.uint64)
        if kind is Imm and operand.kind == "int":
            return np.full(WARP_SIZE, operand.value % (1 << 64), dtype=np.uint64)
        if kind is Sym:
            return np.full(WARP_SIZE, self._symbol(operand), dtype=np.uint64)
        raise ExecutionError(f"operand {operand!r} cannot be used as an address")

    def _symbol(self, sym: Sym) -> int:
        prog = self.program
        for table in (prog.symbols, prog.local_symbols, prog.global_symbols):
            if sym.name in table:
                return table[sym.name]
        raise ExecutionError(f"`{sym.name}` is not a variable this kernel can address")

    def read_bytes(self, address: int, count: int, lane: int) -> bytes:
        """`count` bytes at a generic address, as seen by one lane."""
        from ..units.alu import route
        space, offset = route(address)
        if space == "local":
            if offset + count > self.local.shape[1]:
                raise ExecutionError(f"read past the end of local memory at offset {offset}")
            return self.local[lane, offset:offset + count].tobytes()
        addrs = np.uint64(offset) + np.arange(count, dtype=np.uint64)
        if space == "shared":
            return self.cta.smem.data[self.cta.smem.index(addrs, 1, "read")].tobytes()
        return self.mem.read(addrs, np.dtype(np.uint8)).tobytes()

    def read_cstring(self, address: int, lane: int, limit: int = 1 << 16) -> str:
        out = bytearray()
        while len(out) < limit:
            chunk = self.read_bytes(address + len(out), 1, lane)
            if chunk == b"\0":
                break
            out += chunk
        return out.decode("utf-8", errors="replace")

    def read(self, operand, ty: str) -> np.ndarray:
        """The operand as 32 values of PTX type `ty`."""
        dt = NP_TYPES[ty]
        kind = type(operand)
        if kind is Reg:
            raw = self.regs[operand.name]
            if raw.dtype.kind == "b" or dt.kind == "b":
                if raw.dtype != dt:
                    raise ExecutionError(f"register {operand.name} cannot be read as .{ty}")
                return raw
            if raw.itemsize == dt.itemsize:
                return raw.view(dt)
            if raw.itemsize > dt.itemsize:
                return raw.astype(UINT_BY_SIZE[dt.itemsize]).view(dt)
            raise ExecutionError(f"register {operand.name} is narrower than .{ty}")
        if kind is Imm:
            return self._immediate(operand, dt, ty)
        if kind is Special:
            value = self.specials[operand.name]
            if dt.kind not in "iu" or dt.itemsize < 4:
                raise ExecutionError(f"%{operand.name} cannot be read as .{ty}")
            return value.astype(UINT_BY_SIZE[dt.itemsize]).view(dt)
        if kind is Sym:  # the address of a variable, as in `mov.b32 %r1, tile`
            if dt.kind not in "iu" or dt.itemsize < 4:
                raise ExecutionError(f"address of `{operand.name}` cannot be read as .{ty}")
            return np.full(WARP_SIZE, self._symbol(operand), dtype=UINT_BY_SIZE[dt.itemsize]).view(dt)
        raise ExecutionError(f"operand {operand!r} cannot be used as a .{ty} value")

    @staticmethod
    def _immediate(imm: Imm, dt: np.dtype, ty: str) -> np.ndarray:
        if imm.kind == "int":
            if dt.kind == "b":
                return np.full(WARP_SIZE, bool(imm.value))
            if dt.kind == "f":
                raise ExecutionError(f"integer literal {imm.value} used as a .{ty} value")
            u = UINT_BY_SIZE[dt.itemsize]
            return np.full(WARP_SIZE, imm.value % (1 << (8 * dt.itemsize)), dtype=u).view(dt)
        size = 4 if imm.kind == "f32" else 8
        value = np.full(WARP_SIZE, imm.value, dtype=UINT_BY_SIZE[size])
        if dt.kind == "f":
            return value.view(np.float32 if size == 4 else np.float64).astype(dt)
        if dt.kind in "iu" and dt.itemsize == size:
            return value.view(dt)  # raw bits, as in `mov.b32 %r1, 0f3F800000`
        raise ExecutionError(f"floating-point literal used as a .{ty} value")

    def write(self, dest, value: np.ndarray, mask: np.ndarray) -> None:
        """Store `value` into register `dest` on the lanes in `mask`."""
        if type(dest) is not Reg:
            raise ExecutionError(f"destination {dest!r} is not a register")
        raw = self.regs[dest.name]
        if raw.dtype.kind == "b" or value.dtype.kind == "b":
            if raw.dtype != value.dtype:
                raise ExecutionError(f"type mismatch writing register {dest.name}")
            bits = value
        elif value.itemsize == raw.itemsize:
            bits = value.view(raw.dtype)
        elif value.itemsize < raw.itemsize:
            if value.dtype.kind == "i":  # sign-extend
                bits = value.astype(INT_BY_SIZE[raw.itemsize]).view(raw.dtype)
            else:  # zero-extend
                bits = value.view(UINT_BY_SIZE[value.itemsize]).astype(raw.dtype)
        else:
            raise ExecutionError(f"register {dest.name} is too narrow for the result")
        np.copyto(raw, bits, where=mask)

    # -- execution -----------------------------------------------------------
    def step(self) -> None:
        """Execute one instruction on the lanes of the top SIMT stack entry."""
        stack = self.stack
        top = stack.top
        pc, mask = top.pc, top.mask
        prog = self.program
        ins = prog.instrs[pc]

        if ins.pred is None:
            active = mask
        else:
            p = self.regs[ins.pred.name]
            active = mask & ~p if ins.pred_negated else mask & p

        self.instructions += 1
        self.lane_instructions += int(np.count_nonzero(active))
        if self.trace is not None:
            self.trace(self, pc, ins, active)

        op = ins.opcode
        if op == "bra":
            target = prog.targets[pc]
            if ins.pred is None or active.all() or np.array_equal(active, mask):
                top.pc = target
            elif not active.any():
                top.pc = pc + 1
            else:
                self.divergences += 1
                stack.diverge(pc + 1, target, active, mask & ~active, prog.reconv[pc])
        elif op == "ret" or op == "exit":
            top.pc = pc + 1
            stack.retire(active)
        else:
            try:
                prog.handlers[pc](self, ins, active)
            except (ExecutionError, UnsupportedError) as e:
                if isinstance(e, ExecutionError) and e.where is None:
                    e.where = self._where(ins, active, e)
                raise
            if self.blocked is None or not self.blocked.retry:
                top.pc = pc + 1
        stack.settle()

    def _where(self, ins, active: np.ndarray, err: ExecutionError) -> str:
        lanes = err.lanes if isinstance(err, MemoryFault) and err.lanes else \
            [int(x) for x in np.flatnonzero(active)]
        lines = [f"  at {self.program.module.name}:{ins.line}: {ins.text}",
                 f"  in kernel `{self.program.fn.name}`, CTA {self.cta.id}, warp {self.warp_id}"]
        if lanes:
            first = lanes[0]
            thread = tuple(int(t[first]) for t in self.tid)
            more = f" and {len(lanes) - 1} more lane(s)" if len(lanes) > 1 else ""
            lines.append(f"  lane {first} (thread {thread}){more}")
        return "\n".join(lines)
