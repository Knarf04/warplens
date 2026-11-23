"""tcgen05: the Blackwell tensor core and tensor memory (TMEM).

TMEM is per-CTA storage of 128 lanes x 512 columns of 32 bits. A TMEM address
is `(lane << 16) | column`. The tensor core multiplies operands it reads from
shared memory and accumulates into TMEM; threads then move accumulators
between TMEM and their registers.

    tcgen05.alloc [smem], ncols         reserve columns, write the TMEM address to shared memory
    tcgen05.mma [d], a, b, idesc, p     D = A * B^T (+ D if p), asynchronously
    tcgen05.commit [mbarrier]           arrive on the mbarrier once the MMAs issued so far are done
    tcgen05.ld / tcgen05.st             TMEM <-> registers
    tcgen05.dealloc d, ncols

What is implemented, and where each rule comes from (CUTLASS/CuTe sources):

* `cta_group::1` only. Operands A (M x K) and B (N x K) come from shared memory
  in K-major layout; D is M x N in TMEM as float32, one column per n.
* Instruction descriptor (cute/arch/mma_sm100_desc.hpp, InstrDescriptor):
  c_format bits 4-5, a_format 7-9, b_format 10-12, a/b negate 13/14,
  a/b major 15/16, N/8 in bits 17-22, M/16 in bits 24-28.
* Shared-memory descriptor (SmemDescriptor): start/16 in bits 0-13, leading
  byte offset/16 in 16-29, stride byte offset/16 in 32-45, layout type in 61-63
  (0 none, 6/4/2 = 32B/64B/128B swizzle). With T = 16 / element size, element
  (r, k) of a K-major operand is at
      start + (r % 8) * row_bytes + (r // 8) * SBO + (k % T) * size + (k // T) * step
  where row_bytes is 16/32/64/128 for none/32B/64B/128B and step is 16 (LBO when
  unswizzled); the swizzle is then applied to the address. K = 2T per MMA.
* Accumulator placement (cute/atom/mma_traits_sm100_frag.hpp): for M = 128, row m
  is lane m. For M = 64, row m is lane 32 * (m // 16) + m % 16: the low 16 lanes
  of each warp's 32.
* `tcgen05.ld.32x32b.xN [addr]`: the thread on lane l of its warp reads TMEM lane
  lane(addr) + l, columns column(addr) .. + N - 1. A warp may only touch the 32
  lanes 32 * (warpid % 4) .. + 31.

None of this has been run against hardware. Accumulation is float32; the order
of additions inside one MMA is not meant to match the hardware bit for bit.
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError
from ..frontend.ir import Mem, Vec
from ..host.tensor_map import swizzle_offsets
from .alu import _address, op, unsupported

LANES, COLUMNS = 128, 512
_LAYOUTS = {0: (0, 16), 6: (1, 32), 4: (2, 64), 2: (3, 128)}  # layout type -> (swizzle bits, row bytes)
_ELEMENT = {"f16": {0: np.dtype(np.float16)}, "tf32": {2: np.dtype(np.float32)}}
_NO_OPS = {"wait::ld", "wait::st", "fence::before_thread_sync", "fence::after_thread_sync",
           "relinquish_alloc_permit"}


class Tmem:
    def __init__(self):
        self.data = np.zeros((LANES, COLUMNS), dtype=np.uint32)
        self.written = np.zeros((LANES, COLUMNS), dtype=np.bool_)
        self.pending = np.zeros((LANES, COLUMNS), dtype=np.int32)  # MMA that last targeted the cell
        self.allocated = np.zeros(COLUMNS, dtype=np.bool_)
        self.allocations: dict[int, tuple[int, object]] = {}  # first column -> (columns, alloc instr)

    def alloc(self, ncols: int) -> int:
        if ncols < 32 or ncols > COLUMNS or ncols & (ncols - 1):
            raise ExecutionError(f"tcgen05.alloc of {ncols} columns; the count must be a power of "
                                 f"two from 32 to {COLUMNS}")
        for start in range(0, COLUMNS - ncols + 1, ncols):
            if not self.allocated[start:start + ncols].any():
                self.allocated[start:start + ncols] = True
                return start
        raise ExecutionError(f"tcgen05.alloc of {ncols} columns: tensor memory is full")

    def dealloc(self, address: int, ncols: int) -> None:
        start = address & 0xFFFF
        if start not in self.allocations or self.allocations[start][0] != ncols:
            raise ExecutionError(f"tcgen05.dealloc of {ncols} columns at column {start} does not "
                                 "match an allocation")
        del self.allocations[start]
        self.allocated[start:start + ncols] = False
        self.written[:, start:start + ncols] = False


def _tmem(cta) -> Tmem:
    if cta.tmem is None:
        cta.tmem = Tmem()
    return cta.tmem


def _split(address: int) -> tuple[int, int]:
    return address >> 16, address & 0xFFFF


def _operand_bytes(w, ins, desc: int, rows: int, dtype: np.dtype, which: str) -> np.ndarray:
    """Shared-memory byte indices, shape (rows, K, element size), of a K-major operand."""
    layout = desc >> 61
    if layout not in _LAYOUTS:
        raise unsupported(ins, f"shared-memory descriptor layout type {layout} for operand {which}")
    if (desc >> 49) & 7:
        raise unsupported(ins, "shared-memory descriptor with a non-zero base offset")
    bits, row_bytes = _LAYOUTS[layout]
    start = (desc & 0x3FFF) << 4
    lbo = ((desc >> 16) & 0x3FFF) << 4
    sbo = ((desc >> 32) & 0x3FFF) << 4
    size = dtype.itemsize
    per_chunk = 16 // size
    r = np.arange(rows, dtype=np.uint64)[:, None, None]
    k = np.arange(2 * per_chunk, dtype=np.uint64)[None, :, None]
    b = np.arange(size, dtype=np.uint64)[None, None, :]
    step = 16 if bits else lbo
    offsets = (np.uint64(start) + (r % 8) * row_bytes + (r // 8) * sbo
               + (k % per_chunk) * size + (k // per_chunk) * step + b)
    idx = swizzle_offsets(offsets, bits)
    if int(idx.max()) >= w.cta.smem.size:
        raise ExecutionError(f"tcgen05.mma operand {which} (descriptor {desc:#x}) extends past the "
                             f"CTA's {w.cta.smem.size} bytes of shared memory")
    return idx.astype(np.intp)


def _accumulator_lanes(m_dim: int) -> np.ndarray:
    m = np.arange(m_dim)
    return m if m_dim == 128 else 32 * (m // 16) + m % 16


def _mma(w, ins, lane: int) -> None:
    cta, ops = w.cta, list(ins.operands)
    kind = next((m.split("::")[1] for m in ins.mods if m.startswith("kind::")), None)
    if kind not in _ELEMENT:
        raise unsupported(ins, f"only kind::f16 and kind::tf32 are implemented")
    if len(ops) < 4 or not isinstance(ops[0], Mem):
        raise unsupported(ins, "expected [d], a, b, idesc[, {mask}][, enable]")
    if isinstance(ops[1], Mem):
        raise unsupported(ins, "operand A in tensor memory")
    d_lane, d_col = _split(int(_address(w, ins, ops[0])[lane]))
    desc_a = int(w.read(ops[1], "u64")[lane])
    desc_b = int(w.read(ops[2], "u64")[lane])
    idesc = int(w.read(ops[3], "u32")[lane])
    rest = ops[4:]
    disabled = np.zeros(LANES, dtype=np.bool_)
    if rest and isinstance(rest[0], Vec):
        words = [int(w.read(e, "u32")[lane]) for e in rest.pop(0).elems]
        for i, word in enumerate(words):
            disabled[32 * i:32 * i + 32] = (word >> np.arange(32)) & 1
    accumulate = bool(w.read(rest[0], "pred")[lane]) if rest else False

    if idesc & 0b1100 or (idesc >> 15) & 3:
        raise unsupported(ins, "sparse, saturating or MN-major MMA")
    if (idesc >> 4) & 3 != 1:
        raise unsupported(ins, "only float32 accumulators are implemented")
    formats = _ELEMENT[kind]
    a_fmt, b_fmt = (idesc >> 7) & 7, (idesc >> 10) & 7
    if a_fmt not in formats or b_fmt not in formats:
        raise unsupported(ins, f"operand format {a_fmt}/{b_fmt} for kind::{kind}")
    m_dim, n_dim = ((idesc >> 24) & 31) << 4, ((idesc >> 17) & 63) << 3
    if m_dim not in (64, 128) or not 8 <= n_dim <= 256:
        raise ExecutionError(f"tcgen05.mma with M={m_dim}, N={n_dim}; cta_group::1 needs M of 64 "
                             "or 128 and N from 8 to 256")
    sign = (-1.0 if (idesc >> 13) & 1 else 1.0) * (-1.0 if (idesc >> 14) & 1 else 1.0)

    tmem = _tmem(cta)
    idx_a = _operand_bytes(w, ins, desc_a, m_dim, formats[a_fmt], "A")
    idx_b = _operand_bytes(w, ins, desc_b, n_dim, formats[b_fmt], "B")
    lanes = d_lane + _accumulator_lanes(m_dim)
    if lanes.max() >= LANES or d_col + n_dim > COLUMNS:
        raise ExecutionError(f"tcgen05.mma result ({m_dim} x {n_dim}) at lane {d_lane}, column "
                             f"{d_col} does not fit in tensor memory")
    if not tmem.allocated[d_col:d_col + n_dim].all():
        cta.findings.add("tmem-unallocated", f"tcgen05.mma writes columns {d_col}-{d_col + n_dim - 1}, "
                         "which are not all allocated", w, ins, [lane])
    keep = ~disabled[lanes]
    cells = (lanes[keep][:, None], d_col + np.arange(n_dim)[None, :])

    operation = cta.new_async("mma", w, ins, lane)
    smem = cta.smem
    smem.async_reader[idx_a] = operation.id
    smem.async_reader[idx_b] = operation.id
    smem.async_active = True
    tmem.pending[cells] = operation.id
    w.uncommitted.append(operation)

    def land():
        a = smem.data[idx_a].view(formats[a_fmt])[..., 0].astype(np.float32)
        b = smem.data[idx_b].view(formats[b_fmt])[..., 0].astype(np.float32)
        result = (a @ b.T * np.float32(sign))[keep]
        if accumulate:
            result = result + tmem.data[cells].view(np.float32)
        tmem.data[cells] = result.astype(np.float32).view(np.uint32)
        tmem.written[cells] = True

    cta.schedule(operation, land)


def _transfer(w, ins, mask, store: bool) -> None:
    cta = w.cta
    shape = [m for m in ins.mods if "x" in m and m[0].isdigit()]
    count = next((int(m[1:]) for m in ins.mods if m[0] == "x" and m[1:].isdigit()), None)
    if shape != ["32x32b"] or count is None or any(m.startswith(("pack", "unpack")) for m in ins.mods):
        raise unsupported(ins, "only the 32x32b shape without packing is implemented")
    regs, mem = (ins.operands[1], ins.operands[0]) if store else ins.operands[:2]
    if not isinstance(regs, Vec) or len(regs.elems) != count or not isinstance(mem, Mem):
        raise unsupported(ins, f"expected {count} register(s) and a [tmem address]")
    lanes = np.flatnonzero(mask)
    if not len(lanes):
        return
    tmem = _tmem(cta)
    address = _address(w, ins, mem)[lanes]
    rows = (address >> np.uint64(16)).astype(np.int64) + lanes
    cols = (address & np.uint64(0xFFFF)).astype(np.int64)
    if rows.max() >= LANES or cols.max() + count > COLUMNS:
        raise ExecutionError(f"{ins.name} reaches lane {int(rows.max())}, column "
                             f"{int(cols.max()) + count - 1}, outside tensor memory")
    quadrant = 32 * (w.warp_id % 4)
    outside = (rows < quadrant) | (rows >= quadrant + 32)
    if outside.any():
        cta.findings.add("tmem-lane-rule",
                         f"warp {w.warp_id} may only access tensor-memory lanes {quadrant}-"
                         f"{quadrant + 31}, but this reaches lane {int(rows[outside][0])}",
                         w, ins, lanes[outside])
    for i, reg in enumerate(regs.elems):
        c = cols + i
        if not tmem.allocated[c].all():
            cta.findings.add("tmem-unallocated", f"{ins.name} touches a column that is not allocated",
                             w, ins, lanes[~tmem.allocated[c]])
        if store:
            tmem.data[rows, c] = w.read(reg, "b32")[lanes]
            tmem.written[rows, c] = True
            tmem.pending[rows, c] = 0
            continue
        if not tmem.written[rows, c].all():
            cta.findings.add("tmem-uninitialized-read", "read of tensor memory that was never written",
                             w, ins, lanes[~tmem.written[rows, c]])
        for op_id in np.unique(tmem.pending[rows, c]):
            if op_id and int(op_id) not in w.seen:
                operation = cta.async_ops[int(op_id)]
                state = "is still in flight" if not operation.done else \
                    "has completed, but this warp never waited on the mbarrier of its tcgen05.commit"
                cta.findings.add("tmem-read-before-commit",
                                 f"read of accumulators that the tcgen05.mma at line "
                                 f"{operation.ins.line} writes; that MMA {state}",
                                 w, ins, lanes[tmem.pending[rows, c] == op_id])
        value = np.zeros(len(mask), dtype=np.uint32)
        value[lanes] = tmem.data[rows, c]
        w.write(reg, value, mask)


@op("tcgen05")
def _tcgen05(w, ins, mask):
    kind = ins.mods[0] if ins.mods else ""
    if kind in _NO_OPS:
        return
    if not mask.any():
        return
    cta = w.cta
    lane = int(np.flatnonzero(mask)[0])
    if kind in ("alloc", "dealloc", "mma", "commit") and "cta_group::1" not in ins.mods:
        raise unsupported(ins, "cta_group::2 needs clusters, which are not implemented yet")
    if kind == "alloc":
        if len(ins.operands) != 2:
            raise unsupported(ins)
        ncols = int(w.read(ins.operands[1], "u32")[lane])
        tmem = _tmem(cta)
        column = tmem.alloc(ncols)
        tmem.allocations[column] = (ncols, (w, ins, lane))
        offset = _address(w, ins, ins.operands[0])[lane:lane + 1]
        cta.smem.write(offset, np.array([column], dtype=np.uint32), w.warp_id)
    elif kind == "dealloc":
        if len(ins.operands) != 2:
            raise unsupported(ins)
        _tmem(cta).dealloc(int(w.read(ins.operands[0], "u32")[lane]),
                           int(w.read(ins.operands[1], "u32")[lane]))
    elif kind == "mma":
        for l in np.flatnonzero(mask):
            _mma(w, ins, int(l))
    elif kind == "commit":
        if "mbarrier::arrive::one" not in ins.mods or any("multicast" in m for m in ins.mods) \
                or len(ins.operands) != 1:
            raise unsupported(ins)
        for l in np.flatnonzero(mask):
            l = int(l)
            bar = cta.mbarrier(int(_address(w, ins, ins.operands[0])[l]), w, ins, [l])
            covered, w.uncommitted = w.uncommitted, []
            operation = cta.new_async("commit", w, ins, l)

            def land(bar=bar, covered=covered, l=l):
                bar.knowledge.update(o.id for o in covered)
                bar.arrive(1, w, ins, [l])

            cta.schedule(operation, land)
    elif kind == "ld":
        _transfer(w, ins, mask, store=False)
    elif kind == "st":
        _transfer(w, ins, mask, store=True)
    else:
        raise unsupported(ins)
