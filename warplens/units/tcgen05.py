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
* Shared-memory descriptor (SmemDescriptor): start/16 in bits 0-14, leading
  byte offset/16 in 16-30, stride byte offset/16 in 32-45, layout type in 61-63
  (0 none, 6/4/2 = 32B/64B/128B swizzle). With T = 16 / element size, element
  (r, k) of a K-major operand is at
      start + (r % 8) * row_bytes + (r // 8) * SBO + (k % T) * size + (k // T) * step
  where row_bytes is 16/32/64/128 for none/32B/64B/128B and step is 16 (LBO when
  unswizzled); the swizzle is then applied to the address. K = 2T per MMA.
* Accumulator placement (cute/atom/mma_traits_sm100_frag.hpp): for M = 128, row m
  is lane m. For M = 64, row m is lane 32 * (m // 16) + m % 16: the low 16 lanes
  of each warp's 32.
* `tcgen05.ld.32x32b.xN [addr]`: the thread on lane l of its warp reads TMEM lane
  32 * (warpid % 4) + (lane(addr) % 32) + l, columns column(addr) .. + N - 1.
  NVIDIA CUTLASS's tcgen05_ld primitive documents the local 5-bit row index;
  high row bits do not select another warp's window. Transfers spanning the
  local 32-row boundary are not modeled. The 16x64b, 16x128b and
  16x256b shapes cover 16 lanes and give each thread 1, 2 or 4 registers per
  unit; `transfer_cells` has the mapping (cute/atom/copy_traits_sm100.hpp).
  A warp may only touch the 32 lanes 32 * (warpid % 4) .. + 31.

None of this has been run against hardware. Accumulation is float32; the order
of additions inside one MMA is not meant to match the hardware bit for bit.
"""

from __future__ import annotations

import numpy as np

from ..core.cta import Wait
from ..errors import ExecutionError
from ..frontend.ir import Mem, Reg, Vec
from ..host.tensor_map import swizzle_offsets
from .alu import _address, op, unsupported
from .smem import CLUSTER_WINDOW, GENERIC_WINDOW, local_shared_offset, resolve_shared

LANES, COLUMNS = 128, 512
_LAYOUTS = {0: (0, 16), 6: (1, 32), 4: (2, 64), 2: (3, 128)}  # layout type -> (swizzle bits, row bytes)
_ELEMENT = {"f16": {0: np.dtype(np.float16)}, "tf32": {2: np.dtype(np.float32)}}
_FENCES = {"fence::before_thread_sync", "fence::after_thread_sync"}


class Tmem:
    def __init__(self):
        self.data = np.zeros((LANES, COLUMNS), dtype=np.uint32)
        self.written = np.zeros((LANES, COLUMNS), dtype=np.bool_)
        self.pending = np.zeros((LANES, COLUMNS), dtype=np.int32)  # MMA that last targeted the cell
        self.allocated = np.zeros(COLUMNS, dtype=np.bool_)
        self.allocations: dict[int, tuple[int, object]] = {}  # first column -> (columns, alloc instr)
        self.freed: set[tuple[int, int]] = set()  # (first column, columns) released and not reallocated
        self.alloc_permit = True
        self.last_alloc_size = COLUMNS

    def alloc(self, ncols: int) -> int | None:
        if not self.alloc_permit:
            raise ExecutionError("tcgen05.alloc after this CTA relinquished its allocation permit")
        if ncols < 32 or ncols > COLUMNS or ncols & (ncols - 1):
            raise ExecutionError(f"tcgen05.alloc of {ncols} columns; the count must be a power of "
                                 f"two from 32 to {COLUMNS}")
        if ncols > self.last_alloc_size:
            raise ExecutionError(f"tcgen05.alloc of {ncols} columns increases the previous allocation "
                                 f"size of {self.last_alloc_size} columns")
        for start in range(0, COLUMNS - ncols + 1, ncols):
            if not self.allocated[start:start + ncols].any():
                self.allocated[start:start + ncols] = True
                self.last_alloc_size = ncols
                self.freed = {(first, count) for first, count in self.freed
                              if first + count <= start or first >= start + ncols}
                return start
        return None  # the issuing warp retries after another warp frees memory

    def dealloc(self, address: int, ncols: int) -> None:
        start = address
        if start not in self.allocations or self.allocations[start][0] != ncols:
            raise ExecutionError(f"tcgen05.dealloc of {ncols} columns at column {start} does not "
                                 "match an allocation")
        del self.allocations[start]
        self.freed.add((start, ncols))
        self.allocated[start:start + ncols] = False
        self.written[:, start:start + ncols] = False
        self.pending[:, start:start + ncols] = 0


def _tmem(cta) -> Tmem:
    if cta.tmem is None:
        cta.tmem = Tmem()
    return cta.tmem


def _split(address: int) -> tuple[int, int]:
    return address >> 16, address & 0xFFFF


def _operand_bytes(w, ins, desc: int, rows: int, dtype: np.dtype, which: str,
                   mn_major: bool = False) -> np.ndarray:
    """Shared-memory byte indices, shape (rows, K, element size), of an MMA operand.

    `rows` runs along the operand's M or N dimension. K-major: elements along K are
    contiguous. MN-major: elements along M/N are contiguous (make_umma_desc<Major::MN>
    in cute/arch/mma_sm100_desc.hpp): with T = 16 / element size and a swizzle atom of
    A = 1, 2, 4 or 8 sixteen-byte chunks, element (r, k) is at
        (r % T) * size + (r // T % A) * 16 + (r // (T * A)) * LBO + (k % 8) * 16 * A + (k // 8) * SBO
    and for the unswizzled layout at (r % T) * size + (r // T) * SBO + (k % 8) * 16 + (k // 8) * LBO.
    """
    # PTX tcgen05 shared-memory descriptor: fixed 001 at bits 46..48,
    # reserved bits 15/31/54..60, and relative (not absolute) LBO mode.
    if (desc >> 46) & 7 != 1 or desc & ((1 << 15) | (1 << 31) | (0x7F << 54)):
        raise ExecutionError(f"invalid shared-memory descriptor {desc:#x} for operand {which}: "
                             "fixed or reserved bits are not set as required")
    if (desc >> 52) & 3:
        raise unsupported(ins, "shared-memory descriptor with absolute LBO or decompression mode")
    layout = desc >> 61
    if layout not in _LAYOUTS:
        raise unsupported(ins, f"shared-memory descriptor layout type {layout} for operand {which}")
    if (desc >> 49) & 7:
        raise unsupported(ins, "shared-memory descriptor with a non-zero base offset")
    bits, row_bytes = _LAYOUTS[layout]
    start = (desc & 0x7FFF) << 4
    lbo = ((desc >> 16) & 0x7FFF) << 4
    sbo = ((desc >> 32) & 0x3FFF) << 4
    size = dtype.itemsize
    per_chunk = 16 // size
    r = np.arange(rows, dtype=np.uint64)[:, None, None]
    k = np.arange(2 * per_chunk, dtype=np.uint64)[None, :, None]
    b = np.arange(size, dtype=np.uint64)[None, None, :]
    if mn_major:
        atom = row_bytes // 16  # sixteen-byte chunks along M/N in one swizzle atom
        if bits:
            offsets = (np.uint64(start) + (r % per_chunk) * size + (r // per_chunk % atom) * 16
                       + (r // (per_chunk * atom)) * lbo + (k % 8) * row_bytes + (k // 8) * sbo + b)
        else:
            offsets = (np.uint64(start) + (r % per_chunk) * size + (r // per_chunk) * sbo
                       + (k % 8) * 16 + (k // 8) * lbo + b)
    else:
        step = 16 if bits else lbo
        offsets = (np.uint64(start) + (r % 8) * row_bytes + (r // 8) * sbo
                   + (k % per_chunk) * size + (k // per_chunk) * step + b)
    idx = swizzle_offsets(offsets, bits)
    if int(idx.max()) >= w.cta.smem.size:
        raise ExecutionError(f"tcgen05.mma operand {which} (descriptor {desc:#x}) extends past the "
                             f"CTA's {w.cta.smem.size} bytes of shared memory")
    return idx.astype(np.intp)


_SWIZZLE_NAMES = {0: "no swizzle", 1: "32B swizzle", 2: "64B swizzle", 3: "128B swizzle"}


def _check_operand(w, ins, lane: int, which: str, idx: np.ndarray, bits: int, cta=None) -> None:
    """Is the shared memory an MMA is about to read what its descriptor claims?"""
    from .tma import _require_seen
    cta = cta or w.cta
    smem = cta.smem
    writers = smem.async_writer[idx]
    for op_id in np.unique(writers[writers > 0]):
        load = cta.async_ops[int(op_id)]
        if load.swizzle is not None and load.swizzle != bits:
            cta.findings.add(
                "mma-swizzle-mismatch",
                f"operand {which}'s descriptor says {_SWIZZLE_NAMES[bits]}, but the TMA load at "
                f"line {load.ins.line} wrote this buffer with {_SWIZZLE_NAMES[load.swizzle]}; "
                "the MMA will read the elements in the wrong order", w, ins, [lane])
    _require_seen(w, ins, [lane], writers, f"tcgen05.mma operand {which} reads")
    if smem.has_uninitialized_bytes(idx):
        cta.findings.add("smem-uninitialized-read",
                         f"tcgen05.mma operand {which} reads shared memory that was never written",
                         w, ins, [lane])


def _accumulator_lanes(m_dim: int) -> np.ndarray:
    m = np.arange(m_dim)
    return m if m_dim == 128 else 32 * (m // 16) + m % 16


def _mma(w, ins, lane: int, pair: bool = False) -> None:
    cta, ops = w.cta, list(ins.operands)
    kind = next((m.split("::")[1] for m in ins.mods if m.startswith("kind::")), None)
    if kind not in _ELEMENT:
        raise unsupported(ins, f"only kind::f16 and kind::tf32 are implemented")
    extra = set(ins.mods) - {"mma", "cta_group::1", f"kind::{kind}"}
    if extra:
        raise unsupported(ins, f"MMA modifier .{sorted(extra)[0]}")
    if len(ops) < 5 or not isinstance(ops[0], Mem):
        raise unsupported(ins, "expected [d], a, b, idesc[, {mask}], enable[, scale-input-d]")
    if isinstance(ops[1], Mem):
        raise unsupported(ins, "operand A in tensor memory")
    d_lane, d_col = _split(int(_address(w, ins, ops[0])[lane]))
    desc_a = int(w.read(ops[1], "u64")[lane])
    desc_b = int(w.read(ops[2], "u64")[lane])
    idesc = int(w.read(ops[3], "u32")[lane])
    rest = ops[4:]
    disabled = np.zeros(LANES * (2 if pair else 1), dtype=np.bool_)  # per lane, then per CTA
    if rest and isinstance(rest[0], Vec):
        words = [int(w.read(e, "u32")[lane]) for e in rest.pop(0).elems]
        if len(words) != (8 if pair else 4):
            raise ExecutionError("tcgen05.mma needs four disable-output-lane words per CTA")
        for i, word in enumerate(words):
            disabled[32 * i:32 * i + 32] = (word >> np.arange(32)) & 1
    if len(rest) not in (1, 2):
        raise unsupported(ins, "expected enable-input-d and optional scale-input-d")
    accumulate = bool(w.read(rest[0], "pred")[lane])
    scale = 0
    if len(rest) == 2:
        from ..frontend.ir import Imm
        operand = rest[1]
        if not isinstance(operand, Imm) or operand.kind != "int" or not 0 <= operand.value <= 15:
            raise ExecutionError("tcgen05.mma scale-input-d must be an integer constant from 0 to 15")
        scale = operand.value

    if idesc & ((1 << 6) | (1 << 23) | (7 << 29)):
        raise ExecutionError("tcgen05.mma instruction descriptor has reserved bits, an unsupported "
                             "K dimension or .ws shift controls")
    if idesc & 0b1100:
        raise unsupported(ins, "sparse or saturating MMA")
    a_mn_major, b_mn_major = bool((idesc >> 15) & 1), bool((idesc >> 16) & 1)
    if (idesc >> 4) & 3 != 1:
        raise unsupported(ins, "only float32 accumulators are implemented")
    formats = _ELEMENT[kind]
    a_fmt, b_fmt = (idesc >> 7) & 7, (idesc >> 10) & 7
    if a_fmt not in formats or b_fmt not in formats:
        raise unsupported(ins, f"operand format {a_fmt}/{b_fmt} for kind::{kind}")
    if kind == "tf32" and (a_mn_major or b_mn_major):
        # PTX 9.7.18.10.1: TF32 transpose needs the distinct 128B swizzle
        # with 32-byte atomicity, not any of the 16-byte layouts decoded here.
        raise unsupported(ins, "MN-major TF32 requires 128B swizzling with 32-byte "
                          "atomicity, which is not implemented")
    m_dim, n_dim = ((idesc >> 24) & 31) << 4, ((idesc >> 17) & 63) << 3
    if m_dim not in ((128, 256) if pair else (64, 128)) or not 8 <= n_dim <= 256 \
            or (pair and n_dim % 16):
        raise ExecutionError(f"tcgen05.mma with M={m_dim}, N={n_dim}; cta_group::1 needs M of 64 "
                             "or 128, cta_group::2 needs M of 128 or 256, and N is 8 to 256")
    sign = (-1.0 if (idesc >> 13) & 1 else 1.0) * (-1.0 if (idesc >> 14) & 1 else 1.0)

    # One CTA, or the even/odd pair of a cta_group::2 MMA. Either CTA may issue it. Each
    # CTA holds half of A's rows and half of B's rows in its own shared memory, at the same
    # descriptor addresses, and receives its half of the result in its own tensor memory.
    members = [cta]
    if pair:
        even_rank = cta.rank & ~1
        if even_rank + 1 >= len(cta.cluster.ctas):
            raise ExecutionError("a cta_group::2 MMA requires both CTAs of a "
                                 f"pair; this is rank {cta.rank} in a cluster of "
                                 f"{len(cta.cluster.ctas)}")
        # Matrix placement follows rank parity, not which CTA issued the MMA.
        members = [cta.peer(even_rank), cta.peer(even_rank + 1)]
    rows_m, rows_n = m_dim // len(members), n_dim // len(members)
    n = np.arange(n_dim)[None, :]
    m = np.arange(rows_m)[:, None]
    if not pair:
        lane_grid, column_grid = d_lane + _accumulator_lanes(m_dim)[:, None] + 0 * n, d_col + n + 0 * m
    elif m_dim == 256:  # each CTA: 128 rows on 128 lanes
        lane_grid, column_grid = d_lane + m + 0 * n, d_col + n + 0 * m
    else:  # M = 128: 64 rows per CTA, and the columns are folded onto the other 64 lanes
        lane_grid, column_grid = d_lane + m + 64 * (n >= n_dim // 2), d_col + n % (n_dim // 2) + 0 * m
    if lane_grid.max() >= LANES or column_grid.max() >= COLUMNS:
        raise ExecutionError(f"tcgen05.mma result ({m_dim} x {n_dim}) at lane {d_lane}, column "
                             f"{d_col} does not fit in tensor memory")

    operation = cta.new_async("mma", w, ins, lane)
    parts = []
    for i, member in enumerate(members):
        tmem = _tmem(member)
        idx_a = _operand_bytes(w, ins, desc_a, rows_m, formats[a_fmt], "A", a_mn_major)
        idx_b = _operand_bytes(w, ins, desc_b, rows_n, formats[b_fmt], "B", b_mn_major)
        for which, idx, desc in (("A", idx_a, desc_a), ("B", idx_b, desc_b)):
            _check_operand(w, ins, lane, which, idx, _LAYOUTS[desc >> 61][0], member)
        if not tmem.allocated[column_grid[0]].all():
            member.findings.add("tmem-unallocated", f"tcgen05.mma writes columns {int(column_grid.min())}-"
                                f"{int(column_grid.max())}, which are not all allocated", w, ins, [lane])
        keep = ~disabled[128 * i:128 * i + 128][lane_grid]
        cells = (lane_grid[keep], column_grid[keep])
        member.smem.async_reader[idx_a] = operation.id
        member.smem.async_reader[idx_b] = operation.id
        member.smem.async_active = True
        tmem.pending[cells] = operation.id
        parts.append((member, tmem, idx_a, idx_b, keep, cells))
    k_dim = parts[0][2].shape[1]
    operation.work = {"flops": 2.0 * m_dim * n_dim * k_dim, "kind": kind,
                      "shape": f"{m_dim}x{n_dim}x{k_dim}"}
    w.tcgen_issued.append(operation)

    def land():
        if kind == "tf32":
            # The operands are used as the float32 values they hold. Real TF32 has only
            # 10 fraction bits and what the hardware does with the rest is not specified,
            # so say so rather than silently pick an answer.
            for member, _, idx_a, idx_b, _, _ in parts:
                for which, idx in (("A", idx_a), ("B", idx_b)):
                    if (member.smem.data[idx].view(np.uint32) & np.uint32(0x1FFF)).any():
                        cta.findings.add(
                            "mma-tf32-operand-not-converted",
                            f"operand {which} of a kind::tf32 MMA holds float32 values with bits "
                            "below TF32 precision (the low 13 fraction bits). warplens multiplies "
                            "them as float32; hardware may not. Convert with cvt.rna.tf32.f32 first",
                            w, ins, [lane])
        b = np.concatenate([member.smem.data[idx_b].view(formats[b_fmt])[..., 0]
                            for member, _, _, idx_b, _, _ in parts]).astype(np.float32)
        for member, tmem, idx_a, _, keep, cells in parts:
            a = member.smem.data[idx_a].view(formats[a_fmt])[..., 0].astype(np.float32)
            result = (a @ b.T * np.float32(sign))[keep]
            if accumulate:
                # Check on completion: an earlier pipelined MMA in the same thread
                # may initialize these cells after this instruction is issued.
                if not tmem.written[cells].all():
                    member.findings.add("tmem-uninitialized-read",
                                        "tcgen05.mma accumulates tensor memory that was never written",
                                        w, ins, [lane])
                result = result + tmem.data[cells].view(np.float32) * np.float32(2.0 ** -scale)
            tmem.data[cells] = result.astype(np.float32).view(np.uint32)
            tmem.written[cells] = True

    cta.schedule(operation, land)


_REGISTERS_PER_UNIT = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}


def _transfer_size(shape: str, count: int) -> int:
    if shape not in _REGISTERS_PER_UNIT:
        raise ValueError(f"unsupported tensor-memory transfer shape {shape}")
    registers = _REGISTERS_PER_UNIT[shape] * count
    if count < 1 or count & (count - 1) or registers > 128:
        maximum = 128 // _REGISTERS_PER_UNIT[shape]
        raise ValueError(f"{shape} repeat count must be a power of two from 1 to {maximum}")
    return registers


def _collective(ins, mask) -> None:
    if not mask.all():
        raise ExecutionError(f"{ins.name} requires all 32 lanes of the warp to execute together; "
                             f"only {int(mask.sum())} participate")


def _uniform(w, ins, operand, mask, what: str) -> int:
    values = w.read(operand, "u32")[mask]
    if (values != values[0]).any():
        raise ExecutionError(f"{ins.name} requires the same {what} in every lane of the warp")
    return int(values[0])


def transfer_cells(shape: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Which TMEM cell each register of each thread maps to in `tcgen05.ld/st.<shape>.x<count>`.

    Returns (lane offset, column offset), each of shape (32 threads, registers),
    relative to the instruction's TMEM address. `32x32b` spans the warp's 32
    lanes, one lane per thread. The `16x...` shapes span 16 lanes and give each
    thread several cells of them. Derived from the Copy_Traits of
    SM100_TMEM_LOAD_* in cute/atom/copy_traits_sm100.hpp.
    """
    registers = _transfer_size(shape, count)
    t = np.arange(32)[:, None]
    q = np.arange(registers)[None, :]
    if shape == "32x32b":
        return t + 0 * q, q + 0 * t
    if shape == "16x64b":
        return 8 * (t & 1) + (t >> 2) + 0 * q, ((t >> 1) & 1) + 2 * q
    if shape == "16x128b":
        return (t >> 2) + 8 * (q & 1), (t & 3) + 4 * (q >> 1)
    return (t >> 2) + 8 * ((q >> 1) & 1), 2 * (t & 3) + (q & 1) + 8 * (q >> 2)


def _check_tmem_overwrite(w, ins, ids, lanes, *, dealloc=False) -> None:
    """A store/free must not erase the outstanding producer's dependency."""
    for op_id in np.unique(ids):
        if not op_id or int(op_id) in w.seen:
            continue
        operation = w.cta.async_ops[int(op_id)]
        state = "is still in flight" if not operation.done else "has not been synchronized with"
        action = "deallocation of" if dealloc else "write to"
        kind = "tmem-dealloc-before-commit" if dealloc else "tmem-write-before-commit"
        w.cta.findings.add(kind, f"{action} tensor memory targeted by the tcgen05.mma at line "
                           f"{operation.ins.line}; that MMA {state}", w, ins, lanes)


def _transfer(w, ins, mask, store: bool) -> None:
    cta = w.cta
    shape = next((m for m in ins.mods if m in _REGISTERS_PER_UNIT), None)
    count = next((int(m[1:]) for m in ins.mods if m[0] == "x" and m[1:].isdigit()), None)
    if shape is None or not count or any(m.startswith(("pack", "unpack")) for m in ins.mods):
        raise unsupported(ins, "implemented shapes are 32x32b, 16x64b, 16x128b and 16x256b, "
                               "without packing")
    if tuple(ins.mods) != ("st" if store else "ld", "sync", "aligned", shape, f"x{count}", "b32") \
            or len(ins.operands) != 2:
        raise unsupported(ins, "expected an unmodified .sync.aligned transfer with two operands")
    try:
        _transfer_size(shape, count)
    except ValueError as exc:
        raise unsupported(ins, str(exc)) from exc
    regs, mem = (ins.operands[1], ins.operands[0]) if store else ins.operands[:2]
    lane_offset, column_offset = transfer_cells(shape, count)
    if not isinstance(regs, Vec) or len(regs.elems) != lane_offset.shape[1] \
            or not all(isinstance(reg, Reg) for reg in regs.elems) or not isinstance(mem, Mem):
        raise unsupported(ins, f"expected {lane_offset.shape[1]} register(s) and a [tmem address]")
    lanes = np.flatnonzero(mask)
    if not len(lanes):
        return
    _collective(ins, mask)
    tmem = _tmem(cta)
    address = _address(w, ins, mem)[lanes]
    if (address != address[0]).any():
        raise ExecutionError(f"{ins.name} requires the same tensor-memory address in every lane "
                             "of the warp")
    if (address > np.uint64(0xFFFFFFFF)).any():
        raise ExecutionError(f"{ins.name} requires a 32-bit tensor-memory address")
    # NVIDIA CUTLASS's tcgen05_ld primitive documents row(addr) % 32 as the
    # local row index. All encoded row-window aliases therefore reach the
    # same physical window, fixed by the warp's position in its warpgroup.
    quadrant = 32 * (w.warp_id % 4)
    within = ((address >> np.uint64(16)) & np.uint64(31)).astype(np.int64)
    local_rows = within[:, None] + lane_offset[lanes]
    if (local_rows >= 32).any():
        # The documented base-row alias does not establish how an entire
        # transfer crossing this boundary behaves; do not invent wraparound.
        raise unsupported(ins, "a transfer spanning the local 32-row tensor-memory window")
    all_rows = quadrant + local_rows
    all_cols = (address & np.uint64(0xFFFF)).astype(np.int64)[:, None] + column_offset[lanes]
    if all_cols.max() >= COLUMNS:
        raise ExecutionError(f"{ins.name} reaches column {int(all_cols.max())}, outside tensor memory")
    for i, reg in enumerate(regs.elems):
        rows, c = all_rows[:, i], all_cols[:, i]
        if not tmem.allocated[c].all():
            cta.findings.add("tmem-unallocated", f"{ins.name} touches a column that is not allocated",
                             w, ins, lanes[~tmem.allocated[c]])
        if store:
            _check_tmem_overwrite(w, ins, tmem.pending[rows, c], lanes)
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


_AS_GROUP_1: dict[int, tuple] = {}  # id(instr) -> (instr, the same instr spelled with cta_group::1)


@op("tcgen05")
def _tcgen05(w, ins, mask):
    # cta_group::2 forms take the same operands as cta_group::1. Validate them through
    # one code path, and remember that a pair of CTAs is involved.
    pair = "cta_group::2" in ins.mods
    if pair:
        if id(ins) not in _AS_GROUP_1:
            import dataclasses
            mods = tuple("cta_group::1" if m == "cta_group::2" else m for m in ins.mods)
            _AS_GROUP_1[id(ins)] = (ins, dataclasses.replace(ins, mods=mods))
        ins = _AS_GROUP_1[id(ins)][1]
    kind = ins.mods[0] if ins.mods else ""
    if kind in _FENCES:
        if tuple(ins.mods) != (kind,) or ins.operands:
            raise unsupported(ins)
        if kind == "fence::after_thread_sync" and mask.any():
            w.seen |= w.tcgen_observed
            w.tcgen_observed.clear()
        return
    if not mask.any():
        return
    cta = w.cta
    lane = int(np.flatnonzero(mask)[0])
    if kind in ("alloc", "dealloc", "mma", "commit", "relinquish_alloc_permit") \
            and "cta_group::1" not in ins.mods:
        raise unsupported(ins, "expected .cta_group::1 or .cta_group::2")
    if kind in ("wait::ld", "wait::st", "relinquish_alloc_permit"):
        expected = ((kind, "cta_group::1", "sync", "aligned") if kind == "relinquish_alloc_permit"
                    else (kind, "sync", "aligned"))
        if tuple(ins.mods) != expected or ins.operands:
            raise unsupported(ins)
        _collective(ins, mask)
        if kind == "relinquish_alloc_permit":
            _tmem(cta).alloc_permit = False
        return
    if kind == "alloc":
        if len(ins.operands) != 2 or tuple(ins.mods) not in (
                ("alloc", "cta_group::1", "sync", "aligned", "shared::cta", "b32"),
                ("alloc", "cta_group::1", "sync", "aligned", "b32")):
            raise unsupported(ins)
        _collective(ins, mask)
        ncols = _uniform(w, ins, ins.operands[1], mask, "column count")
        offsets = _address(w, ins, ins.operands[0])[mask]
        if "shared::cta" not in ins.mods:
            window = np.uint64(GENERIC_WINDOW)
            if not ((offsets >= window) & (offsets < window + np.uint64(CLUSTER_WINDOW))).all():
                raise ExecutionError("generic tcgen05.alloc destination does not point to shared memory")
            offsets = offsets - window
        offsets = local_shared_offset(offsets)
        if (offsets % 4).any():
            raise ExecutionError("tcgen05.alloc destination must be aligned to 4 bytes")
        # Check every destination before reserving memory. The PTX contract requires
        # uniform nCols, but does not impose uniformity on the shared destinations.
        offsets = np.unique(offsets)
        cta.smem.index(offsets, 4, "write")
        tmem = _tmem(cta)
        column = tmem.alloc(ncols)
        if column is None:
            w.blocked = Wait("tensor-memory allocation", ins, retry=True,
                             detail=lambda: f"waiting for {ncols} tensor-memory columns to become available")
            return
        tmem.allocations[column] = (ncols, (w, ins, lane))
        cta.smem.write(offsets, np.full(len(offsets), column, dtype=np.uint32), w.warp_id)
    elif kind == "dealloc":
        if len(ins.operands) != 2 or tuple(ins.mods) != (
                "dealloc", "cta_group::1", "sync", "aligned", "b32"):
            raise unsupported(ins)
        _collective(ins, mask)
        ncols = _uniform(w, ins, ins.operands[1], mask, "column count")
        addresses = w.read(ins.operands[0], "u32")[mask]
        if (addresses != addresses[0]).any():
            raise unsupported(ins, "deallocation with different tensor-memory addresses across lanes")
        tmem = _tmem(cta)
        start = int(addresses[0])
        if start not in tmem.allocations and (start, ncols) in tmem.freed:
            # Kernels that run on hardware do this: every warpgroup's first warp releases the
            # one allocation (hgemm_v7 to v9 of the book "Modern GPU Programming for MLSys").
            # What the hardware does with the repeat is not known; here it does nothing.
            cta.findings.add("tmem-dealloc-repeated", f"tensor memory at column {start} ({ncols} "
                             "columns) was already released by an earlier tcgen05.dealloc; this "
                             "one is ignored", w, ins, np.flatnonzero(mask))
            return
        if start in tmem.allocations and tmem.allocations[start][0] == ncols:
            _check_tmem_overwrite(w, ins, tmem.pending[:, start:start + ncols],
                                  np.flatnonzero(mask), dealloc=True)
        tmem.dealloc(start, ncols)
        cta._wake("tensor-memory allocation")
    elif kind == "mma":
        for l in np.flatnonzero(mask):
            _mma(w, ins, int(l), pair)
    elif kind == "commit":
        allowed = {"commit", "cta_group::1", "mbarrier::arrive::one", "shared::cluster", "b64",
                   "multicast::cluster"}
        multicast = "multicast::cluster" in ins.mods
        if not {"mbarrier::arrive::one", "b64"} <= set(ins.mods) \
                or set(ins.mods) - allowed or len(ins.operands) != 1 + multicast:
            raise unsupported(ins)
        addresses = _address(w, ins, ins.operands[0])
        if "shared::cluster" not in ins.mods:
            shared = (addresses >= GENERIC_WINDOW) & (addresses < GENERIC_WINDOW + CLUSTER_WINDOW)
            if not shared[mask].all():
                raise ExecutionError("generic tcgen05.commit address does not point to shared memory")
            addresses = addresses - np.uint64(GENERIC_WINDOW)
        for l in np.flatnonzero(mask):
            l = int(l)
            # With multicast the arrival goes to the mbarrier at this offset in every CTA
            # of the cluster whose rank is set in the mask.
            target, offset = resolve_shared(cta, int(addresses[l]))
            targets = [target]
            if multicast:
                ranks = int(w.read_address(ins.operands[1])[l]) & 0xFFFF  # a 16-bit operand
                targets = [cta.peer(r) for r in range(16) if ranks >> r & 1]
            bars = [t.mbarrier(offset, w, ins, [l]) for t in targets]
            # A commit covers every earlier MMA from this thread, including
            # operations already tracked by another barrier. Other lanes in
            # the warp cannot commit on the issuing thread's behalf.
            covered = tuple(o for o in w.tcgen_issued if o.lane == l)
            operation = cta.new_async("commit", w, ins, l)
            released = frozenset(w.seen)

            def land(bars=bars, covered=covered, l=l, released=released):
                for bar in bars:
                    bar.knowledge.update(o.id for o in covered)
                    # Do not acquire knowledge the issuing warp learns later,
                    # while this asynchronous arrival is waiting to complete.
                    bar.arrive(1, w, ins, [l], knowledge=released)

            cta.schedule(operation, land)
    elif kind == "ld":
        _transfer(w, ins, mask, store=False)
    elif kind == "st":
        _transfer(w, ins, mask, store=True)
    else:
        raise unsupported(ins)
