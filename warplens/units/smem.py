"""Shared memory (SMEM) of one CTA.

Shared addresses are byte offsets from the start of the CTA's shared memory.

Every byte carries shadow state for the race check: who last wrote it and who
read it, each with the accessing warp's clock at that moment (`core/clocks.py`).
Two warps touching the same byte, at least one of them writing, race unless a
barrier that both took part in ordered the earlier access before the later
warp: that is a race on real hardware even if this run's schedule happened to
order them. Reads of never-written bytes are reported too.

One last writer per byte is enough, because two unordered writes are
themselves reported. Reads do not conflict with each other, so several warps
can have unordered reads of a byte on record at once, and a later write must
be checked against each of them. `_Accesses` keeps one (warp, clock) per byte
while the readers are ordered one after another, and a clock per warp for the
bytes where they are not.

`atom` instructions are the exception: two warps updating the same byte atomically
are ordered by the instruction itself, so they do not race with each other. An
atom still races with another warp's ordinary load or store of that byte, in
either order. Atomic updates are therefore recorded like reads, in a second
`_Accesses` (`_atoms`), and checked against ordinary loads as well as stores.
"""

from __future__ import annotations

import numpy as np

from ..core.clocks import Clocks
from ..errors import MemoryFault

# Largest shared memory one CTA may have on sm_100a, static plus dynamic: 227 KB. The CUDA
# C++ Programming Guide's table of technical specifications gives 227 KB per thread block
# and 228 KB per SM for compute capability 10.x, and CUTLASS sizes its sm_100 kernels by
# `sm100_smem_capacity_bytes = 232448` (cutlass/arch/arch.h). The shim libraries report the
# same number to programs as WL_MAX_SMEM_BYTES (warplens/shim/protocol.h), which
# tests/test_sim_bug_regressions.py keeps equal to this one.
MAX_SMEM_BYTES = 227 * 1024

# The shared memory of one CTA that a program gets without asking: what the device reports as
# its shared memory per block (WL_DEFAULT_SMEM_BYTES in warplens/shim/protocol.h, kept equal
# by tests/test_sim_bug_regressions.py). 48 KB is what every compute capability in the CUDA
# C++ Programming Guide's table has; a B200 has not been asked (RT-03).
DEFAULT_SMEM_BYTES = 48 * 1024


def default_max_dynamic(static: int) -> int:
    """The most dynamic shared memory a launch of a kernel with `static` bytes of static
    shared memory may use when the program has not set the kernel's maximum
    (cudaFuncSetAttribute with cudaFuncAttributeMaxDynamicSharedMemorySize, or
    cuFuncSetAttribute): what cudaFuncGetAttributes reports as maxDynamicSharedSizeBytes.

    Measured on an RTX 3080 Ti at its own limits (validation/README.md, "Launch errors"): the
    rest of the default, for a kernel whose static shared memory fits the default; and the
    rest of the opt-in maximum, for a kernel whose static shared memory is larger than the
    default, which therefore launches without any opt-in. Setting the attribute replaces
    this number, with a smaller one too, and is refused above `MAX_SMEM_BYTES - static`.
    Only programs run under the shim are held to it: `Device.launch` has no such step."""
    return max((DEFAULT_SMEM_BYTES if static <= DEFAULT_SMEM_BYTES else MAX_SMEM_BYTES) - static, 0)

# Shadow "writer" value for bytes last written by TMA rather than by a warp.
ASYNC_PROXY = -2

# Shadow "writer" value for bytes that only `atom` instructions have written. Negative, so
# the byte counts as written but not as an ordinary write that an atom of another warp
# would race with.
ATOMIC = -3

# Shared pointers carry the CTA rank above the byte offset. In particular,
# bit 24 selects the peer within a pair, as used by CUTLASS SM100 TMA code.
# Other address bits are simulator-defined, not a hardware encoding claim.
CLUSTER_STRIDE = 1 << 24
CLUSTER_WINDOW = 16 * CLUSTER_STRIDE


def local_shared_offset(address):
    """Remove valid cluster routing bits for an explicitly CTA-local access."""
    if np.isscalar(address):
        return int(address) % CLUSTER_STRIDE if 0 <= address < CLUSTER_WINDOW else int(address)
    return np.where(address < CLUSTER_WINDOW, address % np.uint64(CLUSTER_STRIDE), address)


def resolve_shared(cta, address: int):
    """Resolve a shared::cluster pointer to its target CTA and byte offset."""
    rank, offset = divmod(int(address), CLUSTER_STRIDE)
    return (cta if rank == cta.rank else cta.peer(rank)), offset

# Where shared memory appears in the generic address space (`cvta.shared`).
GENERIC_WINDOW = 0x1_0000_0000


class _Accesses:
    """Per byte, accesses that do not conflict with each other: reads, or atomic updates.

    While each accessor of a byte is ordered after the one before, the byte has one entry
    (`who`, `clock`): whatever is ordered after the latest accessor is ordered after the
    earlier ones too. When a warp accesses a byte whose recorded accessor is not ordered
    before it, the byte switches to one clock per warp in `table`, until `clear`.
    """

    def __init__(self, size: int, clocks: Clocks):
        self.clocks = clocks
        self.used = False  # nothing recorded yet: callers skip `unordered` and `clear`
        self.who = np.full(size, -1, dtype=np.int32)  # -1: none on record, or see `table`
        self.clock = np.zeros(size, dtype=np.int32)
        self.several = np.zeros(size, dtype=np.bool_)  # the byte's accessors are in `table`
        self.any_several = False
        self.table = None  # (size, warps) clocks, 0 = no access; rows are valid where `several`

    def _table(self) -> np.ndarray:
        n = self.clocks.n
        if self.table is None:
            self.table = np.zeros((len(self.who), n), dtype=np.int32)  # pages are touched on use
        elif self.table.shape[1] < n:
            wider = np.zeros((len(self.who), n), dtype=np.int32)
            wider[:, :self.table.shape[1]] = self.table
            self.table = wider
        return self.table

    def unordered(self, idx: np.ndarray, warp: int, row: np.ndarray) -> np.ndarray:
        """Bytes with an access on record by another warp that is not ordered before `warp`."""
        who = self.who[idx]
        out = (who >= 0) & (who != warp) & (self.clock[idx] > row.take(who, mode="clip"))
        if self.any_several:
            several = self.several[idx]
            if several.any():
                later = self._table()[idx[several]] > row
                later[:, warp] = False
                out[several] = later.any(axis=1)
        return out

    def record(self, idx: np.ndarray, warp: int, row: np.ndarray) -> None:
        self.used = True
        flat = idx.ravel()
        who = self.who[flat]
        apart = (who >= 0) & (who != warp) & (self.clock[flat] > row.take(who, mode="clip"))
        if apart.any():
            at = flat[apart]
            table = self._table()
            table[at] = 0
            table[at, who[apart]] = self.clock[at]
            self.several[at] = True
            self.who[at] = -1
            self.any_several = True
        if self.any_several:
            several = self.several[flat]
            if several.any():
                self._table()[flat[several], warp] = row[warp]
                flat = flat[~several]
        self.who[flat] = warp
        self.clock[flat] = row[warp]

    def clear(self, idx: np.ndarray) -> None:
        self.who[idx] = -1
        if self.any_several:
            self.several[idx] = False


class SharedMemory:
    def __init__(self, size: int, clocks: Clocks | None = None):
        self.size = size
        self.data = np.zeros(size, dtype=np.uint8)
        # The clocks of the cluster's warps. A warp is named here by its id among them
        # (`Warp.sync_id`), so that a peer CTA's warp is told apart from a local one.
        self.clocks = clocks if clocks is not None else Clocks()
        # The last ordinary store: warp, -1 = never written, ASYNC_PROXY or ATOMIC.
        self._writer = np.full(size, -1, dtype=np.int32)
        self._write_clock = np.zeros(size, dtype=np.int32)
        # Proxy-fence ownership must not reuse the race history: an atomic
        # preserves an earlier ordinary writer and may have several contenders.
        self._last_writer = np.full(size, -1, dtype=np.int32)
        self._last_clock = np.zeros(size, dtype=np.int32)
        self._reads = _Accesses(size, self.clocks)
        self._atoms = _Accesses(size, self.clocks)
        # TMA (async proxy) shadow state; ids refer to Cta.async_ops.
        self.async_active = False  # stays False for kernels that never use TMA
        self.async_writer = np.zeros(size, dtype=np.int32)  # TMA load that last targeted the byte
        self.async_reader = np.zeros(size, dtype=np.int32)  # TMA store that last read the byte
        self.unfenced = np.zeros(size, dtype=np.bool_)  # ordinary write not yet fenced for TMA

    def use_clocks(self, clocks: Clocks) -> None:
        """Share the clocks of a cluster. Call before any access."""
        self.clocks = self._reads.clocks = self._atoms.clocks = clocks

    def _index(self, offsets: np.ndarray, size: int, verb: str) -> np.ndarray:
        ok = offsets <= np.uint64(self.size - size) if self.size >= size else np.zeros(len(offsets), bool)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            raise MemoryFault(
                f"shared-memory {verb} of {size} byte(s) at offset {int(offsets[bad[0]])} is outside "
                f"the CTA's {self.size} bytes of shared memory", bad, offsets[bad])
        return offsets.astype(np.intp)[:, None] + np.arange(size, dtype=np.intp)

    def _other_write(self, idx: np.ndarray, writer: np.ndarray, warp: int, row: np.ndarray) -> np.ndarray:
        """Bytes whose last ordinary store, by another warp, is not ordered before `warp`."""
        return (writer >= 0) & (writer != warp) & (self._write_clock[idx] > row.take(writer, mode="clip"))

    def read(self, offsets: np.ndarray, dtype: np.dtype, warp: int):
        """Returns (values, accesses that touched a never-written byte, accesses
        that race with another warp's write)."""
        idx = self._index(offsets, dtype.itemsize, "read")
        row = self.clocks.row(warp)
        values = self.data[idx].view(dtype).ravel()
        writer = self._writer[idx]
        unwritten = np.flatnonzero((writer == -1).any(axis=1))
        racy = self._other_write(idx, writer, warp, row)
        if self._atoms.used:
            racy |= self._atoms.unordered(idx, warp, row)
        self._reads.record(idx, warp, row)
        return values, unwritten, np.flatnonzero(racy.any(axis=1))

    def write(self, offsets: np.ndarray, values: np.ndarray, warp: int) -> np.ndarray:
        """Returns the accesses that race with another warp's read or write."""
        size = values.dtype.itemsize
        idx = self._index(offsets, size, "write")
        row = self.clocks.row(warp)
        racy = self._other_write(idx, self._writer[idx], warp, row)
        if self._reads.used:
            racy |= self._reads.unordered(idx, warp, row)
        if self._atoms.used:
            racy |= self._atoms.unordered(idx, warp, row)
        self.data[idx] = np.ascontiguousarray(values).view(np.uint8).reshape(len(offsets), size)
        self._writer[idx] = self._last_writer[idx] = warp
        self._write_clock[idx] = self._last_clock[idx] = row[warp]
        # This store was checked against every access on record. A later access that is not
        # ordered after one of them is not ordered after this store either, unless this
        # store itself raced with that access and was reported.
        if self._reads.used:
            self._reads.clear(idx)
        if self._atoms.used:
            self._atoms.clear(idx)
        self.unfenced[idx] = True
        return np.flatnonzero(racy.any(axis=1))

    def atomic(self, offsets: np.ndarray, dtype: np.dtype, warp: int, update):
        """An `atom`: replace the value at each offset by `update(old values)`.

        The offsets must not overlap one another; the caller applies lanes that share
        bytes one after another. Returns (old values, accesses that touched a
        never-written byte, accesses that race with another warp's ordinary read or
        write). Atoms of other warps are not a race: see the module's docstring.
        """
        size = dtype.itemsize
        idx = self._index(offsets, size, "atomic update")
        row = self.clocks.row(warp)
        old = self.data[idx].view(dtype).ravel()
        writer = self._writer[idx]
        unwritten = np.flatnonzero((writer == -1).any(axis=1))
        racy = self._other_write(idx, writer, warp, row)
        racy |= self._reads.unordered(idx, warp, row)
        new = np.ascontiguousarray(np.asarray(update(old), dtype=dtype))
        self.data[idx] = new.view(np.uint8).reshape(len(offsets), size)
        # An ordinary store stays on record: it races with a later atom of another warp
        # that is not ordered after it, which this atom does not. So do the reads.
        self._writer[idx] = np.where(writer >= 0, writer, ATOMIC)
        self._last_writer[idx] = warp
        self._last_clock[idx] = row[warp]
        self._atoms.record(idx, warp, row)
        self.unfenced[idx] = True
        return old, unwritten, np.flatnonzero(racy.any(axis=1))

    # -- async proxy (TMA) ---------------------------------------------------
    def index(self, offsets: np.ndarray, size: int, verb: str) -> np.ndarray:
        return self._index(offsets, size, verb)

    def has_uninitialized_bytes(self, idx: np.ndarray) -> bool:
        """Whether bytes have neither an ordinary writer nor a queued TMA load.

        A pending TMA load is checked separately for missing synchronization.
        """
        return bool(((self._writer[idx] == -1) & (self.async_writer[idx] == 0)).any())

    def begin_async_write(self, idx: np.ndarray, op_id: int) -> None:
        self.async_writer[idx] = op_id
        self.async_active = True

    def finish_async_write(self, idx: np.ndarray, data: np.ndarray, *, writer_warp=None) -> None:
        self.data[idx] = data
        # Non-bulk cp.async is a generic-proxy write, unlike a TMA load. A later
        # bulk copy/MMA still needs fence.proxy.async before consuming these bytes.
        self._writer[idx] = ASYNC_PROXY if writer_warp is None else writer_warp
        self._last_writer[idx] = self._writer[idx]
        if writer_warp is not None:
            self._write_clock[idx] = self._last_clock[idx] = self.clocks.row(writer_warp)[writer_warp]
        self.unfenced[idx] = writer_warp is not None

    def fence_async(self, warp: int) -> None:
        """fence.proxy.async by `warp`: its own earlier writes, and writes that a
        barrier it took part in ordered before it, become visible to TMA."""
        if self.unfenced.any():
            last = self._last_writer
            ordered = (last == warp) | ((last >= 0) & (
                self._last_clock <= self.clocks.row(warp).take(last, mode="clip")))
            self.unfenced &= ~ordered
