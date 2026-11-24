"""Shared memory (SMEM) of one CTA.

Shared addresses are byte offsets from the start of the CTA's shared memory.

Every byte carries shadow state: which warp last wrote it and last read it,
and in which barrier epoch. The epoch advances each time a barrier completes.
Two warps touching the same byte in the same epoch, at least one of them
writing, have no barrier ordering them: that is a race on real hardware even
if this run's schedule happened to order them. Reads of never-written bytes
are reported too.

`atom` instructions are the exception: two warps updating the same byte atomically
are ordered by the instruction itself, so they do not race with each other. An
atom still races with another warp's ordinary load or store of that byte, in
either order. The shadow state for this is separate (`_atom_warp`): the warp
whose atoms touched the byte in the current epoch, or ATOM_MANY for several.
"""

from __future__ import annotations

import numpy as np

from ..errors import MemoryFault

# Largest shared memory one CTA may have on sm_100a, static plus dynamic: 227 KB. The CUDA
# C++ Programming Guide's table of technical specifications gives 227 KB per thread block
# and 228 KB per SM for compute capability 10.x, and CUTLASS sizes its sm_100 kernels by
# `sm100_smem_capacity_bytes = 232448` (cutlass/arch/arch.h). The shim libraries report the
# same number to programs as WL_MAX_SMEM_BYTES (warplens/shim/protocol.h), which
# tests/test_sim_bug_regressions.py keeps equal to this one.
MAX_SMEM_BYTES = 227 * 1024

# Shadow "writer" value for bytes last written by TMA rather than by a warp.
ASYNC_PROXY = -2

# Shadow "writer" value for bytes whose last write was an `atom` and that no warp has
# written with an ordinary store in the current epoch. Negative, so the byte counts as
# written but not as an ordinary write that an atom of another warp would race with.
ATOMIC = -3
# `_atom_warp` value for bytes that more than one warp updated atomically in an epoch.
ATOM_MANY = -3

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


class SharedMemory:
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.uint8)
        self.epoch = 0  # advanced by the CTA whenever a barrier completes
        self._writer = np.full(size, -1, dtype=np.int32)  # warp id, -1 = never written
        self._write_epoch = np.zeros(size, dtype=np.int32)
        # Proxy-fence ownership must not reuse the race history: an atomic may
        # preserve an earlier ordinary writer or have several atomic contenders.
        self._last_writer = np.full(size, -1, dtype=np.int32)
        self._reader = np.full(size, -1, dtype=np.int32)
        self._read_epoch = np.zeros(size, dtype=np.int32)
        self._atom_warp = np.full(size, -1, dtype=np.int32)  # -1 = no atom yet, or ATOM_MANY
        self._atom_epoch = np.zeros(size, dtype=np.int32)
        # TMA (async proxy) shadow state; ids refer to Cta.async_ops.
        self.async_active = False  # stays False for kernels that never use TMA
        self.async_writer = np.zeros(size, dtype=np.int32)  # TMA load that last targeted the byte
        self.async_reader = np.zeros(size, dtype=np.int32)  # TMA store that last read the byte
        self.unfenced = np.zeros(size, dtype=np.bool_)  # ordinary write not yet fenced for TMA

    def _index(self, offsets: np.ndarray, size: int, verb: str) -> np.ndarray:
        ok = offsets <= np.uint64(self.size - size) if self.size >= size else np.zeros(len(offsets), bool)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            raise MemoryFault(
                f"shared-memory {verb} of {size} byte(s) at offset {int(offsets[bad[0]])} is outside "
                f"the CTA's {self.size} bytes of shared memory", bad, offsets[bad])
        return offsets.astype(np.intp)[:, None] + np.arange(size, dtype=np.intp)

    def read(self, offsets: np.ndarray, dtype: np.dtype, warp: int):
        """Returns (values, accesses that touched a never-written byte, accesses
        that race with another warp's write)."""
        idx = self._index(offsets, dtype.itemsize, "read")
        values = self.data[idx].view(dtype).ravel()
        writer = self._writer[idx]
        unwritten = np.flatnonzero((writer == -1).any(axis=1))
        racy = (writer >= 0) & (writer != warp) & (self._write_epoch[idx] == self.epoch)
        racy |= self._other_atom(idx, warp)
        self._reader[idx] = warp
        self._read_epoch[idx] = self.epoch
        return values, unwritten, np.flatnonzero(racy.any(axis=1))

    def write(self, offsets: np.ndarray, values: np.ndarray, warp: int) -> np.ndarray:
        """Returns the accesses that race with another warp's read or write."""
        size = values.dtype.itemsize
        idx = self._index(offsets, size, "write")
        writer, reader = self._writer[idx], self._reader[idx]
        racy = (writer >= 0) & (writer != warp) & (self._write_epoch[idx] == self.epoch)
        racy |= (reader >= 0) & (reader != warp) & (self._read_epoch[idx] == self.epoch)
        racy |= self._other_atom(idx, warp)
        self.data[idx] = np.ascontiguousarray(values).view(np.uint8).reshape(len(offsets), size)
        self._writer[idx] = warp
        self._last_writer[idx] = warp
        self._write_epoch[idx] = self.epoch
        self.unfenced[idx] = True
        return np.flatnonzero(racy.any(axis=1))

    def _other_atom(self, idx: np.ndarray, warp: int) -> np.ndarray:
        """Bytes that a warp other than `warp` updated atomically in this epoch."""
        by = self._atom_warp[idx]
        return (by != -1) & (by != warp) & (self._atom_epoch[idx] == self.epoch)

    def atomic(self, offsets: np.ndarray, dtype: np.dtype, warp: int, update):
        """An `atom`: replace the value at each offset by `update(old values)`.

        The offsets must not overlap one another; the caller applies lanes that share
        bytes one after another. Returns (old values, accesses that touched a
        never-written byte, accesses that race with another warp's ordinary read or
        write). Atoms of other warps are not a race: see the module's docstring.
        """
        size = dtype.itemsize
        idx = self._index(offsets, size, "atomic update")
        old = self.data[idx].view(dtype).ravel()
        writer, reader = self._writer[idx], self._reader[idx]
        unwritten = np.flatnonzero((writer == -1).any(axis=1))
        plain = (writer >= 0) & (self._write_epoch[idx] == self.epoch)  # ordinary write, this epoch
        racy = plain & (writer != warp)
        racy |= (reader >= 0) & (reader != warp) & (self._read_epoch[idx] == self.epoch)
        new = np.ascontiguousarray(np.asarray(update(old), dtype=dtype))
        self.data[idx] = new.view(np.uint8).reshape(len(offsets), size)
        # An ordinary write of this epoch stays on record: it races with a later atom of
        # another warp, which this atom does not.
        self._writer[idx] = np.where(plain, writer, ATOMIC)
        self._last_writer[idx] = warp
        self._write_epoch[idx] = self.epoch
        by = self._atom_warp[idx]
        several = (by != -1) & (by != warp) & (self._atom_epoch[idx] == self.epoch)
        self._atom_warp[idx] = np.where(several, ATOM_MANY, warp)
        self._atom_epoch[idx] = self.epoch
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
            self._write_epoch[idx] = self.epoch
        self.unfenced[idx] = writer_warp is not None

    def fence_async(self, warp: int) -> None:
        """fence.proxy.async by `warp`: its own earlier writes, and writes that a
        completed barrier already ordered before it, become visible to TMA."""
        if self.unfenced.any():
            ordered = (self._last_writer == warp) | (self._write_epoch < self.epoch)
            self.unfenced &= ~ordered
