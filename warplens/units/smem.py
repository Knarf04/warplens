"""Shared memory (SMEM) of one CTA.

Shared addresses are byte offsets from the start of the CTA's shared memory.

Every byte carries shadow state: which warp last wrote it and last read it,
and in which barrier epoch. The epoch advances each time a barrier completes.
Two warps touching the same byte in the same epoch, at least one of them
writing, have no barrier ordering them: that is a race on real hardware even
if this run's schedule happened to order them. Reads of never-written bytes
are reported too.
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
        self._reader = np.full(size, -1, dtype=np.int32)
        self._read_epoch = np.zeros(size, dtype=np.int32)
        # Atomics must not erase an earlier ordinary access in the same epoch.
        self._ordinary_writer = np.full(size, -1, dtype=np.int32)
        self._ordinary_write_epoch = np.zeros(size, dtype=np.int32)
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

    def read(self, offsets: np.ndarray, dtype: np.dtype, warp: int, *, atomic: bool = False):
        """Returns (values, accesses that touched a never-written byte, accesses
        that race with another warp's write)."""
        idx = self._index(offsets, dtype.itemsize, "read")
        values = self.data[idx].view(dtype).ravel()
        writer = self._writer[idx]
        unwritten = np.flatnonzero((writer == -1).any(axis=1))
        racy = (writer >= 0) & (writer != warp) & (self._write_epoch[idx] == self.epoch)
        if atomic:
            ordinary = self._ordinary_writer[idx]
            racy = (ordinary >= 0) & (ordinary != warp) & (self._ordinary_write_epoch[idx] == self.epoch)
        else:
            # An atomic read is immediately followed by its atomic write. Keep
            # earlier ordinary readers so that the write can still diagnose them.
            self._reader[idx] = warp
            self._read_epoch[idx] = self.epoch
        return values, unwritten, np.flatnonzero(racy.any(axis=1))

    def write(self, offsets: np.ndarray, values: np.ndarray, warp: int, *, atomic: bool = False) -> np.ndarray:
        """Returns the accesses that race with another warp's read or write."""
        size = values.dtype.itemsize
        idx = self._index(offsets, size, "write")
        writer, reader = self._writer[idx], self._reader[idx]
        racy = (writer >= 0) & (writer != warp) & (self._write_epoch[idx] == self.epoch)
        if atomic:
            ordinary = self._ordinary_writer[idx]
            racy = (ordinary >= 0) & (ordinary != warp) & (self._ordinary_write_epoch[idx] == self.epoch)
        read_racy = (reader >= 0) & (reader != warp) & (self._read_epoch[idx] == self.epoch)
        racy |= read_racy
        self.data[idx] = np.ascontiguousarray(values).view(np.uint8).reshape(len(offsets), size)
        self._writer[idx] = warp
        if not atomic:
            self._ordinary_writer[idx] = warp
            self._ordinary_write_epoch[idx] = self.epoch
        self._write_epoch[idx] = self.epoch
        self.unfenced[idx] = True
        return np.flatnonzero(racy.any(axis=1))

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
        self._ordinary_writer[idx] = self._writer[idx]
        if writer_warp is not None:
            self._write_epoch[idx] = self.epoch
            self._ordinary_write_epoch[idx] = self.epoch
        self.unfenced[idx] = writer_warp is not None

    def fence_async(self, warp: int) -> None:
        """fence.proxy.async by `warp`: its own earlier writes, and writes that a
        completed barrier already ordered before it, become visible to TMA."""
        if self.unfenced.any():
            ordered = (self._writer == warp) | (self._write_epoch < self.epoch)
            self.unfenced &= ~ordered
