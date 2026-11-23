"""Global memory (GMEM): a sparse 64-bit address space made of allocations.

Allocations are separated by unmapped guard gaps, so an access that runs off
the end of a buffer faults instead of landing in its neighbor.
"""

from __future__ import annotations

import numpy as np

from ..errors import ExecutionError, MemoryFault


class GlobalMemory:
    BASE = 0x2_0000_0000
    GUARD = 4096

    def __init__(self) -> None:
        self._bufs: list[np.ndarray] = []
        self._bases = np.zeros(0, dtype=np.uint64)
        self._sizes = np.zeros(0, dtype=np.uint64)
        self._next = self.BASE

    def alloc(self, nbytes: int, align: int = 256) -> int:
        addr = -(-self._next // align) * align
        self._bufs.append(np.zeros(nbytes, dtype=np.uint8))
        self._bases = np.append(self._bases, np.uint64(addr))
        self._sizes = np.append(self._sizes, np.uint64(nbytes))
        self._next = addr + nbytes + self.GUARD
        return addr

    def buffer(self, addr: int) -> np.ndarray:
        """The bytes of the allocation that starts at `addr`."""
        (hit,) = np.nonzero(self._bases == np.uint64(addr))
        if len(hit) != 1:
            raise KeyError(f"no allocation starts at {addr:#x}")
        return self._bufs[int(hit[0])]

    def free(self, addr: int) -> None:
        """Release an allocation without reusing its address."""
        if addr == 0:
            return
        (hit,) = np.nonzero(self._bases == np.uint64(addr))
        if len(hit) != 1:
            raise ExecutionError(f"cannot free {addr:#x}: no allocation starts at that address")
        index = int(hit[0])
        self._bufs.pop(index)
        self._bases = np.delete(self._bases, index)
        self._sizes = np.delete(self._sizes, index)

    def validate_range(self, addr: int, nbytes: int, verb: str) -> None:
        """Check a host transfer before allocating an index for every byte."""
        if nbytes:
            self._locate(np.array([addr], dtype=np.uint64), nbytes, verb)

    def _locate(self, addrs: np.ndarray, size: int, verb: str):
        idx = np.searchsorted(self._bases, addrs, side="right").astype(np.intp) - 1
        ok = idx >= 0
        safe = np.where(ok, idx, 0)
        if len(self._bases):
            off = addrs - self._bases[safe]
            sizes = self._sizes[safe]
            ok &= (sizes >= size) & (off <= sizes - np.minimum(sizes, np.uint64(size)))
        else:
            off = addrs
            ok &= False
        if not ok.all():
            bad = np.flatnonzero(~ok)
            raise MemoryFault(self._describe(int(addrs[bad[0]]), size, verb), bad, addrs[bad])
        return safe, off.astype(np.intp)

    def _describe(self, addr: int, size: int, verb: str) -> str:
        msg = f"{verb} of {size} byte(s) at {addr:#x} is outside every allocation"
        i = int(np.searchsorted(self._bases, np.uint64(addr), side="right")) - 1
        if i >= 0:
            base, n = int(self._bases[i]), int(self._sizes[i])
            past = addr - (base + n)
            where = f"{past} byte(s) past the end" if past >= 0 else "straddling the end"
            msg += f" ({where} of the {n}-byte allocation at {base:#x})"
        elif addr < 4096:
            msg += " (null pointer?)"
        return msg

    def read(self, addrs: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Load one `dtype` value from each address."""
        size = dtype.itemsize
        idx, off = self._locate(addrs, size, "read")
        out = np.empty(len(addrs), dtype=dtype)
        span = np.arange(size, dtype=np.intp)
        for i in np.unique(idx):
            sel = idx == i
            out[sel] = self._bufs[i][off[sel, None] + span].view(dtype).ravel()
        return out

    def write(self, addrs: np.ndarray, values: np.ndarray) -> None:
        """Store one value at each address. For duplicate addresses the last one wins."""
        size = values.dtype.itemsize
        idx, off = self._locate(addrs, size, "write")
        raw = np.ascontiguousarray(values).view(np.uint8).reshape(len(addrs), size)
        span = np.arange(size, dtype=np.intp)
        for i in np.unique(idx):
            sel = idx == i
            self._bufs[i][off[sel, None] + span] = raw[sel]
