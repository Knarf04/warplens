"""Tensor maps: the descriptors a TMA instruction reads to find its data.

On real hardware `cuTensorMapEncodeTiled` fills an opaque 128-byte CUtensorMap.
warplens emulates the encode call, so it defines its own 128-byte encoding
here and the TMA unit decodes it. A tensor map made by a real driver cannot
be used, and vice versa.

Dimension order follows TMA: dimension 0 is the innermost (fastest varying).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..errors import ExecutionError, LaunchError

SIZE = 128
MAX_RANK = 5
MAGIC = b"WLTM"
_FORMAT = "<4s5B7xQ5I4x4Q5H5B"  # magic, rank/elem/float/swizzle/nan, address, dims, strides, box, steps
assert struct.calcsize(_FORMAT) <= SIZE

# Swizzle mode -> (name, span in bytes of the inner box dimension, number of swizzled bits)
SWIZZLE_MODES = {0: ("none", None), 1: ("32B", 32), 2: ("64B", 64), 3: ("128B", 128)}
_SWIZZLE_BY_NAME = {name.lower(): bits for bits, (name, _) in SWIZZLE_MODES.items()}


@dataclass(frozen=True, slots=True)
class TensorMapInfo:
    rank: int
    elem_size: int
    is_float: bool
    swizzle: int  # 0 = none, 1/2/3 = 32B/64B/128B
    nan_fill: bool  # fill out-of-bounds elements with NaN instead of zero
    address: int
    dims: tuple[int, ...]  # elements per dimension
    strides: tuple[int, ...]  # bytes between consecutive indices of dimensions 1..rank-1
    box: tuple[int, ...]  # elements traversed per dimension
    steps: tuple[int, ...]  # element stride per dimension (dimension 0 is always 1)

    @property
    def counts(self) -> tuple[int, ...]:
        """Elements actually transferred per dimension."""
        return (self.box[0], *(-(-b // s) for b, s in zip(self.box[1:], self.steps[1:])))

    @property
    def box_bytes(self) -> int:
        return int(np.prod(self.counts, dtype=np.int64)) * self.elem_size

    @property
    def swizzle_period(self) -> int:
        """Bytes after which the swizzle pattern repeats (a buffer should be aligned to it)."""
        return 128 << self.swizzle if self.swizzle else 1


def encode(info: TensorMapInfo) -> bytes:
    pad = lambda values, n: tuple(values) + (0,) * (n - len(values))
    raw = struct.pack(_FORMAT, MAGIC, info.rank, info.elem_size, info.is_float, info.swizzle,
                      info.nan_fill, info.address, *pad(info.dims, 5), *pad(info.strides, 4),
                      *pad(info.box, 5), *pad(info.steps, 5))
    return raw.ljust(SIZE, b"\0")


def decode(raw: bytes) -> TensorMapInfo:
    fields = struct.unpack_from(_FORMAT, raw)
    magic, rank, elem_size, is_float, swizzle, nan_fill, address = fields[:7]
    if magic != MAGIC or not 1 <= rank <= MAX_RANK:
        raise ExecutionError("the tensor-map operand does not point at a tensor map created by "
                             "Device.tensor_map_tiled")
    dims, strides, box, steps = fields[7:12], fields[12:16], fields[16:21], fields[21:26]
    return TensorMapInfo(rank, elem_size, bool(is_float), swizzle, bool(nan_fill), address,
                         dims[:rank], strides[:rank - 1], box[:rank], steps[:rank])


def make_tiled(*, address: int, shape, dtype: np.dtype, box, swizzle=None, element_strides=None,
               oob_fill: str = "zero") -> TensorMapInfo:
    """Validate like cuTensorMapEncodeTiled. `shape`, `box` and `element_strides`
    are in NumPy order (outermost first) and are reversed into TMA order here."""
    rank = len(shape)
    if not 1 <= rank <= MAX_RANK:
        raise LaunchError(f"a tensor map needs 1 to {MAX_RANK} dimensions, got {rank}")
    box = (box,) if isinstance(box, (int, np.integer)) else tuple(box)
    steps = (1,) * rank if element_strides is None else tuple(element_strides)
    if len(box) != rank or len(steps) != rank:
        raise LaunchError(f"box and element_strides need one entry per dimension ({rank})")
    dims, box, steps = tuple(shape)[::-1], box[::-1], steps[::-1]
    elem = dtype.itemsize
    if dtype.kind not in "iuf" or elem not in (1, 2, 4, 8):
        raise LaunchError(f"unsupported tensor-map data type {dtype}")

    key = "none" if swizzle is None else str(swizzle).lower()
    if key not in _SWIZZLE_BY_NAME:
        raise LaunchError(f"unknown swizzle {swizzle!r}; choose None, '32B', '64B' or '128B'")
    bits = _SWIZZLE_BY_NAME[key]
    if oob_fill not in ("zero", "nan"):
        raise LaunchError(f"oob_fill must be 'zero' or 'nan', got {oob_fill!r}")
    if oob_fill == "nan" and dtype.kind != "f":
        raise LaunchError("NaN out-of-bounds fill needs a floating-point tensor")

    if address % 16:
        raise LaunchError("the tensor's address must be 16-byte aligned")
    if any(not 1 <= d <= 2**32 for d in dims):
        raise LaunchError(f"tensor dimensions must be between 1 and 2^32, got {dims[::-1]}")
    if any(not 1 <= b <= 256 for b in box):
        raise LaunchError(f"box dimensions must be between 1 and 256, got {box[::-1]}")
    if any(not 1 <= s <= 8 for s in steps):
        raise LaunchError(f"element strides must be between 1 and 8, got {steps[::-1]}")
    inner = box[0] * elem
    if inner % 16:
        raise LaunchError(f"the innermost box dimension is {inner} bytes; it must be a multiple of 16")
    span = SWIZZLE_MODES[bits][1]
    if span is not None and inner > span:
        raise LaunchError(f"the innermost box dimension is {inner} bytes, more than the "
                          f"{span}-byte span of swizzle {SWIZZLE_MODES[bits][0]}")
    strides, stride = [], elem
    for d in dims[:-1]:
        stride *= d
        if stride % 16 or stride >= 2**40:
            raise LaunchError(f"a stride of {stride} bytes is not allowed: strides must be multiples "
                              "of 16 below 2^40 (pad the inner dimensions)")
        strides.append(stride)
    return TensorMapInfo(rank, elem, dtype.kind == "f", bits, oob_fill == "nan", address,
                         dims, tuple(strides), box, (1, *steps[1:]))


def element_addresses(info: TensorMapInfo, coords) -> tuple[np.ndarray, np.ndarray]:
    """Global address of every element of the box at `coords`, flattened with
    dimension 0 fastest, and which of them lie inside the tensor."""
    addr = np.zeros(1, dtype=np.int64)
    inside = np.ones(1, dtype=np.bool_)
    scale = (info.elem_size, *info.strides)
    for d in range(info.rank - 1, -1, -1):  # outermost first, so dimension 0 ends up fastest
        index = int(coords[d]) + np.arange(info.counts[d], dtype=np.int64) * info.steps[d]
        ok = (index >= 0) & (index < info.dims[d])
        addr = (addr[:, None] + index * scale[d]).ravel()
        inside = (inside[:, None] & ok).ravel()
    return (addr + info.address).astype(np.uint64), inside


def swizzle_offsets(offsets: np.ndarray, bits: int) -> np.ndarray:
    """Where TMA puts each byte in shared memory.

    Shared memory is viewed as 128-byte rows of eight 16-byte chunks. The chunk
    index (address bits 4..) is XORed with the low `bits` bits of the row index
    (address bits 7..). This is CuTe's Swizzle<bits, 4, 3> applied to the address.
    """
    if not bits:
        return offsets
    mask = np.uint64(((1 << bits) - 1) << 4)
    return offsets ^ ((offsets >> np.uint64(3)) & mask)
