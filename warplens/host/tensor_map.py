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
MAX_DIM = 2**31  # cuTensorMapEncodeTiled, CUDA 13.5
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
    if len(raw) != SIZE:
        raise ExecutionError(f"a tensor map must contain {SIZE} bytes, got {len(raw)}")
    fields = struct.unpack_from(_FORMAT, raw)
    magic, rank, elem_size, is_float, swizzle, nan_fill, address = fields[:7]
    if magic != MAGIC or not 1 <= rank <= MAX_RANK:
        raise ExecutionError("the tensor-map operand does not point at a tensor map created by "
                             "Device.tensor_map_tiled")
    dims, strides, box, steps = fields[7:12], fields[12:16], fields[16:21], fields[21:26]
    info = TensorMapInfo(rank, elem_size, bool(is_float), swizzle, bool(nan_fill), address,
                         dims[:rank], strides[:rank - 1], box[:rank], steps[:rank])
    if (elem_size not in (1, 2, 4, 8) or is_float not in (0, 1) or nan_fill not in (0, 1)
            or (is_float and elem_size == 1) or swizzle not in SWIZZLE_MODES):
        raise ExecutionError("invalid tensor-map data type, fill mode or swizzle")
    # Descriptor memory can be overwritten by a kernel. Validate the decoded
    # fields before they become array sizes, divisors, or dtype lookup keys.
    try:
        expected = make_tiled(address=address, shape=info.dims[::-1],
                              dtype=np.dtype(f"{'f' if is_float else 'u'}{elem_size}"),
                              box=info.box[::-1], swizzle=SWIZZLE_MODES[swizzle][0],
                              element_strides=info.steps[::-1],
                              oob_fill="nan" if nan_fill else "zero")
    except LaunchError as e:
        raise ExecutionError(f"invalid tensor map: {e}") from e
    if info.strides != expected.strides or info.steps != expected.steps:
        raise ExecutionError("invalid tensor-map strides")
    return info


def _integer_tuple(values, what: str) -> tuple[int, ...]:
    try:
        values = tuple(values)
    except TypeError:
        raise LaunchError(f"{what} must be a sequence of integers") from None
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           for v in values):
        raise LaunchError(f"{what} must contain integers, got {values!r}")
    return tuple(int(v) for v in values)


def make_tiled(*, address: int, shape, dtype: np.dtype, box, swizzle=None, element_strides=None,
               oob_fill: str = "zero") -> TensorMapInfo:
    """Validate like cuTensorMapEncodeTiled. `shape`, `box` and `element_strides`
    are in NumPy order (outermost first) and are reversed into TMA order here."""
    shape = _integer_tuple(shape, "tensor dimensions")
    rank = len(shape)
    if not 1 <= rank <= MAX_RANK:
        raise LaunchError(f"a tensor map needs 1 to {MAX_RANK} dimensions, got {rank}")
    box = _integer_tuple((box,) if isinstance(box, (int, np.integer)) else box,
                         "box dimensions")
    steps = (1,) * rank if element_strides is None else \
        _integer_tuple(element_strides, "element strides")
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
    if any(not 1 <= d <= MAX_DIM for d in dims):
        raise LaunchError(f"tensor dimensions must be between 1 and 2^31, got {dims[::-1]}")
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


# CUtensorMapDataType -> (element size, is floating point)
DRIVER_TYPES = {0: (1, False), 1: (2, False), 2: (4, False), 3: (4, False), 4: (8, False),
                5: (8, False), 6: (2, True), 7: (4, True), 8: (8, True), 9: (2, True),
                10: (4, True), 11: (4, True), 12: (4, True)}


def from_driver(*, data_type: int, rank: int, address: int, dims, strides, box, steps,
                swizzle: int, nan_fill: bool, interleave: int = 0) -> TensorMapInfo:
    """Build a tensor map from the arguments of cuTensorMapEncodeTiled, which are already
    in TMA order (dimension 0 innermost) with explicit byte strides."""
    if data_type not in DRIVER_TYPES:
        raise LaunchError(f"unknown tensor-map data type {data_type}")
    if interleave:
        raise LaunchError("interleaved tensor maps are not supported")
    if not 1 <= rank <= MAX_RANK or swizzle not in SWIZZLE_MODES:
        raise LaunchError(f"invalid tensor map: rank {rank}, swizzle {swizzle}")
    elem, is_float = DRIVER_TYPES[data_type]
    dims, box, steps = tuple(dims[:rank]), tuple(box[:rank]), tuple(steps[:rank])
    strides = tuple(strides[:rank - 1])
    if address % 16 or any(s % 16 or s >= 2**40 for s in strides):
        raise LaunchError("tensor-map address and strides must be multiples of 16")
    if any(not 1 <= d <= 2**32 for d in dims) or any(not 1 <= b <= 256 for b in box) \
            or any(not 1 <= s <= 8 for s in steps):
        raise LaunchError(f"tensor-map dimensions out of range: dims {dims}, box {box}, strides {steps}")
    inner = box[0] * elem
    span = SWIZZLE_MODES[swizzle][1]
    if inner % 16 or (span is not None and inner > span):
        raise LaunchError(f"the innermost box dimension is {inner} bytes; it must be a multiple of "
                          "16 and fit the swizzle span")
    if nan_fill and not is_float:
        raise LaunchError("NaN out-of-bounds fill needs a floating-point tensor")
    return TensorMapInfo(rank, elem, is_float, swizzle, bool(nan_fill), address, dims, strides, box,
                         (1, *steps[1:]))


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
