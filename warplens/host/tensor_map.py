"""Tensor maps: the descriptors a TMA instruction reads to find its data.

On real hardware `cuTensorMapEncodeTiled` fills an opaque 128-byte CUtensorMap.
warplens emulates the encode call, so it defines its own 128-byte encoding
here and the TMA unit decodes it. A tensor map in the driver's own layout is
read as well (`decode_driver`): a frontend that writes the descriptor itself
instead of calling the encoder (the CuTe DSL does) hands the kernel one of
those. The reverse does not hold: a map encoded here means nothing to a GPU.

Dimension order follows TMA: dimension 0 is the innermost (fastest varying).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..errors import ExecutionError, LaunchError

SIZE = 128
MAX_RANK = 5
MAX_DIM = 2**32  # cuTensorMapEncodeTiled, CUDA 13.5
MAGIC = b"WLT2"
_FORMAT = "<4s5B7xQ5Q4Q5H5B"  # magic, rank/elem/float/swizzle/nan, address, dims, strides, box, steps
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
    if magic != MAGIC:
        return decode_driver(raw)
    if not 1 <= rank <= MAX_RANK:
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
        # CUDA frontends can describe padded tensors. Validate explicit strides
        # rather than reconstructing the contiguous strides of the Python API.
        data_type = ({2: 6, 4: 7, 8: 8} if is_float else {1: 0, 2: 1, 4: 2, 8: 4})[elem_size]
        expected = from_driver(data_type=data_type, rank=rank, address=address, dims=info.dims,
                               strides=info.strides, box=info.box, steps=info.steps,
                               swizzle=swizzle, nan_fill=nan_fill)
    except LaunchError as e:
        raise ExecutionError(f"invalid tensor map: {e}") from e
    if info.steps != expected.steps:
        raise ExecutionError("invalid tensor-map element strides")
    return info


# -- the driver's own layout ---------------------------------------------------------------
#
# The 128 bytes cuTensorMapEncodeTiled writes, as sixteen little-endian 64-bit words. Worked
# out from validation/blackwell/measured/b200/tma/tensormap_encode_sweep.txt: 1087 maps a B200's
# driver (610.57.04) encoded over a sweep of every parameter, each of which this reading
# gives back (tests/test_tensor_map_driver_layout.py). It is not documented.
#
#   word 0            the tensor's address
#   word 1, low half  bits 4-6 rank - 1; 7-10 the data type's code (_DRIVER_TYPE_CODES);
#                     11-12 interleave; 13-14 swizzle; 15 NaN fill; 16 TFLOAT32; 17-18 L2
#                     promotion; 19-20 the atom of the 128-byte swizzles with a smaller atom
#                     (CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B and its two neighbours); 21 set by
#                     the driver for some maps, by a rule the sweep does not show (not read)
#   word 1 high, word 2 low, word 2 high, word 3 low
#                     the byte strides of dimensions 1 to 4, divided by 16: their low 32 bits
#   word 3, bits 32-47  the next 4 bits of each of the four strides
#   words 4, 5, word 6 low   the five dimensions, each less one, 32 bits apiece
#   word 6, bits 32-46  the five element strides, each less one, 3 bits apiece
#   word 6, bits 56-63  the box's dimension 0, less one
#   word 7, bytes 0-3   the box's dimensions 1 to 4, each less one
#   word 8            the bytes one box occupies in shared memory; word 9 the swizzle's
#                     period (16 without one). Both follow from the rest and are not read.
#   words 10 to 15    zero
#
# Only the first eight words are the descriptor a kernel's TMA instructions need: the CuTe
# DSL writes those 64 bytes itself and nothing else, and its kernels ran on a B200 (matrix
# row RT-11; tests/test_tensor_map_driver_layout.py has two of its maps).
#
# The codes of the data types are the enumeration's numbers but for three: FLOAT64 (8) is 9,
# BFLOAT16 (9) is 10, FLOAT32_FTZ (10) is 8. TFLOAT32 and TFLOAT32_FTZ are the codes of
# FLOAT32 and FLOAT32_FTZ with bit 16.
_DRIVER_TYPE_CODES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 10, 9: 8, 10: 9}  # code -> CUtensorMapDataType
_NOT_A_MAP = ("the tensor-map operand points at neither a tensor map created by Device.tensor_map_tiled "
              "nor one in the driver's layout")


def driver_fields(raw: bytes) -> dict:
    """The parameters of cuTensorMapEncodeTiled as a driver wrote them into `raw`, in TMA
    order, without judging them: `data_type` (the enumeration's number), `rank`, `address`,
    `dims`, `strides`, `box`, `steps`, `interleave`, `swizzle`, `swizzle_atom`, `l2_promotion`,
    `nan_fill`. Raises ExecutionError if the bytes cannot be such a map."""
    if len(raw) != SIZE:
        raise ExecutionError(f"a tensor map must contain {SIZE} bytes, got {len(raw)}")
    w = struct.unpack("<16Q", raw)
    flags = w[1] & 0xFFFF_FFFF
    code = flags >> 7 & 0xF
    # What a map never has set: the high bits of the flag word, which no parameter reaches,
    # and the bits above the fields of words 3, 6 and 7. Words 8 to 15 are not looked at:
    # the driver writes two derived numbers and zeros there, and the CuTe DSL, which fills
    # in the descriptor itself, writes the first 64 bytes only and leaves the rest as it
    # found them (the maps captured on a B200 hold host pointers there). The DSL also sets
    # bit 1 of the flag word, which the driver never did in the sweep; bits 0 to 3 are not
    # looked at either.
    if (flags & 0xFFC0_0000 or code not in _DRIVER_TYPE_CODES or w[3] >> 48 or w[6] >> 47 & 0x1FF
            or w[7] >> 32):
        raise ExecutionError(_NOT_A_MAP)
    rank = (flags >> 4 & 7) + 1
    if rank > MAX_RANK:
        raise ExecutionError(_NOT_A_MAP)
    data_type = _DRIVER_TYPE_CODES[code]
    if flags >> 16 & 1:
        if data_type not in (7, 10):
            raise ExecutionError(_NOT_A_MAP)
        data_type = {7: 11, 10: 12}[data_type]
    low = (w[1] >> 32, w[2] & 0xFFFF_FFFF, w[2] >> 32, w[3] & 0xFFFF_FFFF)
    strides = tuple(((w[3] >> (32 + 4 * i) & 0xF) << 32 | low[i]) * 16 for i in range(4))
    dims = tuple(v + 1 for v in (w[4] & 0xFFFF_FFFF, w[4] >> 32, w[5] & 0xFFFF_FFFF, w[5] >> 32,
                                 w[6] & 0xFFFF_FFFF))
    steps = tuple((w[6] >> (32 + 3 * i) & 7) + 1 for i in range(5))
    box = ((w[6] >> 56) + 1, *((w[7] >> (8 * i) & 0xFF) + 1 for i in range(4)))
    return {"data_type": data_type, "rank": rank, "address": w[0], "dims": dims[:rank],
            "strides": strides[:rank - 1], "box": box[:rank], "steps": steps[:rank],
            "interleave": flags >> 11 & 3, "swizzle": flags >> 13 & 3, "swizzle_atom": flags >> 19 & 3,
            "l2_promotion": flags >> 17 & 3, "nan_fill": bool(flags >> 15 & 1)}


def decode_driver(raw: bytes) -> TensorMapInfo:
    """A tensor map in the driver's layout, as the map the simulator runs. What the simulator
    does not model is refused by name: interleaved layouts and the 128-byte swizzles with a
    smaller atom. The L2 promotion is a hint and is ignored; a TFLOAT32 tensor is read as
    the float32 it is stored as, as for a map the shim's own encoder made."""
    f = driver_fields(raw)
    if f["swizzle_atom"]:
        raise ExecutionError("a tensor map with a 128-byte swizzle of a smaller atom "
                             "(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_...) is not supported")
    try:
        return from_driver(data_type=f["data_type"], rank=f["rank"], address=f["address"], dims=f["dims"],
                           strides=f["strides"], box=f["box"], steps=f["steps"], swizzle=f["swizzle"],
                           nan_fill=f["nan_fill"], interleave=f["interleave"])
    except LaunchError as e:
        raise ExecutionError(f"invalid tensor map (in the driver's layout): {e}") from e


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
    dims, box, steps = (_integer_tuple(values, name)[:rank] for values, name in
                        ((dims, "tensor dimensions"), (box, "box dimensions"),
                         (steps, "element strides")))
    strides = _integer_tuple(strides, "byte strides")[:rank - 1]
    if len(dims) != rank or len(box) != rank or len(steps) != rank or len(strides) != rank - 1:
        raise LaunchError("tensor-map geometry must provide one entry per dimension")
    if not isinstance(address, (int, np.integer)) or not 0 <= address < 2**64 or address % 16:
        raise LaunchError("tensor-map address must be a 16-byte aligned unsigned 64-bit integer")
    if any(s <= 0 or s % 16 or s >= 2**40 for s in strides):
        raise LaunchError("tensor-map strides must be positive multiples of 16 below 2^40")
    if any(not 1 <= d <= MAX_DIM for d in dims) or any(not 1 <= b <= 256 for b in box) \
            or any(not 1 <= s <= 8 for s in steps):
        raise LaunchError(f"tensor-map dimensions out of range: dims {dims}, box {box}, strides {steps}")
    preceding = elem
    for dimension, stride in zip(dims, strides):
        if stride < preceding * dimension:
            raise LaunchError("tensor-map byte strides must include the preceding dimension")
        preceding = stride
    if nan_fill not in (0, 1):
        raise LaunchError("invalid tensor-map out-of-bounds fill mode")
    if data_type == 9 and nan_fill:
        raise LaunchError("BF16 tensor maps with NaN out-of-bounds fill are not supported")
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
