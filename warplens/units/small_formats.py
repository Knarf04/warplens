"""The `cvt` forms for the FP8, FP4 and E8M0 formats, as `cuda_fp8.h` and `cuda_fp4.h` emit them.

These instructions convert two values at once. The small-format side is a pair of codes packed
into one register (`e4m3x2`, `e5m2x2` and `ue8m0x2`: two bytes in 16 bits; `e2m1x2`: two
nibbles in 8 bits). The other side is either a packed pair of 16-bit floats in a 32-bit
register (`f16x2`, `bf16x2`) or, when encoding, two separate `.f32` operands:

    cvt.rn.satfinite.{e4m3x2,e5m2x2,e2m1x2}.f32      d, a, b     d = code(a) : code(b)
    cvt.rn.satfinite.{e4m3x2,e5m2x2,e2m1x2}.{f16x2,bf16x2}  d, a
    cvt.{rz,rp}[.satfinite].ue8m0x2.f32              d, a, b
    cvt.{rz,rp}[.satfinite].ue8m0x2.bf16x2           d, a
    cvt.rn.{f16x2,bf16x2}.{e4m3x2,e5m2x2,e2m1x2}     d, a
    cvt.rn.bf16x2.ue8m0x2                            d, a

In the `.f32` forms the last operand supplies the low code (the headers pass the value as `b`
and 0.0 as `a`); in the packed forms the low half converts to the low half.

This is the set an `sm_100a` / `sm_103a` build emits. The values come from the tables in
`warplens.numerics`. What those tables do not say is taken from the host (non-PTX)
implementation of the same functions in the CUDA headers, and has not been compared with a GPU:

  * encoding a NaN gives 0x7f for E4M3 and E5M2 whatever its sign, and +6 (0x7) for E2M1,
    which has no NaN. (`warplens.numerics` gives the same NaN code and
    refuses a NaN for E2M1; it also refuses negative values for UE8M0.)
  * decoding a NaN code gives 0x7fff, in half precision and in bfloat16;
  * UE8M0 ignores the sign of its input. `.rz` gives the exponent field of the input, so zero
    and everything below 2^-126 become code 0 (2^-127); `.rp` gives the next code up unless
    the input is a power of two, and leaves values up to 2^-127 at code 0. A NaN gives 0xff.
    Infinity, and with `.rp` anything above 2^127, gives 0xff, or 0xfe under `.satfinite`;
  * UE8M0 code 0xff decodes to 0x7fff and code 0 to 0x0040 (2^-127, a bfloat16 subnormal).
"""

from __future__ import annotations

import numpy as np

from ..numerics.formats import E2M1, E4M3, E5M2

# name -> (format, bits per code). UE8M0 is handled on bit patterns, not through a table.
_CODES = {"e4m3x2": (E4M3, 8), "e5m2x2": (E5M2, 8), "e2m1x2": (E2M1, 4), "ue8m0x2": (None, 8)}
_PACKED_FLOATS = ("f16x2", "bf16x2")
_NAN16 = 0x7FFF


def is_small_format_cvt(mods: tuple[str, ...]) -> bool:
    return len(mods) >= 2 and (mods[-1] in _CODES or mods[-2] in _CODES)


def _float_parts(bits: np.ndarray, source: str):
    """(value as float64, magnitude bits, fraction width) of f32, f16 or bf16 bit patterns."""
    with np.errstate(invalid="ignore"):  # widening a signalling NaN
        if source == "f32":
            bits = bits.astype(np.uint32)
            return bits.view(np.float32).astype(np.float64), bits & np.uint32(0x7FFF_FFFF), 23
        bits = bits.astype(np.uint16)
        if source == "f16":
            return bits.view(np.float16).astype(np.float64), bits & np.uint16(0x7FFF), 10
        value = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
        return value, bits & np.uint16(0x7FFF), 7  # bfloat16: the top half of a float32


def encode(target: str, rounding: str, saturate: bool, bits: np.ndarray, source: str) -> np.ndarray:
    """One code per element of `bits` (f32, f16 or bf16 patterns); None if the form is not one
    the headers emit."""
    fmt, _ = _CODES[target]
    value, magnitude, fraction_bits = _float_parts(bits, source)
    if fmt is None:  # UE8M0
        if rounding not in ("rz", "rp"):
            return None
        magnitude = magnitude.astype(np.uint32)
        infinity = np.uint32(0xFF << fraction_bits)
        code = magnitude >> np.uint32(fraction_bits)
        if rounding == "rp":
            fraction = magnitude & np.uint32((1 << fraction_bits) - 1)
            code = code + ((fraction != 0) & (magnitude > np.uint32(1 << (fraction_bits - 1)))
                           & (magnitude < infinity))
        if saturate:
            code = np.where((magnitude <= infinity) & (code == 0xFF), 0xFE, code)
        return code.astype(np.uint8)
    if rounding != "rn" or not saturate:
        return None
    nan = np.isnan(value)
    code = fmt.encode(np.where(nan, 0.0, value), saturate=True)
    return np.where(nan, 0x7 if fmt is E2M1 else 0x7F, code).astype(np.uint8)


def decode(source: str, codes: np.ndarray, target: str) -> np.ndarray:
    """The f16 or bf16 bit pattern of each code; None if the form is not one the headers emit."""
    fmt, _ = _CODES[source]
    codes = codes.astype(np.uint8)
    if fmt is None:  # UE8M0: a power of two, so the code is the exponent field
        if target != "bf16x2":
            return None
        out = codes.astype(np.uint16) << np.uint16(7)
        return np.where(codes == 0xFF, _NAN16, np.where(codes == 0, 0x0040, out)).astype(np.uint16)
    value = fmt.decode(codes)  # every value is exact in both 16-bit formats
    with np.errstate(invalid="ignore"):
        if target == "f16x2":
            out = value.astype(np.float16).view(np.uint16)
        else:
            out = (value.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    return np.where(np.isnan(value), _NAN16, out).astype(np.uint16)


def convert(w, ins, mask, unsupported) -> None:
    """Execute one small-format `cvt`. `unsupported` builds the error for a rejected form."""
    *extra, dty, aty = ins.mods
    expected = "see warplens/units/small_formats.py for the forms that are implemented"
    if dty in _CODES:
        _, width = _CODES[dty]
        if aty not in ("f32", *_PACKED_FLOATS) or not extra or len(extra) > 2 \
                or (len(extra) == 2 and extra[1] != "satfinite"):
            raise unsupported(ins, expected)
        if aty == "f32":
            if len(ins.operands) != 3:
                raise unsupported(ins, "expected two .f32 sources")
            d, a, b = ins.operands
            low, high = (w.read(x, "f32").view(np.uint32) for x in (b, a))
            source = "f32"
        else:
            if len(ins.operands) != 2:
                raise unsupported(ins, "expected one packed source")
            d, a = ins.operands
            packed = w.read(a, "b32")
            low, high = packed & np.uint32(0xFFFF), packed >> np.uint32(16)
            source = aty[:-2]
        halves = [encode(dty, extra[0], len(extra) == 2, x, source) for x in (low, high)]
        if halves[0] is None:
            raise unsupported(ins, expected)
        if width == 4:
            w.write(d, (halves[0] | (halves[1] << np.uint8(4))).astype(np.uint8), mask)
        else:
            w.write(d, halves[0].astype(np.uint16) | (halves[1].astype(np.uint16) << np.uint16(8)), mask)
        return
    _, width = _CODES[aty]
    if dty not in _PACKED_FLOATS or extra != ["rn"] or len(ins.operands) != 2:
        raise unsupported(ins, expected)
    d, a = ins.operands
    if width == 4:
        packed = w.read(a, "b8")
        low, high = packed & np.uint8(0xF), packed >> np.uint8(4)
    else:
        packed = w.read(a, "b16")
        low, high = packed & np.uint16(0xFF), packed >> np.uint16(8)
    halves = [decode(aty, x, dty) for x in (low, high)]
    if halves[0] is None:
        raise unsupported(ins, expected)
    w.write(d, halves[0].astype(np.uint32) | (halves[1].astype(np.uint32) << np.uint32(16)), mask)
