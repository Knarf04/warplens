"""Single-rounding IEEE binary arithmetic, without host-specific FMA support."""

from __future__ import annotations

import numpy as np


_FORMATS = {
    np.dtype(np.float16): (np.uint16, 10, 5),
    np.dtype(np.float32): (np.uint32, 23, 8),
    np.dtype(np.float64): (np.uint64, 52, 11),
}


def flush_subnormals(x: np.ndarray, small=None) -> np.ndarray:
    """Subnormal values (or those marked `small`) become zeros of the same sign."""
    if small is None:
        small = np.abs(x) < np.finfo(x.dtype).tiny
    return np.where(small, np.copysign(x.dtype.type(0), x), x)


def _fma_bits(a: int, b: int, c: int, fraction_bits: int, exponent_bits: int) -> int:
    """Compute one round-to-nearest-even FMA from IEEE bit patterns.

    Finite operands are signed integers times powers of two. Python integers
    preserve every bit of the product and sum, including cancellation and
    addends too small for a wider host float to retain at a rounding midpoint.
    """
    fraction_mask = (1 << fraction_bits) - 1
    exponent_mask = (1 << exponent_bits) - 1
    sign_mask = 1 << (fraction_bits + exponent_bits)
    bias = (1 << (exponent_bits - 1)) - 1
    infinity = exponent_mask << fraction_bits
    nan = infinity | (1 << (fraction_bits - 1))
    product_sign = (a ^ b) & sign_mask

    exponents = [(v >> fraction_bits) & exponent_mask for v in (a, b, c)]
    if any(e == exponent_mask and v & fraction_mask for v, e in zip((a, b, c), exponents)):
        return nan
    a_inf, b_inf, c_inf = (e == exponent_mask for e in exponents)
    if a_inf or b_inf:
        if (a & ~sign_mask) == 0 or (b & ~sign_mask) == 0:
            return nan  # infinity times zero
        if c_inf and (c & sign_mask) != product_sign:
            return nan  # opposite infinities
        return product_sign | infinity
    if c_inf:
        return c  # a finite exact product cannot cancel an infinite addend

    def unpack(v, exponent):
        significand = v & fraction_mask
        if exponent:
            significand |= 1 << fraction_bits
        if v & sign_mask:
            significand = -significand
        return significand, max(exponent, 1) - bias - fraction_bits

    x, xe = unpack(a, exponents[0])
    y, ye = unpack(b, exponents[1])
    z, ze = unpack(c, exponents[2])
    product, pe = x * y, xe + ye
    exponent = min(pe, ze)
    total = (product << (pe - exponent)) + (z << (ze - exponent))
    if total == 0:
        # Cancellation rounds to +0; adding two negative zeros preserves -0.
        return product_sign & c & sign_mask if product == 0 and z == 0 else 0

    sign = sign_mask if total < 0 else 0
    magnitude = abs(total)
    # Normal results retain fraction_bits + 1 significant bits. Subnormals
    # instead share a fixed quantum, the value of the smallest subnormal.
    quantum = max(magnitude.bit_length() - 1 + exponent - fraction_bits,
                  1 - bias - fraction_bits)
    shift = quantum - exponent
    if shift > 0:
        rounded, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and rounded & 1):
            rounded += 1
    else:
        rounded = magnitude << -shift
    if rounded == 1 << (fraction_bits + 1):
        rounded >>= 1
        quantum += 1

    result_exponent = quantum + fraction_bits + bias
    if result_exponent >= exponent_mask:
        return sign | infinity
    if rounded < 1 << fraction_bits:
        result_exponent = 0
    return sign | (result_exponent << fraction_bits) | (rounded & fraction_mask)


def fused_multiply_add(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return x*y+z with one rounding to their common f16/f32/f64 dtype."""
    unsigned, fraction_bits, exponent_bits = _FORMATS[x.dtype]
    result = [_fma_bits(int(a), int(b), int(c), fraction_bits, exponent_bits)
              for a, b, c in zip(x.view(unsigned), y.view(unsigned), z.view(unsigned))]
    return np.array(result, dtype=unsigned).view(x.dtype)


# -- TF32 -------------------------------------------------------------------------------------
#
# TF32 is float32 with only 10 fraction bits: sign, 8 exponent bits, 10 fraction
# bits, and the low 13 fraction bits zero. PTX leaves its internal representation
# implementation-defined; what is modeled here is the result of the conversion
# instruction as a 32-bit pattern.

TF32_DISCARDED_BITS = 13
_TF32_LOW = np.uint32((1 << TF32_DISCARDED_BITS) - 1)
_TF32_HALF = np.uint32(1 << (TF32_DISCARDED_BITS - 1))
TF32_MODES = ("rna", "rn", "rz")
# What cvt.rn.tf32.f32 and cvt.rz.tf32.f32 give for any NaN (measured on a B200).
TF32_NAN = np.uint32(0x7FFF_E000)


def to_tf32(bits: np.ndarray, mode: str = "rna") -> np.ndarray:
    """Convert float32 bit patterns to TF32, as `cvt.<mode>.tf32.f32` does.

    rna  nearest, ties away from zero. Checked bit for bit against an RTX 3080 Ti
         (validation/ampere/tf32_golden.txt), including these two behaviours:
         rounding works on the magnitude bits, so subnormals round the same way
         and the largest finite values round up to infinity; and for an exponent
         of all ones the low bits are simply cleared, which leaves infinity as
         it is and can turn a NaN whose payload was only in those bits into infinity.
    rn   nearest, ties to even.   } Need Hopper or newer. Checked bit for bit against a
    rz   toward zero.             } B200 (validation/blackwell/measured/b200/formats/
         tf32_modes.txt): finite values and infinities round on the magnitude bits like
         rna with their own tie rule; every NaN, whatever its sign and payload, gives
         0x7FFFE000. So unlike rna these two never turn a NaN into infinity.
    """
    if mode not in TF32_MODES:
        raise ValueError(f"unknown TF32 rounding mode {mode!r}")
    bits = np.asarray(bits, dtype=np.uint32)
    sign = bits & np.uint32(0x8000_0000)
    magnitude = bits & np.uint32(0x7FFF_FFFF)
    low = magnitude & _TF32_LOW
    kept = magnitude & ~_TF32_LOW
    if mode == "rz":
        up = np.zeros(bits.shape, dtype=bool)
    elif mode == "rna":
        up = low >= _TF32_HALF
    else:
        odd = (kept >> np.uint32(TF32_DISCARDED_BITS)) & np.uint32(1)
        up = (low > _TF32_HALF) | ((low == _TF32_HALF) & (odd == 1))
    special = magnitude >= np.uint32(0x7F80_0000)  # infinity or NaN: no rounding
    rounded = kept + np.where(up & ~special, np.uint32(1 << TF32_DISCARDED_BITS), np.uint32(0))
    result = (sign | rounded).astype(np.uint32)
    if mode != "rna":
        result = np.where(magnitude > np.uint32(0x7F80_0000), TF32_NAN, result).astype(np.uint32)
    return result


def explain_tf32(value, mode: str = "rna") -> str:
    """Show, bit by bit, what converting one float32 to TF32 keeps and discards."""
    x = np.array([value], dtype=np.float32)
    before = int(x.view(np.uint32)[0])
    after = int(to_tf32(x.view(np.uint32), mode)[0])
    y = np.array([after], dtype=np.uint32).view(np.float32)[0]

    def fields(b: int) -> str:
        return f"{b >> 31:01b} {(b >> 23) & 0xFF:08b} {(b >> 13) & 0x3FF:010b} {b & 0x1FFF:013b}"

    low = before & 0x1FFF
    lines = [f"float32 -> TF32 with cvt.{mode}.tf32.f32",
             "              sign exponent fraction kept  fraction discarded",
             f"  float32     {fields(before)}   {float(x[0]):.9g}",
             f"  TF32        {fields(after)}   {float(y):.9g}"]
    if np.isfinite(x[0]) and np.isfinite(y):
        error = float(y) - float(x[0])
        relative = f" ({abs(error) / abs(float(x[0])):.3g} relative)" if x[0] != 0 else ""
        action = "nothing to round" if not low else \
            ("rounded up" if (after & 0x7FFFFFFF) > (before & 0x7FFFE000) else "rounded down")
        lines.append(f"  discarded bits {low:013b} ({low} of 8192 of the last kept place): {action}; "
                     f"error {error:+.9g}{relative}")
    elif not np.isfinite(y) and np.isfinite(x[0]):
        lines.append("  rounding up carried past the largest finite TF32 value: the result is infinity")
    else:
        lines.append("  infinity or NaN: the low 13 bits are cleared without rounding")
    return "\n".join(lines)
