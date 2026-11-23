"""Single-rounding IEEE binary arithmetic, without host-specific FMA support."""

from __future__ import annotations

import numpy as np


_FORMATS = {
    np.dtype(np.float16): (np.uint16, 10, 5),
    np.dtype(np.float32): (np.uint32, 23, 8),
    np.dtype(np.float64): (np.uint64, 52, 11),
}


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
