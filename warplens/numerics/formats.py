"""Floating-point formats of 8 bits or fewer, as used by Blackwell's low-precision MMAs.

A format is a sign bit (optional), an exponent field and a fraction field. With
so few bits, every code can be listed: the value table of a format is built
once from its definition and used for both decoding and encoding.

    E4M3   8 bits, signed    one NaN code, no infinity      largest 448
    E5M2   8 bits, signed    IEEE-style infinity and NaNs   largest 57344
    E2M1   4 bits, signed    no NaN, no infinity            0, 0.5, 1, 1.5, 2, 3, 4, 6
    UE8M0  8 bits, unsigned  a power of two, 2^(code-127)   one NaN code; used as a block scale
    UE4M3  8 bits, unsigned  E4M3 without the sign          one NaN code; used as a block scale

Bit widths and NaN conventions follow cutlass/exmy_base.h. The exponent bias is
the usual 2^(exponent bits - 1) - 1, and exponent field 0 holds subnormals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["E2M1", "E4M3", "E5M2", "FloatFormat", "UE4M3", "UE8M0"]


@dataclass(frozen=True)
class FloatFormat:
    name: str
    bits: int  # bits of storage that carry the value
    exponent_bits: int
    fraction_bits: int
    signed: bool = True
    nan: str = "none"  # "ieee" (infinity and NaNs), "canonical" (one NaN, no infinity) or "none"
    table: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "table", self._build())

    @property
    def bias(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def codes(self) -> int:
        """Number of distinct codes."""
        return 1 << (self.exponent_bits + self.fraction_bits + (1 if self.signed else 0))

    def _build(self) -> np.ndarray:
        table = np.zeros(self.codes, dtype=np.float64)
        top = (1 << self.exponent_bits) - 1
        for code in range(self.codes):
            fraction = code & ((1 << self.fraction_bits) - 1)
            exponent = (code >> self.fraction_bits) & top
            negative = self.signed and code >> (self.exponent_bits + self.fraction_bits)
            if exponent == top and self.nan == "ieee":
                value = np.inf if fraction == 0 else np.nan
            elif exponent == top and fraction == (1 << self.fraction_bits) - 1 and self.nan == "canonical":
                value = np.nan
            elif exponent == 0 and self.fraction_bits:  # subnormal (or zero)
                value = fraction * 2.0 ** (1 - self.bias - self.fraction_bits)
            else:
                value = (1 + fraction * 2.0 ** -self.fraction_bits) * 2.0 ** (exponent - self.bias)
            table[code] = -value if negative else value
        return table

    @property
    def largest(self) -> float:
        return float(np.nanmax(self.table[np.isfinite(self.table)]))

    @property
    def nan_code(self) -> int | None:
        """The code a NaN encodes to: all ones below the sign bit, as cuda_fp8.h and the `cvt`
        instructions give (0x7f for E4M3 and E5M2, 0xff for the unsigned formats)."""
        positive = self.table[:len(self.table) // 2] if self.signed else self.table
        nans = np.flatnonzero(np.isnan(positive))
        return int(nans[-1]) if len(nans) else None

    def decode(self, codes) -> np.ndarray:
        codes = np.asarray(codes)
        if codes.size and codes.dtype.kind not in "biu":
            raise ValueError(f"{self.name} codes must be integers")
        if codes.size and (codes.min() < 0 or codes.max() >= self.codes):
            raise ValueError(f"{self.name} codes must be in 0..{self.codes - 1}")
        return self.table[codes.astype(np.intp)]

    def encode(self, values, saturate: bool = True) -> np.ndarray:
        """Round to nearest, ties to the even significand (UE8M0 ties upward).

        UE8M0 has no fraction bits: midpoint ties go to the next power of two,
        as documented for CUDA's nearest E8M0 conversion, not by exponent parity.

        With `saturate`, values beyond the largest magnitude (and infinities)
        become the largest finite value, like PTX's `.satfinite`. Without it,
        they become infinity, which only E5M2 has. NaN becomes the format's NaN
        code; E2M1 has none, so NaN there is an error. Unsigned formats reject
        negative values.
        """
        x = np.asarray(values, dtype=np.float64)
        magnitude = np.abs(x)
        negative = np.signbit(x)
        if not self.signed and (negative & (magnitude > 0)).any():
            raise ValueError(f"{self.name} is unsigned; negative values cannot be encoded")
        half = self.codes // 2 if self.signed else self.codes
        finite = np.flatnonzero(np.isfinite(self.table[:half]))  # non-negative finite codes, ascending value
        levels = self.table[finite]
        upper = np.clip(np.searchsorted(levels, magnitude, side="left"), 1, len(levels) - 1)
        lower = upper - 1
        below, above = magnitude - levels[lower], levels[upper] - magnitude
        tie_up = (self.fraction_bits == 0) | (finite[upper] % 2 == 0)
        pick_upper = (above < below) | ((above == below) & tie_up)
        code = np.where(pick_upper, finite[upper], finite[lower])
        over = magnitude > levels[-1]
        if not saturate:
            # Round to nearest: only values at least half a step past the top overflow.
            step = levels[-1] - levels[-2]
            over = magnitude >= levels[-1] + step / 2
            if over.any() and self.nan != "ieee":
                raise ValueError(f"{self.name} has no infinity; encode with saturate=True")
            infinity = int(np.flatnonzero(np.isinf(self.table[:half]))[0]) if self.nan == "ieee" else 0
            code = np.where(over, infinity, code)
        else:
            code = np.where(over | np.isinf(magnitude), finite[-1], code)
        if self.signed:
            code = np.where(negative, code + half, code)
        if np.isnan(x).any():
            if self.nan_code is None:
                raise ValueError(f"{self.name} cannot represent NaN")
            code = np.where(np.isnan(x), self.nan_code, code)
        return code.astype(np.uint8)

    def bit_string(self, code: int) -> str:
        """The code as 'sign exponent fraction' bit groups."""
        width = self.exponent_bits + self.fraction_bits + (1 if self.signed else 0)
        text = f"{int(code):0{width}b}"
        parts = ([text[0]] if self.signed else []) + [text[-(self.exponent_bits + self.fraction_bits):
                                                          len(text) - self.fraction_bits]]
        if self.fraction_bits:
            parts.append(text[-self.fraction_bits:])
        return " ".join(parts)


E4M3 = FloatFormat("E4M3", 8, 4, 3, nan="canonical")
E5M2 = FloatFormat("E5M2", 8, 5, 2, nan="ieee")
E2M1 = FloatFormat("E2M1", 4, 2, 1)
UE8M0 = FloatFormat("UE8M0", 8, 8, 0, signed=False, nan="canonical")
UE4M3 = FloatFormat("UE4M3", 8, 4, 3, signed=False, nan="canonical")
