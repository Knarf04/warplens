"""Block-scaled formats: small element codes that share one scale per block.

    MXFP8   32 x E4M3 or E5M2 elements per UE8M0 scale (a power of two)
    MXFP4   32 x E2M1 elements        per UE8M0 scale (a power of two)
    NVFP4   16 x E2M1 elements        per UE4M3 scale (finer), plus one float32 scale per tensor

A value is `element * block scale` (times the tensor scale for NVFP4). This
module quantizes and dequantizes on the host, so the effect of scale selection
and rounding can be inspected. It does not execute the block-scaled
`tcgen05.mma` instructions.

How a scale is chosen is a recipe, not part of the format:

    MX       scale = 2^(floor(log2(max |x|)) - e), where 2^e is the top binade of
             the element format, so the largest value lands in that binade
             (the OCP Microscaling recipe).
    NVFP4    scale = max |x| / 6, since 6 is the largest E2M1 value, rounded to
             UE4M3 after dividing by a per-tensor scale that brings every block
             scale into UE4M3's range.

Other recipes exist; pass your own scales to `quantize` to try them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formats import E2M1, E4M3, E5M2, UE4M3, UE8M0, FloatFormat


@dataclass(frozen=True)
class BlockScaled:
    name: str
    element: FloatFormat
    scale: FloatFormat
    block: int  # elements that share one scale
    tensor_scale: bool = False  # also uses one float32 scale for the whole tensor

    @property
    def bits_per_value(self) -> float:
        return self.element.bits + 8 / self.block


MXFP8_E4M3 = BlockScaled("MXFP8 (E4M3)", E4M3, UE8M0, 32)
MXFP8_E5M2 = BlockScaled("MXFP8 (E5M2)", E5M2, UE8M0, 32)
MXFP4 = BlockScaled("MXFP4", E2M1, UE8M0, 32)
NVFP4 = BlockScaled("NVFP4", E2M1, UE4M3, 16, tensor_scale=True)


def _blocks(x, fmt: BlockScaled) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) % fmt.block:
        raise ValueError(f"{fmt.name} needs a 1-D array whose length is a multiple of {fmt.block}")
    if not np.isfinite(x).all():
        raise ValueError("cannot quantize NaN or infinity")
    return x.reshape(-1, fmt.block)


def _tensor_scale(value: float) -> float:
    with np.errstate(over="ignore", under="ignore"):
        value = float(np.float32(value))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tensor scale must be a positive finite float32 value")
    return value


def choose_scales(x, fmt: BlockScaled, tensor_scale: float | None = None):
    """(scale codes, tensor scale) by the recipe described in this module."""
    blocks = _blocks(x, fmt)
    peak = np.abs(blocks).max(axis=1)
    if fmt.scale is UE8M0:
        # frexp extracts the binade exactly, including values just below powers
        # of two where a rounded log2 can select an exponent one too large.
        top = np.frexp(fmt.element.largest)[1] - 1
        exponent = np.where(peak > 0, np.frexp(peak)[1] - 1 - top, 0)
        return np.clip(exponent + UE8M0.bias, 0, 254).astype(np.uint8), 1.0
    if tensor_scale is None:
        largest = float(peak.max(initial=0.0))
        # Map the largest block scale onto the largest UE4M3 value. Clamp the
        # global scale to finite nonzero float32 so tiny tensors do not become
        # all zero merely because their ideal global scale underflows.
        tensor_scale = largest / fmt.element.largest / fmt.scale.largest if largest > 0 else 1.0
        limits = np.finfo(np.float32)
        tensor_scale = np.clip(tensor_scale, float(limits.smallest_subnormal), float(limits.max))
    tensor_scale = _tensor_scale(tensor_scale)
    return fmt.scale.encode(peak / fmt.element.largest / tensor_scale), tensor_scale


def quantize(x, fmt: BlockScaled, scales=None, tensor_scale: float | None = None):
    """Returns (element codes, scale codes, tensor scale). `scales` overrides the recipe."""
    blocks = _blocks(x, fmt)
    if tensor_scale is not None:
        tensor_scale = _tensor_scale(tensor_scale)
    if scales is None:
        scales, tensor_scale = choose_scales(x, fmt, tensor_scale)
    else:
        scales = np.asarray(scales)
        tensor_scale = 1.0 if tensor_scale is None else tensor_scale
        if scales.shape != (len(blocks),):
            raise ValueError(f"expected {len(blocks)} scale code(s), one per block")
    factor = fmt.scale.decode(scales) * tensor_scale
    scales = scales.astype(np.uint8)  # validate integer codes before narrowing
    if np.isnan(factor).any():
        raise ValueError("a block scale is NaN")
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(factor[:, None] > 0, blocks / factor[:, None], 0.0)
    return fmt.element.encode(scaled).reshape(-1), scales, tensor_scale


def dequantize(codes, scales, fmt: BlockScaled, tensor_scale: float = 1.0) -> np.ndarray:
    codes, scales = np.asarray(codes), np.asarray(scales)
    if codes.ndim != 1 or len(codes) % fmt.block or scales.shape != (len(codes) // fmt.block,):
        raise ValueError(f"{fmt.name} needs {fmt.block} element codes per scale code")
    factor = fmt.scale.decode(scales) * _tensor_scale(tensor_scale)
    return (fmt.element.decode(codes).reshape(-1, fmt.block) * factor[:, None]).reshape(-1)


def explain_block(x, fmt: BlockScaled, block: int = 0, tensor_scale: float | None = None) -> str:
    """Show one block: the scale that was chosen and what each value turned into."""
    x = np.asarray(x, dtype=np.float64)
    codes, scales, tensor_scale = quantize(x, fmt, tensor_scale=tensor_scale)
    back = dequantize(codes, scales, fmt, tensor_scale)
    lo = block * fmt.block
    values, kept = x[lo:lo + fmt.block], back[lo:lo + fmt.block]
    factor = float(fmt.scale.decode(scales[block:block + 1])[0])
    peak = float(np.abs(values).max())
    lines = [f"{fmt.name}, block {block}: {fmt.block} x {fmt.element.name} elements, one {fmt.scale.name} scale",
             f"  largest magnitude in the block: {peak:.6g}",
             f"  block scale: code {int(scales[block])} = {fmt.scale.bit_string(scales[block])} = {factor:.6g}"
             + (f", times tensor scale {tensor_scale:.6g}" if fmt.tensor_scale else "")]
    used = peak / (factor * tensor_scale) / fmt.element.largest if factor else 0.0
    lines.append(f"  the largest value uses {used:.0%} of the element range (largest element {fmt.element.largest:g})")
    lines.append("       value      /scale   code  bits         stored       error")
    for i in range(fmt.block):
        code = int(codes[lo + i])
        scaled = values[i] / (factor * tensor_scale) if factor else 0.0
        lines.append(f"  {values[i]:>10.5f} {scaled:>10.4f}   {code:>4}  "
                     f"{fmt.element.bit_string(code):<10} {kept[i]:>10.5f}  {kept[i] - values[i]:>+10.5f}")
    rms = float(np.sqrt(np.mean((kept - values) ** 2)))
    lines.append(f"  RMS error of the block: {rms:.6g} ({rms / max(float(np.sqrt(np.mean(values ** 2))), 1e-300):.2%} of its RMS value)")
    return "\n".join(lines)


def compare_formats(x, formats=(MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4)) -> str:
    """Quantize the same data with each format and report the error."""
    x = np.asarray(x, dtype=np.float64)
    rms_value = max(float(np.sqrt(np.mean(x ** 2))), 1e-300)
    lines = ["format         bits/value   scale steps      RMS error   relative"]
    for fmt in formats:
        codes, scales, tensor_scale = quantize(x, fmt)
        error = dequantize(codes, scales, fmt, tensor_scale) - x
        rms = float(np.sqrt(np.mean(error ** 2)))
        steps = "powers of two" if fmt.scale is UE8M0 else "8 per octave"
        lines.append(f"{fmt.name:<14} {fmt.bits_per_value:>8.2f}     {steps:<14} {rms:>10.5g}   {rms / rms_value:>7.2%}")
    return "\n".join(lines)
