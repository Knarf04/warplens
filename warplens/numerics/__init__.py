"""The numerical helpers meant to be read and tried: every one is importable from here.

    small formats    `FloatFormat` and `E4M3`, `E5M2`, `E2M1`, `UE8M0`, `UE4M3` (formats.py)
    block scaling    `BlockScaled` and `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP4`, `NVFP4`; `choose_scales`,
                     `quantize`, `dequantize`, `explain_block`, `compare_formats` (blockscale.py)
    TF32             `to_tf32`, `explain_tf32`, `TF32_MODES`
    one rounding     `fused_multiply_add`

The small formats and block scaling are modeled through their bit patterns, on the host. TF32
and the fused multiply-add are what the simulator itself computes with: they are defined in
warplens/units/floating.py, next to the instructions that use them, and exported here.
"""

from ..units.floating import TF32_MODES, explain_tf32, fused_multiply_add, to_tf32
from .blockscale import (MXFP4, MXFP8_E4M3, MXFP8_E5M2, NVFP4, BlockScaled, choose_scales,
                         compare_formats, dequantize, explain_block, quantize)
from .formats import E2M1, E4M3, E5M2, UE4M3, UE8M0, FloatFormat

__all__ = ["BlockScaled", "E2M1", "E4M3", "E5M2", "FloatFormat", "MXFP4", "MXFP8_E4M3",
           "MXFP8_E5M2", "NVFP4", "TF32_MODES", "UE4M3", "UE8M0", "choose_scales",
           "compare_formats", "dequantize", "explain_block", "explain_tf32",
           "fused_multiply_add", "quantize", "to_tf32"]
