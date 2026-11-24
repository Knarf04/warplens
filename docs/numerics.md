# Numerical contract

What warplens computes exactly, what it approximates, and how each rule was
checked. "Checked on a GPU" means an RTX 3080 Ti (Ampere); nothing here has
been checked on Blackwell.

**Bit-exact execution of this model does not establish bit-exact agreement
with Blackwell.** PTX leaves TF32's internal representation
implementation-defined, and the `tcgen05.mma` contract does not say how the
products inside one MMA are accumulated.

## Scalar instructions

| Operation | Rule | Checked |
| --- | --- | --- |
| Integer arithmetic | wraps; division by a nonzero divisor truncates toward zero; shifts clamp at the register width | by test, from the PTX rules |
| Integer `div` and `rem` by zero, and the minimum over -1 | by zero: all ones, for `div` and for `rem`, signed and unsigned, 16, 32 and 64 bits. Minimum over -1: the minimum, remainder 0. PTX leaves zero-divisor results unspecified and machine-specific; active lanes report `integer-divide-by-zero` | on a GPU (Ampere): `validation/ampere/intdiv_golden.txt`, 5808 cases. Blackwell may differ |
| `add`, `sub`, `mul`, `div`, `sqrt` on floats | IEEE round-to-nearest-even, as NumPy does them | NumPy |
| NaN results | see "NaN bit patterns" below | on a GPU (Ampere), bit for bit: `validation/ampere/nan_golden.txt`, 14270 cases. Blackwell may differ |
| `fma`, floating `mad` | exact product and sum with integer significands, rounded once | against exact rational arithmetic, and against libc `fma` by the reviewer |
| `min`, `max` | a NaN operand yields the other; `min(+0, -0)` is `-0`, `max` is `+0`. Two NaNs give `0x7fffffff` for f32, and the second operand made quiet for f64 | on a GPU, bit for bit: `validation/ampere/minmax_cvt_golden.txt` (f32 and f64) |
| float to integer | saturates; NaN gives 0, or the sign-bit pattern when the source is f64 or the result is 64-bit | on a GPU, bit for bit: the same file (f32 and f64 to s32, u32, s64, u64, all four rounding modes) |
| `cvt.rna.tf32.f32` | see below | on a GPU, bit for bit |
| `cvt.rn.tf32.f32`, `cvt.rz.tf32.f32` | see below | against exact rational arithmetic only |

## NaN bit patterns

[PTX leaves f32 NaN encodings unspecified, while f64 NaN payloads are supported](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions)
(except the documented approximate reciprocal operations). This does not establish
the complete operand-selection and quieting policy measured below. In particular,
[the `abs.f64` section](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs)
says a NaN passes through unchanged; the recorded Ampere results quiet signalling
NaNs. The table records this model's empirical policy, not a blanket PTX guarantee.
The simulator returns what an RTX 3080 Ti returned (`validation/ampere/nan_golden.txt`,
`validation/ampere/ftz_golden.txt`, `validation/ampere/minmax_cvt_golden.txt`;
`tests/test_hardware_golden.py`), implemented in
`warplens/units/alu.py:_gpu_nans` and `warplens/units/alu.py:_converted_nans`:

| Result | NaN |
| --- | --- |
| f32, from `add`, `sub`, `mul`, `div`, `fma`, `sqrt`, with or without `.rn` and `.ftz` | `0x7fffffff`, whatever the operands: an invalid operation, a quiet or a signalling NaN of either sign and any payload |
| f32, from `neg` and `abs` of a NaN | `0x7fffffff` too: the sign is not changed |
| f32, from `min` / `max` of two NaNs | `0x7fffffff` |
| f64, invalid operation (`inf - inf`, `0 * inf`, `0 / 0`, `inf / inf`, `sqrt` of a negative number, the same inside `fma`) | `0xfff8000000000000` |
| f64, with a NaN operand | that operand with the quiet bit set, sign and payload kept. Of several NaN operands: the second, else the first, for `add`, `sub` and `mul`; the first, else the second, for `div`; the second, else the third, else the first, for `fma`; the second for `min` / `max` |
| f64, `neg` and `abs` of a NaN | the operand made quiet; the sign is not changed |
| `cvt` between f16 and f32 | `0x7fff` and `0x7fffffff` |
| `cvt` to or from f64 | the sign kept, the quiet bit set, the payload's leading bits kept (shifted to the new width) |
| `mov` | the bits, unchanged |

Not measured: NaNs of `div.full`, `div.approx` and `sqrt.approx` (in the
simulator they share the code of `div.rn` and `sqrt.rn` and so give
`0x7fffffff` as well), rounding modes other than `.rn` (not implemented), and
f16 arithmetic. `cvt` with `.ftz` is recorded in the golden file and not
implemented. A NaN that the tensor core or
TMA's NaN fill produces is a separate claim (TMA-05). All of this is Ampere
observations; NUM-13 records Blackwell NaN bits as informational. The hardware
probes compare NaN classification and non-NaN values bit for bit. The Ampere
golden tests also pin the empirical payload policy, without requiring it of Blackwell.

### Integer zero divisors

[Integer `div`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)
and [`rem`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)
with a zero divisor produce an unspecified, machine-specific value. Active lanes
report `integer-divide-by-zero`; execution continues with the measured Ampere
all-ones fallback, signed or unsigned, at 16, 32 and 64 bits. The scalar-edge
probe records those values as informational only. Floating-point division by
zero is a separate IEEE operation and does not trigger this integer diagnostic.

## TF32

TF32 is float32 with 10 fraction bits instead of 23. As a 32-bit pattern, a
TF32 value has its low 13 fraction bits zero.

`cvt.rna.tf32.f32` rounds to nearest with ties away from zero. On the GPU:

- Rounding acts on the magnitude bits: add half of the last kept place, then
  clear the low 13 bits. A carry into the exponent is the correct result.
- Subnormals round the same way, on the same bit positions.
- The largest finite values round up to infinity.
- When the exponent is all ones, the low 13 bits are cleared with no rounding.
  Infinity stays infinity. A NaN whose payload was only in those bits becomes
  infinity; other NaNs keep their high payload bits.

`.rn` (ties to even) and `.rz` (toward zero) are modeled the same way with a
different tie rule. They need Hopper or newer, so they are **not checked on
hardware**; that they treat subnormals and NaNs like `.rna` is an assumption.

`warplens.units.floating.explain_tf32(0.1)` shows one conversion bit by bit:

```
float32 -> TF32 with cvt.rna.tf32.f32
              sign exponent fraction kept  fraction discarded
  float32     0 01111011 1001100110 0110011001101   0.100000001
  TF32        0 01111011 1001100110 0000000000000   0.0999755859
  discarded bits 0110011001101 (3277 of 8192 of the last kept place): rounded down; error -2.44155526e-05 (0.000244 relative)
```

## `tcgen05.mma`

| Aspect | What warplens does | Status |
| --- | --- | --- |
| Half-precision operands | converted to float32 exactly | exact by construction |
| `kind::tf32` operands | used as the float32 values stored in shared memory. **No conversion is applied.** If any operand has bits below TF32 precision, the launch reports `mma-tf32-operand-not-converted` | an assumption; what hardware does with those bits is not specified |
| Products and their sum | float32, in whatever order NumPy's matrix product uses | **not** the hardware's order; results can differ in the last bits |
| Accumulating into D | one float32 addition of the new sum to the old value, scaled if a scale is given | an assumption |
| Negation flags | sign applied to the product | from the descriptor layout |

So an MMA result is correct to float32 rounding error, and is not meant to
match hardware bit for bit. Tests compare against NumPy with a tolerance.

## Small formats and block scaling (host side only)

`warplens.numerics` models the formats behind Blackwell's low-precision MMAs
through their bit patterns. **It does not execute the block-scaled
`tcgen05.mma` instructions**; it encodes, decodes and quantizes on the host.

| Format | Bits | Notes |
| --- | --- | --- |
| `E4M3` | 8 | one NaN code per sign, no infinity, largest 448 |
| `E5M2` | 8 | IEEE-style infinity and NaNs, largest 57344 |
| `E2M1` | 4 | no NaN or infinity; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| `UE8M0` | 8 | unsigned power of two, `2^(code - 127)`; a block scale |
| `UE4M3` | 8 | E4M3 without the sign; a block scale |

Bit widths and NaN conventions are taken from `cutlass/exmy_base.h`. Every
format has at most 256 codes, so decoding is tested for every code against the
definition evaluated with exact arithmetic, and encoding is tested on every
midpoint between neighbouring values. Element-format ties go to the even
significand; UE8M0 ties go to the next power of two, following the
[CUDA E8M0 conversion rule](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html).
Encoding saturates at the largest finite value, like PTX's `.satfinite`.
Negative inputs to unsigned formats and NaN inputs to E2M1 are rejected by
this host API; it does not reproduce every CUDA conversion special case.
None of this has been compared with a GPU's conversion instructions.

| Block format | Elements | Scale | Per block |
| --- | --- | --- | --- |
| MXFP8 | `E4M3` or `E5M2` | `UE8M0`, powers of two | 32 |
| MXFP4 | `E2M1` | `UE8M0`, powers of two | 32 |
| NVFP4 | `E2M1` | `UE4M3`, 8 steps per octave, plus one float32 scale per tensor | 16 |

How a scale is chosen is a recipe, not part of the format. `quantize` uses the
[OCP Microscaling recipe](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
for MX (put the block's peak in the element format's top binade) and peak / 6
for NVFP4; you can pass your own scales. NVFP4's automatically chosen tensor
scale is clamped to the positive finite float32 range before rounding; explicit
tensor scales must remain positive and finite after float32 conversion. Two effects
are visible with `explain_block` and `compare_formats`:

- Under the MX recipe the peak can saturate. E2M1's top binade is [4, 8) but
  its largest value is 6, so a scaled value of 7.5 is stored as 6.
- NVFP4's finer scale puts each block's peak within a few percent of the top
  of the range, where a power-of-two scale can leave up to half of it unused.

On 6400 standard normal samples:

```
format         bits/value   scale steps      RMS error   relative
MXFP8 (E4M3)       8.25     powers of two    0.030038     3.01%
MXFP8 (E5M2)       8.25     powers of two    0.053614     5.37%
MXFP4              4.25     powers of two     0.11534    11.55%
NVFP4              4.50     8 per octave     0.094871     9.50%
```

## Not built

- A reference mode that evaluates an MMA with explicit, documented rounding at
  every step, for small examples.
- Bit inspection for anything other than a single TF32 conversion.
- Execution of bf16, 8-, 6- and 4-bit and block-scaled MMAs: the packed operands,
  the descriptor fields, and the scale matrices in TMEM with their layout and
  synchronization. The 6-bit formats (E3M2, E2M3) are not in the host layer either.
