# Numerical contract

What warplens computes exactly, what it approximates, and how each rule was
checked. "On a GPU" without a name means an RTX 3080 Ti (Ampere). Three
things here were measured on a B200, on 2026-11-25
([b200-rental-1.md](b200-rental-1.md)), and say so where they are stated:
`cvt.rn.tf32.f32` and `cvt.rz.tf32.f32`, the arithmetic of `tcgen05.mma`, and
the GPU's FP8 and FP4 conversions. On the same day the Ampere programs behind
`nan_golden.txt`, `intdiv_golden.txt`, `minmax_cvt_golden.txt` and
`cvt_ftz_golden.txt` were rerun on the B200 and reproduced those files line
for line (the runs `formats/ampere_replay.*` of
`validation/blackwell/measured/b200/compare-pinned-nvcc-13.5.37.txt`), the
probe of `.ftz` arithmetic was identical to its prediction (`formats/ftz`),
and `cvt.rna.tf32.f32` gave the Ampere result for all 715 inputs. So where a
row below says "Blackwell may differ", that one B200 did not.

**Bit-exact execution of this model does not by itself establish bit-exact
agreement with Blackwell.** PTX leaves TF32's internal representation
implementation-defined, and the `tcgen05.mma` contract does not say how the
products inside one MMA are accumulated. What the model does there is what a
B200 was measured to do (`tcgen05.mma` below), not something a document
promises.

## Scalar instructions

| Operation | Rule | Checked |
| --- | --- | --- |
| Integer arithmetic | wraps; division by a nonzero divisor truncates toward zero; shifts clamp at the register width | by test, from the PTX rules |
| Integer `div` and `rem` by zero, and the minimum over -1 | by zero: all ones, for `div` and for `rem`, signed and unsigned, 16, 32 and 64 bits. Minimum over -1: the minimum, remainder 0. PTX leaves zero-divisor results unspecified and machine-specific; active lanes report `integer-divide-by-zero` | on a GPU (Ampere): `validation/ampere/intdiv_golden.txt`, 5808 cases. Blackwell may differ |
| `add`, `sub`, `mul`, `div`, `sqrt` on floats | IEEE round-to-nearest-even, as NumPy does them | NumPy |
| NaN results | see "NaN bit patterns" below | on a GPU (Ampere), bit for bit: `validation/ampere/nan_golden.txt`, 14270 cases. Blackwell may differ |
| `fma`, floating `mad` | exact product and sum with integer significands, rounded once | against exact rational arithmetic, and against libc `fma` by the reviewer; on a B200, bit for bit, for the 10^4 cases per type of the probe `formats/fma` |
| `min`, `max` | a NaN operand yields the other; `min(+0, -0)` is `-0`, `max` is `+0`. Two NaNs give `0x7fffffff` for f32, and the second operand made quiet for f64 | on a GPU, bit for bit: `validation/ampere/minmax_cvt_golden.txt` (f32 and f64) |
| float to integer | saturates; NaN gives 0, or the sign-bit pattern when the source is f64 or the result is 64-bit | on a GPU, bit for bit: the same file (f32 and f64 to s32, u32, s64, u64, all four rounding modes) |
| `cvt` between f16, f32 and f64 | IEEE round-to-nearest-even, as NumPy does it; NaNs as below | on a GPU, bit for bit, around the subnormal range of each type: `validation/ampere/cvt_ftz_golden.txt` |
| `cvt.{rni,rzi,rmi,rpi}.f32.f32`, `.f64.f64` (`rintf`, `truncf`, `floorf`, `ceilf`) | NumPy's `rint`, `trunc`, `floor`, `ceil`; NaNs as arithmetic gives them | on a GPU, bit for bit: the same file |
| `cvt` with `.ftz` | see "Flush to zero" below | on a GPU, bit for bit: the same file, 8120 cases with the two rows above |
| `cvt.rna.tf32.f32` | see below | on a GPU, bit for bit |
| `cvt.rn.tf32.f32`, `cvt.rz.tf32.f32` | see below | on a B200, bit for bit, 715 inputs; finite values also against exact rational arithmetic |

## NaN bit patterns

[PTX leaves f32 NaN encodings unspecified, while f64 NaN payloads are supported](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions)
(except the documented approximate reciprocal operations). This does not establish
the complete operand-selection and quieting policy measured below. In particular,
[the `abs.f64` section](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs)
says a NaN passes through unchanged; the recorded Ampere results quiet signalling
NaNs. The table records this model's empirical policy, not a blanket PTX guarantee.
The simulator returns what an RTX 3080 Ti returned (`validation/ampere/nan_golden.txt`,
`validation/ampere/ftz_golden.txt`, `validation/ampere/minmax_cvt_golden.txt`,
`validation/ampere/cvt_ftz_golden.txt`; `tests/test_hardware_golden.py`), implemented in
`warplens/units/alu.py:_gpu_nans`, `warplens/units/alu.py:_converted_nans` and
`warplens/units/alu.py:_float_to_float`:

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
| `cvt.ftz.f64.f32` | `0x7fffffffe0000000` for every NaN: under `.ftz` an f32 NaN source first becomes `0x7fffffff`, and that is converted. `cvt.rn.ftz.f32.f64` is as without `.ftz` |
| `cvt.f32.f32`; `cvt.ftz.f32.f32` | the bits, unchanged; `0x7fffffff` |
| `cvt.{rni,rzi,rmi,rpi}` to the same floating type | f32: `0x7fffffff`. f64: the operand made quiet, sign and payload kept |
| `mov` | the bits, unchanged |

Not measured: NaNs of `div.full`, `div.approx` and `sqrt.approx` (in the
simulator they share the code of `div.rn` and `sqrt.rn` and so give
`0x7fffffff` as well), rounding modes other than `.rn` (not implemented), and
f16 arithmetic. A NaN that the tensor core or
TMA's NaN fill produces is a separate claim (TMA-05). All of this is Ampere
observations. The scalar probes compare NaN classification and non-NaN values
bit for bit; `validation/blackwell/formats/ampere_replay` reruns the Ampere
programs on the B200 and compares the NaN bits too (NUM-11, NUM-13), so a
Blackwell payload policy that differs from Ampere's shows as a difference. On
the B200 of 2026-11-25 none showed: the reruns were identical to the Ampere
files. The Ampere golden tests pin the empirical policy without requiring it
of every Blackwell.

### Flush to zero

`.ftz` is about f32 only (the PTX manual: "subnormal inputs and results to
sign-preserving zero"). What counts as subnormal was measured on the RTX 3080 Ti
(`validation/ampere/ftz_golden.txt` for arithmetic,
`validation/ampere/cvt_ftz_golden.txt` for `cvt`):

- A subnormal f32 **source** is a zero of its sign. `cvt.rpi.ftz.f32.f32` and
  `cvt.rpi.ftz.s32.f32` of the smallest positive subnormal give 0, where the forms
  without `.ftz` give 1; `cvt.ftz.f64.f32` gives a zero.
- An f32 **result** is a zero of its sign when it is tiny after rounding: when the
  exact result, rounded to 24 bits as if the exponent had no lower limit, is still
  below 2^-126. `cvt.rn.ftz.f32.f64` gives zero for every magnitude below
  2^-126 (1 - 2^-25) (`0x380ffffff0000000`), although without `.ftz` the values
  from 2^-126 (1 - 2^-24) on round to the smallest normal number. The same rule
  as for `mul`, `div` and `fma`.
- Subnormal **f16** values are left alone, as the source of `cvt.ftz.f32.f16` and
  as the result of `cvt.rn.ftz.f16.f32`.
- ptxas takes `.ftz` on `cvt` only when the source or the result is f32
  ("Modifier '.ftz' does not apply to this type of convert operation"), and the
  simulator refuses the other forms too. `.ftz` with an integer source was not
  measured and is refused.

Compilers emit these forms under fast-math: `cvt.ftz.f64.f32` and
`cvt.rn.ftz.f32.f64` for casts, `cvt.rzi.ftz.s32.f32` for `(int)x`,
`cvt.rmi.ftz.f32.f32` for `floorf`.

### Integer zero divisors

[Integer `div`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)
and [`rem`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)
with a zero divisor produce an unspecified, machine-specific value. Active lanes
report `integer-divide-by-zero`; execution continues with the measured Ampere
all-ones fallback, signed or unsigned, at 16, 32 and 64 bits. The scalar-edge
probe records those values as informational only; the rerun of
`validation/ampere/intdiv_golden.cu` on the B200
(`validation/blackwell/formats/ampere_replay`, NUM-12) compares them with
Ampere's, and was identical. Floating-point division by
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

`.rn` (ties to even) and `.rz` (toward zero) round finite values and
subnormals the same way with a different tie rule, and leave an infinity as it
is. A NaN is different: every NaN input, whatever its sign and payload, gives
`0x7FFFE000`, so these two never turn a NaN into infinity. They need Hopper or
newer; all of this is **measured on a B200** (2026-11-25,
`validation/blackwell/measured/b200/formats/tf32_modes.txt`, 715 inputs).

`warplens.numerics.explain_tf32(0.1)` shows one conversion bit by bit:

```
float32 -> TF32 with cvt.rna.tf32.f32
              sign exponent fraction kept  fraction discarded
  float32     0 01111011 1001100110 0110011001101   0.100000001
  TF32        0 01111011 1001100110 0000000000000   0.0999755859
  discarded bits 0110011001101 (3277 of 8192 of the last kept place): rounded down; error -2.44155526e-05 (0.000244 relative)
```

## `tcgen05.mma`

Measured on a B200 on 2026-11-25 (`docs/b200-rental-1.md`). An element of the
result is **not** a chain of float32 additions; it is one sum, computed like
this (`warplens/units/tcgen05.py:mma_sum`):

| Step | What happens | Status |
| --- | --- | --- |
| Operands | half precision: widened exactly. `kind::tf32`: the low 13 bits of every element are cleared, toward zero like `cvt.rz.tf32.f32`; a launch whose operands have such bits reports `mma-tf32-operand-not-converted`, because a kernel that wanted rounding to nearest had to convert first | measured: all 256 products of `mma/tf32_low_bits` |
| Products | exact: two significands of 11 bits | by construction |
| Lining up | every term (the K products and, when accumulating, the old value) is cut off, toward zero, 25 bits below the leading bit of the largest term. The leading bit of a product is taken from the **sum of its operands' exponents**, as if the product of the two significands had not carried into the next bit | the width: measured (`mma/accumulation_order`, part 1). The exponent rule and the direction for negative terms: from the book's kernels, below |
| Sum | the cut terms are added exactly | measured (parts 1 and 2) |
| Result | the sum is cut off **toward zero** to float32, not rounded to nearest. 2^128 or more is an infinity | measured (part 2; `mma/tf32_low_bits` for the infinity) |
| Accumulating into D | the old value, scaled by 2^-s if a scale is given, is one more term of the same sum: not a separate float32 addition | measured (part 3) |
| Negation flags | sign applied to the product | from the descriptor layout; `mma/negate` identical on the B200 |

Two things follow that a float32 model gets wrong. A term more than 2^25
times smaller than the largest is lost whatever its position: with products
of 5 x 2^24, 5 and -5 x 2^24 the result is 4, in every order. And nothing is
rounded to nearest, so results lean toward zero; over a long K the difference
from a float32 sum grows, which is what the book's `hgemm_v6` to `v9` (K =
256) showed against the first model: up to 30 units in the last place of the
half-precision output.

What this rests on. `validation/blackwell/measured/b200/mma/accumulation_order.txt`
(3016 words) and `tf32_low_bits.txt` (256) are reproduced word for word
(`tests/test_tcgen05.py::test_the_mma_sum_is_the_one_a_b200_computes`). The
nine kernels of the book, captured on the same GPU with their inputs, then
replay bit for bit: 311,296 half-precision elements. With any one rule
changed some elements are a unit off: products normalized before lining up
(9 elements of `v6`, `v8`, `v9`), negative terms cut toward minus infinity
(495), the old value added without being cut (143), a width of 24, 26 or 27
bits (204, 123, 146). Those captures are in the rental's archive and not in
the repository, so parts 4 to 6 of `mma/accumulation_order` ask a GPU for the
same three rules directly.

**Not measured:** operands that are subnormal in half precision (their
exponent is taken from the value), elements with an infinity or a NaN among
their terms (computed in float32 as before), and `kind::i8`.

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

What has been compared with a GPU is one layer down: the `cvt` forms for
these formats that the simulator executes (`warplens/units/small_formats.py`),
which `tests/test_small_format_cvt.py` holds against this host layer. On a
B200 the GPU's conversions gave what those forms give for every code of E4M3,
E5M2 and E2M1, for the E8M0 ties and for the special inputs (the runs
`formats/fp8_codes`, `fp4_codes`, `e8m0_ties` and `special_inputs`, all
identical). UE4M3 has not been compared with anything: ptxas 13.3 and 13.5
have no conversion for it (NUM-05). The host API itself, and the block
formats below, were not run on a GPU.

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
