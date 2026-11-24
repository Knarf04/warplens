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
| `add`, `sub`, `mul`, `div`, `sqrt` on floats | IEEE round-to-nearest-even, as NumPy does them | NumPy |
| `fma`, floating `mad` | exact product and sum with integer significands, rounded once | against exact rational arithmetic, and against libc `fma` by the reviewer |
| `min`, `max` | a NaN operand yields the other; `min(+0, -0)` is `-0`, `max` is `+0`. Two NaNs give `0x7fffffff` for f32, and the second operand made quiet for f64 | on a GPU, bit for bit: `validation/ampere/minmax_cvt_golden.txt` (f32 and f64) |
| float to integer | saturates; NaN gives 0, or the sign-bit pattern when the source is f64 or the result is 64-bit | on a GPU, bit for bit: the same file (f32 and f64 to s32, u32, s64, u64, all four rounding modes) |
| `cvt.rna.tf32.f32` | see below | on a GPU, bit for bit |
| `cvt.rn.tf32.f32`, `cvt.rz.tf32.f32` | see below | against exact rational arithmetic only |

### Nonportable edge results

Single-precision arithmetic may produce different NaN signs and payloads from
hardware. [PTX leaves f32 NaN encodings unspecified](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions),
so `.ftz` validation compares NaN classification and compares non-NaN values
bit for bit. A NumPy-generated `0xffc00000` versus the RTX 3080 Ti's
`0x7fffffff` is not a functional mismatch. The `min`/`max` NaN policy in the
table records the model's Ampere match; it does not promise a Blackwell payload.

[Integer `div`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div)
and [`rem`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem)
with a zero divisor produce an unspecified, machine-specific value. Active
lanes now report `integer-divide-by-zero`; the model continues using divisor
1, yielding the dividend for `div` and zero for `rem`. These are deterministic
fallbacks, not hardware predictions. The scalar-edge probe records those
values as informational only. Floating-point division by zero is a separate
IEEE operation and does not trigger this integer diagnostic.

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
