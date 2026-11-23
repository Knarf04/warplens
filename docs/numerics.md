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
| Integer arithmetic | wraps; division truncates toward zero; shifts clamp at the register width | by test, from the PTX rules |
| `add`, `sub`, `mul`, `div`, `sqrt` on floats | IEEE round-to-nearest-even, as NumPy does them | NumPy |
| `fma`, floating `mad` | exact product and sum with integer significands, rounded once | against exact rational arithmetic, and against libc `fma` by the reviewer |
| `min`, `max` | a NaN operand yields the other; `min(+0, -0)` is `-0`, `max` is `+0` | on a GPU |
| float to integer | saturates; NaN gives 0, or the sign-bit pattern when the source is f64 or the result is 64-bit | on a GPU |
| `cvt.rna.tf32.f32` | see below | on a GPU, bit for bit |
| `cvt.rn.tf32.f32`, `cvt.rz.tf32.f32` | see below | against exact rational arithmetic only |

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

## Not built

- A reference mode that evaluates an MMA with explicit, documented rounding at
  every step, for small examples.
- Bit inspection for anything other than a single TF32 conversion.
- bf16, the 8-, 6- and 4-bit formats, and block scaling.
