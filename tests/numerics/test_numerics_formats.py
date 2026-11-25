"""Small float formats and block scaling. With at most 256 codes per format, decoding,
round trips and every rounding tie are tested exhaustively."""

from fractions import Fraction

import numpy as np
import pytest

from warplens.numerics import (E2M1, E4M3, E5M2, MXFP4, MXFP8_E4M3, MXFP8_E5M2, NVFP4, UE4M3, UE8M0,
                               compare_formats, dequantize, explain_block, quantize)

FORMATS = [E4M3, E5M2, E2M1, UE8M0, UE4M3]
BLOCK_FORMATS = [MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4]


def reference_value(fmt, code):
    """Decode one code from the format's definition, with exact arithmetic.
    Returns a Fraction, or 'inf', '-inf' or 'nan'."""
    e, f = fmt.exponent_bits, fmt.fraction_bits
    fraction = code & ((1 << f) - 1)
    exponent = (code >> f) & ((1 << e) - 1)
    sign = -1 if fmt.signed and (code >> (e + f)) & 1 else 1
    bias = 2 ** (e - 1) - 1
    if exponent == 2 ** e - 1:
        if fmt.nan == "ieee":
            return "nan" if fraction else ("inf" if sign > 0 else "-inf")
        if fmt.nan == "canonical" and fraction == 2 ** f - 1:
            return "nan"
    if exponent == 0 and f:
        return sign * Fraction(fraction, 2 ** f) * Fraction(2) ** (1 - bias)
    return sign * (1 + Fraction(fraction, 2 ** f)) * Fraction(2) ** (exponent - bias)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_every_code_decodes_to_its_defined_value(fmt):
    assert fmt.codes == 2 ** (fmt.exponent_bits + fmt.fraction_bits + fmt.signed)
    for code in range(fmt.codes):
        expect, got = reference_value(fmt, code), float(fmt.decode(code))
        if expect == "nan":
            assert np.isnan(got), code
        elif isinstance(expect, str):
            assert got == float(expect), code
        else:
            assert Fraction(got) == expect, code


def test_the_formats_at_a_glance():
    assert sorted(set(np.abs(E2M1.table).tolist())) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert (E4M3.largest, E5M2.largest, UE4M3.largest) == (448, 57344, 448)
    assert E4M3.decode(0b0_0000_001) == 2.0 ** -9 and E5M2.decode(0b0_00000_01) == 2.0 ** -16
    assert (E4M3.nan_code, UE4M3.nan_code, UE8M0.nan_code, E2M1.nan_code) == (0x7F, 0x7F, 0xFF, None)
    assert np.isinf(E5M2.table).sum() == 2 and np.isnan(E5M2.table).sum() == 6
    assert np.isnan(E4M3.table).sum() == 2 and not np.isinf(E4M3.table).any()  # +NaN and -NaN codes
    assert UE8M0.decode(127) == 1 and UE8M0.decode(130) == 8 and UE8M0.decode(0) == 2.0 ** -127
    assert E4M3.bit_string(0b1_0111_010) == "1 0111 010" and UE8M0.bit_string(127) == "01111111"
    with pytest.raises(ValueError, match="codes must be in 0..15"):
        E2M1.decode(16)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_encoding_round_trips_every_finite_code(fmt):
    finite = np.flatnonzero(np.isfinite(fmt.table))
    np.testing.assert_array_equal(fmt.encode(fmt.table[finite]), finite)  # including -0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_every_midpoint_uses_the_format_tie_rule_and_neighbours_round_to_nearest(fmt):
    half = fmt.codes // 2 if fmt.signed else fmt.codes
    codes = np.flatnonzero(np.isfinite(fmt.table[:half]))
    for lower, upper in zip(codes[:-1], codes[1:]):
        a, b = Fraction(float(fmt.table[lower])), Fraction(float(fmt.table[upper]))
        middle = (a + b) / 2
        even = upper if fmt is UE8M0 else (lower if lower % 2 == 0 else upper)
        assert fmt.encode(float(middle)) == even
        assert fmt.encode(float(middle - (b - a) / 8)) == lower
        assert fmt.encode(float(middle + (b - a) / 8)) == upper
        if fmt.signed:
            assert fmt.encode(-float(middle)) == even + half


def test_saturation_infinity_and_nan():
    assert E4M3.encode([1e9, -1e9, np.inf]).tolist() == [0x7E, 0xFE, 0x7E]  # satfinite: +-448
    assert E2M1.encode([100.0, -100.0, 6.9]).tolist() == [0b0111, 0b1111, 0b0111]
    assert np.isinf(E5M2.decode(E5M2.encode([1e9], saturate=False))).all()
    assert E5M2.decode(E5M2.encode([60000.0], saturate=False)) == 57344  # within half a step: stays finite
    assert E4M3.encode([np.nan]).tolist() == [0x7F]
    with pytest.raises(ValueError, match="E2M1 cannot represent NaN"):
        E2M1.encode([np.nan])
    with pytest.raises(ValueError, match="no infinity"):
        E4M3.encode([1e9], saturate=False)
    with pytest.raises(ValueError, match="UE8M0 is unsigned"):
        UE8M0.encode([-1.0])
    # UE8M0 nearest rounds exact power-of-two midpoints upward, regardless of exponent-code parity.
    assert UE8M0.encode([1.0, 3.0, 0.75, 2.9, 3.1]).tolist() == [127, 129, 127, 128, 129]


# -- block scaling ----------------------------------------------------------------------------

def data(n=128, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


@pytest.mark.parametrize("fmt", BLOCK_FORMATS, ids=lambda f: f.name)
def test_quantize_dequantize(fmt):
    x = data()
    codes, scales, tensor_scale = quantize(x, fmt)
    assert codes.shape == (128,) and scales.shape == (128 // fmt.block,)
    assert codes.max() < fmt.element.codes and (tensor_scale == 1.0) != fmt.tensor_scale
    back = dequantize(codes, scales, fmt, tensor_scale)
    factor = np.repeat(fmt.scale.decode(scales) * tensor_scale, fmt.block)
    levels = np.sort(np.unique(np.abs(fmt.element.table[np.isfinite(fmt.element.table)])))
    worst_step = np.diff(levels).max()
    # A value beyond the largest element saturates; every other one is within half a step.
    saturated = np.abs(x) / factor > fmt.element.largest
    np.testing.assert_allclose(np.abs(back[saturated]), fmt.element.largest * factor[saturated])
    assert (np.abs(back - x)[~saturated] <= worst_step / 2 * factor[~saturated] * (1 + 1e-9)).all()
    assert (np.sign(back) * np.sign(x) >= 0).all()


def test_the_mx_recipe_can_saturate_the_largest_values():
    # The recipe puts the peak in the top binade, [4, 8) for E2M1, but E2M1 stops at 6.
    x = np.zeros(32)
    x[0], x[1] = 7.5, 5.0
    codes, scales, _ = quantize(x, MXFP4)
    back = dequantize(codes, scales, MXFP4)
    assert MXFP4.scale.decode(scales[0]) == 1 and back[0] == 6 and back[1] == 4  # 5 ties to the even code
    codes, scales, t = quantize(x[:16], NVFP4)  # NVFP4 scales the peak onto 6 instead
    assert dequantize(codes, scales, NVFP4, t)[0] == pytest.approx(7.5, rel=1e-6)


@pytest.mark.parametrize("fmt", [MXFP8_E4M3, MXFP8_E5M2, MXFP4], ids=lambda f: f.name)
def test_mx_scale_is_the_power_of_two_that_puts_the_peak_in_the_top_binade(fmt):
    x = data()
    codes, scales, _ = quantize(x, fmt)
    top = 2.0 ** np.floor(np.log2(fmt.element.largest))
    for b in range(len(scales)):
        block = x[b * fmt.block:(b + 1) * fmt.block]
        scale = float(fmt.scale.decode(scales[b]))
        assert np.log2(scale) == int(np.log2(scale))
        assert top <= np.abs(block).max() / scale < 2 * top


def test_nvfp4_scale_tracks_the_peak_much_more_closely():
    x = data(16 * 64, seed=3)
    codes, scales, tensor_scale = quantize(x, NVFP4)
    factor = NVFP4.scale.decode(scales) * tensor_scale
    peak = np.abs(x.reshape(-1, 16)).max(axis=1)
    used = peak / factor / 6  # fraction of the E2M1 range the largest value occupies
    assert 0.93 < used.min() and used.max() < 1.07  # UE4M3 steps are 1/8 of an octave
    codes4, scales4, _ = quantize(x, MXFP4)
    used4 = np.abs(x.reshape(-1, 32)).max(axis=1) / MXFP4.scale.decode(scales4) / 6
    assert used4.min() < 0.75 and used4.max() <= 4 / 3  # a power-of-two scale can waste up to half the range
    assert float(np.float32(tensor_scale)) == tensor_scale and scales.max() == 0x7E  # largest block uses 448


def test_why_nvfp4_and_mxfp4_differ_on_the_same_four_bit_elements():
    x = data(32 * 200, seed=1)
    def rms(fmt):
        codes, scales, t = quantize(x, fmt)
        return float(np.sqrt(np.mean((dequantize(codes, scales, fmt, t) - x) ** 2)))
    assert rms(NVFP4) < 0.9 * rms(MXFP4)  # finer scales and smaller blocks: about 18% less here
    assert rms(MXFP8_E4M3) < 0.5 * rms(NVFP4)  # four more bits per element
    assert rms(MXFP8_E4M3) < rms(MXFP8_E5M2)  # an extra fraction bit beats extra exponent range here
    text = compare_formats(x)
    assert [line.split()[0] for line in text.splitlines()[1:]] == ["MXFP8", "MXFP8", "MXFP4", "NVFP4"]
    assert "4.25" in text and "4.50" in text and "8.25" in text


def test_blocks_are_independent_and_boundaries_are_respected():
    x = data(96)
    codes, scales, _ = quantize(x, MXFP4)
    y = x.copy()
    y[32:64] *= 1000  # only the middle block changes
    codes2, scales2, _ = quantize(y, MXFP4)
    np.testing.assert_array_equal(codes[:32], codes2[:32])
    np.testing.assert_array_equal(codes[64:], codes2[64:])
    assert (scales[0], scales[2]) == (scales2[0], scales2[2]) and scales2[1] > scales[1]
    for bad in (data(33), data(64).reshape(2, 32)):
        with pytest.raises(ValueError, match="multiple of 32"):
            quantize(bad, MXFP4)
    with pytest.raises(ValueError, match="16 element codes per scale"):
        dequantize(codes[:32], scales[:1], NVFP4)
    with pytest.raises(ValueError, match="NaN or infinity"):
        quantize(np.full(32, np.nan), MXFP4)


def test_explicit_scales_saturate_and_zero_blocks_work():
    x = np.linspace(-8, 8, 32)
    codes, scales, _ = quantize(x, MXFP4, scales=np.array([127], dtype=np.uint8))  # scale 1: range +-6
    back = dequantize(codes, scales, MXFP4)
    assert back.max() == 6 and back.min() == -6
    zeros = np.zeros(32)
    for fmt in BLOCK_FORMATS:
        z = np.zeros(fmt.block)
        codes, scales, t = quantize(z, fmt)
        assert not dequantize(codes, scales, fmt, t).any()
    del zeros


def test_explain_block_shows_scale_codes_and_errors():
    x = data(64)
    text = explain_block(x, NVFP4, block=1)
    assert "NVFP4, block 1: 16 x E2M1 elements, one UE4M3 scale" in text
    assert "times tensor scale" in text and "RMS error of the block" in text
    assert len(text.splitlines()) == 5 + 16 + 1
    mx = explain_block(x, MXFP4)
    assert "block scale: code 126 = 01111110 = 0.5" in mx and "uses 78% of the element range" in mx


@pytest.mark.parametrize("fmt", [MXFP8_E4M3, MXFP8_E5M2, MXFP4], ids=lambda f: f.name)
def test_mx_scale_exponent_is_exact_below_a_power_of_two(fmt):
    # log2(nextafter(8, 0)) rounds to 3.0, but the peak's binade is still [4, 8).
    peak = np.nextafter(8.0, 0.0)
    codes, scales, tensor_scale = quantize(np.full(fmt.block, peak), fmt)
    top_exponent = {"E4M3": 8, "E5M2": 15, "E2M1": 2}[fmt.element.name]
    assert scales.tolist() == [127 + 2 - top_exponent]
    assert dequantize(codes, scales, fmt, tensor_scale)[0] == fmt.element.largest * 2.0 ** (2 - top_exponent)


def test_nvfp4_small_tensor_keeps_a_nonzero_float32_global_scale():
    x = np.full(16, np.float32(1e-43), dtype=np.float64)
    with np.errstate(all="raise"):
        codes, scales, tensor_scale = quantize(x, NVFP4)
        back = dequantize(codes, scales, NVFP4, tensor_scale)
    assert tensor_scale == float(np.finfo(np.float32).smallest_subnormal)
    np.testing.assert_allclose(back, x, rtol=0.1, atol=0)


@pytest.mark.parametrize("tensor_scale", [0.0, -1.0, np.nan, np.inf, 1e-100, 1e100])
def test_invalid_tensor_scales_cannot_silently_zero_or_nan_the_output(tensor_scale):
    with pytest.raises(ValueError, match="positive finite float32"):
        quantize(np.ones(16), NVFP4, tensor_scale=tensor_scale)
    with pytest.raises(ValueError, match="positive finite float32"):
        quantize(np.ones(16), NVFP4, scales=[56], tensor_scale=tensor_scale)
    with pytest.raises(ValueError, match="positive finite float32"):
        dequantize(np.ones(16, dtype=np.uint8), [56], NVFP4, tensor_scale)


@pytest.mark.parametrize("codes", [[1.9], [np.nan], [np.inf]])
def test_decode_requires_integer_bit_codes(codes):
    with pytest.raises(ValueError, match="integer"):
        E2M1.decode(codes)


@pytest.mark.parametrize("scales", [np.array([-129]), np.array([383]), np.array([127.9])])
def test_quantize_validates_codes_before_narrowing_the_scale_storage(scales):
    with pytest.raises(ValueError, match="codes"):
        quantize(np.ones(32), MXFP4, scales=scales)


@pytest.mark.parametrize("scales", [127, [[127]], [[127, 128]]])
def test_dequantize_requires_exactly_one_scale_per_block(scales):
    with pytest.raises(ValueError, match="32 element codes per scale"):
        dequantize(np.zeros(32, dtype=np.uint8), scales, MXFP4)


def test_zero_nvfp4_block_can_be_explained_without_dividing_by_zero():
    with np.errstate(all="raise"):
        text = explain_block(np.zeros(16), NVFP4)
    assert "RMS error of the block: 0" in text
    assert "nan" not in text
