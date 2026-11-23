"""TF32 conversion: against a recording from a real GPU, and against exact rational rounding."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.units.floating import explain_tf32, to_tf32

from conftest import kernel

AMPERE = Path(__file__).resolve().parent.parent / "validation" / "ampere"


def golden():
    rows = [line.split() for line in (AMPERE / "tf32_golden.txt").read_text().splitlines()]
    return (np.array([int(a, 16) for a, _ in rows], dtype=np.uint32),
            np.array([int(b, 16) for _, b in rows], dtype=np.uint32))


def test_cvt_rna_tf32_matches_real_gpu_bit_for_bit(sim):
    """tf32_golden.txt is bin/tf32_golden on an RTX 3080 Ti: 715 rounding boundaries,
    subnormals, infinities, NaNs and pseudo-random patterns."""
    inputs, expect = golden()
    assert len(inputs) == 715
    out = sim.empty(len(inputs), "u32")
    stats = sim.launch(AMPERE / "tf32_probe.ptx", "tf32_probe", grid=-(-len(inputs) // 256), block=256,
                       args=[sim.from_numpy(inputs), out, len(inputs)])
    mismatch = np.flatnonzero(out.numpy() != expect)
    assert not len(mismatch), [(hex(inputs[i]), hex(out.numpy()[i]), hex(expect[i])) for i in mismatch[:5]]
    assert not stats.findings
    np.testing.assert_array_equal(to_tf32(inputs, "rna"), expect)


def test_what_the_gpu_recording_shows():
    inputs, expect = golden()
    table = dict(zip(inputs.tolist(), expect.tolist()))
    assert table[0x3F800FFF] == 0x3F800000 and table[0x3F801000] == 0x3F802000  # tie goes away from zero
    assert table[0xBF801000] == 0xBF802000  # also for negatives
    assert table[0x00000FFF] == 0 and table[0x00001000] == 0x00002000  # subnormals round the same way
    assert table[0x7F7FFFFF] == 0x7F800000  # the largest float rounds up to infinity
    assert table[0x7F800001] == 0x7F800000  # a NaN with only low payload bits becomes infinity
    assert table[0x7FA12345] == 0x7FA12000  # other NaNs keep their high payload
    assert not (expect & 0x1FFF).any()


def exact_tf32(bits: int, mode: str) -> int:
    """Round one finite float32 to TF32's grid with rational arithmetic."""
    sign, magnitude = bits >> 31, bits & 0x7FFFFFFF
    exponent, fraction = magnitude >> 23, magnitude & 0x7FFFFF
    value = Fraction(fraction if exponent == 0 else fraction | (1 << 23)) * Fraction(2) ** (max(exponent, 1) - 150)
    spacing = Fraction(2) ** (max(exponent, 1) - 150 + 13)  # one unit of the last kept fraction bit
    steps, remainder = divmod(value, spacing)
    half = spacing / 2
    if mode == "rna":
        steps += remainder >= half
    elif mode == "rn":
        steps += remainder > half or (remainder == half and steps % 2 == 1)
    rounded = np.float32(float(steps * spacing))  # exactly representable, or overflows to infinity
    return int(np.array([rounded]).view(np.uint32)[0]) | (sign << 31)


@pytest.mark.parametrize("mode", ["rna", "rn", "rz"])
def test_rounding_modes_against_exact_rational_arithmetic(mode):
    inputs, _ = golden()
    finite = inputs[(inputs & 0x7F800000) != 0x7F800000]
    with np.errstate(over="ignore"):
        expect = np.array([exact_tf32(int(b), mode) for b in finite], dtype=np.uint32)
    np.testing.assert_array_equal(to_tf32(finite, mode), expect)


def test_rn_and_rz_differ_from_rna_only_where_they_should():
    tie_even, tie_odd, above = 0x3F801000, 0x3F803000, 0x3F801001
    bits = np.array([tie_even, tie_odd, above], dtype=np.uint32)
    assert to_tf32(bits, "rna").tolist() == [0x3F802000, 0x3F804000, 0x3F802000]
    assert to_tf32(bits, "rn").tolist() == [0x3F800000, 0x3F804000, 0x3F802000]  # tie to even
    assert to_tf32(bits, "rz").tolist() == [0x3F800000, 0x3F802000, 0x3F800000]
    assert to_tf32(np.array([0x7F7FFFFF], dtype=np.uint32), "rz").tolist() == [0x7F7FE000]  # stays finite


def test_cvt_instruction_modes(sim):
    mod = kernel("""
        .reg .b32 %x, %a, %b, %c; .reg .b64 %o;
        ld.param.u32 %x, [x]; ld.param.u64 %o, [out];
        cvt.rna.tf32.f32 %a, %x; cvt.rn.tf32.f32 %b, %x; cvt.rz.tf32.f32 %c, %x;
        st.global.u32 [%o], %a; st.global.u32 [%o+4], %b; st.global.u32 [%o+8], %c;
        ret;
    """, ".param .u32 x, .param .u64 out")
    out = sim.empty(3, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[0x3F801000, out])
    assert out.numpy().tolist() == [0x3F802000, 0x3F800000, 0x3F800000]
    bad = kernel(".reg .b32 %x;\ncvt.rm.tf32.f32 %x, %x;\nret;")
    with pytest.raises(warplens.UnsupportedError, match="TF32 conversion needs .rna, .rn or .rz"):
        sim.launch(bad, "k", grid=1, block=1)


def test_bit_explanation():
    text = explain_tf32(0.1)
    assert "0 01111011 1001100110 0110011001101" in text  # sign, exponent, kept, discarded
    assert "0 01111011 1001100110 0000000000000" in text and "rounded down" in text
    assert "3277 of 8192" in text and "relative" in text
    assert "rounded up" in explain_tf32(np.float32(1.0) + np.float32(2.0) ** -11)
    assert "nothing to round" in explain_tf32(1.5)
    assert "result is infinity" in explain_tf32(3.4028235e38)
    assert "without rounding" in explain_tf32(float("nan"))
    assert "cvt.rz.tf32.f32" in explain_tf32(0.1, "rz")
