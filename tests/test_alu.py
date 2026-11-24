"""Instruction semantics, checked lane by lane against values worked out from the PTX ISA rules."""

import numpy as np
import pytest

import warplens

from conftest import kernel, run_lanes

I32_MIN, I32_MAX = -2**31, 2**31 - 1


def i32(*v): return np.array(v, dtype=np.int32)
def u32(*v): return np.array(v, dtype=np.uint32)
def f32(*v): return np.array(v, dtype=np.float32)


def test_signed_division_truncates_toward_zero(sim):
    a, b = i32(7, -7, 7, -7, 6, -6, 0), i32(2, 2, -2, -2, 3, 3, 5)
    assert run_lanes(sim, "div.s32 %d, %a0, %a1;", ["s32", "s32"], "s32", [a, b]).tolist() == [3, -3, -3, 3, 2, -2, 0]
    assert run_lanes(sim, "rem.s32 %d, %a0, %a1;", ["s32", "s32"], "s32", [a, b]).tolist() == [1, -1, 1, -1, 0, 0, 0]


def test_unsigned_division(sim):
    a, b = u32(7, 0xFFFFFFFF, 10), u32(2, 2, 3)
    assert run_lanes(sim, "div.u32 %d, %a0, %a1;", ["u32", "u32"], "u32", [a, b]).tolist() == [3, 0x7FFFFFFF, 3]
    assert run_lanes(sim, "rem.u32 %d, %a0, %a1;", ["u32", "u32"], "u32", [a, b]).tolist() == [1, 1, 1]


def test_integer_add_wraps(sim):
    out = run_lanes(sim, "add.s32 %d, %a0, %a1;", ["s32", "s32"], "s32", [i32(I32_MAX, -1), i32(1, 1)])
    assert out.tolist() == [I32_MIN, 0]


def test_shifts_clamp_at_register_width(sim):
    a, n = i32(-8, -8, -8, 1, 1), u32(1, 32, 100, 31, 32)
    assert run_lanes(sim, "shr.s32 %d, %a0, %a1;", ["s32", "u32"], "s32", [a, n]).tolist() == [-4, -1, -1, 0, 0]
    assert run_lanes(sim, "shr.u32 %d, %a0, %a1;", ["u32", "u32"], "u32", [a.view(np.uint32), n]).tolist() == \
        [0x7FFFFFFC, 0, 0, 0, 0]
    assert run_lanes(sim, "shl.b32 %d, %a0, %a1;", ["u32", "u32"], "u32", [a.view(np.uint32), n]).tolist() == \
        [0xFFFFFFF0, 0, 0, 0x80000000, 0]


def test_multiply_variants(sim):
    a, b = i32(-3, 0x10000, I32_MIN), i32(5, 0x10000, 2)
    assert run_lanes(sim, "mul.lo.s32 %d, %a0, %a1;", ["s32", "s32"], "s32", [a, b]).tolist() == [-15, 0, 0]
    assert run_lanes(sim, "mul.hi.s32 %d, %a0, %a1;", ["s32", "s32"], "s32", [a, b]).tolist() == [-1, 1, -1]
    assert run_lanes(sim, "mul.wide.s32 %d, %a0, %a1;", ["s32", "s32"], "s64", [a, b]).tolist() == \
        [-15, 2**32, -2**32]
    ua, ub = u32(0xFFFFFFFF, 3), u32(0xFFFFFFFF, 5)
    assert run_lanes(sim, "mul.hi.u32 %d, %a0, %a1;", ["u32", "u32"], "u32", [ua, ub]).tolist() == [0xFFFFFFFE, 0]
    assert run_lanes(sim, "mul.wide.u32 %d, %a0, %a1;", ["u32", "u32"], "u64", [ua, ub]).tolist() == \
        [0xFFFFFFFF**2, 15]


def test_mul_hi_64(sim):
    a = np.array([2**63, 2**64 - 1], dtype=np.uint64)
    b = np.array([4, 2**64 - 1], dtype=np.uint64)
    assert run_lanes(sim, "mul.hi.u64 %d, %a0, %a1;", ["u64", "u64"], "u64", [a, b]).tolist() == [2, 2**64 - 2]
    s = np.array([-1, -2**62], dtype=np.int64)
    t = np.array([1, 8], dtype=np.int64)
    assert run_lanes(sim, "mul.hi.s64 %d, %a0, %a1;", ["s64", "s64"], "s64", [s, t]).tolist() == [-1, -2]


def test_mad_and_fma(sim):
    a, b, c = i32(3, -4), i32(5, 6), i32(7, 8)
    assert run_lanes(sim, "mad.lo.s32 %d, %a0, %a1, %a2;", ["s32"] * 3, "s32", [a, b, c]).tolist() == [22, -16]
    x, y, z = f32(1.5, 2.0), f32(2.0, -3.0), f32(0.25, 1.0)
    assert run_lanes(sim, "fma.rn.f32 %d, %a0, %a1, %a2;", ["f32"] * 3, "f32", [x, y, z]).tolist() == [3.25, -5.0]


@pytest.mark.parametrize("opcode", ["fma", "mad"])
def test_float_multiply_add_rounds_once(sim, opcode):
    # Lane 0 cancels bits lost by a rounded f32 product. In lanes 1 and 2,
    # the exact product is a rounding midpoint and a tiny addend selects its
    # neighbor; evaluating the sum in f64 loses that addend and rounds wrong.
    a = f32(1 + 2**-23, 1 + 2**-23, 1 + 2**-23)
    b = f32(1 - 2**-23, 1.5, 1.5)
    c = f32(-1, -2**-80, 2**-80)
    out = run_lanes(sim, f"{opcode}.rn.f32 %d, %a0, %a1, %a2;", ["f32"] * 3, "f32", [a, b, c])
    assert out[0] == -2**-46
    assert out[1:].view(np.uint32).tolist() == [0x3FC00001, 0x3FC00002]


@pytest.mark.parametrize("ty, dtype", [("f16", np.float16), ("f32", np.float32), ("f64", np.float64)])
def test_fma_rounding_boundaries_and_exceptional_values(sim, ty, dtype):
    info = np.finfo(dtype)
    tiny = info.smallest_subnormal
    largest_subnormal = np.nextafter(dtype(info.tiny), dtype(0))
    # Expected results follow exact binary arithmetic, including ties to even
    # at zero and the normal/subnormal boundary, and finite-product overflow
    # that must not occur before the addition.
    a = np.array([info.max, tiny, -tiny, tiny, tiny, np.nan, np.inf,
                  np.inf, info.max, -np.inf, 1], dtype=dtype)
    b = np.array([2, .5, .5, .5, .5, 1, 0, 1, info.max, -1, 1], dtype=dtype)
    c = np.array([-info.max, 0, -0.0, tiny, largest_subnormal, 1, 0,
                  -np.inf, -np.inf, 0, np.inf], dtype=dtype)
    out = run_lanes(sim, f"fma.rn.{ty} %d, %a0, %a1, %a2;", [ty] * 3, ty, [a, b, c])
    expected = np.array([info.max, 0, -0.0, 2 * tiny, info.tiny, np.nan,
                         np.nan, np.nan, -np.inf, np.inf, np.inf], dtype=dtype)
    np.testing.assert_array_equal(out, expected)
    assert np.signbit(out[1:3]).tolist() == [False, True]


@pytest.mark.parametrize("ty, dtype, delta", [("f16", np.float16, 2**-6),
                                            ("f64", np.float64, 2**-27)])
def test_fma_cancellation_other_widths(sim, ty, dtype, delta):
    inputs = [np.array([v], dtype=dtype) for v in (1 + delta, 1 - delta, -1)]
    out = run_lanes(sim, f"fma.rn.{ty} %d, %a0, %a1, %a2;", [ty] * 3, ty, inputs)
    assert out[0] == -(delta**2)


def test_fma_zero_signs(sim):
    a = f32(0, 0, -0.0, -0.0, 0, 0, -0.0, -0.0, 1)
    b = f32(1, 1, 1, 1, -1, -1, -1, -1, 1)
    c = f32(0, -0.0, 0, -0.0, 0, -0.0, 0, -0.0, -1)
    out = run_lanes(sim, "fma.rn.f32 %d, %a0, %a1, %a2;", ["f32"] * 3, "f32", [a, b, c])
    assert (out == 0).all()
    assert np.signbit(out).tolist() == [False, False, False, True, False, True, False, False, False]


def test_float_arithmetic_on_bit_registers(sim):
    # nvcc keeps floats in .b32 registers and lets the instruction type interpret them.
    a, b = f32(1.5, -0.0, np.inf), f32(2.25, 0.0, -np.inf)
    out = run_lanes(sim, "add.f32 %d, %a0, %a1;", ["f32", "f32"], "f32", [a, b])
    assert out[0] == 3.75 and out[1] == 0.0 and np.isnan(out[2])
    assert run_lanes(sim, "max.f32 %d, %a0, %a1;", ["f32", "f32"], "f32",
                     [f32(np.nan, 1.0), f32(2.0, np.nan)]).tolist() == [2.0, 1.0]


@pytest.mark.parametrize("ty, dtype", [("f16", np.float16), ("f32", np.float32), ("f64", np.float64)])
@pytest.mark.parametrize("opcode, signs", [("min", [True, True, True, False]),
                                          ("max", [False, False, True, False])])
def test_min_max_signed_zero_order(sim, ty, dtype, opcode, signs):
    a = np.array([-0.0, 0.0, -0.0, 0.0, np.nan, -0.0], dtype=dtype)
    b = np.array([0.0, -0.0, -0.0, 0.0, -0.0, np.nan], dtype=dtype)
    out = run_lanes(sim, f"{opcode}.{ty} %d, %a0, %a1;", [ty] * 2, ty, [a, b])
    assert (out == 0).all()
    assert np.signbit(out).tolist() == signs + [True, True]


def test_float_immediates(sim):
    a = f32(1.0, 2.0)
    assert run_lanes(sim, "mul.f32 %d, %a0, 0f40400000;", ["f32"], "f32", [a]).tolist() == [3.0, 6.0]
    assert run_lanes(sim, "mov.b32 %d, 0fBF800000;", ["f32"], "f32", [a]).tolist() == [-1.0, -1.0]


def test_cvt_float_to_int_rounds_and_saturates(sim):
    x = f32(2.5, -2.5, 3.5, 1e20, -1e20, np.nan, -0.9)
    assert run_lanes(sim, "cvt.rzi.s32.f32 %d, %a0;", ["f32"], "s32", [x]).tolist() == \
        [2, -2, 3, I32_MAX, I32_MIN, 0, 0]
    assert run_lanes(sim, "cvt.rni.s32.f32 %d, %a0;", ["f32"], "s32", [x]).tolist() == \
        [2, -2, 4, I32_MAX, I32_MIN, 0, -1]
    assert run_lanes(sim, "cvt.rmi.s32.f32 %d, %a0;", ["f32"], "s32", [x])[[0, 1, 6]].tolist() == [2, -3, -1]
    assert run_lanes(sim, "cvt.rzi.u32.f32 %d, %a0;", ["f32"], "u32", [x]).tolist() == \
        [2, 0, 3, 0xFFFFFFFF, 0, 0, 0]


@pytest.mark.parametrize("aty", ["f16", "f32", "f64"])
@pytest.mark.parametrize("dty", ["s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"])
def test_cvt_nan_integer_result_depends_on_types(sim, aty, dty):
    x = np.array([np.nan], dtype="float" + aty[1:])
    out = run_lanes(sim, f"cvt.rzi.{dty}.{aty} %d, %a0;", [aty], dty, [x])
    expected = 0
    if aty == "f64" or dty in ("s64", "u64"):
        expected = 1 << (int(dty[1:]) - 1)
        if dty.startswith("s"):
            expected = -expected
    assert out.tolist() == [expected]


def test_cvt_integer_widths(sim):
    a = i32(-1, 0x1234ABCD, 300)
    assert run_lanes(sim, "cvt.s64.s32 %d, %a0;", ["s32"], "s64", [a]).tolist() == [-1, 0x1234ABCD, 300]
    assert run_lanes(sim, "cvt.u64.u32 %d, %a0;", ["u32"], "u64", [a.view(np.uint32)]).tolist() == \
        [0xFFFFFFFF, 0x1234ABCD, 300]
    # Narrow result into a wider register: sign- or zero-extended by the result type.
    assert run_lanes(sim, "cvt.s8.s32 %d, %a0;", ["s32"], "s32", [a]).tolist() == [-1, -51, 44]
    assert run_lanes(sim, "cvt.u8.u32 %d, %a0;", ["u32"], "u32", [a.view(np.uint32)]).tolist() == [255, 0xCD, 44]
    assert run_lanes(sim, "cvt.rn.f32.s32 %d, %a0;", ["s32"], "f32", [a]).tolist() == [-1.0, float(np.float32(0x1234ABCD)), 300.0]


def test_cvt_ftz_flushes_f32_subnormals_only(sim):
    """The rules measured on an RTX 3080 Ti (validation/ampere/cvt_ftz_golden.txt, which
    tests/test_hardware_golden.py compares in full)."""
    tiny = np.array([1, 0x807FFFFF, 0x00800000, 0x3FC00000], dtype=np.uint32).view(np.float32)
    assert run_lanes(sim, "cvt.ftz.f64.f32 %d, %a0;", ["f32"], "f64", [tiny]).view(np.uint64).tolist() == \
        [0, 1 << 63, 0x3810000000000000, 0x3FF8000000000000]
    assert run_lanes(sim, "cvt.rpi.ftz.s32.f32 %d, %a0;", ["f32"], "s32", [tiny]).tolist() == [0, 0, 1, 2]
    assert run_lanes(sim, "cvt.rpi.s32.f32 %d, %a0;", ["f32"], "s32", [tiny]).tolist() == [1, 0, 1, 2]
    assert run_lanes(sim, "cvt.rmi.ftz.f32.f32 %d, %a0;", ["f32"], "f32", [tiny]).view(np.uint32).tolist() == \
        [0, 1 << 31, 0, 0x3F800000]
    # A result is flushed when it is tiny after rounding: below 2^-126 (1 - 2^-25).
    small = np.array([0x380FFFFFEFFFFFFF, 0x380FFFFFF0000000, 0xB800000000000000], dtype=np.uint64).view(np.float64)
    assert run_lanes(sim, "cvt.rn.ftz.f32.f64 %d, %a0;", ["f64"], "f32", [small]).view(np.uint32).tolist() == \
        [0, 0x00800000, 1 << 31]
    assert run_lanes(sim, "cvt.rn.f32.f64 %d, %a0;", ["f64"], "f32", [small]).view(np.uint32).tolist() == \
        [0x00800000, 0x00800000, 0x80400000]
    # f16 subnormals are not f32 subnormals.
    half = np.array([1, 0x83FF], dtype=np.uint16)
    assert run_lanes(sim, "cvt.ftz.f32.f16 %d, %a0;", ["b16"], "f32", [half]).view(np.uint32).tolist() == \
        [0x33800000, 0xB87FC000]


@pytest.mark.parametrize("instr, types, why", [
    ("cvt.rn.ftz.f16.f64", ("f64", "b16"), r"\.ftz needs"),  # ptxas refuses these three as well
    ("cvt.ftz.f64.f16", ("b16", "f64"), r"\.ftz needs"),
    ("cvt.rzi.ftz.s32.f64", ("f64", "s32"), r"\.ftz needs"),
    ("cvt.rn.ftz.f32.s32", ("s32", "f32"), r"\.ftz needs"),  # not measured
    ("cvt.rmi.f64.f32", ("f32", "f64"), "same floating type"),
    ("cvt.rz.f32.f64", ("f64", "f32"), r"modifier \.rz"),
])
def test_cvt_forms_that_are_refused(sim, instr, types, why):
    source, result = types
    with pytest.raises(warplens.UnsupportedError, match=why):
        run_lanes(sim, f"{instr} %d, %a0;", [source], result, [np.zeros(1, dtype=np.dtype(f"u{int(source[1:]) // 8}"))])


@pytest.mark.parametrize("cmp, expect", [
    ("lt", [1, 0, 0, 0]), ("ge", [0, 1, 1, 0]), ("eq", [0, 1, 0, 0]),
    ("ne", [1, 0, 1, 0]), ("neu", [1, 0, 1, 1]), ("ltu", [1, 0, 0, 1]),
    ("num", [1, 1, 1, 0]), ("nan", [0, 0, 0, 1]),
])
def test_setp_float_ordered_and_unordered(sim, cmp, expect):
    a, b = f32(1, 2, 3, np.nan), f32(2, 2, 1, 1)
    assert _setp(sim, cmp, "f32", a, b).tolist() == expect


def _setp(sim, cmp, ty, a, b):
    bits = ty[1:]
    mod = kernel(f"""
        .reg .pred %p; .reg .b{bits} %a, %b; .reg .b32 %t, %d; .reg .b64 %base, %off, %addr;
        mov.u32 %t, %tid.x;
        mul.wide.u32 %off, %t, {int(bits) // 8};
        ld.param.u64 %base, [pa]; add.u64 %addr, %base, %off; ld.global.b{bits} %a, [%addr];
        ld.param.u64 %base, [pb]; add.u64 %addr, %base, %off; ld.global.b{bits} %b, [%addr];
        setp.{cmp}.{ty} %p, %a, %b;
        selp.u32 %d, 1, 0, %p;
        mul.wide.u32 %off, %t, 4;
        ld.param.u64 %base, [po]; add.u64 %addr, %base, %off; st.global.b32 [%addr], %d;
        ret;
    """, ".param .u64 pa, .param .u64 pb, .param .u64 po")
    out = sim.empty(len(a), "u32")
    sim.launch(mod, "k", grid=1, block=len(a), args=[sim.from_numpy(a), sim.from_numpy(b), out])
    return out.numpy()


def test_setp_signed_vs_unsigned(sim):
    a, b = i32(-1, 1), i32(1, -1)
    assert _setp(sim, "lt", "s32", a, b).tolist() == [1, 0]
    assert _setp(sim, "lt", "u32", a, b).tolist() == [0, 1]
    assert _setp(sim, "lo", "s32", a, b).tolist() == [0, 1]
    assert _setp(sim, "hs", "s32", a, b).tolist() == [1, 0]


def test_unsupported_modifier_is_an_error(sim):
    with pytest.raises(warplens.UnsupportedError, match=r"`add\.sat\.s32`.*modifier \.sat"):
        run_lanes(sim, "add.sat.s32 %d, %a0, %a0;", ["s32"], "s32", [i32(1)])
    with pytest.raises(warplens.UnsupportedError, match="float-to-int needs"):
        run_lanes(sim, "cvt.s32.f32 %d, %a0;", ["f32"], "s32", [f32(1)])


def test_bfi_and_shf_as_nvcc_12_emits_them(sim):
    """`(u64)hi << 32 | lo` and a 32-bit rotation; tests/test_hardware_golden.py has the GPU's
    results for the edge cases."""
    u32 = lambda *v: np.array(v, np.uint32)
    hi, lo = np.array([0x11223344, 0xFFFFFFFF], np.uint64), np.array([0xAABBCCDDEEFF0011, 5], np.uint64)
    out = run_lanes(sim, "bfi.b64 %d, %a0, %a1, 32, 32;", ["b64", "b64"], "b64", [hi, lo])
    assert out.tolist() == [0x11223344EEFF0011, 0xFFFFFFFF00000005]
    out = run_lanes(sim, "bfi.b32 %d, %a0, %a1, 8, 4;", ["b32", "b32"], "b32", [u32(0xFFFFFFFF, 0x5), u32(0, 0xFFFFFFFF)])
    assert out.tolist() == [0x00000F00, 0xFFFFF5FF]
    x, n = u32(0x80000001, 0x12345678, 0xDEADBEEF), u32(1, 36, 0)
    out = run_lanes(sim, "shf.l.wrap.b32 %d, %a0, %a0, %a1;", ["b32", "u32"], "b32", [x, n])
    assert out.tolist() == [0x00000003, 0x23456781, 0xDEADBEEF]
    out = run_lanes(sim, "shf.r.clamp.b32 %d, %a0, %a1, %a2;", ["b32", "b32", "u32"], "b32",
                    [u32(0x000000F0, 7), u32(0x0000000A, 9), u32(4, 99)])  # b:a >> n, n at most 32
    assert out.tolist() == [0xA000000F, 9]


@pytest.mark.parametrize("instruction", ["bfi.u32 %d, %a0, %a0, 0, 1;", "bfi.b16 %d, %a0, %a0, 0, 1;",
                                         "shf.l.b32 %d, %a0, %a0, 1;", "shf.l.wrap.b64 %d, %a0, %a0, 1;",
                                         "shf.wrap.b32 %d, %a0, %a0, 1;"])
def test_other_bfi_and_shf_forms_are_refused(sim, instruction):
    with pytest.raises(warplens.UnsupportedError, match="unsupported instruction"):
        run_lanes(sim, instruction, ["b32"], "b32", [np.zeros(2, np.uint32)])


def test_unimplemented_opcode_is_rejected_before_running(sim):
    mod = kernel(".reg .b32 %r;\npopc.b32 %r, %r;\nret;")
    with pytest.raises(warplens.UnsupportedError, match=r"does not implement yet: popc\n.*test.ptx:7"):
        sim.launch(mod, "k", grid=1, block=1)
