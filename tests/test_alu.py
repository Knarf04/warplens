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


def test_float_arithmetic_on_bit_registers(sim):
    # nvcc keeps floats in .b32 registers and lets the instruction type interpret them.
    a, b = f32(1.5, -0.0, np.inf), f32(2.25, 0.0, -np.inf)
    out = run_lanes(sim, "add.f32 %d, %a0, %a1;", ["f32", "f32"], "f32", [a, b])
    assert out[0] == 3.75 and out[1] == 0.0 and np.isnan(out[2])
    assert run_lanes(sim, "max.f32 %d, %a0, %a1;", ["f32", "f32"], "f32",
                     [f32(np.nan, 1.0), f32(2.0, np.nan)]).tolist() == [2.0, 1.0]


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


def test_cvt_integer_widths(sim):
    a = i32(-1, 0x1234ABCD, 300)
    assert run_lanes(sim, "cvt.s64.s32 %d, %a0;", ["s32"], "s64", [a]).tolist() == [-1, 0x1234ABCD, 300]
    assert run_lanes(sim, "cvt.u64.u32 %d, %a0;", ["u32"], "u64", [a.view(np.uint32)]).tolist() == \
        [0xFFFFFFFF, 0x1234ABCD, 300]
    # Narrow result into a wider register: sign- or zero-extended by the result type.
    assert run_lanes(sim, "cvt.s8.s32 %d, %a0;", ["s32"], "s32", [a]).tolist() == [-1, -51, 44]
    assert run_lanes(sim, "cvt.u8.u32 %d, %a0;", ["u32"], "u32", [a.view(np.uint32)]).tolist() == [255, 0xCD, 44]
    assert run_lanes(sim, "cvt.rn.f32.s32 %d, %a0;", ["s32"], "f32", [a]).tolist() == [-1.0, float(np.float32(0x1234ABCD)), 300.0]


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


def test_unimplemented_opcode_is_rejected_before_running(sim):
    mod = kernel(".reg .b32 %r;\npopc.b32 %r, %r;\nret;")
    with pytest.raises(warplens.UnsupportedError, match=r"does not implement yet: popc\n.*test.ptx:7"):
        sim.launch(mod, "k", grid=1, block=1)
