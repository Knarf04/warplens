"""The `cvt` forms of cuda_fp8.h and cuda_fp4.h: FP8 (E4M3, E5M2), FP4 (E2M1) and E8M0.

Two kinds of test. The first run hand-written PTX, spelled as nvcc emits it for sm_100a, and
need no toolchain. The last compiles the CUDA headers twice, for the device with nvcc (where
they emit these instructions) and for the host with g++ (where they compute the same
functions in software), and requires the simulator to agree with the host on every input.
"""

import shutil
import struct
import subprocess

import numpy as np
import pytest

import warplens
from warplens.numerics import E2M1, E4M3, E5M2
from warplens.units import small_formats

from conftest import cuda_include, kernel
from conftest import run_lanes as run_block

NAN32, INF32 = 0x7FC00000, 0x7F800000


def as_f64(bits32):
    """Float32 bit patterns as float64 values."""
    with np.errstate(invalid="ignore"):  # signalling NaNs
        return np.asarray(bits32, dtype=np.uint32).view(np.float32).astype(np.float64)


def f32_bits(*values):
    return np.array([struct.unpack("<I", struct.pack("<f", v))[0] for v in values], dtype=np.uint32)


def sweep_f32():
    """Float32 patterns around every value a format with up to 4 fraction bits can hold between
    2^-18 and 2^18 (every code, every midpoint between codes, and the neighbours of each), as
    validation/blackwell/formats/small_formats.cu uses, plus special values."""
    values = []
    for sign in range(2):
        for e in range(-18, 18):
            for j in range(32):
                base = (sign << 31) | ((e + 127) << 23) | (j << 18)
                values += [base - 1, base, base + 1]
    values += [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,
               INF32, 0xFF800000, NAN32, 0xFFC00000, 0x7F800001, 0xFFA12345]
    return np.array(values, dtype=np.uint32)


def sweep_16(exponent_bits: int, fraction_bits: int, exponents=None):
    """16-bit float patterns: both signs, the given exponent fields, the top five fraction
    bits taking every value, each with the lowest fraction bits all clear, one set, all set."""
    low_bits = fraction_bits - 5
    exponents = range(1 << exponent_bits) if exponents is None else exponents
    values = [(sign << 15) | (e << fraction_bits) | (j << low_bits) | low
              for sign in range(2) for e in exponents for j in range(32)
              for low in (0, 1, (1 << low_bits) - 1)]
    return np.array(values, dtype=np.uint32)


def run_lanes(sim, instr, in_types, out_type, inputs):
    """conftest.run_lanes for any number of inputs: one block of up to 1024 threads at a time."""
    n = len(inputs[0])
    return np.concatenate([run_block(sim, instr, in_types, out_type, [x[i:i + 1024] for x in inputs])
                           for i in range(0, n, 1024)])


def pair(low: np.ndarray):
    """A second input for the high half: the same values in another order."""
    return np.roll(low[::-1], 7)


# -- hand-written PTX --------------------------------------------------------------------------

@pytest.mark.parametrize("name, fmt", [("e4m3x2", E4M3), ("e5m2x2", E5M2)])
def test_fp8_encoding_from_two_floats_matches_the_format_tables(sim, name, fmt):
    low = sweep_f32()
    high = pair(low)
    out = run_lanes(sim, f"{{cvt.rn.satfinite.{name}.f32 %d, %a1, %a0;}}", ["b32", "b32"], "b16", [low, high])

    def codes(bits):
        value = as_f64(bits)
        nan = np.isnan(value)
        return np.where(nan, 0x7F, fmt.encode(np.where(nan, 0, value)))  # a NaN loses its sign

    assert (out & 0xFF).tolist() == codes(low).tolist()  # the last operand is the low byte
    assert (out >> 8).tolist() == codes(high).tolist()


def test_fp8_encoding_of_chosen_values(sim):
    inputs = f32_bits(1.0, -1.0, 448.0, 464.0, 1e10, -1e10, 0.0, -0.0, 2.0 ** -9, 2.0 ** -10, 0.07421875)
    inputs = np.concatenate([inputs, np.array([INF32, 0xFF800000, NAN32, 0xFFC00000], dtype=np.uint32)])
    zero = np.zeros_like(inputs)
    e4m3 = run_lanes(sim, "cvt.rn.satfinite.e4m3x2.f32 %d, %a1, %a0;", ["b32", "b32"], "b16", [inputs, zero])
    # 448 is the largest value; 2^-9 is the smallest; 2^-10 is a tie that goes to zero (even);
    # 0.07421875 = 1.1875 * 2^-4 is a tie between 0x19 and 0x1a.
    assert [f"{v:02x}" for v in e4m3] == ["38", "b8", "7e", "7e", "7e", "fe", "00", "80", "01", "00", "1a",
                                          "7e", "fe", "7f", "7f"]
    e5m2 = run_lanes(sim, "cvt.rn.satfinite.e5m2x2.f32 %d, %a1, %a0;", ["b32", "b32"], "b16", [inputs, zero])
    assert [f"{v:02x}" for v in e5m2[:8]] == ["3c", "bc", "5f", "5f", "7b", "fb", "00", "80"]
    assert [f"{v:02x}" for v in e5m2[11:]] == ["7b", "fb", "7f", "7f"]  # infinity saturates


@pytest.mark.parametrize("source", ["f16x2", "bf16x2"])
@pytest.mark.parametrize("name, fmt", [("e4m3x2", E4M3), ("e5m2x2", E5M2), ("e2m1x2", E2M1)])
def test_encoding_from_packed_halves(sim, source, name, fmt):
    if source == "f16x2":
        low = sweep_16(5, 10)
        decode = lambda bits: bits.astype(np.uint16).view(np.float16).astype(np.float64)
    else:
        low = sweep_16(8, 7, exponents=[0, 1, *range(100, 150), 254, 255])
        decode = lambda bits: as_f64(bits << np.uint32(16))
    high = pair(low)
    packed = low | (high << np.uint32(16))
    if fmt is E2M1:  # as cuda_fp4.h writes it: the result is 8 bits wide
        instr = (f"{{ .reg .b8 __$temp1;\n cvt.rn.satfinite.{name}.{source} __$temp1, %a0;\n"
                 " mov.b16 %d, {__$temp1, 0}; }")
        shift, nan_code = 4, 0x7
    else:
        instr = f"{{cvt.rn.satfinite.{name}.{source} %d, %a0;}}"
        shift, nan_code = 8, 0x7F
    out = run_lanes(sim, instr, ["b32"], "b16", [packed])

    def codes(bits):
        value = decode(bits)
        nan = np.isnan(value)
        return np.where(nan, nan_code, fmt.encode(np.where(nan, 0, value)))

    assert (out & ((1 << shift) - 1)).tolist() == codes(low).tolist()
    assert (out >> shift).tolist() == codes(high).tolist()


def test_fp4_encoding_from_two_floats(sim):
    low = sweep_f32()
    high = pair(low)
    instr = ("{ .reg .b8 __$temp1;\n cvt.rn.satfinite.e2m1x2.f32 __$temp1, %a1, %a0;\n"
             " mov.b16 %d, {__$temp1, 0}; }")
    out = run_lanes(sim, instr, ["b32", "b32"], "b16", [low, high])

    def codes(bits):
        value = as_f64(bits)
        nan = np.isnan(value)
        return np.where(nan, 0x7, E2M1.encode(np.where(nan, 0, value)))  # NaN becomes +6

    assert (out & 0xF).tolist() == codes(low).tolist()
    assert (out >> 4).tolist() == codes(high).tolist()
    chosen = f32_bits(0.25, 0.26, 0.75, 1.25, 2.5, 5.0, 6.0, 100.0, -100.0, -0.0)
    out = run_lanes(sim, instr, ["b32", "b32"], "b16", [chosen, np.zeros_like(chosen)])
    # Ties go to the even code: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 2.5 -> 2, 5 -> 4.
    assert out.tolist() == [0x0, 0x1, 0x2, 0x2, 0x4, 0x6, 0x7, 0x7, 0xF, 0x8]


@pytest.mark.parametrize("target", ["f16x2", "bf16x2"])
@pytest.mark.parametrize("name, fmt", [("e4m3x2", E4M3), ("e5m2x2", E5M2), ("e2m1x2", E2M1)])
def test_decoding_every_code(sim, target, name, fmt):
    low = np.arange(fmt.codes, dtype=np.uint16)
    high = pair(low)
    if fmt is E2M1:  # as cuda_fp4.h writes it: the source is an 8-bit register
        instr = ("{ .reg .b8 __$temp1, __$tempz;\n mov.b16 {__$temp1, __$tempz}, %a0;\n"
                 f" cvt.rn.{target}.{name} %d, __$temp1; }}")
        packed = low | (high << np.uint16(4))
    else:
        instr = f"{{cvt.rn.{target}.{name} %d, %a0;}}"
        packed = low | (high << np.uint16(8))
    out = run_lanes(sim, instr, ["b16"], "b32", [packed])
    for half, codes in ((out & 0xFFFF, low), (out >> 16, high)):
        value = fmt.decode(codes)
        if target == "f16x2":
            got = half.astype(np.uint16).view(np.float16).astype(np.float64)
        else:
            got = as_f64(half.astype(np.uint32) << np.uint32(16))
        np.testing.assert_array_equal(got, value)  # exact, and NaN where the code is a NaN
        assert (half[np.isnan(value)] == 0x7FFF).all()
        assert (np.signbit(got) == np.signbit(value))[~np.isnan(value)].all()


def test_e5m2_is_the_top_byte_of_a_half(sim):
    codes = np.arange(256, dtype=np.uint16)
    out = run_lanes(sim, "cvt.rn.f16x2.e5m2x2 %d, %a0;", ["b16"], "b32", [codes])
    nan = np.isnan(E5M2.decode(codes))
    assert (out[~nan] == codes[~nan].astype(np.uint32) << 8).all() and (out >> 16 == 0).all()


def test_e8m0_encoding(sim):
    def run(mode, bits):
        bits = np.asarray(bits, dtype=np.uint32)
        out = run_lanes(sim, f"cvt.{mode}.ue8m0x2.f32 %d, %a1, %a0;", ["b32", "b32"], "b16",
                        [bits, np.zeros_like(bits)])
        assert (out >> 8 == 0).all()  # 0.0 in the high half
        return [int(v) for v in out]

    one, three, big = 0x3F800000, 0x40400000, 0x7F000001  # 1, 3, just above 2^127
    below, least, above = 0x00200000, 0x00400000, 0x00400001  # 2^-128, 2^-127, just above it
    values = [one, three, one | 1 << 31, three | 1 << 31, 0, below, least, above, big, 0x7F000000,
              INF32, NAN32, 0xFFC00000]
    # Round toward zero: the exponent field. The sign is ignored; a NaN is 0xff.
    assert run("rz", values) == [127, 128, 127, 128, 0, 0, 0, 0, 254, 254, 255, 255, 255]
    assert run("rz.satfinite", values) == [127, 128, 127, 128, 0, 0, 0, 0, 254, 254, 254, 255, 255]
    # Round up: the next power of two, unless the value is one or is at most 2^-127.
    assert run("rp", values) == [127, 129, 127, 129, 0, 0, 0, 1, 255, 254, 255, 255, 255]
    assert run("rp.satfinite", values) == [127, 129, 127, 129, 0, 0, 0, 1, 254, 254, 254, 255, 255]


def test_e8m0_from_and_to_bfloat16(sim):
    # bfloat16 is the top half of a float32: 1.0 = 0x3f80, 3.0 = 0x4040.
    packed = np.array([0x3F80 | 0x4040 << 16, 0x0040 | 0x0041 << 16, 0x7F80 | 0x7FC0 << 16], dtype=np.uint32)
    rz = run_lanes(sim, "{cvt.rz.satfinite.ue8m0x2.bf16x2 %d, %a0;}", ["b32"], "b16", [packed])
    rp = run_lanes(sim, "{cvt.rp.ue8m0x2.bf16x2 %d, %a0;}", ["b32"], "b16", [packed])
    assert [f"{v:04x}" for v in rz] == ["807f", "0000", "fffe"]
    assert [f"{v:04x}" for v in rp] == ["817f", "0100", "ffff"]

    codes = np.arange(256, dtype=np.uint16)
    out = run_lanes(sim, "{cvt.rn.bf16x2.ue8m0x2 %d, %a0;}", ["b16"], "b32", [codes | (pair(codes) << np.uint16(8))])
    expected = (codes.astype(np.uint32) << 7)
    expected[0], expected[255] = 0x0040, 0x7FFF  # 2^-127 is a bfloat16 subnormal; 0xff is NaN
    assert (out & 0xFFFF).tolist() == expected.tolist()
    assert (out >> 16).tolist() == expected[pair(codes)].tolist()
    value = as_f64(expected[:255] << np.uint32(16))
    np.testing.assert_array_equal(value, 2.0 ** (np.arange(255) - 127.0))


@pytest.mark.parametrize("instr, why", [
    ("cvt.rn.e4m3x2.f32 %d, %a0, %a0;", "small_formats.py"),  # no .satfinite: the headers never emit it
    ("cvt.rz.satfinite.e4m3x2.f32 %d, %a0, %a0;", "small_formats.py"),
    ("cvt.rn.satfinite.ue8m0x2.f32 %d, %a0, %a0;", "small_formats.py"),  # nearest is done in software
    ("cvt.rn.satfinite.e4m3x2.f32 %d, %a0;", "two .f32 sources"),
    ("cvt.rn.satfinite.e4m3x2.f64 %d, %a0, %a0;", "small_formats.py"),
    ("cvt.rz.f16x2.e4m3x2 %d, %a0;", "small_formats.py"),
    ("cvt.rn.f16x2.ue8m0x2 %d, %a0;", "small_formats.py"),
])
def test_forms_the_headers_do_not_emit_are_rejected(sim, instr, why):
    with pytest.raises(warplens.UnsupportedError, match=why):
        run_lanes(sim, instr, ["b32"], "b32", [np.zeros(4, dtype=np.uint32)])


def test_conversion_respects_the_predicate(sim):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 %t, %x; .reg .b16 %d; .reg .pred %p;
        ld.param.u64 %out, [out];
        mov.u32 %t, %tid.x;
        mov.b16 %d, 0x1234;
        mov.b32 %x, 0f3F800000;
        setp.lt.u32 %p, %t, 2;
        @%p cvt.rn.satfinite.e4m3x2.f32 %d, %x, %x;
        st.global.b16 [%out], %d;
        ret;
    """, ".param .u64 out")
    for block, expected in ((1, 0x3838), (4, 0x1234)):  # the last thread to store wins
        out = sim.empty(1, "u16")
        sim.launch(mod, "k", grid=1, block=block, args=[out])
        assert out.numpy()[0] == expected


# -- the CUDA headers, compiled for the device and for the host -----------------------------------

# One source, two builds. `CASES` lists (name, expression in a, b, f2, h2, b2): for nvcc it
# becomes a kernel that switches on the case; for g++ a program that reads inputs from stdin.
HEADER_SOURCE = r"""
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif
HD static inline unsigned pack(__half2_raw v) { return v.x | ((unsigned)v.y << 16); }
HD static inline unsigned pack(__nv_bfloat162_raw v) { return v.x | ((unsigned)v.y << 16); }
#define CASES(C) \
  C(0, __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3)) \
  C(1, __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E5M2)) \
  C(2, __nv_cvt_halfraw2_to_fp8x2(h2, __NV_SATFINITE, __NV_E4M3)) \
  C(3, __nv_cvt_halfraw2_to_fp8x2(h2, __NV_SATFINITE, __NV_E5M2)) \
  C(4, __nv_cvt_bfloat16raw2_to_fp8x2(b2, __NV_SATFINITE, __NV_E4M3)) \
  C(5, __nv_cvt_bfloat16raw2_to_fp8x2(b2, __NV_SATFINITE, __NV_E5M2)) \
  C(6, pack(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)a, __NV_E4M3))) \
  C(7, pack(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)a, __NV_E5M2))) \
  C(8, pack(__nv_cvt_fp8x2_to_bf162raw((__nv_fp8x2_storage_t)a, __NV_E4M3))) \
  C(9, pack(__nv_cvt_fp8x2_to_bf162raw((__nv_fp8x2_storage_t)a, __NV_E5M2))) \
  C(10, __nv_cvt_float2_to_e8m0x2(f2, __NV_SATFINITE, cudaRoundZero)) \
  C(11, __nv_cvt_float2_to_e8m0x2(f2, __NV_NOSAT, cudaRoundZero)) \
  C(12, __nv_cvt_float2_to_e8m0x2(f2, __NV_SATFINITE, cudaRoundPosInf)) \
  C(13, __nv_cvt_float2_to_e8m0x2(f2, __NV_NOSAT, cudaRoundPosInf)) \
  C(14, __nv_cvt_bfloat162raw_to_e8m0x2(b2, __NV_SATFINITE, cudaRoundZero)) \
  C(15, __nv_cvt_bfloat162raw_to_e8m0x2(b2, __NV_NOSAT, cudaRoundZero)) \
  C(16, __nv_cvt_bfloat162raw_to_e8m0x2(b2, __NV_SATFINITE, cudaRoundPosInf)) \
  C(17, __nv_cvt_bfloat162raw_to_e8m0x2(b2, __NV_NOSAT, cudaRoundPosInf)) \
  C(18, pack(__nv_cvt_e8m0x2_to_bf162raw((__nv_fp8x2_storage_t)a))) \
  C(19, __nv_cvt_float2_to_fp4x2(f2, __NV_E2M1, cudaRoundNearest)) \
  C(20, __nv_cvt_halfraw2_to_fp4x2(h2, __NV_E2M1, cudaRoundNearest)) \
  C(21, __nv_cvt_bfloat16raw2_to_fp4x2(b2, __NV_E2M1, cudaRoundNearest)) \
  C(22, pack(__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)a, __NV_E2M1))) \
  C(23, pack(__nv_cvt_fp4x2_to_bf162raw((__nv_fp4x2_storage_t)a, __NV_E2M1)))
#define ONE(id, expr) case id: r = (expr); break;
HD static inline unsigned convert_one(int which, unsigned a, unsigned b) {
    float2 f2;
    memcpy(&f2.x, &a, 4);
    memcpy(&f2.y, &b, 4);
    __half2_raw h2; h2.x = (unsigned short)a; h2.y = (unsigned short)(a >> 16);
    __nv_bfloat162_raw b2; b2.x = (unsigned short)a; b2.y = (unsigned short)(a >> 16);
    unsigned r = 0;
    switch (which) { CASES(ONE) }
    return r;
}
#ifdef __CUDACC__
extern "C" __global__ void convert(int which, const unsigned* a, const unsigned* b, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = convert_one(which, a[i], b[i]);
}
#else
int main(int argc, char** argv) {
    int which = atoi(argv[1]);
    unsigned a, b;
    while (scanf("%x %x", &a, &b) == 2) printf("%x\n", convert_one(which, a, b));
    return 0;
}
#endif
"""

FLOATS, HALVES, BFLOATS, BYTES, NIBBLES, E8M0_FLOATS, E8M0_BFLOATS = range(7)
HEADER_CASES = {0: FLOATS, 1: FLOATS, 2: HALVES, 3: HALVES, 4: BFLOATS, 5: BFLOATS, 6: BYTES, 7: BYTES,
                8: BYTES, 9: BYTES, 10: E8M0_FLOATS, 11: E8M0_FLOATS, 12: E8M0_FLOATS, 13: E8M0_FLOATS,
                14: E8M0_BFLOATS, 15: E8M0_BFLOATS, 16: E8M0_BFLOATS, 17: E8M0_BFLOATS, 18: BYTES,
                19: FLOATS, 20: HALVES, 21: BFLOATS, 22: NIBBLES, 23: NIBBLES}
# What each case must compile to, so that the comparison is of these instructions and not of
# a software path that both builds share.
HEADER_FORMS = [
    "cvt.rn.satfinite.e4m3x2.f32", "cvt.rn.satfinite.e5m2x2.f32", "cvt.rn.satfinite.e4m3x2.f16x2",
    "cvt.rn.satfinite.e5m2x2.f16x2", "cvt.rn.satfinite.e4m3x2.bf16x2", "cvt.rn.satfinite.e5m2x2.bf16x2",
    "cvt.rn.f16x2.e4m3x2", "cvt.rn.f16x2.e5m2x2", "cvt.rn.bf16x2.e4m3x2", "cvt.rn.bf16x2.e5m2x2",
    "cvt.rz.satfinite.ue8m0x2.f32", "cvt.rz.ue8m0x2.f32", "cvt.rp.satfinite.ue8m0x2.f32", "cvt.rp.ue8m0x2.f32",
    "cvt.rz.satfinite.ue8m0x2.bf16x2", "cvt.rz.ue8m0x2.bf16x2", "cvt.rp.satfinite.ue8m0x2.bf16x2",
    "cvt.rp.ue8m0x2.bf16x2", "cvt.rn.bf16x2.ue8m0x2", "cvt.rn.satfinite.e2m1x2.f32",
    "cvt.rn.satfinite.e2m1x2.f16x2", "cvt.rn.satfinite.e2m1x2.bf16x2", "cvt.rn.f16x2.e2m1x2",
    "cvt.rn.bf16x2.e2m1x2",
]


def header_inputs(kind: int):
    """(a, b) input words for a case: `a` holds a packed pair where the source is packed."""
    if kind == FLOATS:
        low = sweep_f32()
        return low, pair(low)
    if kind == E8M0_FLOATS:
        fractions = [0x000000, 0x000001, 0x3504F3, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF]
        low = np.array([(sign << 31) | (e << 23) | fraction for sign in range(2) for e in range(256)
                        for fraction in fractions], dtype=np.uint32)
        return low, pair(low)
    if kind == BYTES:
        low = np.arange(256, dtype=np.uint32)
        return low | (pair(low) << np.uint32(8)), np.zeros(256, dtype=np.uint32)
    if kind == NIBBLES:
        low = np.arange(16, dtype=np.uint32)
        return low | (pair(low) << np.uint32(4)), np.zeros(16, dtype=np.uint32)
    if kind == HALVES:
        low = sweep_16(5, 10)
    elif kind == BFLOATS:
        low = sweep_16(8, 7, exponents=[0, 1, *range(100, 150), 254, 255])
    else:
        low = sweep_16(8, 7)
    return low | (pair(low) << np.uint32(16)), np.zeros(len(low), dtype=np.uint32)


@pytest.fixture(scope="module")
def header_builds(tmp_path_factory):
    """(parsed PTX module, path of the host program), built from HEADER_SOURCE."""
    include = cuda_include("cuda_fp4.h")
    nvcc, cxx = shutil.which("nvcc"), shutil.which("g++") or shutil.which("c++")
    if include is None or nvcc is None or cxx is None:
        pytest.skip("needs nvcc with cuda_fp8.h and cuda_fp4.h, and a host C++ compiler")
    work = tmp_path_factory.mktemp("small_formats")
    source = work / "convert.cu"
    source.write_text(HEADER_SOURCE)
    device = subprocess.run([nvcc, "-arch=sm_100a", "-ptx", "-O2", "-w", str(source), "-o", str(work / "convert.ptx")],
                            capture_output=True, text=True)
    if device.returncode:
        pytest.skip(f"nvcc cannot build for sm_100a: {device.stderr.strip()[-300:]}")
    subprocess.run([cxx, "-O1", "-w", "-x", "c++", f"-I{include}", str(source), "-o", str(work / "convert")],
                   check=True, capture_output=True, text=True)
    return work / "convert.ptx", work / "convert"


def test_the_device_build_uses_the_conversion_instructions(header_builds):
    text = header_builds[0].read_text()
    for form in HEADER_FORMS:
        assert f"{form} " in text, form
    emitted = {line.split()[0].strip("{") for line in text.splitlines() if "cvt." in line}
    small = {name for name in emitted if small_formats.is_small_format_cvt(tuple(name.split(".")[1:]))}
    assert small == set(HEADER_FORMS)  # nothing the simulator would meet is left out below


@pytest.mark.parametrize("which", sorted(HEADER_CASES))
def test_simulated_instructions_match_the_headers_host_implementation(sim, header_builds, which):
    ptx, host = header_builds
    a, b = header_inputs(HEADER_CASES[which])
    n = len(a)
    out = sim.empty(n, "u32")
    sim.launch(ptx, "convert", grid=-(-n // 256), block=256,
               args=[which, sim.from_numpy(a), sim.from_numpy(b), out, n])
    text = "".join(f"{x:x} {y:x}\n" for x, y in zip(a, b))
    reference = subprocess.run([str(host), str(which)], input=text, capture_output=True, text=True, check=True)
    expected = np.array([int(line, 16) for line in reference.stdout.split()], dtype=np.uint32)
    assert len(expected) == n
    wrong = np.flatnonzero(out.numpy() != expected)
    assert not len(wrong), (f"{HEADER_FORMS[which]}: {len(wrong)} of {n} differ; first a={a[wrong[0]]:08x} "
                            f"b={b[wrong[0]]:08x} simulator={out.numpy()[wrong[0]]:x} host={expected[wrong[0]]:x}")
