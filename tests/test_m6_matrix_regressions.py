"""M6 matrix-load address spaces, bank accounting, and MMA layout validation."""

from pathlib import Path
import re

import numpy as np
import pytest

import warplens
from conftest import kernel
from test_tcgen05 import mma_kernel, place

AMPERE = Path(__file__).resolve().parents[1] / "validation" / "ampere"


@pytest.mark.parametrize("space", ["shared", "shared::cta", "generic"])
@pytest.mark.parametrize("mode", ["functional", "timed"])
def test_ldmatrix_cluster_local_addresses_match_hardware(sim, space, mode):
    # Use all six hardware-validated forms, keeping each CTA's output separate.
    text = (AMPERE / "ldmatrix_probe.ptx").read_text()
    text = text.replace(".reg .b64 \t%rd<5>;", ".reg .b64 %rd<5>, %ctaoff, %generic;\n.reg .b32 %rank;")
    text = text.replace("ld.param.b64 \t%rd1, [ldmatrix_probe_param_0];", """
        ld.param.b64 %rd1, [ldmatrix_probe_param_0];
        mov.u32 %rank, %ctaid.x;
        mul.wide.u32 %ctaoff, %rank, 3072;
        add.u64 %rd1, %rd1, %ctaoff;
    """)
    if space == "generic":
        first_load = text.index("ldmatrix.sync")
        text = text[:first_load] + "cvt.u64.u32 %generic, %r6;\ncvta.shared.u64 %generic, %generic;\n" + text[first_load:]
    def address_form(match):
        ins = match.group()
        if space == "generic":
            return ins.replace(".shared.b16", ".b16").replace("[%r6]", "[%generic]")
        return ins.replace(".shared.b16", f".{space}.b16")
    text = re.sub(r"ldmatrix\.sync[^;]+;", address_form, text)
    out = sim.empty((2, 6, 32, 4), "u32")
    stats = sim.launch(warplens.parse_ptx(text), "ldmatrix_probe", grid=2, cluster=2, block=32,
                       args=[out], timing=mode == "timed")
    rows = [line.split() for line in (AMPERE / "ldmatrix_golden.txt").read_text().splitlines()]
    expected = np.array([[int(x, 16) for x in row[3:]] for row in rows if row[0] == "L"], np.uint32)
    np.testing.assert_array_equal(out.numpy(), np.tile(expected.reshape(1, 6, 32, 4), (2, 1, 1, 1)))
    assert not stats.findings


def test_ldmatrix_checks_natural_row_alignment(sim):
    mod = kernel("""
        .reg .b32 %row, %r; .reg .pred %first;
        .shared .align 16 .b8 data[144];
        mov.u32 %row, %laneid;
        setp.eq.u32 %first, %row, 0;
        mad.lo.u32 %row, %row, 16, data;
        @%first add.u32 %row, %row, 2;
        ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r}, [%row];
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="16-byte aligned"):
        sim.launch(mod, "k", grid=1, block=32)


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("stride,waves", [(16, 1), (128, 8)])
def test_ldmatrix_bank_report_counts_matrices_as_one_instruction(sim, count, stride, waves):
    regs = ", ".join(f"%r{i}" for i in range(count))
    mod = kernel(f"""
        .reg .b32 %t, %i, %addr, %row, %r<{count}>; .reg .pred %more;
        .shared .align 16 .b8 data[1024];
        mov.u32 %t, %laneid;
        mov.u32 %i, %t;
      FILL:
        mad.lo.u32 %addr, %i, 4, data;
        st.shared.u32 [%addr], %i;
        add.u32 %i, %i, 32;
        setp.lt.u32 %more, %i, 256;
        @%more bra FILL;
        bar.sync 0;
        and.b32 %row, %t, 7;
        mad.lo.u32 %row, %row, {stride}, data;
        ldmatrix.sync.aligned.m8n8.x{count}.shared.b16 {{{regs}}}, [%row];
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=32)
    (entry,) = [entry for entry in stats.smem.instructions.values() if entry.op == "ldmatrix"]
    assert entry.executions == 1
    # Each matrix transfers 128 bytes, even when its addresses repeat another matrix's.
    assert entry.ideal == count
    assert entry.wavefronts == count * waves
    assert entry.conflicts == count * (waves - 1)
    assert entry.worst_way == waves
    assert not stats.findings


@pytest.mark.parametrize("transposed_operand", ["A", "B"])
@pytest.mark.parametrize("layout", ["none", "128B"])
def test_tf32_mn_major_requires_unimplemented_32_byte_atomicity(sim, transposed_operand, layout):
    image = np.zeros(32768, np.uint8)
    da = place(np.ones((128, 8), np.float32), layout, 0, image)
    db = place(np.ones((16, 8), np.float32), layout, 16384, image)
    idesc = (1 << 4) | (2 << 7) | (2 << 10) | (2 << 17) | (8 << 24)
    idesc |= 1 << (15 if transposed_operand == "A" else 16)
    with pytest.raises(warplens.UnsupportedError, match="32-byte.*atomicity"):
        sim.launch(mma_kernel("tf32"), "k", grid=1, block=128,
                   args=[sim.from_numpy(image), sim.empty((128, 16), "f32"), da, db, idesc,
                         len(image), 32, 16, 0, 1, 0, 1, 0, 0])


def place_half_mn_major(matrix, layout, start, image):
    # Canonical MN-major atoms from PTX 9.7.18.3: store each eight-column
    # submatrix column-first, in row tiles of 8 / 16 / 32 / 64 elements.
    code, bits, row_bytes = {
        "none": (0, 0, 16), "32B": (6, 1, 32),
        "64B": (4, 2, 64), "128B": (2, 3, 128),
    }[layout]
    rows = len(matrix)
    row_tile = row_bytes // 2
    physical = matrix.reshape(rows // row_tile, row_tile, 2, 8).transpose(2, 0, 3, 1).copy()
    raw = physical.view(np.uint8).ravel()
    addresses = start + np.arange(len(raw))
    image[addresses ^ (((addresses >> 7) & ((1 << bits) - 1)) << 4)] = raw
    lbo, sbo = (8 * row_bytes, rows * 16) if bits else (rows * 16, 128)
    return (start >> 4) | ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (code << 61)


@pytest.mark.parametrize("layout", ["none", "32B", "64B", "128B"])
@pytest.mark.parametrize("transposed_operands", ["A", "B", "AB"])
def test_half_mn_major_operand_layouts(sim, layout, transposed_operands):
    rng = np.random.default_rng(104)
    a = rng.integers(-4, 5, (128, 16)).astype(np.float16)
    b = rng.integers(-4, 5, (64, 16)).astype(np.float16)
    image = np.zeros(32768, np.uint8)
    descriptors = [(place_half_mn_major if name in transposed_operands else place)(
        matrix, layout, start, image) for name, matrix, start in [("A", a, 0), ("B", b, 16384)]]
    idesc = (1 << 4) | (8 << 17) | (8 << 24)
    idesc |= (int("A" in transposed_operands) << 15) | (int("B" in transposed_operands) << 16)
    out = sim.empty((128, 64), "f32")
    stats = sim.launch(mma_kernel(), "k", grid=1, block=128,
                       args=[sim.from_numpy(image), out, *descriptors, idesc,
                             len(image), 64, 64, 0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(out.numpy(), a.astype(np.float32) @ b.astype(np.float32).T)
    assert not stats.findings


def test_ldmatrix_generic_address_must_be_shared(sim):
    mod = kernel("""
        .reg .b64 %ptr; .reg .b32 %r;
        ld.param.u64 %ptr, [ptr];
        ldmatrix.sync.aligned.m8n8.x1.b16 {%r}, [%ptr];
        ret;
    """, ".param .u64 ptr")
    with pytest.raises(warplens.ExecutionError, match="generic addresses must point into shared memory"):
        sim.launch(mod, "k", grid=1, block=32, args=[sim.empty(128, "u8")])
