"""TMEM row aliases documented by NVIDIA's CUTLASS tcgen05_ld primitive."""

import numpy as np
import pytest

import warplens

from tests.support import kernel


SHAPES = [("32x32b", 128, 128), ("16x64b", 128, 128),
          ("16x128b", 64, 128), ("16x256b", 32, 128)]


def transfer_kernel(shape, count, registers, half, alias_store):
    regs = ", ".join(f"%r{i}" for i in range(registers))
    initialize = "\n".join(f"add.u32 %r{i}, %value, {i};" for i in range(registers))
    output = "\n".join(f"st.global.u32 [%out+{4*i}], %r{i};" for i in range(registers))
    store_addr, load_addr = ("%alias", "%own") if alias_store else ("%own", "%alias")
    return kernel(f"""
        .reg .b32 %t, %w, %ptr, %base, %own, %alias, %value, %r<{registers}>;
        .reg .b64 %out, %off;
        .reg .pred %first;
        .shared .align 4 .u32 tptr;
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid;
        mov.u32 %ptr, tptr;
        setp.eq.u32 %first, %w, 0;
        @%first tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%ptr], 512;
        bar.sync 0;
        ld.shared.u32 %base, [%ptr];
        shl.b32 %own, %w, 21;
        add.u32 %own, %own, {half << 16};
        add.u32 %own, %own, %base;
        ld.param.u32 %alias, [address];
        add.u32 %alias, %alias, {half << 16};
        add.u32 %alias, %alias, %base;
        mul.lo.u32 %value, %t, 1000;
        {initialize}
        tcgen05.st.sync.aligned.{shape}.x{count}.b32 [{store_addr}], {{{regs}}};
        tcgen05.wait::st.sync.aligned;
        tcgen05.ld.sync.aligned.{shape}.x{count}.b32 {{{regs}}}, [{load_addr}];
        tcgen05.wait::ld.sync.aligned;
        ld.param.u64 %out, [output];
        mul.wide.u32 %off, %t, {4*registers};
        add.u64 %out, %out, %off;
        {output}
        bar.sync 0;
        @%first tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 512;
        ret;
    """, ".param .u64 output, .param .u32 address")


@pytest.mark.parametrize("shape,count,registers,half", [
    (shape, count, regs, half) for shape, count, regs in SHAPES
    for half in ([0] if shape == "32x32b" else [0, 16])
])
@pytest.mark.parametrize("named_row", [32, 96, 0xFFE0])
@pytest.mark.parametrize("alias_store", [False, True])
def test_row_aliases_do_not_name_another_warps_window(sim, shape, count, registers,
                                                      half, named_row, alias_store):
    # The same encoded row in all warps aliases their separate physical windows.
    # Exercise both transfers, both 16-row halves and maximum legal repeat counts.
    out = sim.empty((128, registers), "u32")
    stats = sim.launch(transfer_kernel(shape, count, registers, half, alias_store), "k",
                       grid=1, block=128, args=[out, named_row << 16])
    expected = np.arange(128, dtype=np.uint32)[:, None] * 1000 + np.arange(registers)
    np.testing.assert_array_equal(out.numpy(), expected)
    assert not stats.findings


@pytest.mark.parametrize("shape,count,registers", SHAPES)
@pytest.mark.parametrize("alias_store", [False, True])
def test_row_aliases_do_not_wrap_out_of_range_columns(sim, shape, count, registers, alias_store):
    out = sim.empty((128, registers), "u32")
    with pytest.raises(warplens.ExecutionError, match="column.*outside tensor memory"):
        sim.launch(transfer_kernel(shape, count, registers, 0, alias_store), "k",
                   grid=1, block=128, args=[out, (32 << 16) | 512])


def invalid_transfer(shape, address, store):
    registers = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}[shape]
    regs = "{" + ", ".join(f"%r{i}" for i in range(registers)) + "}"
    operands = f"[%address], {regs}" if store else f"{regs}, [%address]"
    return kernel(f"""
        .reg .b64 %address;
        .reg .b32 %r<{registers}>;
        mov.u64 %address, {address};
        tcgen05.{'st' if store else 'ld'}.sync.aligned.{shape}.x1.b32 {operands};
        ret;
    """)


@pytest.mark.parametrize("shape", [s[0] for s in SHAPES])
@pytest.mark.parametrize("store", [False, True])
def test_tmem_transfer_rejects_an_address_wider_than_32_bits(sim, shape, store):
    # This high bit used to disappear into the modulo row calculation.
    with pytest.raises(warplens.ExecutionError, match="32-bit tensor-memory address"):
        sim.launch(invalid_transfer(shape, 1 << 32, store), "k", grid=1, block=32)


@pytest.mark.parametrize("shape", [s[0] for s in SHAPES])
@pytest.mark.parametrize("store", [False, True])
def test_transfer_crossing_local_row_boundary_is_explicitly_unsupported(sim, shape, store):
    local_row = 1 if shape == "32x32b" else 17
    with pytest.raises(warplens.UnsupportedError, match="spanning the local 32-row"):
        sim.launch(invalid_transfer(shape, (32 + local_row) << 16, store),
                   "k", grid=1, block=32)
