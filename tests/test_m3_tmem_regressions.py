"""TMEM allocation and collective transfer contracts from the NVIDIA PTX ISA."""

import numpy as np
import pytest

import warplens
from conftest import kernel
from warplens.units.tcgen05 import transfer_cells


DECLS = """
    .reg .b32 %dst, %base, %n, %tid, %v, %r<256>;
    .reg .b64 %generic, %out, %off;
    .reg .pred %one;
    .shared .align 4 .b8 ptrs[128];
    mov.b32 %dst, ptrs;
    mov.u32 %tid, %tid.x;
    setp.eq.u32 %one, %tid, 0;
"""
ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%dst], 32;"
LOAD_BASE = "ld.shared.u32 %base, [%dst];"
FREE = "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 32;"


def launch(sim, body, *, block=32, params="", args=()):
    return sim.launch(kernel(DECLS + body + "\nret;", params), "k", grid=1, block=block, args=args)


def test_generic_allocation_destination(sim):
    out = sim.empty(32, "u32")
    stats = launch(sim, """
        mov.u64 %generic, ptrs;
        cvta.shared.u64 %generic, %generic;
        tcgen05.alloc.cta_group::1.sync.aligned.b32 [%generic], 32;
        ld.shared.u32 %base, [%dst];
        tcgen05.st.sync.aligned.32x32b.x1.b32 [%base], {%tid};
        tcgen05.wait::st.sync.aligned;
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%base];
        tcgen05.wait::ld.sync.aligned;
        ld.param.u64 %out, [out];
        mul.wide.u32 %off, %tid, 4;
        add.u64 %out, %out, %off;
        st.global.u32 [%out], %v;
    """ + FREE, params=".param .u64 out", args=[out])
    np.testing.assert_array_equal(out.numpy(), np.arange(32, dtype=np.uint32))
    assert not stats.findings


@pytest.mark.parametrize("prefix", ["@%one ", "@!%one "])
@pytest.mark.parametrize("operation", [
    ALLOC,
    FREE,
    "tcgen05.st.sync.aligned.32x32b.x1.b32 [%base], {%v};",
    "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%base];",
    "tcgen05.wait::ld.sync.aligned;",
    "tcgen05.wait::st.sync.aligned;",
    "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
])
def test_collective_instruction_rejects_predicated_subset(sim, prefix, operation):
    setup = "" if operation == ALLOC else ALLOC + LOAD_BASE
    with pytest.raises(warplens.ExecutionError, match="all 32 lanes"):
        launch(sim, setup + prefix + operation)


@pytest.mark.parametrize("setup, block", [("", 16), ("@%one ret;", 32)])
def test_collectives_also_reject_unlaunched_or_exited_lanes(sim, setup, block):
    with pytest.raises(warplens.ExecutionError, match="all 32 lanes"):
        launch(sim, setup + ALLOC, block=block)


@pytest.mark.parametrize("operation", [
    "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%dst], %n;",
    "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, %n;",
])
def test_allocation_column_count_is_uniform(sim, operation):
    setup = ALLOC + LOAD_BASE if "dealloc" in operation else ""
    with pytest.raises(warplens.ExecutionError, match="same column count"):
        launch(sim, setup + "selp.u32 %n, 32, 64, %one;" + operation)


@pytest.mark.parametrize("instruction", [
    "tcgen05.st.sync.aligned.32x32b.x1.b32 [%base], {%v};",
    "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%base];",
])
def test_transfer_address_is_uniform(sim, instruction):
    with pytest.raises(warplens.ExecutionError, match="same tensor-memory address"):
        launch(sim, ALLOC + LOAD_BASE + "add.u32 %base, %base, %tid;" + instruction)


def test_deallocation_requires_exact_returned_base(sim):
    with pytest.raises(warplens.ExecutionError, match="does not match an allocation"):
        launch(sim, ALLOC + LOAD_BASE + "add.u32 %base, %base, 65536;" + FREE)


def test_relinquished_permit_applies_to_later_allocations(sim):
    with pytest.raises(warplens.ExecutionError, match="relinquished its allocation permit"):
        launch(sim, ALLOC + LOAD_BASE + FREE +
               "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" + ALLOC)


def test_an_allocation_larger_than_the_one_before_is_reported_and_made(sim):
    """The PTX contract asks for sizes that do not increase; a B200 allocates all the same
    (validation/blackwell/measured/b200/tmem/alloc_increasing.txt), so it is a finding. It
    was an error while nothing had been measured."""
    stats = launch(sim, ALLOC + LOAD_BASE + FREE + ALLOC.replace("32;", "64;") + LOAD_BASE + FREE.replace("32;", "64;"))
    (finding,) = stats.findings
    assert finding.kind == "tmem-alloc-increasing" and "64 columns after one of 32" in finding.message


def test_equal_and_decreasing_allocations_remain_valid(sim):
    stats = launch(sim, ALLOC.replace("32;", "64;") + LOAD_BASE + FREE.replace("32;", "64;")
                   + ALLOC + LOAD_BASE + FREE + ALLOC + LOAD_BASE + FREE)
    assert not stats.findings


def test_exhausted_allocation_is_a_descriptive_deadlock(sim):
    with pytest.raises(warplens.DeadlockError, match="waiting for 32 tensor-memory columns"):
        launch(sim, ALLOC.replace("32;", "512;") + ALLOC)


def test_blocked_allocation_resumes_after_another_warp_frees_memory(sim):
    # Warp 1 attempts its allocation while warp 0 owns all columns. A later
    # deallocation by warp 0 must unblock it without restarting the kernel.
    mod = kernel("""
        .reg .b32 %w, %dst, %base, %i;
        .reg .pred %first, %again;
        .shared .align 4 .u32 ptrs[2];
        mov.u32 %w, %warpid;
        mov.u32 %dst, ptrs;
        mad.lo.u32 %dst, %w, 4, %dst;
        setp.eq.u32 %first, %w, 0;
        @%first tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%dst], 512;
        bar.sync 0;
        @!%first bra SECOND;
        mov.u32 %i, 0;
      DELAY:
        add.u32 %i, %i, 1;
        setp.lt.u32 %again, %i, 8;
        @%again bra DELAY;
        ld.shared.u32 %base, [%dst];
        tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 512;
        bra END;
      SECOND:
        tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%dst], 32;
        ld.shared.u32 %base, [%dst];
        tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 32;
      END:
        bar.sync 0;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=64)
    assert not stats.findings


@pytest.mark.parametrize("operation", [
    ALLOC.replace("alloc.", "alloc.exclusive."),
    FREE.replace("dealloc.", "dealloc.exclusive."),
    "tcgen05.relinquish_alloc_permit.cta_group::3.sync.aligned;",
    "tcgen05.wait::ld.sync.aligned.unsupported;",
    "tcgen05.st.sync.aligned.32x32b.x1.b64 [%base], {%v};",
    "tcgen05.ld.sync.aligned.32x32b.x1.unsupported.b32 {%v}, [%base];",
    "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%base], 0;",
])
def test_unimplemented_modifiers_or_extra_operands_are_rejected(sim, operation):
    with pytest.raises(warplens.UnsupportedError):
        launch(sim, ALLOC + LOAD_BASE + operation)


@pytest.mark.parametrize("shape, count, registers", [
    ("32x32b", 3, 3), ("16x64b", 256, 256),
    ("16x128b", 128, 256), ("16x256b", 64, 256),
])
@pytest.mark.parametrize("store", [False, True])
def test_repeat_count_must_be_a_supported_power_of_two(sim, shape, count, registers, store):
    regs = "{" + ", ".join(f"%r{i}" for i in range(registers)) + "}"
    operands = f"[%base], {regs}" if store else f"{regs}, [%base]"
    ins = f"tcgen05.{'st' if store else 'ld'}.sync.aligned.{shape}.x{count}.b32 {operands};"
    with pytest.raises(warplens.UnsupportedError, match="repeat count"):
        launch(sim, ins)


@pytest.mark.parametrize("shape, count", [
    ("32x32b", 0), ("16x64b", -1), ("16x128b", 3), ("16x256b", 64),
])
def test_transfer_layout_helper_rejects_invalid_counts(shape, count):
    with pytest.raises(ValueError, match="repeat count"):
        transfer_cells(shape, count)


@pytest.mark.parametrize("shape, count, nrows, ncols", [
    ("32x32b", 128, 32, 128), ("16x64b", 128, 16, 256),
    ("16x128b", 64, 16, 256), ("16x256b", 32, 16, 256),
])
def test_largest_legal_transfer_covers_expected_rectangle(shape, count, nrows, ncols):
    rows, cols = transfer_cells(shape, count)
    assert rows.shape == cols.shape == (32, 128)
    assert set(zip(rows.ravel().tolist(), cols.ravel().tolist())) == {
        (row, col) for row in range(nrows) for col in range(ncols)
    }


def test_repeated_deallocation_is_a_finding_not_a_failure(sim):
    """hgemm_v7 to v9 of the book release their one allocation from every warpgroup."""
    stats = launch(sim, ALLOC + LOAD_BASE + FREE + FREE)
    assert [f.kind for f in stats.findings] == ["tmem-dealloc-repeated"]


def test_deallocation_after_reallocation_is_ordinary(sim):
    stats = launch(sim, ALLOC + LOAD_BASE + FREE + ALLOC + LOAD_BASE + FREE)
    assert not stats.findings
