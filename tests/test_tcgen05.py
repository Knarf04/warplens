"""tcgen05 MMA, tensor memory, and their checkers, on a hand-written kernel.

The test places operands A and B in shared memory itself, following CuTe's
canonical K-major layouts (cute/arch/mma_sm100_desc.hpp), builds the two
descriptors, and compares the accumulators read back from TMEM with NumPy.
"""

import numpy as np
import pytest

import warplens

from conftest import kernel

LAYOUTS = {"none": (0, 0, 16), "32B": (6, 1, 32), "64B": (4, 2, 64), "128B": (2, 3, 128)}
PARAMS = (".param .u64 img, .param .u64 out, .param .u64 da, .param .u64 db, .param .u32 idesc, "
          ".param .u32 nbytes, .param .u32 ncols, .param .u32 n, .param .u32 twice, "
          ".param .u32 wait, .param .u32 shift, .param .u32 free, .param .u32 dlane, .param .u64 img2")


RAW_COPY = ("@%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%buf], [%img], %nb, [%bar];")
TMA_LOADS = """
        ld.param.u64 %da2, [img2];
        add.u32 %x, %buf, 16384;
        @%one cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%img, {0, 0}], [%bar];
        @%one cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%x], [%da2, {0, 0}], [%bar];"""


def mma_kernel(kind="f16", fill=RAW_COPY):
    """`fill` puts the operands in shared memory: one raw copy of a prepared image,
    or two TMA tensor loads (A at offset 0, B at 16384)."""
    return kernel(f"""
        .reg .b64 %da2;
        .reg .b64 %img, %out, %da, %db, %off;
        .reg .b32 %idesc, %nb, %nc, %n, %twice, %wait, %shift, %free;
        .reg .b32 %t, %w, %buf, %bar, %tp, %base, %taddr, %i, %v, %x, %dlane, %maddr;
        .reg .pred %p, %one, %two, %q, %no, %yes, %dowait, %dofree;
        .shared .align 1024 .b8 buf[49152];
        .shared .align 8 .u64 bar;
        .shared .align 4 .u32 tptr;
        ld.param.u64 %img, [img]; ld.param.u64 %out, [out];
        ld.param.u64 %da, [da]; ld.param.u64 %db, [db]; ld.param.u32 %idesc, [idesc];
        ld.param.u32 %nb, [nbytes]; ld.param.u32 %nc, [ncols]; ld.param.u32 %n, [n];
        ld.param.u32 %twice, [twice]; ld.param.u32 %wait, [wait];
        ld.param.u32 %shift, [shift]; ld.param.u32 %free, [free];
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid;
        setp.eq.u32 %one, %t, 0; setp.eq.u32 %q, %w, 0;
        setp.ne.u32 %no, %t, %t; setp.eq.u32 %yes, %t, %t;
        setp.ne.u32 %two, %twice, 0; and.pred %two, %two, %one;
        setp.ne.u32 %dowait, %wait, 0; setp.ne.u32 %dofree, %free, 0; and.pred %dofree, %dofree, %q;
        mov.b32 %buf, buf; mov.b32 %bar, bar; mov.b32 %tp, tptr;
        @%one mbarrier.init.shared::cta.b64 [%bar], 1;
        @%q tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%tp], %nc;
        bar.sync 0;
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], %nb;
        {fill}
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        ld.shared.u32 %base, [%tp];
        ld.param.u32 %dlane, [dlane];
        shl.b32 %dlane, %dlane, 16;
        add.u32 %maddr, %base, %dlane;
        @%one tcgen05.mma.cta_group::1.kind::{kind} [%maddr], %da, %db, %idesc, {{0, 0, 0, 0}}, %no;
        @%two tcgen05.mma.cta_group::1.kind::{kind} [%maddr], %da, %db, %idesc, {{0, 0, 0, 0}}, %yes;
        @%one tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%bar];
        @%dowait mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 1;
        shl.b32 %x, %w, 21;
        add.u32 %taddr, %base, %x;
        add.u32 %taddr, %taddr, %shift;
        mov.u32 %i, 0;
      LOOP:
        add.u32 %x, %taddr, %i;
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {{%v}}, [%x];
        mad.lo.u32 %x, %t, %n, %i;
        mul.wide.u32 %off, %x, 4;
        add.u64 %off, %out, %off;
        st.global.u32 [%off], %v;
        add.u32 %i, %i, 1;
        setp.lt.u32 %p, %i, %n;
        @%p bra LOOP;
        bar.sync 0;
        @%dofree tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, %nc;
        ret;
    """, PARAMS)


def place(matrix, layout, start, image):
    """Write a K-major operand (rows x K) into `image` the way CuTe lays it out; return its descriptor."""
    code, bits, row_bytes = LAYOUTS[layout]
    rows, k = matrix.shape
    size = matrix.dtype.itemsize
    per_chunk = 16 // size
    assert k == 2 * per_chunk
    sbo = 8 * row_bytes
    lbo = -(-rows // 8) * sbo if not bits else 16  # unswizzled: second K chunk after all rows
    raw = matrix.view(np.uint8).reshape(rows, k, size)
    for r in range(rows):
        for j in range(k):
            off = start + (r % 8) * row_bytes + (r // 8) * sbo + (j % per_chunk) * size + \
                (j // per_chunk) * (16 if bits else lbo)
            off ^= ((off >> 7) & ((1 << bits) - 1)) << 4  # Swizzle<bits, 4, 3>
            image[off:off + size] = raw[r, j]
    return (start >> 4) | ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (code << 61)


def run(sim, m=128, n=32, layout="128B", dtype=np.float16, twice=0, wait=1, shift=0, free=1,
        negate_a=False, ncols=None, seed=0, block=128, dlane=0, to_tf32=True):
    rng = np.random.default_rng(seed)
    k = 32 // np.dtype(dtype).itemsize
    a = rng.uniform(-2, 2, (m, k)).astype(dtype)
    b = rng.uniform(-2, 2, (n, k)).astype(dtype)
    if dtype == np.float32 and to_tf32:  # what a kernel does before a kind::tf32 MMA
        from warplens.units.floating import to_tf32 as convert
        a, b = (convert(x.view(np.uint32)).view(np.float32) for x in (a, b))
    image = np.zeros(32768, dtype=np.uint8)
    desc_a = place(a, layout, 0, image)
    desc_b = place(b, layout, 16384, image)
    fmt = 0 if dtype == np.float16 else 2
    idesc = (1 << 4) | (fmt << 7) | (fmt << 10) | (int(negate_a) << 13) | ((n >> 3) << 17) | ((m >> 4) << 24)
    out = sim.from_numpy(np.full((128, n), np.nan, dtype=np.float32))
    stats = sim.launch(mma_kernel("f16" if dtype == np.float16 else "tf32"), "k", grid=1, block=block,
                       args=[sim.from_numpy(image), out, desc_a, desc_b, idesc, len(image),
                             ncols or max(32, 1 << (n - 1).bit_length()), n, twice, wait, shift, free, dlane, 0])
    expect = a.astype(np.float32) @ b.astype(np.float32).T * (-1 if negate_a else 1) * (2 if twice else 1)
    return out.numpy(), expect, stats


@pytest.mark.parametrize("layout", ["none", "32B", "64B", "128B"])
@pytest.mark.parametrize("n", [8, 32, 96])
def test_mma_128_rows_for_every_operand_layout(sim, layout, n):
    got, expect, stats = run(sim, m=128, n=n, layout=layout)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)  # row m is TMEM lane m
    assert not stats.findings


def test_mma_64_rows_use_the_low_16_lanes_of_each_warp(sim):
    got, expect, stats = run(sim, m=64, n=32, wait=1)
    lanes = 32 * (np.arange(64) // 16) + np.arange(64) % 16
    np.testing.assert_allclose(got[lanes], expect, rtol=1e-5, atol=1e-5)
    # The upper 16 lanes of each warp were never written.
    assert [f.kind for f in stats.findings] == ["tmem-uninitialized-read"]
    assert sorted(set(stats.findings[0].lanes)) == list(range(16, 32))


def test_layout_f_as_the_course_notes_state_it(sim):
    """"Study Notes for Modern GPU Programming For MLSys", cta_group::1 with M = 64 (Layout F):
    lane = (m/16)*32 + a + m%16 with a in {0, 16}; for a = 0, rows 0-15 -> lanes 0-15,
    16-31 -> 32-47, 32-47 -> 64-79, 48-63 -> 96-111. A second tile with a = 16 uses the
    other 16 lanes of each quadrant, in the same columns."""
    from warplens.viz.tmem import accumulator_rows
    m = np.arange(64)
    for a in (0, 16):
        got, expect, _ = run(sim, m=64, n=32, dlane=a)
        lanes = (m // 16) * 32 + a + m % 16
        np.testing.assert_allclose(got[lanes], expect, rtol=1e-5, atol=1e-5)
        assert (got[np.setdiff1d(np.arange(128), lanes)] == 0).all()  # the other lanes stay unwritten
        assert accumulator_rows(64, a)[lanes].tolist() == m.tolist()
    rows = accumulator_rows(64)
    assert [rows[l] for l in (0, 15, 32, 47, 64, 79, 96, 111)] == [0, 15, 16, 31, 32, 47, 48, 63]
    assert accumulator_rows(128).tolist() == list(range(128))  # M = 128: row m -> lane m


def test_accumulate_negate_and_tf32(sim):
    got, expect, stats = run(sim, twice=1)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    got, expect, _ = run(sim, negate_a=True)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    got, expect, stats = run(sim, dtype=np.float32, n=16)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    assert not stats.findings


def test_tf32_mma_with_unconverted_float32_operands_is_reported(sim):
    got, expect, stats = run(sim, dtype=np.float32, n=16, to_tf32=False)
    (f,) = stats.findings
    assert f.kind == "mma-tf32-operand-not-converted" and f.count == 2  # operands A and B
    assert "low 13 fraction bits" in f.message and "cvt.rna.tf32.f32" in f.message
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)  # multiplied as plain float32


def test_reading_accumulators_without_waiting_for_the_commit_is_caught(sim):
    got, expect, stats = run(sim, wait=0)
    kinds = {f.kind for f in stats.findings}
    assert "tmem-read-before-commit" in kinds
    first = next(f for f in stats.findings if f.kind == "tmem-read-before-commit")
    assert first.text.startswith("tcgen05.ld") and "tcgen05.mma at line" in first.message


def test_lane_rule(sim):
    # Three warps, each naming the lanes of the next one.
    got, expect, stats = run(sim, m=128, n=8, shift=32 << 16, ncols=32, block=96)
    f = next(f for f in stats.findings if f.kind == "tmem-lane-rule")
    assert "warp 0 may only access tensor-memory lanes 0-31, but its address reaches lane 63" in f.message
    assert "stays inside the warp's own window" in f.message
    np.testing.assert_allclose(got[:96], expect[:96], rtol=1e-5, atol=1e-5)  # each warp read its own rows


def test_the_same_address_in_every_warp_reaches_each_warps_own_lanes(sim):
    """A warp's lane window comes from its position in the warpgroup; the address selects within
    it. TIRx passes one address (lane 0) in all four warps; CuTe passes each window's base lane.
    Store the first way, load the second way: both must name the same cells."""
    mod = kernel("""
        .reg .b64 %o, %off; .reg .b32 %t, %w, %tp, %base, %own, %v, %x; .reg .pred %q;
        .shared .align 4 .u32 tptr;
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid;
        setp.eq.u32 %q, %w, 0;
        mov.b32 %tp, tptr;
        @%q tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%tp], 32;
        bar.sync 0;
        ld.shared.u32 %base, [%tp];
        add.u32 %v, %t, 500;
        tcgen05.st.sync.aligned.32x32b.x1.b32 [%base], {%v};
        tcgen05.wait::st.sync.aligned;
        shl.b32 %x, %w, 21;
        add.u32 %own, %base, %x;
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%own];
        tcgen05.wait::ld.sync.aligned;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %t, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %v;
        bar.sync 0;
        @%q tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 32;
        ret;
    """, ".param .u64 out")
    out = sim.empty(128, "u32")
    stats = sim.launch(mod, "k", grid=1, block=128, args=[out])
    assert out.numpy().tolist() == [500 + t for t in range(128)] and not stats.findings


def test_leaked_allocation_is_reported(sim):
    _, _, stats = run(sim, free=0)
    (f,) = stats.findings
    assert f.kind == "tmem-leak" and f.text.startswith("@%q tcgen05.alloc") and "32 tensor-memory columns" in f.message


def test_columns_outside_the_allocation_are_reported(sim):
    _, _, stats = run(sim, n=64, ncols=32)
    assert "tmem-unallocated" in {f.kind for f in stats.findings}


def test_allocation_size_must_be_a_power_of_two_of_at_least_32(sim):
    for bad in (16, 48, 1024):
        with pytest.raises(warplens.ExecutionError, match=f"tcgen05.alloc of {bad} columns"):
            run(sim, ncols=bad)


def test_unsupported_tcgen05_forms_are_errors(sim):
    for line, why in [
        ("tcgen05.alloc.cta_group::3.sync.aligned.shared::cta.b32 [%a], 32;", "cta_group::1 or"),
        ("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%a}, [%a], 1;", "implemented shapes are"),
        ("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%a}, [%a];", "without packing"),
        ("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%a}, [%a];", r"expected 2 register\(s\)"),
        ("tcgen05.cp.4x256b.1x64b [%a], %d;", "tcgen05.cp"),
    ]:
        mod = kernel(f".reg .b32 %a; .reg .b64 %d;\n.shared .align 4 .u32 cell;\nmov.b32 %a, cell;\n{line}\nret;")
        with pytest.raises(warplens.UnsupportedError, match=why):
            sim.launch(mod, "k", grid=1, block=32)


# -- load and store shapes ------------------------------------------------------------------

# Copy_Traits of SM100_TMEM_LOAD_<name> in cute/atom/copy_traits_sm100.hpp, copied as data:
# ((value bits per lane, lanes), DstLayout shape, DstLayout stride). The DstLayout maps
# (thread, register bit) to a value index; ValID then splits that index into (bit, lane).
CUTE_TRAITS = {
    ("32x32b", 1): ((32, 32), (32, 32), (32, 1)),
    ("32x32b", 4): ((128, 32), (32, 128), (128, 1)),
    ("16x64b", 1): ((64, 16), ((2, 2, 8), 32), ((512, 32, 64), 1)),
    ("16x64b", 2): ((128, 16), ((2, 2, 8), (32, 2)), ((1024, 32, 128), (1, 64))),
    ("16x64b", 4): ((256, 16), ((2, 2, 8), (32, 4)), ((2048, 32, 256), (1, 64))),
    ("16x128b", 1): ((128, 16), ((4, 8), (32, 2)), ((32, 128), (1, 1024))),
    ("16x128b", 2): ((256, 16), ((4, 8), (32, 2, 2)), ((32, 256), (1, 2048, 128))),
    ("16x128b", 4): ((512, 16), ((4, 8), (32, 2, 4)), ((32, 512), (1, 4096, 128))),
    ("16x256b", 2): ((512, 16), ((4, 8), (64, 2, 2)), ((64, 512), (1, 4096, 256))),
    ("16x256b", 4): ((1024, 16), ((4, 8), (64, 2, 4)), ((64, 1024), (1, 8192, 256))),
}


def cute_index(shape, stride, flat):
    """Evaluate a CuTe layout mode at a flat (column-major) coordinate."""
    if isinstance(shape, int):
        return stride * flat
    total = 0
    for sub_shape, sub_stride in zip(shape, stride):
        size = sub_shape if isinstance(sub_shape, int) else int(np.prod(np.array(sub_shape).ravel()))
        total += cute_index(sub_shape, sub_stride, flat % size)
        flat //= size
    return total


def cute_cells(traits):
    """(lane, column) of every register of every thread, from the CuTe layouts alone."""
    (bits, _), (thread_shape, value_shape), (thread_stride, value_stride) = traits
    value_bits = value_shape if isinstance(value_shape, int) else int(np.prod(value_shape))
    lanes, columns = [], []
    for thread in range(32):
        row_l, row_c = [], []
        for register in range(value_bits // 32):
            index = cute_index(thread_shape, thread_stride, thread) + \
                cute_index(value_shape, value_stride, register * 32)
            row_l.append(index // bits)
            row_c.append(index % bits // 32)
        lanes.append(row_l)
        columns.append(row_c)
    return np.array(lanes), np.array(columns)


@pytest.mark.parametrize("shape, count", sorted(CUTE_TRAITS))
def test_transfer_shapes_agree_with_cute_layouts(shape, count):
    from warplens.units.tcgen05 import transfer_cells
    lanes, columns = transfer_cells(shape, count)
    expect_lanes, expect_columns = cute_cells(CUTE_TRAITS[shape, count])
    np.testing.assert_array_equal(lanes, expect_lanes)
    np.testing.assert_array_equal(columns, expect_columns)
    # Every cell of the covered block is reached exactly once.
    cells = set(zip(lanes.ravel().tolist(), columns.ravel().tolist()))
    assert len(cells) == lanes.size
    assert cells == {(l, c) for l in range(lanes.max() + 1) for c in range(columns.max() + 1)}


def shape_kernel(shape, count, half):
    """Each warp stores lane * 1000 + column over its 32 lanes with 32x32b, then reads its
    lower or upper 16 lanes back with `shape`, and writes the registers to `out`."""
    registers = {"16x64b": 1, "16x128b": 2, "16x256b": 4}[shape] * count
    width = {"16x64b": 2, "16x128b": 4, "16x256b": 8}[shape] * count
    fill = "\n".join(f"""
        add.u32 %v, %lane, {c};
        add.u32 %x, %taddr, {c};
        tcgen05.st.sync.aligned.32x32b.x1.b32 [%x], {{%v}};""" for c in range(width))
    names = ", ".join(f"%g{i}" for i in range(registers))
    stores = "\n".join(f"st.global.u32 [%o+{4 * i}], %g{i};" for i in range(registers))
    return kernel(f"""
        .reg .b64 %o, %off; .reg .b32 %t, %w, %l, %lane, %tp, %base, %taddr, %v, %x, %g<{registers}>;
        .reg .pred %q;
        .shared .align 4 .u32 tptr;
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid; mov.u32 %l, %laneid;
        setp.eq.u32 %q, %w, 0;
        mov.b32 %tp, tptr;
        @%q tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%tp], 32;
        bar.sync 0;
        ld.shared.u32 %base, [%tp];
        shl.b32 %x, %w, 21;
        add.u32 %taddr, %base, %x;
        mad.lo.u32 %lane, %w, 32, %l;
        mul.lo.u32 %lane, %lane, 1000;
        {fill}
        tcgen05.wait::st.sync.aligned;
        add.u32 %x, %taddr, {half * 16 << 16};
        tcgen05.ld.sync.aligned.{shape}.x{count}.b32 {{{names}}}, [%x];
        tcgen05.wait::ld.sync.aligned;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %t, {4 * registers};
        add.u64 %o, %o, %off;
        {stores}
        bar.sync 0;
        @%q tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 32;
        ret;
    """, ".param .u64 out"), registers


@pytest.mark.parametrize("shape, count", [("16x64b", 1), ("16x64b", 4), ("16x128b", 1),
                                          ("16x128b", 2), ("16x256b", 1), ("16x256b", 4)])
@pytest.mark.parametrize("half", [0, 1])
def test_store_then_load_with_each_shape(sim, shape, count, half):
    from warplens.units.tcgen05 import transfer_cells
    mod, registers = shape_kernel(shape, count, half)
    out = sim.empty((128, registers), "u32")
    stats = sim.launch(mod, "k", grid=1, block=128, args=[out])
    lanes, columns = transfer_cells(shape, count)
    warp = np.arange(128)[:, None] // 32
    expect = (32 * warp + 16 * half + np.tile(lanes, (4, 1))) * 1000 + np.tile(columns, (4, 1))
    np.testing.assert_array_equal(out.numpy(), expect)
    assert not stats.findings


# -- TMA feeding the MMA --------------------------------------------------------------------

def run_with_tma(sim, tma_swizzle, descriptor_layout, n=32):
    """A (128 x K) and B (n x K) reach shared memory by TMA tensor loads, one box each
    whose inner dimension is the swizzle span, then one MMA reads the first 16 of K."""
    inner = {"32B": 32, "64B": 64, "128B": 128}[tma_swizzle]
    k = inner // 2
    rng = np.random.default_rng(1)
    a = rng.uniform(-2, 2, (128, k)).astype(np.float16)
    b = rng.uniform(-2, 2, (n, k)).astype(np.float16)
    map_a = sim.tensor_map_tiled(sim.from_numpy(a), (128, k), swizzle=tma_swizzle)
    map_b = sim.tensor_map_tiled(sim.from_numpy(b), (n, k), swizzle=tma_swizzle)
    code, _, row_bytes = LAYOUTS[descriptor_layout]
    descriptor = lambda start: (start >> 4) | (1 << 16) | ((8 * row_bytes >> 4) << 32) | (1 << 46) | (code << 61)
    idesc = (1 << 4) | ((n >> 3) << 17) | (8 << 24)
    out = sim.from_numpy(np.full((128, n), np.nan, dtype=np.float32))
    stats = sim.launch(mma_kernel(fill=TMA_LOADS), "k", grid=1, block=128,
                       args=[map_a, out, descriptor(0), descriptor(16384), idesc,
                             a.nbytes + b.nbytes, 32, n, 0, 1, 0, 1, 0, map_b])
    expect = a[:, :16].astype(np.float32) @ b[:, :16].astype(np.float32).T
    return out.numpy(), expect, stats


@pytest.mark.parametrize("swizzle", ["32B", "64B", "128B"])
def test_tma_loaded_operands_feed_the_mma(sim, swizzle):
    # What TMA writes for a swizzle mode is what the MMA expects for the same layout type.
    got, expect, stats = run_with_tma(sim, swizzle, swizzle)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    assert not stats.findings


def test_descriptor_that_disagrees_with_the_tma_swizzle_is_caught(sim):
    got, expect, stats = run_with_tma(sim, "128B", "64B")
    kinds = [f.kind for f in stats.findings]
    assert kinds.count("mma-swizzle-mismatch") == 1  # one finding per MMA instruction
    f = stats.findings[kinds.index("mma-swizzle-mismatch")]
    assert "descriptor says 64B swizzle" in f.message and "with 128B swizzle" in f.message
    assert f.text.startswith("@%one tcgen05.mma") and "TMA load at line" in f.message
    assert not np.allclose(got, expect, atol=1e-3)  # and the result really is wrong
