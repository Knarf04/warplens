"""TMA loads and stores, tensor maps, swizzle, mbarriers and the hazard checkers."""

import numpy as np
import pytest

import warplens
from warplens.frontend.ir import Reg, Sym, TensorRef

from tests.support import kernel
from tests.support.tma import (EARLY_REUSE, EXPECT_TOO_FEW, EXPECT_TOO_MANY, NO_FENCE, NO_WAIT, box_kernel,
                               mb_kernel, record, tma_copy)


# -- M2 acceptance -------------------------------------------------------------------

@pytest.mark.parametrize("swizzle, box", [
    (None, (16, 32)), ("32B", (16, 8)), ("64B", (16, 16)), ("128B", (16, 32)),
    ("128B", (20, 24)),  # tiles that hang over the edge of the tensor
])
def test_round_trip_for_every_swizzle_mode(swizzle, box):
    ok, stats = tma_copy(swizzle, box)
    assert ok and not stats.findings


@pytest.mark.parametrize("delay", [0, 1, 50])
def test_round_trip_does_not_depend_on_async_timing(delay):
    ok, stats = tma_copy("128B", (16, 32), async_delay=delay)
    assert ok and not stats.findings


def test_expect_tx_too_small_is_caught():
    """A copy reports its bytes in pieces, so a count of half the copy reaches zero part of
    the way through and the phase completes: the wait returns early, as on a B200
    (validation/blackwell/measured/b200/mbarrier/expect_tx_small.txt), and that is reported.
    The other half is charged to a phase that never comes."""
    ok, stats = tma_copy("128B", (16, 32), EXPECT_TOO_FEW)
    (f,) = stats.findings
    assert f.kind == "mbarrier-tx-mismatch"
    assert "a copy of 2048 byte(s) landed" in f.message and "expected only 1024 more" in f.message
    assert "a wait on it returns before the other 1024 are there" in f.message
    assert "cp.async.bulk.tensor.2d.shared" in f.text and f.cta == (0, 0, 0)
    assert ok  # the simulator delivers a copy whole; on a GPU the tile may be read half written


def test_an_undersized_expect_tx_completes_only_at_a_whole_number_of_pieces():
    """TX_PIECE_BYTES is 32, as a B200 has it (measured/b200/mbarrier/expect_tx_sweep.txt): a
    count that is not a multiple of it is passed over, and the wait deadlocks as before. At
    a multiple of 32 a GPU completes the phase in some runs and not in others; the simulator
    always does, and reports it."""
    from warplens.units.mbarrier import Mbarrier, TX_PIECE_BYTES

    class Cta:
        def __init__(self):
            self.completed, self.findings = 0, self
            self.kinds = []
        def phase_completed(self, offset=None, phase=None):
            self.completed += 1
        def add(self, kind, *rest):
            self.kinds.append(kind)

    def phases(expect, copy=256):
        cta = Cta()
        bar = Mbarrier(cta, 0, 1, 0)
        bar.expect_tx(expect)
        bar.pending = 0  # the arrival that came with expect_tx
        bar.deliver(copy, None, None, [0])
        return cta.completed, bar.tx, cta.kinds

    assert TX_PIECE_BYTES == 32
    assert phases(256) == (1, 0, [])
    assert phases(128) == (1, -128, ["mbarrier-tx-mismatch"]) and phases(64) == (1, -192, ["mbarrier-tx-mismatch"])
    assert phases(96) == (1, -160, ["mbarrier-tx-mismatch"]) and phases(32) == (1, -224, ["mbarrier-tx-mismatch"])
    # What a B200 never completed in 80 runs: a count that is not a multiple of 32.
    assert phases(48) == (0, -208, []) and phases(16) == (0, -240, []) and phases(255) == (0, -1, [])
    assert phases(64, copy=128) == (1, -64, ["mbarrier-tx-mismatch"]) and phases(80, copy=128) == (0, -48, [])
    assert phases(512) == (0, 256, [])


def test_expect_tx_too_large_is_a_deadlock_that_names_the_mbarrier():
    with pytest.raises(warplens.DeadlockError) as info:
        tma_copy("128B", (16, 32), EXPECT_TOO_MANY)
    assert not info.value.findings  # nothing was delivered beyond what was expected
    msg = str(info.value)
    assert "warps 0-3 wait at line" in msg and "mbarrier.try_wait.parity" in msg
    assert "still needs 0 arrival(s) and 2048 byte(s) of TMA data" in msg
    assert "no TMA load in flight will deliver them" in msg


@pytest.mark.parametrize("delay", [0, 4, 1000])
def test_missing_wait_is_caught_whatever_the_timing(delay):
    # With no delay the data happens to be there in time; the missing wait is still a bug.
    _, stats = tma_copy("128B", (16, 32), NO_WAIT, async_delay=delay)
    kinds = {f.kind for f in stats.findings}
    assert "tma-load-unsynchronized" in kinds
    first = next(f for f in stats.findings if f.kind == "tma-load-unsynchronized")
    assert "ld.shared.b32" in first.text and "TMA load at line" in first.message


def test_missing_fence_is_caught():
    _, stats = tma_copy("128B", (16, 32), NO_FENCE)
    (f,) = stats.findings
    assert f.kind == "tma-store-missing-fence" and "global.shared::cta.bulk_group" in f.text


def test_early_reuse_of_store_source_is_caught():
    _, stats = tma_copy("128B", (16, 32), EARLY_REUSE)
    (f,) = stats.findings
    assert f.kind == "tma-store-source-overwritten" and f.text.startswith("st.shared.b32")


# -- loads and stores of every rank ----------------------------------------------------

def extract_box(data, origin, box, steps, fill=0):
    """What TMA should load: NumPy-order origin/box/steps, out-of-bounds filled."""
    counts = [-(-b // s) for b, s in zip(box, steps)]
    out = np.full(counts, fill, dtype=data.dtype)
    for index in np.ndindex(*counts):
        source = tuple(o + i * s for o, i, s in zip(origin, index, steps))
        if all(0 <= x < n for x, n in zip(source, data.shape)):
            out[index] = data[source]
    return out


@pytest.mark.parametrize("shape, box, origin", [
    ((64,), (16,), (8,)),
    ((64,), (32,), (48,)),  # runs off the end
    ((12, 16), (4, 8), (3, 4)),
    ((12, 16), (5, 8), (-2, 12)),  # negative coordinate and overhang
    ((6, 10, 8), (2, 3, 8), (4, 1, 0)),
    ((3, 4, 5, 16), (2, 2, 3, 4), (1, 3, 2, 12)),
    ((2, 3, 2, 4, 8), (2, 2, 2, 3, 4), (0, 1, 1, 2, 4)),
])
@pytest.mark.parametrize("swizzle", [None, "128B"])
def test_tile_load_and_store_1d_to_5d(sim, shape, box, origin, swizzle):
    rank = len(shape)
    data = np.random.default_rng(rank).integers(1, 2**31, shape).astype(np.uint32)
    src = sim.from_numpy(data)
    dst = sim.from_numpy(np.full(box, 0xDEADBEEF, dtype=np.uint32))
    src_map = sim.tensor_map_tiled(src, box, swizzle=swizzle)
    dst_map = sim.tensor_map_tiled(dst, box, swizzle=swizzle)
    stats = sim.launch(box_kernel(rank), "k", grid=1, block=1,
                       args=[src_map, dst_map, dst.nbytes, *origin[::-1]])  # TMA order: innermost first
    np.testing.assert_array_equal(dst.numpy(), extract_box(data, origin, box, (1,) * rank))
    assert not stats.findings


def test_element_strides_and_nan_fill(sim):
    data = np.arange(8 * 12 * 16, dtype=np.float32).reshape(8, 12, 16) + 1
    src = sim.from_numpy(data)
    box, steps, origin = (4, 6, 8), (2, 3, 1), (5, 9, 8)
    expect = extract_box(data, origin, box, steps, fill=np.nan)
    dst = sim.empty(expect.shape, "f32")
    src_map = sim.tensor_map_tiled(src, box, element_strides=steps, oob_fill="nan")
    assert src_map.info.counts == (8, 2, 2) and src_map.info.box_bytes == 128
    stats = sim.launch(box_kernel(3), "k", grid=1, block=1,
                       args=[src_map, sim.tensor_map_tiled(dst, expect.shape), 128, *origin[::-1]])
    np.testing.assert_array_equal(dst.numpy(), expect)
    assert np.isnan(expect).any() and not stats.findings
    # The bits of the fill are the B200's (validation/blackwell/measured/b200/tma/oob_fill.txt):
    # 0x7FF7 in every 16 bits of the element, not NumPy's default NaN.
    assert set(dst.numpy().view(np.uint32)[np.isnan(expect)].tolist()) == {0x7FF77FF7}


# -- swizzle ---------------------------------------------------------------------------

DUMP = """
    .reg .b64 %s, %o, %off; .reg .b32 %n, %t, %a, %v, %buf, %bar; .reg .pred %p;
    .shared .align 1024 .b8 pad[{pad}];
    .shared .align 16 .b8 buf[2048];
    .shared .align 8 .u64 bar;
    ld.param.u64 %s, [src]; ld.param.u64 %o, [out]; ld.param.u32 %n, [bytes];
    mov.u32 %t, %tid.x;
    mov.b32 %buf, buf; mov.b32 %bar, bar;
    setp.ne.u32 %p, %t, 0;
    @%p bra WAIT;
    mbarrier.init.shared::cta.b64 [%bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], %n;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%s, {{0, 0}}], [%bar];
  WAIT:
    bar.sync 0;
    mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
    add.u32 %a, %buf, %t;
    ld.shared.u8 %v, [%a];
    cvt.u64.u32 %off, %t;
    add.u64 %o, %o, %off;
    st.global.u8 [%o], %v;
    ret;
"""
DUMP_PARAMS = ".param .u64 src, .param .u64 out, .param .u32 bytes"


def cute_swizzle(offset: int, bits: int) -> int:
    """CuTe Swizzle<bits, 4, 3>, written independently of the simulator's version."""
    return offset ^ (((offset >> 7) & ((1 << bits) - 1)) << 4)


@pytest.mark.parametrize("swizzle, bits, inner", [(None, 0, 128), ("32B", 1, 32), ("64B", 2, 64), ("128B", 3, 128)])
def test_swizzled_bytes_land_where_cute_says(sim, swizzle, bits, inner):
    # A box of `inner` bytes x rows; then every thread copies one raw SMEM byte back out.
    rows = 1024 // inner
    data = np.arange(1024, dtype=np.uint16).reshape(rows, inner).astype(np.uint8) ^ \
        (np.arange(rows, dtype=np.uint8)[:, None] * 37)
    src_map = sim.tensor_map_tiled(sim.from_numpy(data), (rows, inner), swizzle=swizzle)
    out = sim.empty(1024, "u8")
    mod = kernel(DUMP.format(pad=1024), DUMP_PARAMS)
    stats = sim.launch(mod, "k", grid=1, block=1024, args=[src_map, out, 1024])
    smem = out.numpy()
    flat = data.reshape(-1)
    expect = np.zeros(1024, dtype=np.uint8)
    for o in range(1024):
        expect[cute_swizzle(o, bits)] = flat[o]
    np.testing.assert_array_equal(smem, expect)
    assert not stats.findings
    if bits:
        assert not np.array_equal(smem, flat)  # the swizzle really moved bytes


def test_buffer_not_aligned_to_swizzle_period_is_reported(sim):
    data = np.arange(256, dtype=np.uint8).reshape(2, 128)
    src_map = sim.tensor_map_tiled(sim.from_numpy(data), (2, 128), swizzle="128B")
    # buf lands at offset 128: aligned as a tensor copy must be (a B200 ends the kernel for
    # less, and so does the simulator), and not to the swizzle's period.
    mod = kernel(DUMP.format(pad=128), DUMP_PARAMS)
    out = sim.empty(256, "u8")
    stats = sim.launch(mod, "k", grid=1, block=256, args=[src_map, out, 256])
    f = stats.findings[0]
    assert f.kind == "tma-swizzle-misaligned" and "not aligned to 1024 bytes" in f.message
    # The consequence: bytes are not where offset-based indexing looks for them. The rows are
    # 1 and 2 of the period, not 0 and 1, so the chunks of each are in another order than a
    # buffer at offset 0 has them.
    at_zero = sim.empty(256, "u8")
    assert not sim.launch(kernel(DUMP.format(pad=1024), DUMP_PARAMS), "k", grid=1, block=256,
                          args=[src_map, at_zero, 256]).findings
    assert sorted(out.numpy().tolist()) == sorted(at_zero.numpy().tolist())
    assert not np.array_equal(out.numpy(), at_zero.numpy())


def test_a_tensor_copy_16_bytes_into_a_row_ends_the_kernel_as_on_a_b200(sim):
    """validation/blackwell/measured/b200/tma/swizzle_misaligned.sixteen.txt: a destination 16
    bytes into the buffer is cudaErrorMisalignedAddress, with or without a swizzle. The
    simulator refuses what is not a multiple of 128 (TENSOR_COPY_ALIGNMENT)."""
    data = np.arange(256, dtype=np.uint8).reshape(2, 128)
    for swizzle in (None, "128B"):
        src_map = sim.tensor_map_tiled(sim.from_numpy(data), (2, 128), swizzle=swizzle)
        with pytest.raises(warplens.ExecutionError, match="aligned to 128 bytes, got offset 16.*cudaErrorMisalignedAddress"):
            sim.launch(kernel(DUMP.format(pad=16), DUMP_PARAMS), "k", grid=1, block=256,
                       args=[src_map, sim.empty(256, "u8"), 256])


@pytest.mark.parametrize("pad", [32, 48, 64, 80, 96, 112, 144, 192])
def test_a_tensor_copy_needs_a_shared_address_aligned_to_128_bytes_as_on_a_b200(sim, pad):
    """The second rental (validation/blackwell/measured/b200/tma/swizzle_offsets.txt): every
    offset tried that is not a multiple of 128 ended the kernel, 32 and 64 among them, which
    the simulator let run before; 128 and 256 ran."""
    from warplens.units.tma import TENSOR_COPY_ALIGNMENT
    assert TENSOR_COPY_ALIGNMENT == 128
    data = np.arange(256, dtype=np.uint8).reshape(2, 128)
    src_map = sim.tensor_map_tiled(sim.from_numpy(data), (2, 128))
    with pytest.raises(warplens.ExecutionError, match=f"aligned to 128 bytes, got offset {pad}.*cudaErrorMisalignedAddress"):
        sim.launch(kernel(DUMP.format(pad=pad), DUMP_PARAMS), "k", grid=1, block=256,
                   args=[src_map, sim.empty(256, "u8"), 256])
    out = sim.empty(256, "u8")
    assert not [f for f in sim.launch(kernel(DUMP.format(pad=128), DUMP_PARAMS), "k", grid=1, block=256,
                                      args=[src_map, out, 256]).findings if "aligned" in f.message]


# -- mbarrier semantics ----------------------------------------------------------------

def test_phases_flip_parity(sim):
    mod = mb_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 2;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(0)}
        mbarrier.arrive.shared::cta.b64 %tok, [%bar];
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(1)}
        mbarrier.arrive.shared::cta.b64 %tok, [%bar];
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(2)}
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 1;
        {record(3)}
        mbarrier.test_wait.shared::cta.b64 %p, [%bar], %tok;
        {record(4)}
        mbarrier.arrive.shared::cta.b64 %tok, [%bar], 2;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 1;
        {record(5)}
    """)
    out = sim.empty(6, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    # not complete, still not, phase 0 complete, phase 1 not, token of phase 0 complete, phase 1 complete
    assert out.numpy().tolist() == [0, 0, 1, 0, 1, 1] and not stats.findings


def test_phase_needs_both_arrivals_and_bytes(sim):
    mod = mb_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%bar], 64;
        mbarrier.arrive.shared::cta.b64 _, [%bar];
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(0)}
        mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%bar], 48;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(1)}
        mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%bar], 16;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(2)}
    """)
    out = sim.empty(3, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [0, 0, 1]


def test_over_arrival_ends_the_kernel_as_on_a_b200(sim):
    """Three arrivals at once on an mbarrier that expected two: a B200 ends the kernel with
    cudaErrorLaunchFailure (validation/blackwell/measured/b200/mbarrier/over_arrival.txt).
    The first model carried the surplus into the next phase and reported a finding."""
    mod = mb_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 2;
        mbarrier.arrive.shared::cta.b64 _, [%bar], 3;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(0)}
    """)
    with pytest.raises(warplens.ExecutionError, match=r"3 arrival\(s\).*needed only 2 more.*mbarrier-over-arrival"):
        sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(2, "u32")])
    exact = mb_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 2;
        mbarrier.arrive.shared::cta.b64 _, [%bar], 2;
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        {record(0)}
    """)
    out = sim.empty(2, "u32")
    assert not sim.launch(exact, "k", grid=1, block=1, args=[out]).findings and out.numpy()[0] == 1


def test_mbarrier_errors(sim):
    out = sim.empty(1, "u32")
    with pytest.raises(warplens.ExecutionError, match="used before mbarrier.init"):
        sim.launch(mb_kernel("mbarrier.arrive.shared::cta.b64 _, [%bar];"), "k", grid=1, block=1, args=[out])
    with pytest.raises(warplens.ExecutionError, match="used before mbarrier.init"):
        sim.launch(mb_kernel("mbarrier.init.shared::cta.b64 [%bar], 1;\nmbarrier.inval.shared::cta.b64 [%bar];\n"
                             "mbarrier.arrive.shared::cta.b64 _, [%bar];"), "k", grid=1, block=1, args=[out])
    with pytest.raises(warplens.ExecutionError, match="arrival count of 0"):
        sim.launch(mb_kernel("mbarrier.init.shared::cta.b64 [%bar], 0;"), "k", grid=1, block=1, args=[out])
    with pytest.raises(warplens.UnsupportedError, match="mbarrier.arrive_drop"):
        sim.launch(mb_kernel("mbarrier.init.shared::cta.b64 [%bar], 1;\n"
                             "mbarrier.arrive_drop.shared::cta.b64 _, [%bar];"), "k", grid=1, block=1, args=[out])


def test_waiting_on_a_phase_nobody_completes_is_a_deadlock(sim):
    mod = mb_kernel("mbarrier.init.shared::cta.b64 [%bar], 2;\n"
                    "mbarrier.arrive.shared::cta.b64 _, [%bar];\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;")
    with pytest.raises(warplens.DeadlockError, match=r"still needs 1 arrival\(s\) and 0 byte"):
        sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(1, "u32")])


WARP_PAIR = """
    .reg .b32 %bar, %w, %v; .reg .pred %p, %q, %one;
    .shared .align 8 .u64 bar;
    .shared .align 4 .u32 cell;
    mov.b32 %bar, bar;
    mov.u32 %w, %warpid;
    setp.eq.u32 %q, %w, 0;
    mov.u32 %v, %tid.x;
    setp.eq.u32 %one, %v, 0;
"""


def test_mbarrier_orders_one_warp_after_another(sim):
    # Warp 0 writes a cell then arrives; warp 1 waits, then reads it. No race, no deadlock.
    mod = kernel(WARP_PAIR + """
        @%one mbarrier.init.shared::cta.b64 [%bar], 1;
        bar.sync 0;
        @%q st.shared.u32 [cell], 7;
        @%one mbarrier.arrive.shared::cta.b64 _, [%bar];
        @!%q mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        @!%q ld.shared.u32 %v, [cell];
        ret;
    """)
    assert not sim.launch(mod, "k", grid=1, block=64).findings


def test_every_lane_that_executes_an_arrive_arrives(sim):
    # 32 lanes arrive on a barrier that expects 32: exactly one phase completes.
    mod = kernel(WARP_PAIR + """
        .reg .b64 %o;
        @%one mbarrier.init.shared::cta.b64 [%bar], 32;
        mbarrier.arrive.shared::cta.b64 _, [%bar];
        mbarrier.test_wait.parity.shared::cta.b64 %p, [%bar], 0;
        selp.u32 %v, 1, 0, %p;
        ld.param.u64 %o, [out];
        @%one st.global.u32 [%o], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(1, "u32")
    stats = sim.launch(mod, "k", grid=1, block=32, args=[out])
    assert out.numpy().tolist() == [1] and not stats.findings


def test_mbarrier_used_by_another_warp_right_after_init_is_reported(sim):
    mod = kernel(WARP_PAIR + """
        @%q mbarrier.init.shared::cta.b64 [%bar], 1;
        @!%q mbarrier.arrive.shared::cta.b64 _, [%bar];
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=64)
    assert [f.kind for f in stats.findings] == ["mbarrier-init-race"]


# -- raw bulk copies, groups, leftovers ---------------------------------------------------

def test_raw_bulk_copy_round_trip(sim):
    data = np.arange(256, dtype=np.uint8)
    src, dst = sim.from_numpy(data), sim.empty(256, "u8")
    mod = kernel("""
        .reg .b64 %s, %d; .reg .b32 %buf, %bar; .reg .pred %p;
        .shared .align 16 .b8 buf[256];
        .shared .align 8 .u64 bar;
        ld.param.u64 %s, [src]; ld.param.u64 %d, [dst];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 256;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%s], 256, [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        cp.async.bulk.global.shared::cta.bulk_group [%d], [%buf], 256;
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group.read 0;
        ret;
    """, ".param .u64 src, .param .u64 dst")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src, dst])
    np.testing.assert_array_equal(dst.numpy(), data)
    assert not stats.findings


def test_returning_with_a_store_in_flight_is_reported(sim):
    data = np.arange(64, dtype=np.uint32)
    src, dst = sim.from_numpy(data), sim.empty(16, "u32")
    stats = sim.launch(box_kernel(1, raw_wait=""), "k", grid=1, block=1,
                       args=[sim.tensor_map_tiled(src, 16), sim.tensor_map_tiled(dst, 16), 64, 0])
    assert [f.kind for f in stats.findings] == ["tma-store-not-waited"]
    np.testing.assert_array_equal(dst.numpy(), data[:16])  # the copy still completes


def test_tma_errors(sim):
    data = sim.from_numpy(np.zeros((4, 16), dtype=np.uint32))
    dst = sim.empty((4, 16), "u32")
    good = sim.tensor_map_tiled(data, (4, 16))
    with pytest.raises(warplens.ExecutionError, match="1d TMA instruction used with a 2-dimensional"):
        sim.launch(box_kernel(1), "k", grid=1, block=1, args=[good, good, 256, 0])
    # Tensor data where a tensor map belongs: not one of ours, and not a driver's either.
    with pytest.raises(warplens.ExecutionError, match="invalid tensor map|neither a tensor map"):
        sim.launch(box_kernel(2), "k", grid=1, block=1, args=[data, good, 256, 0, 0])
    big = sim.tensor_map_tiled(sim.empty((256, 128), "u8"), (256, 128))
    with pytest.raises(warplens.ExecutionError, match=r"runs past\s+the CTA's 16392 bytes"):
        sim.launch(box_kernel(2), "k", grid=1, block=1, args=[big, big, 32768, 0, 0])
    with pytest.raises(warplens.UnsupportedError, match="im2col"):
        sim.launch(kernel("""
            .reg .b64 %s; .reg .b32 %a;
            cp.async.bulk.tensor.1d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%a], [%s, {0}], [%a];
            ret;"""), "k", grid=1, block=1)
    del dst


@pytest.mark.parametrize("kwargs, message", [
    (dict(box=(4, 6)), "innermost box dimension is 24 bytes; it must be a multiple of 16"),
    (dict(box=(4, 16), swizzle="32B"), "more than the 32-byte span of swizzle 32B"),
    (dict(box=(4, 300)), "box dimensions must be between 1 and 256"),
    (dict(box=(4,)), "one entry per dimension"),
    (dict(box=(4, 8), swizzle="256B"), "unknown swizzle"),
    (dict(box=(4, 8), element_strides=(9, 1)), "element strides must be between 1 and 8"),
    (dict(box=(4, 8), oob_fill="nan"), "needs a floating-point tensor"),
])
def test_tensor_map_validation(sim, kwargs, message):
    with pytest.raises(warplens.LaunchError, match=message):
        sim.tensor_map_tiled(sim.empty((8, 16), "u32"), **kwargs)


def test_tensor_map_rejects_rows_that_are_not_16_byte_multiples(sim):
    with pytest.raises(warplens.LaunchError, match="stride of 40 bytes is not allowed"):
        sim.tensor_map_tiled(sim.empty((8, 10), "u32"), (4, 4))


# -- parser ------------------------------------------------------------------------------

def test_inline_asm_blocks_have_their_own_registers_and_labels():
    mod = kernel("""
        .reg .pred %p; .reg .b32 %r;
        { .reg .pred P1; L: setp.eq.u32 P1, %r, 0; @P1 bra DONE; bra L; DONE: }
        { .reg .pred P1; L: setp.eq.u32 P1, %r, 1; @P1 bra DONE; bra L; DONE: }
        bra L;
      L:
        ret;
    """)
    fn = mod.functions["k"]
    assert {"P1@1", "P1@2"} <= set(fn.regs) and fn.labels == {"L@1": 0, "DONE@1": 3, "L@2": 3,
                                                               "DONE@2": 6, "L": 7}
    assert [i.operands[0].name for i in fn.instrs if i.opcode == "bra"] == \
        ["DONE@1", "L@1", "DONE@2", "L@2", "L"]
    assert fn.instrs[1].pred == Reg("P1@1") and fn.instrs[4].pred == Reg("P1@2")


def test_tensor_operand_and_discarded_destination_parse():
    mod = kernel("""
        .reg .b64 %m; .reg .b32 %a, %c;
        cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%m, {%c, 4}], [%a];
        mbarrier.arrive.shared::cta.b64 _, [%a];
    """)
    store, arrive = mod.functions["k"].instrs
    assert isinstance(store.operands[0], TensorRef) and store.operands[0].base == Reg("%m")
    assert len(store.operands[0].coords) == 2 and arrive.operands[0] == Sym("_")
    assert store.mods[-1] == "bulk_group" and "shared::cta" in store.mods


def test_tensor_map_passed_by_value_in_a_struct_parameter(sim):
    # As CUTLASS does: the tensor map sits inside a kernel argument, and the kernel
    # takes that parameter's address with `mov` + `cvta.param`.
    data = np.arange(64, dtype=np.uint32)
    src, dst = sim.from_numpy(data), sim.empty(16, "u32")
    mod = kernel("""
        .reg .b64 %s, %d; .reg .b32 %buf, %bar; .reg .pred %p;
        .shared .align 1024 .b8 buf[64];
        .shared .align 8 .u64 bar;
        mov.b64 %s, atom; cvta.param.u64 %s, %s;
        add.u64 %s, %s, 128;
        ld.param.u64 %d, [dst];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 64;
        cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%s, {32}], [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%d, {0}], [%buf];
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
        ret;
    """, ".param .u32 pad, .param .align 128 .b8 atom[256], .param .u64 dst")
    atom = b"\0" * 128 + sim.tensor_map_tiled(src, 16).tobytes()  # tensor map at offset 128
    stats = sim.launch(mod, "k", grid=1, block=1, args=[7, atom, sim.tensor_map_tiled(dst, 16)])
    np.testing.assert_array_equal(dst.numpy(), data[32:48])
    assert not stats.findings
