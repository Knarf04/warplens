"""Shared memory, bank conflicts, named barriers, shuffles, deadlock."""


import numpy as np
import pytest

import warplens
from warplens.units.smem_banks import analyze

from tests.support import AMPERE, EXAMPLES, HEADER, kernel

TRANSPOSE = EXAMPLES / "01_smem_transpose" / "transpose.ptx"
REDUCE = EXAMPLES / "01_warp_reduce" / "reduce.ptx"


# -- M1 acceptance -------------------------------------------------------------------

@pytest.mark.parametrize("layout, smem_bytes, read_way, conflicts_per_cta", [
    ("linear", 4096, 32, 32 * 31),
    ("padded", 4224, 1, 0),
    ("swizzled", 4096, 1, 0),
])
def test_transpose_bank_conflicts(sim, layout, smem_bytes, read_way, conflicts_per_cta):
    ctas = 3
    data = np.random.default_rng(0).standard_normal((ctas, 32, 32)).astype(np.float32)
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(TRANSPOSE, f"transpose_{layout}", grid=ctas, block=(32, 32),
                       args=[sim.from_numpy(data), out])
    np.testing.assert_array_equal(out.numpy(), data.transpose(0, 2, 1))
    assert stats.smem_bytes == smem_bytes and not stats.findings
    store, load = sorted(stats.smem.instructions.values(), key=lambda e: e.line)
    assert (store.op, load.op) == ("st", "ld")
    assert store.executions == load.executions == ctas * 32
    assert store.worst_way == 1 and store.conflicts == 0  # rows are always conflict-free
    assert load.worst_way == read_way
    assert stats.smem.conflicts == ctas * conflicts_per_cta
    assert ("32-way" in stats.report()) == (layout == "linear")


def test_warp_reduce_with_dynamic_smem(sim):
    x = np.random.default_rng(1).integers(-1000, 1000, (6, 256)).astype(np.int32)
    out = sim.empty(6, "s32")
    stats = sim.launch(REDUCE, "block_sum", grid=6, block=256, smem=8 * 4,
                       args=[sim.from_numpy(x), out])
    np.testing.assert_array_equal(out.numpy(), x.sum(axis=1))
    assert stats.smem_bytes == 32 and not stats.findings


def test_dynamic_smem_too_small_faults(sim):
    x = sim.from_numpy(np.ones(256, dtype=np.int32))
    with pytest.raises(warplens.MemoryFault, match=r"offset 16 is outside the CTA's 16 bytes"):
        sim.launch(REDUCE, "block_sum", grid=1, block=256, smem=16, args=[x, sim.empty(1, "s32")])


def test_smem_limit(sim):
    x = sim.empty(256, "s32")
    with pytest.raises(warplens.LaunchError, match="the limit is 232448"):
        sim.launch(REDUCE, "block_sum", grid=1, block=256, smem=300_000, args=[x, x])


# -- bank model ----------------------------------------------------------------------

def offsets(*v):
    return np.array(v, dtype=np.uint64)


def test_bank_analysis():
    row = np.arange(32, dtype=np.uint64) * 4
    assert analyze(row, 4)[:2] == (1, 1)
    assert analyze(row * 32, 4)[:2] == (32, 1)  # one column: same bank, 32 words
    assert analyze(row * 2, 4)[:2] == (2, 1)  # stride 2: 16 banks, 2 words each
    assert analyze(offsets(8, 8, 8, 8), 4)[:2] == (1, 1)  # same word is not a conflict
    assert analyze(offsets(0, 128), 4)[:2] == (2, 1)
    # 8-byte accesses by 32 lanes cover 64 words: two wavefronts even without conflicts.
    assert analyze(np.arange(32, dtype=np.uint64) * 8, 8)[:2] == (2, 2)
    assert analyze(offsets(0, 1, 2, 3), 1)[:2] == (1, 1)  # bytes within one word
    assert analyze(offsets(2), 4)[:2] == (1, 1)  # unaligned: straddles two words, two banks
    assert analyze(offsets(2), 4)[2].tolist()[:3] == [1, 1, 0]


# -- address spaces --------------------------------------------------------------------

SHARED_PARAMS = ".param .u64 out"


def test_static_layout_respects_alignment_and_generic_window(sim):
    mod = kernel("""
        .reg .b32 %a, %b, %v; .reg .b64 %g, %o;
        .shared .align 4 .b8 small[5];
        .shared .align 16 .f32 big[4];
        mov.b32 %a, small; mov.b32 %b, big;
        ld.param.u64 %o, [out];
        st.global.u32 [%o], %a;
        st.global.u32 [%o+4], %b;
        st.shared.u32 [big+4], 77;
        cvt.u64.u32 %g, %b;
        cvta.shared.u64 %g, %g;
        ld.u32 %v, [%g+4];
        st.global.u32 [%o+8], %v;
        cvta.to.shared.u64 %g, %g;
        st.global.u64 [%o+16], %g;
        ret;
    """, SHARED_PARAMS)
    out = sim.empty(6, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [0, 16, 77, 0, 16, 0]
    assert stats.smem_bytes == 32


def test_module_level_and_extern_shared(sim):
    mod = warplens.parse_ptx(HEADER + """
        .shared .align 8 .u64 counter;
        .extern .shared .align 16 .b8 dyn[];
        .visible .entry k(.param .u64 out) {
            .reg .b32 %a, %b; .reg .b64 %o;
            mov.b32 %a, counter; mov.b32 %b, dyn;
            ld.param.u64 %o, [out];
            st.global.u32 [%o], %a;
            st.global.u32 [%o+4], %b;
            ret;
        }""")
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, smem=100, args=[out])
    assert out.numpy().tolist() == [0, 16] and stats.smem_bytes == 116


def test_each_cta_has_its_own_shared_memory(sim):
    # Only CTA 0 writes; CTA 1 then reads the same offset and must not see the value.
    mod = kernel("""
        .reg .pred %p; .reg .b32 %c, %v; .reg .b64 %o;
        .shared .align 4 .u32 cell;
        mov.u32 %c, %ctaid.x;
        setp.ne.u32 %p, %c, 0;
        @!%p st.shared.u32 [cell], 99;
        ld.shared.u32 %v, [cell];
        ld.param.u64 %o, [out];
        cvt.u64.u32 %o, %c;
        ret;
    """, SHARED_PARAMS)
    stats = sim.launch(mod, "k", grid=2, block=1, args=[sim.empty(1, "u32")])
    (f,) = stats.findings
    assert f.kind == "smem-uninitialized-read" and f.cta == (1, 0, 0) and f.line == 12
    assert "never wrote" in str(f)


def test_shared_access_out_of_bounds(sim):
    mod = kernel(".reg .b32 %v;\n.shared .align 4 .u32 cell[4];\nld.shared.u32 %v, [cell+16];\nret;")
    with pytest.raises(warplens.MemoryFault, match=r"offset 16 is outside the CTA's 16 bytes.*\n.*test.ptx:8"):
        sim.launch(mod, "k", grid=1, block=1)


def test_global_load_cannot_reach_shared_memory(sim):
    mod = kernel(".reg .b32 %v;\n.shared .align 4 .u32 cell;\nld.global.u32 %v, [cell];\nret;")
    with pytest.raises(warplens.MemoryFault, match="null pointer"):
        sim.launch(mod, "k", grid=1, block=1)


# -- barriers ------------------------------------------------------------------------

EXCHANGE = """
    .reg .pred %p; .reg .b32 %t, %a, %v, %n; .reg .b64 %o, %off;
    .shared .align 4 .u32 box[128];
    mov.u32 %t, %tid.x;
    shl.b32 %a, %t, 2;
    st.shared.u32 [%a], %t;
    {barrier}
    mov.u32 %n, %ntid.x;
    sub.u32 %v, %n, %t;
    sub.u32 %v, %v, 1;
    shl.b32 %a, %v, 2;
    ld.shared.u32 %v, [%a];
    ld.param.u64 %o, [out];
    mul.wide.u32 %off, %t, 4;
    add.u64 %o, %o, %off;
    st.global.u32 [%o], %v;
    ret;
"""


def test_barrier_orders_writes_before_reads_across_warps(sim):
    # Thread t reads the slot written by thread n-1-t, which is in another warp.
    out = sim.empty(128, "u32")
    stats = sim.launch(kernel(EXCHANGE.format(barrier="bar.sync 0;"), SHARED_PARAMS), "k",
                       grid=1, block=128, args=[out])
    assert out.numpy().tolist() == list(range(127, -1, -1)) and not stats.findings


def test_missing_barrier_is_reported_as_a_race(sim):
    # The round-robin schedule happens to give the right answer; the race is still there.
    out = sim.empty(128, "u32")
    stats = sim.launch(kernel(EXCHANGE.format(barrier=""), SHARED_PARAMS), "k",
                       grid=1, block=128, args=[out])
    (f,) = stats.findings
    assert f.kind == "smem-race" and f.count == 4 and "ld.shared.u32" in f.text
    assert "no barrier in between" in str(f)


def test_write_write_and_read_write_races(sim):
    # Every warp stores to slot 0: write-write. Then warps read slot 1 and warp 0 overwrites it.
    mod = kernel("""
        .reg .pred %p; .reg .b32 %l, %w, %v;
        .shared .align 4 .u32 cell[2];
        mov.u32 %l, %laneid; mov.u32 %w, %warpid;
        setp.eq.u32 %p, %l, 0;
        @%p st.shared.u32 [cell], %w;
        bar.sync 0;
        @%p st.shared.u32 [cell+4], 5;
        bar.sync 0;
        @%p ld.shared.u32 %v, [cell+4];
        setp.eq.u32 %p, %w, 0;
        @%p st.shared.u32 [cell+4], 6;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=64)
    assert [(f.kind, f.line) for f in stats.findings] == \
        [("smem-race", 11), ("smem-race", 13), ("smem-race", 17)]
    assert "write to shared memory" in stats.findings[0].message


def test_same_warp_and_barrier_separated_accesses_do_not_race(sim):
    mod = kernel("""
        .reg .b32 %v;
        .shared .align 4 .u32 cell;
        st.shared.u32 [cell], 1;
        ld.shared.u32 %v, [cell];
        st.shared.u32 [cell], 2;
        ret;
    """)
    assert not sim.launch(mod, "k", grid=1, block=32).findings


def test_barrier_with_thread_count_and_arrive(sim):
    # Warps 0 and 1 sync on barrier 1 with a count of 64; warps 2 and 3 never touch it.
    mod = kernel("""
        .reg .pred %p; .reg .b32 %t; .reg .b64 %o, %off;
        mov.u32 %t, %tid.x;
        setp.ge.u32 %p, %t, 64;
        @%p bra DONE;
        barrier.cta.sync 1, 64;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %t, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], 1;
      DONE:
        ret;
    """, SHARED_PARAMS)
    out = sim.empty(128, "u32")
    sim.launch(mod, "k", grid=1, block=128, args=[out])
    assert out.numpy().tolist() == [1] * 64 + [0] * 64

    # One warp waits for 64 threads; the other arrives without waiting.
    mod = kernel("""
        .reg .pred %p; .reg .b32 %t;
        mov.u32 %t, %tid.x;
        setp.ge.u32 %p, %t, 32;
        @%p bra ARRIVE;
        bar.sync 2, 64;
        ret;
      ARRIVE:
        bar.arrive 2, 64;
        ret;
    """)
    sim.launch(mod, "k", grid=1, block=64)


def test_deadlock_is_reported_with_who_waits_where(sim):
    # Barrier 3 needs 128 threads but the CTA only has 64.
    mod = kernel("bar.sync 3, 128;\nret;")
    with pytest.raises(warplens.DeadlockError) as info:
        sim.launch(mod, "k", grid=1, block=64)
    msg = str(info.value)
    assert "deadlock in CTA (0, 0, 0)" in msg
    assert "barrier 3: 2 warp(s) arrived, needs 128 threads" in msg
    assert "warp 1 waits at line 6: bar.sync 3, 128;" in msg


def test_deadlock_between_two_barriers(sim):
    mod = kernel("""
        .reg .pred %p; .reg .b32 %t;
        mov.u32 %t, %tid.x;
        setp.ge.u32 %p, %t, 32;
        @%p bra OTHER;
        bar.sync 0;
        ret;
      OTHER:
        bar.sync 1;
        ret;
    """)
    with pytest.raises(warplens.DeadlockError, match=r"barrier 0: 1 warp\(s\) arrived, needs all 2 running"):
        sim.launch(mod, "k", grid=1, block=64)


@pytest.mark.parametrize("which", [0, 1])
def test_barrier_completes_when_threads_already_returned(sim, which):
    """Matches the real GPU: validation/ampere/barrier_exit_hw.cu completes in both cases."""
    out = sim.empty(128, "s32")
    stats = sim.launch(AMPERE / "barrier_exit.ptx", "barrier_exit", grid=1, block=128, args=[out, which])
    exited = [t >= 64 if which == 0 else bool(t & 1) for t in range(128)]
    assert out.numpy().tolist() == [0 if exited[t] else (t ^ 2) + 1000 for t in range(128)]
    # The pending path exits before the barrier counts the warp. Exited lanes
    # are exempt, so this is not a divergent-barrier violation.
    assert not stats.findings


def test_barrier_number_out_of_range(sim):
    with pytest.raises(warplens.ExecutionError, match="barrier number 16 is out of range"):
        sim.launch(kernel("bar.sync 16;\nret;"), "k", grid=1, block=32)


# -- shuffles ------------------------------------------------------------------------

def shuffle(sim, instr, n=32, prologue=""):
    mod = kernel(f"""
        .reg .pred %p, %q; .reg .b32 %t, %v, %r, %f; .reg .b64 %o, %off;
        mov.u32 %t, %tid.x;
        add.u32 %v, %t, 100;
        mov.u32 %r, 7;
        {prologue}
        {instr}
        selp.u32 %f, 1, 0, %p;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %t, 8;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %r;
        st.global.u32 [%o+4], %f;
        ret;
    """, SHARED_PARAMS)
    out = sim.empty((n, 2), "u32")
    stats = sim.launch(mod, "k", grid=1, block=n, args=[out])
    return out.numpy()[:, 0].tolist(), out.numpy()[:, 1].tolist(), stats


def test_shfl_predicate_reports_out_of_range_sources(sim):
    value, valid, stats = shuffle(sim, "shfl.sync.down.b32 %r|%p, %v, 4, 31, -1;")
    assert value == [104 + t for t in range(28)] + [128, 129, 130, 131]
    assert valid == [1] * 28 + [0] * 4 and not stats.findings
    value, valid, _ = shuffle(sim, "shfl.sync.up.b32 %r|%p, %v, 1, 0, -1;")
    assert value == [100] + [100 + t for t in range(31)] and valid == [0] + [1] * 31


def test_shfl_segments(sim):
    # c = ((32 - 8) << 8) | 31: segments of 8 lanes, as __shfl_sync(mask, v, 1, 8) emits.
    value, valid, _ = shuffle(sim, "shfl.sync.idx.b32 %r|%p, %v, 1, 0x181f, -1;")
    assert value == [100 + (t & ~7) + 1 for t in range(32)] and valid == [1] * 32


def test_shfl_without_predicate_destination(sim):
    value, _, _ = shuffle(sim, "shfl.sync.bfly.b32 %r, %v, 1, 31, -1;", prologue="setp.eq.u32 %p, %t, %t;")
    assert value == [100 + (t ^ 1) for t in range(32)]


def test_shfl_from_inactive_lane_is_reported(sim):
    # Only even lanes execute the shuffle, but each reads its odd neighbour.
    value, _, stats = shuffle(sim, """
        and.b32 %f, %t, 1;
        setp.eq.u32 %q, %f, 0;
        @%q shfl.sync.bfly.b32 %r|%p, %v, 1, 31, 0x55555555;
    """)
    assert [f.kind for f in stats.findings] == ["shfl-inactive-source"]
    assert stats.findings[0].lanes == list(range(0, 32, 2))
    assert value[1::2] == [7] * 16  # lanes that did not execute keep their old value


def test_shfl_membermask_mismatch_is_reported(sim):
    _, _, stats = shuffle(sim, "shfl.sync.bfly.b32 %r|%p, %v, 1, 31, 0xffff;")
    assert [f.kind for f in stats.findings] == ["shfl-membermask-mismatch"]
