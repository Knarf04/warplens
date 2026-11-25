"""Named-barrier phase boundaries and operand validation."""

import pytest

import warplens

from tests.support import kernel


@pytest.mark.parametrize("operation", ["sync", "arrive"])
def test_counted_barrier_resets_before_later_arrivals_in_the_same_round(sim, operation):
    # Three warps reach this 64-thread barrier together, then the fourth
    # arrives later. The first two complete phase 0; the last two complete
    # phase 1. Batching three arrivals into phase 0 strands the fourth warp.
    mod = kernel(f"""
        .reg .pred %p;
        .reg .u32 %w, %n;
        mov.u32 %w, %warpid;
        setp.eq.u32 %p, %w, 3;
        @%p bra LATE;
        bra SYNC;
      LATE:
        mov.u32 %n, 0;
        add.u32 %n, %n, 1;
        add.u32 %n, %n, 1;
        add.u32 %n, %n, 1;
        barrier.sync 0, 64;
        ret;
      SYNC:
        barrier.{operation} 0, 64;
        ret;
    """)
    sim.launch(mod, "k", grid=1, block=128)


@pytest.mark.parametrize("count", ["0", "%count"])
def test_zero_sync_count_waits_for_the_entire_cta(sim, count):
    mod = kernel(f"""
        .reg .pred %p;
        .reg .u32 %w, %v, %count;
        .reg .u64 %out;
        .shared .align 4 .u32 cell;
        mov.u32 %w, %warpid;
        mov.u32 %count, 0;
        setp.eq.u32 %p, %w, 1;
        @%p bra WRITE;
        barrier.sync 0, {count};
        ld.shared.u32 %v, [cell];
        ld.param.u64 %out, [out];
        st.global.u32 [%out], %v;
        ret;
      WRITE:
        mov.u32 %v, 40;
        add.u32 %v, %v, 1;
        add.u32 %v, %v, 1;
        st.shared.u32 [cell], %v;
        barrier.sync 0;
        ret;
    """, ".param .u64 out")
    out = sim.empty(1, "u32")
    stats = sim.launch(mod, "k", grid=1, block=64, args=[out])
    assert out.numpy().tolist() == [42]
    assert not stats.findings


@pytest.mark.parametrize("operation", ["sync", "arrive"])
@pytest.mark.parametrize("count", [1, 31, 33])
@pytest.mark.parametrize("register", [False, True])
def test_barrier_rejects_counts_that_are_not_warp_multiples(sim, operation, count, register):
    body = f".reg .u32 %n; mov.u32 %n, {count};" if register else ""
    body += f"bar.{operation} 0, {'%n' if register else count}; ret;"
    with pytest.raises(warplens.ExecutionError, match=f"count {count} must be a multiple of 32") as exc:
        sim.launch(kernel(body), "k", grid=1, block=64)
    assert "test.ptx" in str(exc.value)


@pytest.mark.parametrize("count", ["0", "%count"])
def test_arrive_rejects_zero_count(sim, count):
    mod = kernel(f".reg .u32 %count; mov.u32 %count, 0; bar.arrive 0, {count}; ret;")
    with pytest.raises(warplens.ExecutionError, match="requires a non-zero barrier thread count"):
        sim.launch(mod, "k", grid=1, block=64)


@pytest.mark.parametrize("scope", ["", ".cta"])
@pytest.mark.parametrize("operation", ["sync", "arrive"])
def test_explicit_aligned_barrier_alias(sim, scope, operation):
    mod = kernel(f"barrier{scope}.{operation}.aligned 0, 64; ret;")
    assert not sim.launch(mod, "k", grid=1, block=64).findings


def test_repeated_arrival_does_not_count_as_another_warp(sim):
    mod = kernel("""
        .reg .pred %p;
        .reg .u32 %w;
        mov.u32 %w, %warpid;
        setp.eq.u32 %p, %w, 0;
        @%p bra PRODUCER;
        barrier.sync 0, 96;
        ret;
      PRODUCER:
        barrier.arrive 0, 96;
        barrier.arrive 0, 96;
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="arrived at barrier 0 more than once") as exc:
        sim.launch(mod, "k", grid=1, block=64)
    assert "test.ptx" in str(exc.value)


@pytest.mark.parametrize("counts", [(64, 128), (128, 64)])
def test_explicit_thread_counts_must_agree_within_a_phase(sim, counts):
    mod = kernel(f"""
        .reg .pred %p;
        .reg .u32 %w, %count;
        mov.u32 %w, %warpid;
        setp.eq.u32 %p, %w, 0;
        selp.u32 %count, {counts[0]}, {counts[1]}, %p;
        barrier.sync 0, %count;
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="differs from the current phase's thread count"):
        sim.launch(mod, "k", grid=1, block=64)


def test_thread_count_may_change_after_a_phase_completes(sim):
    mod = kernel("barrier.sync 0, 32; barrier.sync 0, 64; ret;")
    sim.launch(mod, "k", grid=1, block=64)


@pytest.mark.parametrize("producer", [0, 1])
def test_explicit_arrive_can_mix_with_implicit_full_cta_sync(sim, producer):
    mod = kernel(f"""
        .reg .pred %p;
        .reg .u32 %w;
        mov.u32 %w, %warpid;
        setp.eq.u32 %p, %w, {producer};
        @%p bra PRODUCER;
        barrier.sync 0;
        ret;
      PRODUCER:
        barrier.arrive 0, 64;
        ret;
    """)
    sim.launch(mod, "k", grid=1, block=64)
