"""The shared-memory race check orders only the warps that took part in a barrier, and
keeps every reader a later write could race with.

Each kernel here runs three warps (block=96). `%w0`, `%w1`, `%w2` select one warp and
`%w12` warps 1 and 2. Every kernel starts by writing `cell` and passing a full barrier, so
that nothing is uninitialized and the accesses under test are the only ones in question.
"""

import numpy as np
import pytest

from warplens.core.clocks import Clocks
from warplens.units.smem import SharedMemory

from tests.support import kernel

WARPS = """
    .reg .u32 %t, %w, %v; .reg .b32 %bar; .reg .pred %w0, %w1, %w2, %w12, %first, %p;
    .shared .align 8 .b8 mbar[8];
    .shared .u32 cell;
    mov.u32 %t, %tid.x; shr.u32 %w, %t, 5;
    setp.eq.u32 %first, %t, 0;
    setp.eq.u32 %w0, %w, 0; setp.eq.u32 %w1, %w, 1; setp.eq.u32 %w2, %w, 2;
    setp.ge.u32 %w12, %w, 1;
    mov.b32 %bar, mbar;
    @%first st.shared.u32 [cell], 1;
    @%first mbarrier.init.shared::cta.b64 [%bar], 32;
    bar.sync 0;
"""


def races(sim, body: str, **launch):
    stats = sim.launch(kernel(WARPS + body + "\n ret;"), "k", grid=1, block=96, **launch)
    assert {f.kind for f in stats.findings} <= {"smem-race"}, stats.findings
    return [(f.line, f.warp) for f in stats.findings]


# -- a barrier orders the warps that took part in it, and no others -------------------------

def test_counted_barrier_does_not_order_a_warp_that_did_not_arrive(sim):
    """Warps 1 and 2 meet at a 64-thread barrier. Warp 0's store is not ordered by it."""
    found = races(sim, """
        @%w0 st.shared.u32 [cell], 7;
        @%w12 bar.sync 1, 64;
        @%w1 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1 and found[0][1] == 1  # the load, in warp 1


def test_counted_barrier_orders_the_warps_that_arrived(sim):
    assert not races(sim, """
        @%w2 st.shared.u32 [cell], 7;
        @%w12 bar.sync 1, 64;
        @%w1 ld.shared.u32 %v, [cell];
    """)


def test_full_barrier_orders_every_warp(sim):
    assert not races(sim, """
        @%w0 st.shared.u32 [cell], 7;
        bar.sync 0;
        @%w1 ld.shared.u32 %v, [cell];
    """)


def test_bar_arrive_releases_but_does_not_acquire(sim):
    """Warp 1 only arrives: warp 2, which waits, is ordered after warp 1's store, but warp 1
    is not ordered after warp 2's."""
    assert not races(sim, """
        @%w1 st.shared.u32 [cell], 7;
        @%w1 bar.arrive 1, 64;
        @%w2 bar.sync 1, 64;
        @%w2 ld.shared.u32 %v, [cell];
    """)
    found = races(sim, """
        @%w2 st.shared.u32 [cell], 7;
        @%w1 bar.arrive 1, 64;
        @%w2 bar.sync 1, 64;
        @%w1 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1


def test_store_after_arriving_is_not_covered_by_the_arrival(sim):
    found = races(sim, """
        @%w1 bar.arrive 1, 64;
        @%w1 st.shared.u32 [cell], 7;
        @%w2 bar.sync 1, 64;
        @%w2 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1


def test_mbarrier_phase_does_not_order_a_warp_that_neither_arrived_nor_waited(sim):
    """Warp 1 completes an mbarrier phase and warp 2 waits for it. Warp 0 did neither, so
    its store and warp 2's load are unordered: the phase used to order the whole CTA."""
    found = races(sim, """
        @%w0 st.shared.u32 [cell], 7;
        @%w1 mbarrier.arrive.shared::cta.b64 _, [%bar];
        @%w2 mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        @%w2 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1 and found[0][1] == 2


def test_mbarrier_phase_orders_arrival_before_successful_wait(sim):
    assert not races(sim, """
        @%w1 st.shared.u32 [cell], 7;
        @%w1 mbarrier.arrive.shared::cta.b64 _, [%bar];
        @%w2 mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        @%w2 ld.shared.u32 %v, [cell];
    """)


def test_relaxed_mbarrier_arrival_orders_nothing(sim):
    found = races(sim, """
        @%w1 st.shared.u32 [cell], 7;
        @%w1 mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%bar];
        @%w2 mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        @%w2 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1


def test_full_barrier_counts_what_exited_warps_did(sim):
    """A warp that returned cannot arrive, and the barrier completes without it: its
    accesses are before the barrier, as they were when every completion ordered the CTA."""
    assert not races(sim, """
        @%w0 st.shared.u32 [cell], 7;
        @%w0 ret;
        bar.sync 0;
        @%w1 ld.shared.u32 %v, [cell];
    """)


def test_cluster_barrier_orders_the_ctas_that_met(sim):
    """One warp per CTA; CTA 0 writes its peer's cell and the peer reads it."""
    body = """
        .reg .u32 %rank, %v; .reg .u64 %a; .reg .pred %r0;
        .shared .u32 cell;
        mov.u32 %rank, %cluster_ctarank; setp.eq.u32 %r0, %rank, 0;
        st.shared.u32 [cell], 1;
        barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;
        mov.u64 %a, cell; mapa.shared::cluster.u64 %a, %a, 1;
        @%r0 st.shared::cluster.u32 [%a], 7;
        {barrier}
        @!%r0 ld.shared.u32 %v, [cell];
        ret;
    """
    both = "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;"
    for barrier, expect in ((both, 0), ("", 1)):
        stats = sim.launch(kernel(body.format(barrier=barrier)), "k", grid=2, block=32, cluster=2)
        assert [f.kind for f in stats.findings] == ["smem-race"] * expect


# -- every reader counts ---------------------------------------------------------------------

def test_write_races_with_an_earlier_reader_behind_the_writers_own_read(sim):
    """Warp 0 reads, warp 1 reads, warp 1 writes. The last reader is the writer itself; the
    race is with warp 0's read, which a single last reader per byte had forgotten."""
    found = races(sim, """
        @%w0 ld.shared.u32 %v, [cell];
        @%w1 ld.shared.u32 %v, [cell];
        @%w1 st.shared.u32 [cell], 7;
    """)
    assert len(found) == 1 and found[0][1] == 1


def test_write_races_with_a_reader_behind_an_ordered_reader(sim):
    """Warps 0 and 1 read. Warps 1 and 2 meet at a barrier, then warp 2 writes: ordered after
    warp 1's read, the last one, but not after warp 0's."""
    found = races(sim, """
        @%w0 ld.shared.u32 %v, [cell];
        @%w1 ld.shared.u32 %v, [cell];
        @%w12 bar.sync 1, 64;
        @%w2 st.shared.u32 [cell], 7;
    """)
    assert len(found) == 1 and found[0][1] == 2


def test_readers_ordered_before_the_write_are_no_race(sim):
    assert not races(sim, """
        ld.shared.u32 %v, [cell];
        bar.sync 0;
        @%w2 st.shared.u32 [cell], 7;
        bar.sync 0;
        ld.shared.u32 %v, [cell];
    """)


def test_atomic_races_with_an_earlier_reader_behind_another_reader(sim):
    found = races(sim, """
        @%w0 ld.shared.u32 %v, [cell];
        @%w1 ld.shared.u32 %v, [cell];
        @%w1 atom.shared.add.u32 %v, [cell], 1;
    """)
    assert len(found) == 1 and found[0][1] == 1


def test_load_races_with_an_earlier_atomic_behind_its_own(sim):
    found = races(sim, """
        @%w0 atom.shared.add.u32 %v, [cell], 1;
        @%w1 atom.shared.add.u32 %v, [cell], 1;
        @%w1 ld.shared.u32 %v, [cell];
    """)
    assert len(found) == 1 and found[0][1] == 1


# -- the shadow state itself -----------------------------------------------------------------

AT, U32 = np.array([0], dtype=np.uint64), np.dtype("u4")
ONE = np.array([1], dtype=U32)


def barrier(clocks: Clocks, *warps: int) -> None:
    released = None
    for w in warps:
        released = clocks.release(w, released)
    for w in warps:
        clocks.acquire(w, released)


def test_readers_in_sequence_keep_one_entry_and_unordered_ones_get_a_table():
    smem = SharedMemory(8, Clocks(4))
    smem.write(AT, ONE, 0)
    barrier(smem.clocks, 0, 1, 2, 3)
    smem.read(AT, U32, 0)
    barrier(smem.clocks, 0, 1)
    smem.read(AT, U32, 1)  # ordered after warp 0's read: replaces it
    assert smem._reads.table is None and smem._reads.who[0] == 1
    smem.read(AT, U32, 2)  # not ordered after warp 1's
    assert smem._reads.several[:4].all() and not smem._reads.several[4:].any()
    assert smem._reads.table[0].tolist() == [0, 3, 2, 0]  # each warp's clock at its read
    assert smem.write(AT, ONE, 2).tolist() == [0]  # races with warp 1's read
    assert not smem._reads.several.any()  # a store starts the record again


def test_write_ordered_after_all_readers_is_clean_and_one_missing_reader_is_not():
    for missing in (None, 0, 1, 2):
        smem = SharedMemory(4, Clocks(4))
        smem.write(AT, ONE, 3)
        barrier(smem.clocks, 0, 1, 2, 3)
        for reader in (0, 1, 2):
            smem.read(AT, U32, reader)
        barrier(smem.clocks, *(w for w in (0, 1, 2, 3) if w != missing))
        assert smem.write(AT, ONE, 3).tolist() == ([] if missing is None else [0])


def test_clocks_grow_with_the_warps_that_use_them():
    smem = SharedMemory(4)
    smem.write(AT, ONE, 0)
    assert smem.read(AT, U32, 5)[2].tolist() == [0]
    assert smem.read(AT, U32, 9)[2].tolist() == [0]  # the reads table widens too
    assert smem.write(AT, ONE, 0).tolist() == [0]
    barrier(smem.clocks, 0, 9)
    assert smem.read(AT, U32, 9)[2].tolist() == []


@pytest.mark.parametrize("ordered", [False, True])
def test_proxy_fence_covers_only_writes_ordered_before_the_fencing_warp(ordered):
    smem = SharedMemory(4, Clocks(3))
    smem.write(AT, ONE, 0)
    barrier(smem.clocks, 0, 1) if ordered else barrier(smem.clocks, 1, 2)
    smem.fence_async(1)
    assert smem.unfenced.any() != ordered
