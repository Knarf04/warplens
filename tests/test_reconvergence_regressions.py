"""Warp rendezvous at synchronizing instructions beyond a static postdominator."""

import numpy as np
import pytest

import warplens

from tests.support import kernel
from tests.support.probes import reconvergence


@pytest.mark.parametrize("shape", ["C", "D"])
@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("barrier", ["bar.sync 0;", "bar.cta.sync 0;",
                                    "barrier.cta.sync 0;", "barrier.sync.aligned 0;",
                                    "bar.arrive 0, 32;"])
def test_named_barrier_collects_paths_before_counting_warp(shape, timing, barrier):
    repro = reconvergence()
    sim = warplens.Device("B200")
    out = sim.empty(32, "u32")
    trace = []
    mod = warplens.parse_ptx(repro.HEADER + repro.SHAPES[shape].replace("BARRIER", barrier))
    stats = sim.launch(mod, "k", grid=1, block=32, args=[out], timing=timing,
                       trace=lambda w, pc, ins, mask: trace.append((ins.name, int(mask.sum()))))
    np.testing.assert_array_equal(out.numpy(), np.full(32, 7, dtype=np.uint32))
    assert not stats.findings
    arrivals = [n for op, n in trace if op.startswith(("bar.", "barrier."))]
    assert arrivals == [32]  # Deferred arrivals are not extra issued instructions.
    assert stats.instructions == len(trace)
    assert stats.lane_instructions == sum(n for _, n in trace)


@pytest.mark.parametrize("timing", [False, True])
def test_multiwarp_arrive_waits_for_all_its_lanes_before_releasing_consumer(timing):
    # The producer's late lane writes the data. All producer lanes must reach
    # bar.arrive before the waiting consumer warp is allowed to load it.
    sim = warplens.Device("B200")
    mod = kernel("""
        .reg .pred %p, %q; .reg .b32 %tid, %lane, %v; .reg .b64 %out;
        .shared .align 4 .u32 cell;
        mov.u32 %tid, %tid.x;
        mov.u32 %lane, %laneid;
        setp.ge.u32 %p, %tid, 32;
        @%p bra CONSUMER;
        setp.ne.u32 %p, %lane, 0;
        @%p bra EARLY;
        st.shared.u32 [cell], 91;
        bra JOIN;
      EARLY:
        setp.ge.u32 %q, %lane, 32;
        @%q bra END;
      JOIN:
        bar.arrive 0, 64;
        bar.sync 1, 64;
        ret;
      CONSUMER:
        bar.sync 0, 64;
        ld.shared.u32 %v, [cell];
        ld.param.u64 %out, [out];
        st.global.u32 [%out], %v;
        bar.arrive 1, 64;
      END:
        ret;
    """, ".param .u64 out")
    out = sim.empty(1, "u32")
    stats = sim.launch(mod, "k", grid=1, block=64, args=[out], timing=timing)
    assert out.numpy().tolist() == [91]
    assert not stats.findings


@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("threads", [32, 64, 40])
def test_repeated_nested_rendezvous_preserves_per_lane_work_and_barrier_phases(timing, threads):
    sim = warplens.Device("B200")
    # Nested branches have a possible edge past JOIN, but every actual path
    # reaches the barrier. Each lane increments its value once per iteration.
    mod = kernel("""
        .reg .pred %p, %q; .reg .b32 %tid, %v, %i, %tmp;
        .reg .b64 %out, %offset;
        mov.u32 %tid, %tid.x;
        mov.u32 %v, 0;
        mov.u32 %i, 0;
      LOOP:
        and.b32 %tmp, %tid, 1;
        setp.ne.u32 %p, %tmp, 0;
        @%p bra ODD;
        and.b32 %tmp, %tid, 2;
        setp.ne.u32 %p, %tmp, 0;
        @%p bra EVEN2;
        add.u32 %v, %v, 1;
        bra JOIN;
      EVEN2:
        add.u32 %v, %v, 3;
        setp.gt.u32 %q, %tid, 128;
        @%q bra END;
        bra JOIN;
      ODD:
        add.u32 %v, %v, 2;
        setp.gt.u32 %q, %tid, 128;
        @%q bra END;
      JOIN:
        bar.sync 0;
        add.u32 %i, %i, 1;
        setp.lt.u32 %p, %i, 5;
        @%p bra LOOP;
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %tid, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %v;
      END:
        ret;
    """, ".param .u64 out")
    out = sim.empty(threads, "u32")
    stats = sim.launch(mod, "k", grid=1, block=threads, args=[out], timing=timing,
                       max_steps=1000)
    assert out.numpy().tolist() == [5 * (2 if t & 1 else 3 if t & 2 else 1) for t in range(threads)]
    assert not stats.findings


@pytest.mark.parametrize("timing", [False, True])
def test_tcgen05_allocation_synchronizes_the_compiler_merged_paths(timing):
    repro = reconvergence()
    # The .sync operation itself must collect all 32 lanes. No extra bar.sync
    # is required merely to make the interpreter reach the allocation together.
    text = repro.SHAPES["C"].replace("BARRIER", """
        tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tp], 32;
        ld.shared.u32 %ptr, [tp];
        tcgen05.dealloc.cta_group::1.sync.aligned.b32 %ptr, 32;
    """).replace(".shared .align 4 .b8 x[4];", """
        .shared .align 4 .b8 x[4];
        .shared .align 4 .u32 tp;
        .reg .b32 %ptr;
    """)
    sim = warplens.Device("B200")
    out = sim.empty(32, "u32")
    stats = sim.launch(warplens.parse_ptx(repro.HEADER + text), "k", grid=1, block=32,
                       args=[out], timing=timing)
    assert out.numpy().tolist() == [7] * 32
    assert not stats.findings


def test_different_instructions_within_one_warp_are_explicitly_unsupported(sim):
    # Unlike .aligned bar.sync, non-aligned barrier.sync may be reached from
    # different instruction addresses. This is still outside our same-PC model.
    mod = kernel("""
        .reg .pred %p; .reg .b32 %tid;
        mov.u32 %tid, %tid.x;
        setp.ge.u32 %p, %tid, 16;
        @%p bra OTHER;
        barrier.sync 0;
        ret;
      OTHER:
        barrier.sync 0;
        ret;
    """)
    with pytest.raises(warplens.UnsupportedError, match="different synchronizing instructions"):
        sim.launch(mod, "k", grid=1, block=32)


@pytest.mark.parametrize("timing", [False, True])
def test_recursive_rendezvous_preserves_activation_state(timing):
    # Every recursive activation enters uniformly, then its lanes diverge.
    # The possible base-case edge makes the static join later than bar.sync.
    # Rebuilding that activation's SIMT region must preserve its saved %n.
    text = """
.version 9.5
.target sm_100a
.address_size 64
.func (.param .b32 result) factorial(.param .b32 argument)
{
    .reg .u32 %n, %value, %tid, %odd; .reg .pred %p;
    ld.param.u32 %n, [argument];
    setp.gt.u32 %p, %n, 1;
    @%p bra MORE;
  BASE:
    st.param.u32 [result], 1;
    ret;
  MORE:
    mov.u32 %tid, %tid.x;
    and.b32 %odd, %tid, 1;
    setp.ne.u32 %p, %odd, 0;
    @%p bra ODD;
    mov.u32 %value, 0;
    bra JOIN;
  ODD:
    setp.gt.u32 %p, %tid, 1000;
    @%p bra BASE;
  JOIN:
    bar.sync 0;
    sub.u32 %value, %n, 1;
    {
        .param .b32 child_arg; .param .b32 child_result;
        st.param.u32 [child_arg], %value;
        call.uni (child_result), factorial, (child_arg);
        ld.param.u32 %value, [child_result];
    }
    mul.lo.u32 %value, %value, %n;
    st.param.u32 [result], %value;
    ret;
}
.visible .entry k(.param .u64 out)
{
    .reg .u32 %value, %tid; .reg .u64 %out, %offset;
    ld.param.u64 %out, [out];
    mov.u32 %tid, %tid.x;
    mul.wide.u32 %offset, %tid, 4;
    add.u64 %out, %out, %offset;
    {
        .param .b32 argument; .param .b32 result;
        st.param.u32 [argument], 5;
        call.uni (result), factorial, (argument);
        ld.param.u32 %value, [result];
    }
    st.global.u32 [%out], %value;
    ret;
}
"""
    sim = warplens.Device("B200")
    out = sim.empty(64, "u32")
    barriers = []
    def record(w, pc, ins, mask):
        if ins.name == "bar.sync":
            barriers.append(int(mask.sum()))
    stats = sim.launch(warplens.parse_ptx(text), "k", grid=1, block=64, args=[out],
                       timing=timing, max_steps=10000, trace=record)
    assert out.numpy().tolist() == [120] * 64
    assert barriers == [32] * 8
    assert not stats.findings


@pytest.mark.parametrize("barrier", ["bar.sync", "barrier.sync.aligned", "barrier.sync"])
def test_guard_false_lanes_exit_before_partial_arrival_is_counted(sim, barrier):
    mod = kernel(f"""
        .reg .b32 %lane; .reg .pred %one;
        mov.u32 %lane, %laneid;
        setp.eq.u32 %one, %lane, 0;
        @%one {barrier} 0;
        ret;
    """)
    trace = []
    stats = sim.launch(mod, "k", grid=1, block=32,
                       trace=lambda w, pc, ins, mask: trace.append((ins.name, int(mask.sum()))))
    assert trace[-3:] == [("ret", 31), (barrier, 1), ("ret", 1)]
    # For aligned forms the unequal predicate is undefined even though the
    # interpreter can let the remaining threads exit and finish the barrier.
    assert [f.kind for f in stats.findings] == ([] if barrier == "barrier.sync" else
                                              ["barrier-in-divergent-code"])


@pytest.mark.parametrize("timing", [False, True])
def test_cyclic_early_join_still_requires_explicit_barrier_rendezvous(sim, timing):
    # CFG regrouping deliberately leaves paths in the same cycle in their old
    # order. It must not replace the barrier's rendezvous: without it lanes1–31
    # run both iterations before lane0 ever initializes the shared word.
    mod = kernel("""
        .reg .pred %p, %q; .reg .b32 %tid, %i, %v;
        .reg .b64 %out, %off; .shared .align 4 .u32 cell;
        mov.u32 %tid, %tid.x;
        mov.u32 %i, 0;
      LOOP:
        setp.ne.u32 %p, %tid, 0;
        @%p bra ELSE;
        st.shared.u32 [cell], 7;
        bra JOIN;
      ELSE:
        setp.gt.u32 %q, %tid, 31;
        @%q bra END;
      JOIN:
        bar.sync 0;
        ld.shared.u32 %v, [cell];
        ld.param.u64 %out, [out];
        mul.wide.u32 %off, %tid, 4;
        add.u64 %out, %out, %off;
        st.global.u32 [%out], %v;
        add.u32 %i, %i, 1;
        setp.lt.u32 %q, %i, 2;
        @%q bra LOOP;
      END:
        ret;
    """, ".param .u64 out")
    out = sim.empty(32, "u32")
    arrivals = []
    def record(w, pc, ins, mask):
        if ins.name == "bar.sync":
            arrivals.append(int(mask.sum()))
    stats = sim.launch(mod, "k", grid=1, block=32, args=[out], timing=timing,
                       trace=record, max_steps=200)
    assert out.numpy().tolist() == [7] * 32
    assert arrivals == [32, 32]
    assert not stats.findings
