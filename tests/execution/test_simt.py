"""SIMT divergence and reconvergence on hand-written control flow."""

import numpy as np
import pytest

from tests.support import kernel

PARAMS = ".param .u64 out"
PROLOGUE = """
    .reg .pred %p, %q; .reg .b32 %t, %v, %i; .reg .b64 %base, %off, %addr;
    mov.u32 %t, %tid.x;
    ld.param.u64 %base, [out];
    mul.wide.u32 %off, %t, 4;
    add.u64 %addr, %base, %off;
"""


def launch(sim, body, n=32, **kw):
    out = sim.from_numpy(np.full(n, 0xDEAD, dtype=np.uint32))
    stats = sim.launch(kernel(PROLOGUE + body, PARAMS), "k", grid=1, block=n, args=[out], **kw)
    return out.numpy(), stats


def test_if_else(sim):
    out, stats = launch(sim, """
        and.b32 %v, %t, 1;
        setp.eq.u32 %p, %v, 0;
        @%p bra EVEN;
        mov.u32 %v, 111;
        bra JOIN;
      EVEN:
        mov.u32 %v, 222;
      JOIN:
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert out.tolist() == [222, 111] * 16
    assert stats.divergent_branches == 1 and stats.max_simt_depth == 3


def test_uniform_branch_does_not_diverge(sim):
    out, stats = launch(sim, """
        setp.lt.u32 %p, %t, 1000;
        @%p bra T;
        mov.u32 %v, 1;
        st.global.u32 [%addr], %v;
      T:
        mov.u32 %v, 2;
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert (out == 2).all() and stats.divergent_branches == 0 and stats.max_simt_depth == 1


def test_nested_divergence(sim):
    # v = 0..3 by (tid & 1, tid & 2), through two levels of if/else.
    out, stats = launch(sim, """
        and.b32 %v, %t, 1;
        setp.ne.u32 %p, %v, 0;
        and.b32 %v, %t, 2;
        setp.ne.u32 %q, %v, 0;
        @%p bra ODD;
        @%q bra E2;
        mov.u32 %v, 0;
        bra EJ;
      E2:
        mov.u32 %v, 2;
      EJ:
        bra JOIN;
      ODD:
        @%q bra O2;
        mov.u32 %v, 1;
        bra JOIN;
      O2:
        mov.u32 %v, 3;
      JOIN:
        add.u32 %v, %v, 10;
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert out.tolist() == [10 + (t & 3) for t in range(32)]
    assert stats.divergent_branches == 3


def test_loop_with_per_lane_trip_count_keeps_stack_flat(sim):
    # Lane t iterates t times; the stack must not grow with the iteration count.
    out, stats = launch(sim, """
        mov.u32 %v, 0;
        mov.u32 %i, 0;
      HEAD:
        setp.ge.u32 %p, %i, %t;
        @%p bra DONE;
        add.u32 %v, %v, %i;
        add.u32 %i, %i, 1;
        bra HEAD;
      DONE:
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert out.tolist() == [t * (t - 1) // 2 for t in range(32)]
    assert stats.max_simt_depth == 2


def test_return_inside_divergent_region(sim):
    # Lanes with tid % 4 == 0 return early without storing; they must not run later code.
    out, _ = launch(sim, """
        and.b32 %v, %t, 3;
        setp.eq.u32 %p, %v, 0;
        and.b32 %v, %t, 1;
        setp.eq.u32 %q, %v, 0;
        @%q bra EVEN;
        mov.u32 %v, 5;
        bra STORE;
      EVEN:
        @%p ret;
        mov.u32 %v, 6;
      STORE:
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert out.tolist() == [0xDEAD, 5, 6, 5] * 8


def test_predicated_non_branch_instruction(sim):
    out, stats = launch(sim, """
        mov.u32 %v, 1;
        setp.ge.u32 %p, %t, 16;
        @%p mov.u32 %v, 2;
        @!%p add.u32 %v, %v, 10;
        st.global.u32 [%addr], %v;
        ret;
    """)
    assert out.tolist() == [11] * 16 + [2] * 16 and stats.divergent_branches == 0


def test_partial_warp_only_runs_real_threads(sim):
    out, stats = launch(sim, "st.global.u32 [%addr], %t;\nret;", n=40)
    assert out.tolist() == list(range(40))
    assert stats.warps == 2 and stats.threads == 40
    assert stats.lane_instructions == 40 * 6


def test_trace_sees_every_instruction_with_its_mask(sim):
    seen = []
    launch(sim, """
        setp.lt.u32 %p, %t, 4;
        @%p st.global.u32 [%addr], %t;
        ret;
    """, n=8, trace=lambda warp, pc, ins, mask: seen.append((ins.name, int(mask.sum()))))
    assert seen[-3:] == [("setp.lt.u32", 8), ("st.global.u32", 4), ("ret", 8)]


# ---------------------------------------------------------------------------------------------
# Lanes that meet before the point where their branch is known to rejoin
# (validation/ampere/reconvergence_golden.py has what a GPU does with these layouts)
# ---------------------------------------------------------------------------------------------

SHARED = ".shared .align 4 .b8 x[4]; .shared .align 4 .b8 ptrs[4]; .reg .b32 %seven, %tm;"
LANE0_WRITES = """
        setp.ne.u32 %p, %t, 0;
        @%p bra ELSE;
        mov.b32 %seven, 7;
        st.shared.u32 [x], %seven;
"""
READ = """
        ld.shared.u32 %v, [x];
        st.global.u32 [%addr], %v;
"""


def _lanes_fall_into_join_after_untaken_branch(at_join: str) -> str:
    """Lane 0 jumps to the join; the others fall into it past a branch that could go around
    it and that none of them takes. The first branch's post-dominator is AFTER."""
    return SHARED + LANE0_WRITES + """
        bra.uni JOIN;
      ELSE:
        setp.gt.u32 %q, %t, 31;
        @%q bra AFTER;
      JOIN:
    """ + at_join + READ + """
      AFTER:
        ret;
    """


def _lanes_branch_back_to_join(at_join: str) -> str:
    """Lane 0 falls into the join; the others' block is at the end of the function and
    branches back. The first branch's post-dominator is the end of the function."""
    return SHARED + LANE0_WRITES + """
      JOIN:
    """ + at_join + READ + """
        ret;
      ELSE:
        setp.lt.u32 %q, %t, 32;
        @%q bra JOIN;
        ret;
    """


LAYOUTS = {"falls into the join": _lanes_fall_into_join_after_untaken_branch,
           "branches back to the join": _lanes_branch_back_to_join}


def _executions(sim, body, opcode):
    """(number of active lanes) for every execution of `opcode`, and the launch's results."""
    seen = []

    def trace(warp, pc, ins, active):
        if ins.opcode == opcode:
            seen.append(int(active.sum()))
    out, stats = launch(sim, body, trace=trace)
    return seen, out, stats


@pytest.mark.parametrize("layout", LAYOUTS)
def test_barrier_at_an_early_join_is_reached_by_the_whole_warp(sim, layout):
    """Lanes 1 to 31 used to run through `bar.sync` and read the word before lane 0 had
    written it, with the warp counted as arrived."""
    seen, out, stats = _executions(sim, LAYOUTS[layout]("bar.sync 0;"), "bar")
    assert seen == [32]
    assert out.tolist() == [7] * 32
    assert not stats.findings


@pytest.mark.parametrize("layout", LAYOUTS)
def test_lanes_meet_at_an_early_join_without_a_barrier(sim, layout):
    """Match the measured Ampere schedule; this is not an implicit sync guarantee."""
    seen, out, stats = _executions(sim, LAYOUTS[layout](""), "ld")
    assert seen == [32, 32]  # the kernel's parameter, then the shared word
    assert out.tolist() == [7] * 32
    assert not stats.findings


@pytest.mark.parametrize("layout", LAYOUTS)
def test_warp_wide_allocation_at_an_early_join(sim, layout):
    """`if (tid == 0) {...} if (warp == 0) tmem_alloc(...)` as nvcc 13.0 lays it out: the
    allocation was refused with "requires all 32 lanes of the warp to execute together;
    only 31 participate"."""
    alloc = """
        mov.b32 %tm, ptrs;
        tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%tm], 32;
        ld.shared.u32 %tm, [ptrs];
        tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tm, 32;
    """
    seen, out, stats = _executions(sim, LAYOUTS[layout](alloc), "tcgen05")
    assert seen == [32, 32]
    assert out.tolist() == [7] * 32
    assert not stats.findings


@pytest.mark.parametrize("layout", LAYOUTS)
def test_barrier_waits_for_bypassing_lanes_to_exit(sim, layout):
    """Lanes 16 to 31 bypass the join and exit. The barrier first lets their
    pending path exit, then counts the remaining lanes without a false warning."""
    body = LAYOUTS[layout]("bar.sync 0;")
    body = body.replace("setp.gt.u32 %q, %t, 31;", "setp.gt.u32 %q, %t, 15;")
    body = body.replace("setp.lt.u32 %q, %t, 32;", "setp.lt.u32 %q, %t, 16;")
    seen, out, stats = _executions(sim, body, "bar")
    assert seen == [16]
    assert out.tolist() == [7] * 16 + [0xDEAD] * 16
    assert not stats.findings


def test_diverged_lanes_in_a_loop_keep_their_order(sim):
    """Entries on a common cycle are not reordered: a loop that lanes leave at different
    iterations runs as it did, one stack entry deep, and every lane counts to its own bound."""
    out, stats = launch(sim, """
        mov.u32 %i, 0;
        mov.u32 %v, 0;
      HEAD:
        setp.ge.u32 %p, %i, %t;
        @%p bra DONE;
        and.b32 %v, %i, 1;
        setp.eq.u32 %q, %v, 0;
        @%q bra SKIP;
        add.u32 %v, %v, 0;
      SKIP:
        add.u32 %i, %i, 1;
        bra HEAD;
      DONE:
        st.global.u32 [%addr], %i;
        ret;
    """)
    assert out.tolist() == list(range(32))
