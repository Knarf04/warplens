"""SIMT divergence and reconvergence on hand-written control flow."""

import numpy as np

from conftest import kernel

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
