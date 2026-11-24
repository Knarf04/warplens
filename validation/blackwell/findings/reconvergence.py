"""A simulator bug found while preparing the rental, fixed since (branch work/reconvergence):
lanes of one warp that reach the same instruction by different paths were not always brought
together, and `bar.sync` then let some of them through before the others arrived.

    PYTHONPATH=<repo> python validation/blackwell/findings/reconvergence.py

Each kernel is `if (tid == 0) shared = 7;` followed by code every lane reaches, in four
layouts of the same control flow. All 32 lanes must end up reading 7 (with the barrier that
is guaranteed; without it a GPU reconverges at the join all the same: an RTX 3080 Ti gives
32 of 32 for all four either way, validation/ampere/reconvergence_golden.py). On 2026-11-23
the simulator gave 32 of 32 for A and B and **1 of 32 for C and D, with or without bar.sync**:
lanes 1 to 31 read the word before lane 0 had written it. This script did not look at the
findings; with the barrier there was a `barrier-in-divergent-code` finding, on a program that
has no barrier in divergent code. It gives 32 of 32 for all eight now.

A  the join is the branch target (how nvcc 13.5 lays this out)
B  the then-side jumps to the join; the other lanes fall into it from a block of their own
C  as B, and the other lanes' block ends in a conditional branch that none of them takes.
   nvcc 13.0 produces this from `if (tid == 0) {...} if (warp == 0) tmem_alloc(...)`: it
   merges the two tests, lane 0 jumps to the allocation and lanes 1 to 31 fall into it. The
   simulator then refuses the allocation ("requires all 32 lanes of the warp to execute
   together; only 31 participate")
D  the other lanes' block is at the END of the function and branches BACK to the join.
   nvcc 13.0 produced this in cluster/clc_per_thread.cu, where lanes 1 to 3 then issued
   their request before lane 0 had initialized the mbarrier, behind a __syncthreads()

Consequence for the rental: with a toolkit other than the one the predictions were made
with, a probe's code can take one of these shapes and the simulator's prediction for it is
wrong or missing, although the program is correct. The probes now put a __syncthreads()
between the two tests, which keeps a compiler from merging them; with that, nvcc 13.0.88 and
nvcc 13.5.37 give identical predictions for all 98 predicted runs. The cause was that the
sides of a branch were only joined at its immediate post-dominator, which in C and D is past
the join; warplens/core/warp.py (`_regroup`) now joins them where they meet.
tests/test_rental_harness.py runs the four shapes.
"""
import numpy as np
import warplens

HEADER = ".version 9.0\n.target sm_100a\n.address_size 64\n"
DECL = """
.visible .entry k(.param .u64 out)
{
    .reg .pred %p, %q;
    .reg .b32 %tid, %v, %one, %dummy;
    .reg .b64 %base, %off, %addr;
    .shared .align 4 .b8 x[4];
    mov.u32 %tid, %tid.x;
    setp.ne.s32 %p, %tid, 0;
"""
THEN = """
    mov.b32 %one, 7;
    st.shared.u32 [x], %one;
"""
JOIN = """
JOIN:
    BARRIER
    ld.shared.u32 %v, [x];
    ld.param.u64 %base, [out];
    mul.wide.u32 %off, %tid, 4;
    add.u64 %addr, %base, %off;
    st.global.u32 [%addr], %v;
AFTER:
    ret;
"""
SHAPES = {
    "A": DECL + "    @%p bra JOIN;" + THEN + JOIN + "}\n",
    "B": DECL + "    @%p bra ELSE;" + THEN + "    bra.uni JOIN;\nELSE:\n    mov.b32 %dummy, 0;" + JOIN + "}\n",
    "C": DECL + "    @%p bra ELSE;" + THEN + "    bra.uni JOIN;\nELSE:\n    setp.gt.u32 %q, %tid, 31;\n    @%q bra AFTER;"
         + JOIN + "}\n",
    "D": DECL + "    @%p bra ELSE;" + THEN + JOIN + "ELSE:\n    setp.lt.u32 %q, %tid, 32;\n    @%q bra JOIN;\n    ret;\n}\n",
}



def lanes_reading_seven(shape: str, barrier: bool) -> int:
    """How many of the 32 lanes read what lane 0 wrote (32 is right)."""
    sim = warplens.Device("B200")
    out = sim.from_numpy(np.full(32, 99, dtype=np.uint32))
    module = warplens.parse_ptx(HEADER + SHAPES[shape].replace("BARRIER", "bar.sync 0;" if barrier else ""),
                                name="reconvergence.ptx")
    sim.launch(module, "k", grid=1, block=32, args=[out])
    return int((out.numpy() == 7).sum())


if __name__ == "__main__":
    for shape in SHAPES:
        for barrier in (True, False):
            label = f"shape {shape}, {'with bar.sync' if barrier else 'no barrier   '}"
            try:
                print(f"{label}: lanes reading 7: {lanes_reading_seven(shape, barrier)} of 32")
            except Exception as error:  # a refusal is also an answer
                print(f"{label}: {type(error).__name__}: {str(error).splitlines()[0][:150]}")
