"""Regression reproducer for warp rendezvous beyond a static reconvergence point.

    PYTHONPATH=<repo> python validation/blackwell/findings/reconvergence.py

Each kernel is `if (tid == 0) shared = 7;` followed by code every lane reaches,
using four layouts of the same control flow. With bar.sync all 32 lanes must
read 7. Without synchronization, PTX does not guarantee this communication;
the simulator's deterministic scheduling order is not a hardware prediction.

A  the join is the branch target (how nvcc 13.5 lays this out)
B  the then-side jumps to the join; other lanes fall into it from their own block
C  as B, but that block ends in a conditional branch no lane actually takes
D  the other lanes' block is at the end and branches backward to the join

Before the rental-gap fix, C and D produced only 1 correct lane even with
bar.sync: their static immediate postdominator lies beyond the barrier, and
part of the warp was incorrectly counted as a whole-warp arrival. This also
rejected tcgen05.alloc.sync.aligned in compiler-merged control flow.

The executor now collects pending paths at named CTA barriers and tcgen05
.sync.aligned operations before executing them. All four synchronized cases
produce 32 correct lanes. Ordinary unsynchronized instructions keep the
static reconvergence schedule; C and D without a barrier still produce 1.
Tests cover forward/backward joins, multiple warps, bar.arrive, exited lanes,
repeated phases, and timed execution. Divergent rendezvous at different
instruction addresses or across recursive activations remains unsupported.
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
    """How many lanes read lane 0's value (32 is required with a barrier)."""
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
