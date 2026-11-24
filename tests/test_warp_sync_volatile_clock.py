"""Forms ordinary CUDA C++ compiles to: `bar.warp.sync` (`__syncwarp()`), `.volatile` loads
and stores, and the `%clock64` cycle counter (`clock64()` and `clock()`)."""

import numpy as np
import pytest

import warplens

from conftest import HEADER, kernel


# -- bar.warp.sync -----------------------------------------------------------------------------

# What nvcc 13.5 emits for sm_100a from
#   __syncwarp(); __syncwarp(0xffff); p[threadIdx.x] = q[threadIdx.x] + 1; q[1] = p[0];
# with `volatile int* p`.
SYNCWARP = HEADER + """
.visible .entry k_syncwarp(
    .param .u64 .ptr .align 1 k_syncwarp_param_0,
    .param .u64 .ptr .align 1 k_syncwarp_param_1
)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;

    ld.param.b64 %rd1, [k_syncwarp_param_0];
    ld.param.b64 %rd2, [k_syncwarp_param_1];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    bar.warp.sync -1;
    bar.warp.sync 65535;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.b32 %r2, [%rd6];
    add.s32 %r3, %r2, 1;
    add.s64 %rd7, %rd3, %rd5;
    st.volatile.global.b32 [%rd7], %r3;
    ld.volatile.global.b32 %r4, [%rd3];
    st.global.b32 [%rd4+4], %r4;
    ret;
}
"""


def test_syncwarp_and_volatile_as_nvcc_emits_them(sim):
    p = sim.empty(16, "s32")
    q = sim.from_numpy(np.arange(100, 116, dtype=np.int32))
    stats = sim.launch(warplens.parse_ptx(SYNCWARP), "k_syncwarp", grid=1, block=16, args=[p, q])
    assert p.numpy().tolist() == list(range(101, 117))
    assert q.numpy().tolist() == [100, 101, *range(102, 116)]  # q[1] = p[0] = q[0] + 1
    assert stats.findings == []


def warp_sync(sim, body: str, block: int = 32):
    """Run `body` (with %t = tid.x and predicates %p, %q declared) and return the findings."""
    mod = kernel(f"""
        .reg .b32 %t, %m; .reg .pred %p, %q;
        mov.u32 %t, %tid.x;
        {body}
        ret;
    """)
    return sim.launch(mod, "k", grid=1, block=block).findings


def test_warp_sync_of_a_whole_or_partial_warp_is_quiet(sim):
    assert warp_sync(sim, "bar.warp.sync -1;") == []
    assert warp_sync(sim, "bar.warp.sync 0xffffffff;", block=20) == []  # lanes that never ran may be named
    assert warp_sync(sim, "bar.warp.sync -1;", block=96) == []
    # The mask may come from a register, and each half of the warp may name only itself.
    assert warp_sync(sim, """
        setp.lt.u32 %p, %t, 16;
        selp.b32 %m, 0x0000ffff, 0xffff0000, %p;
        bar.warp.sync %m;
    """) == []


def test_warp_sync_of_a_subset_that_all_arrives_is_quiet(sim):
    # if (lane < 16) __syncwarp(0xffff);
    assert warp_sync(sim, """
        setp.ge.u32 %p, %t, 16;
        @%p bra SKIP;
        bar.warp.sync 0xffff;
      SKIP:
    """) == []
    # Lanes that have returned are not waited for.
    assert warp_sync(sim, """
        setp.ge.u32 %p, %t, 16;
        @%p ret;
        bar.warp.sync -1;
    """) == []


def test_warp_sync_that_would_wait_for_another_path_is_reported(sim):
    # Only the lower half executes it, but the mask names the whole warp, which is still running.
    findings = warp_sync(sim, """
        setp.ge.u32 %p, %t, 16;
        @%p bra SKIP;
        bar.warp.sync -1;
      SKIP:
    """)
    assert [f.kind for f in findings] == ["warp-sync-membermask-mismatch"]
    assert findings[0].lanes == list(range(16))


def test_warp_sync_caller_outside_its_mask_is_reported(sim):
    findings = warp_sync(sim, "bar.warp.sync 0xffff;")
    assert [f.kind for f in findings] == ["warp-sync-membermask-mismatch"]
    assert findings[0].lanes == list(range(16, 32))


def test_warp_sync_under_a_false_predicate_does_nothing(sim):
    assert warp_sync(sim, "setp.gt.u32 %p, %t, 99;\n@%p bar.warp.sync 1;") == []


def test_warp_sync_does_not_release_a_cta_barrier(sim):
    mod = kernel("""
        .reg .b32 %t; .reg .pred %p;
        mov.u32 %t, %tid.x;
        setp.ge.u32 %p, %t, 32;
        @%p bra OTHER;
        bar.sync 0;
        ret;
      OTHER:
        bar.warp.sync -1;
        bar.sync 1;
        ret;
    """)
    with pytest.raises(warplens.DeadlockError, match="barrier 0"):
        sim.launch(mod, "k", grid=1, block=64)


@pytest.mark.parametrize("text", ["bar.warp.sync;", "bar.warp.sync -1, 32;", "bar.warp.arrive -1;",
                                  "barrier.warp.sync -1;"])
def test_other_spellings_of_warp_sync_are_rejected(sim, text):
    with pytest.raises(warplens.UnsupportedError):
        sim.launch(kernel(f"{text}\nret;"), "k", grid=1, block=32)


# -- .volatile -----------------------------------------------------------------------------------

@pytest.mark.parametrize("ty", ["b8", "b16", "b32", "b64", "f32"])
def test_volatile_global_load_and_store(sim, ty):
    size = 4 if ty == "f32" else int(ty[1:]) // 8
    mod = kernel(f"""
        .reg .b64 %src, %dst, %off, %v; .reg .b32 %t;
        ld.param.u64 %src, [src];
        ld.param.u64 %dst, [dst];
        mov.u32 %t, %tid.x;
        mul.wide.u32 %off, %t, {size};
        add.u64 %src, %src, %off;
        add.u64 %dst, %dst, %off;
        ld.volatile.global.{ty} %v, [%src];
        st.volatile.global.{ty} [%dst], %v;
        ret;
    """, ".param .u64 src, .param .u64 dst")
    data = np.random.default_rng(size).integers(0, 256, 40 * size, dtype=np.uint8)
    out = sim.empty(40 * size, "u8")
    sim.launch(mod, "k", grid=1, block=40, args=[sim.from_numpy(data), out])
    assert out.numpy().tolist() == data.tolist()


def test_volatile_shared_local_and_generic(sim):
    # A value goes thread -> shared -> the neighbour's local memory -> global, all volatile.
    mod = kernel("""
        .shared .align 4 .b32 tile[64];
        .local .align 4 .b32 slot[2];
        .reg .b64 %out, %off, %g; .reg .b32 %t, %a, %v, %n;
        ld.param.u64 %out, [out];
        mov.u32 %t, %tid.x;
        shl.b32 %a, %t, 2;
        add.u32 %v, %t, 500;
        mov.u32 %n, tile;
        add.u32 %a, %a, %n;
        st.volatile.shared.u32 [%a], %v;
        bar.sync 0;
        xor.b32 %a, %t, 1;
        shl.b32 %a, %a, 2;
        add.u32 %a, %a, %n;
        ld.volatile.shared.u32 %v, [%a];
        st.volatile.local.u32 [slot+4], %v;
        ld.volatile.local.u32 %n, [slot+4];
        cvta.local.u64 %g, slot;
        ld.volatile.u32 %v, [%g+4];
        add.u32 %v, %v, %n;
        mul.wide.u32 %off, %t, 4;
        add.u64 %out, %out, %off;
        st.volatile.u32 [%out], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(64, "u32")
    stats = sim.launch(mod, "k", grid=1, block=64, args=[out])
    assert out.numpy().tolist() == [2 * ((t ^ 1) + 500) for t in range(64)]
    assert stats.findings == []


def test_volatile_vector_load(sim):
    mod = kernel("""
        .reg .b64 %p; .reg .b32 %a, %b;
        ld.param.u64 %p, [p];
        ld.volatile.global.v2.u32 {%a, %b}, [%p];
        st.volatile.global.v2.u32 [%p+8], {%b, %a};
        ret;
    """, ".param .u64 p")
    data = sim.from_numpy(np.array([7, 9, 0, 0], dtype=np.uint32))
    sim.launch(mod, "k", grid=1, block=1, args=[data])
    assert data.numpy().tolist() == [7, 9, 9, 7]


@pytest.mark.parametrize("text", ["ld.global.volatile.u32 %a, [%p];", "ld.volatile.weak.global.u32 %a, [%p];",
                                  "ld.volatile.volatile.u32 %a, [%p];", "st.weak.global.u32 [%p], %a;"])
def test_misplaced_ordering_modifiers_are_rejected(sim, text):
    mod = kernel(f".reg .b64 %p; .reg .b32 %a;\nld.param.u64 %p, [p];\n{text}\nret;", ".param .u64 p")
    with pytest.raises(warplens.UnsupportedError):
        sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(4, "u32")])


# -- %clock64, %clock, %clock_hi -------------------------------------------------------------------

CLOCK = """
    .reg .b64 %out, %off, %c0, %c1; .reg .b32 %t, %lo, %hi, %x;
    ld.param.u64 %out, [out];
    mov.u32 %t, %tid.x;
    mul.wide.u32 %off, %t, 32;
    add.u64 %out, %out, %off;
    mov.u64 %c0, %clock64;
    add.u32 %x, %t, 1;
    add.u32 %x, %x, 1;
    add.u32 %x, %x, 1;
    mov.u64 %c1, %clock64;
    mov.u32 %lo, %clock;
    mov.u32 %hi, %clock_hi;
    st.global.u64 [%out], %c0;
    st.global.u64 [%out+8], %c1;
    st.global.u32 [%out+16], %lo;
    st.global.u32 [%out+20], %hi;
    ret;
"""


def clocks(sim, **launch):
    out = sim.empty((launch.get("grid", 1) * 64, 4), "u64")
    mod = kernel(CLOCK.replace("%tid.x", "%tid.x;\nmov.u32 %x, %ctaid.x;\nmad.lo.u32 %t, %x, 64, %t")
                 if launch.get("grid", 1) > 1 else CLOCK, ".param .u64 out")
    stats = sim.launch(mod, "k", block=64, args=[out], **{"grid": 1, **launch})
    return out.numpy(), stats


def test_clock64_counts_scheduling_rounds_in_an_untimed_run(sim):
    values, _ = clocks(sim)
    c0, c1, low = values[:, 0], values[:, 1], values[:, 2]
    # Four instructions were executed before the first read; every lane of both warps agrees.
    assert set(c0.tolist()) == {4}
    assert set(c1.tolist()) == {8}  # the first read and three adds later
    # %clock is the low half, read one instruction later, and %clock_hi the high half.
    assert set((low & 0xFFFFFFFF).tolist()) == {9} and set((low >> 32).tolist()) == {0}


def test_clock64_is_deterministic_and_keeps_counting_across_ctas(sim):
    first, _ = clocks(sim, grid=3)
    again, _ = clocks(sim, grid=3)
    assert first.tolist() == again.tolist()
    starts = [set(first[64 * c:64 * (c + 1), 0].tolist()) for c in range(3)]
    assert all(len(s) == 1 for s in starts)
    a, b, c = (s.pop() for s in starts)
    assert a < b < c and b - a == c - b  # each CTA runs the same number of rounds


def test_clock64_advances_while_a_warp_waits(sim):
    # Warp 1 goes straight to the barrier; warp 0 first runs a 50-iteration loop.
    mod = kernel("""
        .reg .b64 %out, %off, %c; .reg .b32 %t, %i; .reg .pred %p;
        ld.param.u64 %out, [out];
        mov.u32 %t, %tid.x;
        mul.wide.u32 %off, %t, 8;
        add.u64 %out, %out, %off;
        setp.ge.u32 %p, %t, 32;
        @%p bra WAIT;
        mov.u32 %i, 50;
      LOOP:
        sub.u32 %i, %i, 1;
        setp.ne.u32 %p, %i, 0;
        @%p bra LOOP;
      WAIT:
        bar.sync 0;
        mov.u64 %c, %clock64;
        st.global.u64 [%out], %c;
        ret;
    """, ".param .u64 out")
    out = sim.empty(64, "u64")
    sim.launch(mod, "k", grid=1, block=64, args=[out])
    values = out.numpy()
    assert len(set(values.tolist())) == 1  # both warps resume in the same round
    assert values[0] > 150  # 50 iterations of three instructions passed for the waiting warp too


def test_clock64_follows_the_timing_model_in_a_timed_run(sim):
    values, stats = clocks(sim, timing=True)
    c0, c1, low = values[:, 0], values[:, 1], values[:, 2]
    assert len(set(c0[:32].tolist())) == 1 and len(set(c1[:32].tolist())) == 1
    # One warp per issue slot here, and each of these instructions costs one clock.
    assert (c1 - c0 == 4).all()
    assert ((low & 0xFFFFFFFF) == c1 + 1).all() and (low >> 32 == 0).all()
    assert c1.max() <= stats.timing.clocks


def test_clock_needs_a_type_of_at_least_32_bits(sim):
    with pytest.raises(warplens.ExecutionError, match="%clock64 cannot be read as .u16"):
        sim.launch(kernel(".reg .b16 %h;\nmov.u16 %h, %clock64;\nret;"), "k", grid=1, block=1)
