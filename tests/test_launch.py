"""Host-side API: memory, arguments, grid and block geometry, faults."""

import numpy as np
import pytest

import warplens

from conftest import EXAMPLES, kernel

VECADD = EXAMPLES / "00_vecadd" / "vecadd.ptx"


def test_device_arrays_round_trip(sim):
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    d = sim.from_numpy(a)
    assert d.shape == (2, 3, 4) and d.nbytes == 192 and np.array_equal(d.numpy(), a)
    e = sim.empty((3, 5), "s32")
    assert e.numpy().dtype == np.int32 and e.numpy().shape == (3, 5)
    assert sim.from_numpy(a[:, ::2]).numpy().tolist() == a[:, ::2].tolist()  # non-contiguous input


@pytest.mark.parametrize("foreign_slot", [0, 1, 2])
def test_foreign_device_arguments_are_rejected_before_execution(sim, foreign_slot):
    local = [sim.from_numpy(np.array([v], dtype=np.float32)) for v in (10, 2, -99)]
    other = warplens.Device()
    foreign = other.from_numpy(np.array([100], dtype=np.float32))
    # Independent devices reuse addresses, so accepting this pointer would alias
    # local[0], whether the argument is used as an input or an output.
    assert foreign.addr == local[0].addr
    args = [*local, 1]
    args[foreign_slot] = foreign
    with pytest.raises(warplens.LaunchError, match=f"argument {foreign_slot} .*different device"):
        sim.launch(VECADD, "vecadd", grid=1, block=1, args=iter(args))
    assert [a.numpy().item() for a in local] == [10, 2, -99]
    assert foreign.numpy().item() == 100


def test_same_device_arguments_accept_an_iterator(sim):
    a, b = (sim.from_numpy(np.array([v], dtype=np.float32)) for v in (10, 2))
    out = sim.empty(1, "f32")
    sim.launch(VECADD, "vecadd", grid=1, block=1, args=iter([a, b, out, 1]))
    assert out.numpy().tolist() == [12]


def test_3d_grid_and_block_special_registers(sim):
    # Every thread writes one record of its coordinates at a unique global index.
    mod = kernel("""
        .reg .b32 %tx, %ty, %tz, %cx, %cy, %cz, %nx, %ny, %nz, %gx, %gy, %cta, %tid, %idx, %v;
        .reg .b64 %base, %off, %addr;
        mov.u32 %tx, %tid.x; mov.u32 %ty, %tid.y; mov.u32 %tz, %tid.z;
        mov.u32 %cx, %ctaid.x; mov.u32 %cy, %ctaid.y; mov.u32 %cz, %ctaid.z;
        mov.u32 %nx, %ntid.x; mov.u32 %ny, %ntid.y; mov.u32 %nz, %ntid.z;
        mov.u32 %gx, %nctaid.x; mov.u32 %gy, %nctaid.y;
        mad.lo.u32 %cta, %cz, %gy, %cy;
        mad.lo.u32 %cta, %cta, %gx, %cx;
        mad.lo.u32 %tid, %tz, %ny, %ty;
        mad.lo.u32 %tid, %tid, %nx, %tx;
        mul.lo.u32 %v, %nx, %ny;
        mul.lo.u32 %v, %v, %nz;
        mad.lo.u32 %idx, %cta, %v, %tid;
        ld.param.u64 %base, [out];
        mul.wide.u32 %off, %idx, 4;
        add.u64 %addr, %base, %off;
        shl.b32 %v, %cta, 16;
        or.b32 %v, %v, %tid;
        st.global.u32 [%addr], %v;
        ret;
    """, ".param .u64 out")
    grid, block = (2, 3, 2), (5, 4, 3)
    ctas, threads = 12, 60
    out = sim.empty(ctas * threads, "u32")
    stats = sim.launch(mod, "k", grid=grid, block=block, args=[out])
    expect = [(c << 16) | t for c in range(ctas) for t in range(threads)]
    assert out.numpy().tolist() == expect
    assert (stats.ctas, stats.warps, stats.threads) == (12, 24, 720)


def test_laneid_and_warpid(sim):
    mod = kernel("""
        .reg .b32 %t, %l, %w, %v; .reg .b64 %base, %off, %addr;
        mov.u32 %t, %tid.x; mov.u32 %l, %laneid; mov.u32 %w, %warpid;
        mad.lo.u32 %v, %w, 100, %l;
        ld.param.u64 %base, [out]; mul.wide.u32 %off, %t, 4; add.u64 %addr, %base, %off;
        st.global.u32 [%addr], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(70, "u32")
    sim.launch(mod, "k", grid=1, block=70, args=[out])
    assert out.numpy().tolist() == [(t // 32) * 100 + t % 32 for t in range(70)]


def test_scalar_arguments(sim):
    mod = kernel("""
        .reg .b32 %i, %f; .reg .b64 %base, %l;
        ld.param.u64 %base, [out];
        ld.param.s32 %i, [pi]; ld.param.f32 %f, [pf]; ld.param.u64 %l, [pl];
        st.global.s32 [%base], %i;
        st.global.f32 [%base+4], %f;
        st.global.u64 [%base+8], %l;
        ret;
    """, ".param .u64 out, .param .s32 pi, .param .f32 pf, .param .u64 pl")
    out = sim.empty(16, "u8")
    sim.launch(mod, "k", grid=1, block=1, args=[out, -7, 1.5, 2**40 + 3])
    raw = out.numpy()
    assert raw[:4].view(np.int32)[0] == -7
    assert raw[4:8].view(np.float32)[0] == 1.5
    assert raw[8:].view(np.uint64)[0] == 2**40 + 3


def test_argument_and_geometry_errors(sim):
    a = sim.empty(4, "f32")
    with pytest.raises(warplens.LaunchError, match="takes 4 argument"):
        sim.launch(VECADD, "vecadd", grid=1, block=4, args=[a, a, a])
    with pytest.raises(warplens.LaunchError, match="is an integer but got the float"):
        sim.launch(VECADD, "vecadd", grid=1, block=4, args=[a, a, a, 4.0])
    with pytest.raises(warplens.LaunchError, match="cannot hold a device pointer"):
        sim.launch(VECADD, "vecadd", grid=1, block=4, args=[a, a, a, a])
    with pytest.raises(warplens.LaunchError, match="2048 threads"):
        sim.launch(VECADD, "vecadd", grid=1, block=(64, 32), args=[a, a, a, 4])
    with pytest.raises(warplens.LaunchError, match="grid must be"):
        sim.launch(VECADD, "vecadd", grid=0, block=4, args=[a, a, a, 4])
    with pytest.raises(warplens.LaunchError, match="no kernel named `nope`.*found: vecadd"):
        sim.launch(VECADD, "nope", grid=1, block=4)
    with pytest.raises(warplens.LaunchError, match="unknown GPU model"):
        warplens.Device("A100")


def test_out_of_bounds_access_reports_line_and_thread(sim):
    n = 100
    a, b, c = (sim.empty(n, "f32") for _ in range(3))
    with pytest.raises(warplens.MemoryFault) as info:
        sim.launch(VECADD, "vecadd", grid=1, block=128, args=[a, b, c, n + 8])
    e = info.value
    msg = str(e)
    assert "read of 4 byte(s)" in msg and "0 byte(s) past the end of the 400-byte allocation" in msg
    assert "vecadd.ptx:41: ld.global.b32" in msg and "warp 3" in msg and "thread (100, 0, 0)" in msg
    assert e.lanes == list(range(4, 12))


def test_null_pointer_store(sim):
    mod = kernel(".reg .b64 %a;\nmov.u64 %a, 0;\nst.global.u32 [%a+8], 1;\nret;")
    with pytest.raises(warplens.MemoryFault, match=r"write of 4 byte\(s\) at 0x8 .*null pointer"):
        sim.launch(mod, "k", grid=1, block=1)


def test_access_straddling_the_end_faults(sim):
    mod = kernel(".reg .b64 %a; .reg .b32 %v;\nld.param.u64 %a, [p];\nld.global.u32 %v, [%a+2];\nret;",
                 ".param .u64 p")
    with pytest.raises(warplens.MemoryFault, match="straddling the end"):
        sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(4, "u8")])


def test_max_steps_stops_a_runaway_kernel(sim):
    mod = kernel("SPIN:\nbra SPIN;")
    with pytest.raises(warplens.ExecutionError, match="max_steps=1000"):
        sim.launch(mod, "k", grid=1, block=32, max_steps=1000)
