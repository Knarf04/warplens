"""Atomic additions preserve RMW values, masks and shared-memory diagnostics."""

import shutil
import subprocess

import numpy as np
import pytest

import warplens
from conftest import kernel
from warplens.frontend.ir import NP_TYPES


def add_kernel(ty, *, space="global", operation="atom", qualifiers="", predicate="", setup=""):
    bits = NP_TYPES[ty].itemsize * 8
    is_shared = space in ("shared", "shared::cta", "generic_shared")
    declaration = f".shared .align {bits // 8} .b{bits} cell;" if is_shared else ""
    initialize = f"@%first st.shared.b{bits} [cell], %initial; bar.sync 0;" if is_shared else ""
    target = "cell" if is_shared else "%target"
    actual_space = space
    if space == "generic_shared":
        initialize += " cvta.shared.u64 %target, cell;"
        target, actual_space = "%target", ""
    if space == "generic_global":
        actual_space = ""
    name = ".".join(p for p in [operation, qualifiers, actual_space, "add", ty] if p)
    dest = "%old, " if operation == "atom" else ""
    finish = f"bar.sync 0; @%first ld.shared.b{bits} %final, [cell]; @%first st.global.b{bits} [%result], %final;" if is_shared else ""
    return kernel(f"""
        .reg .b64 %target, %result, %input, %out, %offset;
        .reg .u32 %t, %block, %width;
        .reg .b{bits} %initial, %value, %old, %final;
        .reg .pred %first, %odd;
        {declaration}
        ld.param.u64 %target, [target]; mov.b64 %result, %target;
        ld.param.u64 %input, [input]; ld.param.u64 %out, [out];
        mov.u32 %t, %tid.x; mov.u32 %block, %ctaid.x; mov.u32 %width, %ntid.x;
        setp.eq.u32 %first, %t, 0;
        mad.lo.u32 %t, %block, %width, %t;
        mul.wide.u32 %offset, %t, {bits // 8};
        add.u64 %input, %input, %offset; add.u64 %out, %out, %offset;
        ld.global.b{bits} %value, [%input]; ld.global.b{bits} %initial, [%target];
        mov.b{bits} %old, 99;
        {initialize}
        {setup}
        {predicate} {name} {dest}[{target}], %value;
        st.global.b{bits} [%out], %old;
        {finish}
        ret;
    """, ".param .u64 target, .param .u64 input, .param .u64 out")


def run_add(sim, ty, initial, inputs, *, space="global", operation="atom", grid=1, **kwargs):
    dt = NP_TYPES[ty]
    inputs = np.asarray(inputs, dtype=dt)
    target = sim.from_numpy(np.asarray([initial], dtype=dt))
    source = sim.from_numpy(inputs)
    old = sim.empty(len(inputs), dt)
    stats = sim.launch(add_kernel(ty, space=space, operation=operation, **kwargs), "k",
                       grid=grid, block=len(inputs) // grid, args=[target, source, old])
    return target.numpy()[0], old.numpy(), stats


@pytest.mark.parametrize("ty", ["u32", "s32", "u64", "f32", "f64"])
@pytest.mark.parametrize("space", ["global", "shared", "shared::cta", "generic_global", "generic_shared"])
def test_contended_add_returns_each_old_value_once(sim, ty, space):
    result, old, stats = run_add(sim, ty, 0, np.ones(64), space=space)
    assert result == 64
    np.testing.assert_array_equal(np.sort(old), np.arange(64))
    assert not stats.findings


@pytest.mark.parametrize("operation", ["atom", "red"])
@pytest.mark.parametrize("ty", ["u32", "s32", "u64", "f32", "f64"])
def test_multiple_ctas_add_without_lost_updates(sim, ty, operation):
    result, old, stats = run_add(sim, ty, 0, np.ones(128), grid=4, operation=operation,
                                 qualifiers="relaxed.gpu")
    assert result == 128
    if operation == "atom":
        np.testing.assert_array_equal(np.sort(old), np.arange(128))
    assert not stats.findings


@pytest.mark.parametrize("operation", ["atom", "red"])
@pytest.mark.parametrize("space", ["global", "shared"])
def test_only_predicated_lanes_update_memory(sim, operation, space):
    result, old, stats = run_add(sim, "u32", 0, np.ones(32), operation=operation, space=space,
        predicate="@%odd", setup="and.b32 %t, %t, 1; setp.eq.u32 %odd, %t, 1;")
    assert result == 16
    if operation == "atom":
        np.testing.assert_array_equal(old[::2], 99)
        np.testing.assert_array_equal(np.sort(old[1::2]), np.arange(16))
    assert not stats.findings


@pytest.mark.parametrize("ty", ["s32", "u32", "u64"])
def test_integer_add_wraps_at_the_operand_width(sim, ty):
    dt = NP_TYPES[ty]
    initial = np.iinfo(dt).max - 1
    result, old, _ = run_add(sim, ty, initial, [1, 1, 1, 1])
    unsigned = np.dtype(f"u{dt.itemsize}")
    mask = (1 << (8 * dt.itemsize)) - 1
    expected = np.asarray([(int(initial) + i) & mask for i in range(4)], dtype=unsigned).view(dt)
    np.testing.assert_array_equal(old, expected)
    assert result == np.asarray([(int(initial) + 4) & mask], dtype=unsigned).view(dt)[0]


@pytest.mark.parametrize("space,expected", [("global", 0), ("shared", 2)])
@pytest.mark.parametrize("operation", ["atom", "red"])
def test_f32_subnormal_inputs_follow_memory_space(sim, space, expected, operation):
    tiny = np.asarray([1], dtype=np.uint32).view(np.float32)[0]
    result, old, _ = run_add(sim, "f32", tiny, [tiny], space=space, operation=operation)
    assert result.view(np.uint32) == expected
    if operation == "atom":
        assert old.view(np.uint32)[0] == 1  # the returned value is the original bits


@pytest.mark.parametrize("old_bits,value_bits,expected", [
    (0x00800001, 0x80800000, 0), (0x80800001, 0x00800000, 0x80000000),
])
def test_f32_global_flushes_subnormal_results_with_their_sign(sim, old_bits, value_bits, expected):
    a, b = np.asarray([old_bits, value_bits], dtype=np.uint32).view(np.float32)
    result, _, _ = run_add(sim, "f32", a, [b])
    assert result.view(np.uint32) == expected


def test_f64_add_preserves_subnormal_inputs_and_results(sim):
    tiny = np.asarray([1], dtype=np.uint64).view(np.float64)[0]
    result, old, _ = run_add(sim, "f64", tiny, [tiny])
    assert result.view(np.uint64) == 2 and old.view(np.uint64)[0] == 1


@pytest.mark.parametrize("operation", ["atom", "red"])
def test_explicit_f32_noftz_preserves_subnormals(sim, operation):
    tiny = np.asarray([1], dtype=np.uint32).view(np.float32)
    target = sim.from_numpy(tiny)
    dest = "%old, " if operation == "atom" else ""
    mod = kernel(f""".reg .u64 %p; .reg .b32 %old;
        ld.param.u64 %p, [p]; {operation}.global.add.noftz.f32 {dest}[%p], 0f00000001; ret;
    """, ".param .u64 p")
    sim.launch(mod, "k", grid=1, block=1, args=[target])
    assert target.numpy().view(np.uint32)[0] == 2


@pytest.mark.parametrize("ordinary", ["ld.shared.u32 %old, [cell];", "st.shared.u32 [cell], 7;"])
@pytest.mark.parametrize("atomic_first", [False, True])
def test_ordinary_and_atomic_accesses_still_report_a_shared_race(sim, ordinary, atomic_first):
    atomic = "atom.shared.add.u32 %old, [cell], 1;"
    first, second = (atomic, ordinary) if atomic_first else (ordinary, atomic)
    mod = kernel(f"""
        .reg .pred %first, %warp0; .reg .u32 %t, %old; .shared .u32 cell;
        mov.u32 %t, %tid.x; setp.eq.u32 %first, %t, 0; setp.lt.u32 %warp0, %t, 32;
        @%first st.shared.u32 [cell], 0; bar.sync 0;
        @%warp0 {first} @!%warp0 {second} ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=64)
    assert any(f.kind == "smem-race" for f in stats.findings)


def test_atomic_add_reports_uninitialized_shared_memory(sim):
    mod = kernel(".shared .u32 cell; red.shared.add.u32 [cell], 1; ret;")
    stats = sim.launch(mod, "k", grid=1, block=32)
    assert any(f.kind == "smem-uninitialized-read" for f in stats.findings)


@pytest.mark.parametrize("name", ["atom.global.sub.u32", "atom.global.add.s64", "atom.global.add.f16",
    "atom.acquire.gpu.global.add.u32", "atom.release.gpu.global.add.u32", "atom.acq_rel.gpu.global.add.u32",
    "atom.relaxed.cluster.global.add.u32", "atom.shared::cluster.add.u32", "atom.global.add.ftz.f32",
    "atom.global.v2.add.f32", "atom.global.global.add.u32", "atom.global.add.noftz.f64"])
def test_unimplemented_atomic_forms_are_explicit_errors(sim, name):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint64))
    mod = kernel(f".reg .u64 %p, %v; ld.param.u64 %p, [p]; {name} %v, [%p], 1; ret;", ".param .u64 p")
    with pytest.raises(warplens.UnsupportedError):
        sim.launch(mod, "k", grid=1, block=1, args=[target])


def test_misaligned_atomic_address_fails_without_mutation(sim):
    target = sim.from_numpy(np.zeros(8, dtype=np.uint8))
    mod = kernel(".reg .u64 %p; .reg .u32 %v; ld.param.u64 %p, [p]; atom.global.add.u32 %v, [%p+1], 1; ret;", ".param .u64 p")
    with pytest.raises(warplens.ExecutionError, match="aligned"):
        sim.launch(mod, "k", grid=1, block=1, args=[target])
    assert not target.numpy().any()


def test_inactive_invalid_address_does_not_fault(sim):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    mod = kernel(""".reg .u64 %p; .reg .u32 %t, %v; .reg .pred %first;
        ld.param.u64 %p, [p]; mov.u32 %t, %tid.x; setp.eq.u32 %first, %t, 0;
        @!%first mov.u64 %p, 1; @%first atom.global.add.u32 %v, [%p], 1; ret;
    """, ".param .u64 p")
    sim.launch(mod, "k", grid=1, block=32, args=[target])
    assert target.numpy()[0] == 1


CUDA = r"""
extern "C" __global__ void atomic_ints(int* sums, int* old) {
  int t=threadIdx.x+blockIdx.x*blockDim.x;
  old[t]=atomicAdd(sums,1);
  atomicAdd(sums+1,2);
}
extern "C" __global__ void atomic_floats(float* sums, float* old) {
  int t=threadIdx.x+blockIdx.x*blockDim.x;
  old[t]=atomicAdd(sums,1.0f);
  atomicAdd(sums+1,2.0f);
}
extern "C" __global__ void atomic_doubles(double* sums, double* old) {
  int t=threadIdx.x+blockIdx.x*blockDim.x;
  old[t]=atomicAdd(sums,1.0);
  atomicAdd(sums+1,2.0);
}
extern "C" __global__ void atomic_shared(unsigned long long* sums, unsigned long long* old) {
  __shared__ unsigned long long sum;
  if (threadIdx.x==0) sum=0;
  __syncthreads();
  old[threadIdx.x]=atomicAdd(&sum,1ull);
  atomicAdd(&sum,2ull);
  __syncthreads();
  if(threadIdx.x==0) *sums=sum;
}
"""


@pytest.fixture(scope="module")
def compiled_atomic_add(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc is required to compile CUDA atomicAdd")
    tmp = tmp_path_factory.mktemp("atomic_add_cuda")
    source, ptx = tmp / "atomic.cu", tmp / "atomic.ptx"
    source.write_text(CUDA)
    subprocess.run([nvcc, "-arch=sm_100a", "-ptx", str(source), "-o", str(ptx)], check=True,
                   capture_output=True, text=True, timeout=60)
    text = ptx.read_text()
    assert "atom." in text
    return warplens.parse_ptx(text)


@pytest.mark.parametrize("entry,ty", [("atomic_ints", "s32"), ("atomic_floats", "f32"),
                                       ("atomic_doubles", "f64"), ("atomic_shared", "u64")])
def test_compiled_cuda_atomic_add_runs_without_a_gpu(sim, compiled_atomic_add, entry, ty):
    count = 64
    target = sim.from_numpy(np.zeros(2, dtype=NP_TYPES[ty]))
    old = sim.empty(count, ty)
    shared = entry == "atomic_shared"
    stats = sim.launch(compiled_atomic_add, entry, grid=1 if shared else 2, block=count if shared else count // 2,
                       args=[target, old])
    assert not stats.findings
    if shared:
        assert target.numpy()[0] == 3 * count
        assert len(np.unique(old.numpy())) == count
        assert np.all(old.numpy() < 3 * count)
    else:
        np.testing.assert_array_equal(target.numpy(), [count, 2 * count])
        np.testing.assert_array_equal(np.sort(old.numpy()), np.arange(count))


def test_discarded_atomic_return_still_updates_memory(sim):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    mod = kernel(".reg .u64 %p; ld.param.u64 %p, [p]; atom.global.add.u32 _, [%p], 1; ret;", ".param .u64 p")
    sim.launch(mod, "k", grid=1, block=32, args=[target])
    assert target.numpy()[0] == 32


def test_atomic_add_checks_pending_async_writers(sim):
    source = sim.from_numpy(np.ones(1, dtype=np.uint32))
    mod = kernel(""".reg .u64 %p; .reg .u32 %old; .shared .align 16 .b8 cell[16];
        ld.param.u64 %p, [p];
        cp.async.ca.shared.global [cell], [%p], 4;
        cp.async.commit_group;
        atom.shared.add.u32 %old, [cell], 1;
        cp.async.wait_group 0;
        ret;
    """, ".param .u64 p")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[source], async_delay=100)
    assert any(f.kind == "tma-load-unsynchronized" for f in stats.findings)


def test_out_of_bounds_atomic_preserves_memory_and_reports_the_lane(sim):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    mod = kernel(""".reg .u64 %p, %off; .reg .u32 %t, %old;
        ld.param.u64 %p, [p]; mov.u32 %t, %tid.x;
        mul.wide.u32 %off, %t, 4; add.u64 %p, %p, %off;
        atom.global.add.u32 %old, [%p], 1; ret;
    """, ".param .u64 p")
    with pytest.raises(warplens.MemoryFault) as error:
        sim.launch(mod, "k", grid=1, block=2, args=[target])
    assert error.value.lanes == [1]
    assert target.numpy()[0] == 0


def test_global_atomic_timing_includes_existing_access_latency(sim):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    def run(instruction):
        mod = kernel(".reg .u64 %p; .reg .u32 %old; ld.param.u64 %p, [p]; " + instruction + " ret;", ".param .u64 p")
        return sim.launch(mod, "k", grid=1, block=32, args=[target], timing=True).timing.time_ns
    assert run("atom.global.add.u32 %old, [%p], 1;") > run("mov.u32 %old, 0;")


@pytest.mark.parametrize("scope", ["cta", "sys"])
def test_explicit_relaxed_atomic_scopes(sim, scope):
    result, old, _ = run_add(sim, "u32", 0, np.ones(32), qualifiers="relaxed." + scope)
    assert result == 32
    np.testing.assert_array_equal(np.sort(old), np.arange(32))


@pytest.mark.parametrize("space", ["local", "param"])
def test_generic_atomic_rejects_non_global_non_shared_spaces(sim, space):
    variable = ".local .u32 cell;" if space == "local" else ""
    name = "cell" if space == "local" else "value"
    mod = kernel(f""".reg .u64 %p; .reg .u32 %old; {variable}
        cvta.{space}.u64 %p, {name}; atom.add.u32 %old, [%p], 1; ret;
    """, ".param .u32 value")
    with pytest.raises(warplens.ExecutionError, match="global and shared"):
        sim.launch(mod, "k", grid=1, block=1, args=[0])


def test_generic_local_shared_atomic_keeps_cluster_rank(sim):
    out = sim.empty(2, "u32")
    mod = kernel(""".reg .u64 %p, %out, %off; .reg .u32 %old, %rank, %tid; .reg .pred %first;
        .shared .u32 cell;
        mov.u32 %rank, %cluster_ctarank; mov.u32 %tid, %tid.x; setp.eq.u32 %first, %tid, 0;
        @%first st.shared.u32 [cell], 0; bar.sync 0;
        cvta.shared.u64 %p, cell;
        atom.add.u32 %old, [%p], 1;
        bar.sync 0; @%first ld.shared.u32 %old, [cell];
        ld.param.u64 %out, [out]; mul.wide.u32 %off, %rank, 4; add.u64 %out, %out, %off;
        @%first st.global.u32 [%out], %old; ret;
    """, ".param .u64 out")
    stats = sim.launch(mod, "k", grid=2, block=64, cluster=2, args=[out])
    np.testing.assert_array_equal(out.numpy(), [64, 64])
    assert not stats.findings


def test_bad_atomic_destination_does_not_modify_memory(sim):
    target = sim.from_numpy(np.zeros(1, dtype=np.uint64))
    mod = kernel(".reg .u64 %p; .reg .u32 %old; ld.param.u64 %p, [p]; atom.add.u64 %old, [%p], 1; ret;", ".param .u64 p")
    with pytest.raises(warplens.ExecutionError, match="narrower"):
        sim.launch(mod, "k", grid=1, block=1, args=[target])
    assert target.numpy()[0] == 0


@pytest.mark.parametrize("synchronize", [False, True])
def test_atomic_write_does_not_hide_an_earlier_ordinary_writer(sim, synchronize):
    barrier = "bar.sync 0;" if synchronize else ""
    mod = kernel(f""".reg .u32 %t, %old; .reg .pred %first, %warp0; .shared .u32 cell;
        mov.u32 %t, %tid.x; setp.eq.u32 %first, %t, 0; setp.lt.u32 %warp0, %t, 32;
        @%first st.shared.u32 [cell], 0; bar.sync 0;
        @%warp0 st.shared.u32 [cell], 10;
        @%warp0 atom.shared.add.u32 %old, [cell], 1;
        {barrier}
        @!%warp0 atom.shared.add.u32 %old, [cell], 1;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=64)
    assert any(f.kind == "smem-race" for f in stats.findings) != synchronize


def test_proxy_fence_uses_latest_atomic_writer_after_multiple_contenders():
    from warplens.units.smem import SharedMemory
    smem = SharedMemory(4)
    at, dt = np.array([0], dtype=np.uint64), np.dtype("u4")
    smem.write(at, np.array([0], dtype=dt), 0)
    smem.epoch += 1
    smem.atomic(at, dt, 0, lambda old: old + 1)
    smem.atomic(at, dt, 1, lambda old: old + 1)
    smem.fence_async(0)
    assert smem.unfenced.all()  # another warp's later write is not ordered
    smem.fence_async(1)
    assert not smem.unfenced.any()  # its own final RMW is ordered


def test_proxy_fence_ownership_does_not_reuse_preserved_ordinary_race_history():
    from warplens.units.smem import SharedMemory
    smem = SharedMemory(4)
    at, dt = np.array([0], dtype=np.uint64), np.dtype("u4")
    smem.write(at, np.array([0], dtype=dt), 0)
    _, _, races = smem.atomic(at, dt, 1, lambda old: old + 1)
    assert races.tolist() == [0]
    smem.fence_async(0)
    assert smem.unfenced.all()
    smem.fence_async(1)
    assert not smem.unfenced.any()
    # Fencing does not erase the independent ordinary-store race provenance.
    _, _, races = smem.atomic(at, dt, 2, lambda old: old + 1)
    assert races.tolist() == [0]
