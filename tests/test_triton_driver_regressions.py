"""Triton driver behavior with hardware discovery explicitly disabled."""

import importlib
import gc
import weakref

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl
import warplens.triton_driver as wt
from warplens.errors import LaunchError


@triton.jit
def scalar_kernel(out, value):
    tl.store(out, value)


@triton.jit
def copy_kernel(source, out):
    tl.store(out, tl.load(source))


@triton.jit
def tuple_kernel(out, values):
    tl.store(out, values[0] + values[1][0] + values[1][1])


@triton.jit
def cluster_kernel(out):
    pid = tl.program_id(0)
    tl.store(out + pid, pid + 1)


@triton.jit
def print_kernel():
    tl.device_print("driver-value", 7)


@pytest.fixture(autouse=True)
def deny_hardware(monkeypatch):
    module = importlib.import_module("triton.runtime.driver")

    def forbidden():
        raise AssertionError("hardware driver discovery is forbidden")

    monkeypatch.setattr(module, "_create_driver", forbidden)
    monkeypatch.setattr(triton.runtime.driver, "_active", None)
    monkeypatch.setattr(triton.runtime.driver, "_default", None)
    if hasattr(wt, "_installed"):
        monkeypatch.setattr(wt, "_installed", [])


@pytest.fixture
def gpu():
    driver = wt.WarplensDriver()
    triton.runtime.driver.set_active(driver)
    return driver


def test_install_uninstall_preserves_lazy_hardware_state():
    assert wt.active_driver() is None
    wt.install()
    wt.uninstall()
    assert wt.active_driver() is None
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    wt.uninstall()  # harmless when already restored


def test_nested_install_restores_the_previous_driver(gpu):
    inner = wt.install("B300")
    assert triton.runtime.driver.active is inner
    wt.uninstall()
    assert triton.runtime.driver.active is gpu


@pytest.mark.parametrize("launch", [False, True])
def test_compiled_kernel_cleanup_never_initializes_hardware(launch):
    driver = wt.install()
    out = driver.empty(1, "int32")
    if launch:
        kernel = scalar_kernel[(1,)](out, 61)
    else:
        kernel = scalar_kernel.warmup(out, 67, grid=(1,))
        kernel._init_handles()
    wt.uninstall()
    kernel.__del__()
    assert triton.runtime.driver._active is None


def test_cached_kernel_collection_after_uninstall_with_launch_hooks(monkeypatch):
    driver = wt.install()
    events = []
    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", lambda data: events.append(data.get()["function"]))
    monkeypatch.setattr(triton.knobs.runtime, "launch_exit_hook", lambda data: None)
    out = driver.empty(1, "int32")
    kernel = scalar_kernel[(1,)](out, 71)
    assert events == [kernel.function]
    reference = weakref.ref(kernel)
    kernels = scalar_kernel.device_caches[driver.get_current_device()][0]
    for key, value in list(kernels.items()):
        if value is kernel:
            del kernels[key]
    del value
    wt.uninstall()
    # No hardware backend is active while the compiled kernel is destroyed.
    del kernel
    gc.collect()
    assert reference() is None
    assert triton.runtime.driver._active is None
    driver = wt.install()
    scalar_kernel[(1,)](driver.empty(1, "int32"), 71)
    wt.uninstall()


def test_hardware_and_simulator_kernel_caches_do_not_collide(gpu):
    # JITFunction keys its binder/target/compiled launcher by current device.
    # Device zero may already contain kernels compiled for a real GPU.
    previous = dict(scalar_kernel.device_caches)
    scalar_kernel.device_caches.clear()
    scalar_kernel.device_caches[0] = "hardware cache must not be used"
    try:
        out = gpu.empty(1, "int32")
        scalar_kernel[(1,)](out, 31)
        assert out.numpy().tolist() == [31]
        assert scalar_kernel.device_caches[0] == "hardware cache must not be used"
    finally:
        scalar_kernel.device_caches.clear()
        scalar_kernel.device_caches.update(previous)


def test_reinstalled_driver_can_reuse_simulator_kernel_cache(gpu):
    out = gpu.empty(1, "int32")
    scalar_kernel[(1,)](out, 17)
    another = wt.WarplensDriver("B300")
    triton.runtime.driver.set_active(another)
    out2 = another.empty(1, "int32")
    scalar_kernel[(1,)](out2, 17)
    assert out2.numpy().tolist() == [17]


def test_foreign_device_tensor_cannot_alias_an_active_allocation(gpu):
    foreign = wt.WarplensDriver().tensor(np.array([99], np.int32))
    local = gpu.tensor(np.array([7], np.int32))
    assert foreign.data_ptr() == local.data_ptr()
    out = gpu.empty(1, "int32")
    with pytest.raises(LaunchError, match="different.*device"):
        copy_kernel[(1,)](foreign, out)
    assert out.numpy().tolist() == [0]


def test_foreign_pointer_wrapper_is_rejected(gpu):
    local = gpu.tensor(np.array([7], np.int32))

    class ForeignTensor:
        dtype = np.dtype("int32")
        def data_ptr(self):
            return local.data_ptr()

    with pytest.raises(LaunchError, match="warplens.*tensor"):
        copy_kernel[(1,)](ForeignTensor(), gpu.empty(1, "int32"))


def test_nested_tuple_parameters_and_constexprs(gpu):
    out = gpu.empty(1, "int32")
    tuple_kernel[(1,)](out, (5, (7, tl.constexpr(3))))
    assert out.numpy().tolist() == [15]


def test_zero_grid_is_a_noop(gpu):
    out = gpu.tensor(np.array([9], np.int32))
    scalar_kernel[(0,)](out, 17)
    assert out.numpy().tolist() == [9]
    assert gpu.launches == []


def test_cluster_grid_counts_triton_programs_not_individual_ctas(gpu):
    out = gpu.empty(3, "int32")
    cluster_kernel[(3,)](out, num_ctas=2)
    assert out.numpy().tolist() == [1, 2, 3]
    assert gpu.last_stats.ctas == 6


def test_launch_hooks_surround_simulation(gpu, monkeypatch):
    out = gpu.empty(1, "int32")
    events = []
    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", lambda data: events.append(("enter", out.numpy().item())))
    monkeypatch.setattr(triton.knobs.runtime, "launch_exit_hook", lambda data: events.append(("exit", out.numpy().item())))
    scalar_kernel[(1,)](out, 23)
    assert events == [("enter", 0), ("exit", 23)]


def test_device_print_reaches_stdout(gpu, capsys):
    print_kernel[(1,)]()
    assert "driver-value" in capsys.readouterr().out


def test_profile_scratch_is_rejected_before_execution(gpu):
    out = gpu.empty(1, "int32")
    kernel = scalar_kernel[(1,)](out, 11)
    launcher = kernel.run
    original = launcher.metadata
    launcher.metadata = original._replace(profile_scratch_size=16)
    try:
        with pytest.raises(LaunchError, match="profile scratch"):
            scalar_kernel[(1,)](out, 11)
    finally:
        launcher.metadata = original
