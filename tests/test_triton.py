"""Triton kernels through the warplens driver. Skipped when Triton is not installed."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

triton = pytest.importorskip("triton")
warnings.filterwarnings("ignore", module="triton")

import warplens  # noqa: E402
import warplens.triton_driver as wt  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "05_triton" / "run.py"
spec = importlib.util.spec_from_file_location("triton_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def gpu():
    driver = wt.install("B200")
    yield driver
    wt.uninstall()


def test_vector_add_through_triton_launch_syntax(gpu):
    got, reference, stats = example.vector_add(gpu, n=1000, block=128)
    np.testing.assert_array_equal(got, reference)
    assert stats.ctas == 8 and stats.threads == 8 * 128 and not stats.findings
    assert gpu.launches[-1][0] == "add_kernel"


@pytest.mark.parametrize("dims", [(128, 64, 32), (256, 128, 96), (130, 70, 50)])
def test_matmul_uses_the_blackwell_tensor_core_and_matches_numpy(gpu, dims):
    got, reference, stats = example.matmul(gpu, *dims)
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)
    assert not stats.findings
    kernel = next(iter(example.matmul_kernel.device_caches[0][0].values()))
    ptx = kernel.asm["ptx"]
    assert ".target sm_100a" in ptx and "tcgen05.mma" in ptx  # compiled for Blackwell, no GPU involved
    assert kernel.metadata.tmem_size > 0


def test_launch_options_reach_the_simulator():
    driver = wt.install("B200", timing=True)
    try:
        _, _, stats = example.vector_add(driver, n=256, block=128)
        assert stats.timing is not None and stats.timing.time_ns > 0
    finally:
        wt.uninstall()


def test_simulator_errors_surface_through_triton(gpu):
    a = np.zeros(100, dtype=np.float32)
    small = gpu.empty(10, "float32")  # too small for 100 elements
    with pytest.raises(warplens.MemoryFault, match="outside every allocation"):
        example.add_kernel[(1,)](gpu.tensor(a), gpu.tensor(a), small, 100, BLOCK=128)


def test_tensor_wrapper(gpu):
    t = gpu.tensor(np.arange(6, dtype=np.int32).reshape(2, 3))
    assert t.shape == (2, 3) and str(t.dtype) == "int32" and t.data_ptr() == t.array.addr
    assert t.numpy().tolist() == [[0, 1, 2], [3, 4, 5]] and "warplens Tensor" in repr(t)
    with pytest.raises(NotImplementedError, match="approximate timing model"):
        gpu.get_benchmarker()
    assert gpu.get_current_target().arch == 100
