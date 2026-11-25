"""M0 acceptance: kernels compiled by nvcc for sm_100a run on the CPU and match NumPy."""

import importlib.util

import numpy as np
import pytest

from tests.support import EXAMPLES

VECADD = EXAMPLES / "00_vecadd" / "vecadd.ptx"
DIVERGENT = EXAMPLES / "00_divergent" / "divergent.ptx"


@pytest.mark.parametrize("n, grid, block", [(1, 1, 1), (33, 1, 64), (1000, 4, 256), (4096, 4, 1024), (777, 13, 60)])
def test_vecadd(sim, n, grid, block):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = sim.from_numpy(np.full(n, np.nan, dtype=np.float32))
    stats = sim.launch(VECADD, "vecadd", grid=grid, block=block,
                       args=[sim.from_numpy(a), sim.from_numpy(b), c, n])
    np.testing.assert_array_equal(c.numpy(), a + b)
    assert stats.threads == grid * block


def collatz_steps(v: int) -> int:
    count = 0
    while v > 1:
        v = (3 * v + 1) & 0xFFFFFFFF if v & 1 else v // 2
        count += 1
    return count


def test_collatz_data_dependent_loop(sim):
    n = 500
    x = np.random.default_rng(1).integers(0, 100_000, n).astype(np.uint32)
    x[:4] = [0, 1, 2, 27]
    steps = sim.empty(n, "u32")
    stats = sim.launch(DIVERGENT, "collatz", grid=4, block=128, args=[sim.from_numpy(x), steps, n])
    assert steps.numpy().tolist() == [collatz_steps(int(v)) for v in x]
    assert steps.numpy()[3] == 111
    assert stats.divergent_branches > 0
    assert stats.max_simt_depth <= 3  # loop divergence must not grow the stack


def test_classify_nested_branches_with_early_exit(sim):
    rows, cols = 150, 23
    data = np.random.default_rng(2).integers(-6, 7, (rows, cols)).astype(np.int32)
    data[0] = 5  # never breaks
    data[1, 0] = 0  # breaks immediately
    total, neg = sim.empty(rows, "s32"), sim.empty(rows, "s32")
    sim.launch(DIVERGENT, "classify", grid=3, block=64,
               args=[sim.from_numpy(data), cols, total, neg, rows])
    stop = np.where((data == 0).any(axis=1), (data == 0).argmax(axis=1), cols)
    live = np.arange(cols) < stop[:, None]
    np.testing.assert_array_equal(total.numpy(), np.where(live & (data > 0), data, 0).sum(axis=1))
    np.testing.assert_array_equal(neg.numpy(), (live & (data < 0)).sum(axis=1))


@pytest.mark.parametrize("stages", [0, -1, 5, 1.5, True])
def test_example_rejects_stage_counts_outside_its_kernel_storage(stages):
    path = EXAMPLES / "04_pipelined_gemm/run.py"
    spec = importlib.util.spec_from_file_location("gemm_run_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="stages must be an integer from 1 to 4"):
        module.run(stages)
