"""Launch shape limits must agree with the attributes advertised by the shim."""

import pytest

from conftest import kernel
from warplens.errors import CUDA_INVALID_CONFIGURATION, LaunchError
from warplens.host.launch import MAX_BLOCK_DIM, MAX_GRID_DIM, _check_dimensions


@pytest.mark.parametrize("block", [(1024, 1, 1), (1, 1024, 1), (1, 1, 64)])
def test_block_axis_boundaries_execute(sim, block):
    stats = sim.launch(kernel("ret;"), "k", grid=1, block=block, args=[])
    assert stats.threads == block[0] * block[1] * block[2]
    assert not stats.findings


@pytest.mark.parametrize("what,dims", [
    ("block", (1025, 1, 1)), ("block", (1, 1025, 1)), ("block", (1, 1, 65)),
    ("grid", (2**31, 1, 1)), ("grid", (1, 65536, 1)), ("grid", (1, 1, 65536)),
])
def test_oversized_axis_is_refused_before_execution(sim, what, dims):
    # Invalid grids must be rejected before allocating a potentially enormous queue.
    config = {"block": (1, 1, 1), "grid": (1, 1, 1), what: dims}
    with pytest.raises(LaunchError) as caught:
        sim.launch(kernel("ret;"), "k", **config, args=[])
    assert caught.value.cuda_error == CUDA_INVALID_CONFIGURATION
    assert "limit" in str(caught.value)


@pytest.mark.parametrize("axis", range(3))
def test_grid_boundary_is_accepted_without_executing_a_huge_grid(axis):
    dims = tuple(MAX_GRID_DIM[i] if i == axis else 1 for i in range(3))
    _check_dimensions(dims, MAX_GRID_DIM, "grid")


def test_per_axis_checks_do_not_replace_the_total_thread_limit(sim):
    _check_dimensions((32, 33, 1), MAX_BLOCK_DIM, "block")
    with pytest.raises(LaunchError, match="1056 threads") as caught:
        sim.launch(kernel("ret;"), "k", grid=1, block=(32, 33, 1), args=[])
    assert caught.value.cuda_error == CUDA_INVALID_CONFIGURATION
