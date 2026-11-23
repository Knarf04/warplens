"""CUTLASS's own Blackwell tutorials, unmodified. Needs examples/03_cutlass_mma/build.sh
to have been run (it fetches CUTLASS and compiles the tutorials); skipped otherwise."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "03_cutlass_mma"


def load_runner():
    spec = importlib.util.spec_from_file_location("cutlass_run", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


@pytest.mark.parametrize("tutorial", [1, 2])  # 1: threads fill shared memory; 2: TMA does
@pytest.mark.parametrize("dims, alpha, beta", [
    ((128, 256, 64), 1.0, 0.0),  # one tile
    ((128, 256, 256), 1.0, 0.0),  # several K tiles accumulate
    ((256, 512, 64), 0.5, -2.0),  # four CTAs, and the alpha/beta epilogue
])
def test_tutorial_gemm_matches_numpy(tutorial, dims, alpha, beta):
    if not (EXAMPLE / f"tutorial_0{tutorial}.ptx").exists():
        pytest.skip(f"tutorial_0{tutorial}.ptx not built")
    got, reference, stats = load_runner()(tutorial, *dims, alpha=alpha, beta=beta)
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)
    assert not stats.findings
    assert stats.output  # the tutorials print their layouts from thread 0


@pytest.mark.parametrize("tutorial", [3, 4])  # 3: TMA multicast in 4 x 4 clusters; 4: two-SM MMA
@pytest.mark.parametrize("dims, alpha, beta", [
    ((512, 1024, 64), 1.0, 0.0),  # one cluster of 16 CTAs
    ((512, 1024, 192), 0.5, -2.0),  # several K tiles, and the alpha/beta epilogue
])
def test_cluster_tutorial_gemm_matches_numpy(tutorial, dims, alpha, beta):
    if not (EXAMPLE / f"tutorial_0{tutorial}.ptx").exists():
        pytest.skip(f"tutorial_0{tutorial}.ptx not built")
    got, reference, stats = load_runner()(tutorial, *dims, alpha=alpha, beta=beta)
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)
    assert not stats.findings and stats.ctas == 16
