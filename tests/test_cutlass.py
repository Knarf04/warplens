"""CUTLASS's own Blackwell tutorial, unmodified. Needs examples/03_cutlass_mma/build.sh
to have been run (it fetches CUTLASS and compiles the tutorial); skipped otherwise."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "03_cutlass_mma"
pytestmark = pytest.mark.skipif(not (EXAMPLE / "tutorial_01.ptx").exists(),
                                reason="tutorial_01.ptx not built")


def load_runner():
    spec = importlib.util.spec_from_file_location("cutlass_run", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


@pytest.mark.parametrize("dims, alpha, beta", [
    ((128, 256, 64), 1.0, 0.0),  # one tile
    ((128, 256, 256), 1.0, 0.0),  # several K tiles accumulate
    ((256, 512, 64), 0.5, -2.0),  # four CTAs, and the alpha/beta epilogue
])
def test_tutorial_01_gemm_matches_numpy(dims, alpha, beta):
    got, reference, stats = load_runner()(*dims, alpha=alpha, beta=beta)
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)
    assert not stats.findings
    assert stats.output  # the tutorial prints its layouts from thread 0
