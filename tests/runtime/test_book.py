"""The GEMM kernels of the book *Modern GPU Programming for MLSys*, read from a clone of the
book and launched by TVM's runtime under `warplens run` (examples/08_book). The book has no
license, so it is not in this repository: set WARPLENS_BOOK to a clone. CI checks out the
commit below. Skipped without the clone or without TIRx."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import EXAMPLES

pytest.importorskip("tvm_tirx")
pytest.importorskip("cuda.bindings")

COMMIT = "3a24144defa6da03d4ab8a1dfcf513b18007d15f"  # what these expectations were recorded against
RUNNER = EXAMPLES / "08_book" / "run.py"
BOOK = os.environ.get("WARPLENS_BOOK", "")

pytestmark = pytest.mark.skipif(not (Path(BOOK) / "chapter_gemm_basics" / "index.md").is_file() or not BOOK,
                                reason="WARPLENS_BOOK is not a clone of the book")


def run(*versions):
    return subprocess.run([sys.executable, "-m", "warplens.cli", "run", sys.executable, str(RUNNER), BOOK, *versions],
                          capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_all_nine_gemm_kernels_match_numpy_in_the_simulator():
    done = run()
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    sizes = {**{f"v{n}": "128x128x64" for n in range(1, 6)}, "v6": "128x128x256", "v7": "128x128x256",
             "v8": "256x256x256", "v9": "512x256x256"}
    for version, size in sizes.items():
        assert f"hgemm_{version} {size}: matches NumPy" in done.stdout
    assert "9 kernel launch(es) simulated on B200" in done.stderr
    # v7 to v9 release their tensor memory once per warpgroup; nothing else is reported.
    assert "3 finding(s)" in done.stderr and done.stderr.count("tmem-dealloc-repeated") == 3
