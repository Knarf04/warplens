"""The CuTe DSL example: compile with the DSL, run the PTX in warplens. Skipped when the
DSL (`nvidia-cutlass-dsl`) is not installed."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "06_cute_dsl" / "run.py"


def test_cute_dsl_kernel_compiles_without_a_gpu_and_runs_in_the_simulator():
    done = subprocess.run([sys.executable, str(EXAMPLE)], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "lines of PTX" in done.stdout and "warplens ran `" in done.stdout
    for thread in range(3):
        assert f"thread {thread} says n = 7" in done.stdout
    assert "thread 3 says" not in done.stdout
