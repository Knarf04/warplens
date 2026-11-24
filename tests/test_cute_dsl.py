"""The CuTe DSL example: one script, compiled and launched by the DSL itself, with its CUDA
runtime calls served by `warplens run`. Skipped when the DSL (`nvidia-cutlass-dsl`) is not
installed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "06_cute_dsl" / "run.py"


def run(*arguments):
    return subprocess.run([sys.executable, "-m", "warplens.cli", "run", *arguments, sys.executable, str(EXAMPLE)],
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


@pytest.mark.parametrize("model", ["B200", "B300"])
def test_cute_dsl_script_launches_in_the_simulator_without_a_gpu(model):
    done = run("--model", model)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    for block in range(2):
        for thread in range(3):
            assert f"thread {thread} of block {block} says n = 7" in done.stdout
    assert "thread 3 of" not in done.stdout and "launched 2 blocks of 32 threads" in done.stdout
    assert f"1 kernel launch(es) simulated on {model}, 0 finding(s)" in done.stderr
    assert ": 2 CTA(s)" in done.stderr
