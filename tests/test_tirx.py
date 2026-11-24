"""The TIRx example: one script, launched by TVM's runtime through the CUDA driver calls that
`warplens run` serves. Skipped when TIRx (`apache-tvm-tirx`) is not installed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tvm_tirx")
pytest.importorskip("cuda.bindings")  # TIRx compiles through NVRTC

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "07_tirx" / "run.py"


def run(*arguments, **env):
    return subprocess.run([sys.executable, "-m", "warplens.cli", "run", *arguments, sys.executable, str(EXAMPLE)],
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env})


def test_tirx_script_compiles_for_blackwell_and_launches_in_the_simulator():
    done = run()
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "on compute capability 10.0: matches NumPy" in done.stdout
    assert "1 kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr
    assert "saxpy_kernel: 4 CTA(s)" in done.stderr


def test_tirx_sees_the_selected_model():
    done = run("--model", "B300")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "on compute capability 10.3: matches NumPy" in done.stdout
