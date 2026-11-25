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
    assert "the last thread is 63 of block 3" in done.stdout  # through a call to a device function
    assert "1 kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr
    assert "saxpy_kernel: 4 CTA(s)" in done.stderr


def test_tirx_sees_the_selected_model():
    done = run("--model", "B300")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "on compute capability 10.3: matches NumPy" in done.stdout


def test_tirx_script_needs_no_python_in_front_of_it():
    """`warplens run script.py` is `warplens run python script.py`, as for a Triton script."""
    done = subprocess.run([sys.executable, "-m", "warplens.cli", "run", "--timing", str(EXAMPLE)],
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "on compute capability 10.0: matches NumPy" in done.stdout
    assert "1 kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr
    assert "saxpy_kernel: 4 CTA(s)" in done.stderr and " ns simulated (" in done.stderr


def test_run_program_returns_what_the_command_only_prints(monkeypatch):
    """`warplens.run_program` is `warplens run` from Python: the same launch, with its
    statistics, PTX and timing report kept (examples/08_book/book_tour.ipynb is built on it)."""
    import warplens
    from warplens.viz import timeline_svg
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seen = []
    done = warplens.run_program([sys.executable, EXAMPLE], timing=True,
                                trace=lambda warp, pc, ins, active: seen.append(ins.name))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "on compute capability 10.0: matches NumPy" in done.stdout
    launch = done.launch
    assert launch.kernel.endswith("saxpy_kernel") and launch.grid[0] == 4 and launch.cluster is None
    assert launch.stats.ctas == 4 and not launch.stats.findings
    assert launch.stats.timing.time_ns > 0 and timeline_svg(launch.stats.timing).startswith("<svg")
    # The trace also sees the device function the kernel calls; `instructions` is the kernel's own text.
    assert {i.name for i in launch.instructions} & set(seen) and any(i.name.startswith("call") for i in launch.instructions)
    failed = warplens.run_program([sys.executable, "-c", "import sys; sys.exit('no')"])
    assert failed.returncode == 1 and failed.stderr.strip() == "no" and not failed.launches
    with pytest.raises(RuntimeError, match="0 launches"):
        failed.launch
