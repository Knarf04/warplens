"""The matrix rows that are not probes (NUM-05, RT-10, RT-11): validation/blackwell/run_rows.py
and the scripts it runs. What needs a GPU is not run here; what must hold without one is:
a step that cannot run says `skipped` with the reason, and that fails the run only for
RT-11 where its capture is required (tests/validation/capture/test_launch_capture.py has that side).
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import REPO

SUITE = REPO / "validation" / "blackwell"


def run(*command, env=None):
    return subprocess.run([sys.executable, *map(str, command)], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SUITE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SUITE))
    return module


def test_ue4m3_has_no_conversion_and_the_controls_assemble():
    """NUM-05. The verdict rests on the controls: a kernel that ptxas rejects for another
    reason must not read as 'no UE4M3 conversion'."""
    if not shutil.which("ptxas"):
        pytest.skip("needs ptxas (the CUDA toolkit)")
    done = run(SUITE / "formats" / "ue4m3_ptxas.py")
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    controls = [line for line in lines if line.split()[0] in ("e4m3x2", "e5m2x2", "e2m1x2", "ue8m0x2")]
    assert len(controls) == 8 and all("ptxas accepts" in line for line in controls)
    assert sum("ptxas rejects" in line for line in lines if line.startswith("ue4m3")) == 6
    assert lines[-1].startswith("NUM-05: no UE4M3 conversion")
    # The committed record is this output for the pinned toolkit (its first line names the build).
    committed = (SUITE / "formats" / "ue4m3_ptxas.txt").read_text().splitlines()
    assert committed[1:] == lines[1:] or committed[0] != lines[0]


def test_ue4m3_check_is_undecided_without_working_controls(tmp_path):
    """An assembler that rejects everything decides nothing (exit 2), and one that accepts
    a UE4M3 spelling is news (exit 1)."""
    rejects = tmp_path / "ptxas-no"
    rejects.write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo fake; exit 0; }\necho 'error : no' >&2\nexit 1\n")
    accepts = tmp_path / "ptxas-yes"
    accepts.write_text("#!/bin/sh\necho fake\nexit 0\n")
    for program in (rejects, accepts):
        program.chmod(0o755)
    undecided = run(SUITE / "formats" / "ue4m3_ptxas.py", "--ptxas", rejects)
    assert undecided.returncode == 2 and "NUM-05: UNDECIDED" in undecided.stdout
    news = run(SUITE / "formats" / "ue4m3_ptxas.py", "--ptxas", accepts)
    assert news.returncode == 1 and "UE4M3 CONVERSION ACCEPTED" in news.stdout
    missing = run(SUITE / "formats" / "ue4m3_ptxas.py", "--ptxas", tmp_path / "none")
    assert missing.returncode == 2


def test_the_tracer_says_skipped_without_gdb(tmp_path):
    for script, extra in (("driver_call_trace.py", ["--", "true"]), ("cute_dsl_tma.py", [])):
        done = run(SUITE / "e2e" / script, "--out", tmp_path / "out", *extra,
                   env={"WARPLENS_GDB": str(tmp_path / "no-gdb")})
        assert done.returncode == 2, done.stdout + done.stderr
        assert done.stdout.startswith("SKIPPED: gdb not found")


def test_the_tracer_logs_nothing_for_a_program_without_cuda(tmp_path):
    """With a real gdb and a program that never loads the driver: no calls, exit 1, and the
    files a reader expects are still there."""
    tracer = load("driver_call_trace_under_test", SUITE / "e2e" / "driver_call_trace.py")
    if not tracer.find_gdb()[0]:
        pytest.skip("needs gdb with Python support")
    status, facts = tracer.trace(["/bin/true"], tmp_path / "out", 120)
    if status.startswith("skipped"):
        pytest.skip(status)                      # a sandbox that forbids ptrace
    assert status.startswith("no driver call was logged") and "exit 0" in status
    assert facts == {"launches": [], "calls": []}
    assert (tmp_path / "out" / "launches.txt").read_text() == "no kernel launch reached the driver\n"
    assert (tmp_path / "out" / "exit.txt").read_text() == "exit 0\n"


def test_run_rows_records_skips_and_never_fails(tmp_path):
    """Without gdb every launching step is skipped with its reason; --no-e2e skips them
    outright. Either way the runner exits 0 and writes one line per step. (`--rt11 optional`:
    this test also runs in the self-test on the B200, where a skipped RT-11 is otherwise a
    failure; tests/validation/capture/test_launch_capture.py.)"""
    out = tmp_path / "results"
    done = run(SUITE / "run_rows.py", "--out", out, "--only", "e2e/*", "--rt11", "optional",
               env={"WARPLENS_GDB": str(tmp_path / "no-gdb")})
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [line.split("\t") for line in (out / "rows" / "status.tsv").read_text().splitlines()[1:]]
    # RT-11 first: the one row a rental cannot end without (it used to be last).
    assert [(row[0], row[1]) for row in rows] == [("e2e/cute_dsl_tma", "RT-11"),
                                                 ("e2e/driver_call_trace.tirx", "RT-10"),
                                                 ("e2e/driver_call_trace.cute_dsl", "RT-10")]
    assert all(row[2].startswith("skipped (gdb not found") for row in rows)
    assert (out / "rows" / "e2e" / "cute_dsl_tma.stdout").read_text().startswith("SKIPPED:")

    quiet = run(SUITE / "run_rows.py", "--out", tmp_path / "second", "--no-e2e", "--only", "e2e/*")
    assert quiet.returncode == 0
    assert quiet.stdout.count("skipped (--no-e2e)") == 3


def test_run_all_runs_the_rows_only_on_a_whole_gpu_run():
    """run_all.py calls run_rows.py in GPU mode, not for --only. What a step ends as does not
    fail the run: run_all.py reads no status of a row, only whether run_rows.py exited 0,
    which it does unless the RT-11 capture is required and missing."""
    source = (SUITE / "run_all.py").read_text()
    assert "with_rows = gpu and not args.only" in source
    block = source[source.index("    def run_rows() -> None:"):source.index("    for kind, batch in execution_phases(")]
    assert "run_rows.py" in block and "--no-e2e" in block
    # The run fails for the rows in two ways only: run_rows.py did not exit 0, or (--resume)
    # the rows are kept from an earlier run that left the RT-11 flag. No status of a row is read.
    assert block.count("infrastructure_failures += ") == 2 and "split(" not in block
    assert 'if status != "ok":\n            infrastructure_failures += 1' in block
    assert "infrastructure_failures += rt11_marker.is_file()" in block
    rows = (SUITE / "run_rows.py").read_text()
    assert rows.count("return 1") == 1 and "return rt11_verdict(" in rows       # the one way to fail


def test_every_row_without_a_probe_is_scripted_and_documented():
    readme = (SUITE / "README.md").read_text()
    runbook = (REPO / "env" / "remote" / "RUNBOOK.md").read_text()
    for row, script in (("NUM-05", "formats/ue4m3_ptxas.py"), ("RT-10", "e2e/driver_call_trace.py"),
                        ("RT-11", "e2e/cute_dsl_tma.py")):
        assert (SUITE / script).is_file()
        assert row in readme and script in readme
        assert row in runbook
    assert "rows/status.tsv" in runbook
