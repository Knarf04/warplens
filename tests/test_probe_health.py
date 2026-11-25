"""What keeps a rental's results when a probe hurts the GPU (validation/blackwell/run_all.py,
probe_process.py, compare.py, run_rows.py).

* the order of a GPU run: the probes without a risk, the end-to-end programs, the rows
  without a probe, then the probes that may fault, then those that may hang;
* the canary after every run that did not end as ok or has a risk, health.tsv, xid.log, and
  the circuit breaker that stops launching when the canary does not come back;
* a killed process that does not exit is not waited for without a bound, and a killed
  probe keeps the lines it had printed;
* compare.py: a timeout is "as predicted" only where the simulator predicts a deadlock, a
  matching KERNEL FAILED is "identical" only where the simulator models the refusal, and the
  status.tsv of a simulator's run in a rental's archive is never taken for the GPU's.

No GPU and no nvcc are needed: the probes here are shell scripts, as in
tests/test_rental_harness.py. None of this has met a GPU that really hung; what was run on
the RTX 3080 Ti is in validation/blackwell/README.md, "Dry run".
"""

import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "validation" / "blackwell"
sys.path.insert(0, str(SUITE))
import manifest  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SUITE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SUITE))
    return module


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# ---------------------------------------------------------------------------------------------
# The order of a run
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_all_module():
    return load("run_all_health_under_test", SUITE / "run_all.py")


def test_a_gpu_run_puts_what_may_hurt_the_gpu_last(run_all_module):
    phases = run_all_module.execution_phases(manifest.PROBES, "gpu", True, True)
    assert [kind for kind, _ in phases] == ["probes", "e2e", "rows", "probes", "probes"]
    safe, fault, hang = (batch for kind, batch in phases if kind == "probes")
    assert safe and all(p.risk == "" for p in safe)
    assert fault and all(p.risk == "fault" for p in fault)
    assert hang and all(p.risk == "hang" for p in hang)
    # Every probe once, and inside a group the manifest's order.
    assert sorted(p.name for p in safe + fault + hang) == sorted(p.name for p in manifest.PROBES)
    for batch in (safe, fault, hang):
        assert batch == [p for p in manifest.PROBES if p in batch]
    # --only and --no-e2e leave the other steps out and keep the order of the probes.
    some = [p for p in manifest.PROBES if p.name.startswith("tmem/")]
    only = run_all_module.execution_phases(some, "gpu", False, False)
    assert [kind for kind, _ in only] == ["probes"] * 3
    assert [p.risk for _, batch in only for p in batch] == sorted(p.risk for p in some)
    assert [kind for kind, _ in run_all_module.execution_phases(some, "gpu", False, True)] == \
        ["probes", "rows", "probes", "probes"]


def test_the_simulator_modes_keep_the_manifest_order(run_all_module):
    """The predictions are written by a run in this order; nothing about it may move."""
    for mode in ("sim", "predict"):
        assert run_all_module.execution_phases(manifest.PROBES, mode, True, False) == \
            [("probes", manifest.PROBES), ("e2e", [])]
        assert run_all_module.execution_phases(manifest.PROBES, mode, False, False) == [("probes", manifest.PROBES)]


def test_the_step_that_may_fault_is_the_last_end_to_end_step():
    source = (SUITE / "e2e" / "run_e2e.py").read_text()
    assert 'assert steps[-1][0] == "e2e/book_hgemm"' in source
    assert list(manifest.E2E)[-1] == "e2e/book_hgemm"


def test_rt11_runs_first_and_the_optional_rows_get_minutes(tmp_path, monkeypatch):
    rows = load("run_rows_order_under_test", SUITE / "run_rows.py")
    calls = []

    def run_process(command, timeout, **_):
        calls.append(([str(word) for word in command], timeout))
        return "ok", 0.0, b"", b""

    monkeypatch.setattr(rows, "run_process", run_process)
    monkeypatch.setattr(sys, "argv", ["run_rows.py", "--out", str(tmp_path), "--rt11", "optional"])
    assert rows.main() == 0
    assert ["cute_dsl_tma.py" in " ".join(command) for command, _ in calls] == [True, False, False, False]
    assert [timeout for _, timeout in calls] == [3000, 300, 300, 300]
    for command, _ in calls[2:]:                    # the traces: their own limit is inside the step's
        assert float(command[command.index("--timeout") + 1]) < 300
    names = [line.split("\t")[0] for line in (tmp_path / "rows" / "status.tsv").read_text().splitlines()[1:]]
    assert names[0] == "e2e/cute_dsl_tma" and len(names) == 4


# ---------------------------------------------------------------------------------------------
# The canary and the circuit breaker, with stand-in programs
# ---------------------------------------------------------------------------------------------

SAFE, FAULT_FIRST, FAULT, HANG, HANG_LATER = ("mbarrier/phase_parity", "mbarrier/complete_before_expect",
                                              "mbarrier/over_arrival", "mbarrier/expect_tx_large", "tmem/alloc_leak")


@pytest.fixture
def suite(tmp_path):
    """A copy of the harness with five stand-in runs: one without a risk, two that may fault,
    two that may hang. `mbarrier/script basic` is also the canary. The 'GPU' is the file
    state/dead: while it holds a number above zero the canary fails (and counts it down);
    `mbarrier/script over` (mbarrier/over_arrival) writes what state/break-with holds into it,
    if that is above zero."""
    root = tmp_path / "validation" / "blackwell"
    root.mkdir(parents=True)
    for name in ("manifest.py", "run_all.py", "compare.py", "probe_process.py", "capture.py", "e2e_checks.py",
                 "e2e_compare.py"):
        shutil.copy(SUITE / name, root / name)
    (root / "provenance.py").write_text("def archive_probes(*args, **kwargs): return []\n"
                                        "def validate_probe_archive(*args, **kwargs): return []\n")
    shutil.copytree(SUITE / "predicted", root / "predicted")
    state = tmp_path / "state"
    state.mkdir()
    (state / "dead").write_text("0")
    (state / "break-with").write_text("0")
    predicted = root / "predicted"
    script(root / "bin" / "mbarrier" / "script", f'''echo "$1" >> {state}/calls
case "$1" in
  basic) n=$(cat {state}/dead)
         if [ "$n" -gt 0 ]; then echo $((n - 1)) > {state}/dead
           echo "HOST ERROR: cudaMalloc(&p, 4) returned cudaErrorUnknown"; exit 2; fi
         cat {predicted}/mbarrier/phase_parity.txt ;;
  negative) cat {predicted}/mbarrier/complete_before_expect.txt ;;
  over) [ "$(cat {state}/break-with)" -gt 0 ] && cat {state}/break-with > {state}/dead
        echo "KERNEL FAILED: mbar_script"; echo "info: error cudaErrorLaunchFailure"
        echo "KERNEL FAILED: mbar_script: cudaErrorLaunchFailure at synchronize (unspecified launch failure)" >&2
        exit 3 ;;
esac
''')
    script(root / "bin" / "mbarrier" / "expect_tx", f'echo "$1" >> {state}/calls\n'
           f'cat {predicted}/mbarrier/expect_tx_large.txt\nexit 3\n')
    script(root / "bin" / "tmem" / "alloc", f'echo "$1" >> {state}/calls\ncat {predicted}/tmem/alloc_leak.txt\n')
    nvcc = json.loads((SUITE / "predicted" / "provenance.json").read_text())["nvcc_version"]
    (root / "bin" / "toolkit.txt").write_text(f"nvcc_version={nvcc}\n")
    # The machine: a kernel log that names an Xid once the 'GPU' is dead, and an nvidia-smi.
    tools = tmp_path / "tools"
    script(tools / "dmesg", f'echo "[    1.0] NVRM: loading NVIDIA UNIX x86_64 Kernel Module"\n'
           f'[ "$(cat {state}/dead)" -gt 0 ] && echo "[  99.5] NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."\n'
           'exit 0\n')
    script(tools / "nvidia-smi", 'echo "stand-in nvidia-smi $*"\n')
    return root


def run_all(suite: Path, *args: str, only=(SAFE, FAULT_FIRST, FAULT, HANG, HANG_LATER)):
    command = [sys.executable, str(suite / "run_all.py"), "--no-build", "--binaries", "ptx", "--timeout-scale", "0.004",
               "--out", str(suite / "results" / "run"), *args]
    for name in only:
        command += ["--only", name]
    tools = suite.parent.parent / "tools"
    return subprocess.run(command, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"})


def table(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines()[1:]]


def compare(suite: Path):
    return subprocess.run([sys.executable, str(suite / "compare.py"), str(suite / "results" / "run")],
                          capture_output=True, text=True, timeout=300)


def test_a_healthy_run_checks_the_canary_after_every_risky_run(suite):
    done = run_all(suite)
    assert done.returncode == 0, done.stdout + done.stderr
    out = suite / "results" / "run"
    rows = table(out / "status.tsv")
    # Without a risk, then may fault (manifest order), then may hang; a start time on each.
    assert [row[0] for row in rows] == [SAFE, FAULT_FIRST, FAULT, HANG, HANG_LATER]
    assert [row[1] for row in rows] == ["ok", "ok", "exit 3", "exit 3", "ok"]
    assert (out / "status.tsv").read_text().splitlines()[0].split("\t") == \
        ["name", "status", "seconds", "command", "started_utc"]
    for row in rows:
        assert len(row) == 5 and row[4].endswith("Z") and row[4][4] == "-" and "T" in row[4]
    assert [row[4] for row in rows] == sorted(row[4] for row in rows)
    health = table(out / "health.tsv")
    assert [row[0] for row in health] == ["(start)", FAULT_FIRST, FAULT, HANG, HANG_LATER]     # not after SAFE: ok, no risk
    assert all(row[1] == "ok" and row[3] == "1" and row[5] == "" for row in health)
    assert not (out / "GPU-UNHEALTHY.txt").exists()
    log = (out / "xid.log").read_text().splitlines()
    assert log[0].startswith("# run started ") and log[1].startswith("(before the run)\t") and len(log) == 2
    with tarfile.open(str(out) + ".tar.gz") as tar:
        assert {"run/health.tsv", "run/xid.log", "run/status.tsv"} <= set(tar.getnames())


def test_a_gpu_that_does_not_come_back_stops_the_run(suite):
    state = suite.parent.parent / "state"
    (state / "break-with").write_text("1000")                # mbarrier/over_arrival kills the 'GPU' for good
    done = run_all(suite)
    assert done.returncode == 1
    out = suite / "results" / "run"
    rows = table(out / "status.tsv")
    assert [row[0] for row in rows] == [SAFE, FAULT_FIRST, FAULT, HANG, HANG_LATER]      # every run has a line
    assert [row[1] for row in rows] == ["ok", "ok", "exit 3"] + [f"not run (GPU unhealthy after {FAULT})"] * 2
    calls = (state / "calls").read_text().split()
    assert "wait-large" not in calls and "leak" not in calls                          # nothing was launched after it
    last = table(out / "health.tsv")[-1]
    assert last[0] == FAULT and last[1] == "host-error" and last[3] == "5" and last[5].startswith("GPU UNHEALTHY")
    flag = (out / "GPU-UNHEALTHY.txt").read_text()
    assert flag.startswith(f"THE GPU WAS UNHEALTHY AFTER {FAULT}\n") and "--resume" in flag
    assert flag in done.stdout and "2 WERE NOT BUILT OR NOT RUN" in done.stdout
    assert {f.name for f in (out / "health").iterdir()} == {f"nvidia-smi.mbarrier_over_arrival.{when}.txt"
                                                            for when in ("first-failure", "unhealthy")}
    # The kernel log line that appeared is filed under the run after which it was first seen.
    assert f"{FAULT}\t[  99.5] NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus." in \
        (out / "xid.log").read_text().splitlines()

    compared = compare(suite)
    assert compared.returncode == 1
    assert compared.stdout.startswith("!" * 100 + f"\nTHE GPU WAS UNHEALTHY AFTER {FAULT}\n")
    lines = {line.split()[0]: line for line in compared.stdout.splitlines() if line.startswith(("mbarrier/", "tmem/"))}
    for name in (HANG, HANG_LATER):
        assert "NOT RUN, GPU unhealthy" in lines[name] and f"after {FAULT}" in lines[name]
    assert f"{HANG}:" not in compared.stdout.split("Details")[-1]                      # not a finding about the probe
    assert "GPU health (health.tsv)" in compared.stdout and "1 kernel log line(s) naming Xid or NVRM" in compared.stdout
    assert "; 1 name an Xid, the first:\n    mbarrier/over_arrival\t[  99.5] NVRM: Xid (PCI:0000:01:00): 79" in compared.stdout

    # After a reset: --resume runs what was left out and repeats nothing that has a status.
    (state / "dead").write_text("0")
    (state / "break-with").write_text("0")
    (state / "calls").write_text("")
    again = run_all(suite, "--resume")
    assert again.returncode == 0, again.stdout + again.stderr
    assert [row[1] for row in table(out / "status.tsv")] == ["ok", "ok", "exit 3", "exit 3", "ok"]
    calls = (state / "calls").read_text().split()
    assert "over" not in calls and "negative" not in calls and {"wait-large", "leak"} <= set(calls)
    assert not (out / "GPU-UNHEALTHY.txt").exists()


def test_a_canary_that_recovers_is_recorded_and_the_run_goes_on(suite):
    state = suite.parent.parent / "state"
    (state / "break-with").write_text("2")                   # the canary fails twice after over_arrival, then passes
    done = run_all(suite)
    assert done.returncode == 0, done.stdout + done.stderr
    out = suite / "results" / "run"
    assert [row[1] for row in table(out / "status.tsv")] == ["ok", "ok", "exit 3", "exit 3", "ok"]
    event = next(row for row in table(out / "health.tsv") if row[0] == FAULT)
    assert event[1] == "ok" and event[3] == "3"
    assert event[5] == "FAILED THEN RECOVERED: first host-error; passed at attempt 3"
    assert not (out / "GPU-UNHEALTHY.txt").exists()
    assert f"the canary failed and then passed after: {FAULT}" in done.stdout
    assert (out / "health" / "nvidia-smi.mbarrier_over_arrival.first-failure.txt").is_file()
    compared = compare(suite)
    assert f"    {FAULT}: FAILED THEN RECOVERED" in compared.stdout and "NOT RUN, GPU unhealthy" not in compared.stdout


def test_a_canary_that_never_ran_stops_nothing(suite):
    """The rehearsal on a GPU that runs none of the programs (bin/ on an RTX 3080 Ti): the
    canary fails from the start, so its failures say nothing and the breaker stays open."""
    (suite.parent.parent / "state" / "dead").write_text("1000")
    done = run_all(suite)
    out = suite / "results" / "run"
    rows = table(out / "status.tsv")
    assert [row[0] for row in rows] == [SAFE, FAULT_FIRST, FAULT, HANG, HANG_LATER]
    assert not any(row[1].startswith("not run") for row in rows)
    health = table(out / "health.tsv")
    assert len(health) == 6 and all(row[1] == "host-error" and "breaker is not armed" in row[5] for row in health)
    assert not (out / "GPU-UNHEALTHY.txt").exists() and "UNHEALTHY" not in done.stdout
    assert done.returncode == 1                              # the safe run had a host error: that still fails


def test_an_unreadable_kernel_log_is_said_once_and_fails_nothing(suite):
    tools = suite.parent.parent / "tools"
    script(tools / "dmesg", 'echo "dmesg: read kernel buffer failed: Operation not permitted" >&2\nexit 1\n')
    script(tools / "journalctl", "exit 1\n")
    done = run_all(suite)
    assert done.returncode == 0, done.stdout + done.stderr
    log = (suite / "results" / "run" / "xid.log").read_text().splitlines()
    assert len(log) == 2 and log[1].startswith("# dmesg is unreadable here (dmesg: read kernel buffer failed")


def test_a_hang_that_is_predicted_to_finish_is_named_by_the_runner(suite):
    """tmem/alloc_leak may hang by design, and the simulator predicts that it returns. A
    timeout is not a failed collection, but it is said, and the canary runs after it."""
    script(suite / "bin" / "tmem" / "alloc", "echo 'tmem_alloc_script: first case'\nsleep 60\n")
    done = run_all(suite, "--timeout-scale", "0.05", only=(SAFE, HANG_LATER))
    assert done.returncode == 0, done.stdout + done.stderr
    out = suite / "results" / "run"
    assert [row[:2] for row in table(out / "status.tsv")] == [[SAFE, "ok"], [HANG_LATER, "timeout"]]
    assert f"TIMED OUT, PREDICTED TO FINISH: {HANG_LATER}" in done.stdout
    assert "1 run(s) TIMED OUT, PREDICTED TO FINISH" in done.stdout
    assert [row[0] for row in table(out / "health.tsv")] == ["(start)", HANG_LATER]
    assert (out / "tmem" / "alloc_leak.stdout").read_text() == "tmem_alloc_script: first case\n"
    compared = compare(suite)
    line = next(l for l in compared.stdout.splitlines() if l.startswith(HANG_LATER + " "))
    assert "TIMED OUT, predicted to finish" in line and compared.returncode == 1


# ---------------------------------------------------------------------------------------------
# The flag for RT-11 exists before the run can be killed
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rt11, capability, no_e2e, flagged", [
    ("auto", "10.0", False, True),          # a B200
    ("auto", "8.6", False, False),          # the development machine
    ("required", "8.6", False, True),
    ("optional", "10.0", False, False),
    ("required", "10.0", True, False),      # --no-e2e asks for a run without the row
])
def test_the_rt11_flag_is_there_before_the_first_probe(tmp_path, monkeypatch, rt11, capability, no_e2e, flagged):
    run_all = load("run_all_rt11_start", SUITE / "run_all.py")
    out = tmp_path / "results"
    seen = {}

    def run_process(command, timeout, **_):
        words = [str(word) for word in command]
        if words[:2] == ["nvidia-smi", "--query-gpu=compute_cap"]:
            return "ok", 0.0, capability.encode() + b"\n", b""
        if any(word.endswith(("run_e2e.py", "run_rows.py")) for word in words):
            # The stage's limit: the run is killed before the row. What is on disk then?
            seen["flag"] = (out / "RT-11-NOT-CAPTURED.txt").read_text() if (out / "RT-11-NOT-CAPTURED.txt").exists() else ""
            raise SystemExit(143)
        return "ok", 0.0, b"", b""

    monkeypatch.setattr(run_all, "run_process", run_process)
    monkeypatch.setattr(run_all, "PROBES", [])
    monkeypatch.setattr(run_all, "archive_probes", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_all, "record_toolkit", lambda out, binaries: {"predictions_comparable": "yes", "reason": ""})
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--mode", "gpu", "--no-build", "--binaries", "ampere", "--rt11", rt11,
                                      "--out", str(out)] + (["--no-e2e"] if no_e2e else []))
    with pytest.raises(SystemExit):
        run_all.main()
    assert seen["flag"].startswith("MATRIX ROW RT-11 IS NOT CAPTURED YET") == flagged
    assert (seen["flag"] == "") == (not flagged)


def test_run_rows_removes_the_early_flag_when_the_row_is_captured(tmp_path):
    rows = load("run_rows_flag_under_test", SUITE / "run_rows.py")
    marker = tmp_path / "RT-11-NOT-CAPTURED.txt"
    marker.write_text("MATRIX ROW RT-11 IS NOT CAPTURED YET: the run has not reached the row\n")
    assert rows.rt11_verdict(SimpleNamespace(rt11="required"), tmp_path, {rows.RT11: "ok"}) == 0
    assert not marker.exists()


# ---------------------------------------------------------------------------------------------
# probe_process.py: a killed process
# ---------------------------------------------------------------------------------------------

def test_a_process_that_does_not_exit_after_the_kill_is_not_waited_for(tmp_path, monkeypatch):
    """Stand-in for a process stuck in the driver's teardown: the kill is made to do nothing."""
    probes = load("probe_process_under_test", SUITE / "probe_process.py")
    monkeypatch.setattr(probes, "_terminate", lambda children: None)
    monkeypatch.setattr(probes, "REAP_SECONDS", 0.3)
    pid_file = tmp_path / "pid"
    status, seconds, stdout, _ = probes.run_process(
        ["sh", "-c", f"echo $$ > {pid_file}; echo first case; exec sleep 600"], 0.5)
    try:
        assert status == "timeout (process did not exit after SIGKILL)" == probes.NOT_REAPED
        assert stdout == b"first case\n" and seconds < 5
    finally:
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


def test_a_killed_process_keeps_what_it_printed():
    probes = load("probe_process_output", SUITE / "probe_process.py")
    status, seconds, stdout, stderr = probes.run_process(["sh", "-c", "echo one; echo two >&2; exec sleep 600"], 0.5)
    assert (status, stdout, stderr) == ("timeout", b"one\n", b"two\n") and seconds < 10


def test_probes_print_line_by_line(tmp_path):
    """A probe's standard output is a pipe, where the C library buffers by the block: without
    the line in common.h a program killed in its third case has printed nothing. The stand-in
    is that very line in a program of its own (the probes need nvcc), next to one without it."""
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:
        pytest.skip("needs a C++ compiler")
    header = (SUITE / "common.h").read_text()
    line = next(l for l in header.splitlines() if "setvbuf(stdout" in l)
    assert line.startswith("static const int ") and "_IOLBF" in line
    assert header.index(line) < header.index("#define HOST_CHECK")          # before anything prints
    for source in SUITE.glob("*/*.cu"):                                     # every probe gets it
        text = source.read_text()
        assert any(f'#include "{name}"' in text for name in ("../common.h", "../tcgen05_common.h", "tma_dump.h",
                                                             "clc_common.h")), source.name
    for name in ("tcgen05_common.h", "tma/tma_dump.h", "cluster/clc_common.h"):
        assert "common.h" in (SUITE / name).read_text()
    body = ('#include <stdio.h>\n#include <unistd.h>\n%s\n'
            'int main() { printf("case 1\\n"); printf("case 2\\n"); sleep(600); printf("case 3\\n"); return 0; }\n')
    probes = load("probe_process_buffering", SUITE / "probe_process.py")
    kept = {}
    for name, first in (("with", line), ("without", "")):
        (tmp_path / f"{name}.cpp").write_text(body % first)
        subprocess.run([compiler, "-O1", str(tmp_path / f"{name}.cpp"), "-o", str(tmp_path / name)], check=True)
        status, _, stdout, _ = probes.run_process([str(tmp_path / name)], 1.0)
        assert status == "timeout"
        kept[name] = stdout
    assert kept == {"with": b"case 1\ncase 2\n", "without": b""}


def test_base_offset_runs_each_field_case_in_its_own_process():
    """`base_offset field` stops at its first failed kernel (in the simulator: the first
    case), so the manifest runs the eight cases one by one. None has a prediction."""
    cases = [p for p in manifest.PROBES if p.name.startswith("mma/base_offset.field")]
    assert [p.args for p in cases] == [("field", str(index)) for index in range(8)]
    assert all(p.predict == "none" and p.risk == "fault" and p.program == "mma/base_offset" for p in cases)
    source = (SUITE / "mma" / "base_offset.cu").read_text()
    listed = source[source.index("WITH_FIELD[] = "):]
    assert listed[:listed.index(";")].count("{") == 1 + len(cases)
    assert not list((SUITE / "predicted" / "mma").glob("base_offset.field*"))


# ---------------------------------------------------------------------------------------------
# compare.py: verdicts that depend on what the simulator predicts
# ---------------------------------------------------------------------------------------------

def compared(tmp_path, monkeypatch, capsys, runs: dict):
    """compare.py on results made of `runs`: {name: (status, stdout or None for the prediction,
    stderr)}. Returns (exit status, {name: (verdict, detail)}, everything printed)."""
    import re
    module = load("compare_verdicts_under_test", SUITE / "compare.py")
    monkeypatch.setattr(module, "PROBES", [manifest.BY_NAME[name] for name in runs])
    monkeypatch.setattr(module, "E2E", [])
    monkeypatch.setattr(module, "validate_probe_archive", lambda *args: [])
    out = tmp_path / "results"
    lines = ["name\tstatus\tseconds\tcommand\tstarted_utc"]
    for name, (status, stdout, stderr) in runs.items():
        target = out / (name + ".stdout")
        target.parent.mkdir(parents=True, exist_ok=True)
        prediction = SUITE / "predicted" / (name + ".txt")
        target.write_text(prediction.read_text() if stdout is None else stdout)
        target.with_suffix(".stderr").write_text(stderr)
        lines.append(f"{name}\t{status}\t1.0\tx\t2026-11-24T00:00:00Z")
    (out / "status.tsv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(sys, "argv", ["compare.py", str(out)])
    code = module.main()
    printed = capsys.readouterr().out
    verdicts = {}
    for line in printed.split("\n\n")[0].splitlines()[1:]:
        fields = re.split(r"\s{2,}", line, maxsplit=2)
        verdicts[fields[0]] = (fields[1], fields[2] if len(fields) > 2 else "")
    return code, verdicts, printed


def test_a_timeout_is_as_predicted_only_where_the_simulator_predicts_a_deadlock(tmp_path, monkeypatch, capsys):
    hang = [p for p in manifest.PROBES if p.risk == "hang"]
    code, verdicts, printed = compared(tmp_path, monkeypatch, capsys, {p.name: ("timeout", "", "") for p in hang})
    by_verdict = {}
    for name, (verdict, _) in verdicts.items():
        by_verdict.setdefault(verdict, set()).add(name)
    assert by_verdict["timed out (as predicted)"] == {"mbarrier/expect_tx_small.wait", "mbarrier/expect_tx_large",
                                                      "tmem/alloc_blocks"}
    # No prediction (try_wait.incomplete), or the simulator predicts a refusal (dealloc_mismatch).
    assert by_verdict["timed out (by design)"] == {"mbarrier/try_wait.incomplete", "tmem/dealloc_mismatch"}
    assert verdicts["mbarrier/try_wait.incomplete"][1] == "no prediction"
    assert verdicts["tmem/dealloc_mismatch"][1].startswith("simulator: tcgen05.dealloc of 32 columns")
    finish = by_verdict["TIMED OUT, predicted to finish"]
    assert finish == {f"cluster/pair_mma.m{m}_issuer{i}" for m in (256, 128) for i in (0, 1)} | {
        "tmem/dealloc_repeated", "tmem/alloc_leak", "cluster/clc_failed_response.again",
        "cluster/clc_failed_response.again_grid"}
    assert len(finish) == 8 and len(hang) == 13 and code == 1
    details = printed[printed.index("\nDetails\n"):]
    for name in finish:
        assert f"\n{name}: " in details and "the simulator predicts that it finishes" in details
    for name in by_verdict["timed out (as predicted)"] | by_verdict["timed out (by design)"]:
        assert f"\n{name}: " not in details


def test_timeouts_that_are_predicted_or_allowed_do_not_fail_the_comparison(tmp_path, monkeypatch, capsys):
    runs = {name: ("timeout", "", "") for name in ("mbarrier/expect_tx_large", "mbarrier/try_wait.incomplete")}
    runs["tmem/alloc_blocks"] = ("timeout (process did not exit after SIGKILL)", "", "")
    code, verdicts, _ = compared(tmp_path, monkeypatch, capsys, runs)
    assert code == 0
    assert verdicts["tmem/alloc_blocks"][0] == "timed out (as predicted)"
    assert verdicts["tmem/alloc_blocks"][1].endswith("; process did not exit after SIGKILL")


def test_a_timeout_of_a_run_without_that_risk_is_a_failure(tmp_path, monkeypatch, capsys):
    code, verdicts, _ = compared(tmp_path, monkeypatch, capsys, {"tma/ranks": ("timeout", "", ""),
                                                                "tmem/alloc_increasing": ("timeout", "", "")})
    assert code == 1 and verdicts["tma/ranks"][0] == "timed out" and verdicts["tmem/alloc_increasing"][0] == "timed out"


def failed(label: str, error: str, where: str) -> str:
    return f"KERNEL FAILED: {label}: {error} {where} (the driver's words)\n"


def test_a_matching_failed_kernel_says_which_error_and_whether_the_simulator_models_it(tmp_path, monkeypatch, capsys):
    code, verdicts, _ = compared(tmp_path, monkeypatch, capsys, {
        # The simulator names the rule (a larger allocation after a smaller one): a prediction.
        "tmem/alloc_increasing": ("exit 3", None, failed("32 columns, then 64 (a larger allocation after a smaller one)",
                                                         "cudaErrorIllegalInstruction", "at synchronize")),
        # The simulator says "unsupported instruction": any fault of the GPU prints the same line.
        "tmem/lane_window.16": ("exit 3", None, failed("mma_dump", "cudaErrorIllegalAddress", "at synchronize")),
        "mma/operand_layouts.tf32_mn": ("exit 3", None, failed("mma_dump", "cudaErrorInvalidValue", "at launch")),
        # The simulator predicts a deadlock and the GPU reported an error instead.
        "tmem/alloc_blocks": ("exit 3", None, failed("x", "cudaErrorLaunchFailure", "after synchronize")),
        # Standard error lost: the name is still on the info: line.
        "mbarrier/init_count.zero": ("exit 3", None, ""),
    })
    assert code == 0                                                     # all of these may fault or hang by design
    verdict, detail = verdicts["tmem/alloc_increasing"]
    assert verdict == "identical" and detail.startswith("both refuse the kernel; GPU: cudaErrorIllegalInstruction at "
                                                        "synchronize; simulator: tcgen05.alloc of 64 columns")
    for name, error in (("tmem/lane_window.16", "cudaErrorIllegalAddress at synchronize"),
                        ("mma/operand_layouts.tf32_mn", "cudaErrorInvalidValue at launch")):
        verdict, detail = verdicts[name]
        assert verdict == "GPU faulted, not modelled"
        assert detail.startswith(f"GPU: {error}; the simulator does not model this (unsupported instruction")
    # The lines match (the simulator prints its deadlock as a failed kernel), so this stays
    # `identical` (tests/test_rental_harness.py holds that); the detail says what differs.
    verdict, detail = verdicts["tmem/alloc_blocks"]
    assert verdict == "identical"
    assert detail.startswith("GPU: cudaErrorLaunchFailure after synchronize, where the simulator predicts a deadlock")
    verdict, detail = verdicts["mbarrier/init_count.zero"]
    assert verdict == "identical" and "GPU: cudaErrorLaunchFailure (place not recorded)" in detail


def test_the_simulators_own_run_is_identical_to_its_predictions(tmp_path, monkeypatch, capsys):
    """run_all.py --mode sim held against predicted/ (tests/test_predict_path.py does it whole):
    no GPU faulted there, and the refusals are the predictions themselves."""
    runs = {name: ("exit 3", None, (SUITE / "predicted" / (name + ".sim.txt")).read_text().split("\n", 1)[1])
            for name in ("tmem/lane_window.16", "tmem/alloc_increasing", "tmem/alloc_blocks")}
    code, verdicts, _ = compared(tmp_path, monkeypatch, capsys, runs)
    assert code == 0 and {verdict for verdict, _ in verdicts.values()} == {"identical"}
    assert verdicts["tmem/lane_window.16"][1].startswith("a simulator's run, refusing the kernel as predicted (unsupported")


def test_a_failed_kernel_without_a_prediction_is_not_plain_informational(tmp_path, monkeypatch, capsys):
    text = "KERNEL FAILED: mma_dump (base offset)\ninfo: error cudaErrorIllegalInstruction\n"
    stderr = failed("mma_dump (base offset)", "cudaErrorIllegalInstruction", "at synchronize")
    code, verdicts, _ = compared(tmp_path, monkeypatch, capsys, {
        "mma/base_offset.field.0": ("exit 3", text, stderr),             # may fault by design
        "mma/base_offset.field.1": ("ok", "layout 128B, ...\n", ""),
    })
    assert code == 0
    assert verdicts["mma/base_offset.field.0"] == (
        "kernel failed (by design)", "no prediction; GPU: cudaErrorIllegalInstruction at synchronize; 2 line(s) recorded")
    assert verdicts["mma/base_offset.field.1"] == ("informational", "no prediction; 1 line(s) recorded")
    # Where the manifest allows no fault, it is a failure.
    code, verdicts, _ = compared(tmp_path / "second", monkeypatch, capsys,
                                 {"tma/tensormap_encode_sweep": ("exit 3", text, stderr)})
    assert code == 1 and verdicts["tma/tensormap_encode_sweep"][0] == "kernel failed"


def test_a_failed_kernel_that_the_simulator_runs_is_a_difference(tmp_path, monkeypatch, capsys):
    text = "KERNEL FAILED: tma_store\ninfo: error cudaErrorIllegalAddress\n"
    code, verdicts, printed = compared(tmp_path, monkeypatch, capsys, {
        "tma/store_wait_group.edge": ("exit 3", text, failed("tma_store", "cudaErrorIllegalAddress", "at synchronize"))})
    assert verdicts["tma/store_wait_group.edge"] == (
        "kernel failed (by design)", "GPU: cudaErrorIllegalAddress at synchronize; the simulator's kernel returns")
    assert code == 0 and "the GPU reported a failed kernel; outputs differ" in printed


# ---------------------------------------------------------------------------------------------
# compare.py: which status.tsv
# ---------------------------------------------------------------------------------------------

def tree(root: Path, *directories: str) -> Path:
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "status.tsv").write_text("name\tstatus\tseconds\tcommand\n")
    return root


def test_the_gpu_run_of_a_rental_archive_is_the_one_compared(tmp_path):
    module = load("compare_selection_under_test", SUITE / "compare.py")
    rental = tree(tmp_path / "rental", "run/out/predicted-toolkit/sim-run", "run/out/predicted-toolkit/blackwell",
                  "run/out/blackwell", "run/out/blackwell/rows")
    assert module.find_results(rental, rental) == rental / "run" / "out" / "blackwell"
    # Given directly, a directory is itself.
    assert module.find_results(rental / "run" / "out" / "blackwell", rental) == rental / "run" / "out" / "blackwell"
    # run_all.py's own archive: one directory, with its rows/.
    own = tree(tmp_path / "own", "gpu-host-1", "gpu-host-1/rows")
    assert module.find_results(own, own) == own / "gpu-host-1"
    # Never a simulator's run, never the rows': without the GPU's there is nothing to compare.
    for name, directories in (("sim", ["out/predicted-toolkit/sim-run"]), ("rows", ["out/blackwell/rows"]), ("none", [])):
        with pytest.raises(SystemExit) as error:
            module.find_results(tree(tmp_path / name, *directories), Path("given"))
        assert "given: no status.tsv of a GPU run inside" in str(error.value)
    # Two that could be it: say which, pick none.
    two = tree(tmp_path / "two", "first", "second", "out/predicted-toolkit/sim-run")
    with pytest.raises(SystemExit) as error:
        module.find_results(two, Path("given"))
    assert "more than one status.tsv" in str(error.value)
    assert [line.strip() for line in str(error.value).splitlines()[1:]] == ["first", "second"]
    # ... unless exactly one of them is out/blackwell.
    assert module.find_results(tree(two, "x/out/blackwell"), two) == two / "x" / "out" / "blackwell"


def test_compare_reads_the_gpu_run_from_a_packed_rental(tmp_path):
    """The archive of a rental with a fallback toolkit holds a simulator's run of every probe,
    all `ok`. Picked by accident it would report the GPU run as whatever the simulator did."""
    rental = tmp_path / "run"
    sim, gpu = rental / "out" / "predicted-toolkit" / "sim-run", rental / "out" / "blackwell"
    name = "mbarrier/phase_parity"
    for directory, status, text in ((sim, "ok", (SUITE / "predicted" / (name + ".txt")).read_text()),
                                    (gpu, "host-error", "HOST ERROR: cudaMalloc(&p, 4) returned cudaErrorUnknown\n")):
        (directory / "mbarrier").mkdir(parents=True)
        (directory / "status.tsv").write_text(f"name\tstatus\tseconds\tcommand\n{name}\t{status}\t0.1\tx\n")
        (directory / (name + ".stdout")).write_text(text)
    archive = tmp_path / "rental.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(rental, arcname="a-run")                    # the simulator's run sorts first in the archive
    for given in (archive, rental):
        done = subprocess.run([sys.executable, str(SUITE / "compare.py"), str(given)], capture_output=True, text=True)
        line = next(l for l in done.stdout.splitlines() if l.startswith(name + " "))
        assert "HOST ERROR" in line and "identical" not in line, done.stdout[:2000] + done.stderr
