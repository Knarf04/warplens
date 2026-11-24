"""Failures of unattended validation must not leave GPU jobs or mislabel measurements."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "validation" / "blackwell"
TIMING = ROOT / "validation" / "blackwell_timing"


def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executable(path, code):
    path.write_text(code)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_interrupted_runner_stops_detached_nested_probe(tmp_path, signum, ignore_term):
    # Model run_all -> e2e runner -> GPU program, each in a separate session.
    pidfile = tmp_path / "pid"
    child = executable(tmp_path / "probe.py", f"""import os, time, signal
from pathlib import Path
if {ignore_term!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(pidfile)!r}).write_text(str(os.getpid()))
time.sleep(60)
""")
    nested = tmp_path / "nested.py"
    nested.write_text(f"""import sys
sys.path.insert(0, {str(PROBES)!r})
from probe_process import forward_interrupts, run_process
with forward_interrupts():
    run_process([sys.executable, {str(child)!r}], 120)
""")
    runner = tmp_path / "outer.py"
    runner.write_text(f"""import sys
sys.path.insert(0, {str(PROBES)!r})
from probe_process import forward_interrupts, run_process
with forward_interrupts():
    run_process([sys.executable, {str(nested)!r}], 120)
""")
    outer = subprocess.Popen([sys.executable, str(runner)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        outer.send_signal(signum)
        assert outer.wait(timeout=10) == 128 + signum
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if outer.poll() is None:
            outer.kill()
            outer.wait()
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_probe_timeout_reaps_the_process(tmp_path):
    helper = load("probe_process_timeout", PROBES / "probe_process.py")
    status, _, out, _ = helper.run_process(
        [sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(60)"], 0.1)
    assert status == "timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(int(out.strip()), 0)


def ncu_tree(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    script = suite / "run_ncu.sh"
    script.write_text((TIMING / "run_ncu.sh").read_text())
    binaries = suite / "bin"
    binaries.mkdir()
    for name in ("launch", "alu_chain", "smem_wavefronts", "global_load", "cp_async", "mbar",
                 "tma", "mma_rate", "tmem_access"):
        executable(binaries / name, "#!/bin/sh\nexit 0\n")
    transpose = tmp_path / "ampere" / "bin"
    transpose.mkdir(parents=True)
    executable(transpose / "transpose_hw", "#!/bin/sh\nexit 0\n")
    return script, binaries, tmp_path / "results"


def test_ncu_permission_canary_has_a_timeout(tmp_path):
    script, binaries, out = ncu_tree(tmp_path)
    bindir = tmp_path / "tools"
    bindir.mkdir()
    log = tmp_path / "timeout.log"
    executable(bindir / "timeout", f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {log}\nexit 124\n")
    ncu = executable(bindir / "ncu", "#!/bin/sh\necho SHOULD_NOT_RUN_DIRECTLY\nexit 99\n")
    done = subprocess.run(["bash", str(script), str(binaries), str(out)], capture_output=True, text=True,
                          env={**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
                               "NCU_BIN": str(ncu)}, timeout=10)
    assert done.returncode == 0
    assert log.exists() and "--kill-after=10 60" in log.read_text()
    assert "exit 124" in (out / "STATUS.txt").read_text()


def test_ncu_partial_csv_with_error_is_not_success(tmp_path):
    script, binaries, out = ncu_tree(tmp_path)
    ncu = executable(tmp_path / "ncu", '''#!/bin/sh
printf '"ID","Kernel Name"\n'
case "$*" in *"/launch "*) exit 0;; *) exit 9;; esac
''')
    done = subprocess.run(["bash", str(script), str(binaries), str(out)], capture_output=True, text=True,
                          env={**os.environ, "NCU_BIN": str(ncu)}, timeout=10)
    assert done.returncode == 0, done.stderr
    assert "0 benchmark(s) profiled, 9 failed" in (out / "STATUS.txt").read_text()


def test_negative_clock_fit_does_not_poison_other_costs():
    fit = load("fit_negative_clock", TIMING / "fit.py")
    cfg = fit.load_config_values(fit.DEFAULT_CONFIG)
    ident = "global_load/load_chain/footprint=l2"
    meas = {"alu_chain/warps=1": {"ns_per_rep": -2}, ident: {"ns_per_rep": 100}}
    pred = {"alu_chain/warps=1": {"ns_per_rep": 2}, ident: {"ns_per_rep": 100}}
    params, diag = fit.derive(meas, pred, cfg, {})
    by_key = {p["key"]: p["value"] for p in params}
    assert "clock_ghz" not in by_key
    assert by_key["global_access_latency_clk"] == cfg["global_access_latency_clk"]
    assert any(d[0] == "clock_ghz" and "no positive finite" in d[3] for d in diag)


def test_prediction_child_receives_the_requested_model_and_config(tmp_path, monkeypatch):
    predict = load("predict_requested_model", TIMING / "predict.py")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "launch").touch()
    seen = []

    def run(command, timeout, env):
        seen.append(command)
        record = Path(command[command.index("--record") + 1])
        record.write_text(json.dumps([{"time_ns": 1, "instructions": 1, "bound": "issue", "findings": 0}]))
        return "ok", 0, b"BENCH id=launch reps=1 trial=0 event_ns=1\n", b""

    monkeypatch.setattr(predict, "run_process", run)
    meta = predict.predict(tmp_path / "out", ["launch"], bin_dir=binaries, model="B300", quiet=True)
    assert seen[0][seen[0].index("--model") + 1] == "B300"
    assert Path(seen[0][seen[0].index("--config") + 1]).name == "b300.yaml"
    assert meta["config"] == predict.fit.load_config_values(predict.fit.DEFAULT_CONFIG.with_name("b300.yaml"))


def test_signal_between_spawn_and_registration_reaps_child(tmp_path):
    pidfile = tmp_path / "pid"
    runner = tmp_path / "spawn.py"
    runner.write_text(f"""import os, signal, sys
from pathlib import Path
sys.path.insert(0, {str(PROBES)!r})
import probe_process as helper
original = helper.subprocess.Popen
def spawn(*args, **kwargs):
    child = original(*args, **kwargs)
    Path({str(pidfile)!r}).write_text(str(child.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return child
helper.subprocess.Popen = spawn
with helper.forward_interrupts():
    helper.run_process([sys.executable, '-c', 'import time; time.sleep(60)'], 120)
""")
    done = subprocess.run([sys.executable, str(runner)], timeout=10)
    assert done.returncode == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_missing_timing_program_does_not_reuse_previous_measurement(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    script = suite / "run_all.sh"
    script.write_text((TIMING / "run_all.sh").read_text())
    binaries = suite / "bin"
    binaries.mkdir()
    executable(binaries / "launch", "#!/bin/sh\necho 'BENCH id=launch reps=1 trial=0 event_ns=1'\n")
    out = tmp_path / "results"
    out.mkdir()
    (out / "alu_chain.out").write_text("BENCH id=alu_chain reps=1 trial=0 event_ns=999\n")
    (out / "alu_chain.err").write_text("stale error")
    done = subprocess.run(["bash", str(script), "--bin", str(binaries), "--out", str(out),
                           "--no-ncu", "--no-fit"], capture_output=True, text=True, timeout=10)
    assert done.returncode != 0
    assert (out / "alu_chain.out").read_text() == ""
    assert (out / "alu_chain.err").read_text() == ""
    assert "alu_chain: missing program" in (out / "status.txt").read_text()


@pytest.mark.parametrize("gpu,prediction,exit_code,verdict", [
    ("3f800000", "00000000", 2, "unverified"),
    ("00000001", "00000000", 2, "unverified"),
    ("7f800000", "7f7fffff", 1, "DIFFERS"),
    ("7fc00000", "00000000", 1, "DIFFERS"),
    ("3f800000", "3f800000", 0, "identical"),
])
def test_numeric_observations_do_not_assert_rounding_or_success(
        tmp_path, monkeypatch, capsys, gpu, prediction, exit_code, verdict):
    compare = load("compare_numeric_validation", PROBES / "compare.py")
    from types import SimpleNamespace
    probe = SimpleNamespace(name="numeric", risk="", predict="sim", compare="ulp", ids=(), claim="observation")
    monkeypatch.setattr(compare, "PROBES", [probe])
    monkeypatch.setattr(compare, "E2E", [])
    out, pred = tmp_path / "results", tmp_path / "predicted"
    out.mkdir()
    pred.mkdir()
    (out / "status.tsv").write_text("name\tstatus\n" + "numeric\tok\n")
    (out / "numeric.stdout").write_text(gpu + "\n")
    (pred / "numeric.txt").write_text(prediction + "\n")
    monkeypatch.setattr(sys, "argv", ["compare.py", str(out), "--predicted", str(pred)])
    assert compare.main() == exit_code
    assert verdict in capsys.readouterr().out


@pytest.mark.parametrize("missing,expected", [(True, 1), (False, 0)])
def test_probe_collector_archives_infrastructure_failure_and_expected_hang(tmp_path, monkeypatch, missing, expected):
    runner = load("run_all_collection_status", PROBES / "run_all.py")
    from types import SimpleNamespace
    suite = tmp_path / "suite"
    (suite / "bin").mkdir(parents=True)
    (suite / "manifest.py").write_text("# fixture manifest\n")
    if not missing:
        (suite / "bin" / "probe").touch()
    monkeypatch.setattr(runner, "HERE", suite)
    probe = SimpleNamespace(name="probe", program="probe", predict="sim", args=(), risk="hang")
    monkeypatch.setattr(runner, "PROBES", [probe])
    monkeypatch.setattr(runner, "run_probe", lambda *args: ("timeout", 0, b"", b""))
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--mode", "sim", "--no-build", "--no-e2e", "--out", str(out)])
    assert runner.main() == expected
    assert out.with_suffix(".tar.gz").exists()
    assert ("not built" if missing else "timeout") in (out / "status.tsv").read_text()


def test_e2e_runner_failure_is_reported_after_archive(tmp_path, monkeypatch):
    runner = load("run_all_e2e_status", PROBES / "run_all.py")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.py").write_text("# fixture manifest\n")
    monkeypatch.setattr(runner, "HERE", suite)
    monkeypatch.setattr(runner, "PROBES", [])
    monkeypatch.setattr(runner, "run_process", lambda *args: ("exit 1", 0, b"runner failed", b""))
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--mode", "sim", "--no-build", "--out", str(out)])
    assert runner.main() == 1
    assert out.with_suffix(".tar.gz").exists()
    assert (out / "e2e-runner.log").read_text() == "runner failed"
