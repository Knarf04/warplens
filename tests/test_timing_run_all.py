"""validation/blackwell_timing/run_all.sh, without a GPU and without Nsight Compute.

The benchmarks here are shell scripts that print what a benchmark prints, and ncu is the
stand-in of validation/blackwell_timing/fixture_ncu. What that covers is the runner's own
logic, as the pre-rental review of 2026-11-24 asked for it: the order of the run, the limit
of a benchmark and how its end is recorded, that nothing is left running when the runner is
stopped, and the counter verdict it hands to env/remote/run.sh. What it cannot cover: a real
benchmark that hangs inside the driver, and a real ncu.
"""

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TIMING = ROOT / "validation" / "blackwell_timing"
FIXTURE = TIMING / "fixture_ncu"
AMPERE_VALID = ["launch", "alu_chain", "smem_wavefronts", "global_load", "gstream", "cp_async", "ncu_banks"]
BLACKWELL_ONLY = ["mbar", "tma", "mma_rate", "tmem_access", "clc_latency", "cluster_costs", "pipelined_gemm"]

# A benchmark: logs that it ran, prints a DEVICE line and two BENCH lines, and then does
# what the test asks of the benchmark of that name.
PROGRAM = r"""#!/bin/sh
name=$(basename "$0")
echo "$name $*" >> "$STAND_IN_ORDER"
echo "DEVICE bench=$name sim=0"
echo "BENCH id=$name/a reps=1 trial=0"
echo "BENCH id=$name/a reps=2 trial=0"
case " $STAND_IN_HANG " in *" $name "*)
  [ -z "$STAND_IN_IGNORE_TERM" ] || trap '' TERM
  sleep 600 &
  echo "$$ $!" >> "$STAND_IN_PIDS"
  wait
  wait ;;
esac
case " $STAND_IN_LAUNCH_FAILS " in *" $name "*)
  echo 'BENCH_ERROR where="cudaGetLastError()" error=cudaErrorNoKernelImageForDevice class=launch-failed'
  exit 4 ;;
esac
case " $STAND_IN_FAILS " in *" $name "*) exit 1 ;; esac
exit 0
"""


def gone(pid: int) -> bool:
    """No such process, or one that has ended and only waits for its parent to collect it."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


def wait_until(condition, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


@pytest.fixture
def suite(tmp_path):
    timing = tmp_path / "validation" / "blackwell_timing"
    ampere = tmp_path / "validation" / "ampere" / "bin"
    (timing / "bin" / "hw").mkdir(parents=True)
    (timing / "predicted").mkdir()
    ampere.mkdir(parents=True)
    for name in ("run_all.sh", "run_ncu.sh"):
        shutil.copy(TIMING / name, timing / name)
    shutil.copy(TIMING / "predicted" / "meta.json", timing / "predicted" / "meta.json")
    for program in [timing / "bin" / "hw" / name for name in AMPERE_VALID + BLACKWELL_ONLY] + [ampere / "transpose_hw"]:
        program.write_text(PROGRAM)
        program.chmod(0o755)
    return timing


def environment(suite: Path, **env: str) -> dict[str, str]:
    full = {**os.environ, "NCU_BIN": str(FIXTURE / "fake_ncu.sh"), "FAKE_NCU_MODE": "ok", "PYTHON": "false",
            "STAND_IN_ORDER": str(suite / "order.log"), "STAND_IN_PIDS": str(suite / "pids"),
            "STAND_IN_HANG": "", "STAND_IN_IGNORE_TERM": "", "STAND_IN_LAUNCH_FAILS": "", "STAND_IN_FAILS": "", **env}
    full.pop("WARPLENS_NCU", None)
    full.update({key: value for key, value in env.items() if key == "WARPLENS_NCU"})
    return full


def run_all(suite: Path, *args: str, **env: str):
    done = subprocess.run(["bash", str(suite / "run_all.sh"), "--no-fit", "--out", str(suite / "out"), *args],
                          capture_output=True, text=True, env=environment(suite, **env), timeout=120)
    return done, (suite / "out" / "status.txt").read_text().splitlines()


def ncu_lines(status: list[str]) -> list[str]:
    return [line for line in status if re.match(r"ncu (ok|partial|failed|not run): ", line)]


def test_what_is_proven_runs_first(suite):
    """The Ampere-valid benchmarks, their counters (TIM-16/TIM-17 first), and only then what
    has never run on a GPU: a run that is cut short has lost the unproven part."""
    done, status = run_all(suite)
    assert done.returncode == 0, done.stderr
    ran = [(line.split()[0], "--trials 5 " in line) for line in (suite / "order.log").read_text().splitlines()]
    timed = lambda names: [(name, True) for name in names]
    profiled = lambda names: [(name, False) for name in names]
    assert ran == (timed(AMPERE_VALID)
                   + profiled(["launch", "ncu_banks", "transpose_hw", "alu_chain", "smem_wavefronts", "global_load", "cp_async"])
                   + timed(BLACKWELL_ONLY) + timed(["mma_rate"])
                   + profiled(["mbar", "tma", "mma_rate", "tmem_access", "clc_latency"]))
    assert (suite / "order.log").read_text().splitlines()[-6].endswith("--only kind=i8")
    # status.txt in the same order, and the counter verdict once, as STATUS.txt has it.
    names = [line.split(":")[0] for line in status if re.match(r"\w+: exit=", line)]
    assert names == AMPERE_VALID + BLACKWELL_ONLY + ["mma_rate_i8"]
    first = (suite / "out" / "ncu" / "STATUS.txt").read_text().splitlines()[0]
    assert first == f"ncu ok: 11 benchmark(s) profiled, 0 failed; raw CSVs in {suite / 'out' / 'ncu'}"
    assert ncu_lines(status) == [first]
    part = next(line for line in status if line.startswith("ncu, first part (the programs valid on Ampere): "))
    assert "ncu partial: 6 benchmark(s) profiled, 0 failed, 5 not profiled" in part
    assert status.index(part) == status.index(next(l for l in status if l.startswith("ncu_banks: "))) + 1
    assert (suite / "out" / "ncu" / "mbar.csv").exists() and (suite / "out" / "ncu_banks.out").read_text().count("BENCH ") == 2
    assert (suite / "out.tar.gz").exists()


def test_ampere_run_leaves_the_blackwell_programs_out(suite):
    hw = suite / "bin" / "hw"
    hw.rename(suite / "bin" / "ampere")
    for name in BLACKWELL_ONLY:
        (suite / "bin" / "ampere" / name).unlink()
    done, status = run_all(suite, "--ampere")
    assert done.returncode == 0, done.stderr
    assert [line.split(":")[0] for line in status if re.match(r"\w+: exit=", line)] == AMPERE_VALID
    assert not any("missing program" in line for line in status)
    assert ncu_lines(status)[0].startswith("ncu ok: 6 benchmark(s) profiled, 0 failed; ")
    assert "ncu: not built in bin/ampere, so not profiled: mbar tma mma_rate tmem_access clc_latency" in \
        (suite / "out" / "ncu" / "STATUS.txt").read_text().splitlines()


def test_a_benchmark_that_hangs_is_cut_recorded_and_passed(suite):
    """Timed out, launch failed and failed are three different notes; the lines the hanging
    benchmark had printed are kept; the benchmarks after it run."""
    start = time.monotonic()
    done, status = run_all(suite, "--no-ncu", "--timeout", "1", STAND_IN_HANG="mbar", STAND_IN_LAUNCH_FAILS="tma",
                           STAND_IN_FAILS="tmem_access")
    assert time.monotonic() - start < 30
    assert done.returncode == 3
    line = lambda name: next(l for l in status if l.startswith(name + ": "))
    assert re.fullmatch(r"mbar: exit=124 seconds=[12] lines=2 TIMED OUT after 1s", line("mbar"))
    assert line("tma").endswith("lines=2 LAUNCH FAILED (the driver could not load the kernels: cudaErrorNoKernelImageForDevice)")
    assert re.fullmatch(r"tmem_access: exit=1 seconds=\d+ lines=2 FAILED", line("tmem_access"))
    assert re.fullmatch(r"pipelined_gemm: exit=0 seconds=\d+ lines=2 ok", line("pipelined_gemm"))
    assert (suite / "out" / "mbar.out").read_text().count("BENCH id=mbar/a") == 2
    assert all(wait_until(lambda pid=pid: gone(int(pid))) for pid in (suite / "pids").read_text().split())
    assert ncu_lines(status) == ["ncu not run: not requested (run_all.sh --no-ncu)"]
    assert (suite / "out" / "ncu" / "STATUS.txt").read_text() == "ncu not run: not requested (run_all.sh --no-ncu)\n"


def test_the_default_limit(suite):
    """180 s a benchmark: the slowest measured took 13 s (3080 Ti), the longest estimate for
    the B200 is 30 s. A run asked to take four times as long gets four times the limit."""
    limit = lambda: re.search(r"timeout: (\d+)", (suite / "out" / "env.txt").read_text()).group(1)
    run_all(suite, "--no-ncu")
    assert limit() == "180"
    run_all(suite, "--no-ncu", "--scale", "4")
    assert limit() == "720"
    run_all(suite, "--no-ncu", "--scale", "0.5")
    assert limit() == "180"
    run_all(suite, "--no-ncu", "--scale", "4", "--timeout", "50")
    assert limit() == "50"
    bad = subprocess.run(["bash", str(suite / "run_all.sh"), "--timeout", "never"], capture_output=True, text=True)
    assert bad.returncode == 2


def stop(suite: Path, signum: int, wanted_pids: int, *args: str, **env: str):
    """Start run_all.sh in a session of its own, wait for the stand-ins to hang, signal the
    script and nothing else. -> (exit status, the process ids that were hanging, seconds)"""
    pids = suite / "pids"
    script = subprocess.Popen(["bash", str(suite / "run_all.sh"), "--no-fit", "--out", str(suite / "out"), *args],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              env=environment(suite, **env), start_new_session=True)
    hanging: list[int] = []
    try:
        assert wait_until(lambda: pids.exists() and len(pids.read_text().split()) == wanted_pids, 30)
        hanging = [int(pid) for pid in pids.read_text().split()]
        assert not any(gone(pid) for pid in hanging)
        start = time.monotonic()
        script.send_signal(signum)
        code = script.wait(timeout=25)
        took = time.monotonic() - start
        survivors = [pid for pid in hanging if not wait_until(lambda pid=pid: gone(pid), 3)]
        return code, hanging, survivors, took
    finally:
        if script.poll() is None:
            script.kill()
        for pid in hanging:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("signum, status, name", [(signal.SIGTERM, 143, "SIGTERM"), (signal.SIGINT, 130, "SIGINT")])
def test_stopping_the_runner_leaves_no_benchmark_behind(suite, signum, status, name):
    """`timeout` puts the benchmark in a process group of its own, so the stage limit of
    env/remote/run.sh (SIGTERM to the runner's group) or Ctrl-C does not reach it. The runner
    must: a benchmark left on the GPU disturbs the rerun and writes the same output file."""
    code, hanging, survivors, took = stop(suite, signum, 2, "--no-ncu", STAND_IN_HANG="smem_wavefronts")
    assert code == status and len(hanging) == 2 and survivors == []
    assert took < 5                                           # at once, not at the benchmark's limit
    lines = (suite / "out" / "status.txt").read_text().splitlines()
    assert any(re.fullmatch(rf"smem_wavefronts: STOPPED by {name} after \d+ s \(run_all.sh was told to stop\)", l) for l in lines)
    assert lines[-1] == (f"run_all: stopped by {name}; not run: global_load gstream cp_async ncu_banks "
                         + " ".join(BLACKWELL_ONLY) + " mma_rate_i8; no fit, no archive")
    assert [l.split(":")[0] for l in lines if ": exit=" in l] == ["launch", "alu_chain"]
    assert not (suite / "out.tar.gz").exists()


def test_a_benchmark_that_ignores_sigterm_is_killed(suite):
    code, hanging, survivors, took = stop(suite, signal.SIGTERM, 2, "--no-ncu", STAND_IN_HANG="launch",
                                          STAND_IN_IGNORE_TERM="1")
    assert code == 143 and survivors == []
    assert 7 < took < 15                                      # the 8 s it is given to end by itself


def test_stopping_the_runner_during_ncu_leaves_nothing_behind(suite):
    """Three levels: run_all.sh -> run_ncu.sh -> ncu -> the profiled program."""
    code, hanging, survivors, took = stop(suite, signal.SIGTERM, 2, FAKE_NCU_HANG="alu_chain",
                                          FAKE_NCU_PIDS=str(suite / "pids"))
    assert code == 143 and len(hanging) == 2 and survivors == []
    assert took < 8
    lines = (suite / "out" / "status.txt").read_text().splitlines()
    assert any(l.startswith("run_ncu.sh --group ampere: STOPPED by SIGTERM") for l in lines)
    assert lines[-1].startswith("run_all: stopped by SIGTERM; not run: mbar tma ")
    # What run_ncu.sh had read is on record, and so is that it was stopped.
    ncu = (suite / "out" / "ncu" / "STATUS.txt").read_text().splitlines()
    assert ncu[0].startswith("ncu partial: 2 benchmark(s) profiled, 0 failed, 9 not profiled; ")
    assert "ncu: run_ncu.sh was stopped by SIGTERM" in ncu


def test_the_exit_status_says_when_required_counters_were_not_read(suite):
    """env/remote/run.sh passes WARPLENS_NCU=ok when preflight read a counter. Then, and only
    then, `ncu failed:` and `ncu not run:` make the run fail, after every benchmark has run."""
    everything = " ".join(["ncu_banks", "transpose_hw", "alu_chain", "smem_wavefronts", "global_load", "cp_async",
                           "mbar", "tma", "mma_rate", "tmem_access", "clc_latency"])
    ran = lambda status: len([line for line in status if ": exit=0 " in line])
    done, status = run_all(suite, WARPLENS_NCU="ok", FAKE_NCU_MODE="permission")
    assert done.returncode == 1 and ran(status) == 15
    assert ncu_lines(status) == ["ncu not run: no permission to read GPU performance counters (ERR_NVGPUCTRPERM); "
                                 "needs root or NVreg_RestrictProfilingToAdminUsers=0"]
    assert any(line.startswith("run_all: COUNTERS REQUIRED AND NOT READ") for line in status)

    done, status = run_all(suite, WARPLENS_NCU="ok", FAKE_NCU_FAIL=everything)
    assert done.returncode == 1 and ran(status) == 15
    assert ncu_lines(status)[0].startswith("ncu failed: 0 benchmark(s) profiled, 11 failed; ")

    done, status = run_all(suite, WARPLENS_NCU="ok", FAKE_NCU_FAIL="tma")          # some read: not a failure
    assert done.returncode == 0 and ncu_lines(status)[0].startswith("ncu partial: 10 benchmark(s) profiled, 1 failed; ")
    done, status = run_all(suite, WARPLENS_NCU="ok")
    assert done.returncode == 0 and ncu_lines(status)[0].startswith("ncu ok: ")
    # Not required: preflight read none, the variable is not set, or the run said --no-ncu.
    for env in ({"WARPLENS_NCU": "unavailable"}, {}):
        done, status = run_all(suite, FAKE_NCU_MODE="permission", **env)
        assert done.returncode == 0 and ncu_lines(status)[0].startswith("ncu not run: no permission")
        assert not any("COUNTERS REQUIRED" in line for line in status)
    done, status = run_all(suite, "--no-ncu", WARPLENS_NCU="ok")
    assert done.returncode == 0 and ncu_lines(status) == ["ncu not run: not requested (run_all.sh --no-ncu)"]
    # A failed benchmark still counts, next to the counters.
    done, status = run_all(suite, WARPLENS_NCU="ok", FAKE_NCU_MODE="permission", STAND_IN_FAILS="gstream")
    assert done.returncode == 2


def test_the_limits_of_ncu_are_passed_on(suite):
    done, status = run_all(suite, "--ncu-timeout", "1", "--ncu-budget", "60", FAKE_NCU_HANG="mbar")
    assert done.returncode == 0
    assert "mbar (timed out after 1 s; not run again)" in ncu_lines(status)[0]
    assert "of the budget of 60 s used" in (suite / "out" / "ncu" / "STATUS.txt").read_text()
