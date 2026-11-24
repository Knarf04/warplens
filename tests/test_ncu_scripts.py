"""The Nsight Compute scripts, without Nsight Compute.

validation/blackwell_timing/run_ncu.sh and validation/ampere/run_ncu.sh need permission to
read GPU performance counters, which the development machine does not give. So their
profiling path has not run against a real ncu. Here they run against a stand-in
(validation/blackwell_timing/fixture_ncu/fake_ncu.sh) that replays recorded ncu output: the
permission error, the metric table of the pinned ncu for the B200's chip, and one complete
counter table from the RTX 3080 Ti. What that covers is the scripts' own logic: argument
handling, the permission check, the retries, where files go, and the parsing of ncu's CSV.
What it cannot cover is listed in the fixture's README.
"""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TIMING = ROOT / "validation" / "blackwell_timing"
AMPERE = ROOT / "validation" / "ampere"
FIXTURE = TIMING / "fixture_ncu"
RECORDED = AMPERE / "transpose_ncu_3080ti_grid64.csv"
BENCHES = ["launch", "alu_chain", "smem_wavefronts", "global_load", "cp_async", "mbar", "tma", "mma_rate",
           "tmem_access", "clc_latency", "ncu_banks"]


def metric_lists(text: str) -> dict[str, list[str]]:
    return {name: value.split(",") for name, value in re.findall(r"^(SMEM|CORE|GLOBAL|BASIC)=(\S+)$", text, flags=re.M)}


@pytest.fixture
def work(tmp_path):
    """A copy of the two suites' ncu scripts with stand-in benchmark programs."""
    timing = tmp_path / "validation" / "blackwell_timing"
    ampere = tmp_path / "validation" / "ampere"
    (timing / "bin" / "hw").mkdir(parents=True)
    (ampere / "bin").mkdir(parents=True)
    shutil.copy(TIMING / "run_ncu.sh", timing / "run_ncu.sh")
    for name in ("run_ncu.sh", "compare_ncu.py"):
        shutil.copy(AMPERE / name, ampere / name)
    programs = [timing / "bin" / "hw" / bench for bench in BENCHES] + [ampere / "bin" / "transpose_hw"]
    for program in programs:
        program.write_text('#!/bin/sh\necho "DEVICE bench=$(basename "$0") sim=0"\n'
                           'echo "BENCH id=stand_in reps=1 trial=0 args=$*"\n')
        program.chmod(program.stat().st_mode | stat.S_IXUSR)
    return tmp_path


def run_ncu(work: Path, *args: str, mode: str = "ok", **env: str):
    log = work / "ncu-commands.log"
    full = {**os.environ, "NCU_BIN": str(FIXTURE / "fake_ncu.sh"), "FAKE_NCU_MODE": mode, "FAKE_NCU_LOG": str(log),
            "FAKE_NCU_TRANSPOSE_CSV": str(RECORDED), "PYTHON": "false", **env}
    done = subprocess.run(["bash", str(work / "validation" / "blackwell_timing" / "run_ncu.sh"), *args],
                          capture_output=True, text=True, env=full)
    return done, (log.read_text().splitlines() if log.exists() else [])


def test_dry_run_prints_every_command_and_runs_nothing(work):
    done, commands = run_ncu(work, "--dry-run", "bin/hw", str(work / "out"))
    assert done.returncode == 0 and commands == []           # ncu was never started
    assert not (work / "out").exists()                        # and nothing was written
    lines = [line for line in done.stdout.splitlines() if line.startswith("dry-run: ")]
    # The permission probe, 11 profiles, and the two comparisons with the simulator.
    assert len(lines) == 14
    lists = metric_lists((TIMING / "run_ncu.sh").read_text())
    assert set(lists) == {"SMEM", "CORE", "GLOBAL", "BASIC"}
    for bench, which in (("alu_chain", "CORE"), ("smem_wavefronts", "SMEM"), ("global_load", "GLOBAL"),
                         ("cp_async", "GLOBAL"), ("mbar", "CORE"), ("tma", "BASIC"), ("mma_rate", "CORE"),
                         ("tmem_access", "CORE")):
        line = next(l for l in lines if f"bin/hw/{bench} " in l)
        assert f"--csv --metrics {','.join(lists[which])} bin/hw/{bench} --trials 1 --scale 0.01" in line
        assert line.rstrip().endswith("built")
    assert "--only ctas=1" in next(l for l in lines if "bin/hw/tma " in l)
    assert "--only m=128/n=256" in next(l for l in lines if "bin/hw/mma_rate " in l)
    assert "--only fail" in next(l for l in lines if "bin/hw/clc_latency " in l)
    banks = next(l for l in lines if "bin/hw/ncu_banks " in l)
    assert f"--metrics {','.join(lists['SMEM'])} bin/hw/ncu_banks --trials 1" in banks
    assert "ncu_banks_compare.py" in lines[lines.index(banks) + 1]
    assert "transpose_hw 64" in lines[-2] and "compare_ncu.py" in lines[-1]
    # A program that is not there is said to be missing, not silently left out.
    (work / "validation" / "blackwell_timing" / "bin" / "hw" / "mbar").unlink()
    done, _ = run_ncu(work, "--dry-run", "bin/hw", str(work / "out"))
    assert next(l for l in done.stdout.splitlines() if "bin/hw/mbar " in l).rstrip().endswith("NOT BUILT")


def test_every_counter_name_is_well_formed_and_listed_for_the_b200():
    """The names against what the pinned ncu lists for chip gb100 (recorded). One name is
    known not to be listed; a new unlisted name is a typo until shown otherwise."""
    table = (FIXTURE / "query_metrics_gb100.txt").read_text()
    listed = {line.split()[0] for line in table.splitlines() if re.match(r"^[a-z0-9]+__", line)}
    names = set()
    for script in (TIMING / "run_ncu.sh", AMPERE / "run_ncu.sh", ROOT / "env" / "remote" / "preflight.sh"):
        names |= set(re.findall(r"\b[a-z0-9]+__[a-z0-9_]+\.[a-z]+\b", script.read_text()))
    assert len(names) >= 19
    for name in names:
        assert re.fullmatch(r"[a-z0-9]+__[a-z0-9_]+\.(sum|max|min|avg|pct)", name), name
    unlisted = {name for name in names if name.rsplit(".", 1)[0] not in listed}
    assert unlisted == {"l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum"}


def test_check_metrics_asks_ncu_about_a_chip(work):
    done, commands = run_ncu(work, "--check-metrics", "gb100")
    assert commands == ["--query-metrics --chips gb100", "--version"]
    assert done.returncode == 1                               # one name is not listed
    lines = done.stdout.splitlines()
    assert sum(" ok " in line for line in lines) == 18
    assert [line for line in lines if "UNLISTED" in line] == [
        "gb100 UNLISTED l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum"]
    unknown, _ = run_ncu(work, "--check-metrics", "zz999")
    assert unknown.returncode == 2 and "no metric table for chip 'zz999'" in unknown.stderr


def test_without_permission_the_script_says_so_and_goes_on(work):
    """The recorded output of the real ncu as a normal user."""
    out = work / "out"
    done, commands = run_ncu(work, "bin/hw", str(out), mode="permission")
    assert done.returncode == 0                               # run_all.sh must go on
    assert len(commands) == 1 and "smsp__inst_executed.sum bin/hw/launch" in commands[0]
    status = (out / "STATUS.txt").read_text()
    assert status.startswith("ncu not run: no permission to read GPU performance counters (ERR_NVGPUCTRPERM)")
    assert "ERR_NVGPUCTRPERM" in (out / "probe.csv").read_text()


def test_missing_ncu_or_programs(work):
    out = work / "out"
    done, _ = run_ncu(work, "bin/hw", str(out), NCU_BIN="no-such-ncu")
    assert done.returncode == 0 and "is not installed or not on PATH" in (out / "STATUS.txt").read_text()
    done, _ = run_ncu(work, "bin/elsewhere", str(out))
    assert "bin/elsewhere/launch is not built" in (out / "STATUS.txt").read_text()
    done, _ = run_ncu(work, "bin/hw", str(out), mode="none")
    assert "the probe produced no counter table" in (out / "STATUS.txt").read_text()


def test_profiling_writes_a_table_and_the_programs_own_lines(work):
    out = work / "out"
    done, commands = run_ncu(work, "bin/hw", str(out))
    assert done.returncode == 0, done.stderr
    assert (out / "STATUS.txt").read_text() == f"ncu: 11 benchmark(s) profiled, 0 failed; raw CSVs in {out}\n"
    lists = metric_lists((TIMING / "run_ncu.sh").read_text())
    for bench, which in (("alu_chain", "CORE"), ("smem_wavefronts", "SMEM"), ("tma", "BASIC")):
        table = (out / f"{bench}.csv").read_text()
        rows = [line for line in table.splitlines() if line.startswith('"0"')]
        assert [row.split('","')[12] for row in rows] == lists[which]
        # The program's own output is mixed into ncu's; it is kept apart as well.
        own = (out / f"{bench}.out").read_text().splitlines()
        assert own[0] == f"DEVICE bench={bench} sim=0" and own[1].startswith("BENCH id=stand_in")
    assert "args=--trials 1 --scale 0.01 --only ctas=1" in (out / "tma.out").read_text()
    # The transpose table is the recording from the 3080 Ti, handed to compare_ncu.py.
    assert (out / "transpose.csv").read_text().count("transpose_linear") == 7
    assert any(command.endswith("transpose_hw 64") for command in commands)


def test_a_rejected_counter_list_is_retried_without_the_unlisted_names(work):
    out = work / "out"
    done, commands = run_ncu(work, "bin/hw", str(out), mode="strict")
    assert done.returncode == 0, done.stderr
    status = (out / "STATUS.txt").read_text()
    assert status.startswith("ncu: 11 benchmark(s) profiled, 0 failed")
    assert "reduced counter list for smem_wavefronts: without l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum" in status
    assert "reduced counter list for ncu_banks: without" in status
    assert "reduced counter list for transpose: without" in status
    assert "alu_chain" not in status.split("\n", 1)[1]        # lists without the unlisted name ran as they were
    metrics = [row.split('","')[12] for row in (out / "smem_wavefronts.csv").read_text().splitlines()
               if row.startswith('"0"')]
    lists = metric_lists((TIMING / "run_ncu.sh").read_text())
    assert metrics == [m for m in lists["SMEM"] if "ideal" not in m]   # everything else was kept
    assert (out / "smem_wavefronts.first_attempt.csv").read_text().startswith("==ERROR==")
    attempts = [c for c in commands if "bin/hw/smem_wavefronts" in c]
    assert len(attempts) == 2 and "ideal" in attempts[0] and "ideal" not in attempts[1]


def test_ampere_script_dry_run(work):
    done = subprocess.run(["bash", str(work / "validation" / "ampere" / "run_ncu.sh"), "--dry-run", "16"],
                          capture_output=True, text=True, env={**os.environ, "NCU": "no-such-ncu", "PYTHON": "python3"})
    assert done.returncode == 0, done.stderr
    first, second = done.stdout.splitlines()
    assert first.startswith("dry-run: no-such-ncu --csv --metrics smsp__inst_executed_op_shared_ld.sum,")
    assert "./bin/transpose_hw 16 > ncu_transpose.csv" in first and first.rstrip().endswith("built")
    assert second == "dry-run: python3 compare_ncu.py ncu_transpose.csv 16"
    assert not (work / "validation" / "ampere" / "ncu_transpose.csv").exists()


def test_the_recorded_table_is_parsed_and_compared_with_the_simulator():
    """compare_ncu.py on a real `ncu --csv` table: the thousands separators, the "n/a" rows,
    the text before the header. The counters agree with the simulator, the total wavefronts
    included: `stats.smem.ncu_wavefronts` adds the 4160 that ncu counts beyond the shared
    accesses on the 3080 Ti (two per warp, one per CTA per barrier; matrix row TIM-17)."""
    import sys
    done = subprocess.run([sys.executable, str(AMPERE / "compare_ncu.py"), str(RECORDED), "64"], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0, done.stdout + done.stderr
    text = done.stdout
    assert text.count("grid 64, block 32x32") == 3
    row = lambda kernel, metric: next(l.split() for l in text.split(kernel)[1].splitlines() if l.strip().startswith(metric))
    assert row("transpose_linear", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")[1:] == ["63488", "63488", "="]
    assert row("transpose_linear", "smsp__inst_executed_op_shared_ld.sum")[1:] == ["2048", "2048", "="]
    assert row("transpose_linear", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")[1:] == ["71744", "71744", "="]
    assert row("transpose_swizzled", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")[1:] == ["8256", "8256", "="]
    assert row("transpose_padded", "l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum")[1] == "n/a"
    assert text.rstrip().endswith("all counters match")


def test_a_table_with_ncu_chatter_before_it_and_no_table_at_all(tmp_path):
    import sys
    chatter = tmp_path / "with_chatter.csv"
    chatter.write_text("==PROF== Connected to process 1 (/path/to/transpose_hw)\nok 3 layouts\n"
                       "==PROF== Disconnected from process 1\n" + RECORDED.read_text())
    done = subprocess.run([sys.executable, str(AMPERE / "compare_ncu.py"), str(chatter), "64"], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert done.stdout.rstrip().endswith("all counters match")
    denied = subprocess.run([sys.executable, str(AMPERE / "compare_ncu.py"), str(FIXTURE / "permission_denied.txt"), "64"],
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert denied.returncode != 0 and "no ncu CSV table found" in denied.stderr and "ERR_NVGPUCTRPERM" in denied.stderr
