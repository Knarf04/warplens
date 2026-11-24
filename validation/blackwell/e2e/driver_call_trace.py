#!/usr/bin/env python3
"""Which CUDA driver calls does a program make, and what does it hand the driver at each
kernel launch? (matrix row RT-10; e2e/cute_dsl_tma.py uses it for RT-11)

    validation/blackwell/e2e/driver_call_trace.py --out DIR [--timeout SECONDS] -- COMMAND...

    # the two launches of row RT-10, as validation/blackwell/run_rows.py runs them
    driver_call_trace.py --out out/tirx     -- python examples/07_tirx/run.py
    driver_call_trace.py --out out/cute_dsl -- python examples/06_cute_dsl/saxpy.py

COMMAND runs on the GPU under gdb with e2e/gdb_driver_trace.py, which says what is written
to DIR (calls.txt, summary.txt, launches.txt, launches.json, the raw parameter bytes), and
what more it can save of a launch (the code, the memory; e2e/cute_dsl_tma.py asks for that).
Needs gdb with Python support (Ubuntu's `gdb` package; WARPLENS_GDB names another one) and
permission to trace a child process, which a normal user has. No root, no symbols.

Standard output: the distinct driver calls in order of first use, then one line per launch.
Exit status: 0 the program ran to exit 0 and its calls were logged; 1 it did not; 2 with a
first line `SKIPPED: why` when it cannot be done here (no gdb, gdb without Python).

Measured on the development machine (RTX 3080 Ti, driver 580.173.02): saxpy.py takes 4 s
under the tracer, against 3 s without.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from probe_process import forward_interrupts, run_process  # noqa: E402

TRACER = HERE / "gdb_driver_trace.py"


def find_gdb() -> tuple[str, str]:
    """(path of a gdb that can run the tracer, "") or ("", why not)."""
    gdb = os.environ.get("WARPLENS_GDB") or shutil.which("gdb")
    if not gdb or not shutil.which(gdb):
        return "", ("gdb not found. As root: apt-get install -y gdb. Otherwise set WARPLENS_GDB to one "
                    "(env/remote/RUNBOOK.md, section 3)")
    try:
        check = subprocess.run([gdb, "-q", "-batch", "-ex", "python print('gdb-python-ok')"],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", f"{gdb} does not start: {error}"
    if "gdb-python-ok" not in check.stdout:
        return "", f"{gdb} has no Python support, which the tracer is written in"
    return gdb, ""


def trace(command: list[str], out: Path, timeout: float, env: dict | None = None,
          capture: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Run `command` under the tracer. Returns (status, facts): status is "ok", "skipped (why)"
    or what went wrong; facts has "launches" (launches.json) and "calls" (distinct names).
    `capture` names what else the tracer saves ("module", "memory": gdb_driver_trace.py)."""
    gdb, why = find_gdb()
    if not gdb:
        return f"skipped ({why})", {}
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("calls.txt", "summary.txt", "launches.txt", "launches.json", "exit.txt", "capture.json",
                  "capture.log"):
        (out / stale).unlink(missing_ok=True)
    for stale in [*out.glob("launch*.param*.bin"), *out.glob("module*.bin")]:
        stale.unlink()
    shutil.rmtree(out / "mem", ignore_errors=True)
    run_env = dict(os.environ if env is None else env)
    run_env["WARPLENS_TRACE_DIR"] = str(out)
    run_env["WARPLENS_TRACE_CAPTURE"] = ",".join(capture)
    status, _, stdout, stderr = run_process([gdb, "-q", "-batch", "-x", str(TRACER), "--args", *command],
                                            timeout, env=run_env)
    (out / "gdb.stdout").write_bytes(stdout)
    (out / "gdb.stderr").write_bytes(stderr)
    facts = {"launches": [], "calls": []}
    if (out / "launches.json").is_file():
        facts["launches"] = json.loads((out / "launches.json").read_text())
    if (out / "summary.txt").is_file():
        facts["calls"] = [line.split("\t")[0] for line in (out / "summary.txt").read_text().splitlines()]
    ended = (out / "exit.txt").read_text().strip() if (out / "exit.txt").is_file() else "no exit.txt"
    if status == "timeout":
        return "timeout", facts
    if not facts["calls"]:
        text = (stdout + stderr).decode(errors="replace")
        if "ptrace: Operation not permitted" in text or "Could not trace the inferior" in text:
            return "skipped (this machine does not let gdb trace a process: ptrace is not permitted)", facts
        return f"no driver call was logged (the program: {ended}; gdb: {status}); see gdb.stdout", facts
    return ("ok" if ended == "exit 0" else f"the program ended with: {ended}"), facts


def describe(launch: dict) -> str:
    sizes = [p["size"] for p in launch.get("parameters", [])]
    return (f"launch {launch['index']}: {launch['call']} {launch.get('kernel', '?')} grid={launch.get('grid')} "
            f"block={launch.get('block')} parameter sizes={sizes}"
            + (f" ({launch['note']})" if "note" in launch else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to trace after --")
    status, facts = trace(command, Path(args.out), args.timeout)
    if status.startswith("skipped"):
        print("SKIPPED: " + status[len("skipped ("):-1])
        return 2
    print(f"traced: {' '.join(command)}")
    print(f"status: {status}")
    print(f"driver calls, in order of first use ({len(facts['calls'])}): " + " ".join(facts["calls"]))
    for launch in facts["launches"]:
        print(describe(launch))
    if not facts["launches"]:
        print("no kernel launch reached the driver")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
