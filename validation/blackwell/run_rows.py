#!/usr/bin/env python3
"""The matrix rows that are not probes: NUM-05, RT-10 and RT-11. run_all.py calls this at
the end of a GPU run; it can also be run alone.

    validation/blackwell/run_rows.py --out DIR [--no-e2e] [--only STEP] [--timeout-scale X]

These rows have no prediction to compare with (RT-10 and RT-11 are informational, NUM-05
is open until a GPU conversion exists), so nothing here is a manifest entry and
compare.py does not read it. Each step runs in its own process under a timeout; its output
goes to DIR/rows/, one line per step to DIR/rows/status.tsv.

  formats/ue4m3_ptxas            NUM-05  does the installed ptxas have a UE4M3 conversion?
                                         (formats/ue4m3_ptxas.py; no GPU)
  e2e/driver_call_trace.tirx     RT-10   the driver calls of examples/07_tirx/run.py
  e2e/driver_call_trace.cute_dsl RT-10   and of examples/06_cute_dsl/saxpy.py, under gdb
                                         (e2e/driver_call_trace.py)
  e2e/cute_dsl_tma               RT-11   CUTLASS's fp16_gemm_0.py on the GPU, and the tensor
                                         maps it is handed (e2e/cute_dsl_tma.py)

Statuses: ok; `skipped (why)` when a step says it cannot run here (no gdb, no PyTorch, not
a B200: on the development machine RT-11 is "skipped (needs sm_100 ...)"); `news (...)` for
NUM-05 when ptxas accepts a UE4M3 conversion; anything else is a failure of that step.
The exit status is 0 unless this script itself breaks: a failed or skipped step is a line
in status.tsv, never a failed run. --no-e2e leaves out the three steps that launch kernels.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "e2e"))
from probe_process import forward_interrupts, run_process  # noqa: E402


def tvm_library_path(env: dict) -> str:
    """TVM's CUDA runtime needs the toolkit's libcudart on LD_LIBRARY_PATH (e2e/run_e2e.py,
    cudart_directory); the same environment as the untraced step e2e/unchanged_scripts.tirx."""
    try:
        from run_e2e import cudart_directory
        directory = cudart_directory()
    except Exception:  # the harness changed; the trace then shows what TVM does without it
        directory = ""
    return f"{directory}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":") if directory else env.get("LD_LIBRARY_PATH", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-e2e", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()
    rows = Path(args.out) / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    tvm_env = dict(env, LD_LIBRARY_PATH=tvm_library_path(env))
    python = sys.executable
    trace = [python, str(HERE / "e2e" / "driver_call_trace.py")]

    # (name, matrix row, command, environment, launches kernels)
    steps = [
        ("formats/ue4m3_ptxas", "NUM-05", [python, str(HERE / "formats" / "ue4m3_ptxas.py")], env, False),
        ("e2e/driver_call_trace.tirx", "RT-10",
         trace + ["--out", str(rows / "e2e" / "driver_call_trace.tirx"), "--",
                  python, str(REPO / "examples" / "07_tirx" / "run.py")], tvm_env, True),
        ("e2e/driver_call_trace.cute_dsl", "RT-10",
         trace + ["--out", str(rows / "e2e" / "driver_call_trace.cute_dsl"), "--",
                  python, str(REPO / "examples" / "06_cute_dsl" / "saxpy.py")], env, True),
        ("e2e/cute_dsl_tma", "RT-11",
         [python, str(HERE / "e2e" / "cute_dsl_tma.py"), "--out", str(rows / "e2e" / "cute_dsl_tma")], env, True),
    ]
    lines = ["name\trow\tstatus\tseconds\tcommand"]
    for name, row, command, step_env, launches in steps:
        if args.only and not any(fnmatch.fnmatch(name, pattern) for pattern in args.only):
            continue
        if launches and args.no_e2e:
            status, seconds, stdout, stderr = "skipped (--no-e2e)", 0.0, b"", b""
        else:
            # Twice the tracer's own limit: cute_dsl_tma.py runs the tutorial plain, then traced.
            status, seconds, stdout, stderr = run_process(command, 2000 * args.timeout_scale, env=step_env, cwd=REPO)
            first = (stdout.decode(errors="replace").splitlines() or [""])[0]
            if status == "exit 2" and first.startswith("SKIPPED:"):
                status = f"skipped ({first[len('SKIPPED:'):].strip()[:300]})"
            elif status == "exit 1" and name == "formats/ue4m3_ptxas":
                status = "news (ptxas accepts a UE4M3 conversion: NUM-05 can be tested; see the output)"
            elif status == "exit 1":
                # What the step itself says went wrong ("status: ..." or "FAILED: ...").
                said = next((line for line in stdout.decode(errors="replace").splitlines()
                             if line.startswith(("status:", "FAILED:"))), "")
                status = f"failed ({said.split(':', 1)[1].strip()[:300]})" if said else status
        target = rows / (name + ".stdout")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stdout)
        target.with_suffix(".stderr").write_bytes(stderr)
        lines.append(f"{name}\t{row}\t{status}\t{seconds:.1f}\t{' '.join(command)}")
        (rows / "status.tsv").write_text("\n".join(lines) + "\n")
        print(f"{name:42} {row:7} {status} ({seconds:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
