#!/usr/bin/env python3
"""Run a CuTe DSL kernel that uses TMA on the B200 and save the tensor maps it is handed
(matrix row RT-11). Informational: there is no prediction.

    validation/blackwell/e2e/cute_dsl_tma.py --out DIR [--tutorial FILE] [--python PROGRAM] [--force]

Why: the DSL does not ask the driver for its tensor maps; its compiled host code writes the
128-byte descriptors itself, in the driver's layout, and the simulator only reads maps it
made (docs/m6.md, "Not working: kernels that use TMA"). The bytes saved here, next to the
driver's own encodings from tma/tensormap_encode_sweep (TMA-12), are the test data for a
decoder, after which such kernels can run in the simulator. Nothing else on the rental
produces them.

Steps, each under a timeout:
  1. CUTLASS's `fp16_gemm_0.py`, unmodified (it seeds its inputs itself: torch.manual_seed(1111)),
     with plain Python. Output in DIR/tutorial.stdout; it ends with PASS when the GEMM matches.
  2. The same under e2e/driver_call_trace.py: DIR/trace/launches.txt and launches.json hold
     every kernel parameter's offset, size and bytes; parameters of 64 bytes or more are
     also saved raw as DIR/trace/launchNN.paramMM.bin. A tensor map is a parameter of 128
     bytes, or 128 bytes inside a larger one.

The tutorial is the one in the CUTLASS clone that examples/03_cutlass_mma/build.sh makes
(same pinned commit; cloned here if it is missing). It needs PyTorch with CUDA: --python,
else WARPLENS_TORCH_PYTHON (env/remote/run.sh sets it to the PyTorch env of setup.sh), else
this interpreter.

Exit status: 0 the tutorial passed and its launch parameters were saved; 1 something
failed (the output says what); 2 with a first line `SKIPPED: why` when it cannot run here:
the GPU is not compute capability 10.x (--force runs it anyway), no CUDA PyTorch, no gdb.
`--script FILE` traces another DSL script in place of the tutorial and skips the checks for
sm_100 and PyTorch; the dry run on the development machine uses it with
examples/06_cute_dsl/saxpy.py.

Not done here: the DSL's experimental `create_tensor_map_tiled`, which docs/m6.md says may
go through the driver. It needs a kernel written for it; env/remote/RUNBOOK.md lists it as
an optional manual step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from driver_call_trace import describe, find_gdb, trace  # noqa: E402
from probe_process import forward_interrupts, run_process  # noqa: E402

CUTLASS = REPO / "examples" / "03_cutlass_mma"
TUTORIAL = CUTLASS / ".cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py"


def compute_capability() -> str:
    """Of the first GPU nvidia-smi lists ("" if it cannot say)."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip().splitlines()[0].strip() if done.returncode == 0 and done.stdout.strip() else ""


def skipped(why: str) -> int:
    print(f"SKIPPED: {why}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tutorial", default=str(TUTORIAL))
    parser.add_argument("--script", help="trace this DSL script instead (no sm_100 or PyTorch check)")
    parser.add_argument("--python", default=os.environ.get("WARPLENS_TORCH_PYTHON") or sys.executable)
    parser.add_argument("--force", action="store_true", help="run on a GPU that is not compute capability 10.x")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    out = Path(args.out)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")

    if not find_gdb()[0]:
        return skipped(find_gdb()[1])
    if args.script:
        program = Path(args.script)
    else:
        capability = compute_capability()
        if not capability.startswith("10.") and not args.force:
            return skipped(f"needs sm_100 (a B200, compute capability 10.x); this GPU reports "
                           f"'{capability or 'nothing'}'. The tutorial's kernel is tcgen05 and TMA")
        check = subprocess.run([args.python, "-c", "import torch, cutlass; assert torch.cuda.is_available()"],
                               capture_output=True, text=True, env=env)
        if check.returncode != 0:
            return skipped(f"{args.python} has no PyTorch that sees the GPU, or no CuTe DSL "
                           f"({(check.stderr.strip().splitlines() or ['no message'])[-1][:160]}). "
                           "Set WARPLENS_TORCH_PYTHON to the interpreter of the PyTorch env")
        program = Path(args.tutorial)
        if not program.is_file() and program == TUTORIAL:
            # build.sh holds the pinned CUTLASS commit; it clones, then compiles four tutorials to PTX.
            status, _, stdout, stderr = run_process([str(CUTLASS / "build.sh")], 1800)
            out.mkdir(parents=True, exist_ok=True)
            (out / "cutlass_clone.log").write_bytes(stdout + stderr)
    if not program.is_file():
        print(f"FAILED: {program} not found (the CUTLASS clone: examples/03_cutlass_mma/build.sh; or --tutorial)")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    command = [args.python, str(program)]

    plain, seconds, stdout, stderr = run_process(command, args.timeout, env=env, cwd=REPO)
    (out / "tutorial.stdout").write_bytes(stdout)
    (out / "tutorial.stderr").write_bytes(stderr)
    last = (stdout.decode(errors="replace").strip().splitlines() or ["(no output)"])[-1]
    print(f"plain run of {program.name}: {plain} ({seconds:.0f} s), last line: {last[:120]}")

    status, facts = trace(command, out / "trace", args.timeout, env=env)
    print(f"traced run: {status}")
    launches = facts.get("launches", [])
    for launch in launches:
        print(describe(launch))
    read = [p for launch in launches for p in launch.get("parameters", [])]
    candidates = [p for p in read if p["size"] >= 128]
    print(f"{len(launches)} launch(es), {len(read)} parameter(s) read, {len(candidates)} of 128 bytes or more "
          f"(where a tensor map can be); bytes in {out / 'trace' / 'launches.txt'}")
    if plain != "ok" or status != "ok" or not read:
        print("FAILED: " + ("the plain run did not pass; " if plain != "ok" else "")
              + ("the traced run did not finish cleanly; " if status != "ok" else "")
              + ("no kernel parameter was read" if not read else ""))
        return 1
    return 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
