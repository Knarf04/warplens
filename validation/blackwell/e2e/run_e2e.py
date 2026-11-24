#!/usr/bin/env python3
"""End-to-end programs: the four CUTLASS tutorials, the Triton kernels, the book's GEMMs.

    validation/blackwell/e2e/run_e2e.py --out DIR [--book CLONE]              on the GPU
    validation/blackwell/e2e/run_e2e.py --mode sim --out DIR [--book CLONE]   in the simulator

Each step runs in its own process under a timeout; its output goes to DIR/e2e/<step>.stdout
and .stderr, and one line per step is appended to DIR/status.tsv. run_all.py calls this.

  cutlass_tutorials   examples/03_cutlass_mma/bin/tutorial_01 .. _04, built from unmodified
                      CUTLASS sources by `examples/03_cutlass_mma/build.sh --programs` (run
                      here if the programs are missing; needs nvcc and network access).
                      The same binaries run on the GPU and under `warplens run`.
                      --cutlass-bin DIR uses programs built elsewhere.
  triton              GPU: e2e/triton_gpu.py (PyTorch tensors, Triton's CUDA driver).
                      Simulator: `warplens run examples/05_triton/run.py`. Same kernels.
  book_hgemm          hgemm_v1 .. v9 of "Modern GPU Programming for MLSys", through
                      examples/08_book/run.py. The book has no license and is never copied:
                      pass your clone with --book (or WARPLENS_BOOK). Skipped without it.
                      v7 to v9 release tensor memory twice; on hardware that may fault.

--predicted DIR (simulator mode) also writes DIR/e2e/<step>.txt and .sim.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CUTLASS = REPO / "examples" / "03_cutlass_mma"
sys.path.insert(0, str(HERE.parent))
from probe_process import forward_interrupts, run_process  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gpu", "sim"], default="gpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--book", default=os.environ.get("WARPLENS_BOOK", ""))
    parser.add_argument("--cutlass-bin", default=os.environ.get("CUTLASS_BIN", str(CUTLASS / "bin")))
    parser.add_argument("--predicted")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "e2e").mkdir(parents=True, exist_ok=True)
    sim = args.mode == "sim"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if sim:
        env["CUDA_VISIBLE_DEVICES"] = ""
    warplens_run = [sys.executable, "-m", "warplens.cli", "run"]
    limit = (3600 if sim else 900) * args.timeout_scale

    infrastructure_failures = 0
    steps: list[tuple[str, list[str] | None, str]] = []  # (name, command or None, why not)
    bin_dir = Path(args.cutlass_bin)
    if not all((bin_dir / f"tutorial_0{n}").exists() for n in range(1, 5)) and bin_dir == CUTLASS / "bin":
        status, seconds, stdout, stderr = run_process([str(CUTLASS / "build.sh"), "--programs"], 3600)
        infrastructure_failures += status != "ok"
        (out / "e2e" / "cutlass_build.log").write_bytes(stdout + stderr)
        print(f"e2e: build the CUTLASS tutorials: {status} ({seconds:.0f} s)", flush=True)
    for n in range(1, 5):
        program = bin_dir / f"tutorial_0{n}"
        name = f"e2e/cutlass_tutorials.tutorial_0{n}"
        if not program.exists():
            infrastructure_failures += 1
            steps.append((name, None, f"{program} is not built"))
        else:
            steps.append((name, (warplens_run if sim else []) + [str(program)], ""))
    if sim:
        steps.append(("e2e/triton", warplens_run + [str(REPO / "examples" / "05_triton" / "run.py")], ""))
    else:
        steps.append(("e2e/triton", [sys.executable, str(HERE / "triton_gpu.py")], ""))
    book = Path(args.book) if args.book else None
    if book is None or not (book / "chapter_gemm_basics" / "index.md").is_file():
        steps.append(("e2e/book_hgemm", None, "no clone of the book given (--book or WARPLENS_BOOK)"))
    else:
        runner = [sys.executable, str(REPO / "examples" / "08_book" / "run.py"), str(book)]
        steps.append(("e2e/book_hgemm", (warplens_run if sim else []) + runner, ""))

    status_file = out / "status.tsv"
    if not status_file.exists():
        status_file.write_text("name\tstatus\tseconds\tcommand\n")
    for name, command, why in steps:
        if command is None:
            status, seconds, stdout, stderr = f"skipped ({why})", 0.0, b"", b""
        else:
            status, seconds, stdout, stderr = run_process(command, limit, env=env, cwd=REPO)
            infrastructure_failures += status.startswith("not run")
        (out / (name + ".stdout")).write_bytes(stdout)
        (out / (name + ".stderr")).write_bytes(stderr)
        shown = " ".join(command) if command else "-"
        with status_file.open("a") as f:
            f.write(f"{name}\t{status}\t{seconds:.1f}\t{shown}\n")
        print(f"{name:42} {status:10} {seconds:6.1f} s", flush=True)
        if args.predicted and sim and command is not None:
            target = Path(args.predicted) / (name + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stdout)
            target.with_suffix(".sim.txt").write_text(f"status: {status}\n" + stderr.decode(errors="replace"))
    return 1 if infrastructure_failures else 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
