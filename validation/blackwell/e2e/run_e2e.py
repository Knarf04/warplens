#!/usr/bin/env python3
"""End-to-end programs: the four CUTLASS tutorials, the Triton kernels, the book's GEMMs.

    validation/blackwell/e2e/run_e2e.py --out DIR [--book CLONE]              on the GPU
    validation/blackwell/e2e/run_e2e.py --mode sim --out DIR [--book CLONE]   in the simulator

Each step runs in its own process under a timeout; its output goes to DIR/e2e/<step>.stdout
and .stderr, and one line per step (name, status, seconds, command, UTC start) is appended
to DIR/status.tsv. run_all.py calls this. The steps run in the order below: the one that may
fault on hardware, the book's, is last, so the others' results and captures are on disk first.

  cutlass_tutorials   host-instrumented copies of the four pinned CUTLASS tutorials;
                      capture_cutlass.py preserves the kernels and dumps full tensors/ABI.
                      --cutlass-source CLONE supplies the source; the default example's
                      build.sh acquires its pinned clone if its programs are absent.
                      Exact capture executables, sources and embedded PTX are archived.
                      The same capture binaries run on the GPU and under `warplens run`.
  triton              GPU: e2e/triton_gpu.py (PyTorch tensors, Triton's CUDA driver), run by
                      the interpreter WARPLENS_TORCH_PYTHON names (the PyTorch env that
                      env/remote/setup.sh makes; env/remote/run.sh sets the variable), or by
                      this one if it is not set. Without PyTorch the step is "skipped".
                      Simulator: `warplens run examples/05_triton/run.py`. Same kernels.
  unchanged_scripts   examples/07_tirx/run.py (TIRx, asking for PTX) and
                      examples/06_cute_dsl/saxpy.py (the CuTe DSL, arrays from driver
                      memory): the same script with plain `python` on the GPU and under
                      `warplens run` in the simulator (validation matrix RT-09). TIRx hands
                      the driver PTX from NVRTC, so a driver older than the toolkit refuses
                      it; that shows as a failed step whose output names the error.
  book_hgemm          hgemm_v1 .. v9 of "Modern GPU Programming for MLSys", through
                      examples/08_book/run.py. The book has no license and is never copied:
                      pass your clone with --book (or WARPLENS_BOOK). Skipped without it.
                      v7 to v9 release tensor memory twice; on hardware that may fault.
                      They are the last kernels of the last step (examples/08_book/run.py
                      runs v1 to v9 in order, in one process).

--predicted DIR (simulator mode) also writes DIR/e2e/<step>.txt and .sim.txt.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CUTLASS = REPO / "examples" / "03_cutlass_mma"
sys.path.insert(0, str(HERE.parent))
from probe_process import forward_interrupts, run_process  # noqa: E402
from e2e_checks import EXPECTED_CASES, completion_errors
from capture import validate_capture_tree  # noqa: E402


def cudart_directory() -> str:
    """The directory that holds the toolkit's shared CUDA runtime (libcudart.so.<major>), or "".

    TVM's CUDA runtime (tvm_tirx/lib/libtvm_runtime_cuda.so) links libcudart.so.13 and has no
    run path to it, so plain `python` in the env does not load it: TIRx then reports
    "CUDA compilation requires an explicit backend_config arch or Target arch when no CUDA
    device is available", on a machine with a working GPU. Measured on the development
    machine (RTX 3080 Ti): with this directory on LD_LIBRARY_PATH the same script runs.
    Programs built by nvcc carry a run path and do not need it. Under `warplens run` the
    simulator's own libcudart is used."""
    candidates = [Path(sys.prefix) / "lib"]
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc:
        toolkit = Path(nvcc).resolve().parent.parent
        candidates += [toolkit / "lib", toolkit / "lib64", toolkit / "targets" / "x86_64-linux" / "lib"]
    for directory in candidates:
        if glob.glob(str(directory / "libcudart.so.*")):
            return str(directory)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gpu", "sim"], default="gpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--book", default=os.environ.get("WARPLENS_BOOK", ""))
    parser.add_argument("--cutlass-bin", default=os.environ.get("CUTLASS_BIN", str(CUTLASS / "bin")),
                        help="original tutorial availability gate; capture programs are rebuilt from --cutlass-source")
    parser.add_argument("--cutlass-source", default=str(CUTLASS / ".cutlass"),
                        help="source clone used to build the executed, host-instrumented capture programs")
    parser.add_argument("--predicted")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    (out / "e2e").mkdir(parents=True, exist_ok=True)
    sim = args.mode == "sim"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if sim:
        env["CUDA_VISIBLE_DEVICES"] = ""
    warplens_run = [sys.executable, "-m", "warplens.cli", "run"]
    limit = (3600 if sim else 900) * args.timeout_scale

    infrastructure_failures = 0
    # GPU mode, the steps that launch through TVM's runtime: see cudart_directory().
    tvm_env = dict(env)
    if not sim and cudart_directory():
        tvm_env["LD_LIBRARY_PATH"] = f"{cudart_directory()}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    uses_tvm = ("e2e/unchanged_scripts.tirx", "e2e/book_hgemm")

    steps: list[tuple[str, list[str] | None, str]] = []  # (name, command or None, why not)
    bin_dir = Path(args.cutlass_bin).resolve()
    if not all((bin_dir / f"tutorial_0{n}").exists() for n in range(1, 5)) and bin_dir == CUTLASS / "bin":
        status, seconds, stdout, stderr = run_process([str(CUTLASS / "build.sh"), "--programs"], 3600)
        infrastructure_failures += status != "ok"
        (out / "e2e" / "cutlass_build.log").write_bytes(stdout + stderr)
        print(f"e2e: build the CUTLASS tutorials: {status} ({seconds:.0f} s)", flush=True)
    capture_build = out / "e2e" / "cutlass_capture_build"
    capture_tool = HERE / "capture_cutlass.py"
    requested = {n: bin_dir / f"tutorial_0{n}" for n in range(1, 5)}
    available = {n for n, program in requested.items() if program.is_file()}
    build_status = "not run (no requested tutorial programs)"
    if available:
        source = Path(args.cutlass_source).resolve()
        identity = (f"Capture programs are rebuilt from {source}.\n"
                    f"Original tutorial availability is checked in {bin_dir}; those binaries are not executed.\n")
        print("e2e: " + identity.replace("\n", " ").strip(), flush=True)
        build_command = [sys.executable, str(capture_tool), "build", "--cutlass", str(source),
                         "--out", str(capture_build)]
        build_status, _, build_stdout, build_stderr = run_process(build_command, 3600, env=env, cwd=REPO)
        (out / "e2e" / "cutlass_capture_build.log").write_bytes(identity.encode() + build_stdout + build_stderr)
        infrastructure_failures += build_status != "ok"
    for n in range(1, 5):
        program = capture_build / "bin" / f"tutorial_0{n}"
        name = f"e2e/cutlass_tutorials.tutorial_0{n}"
        if n not in available:
            infrastructure_failures += 1
            steps.append((name, None, f"{requested[n]} is not built"))
        elif build_status != "ok" or not program.exists():
            infrastructure_failures += 1
            steps.append((name, None, f"{program} is not built"))
        else:
            steps.append((name, (warplens_run if sim else []) + [str(program)], ""))
    if sim:
        steps.append(("e2e/triton", warplens_run + [str(REPO / "examples" / "05_triton" / "run.py")], ""))
    else:
        torch_python = os.environ.get("WARPLENS_TORCH_PYTHON") or sys.executable
        steps.append(("e2e/triton", [torch_python, str(HERE / "triton_gpu.py")], ""))
    for name, path in (("tirx", REPO / "examples" / "07_tirx" / "run.py"),
                       ("cute_dsl", REPO / "examples" / "06_cute_dsl" / "saxpy.py")):
        steps.append((f"e2e/unchanged_scripts.{name}", (warplens_run if sim else []) + [sys.executable, str(path)], ""))
    book = Path(args.book).resolve() if args.book else None
    if book is None or not (book / "chapter_gemm_basics" / "index.md").is_file():
        steps.append(("e2e/book_hgemm", None, "no clone of the book given (--book or WARPLENS_BOOK)"))
    else:
        runner = [sys.executable, str(REPO / "examples" / "08_book" / "run.py"), str(book)]
        steps.append(("e2e/book_hgemm", (warplens_run if sim else []) + runner, ""))

    status_file = out / "status.tsv"
    if not status_file.exists():
        status_file.write_text("name\tstatus\tseconds\tcommand\tstarted_utc\n")
    # The order of `steps` is the order of execution. The book's step stays last: it is the one
    # with kernels that may fault on hardware (the docstring), and a fault can cost the GPU.
    assert steps[-1][0] == "e2e/book_hgemm"
    for name, command, why in steps:
        capture_dir = out / "captures" / name.removeprefix("e2e/")
        if capture_dir.exists():
            shutil.rmtree(capture_dir)  # a rerun cannot pass using a previous run's fixtures
        step_env = dict(tvm_env if name in uses_tvm else env)
        step_env["WARPLENS_CAPTURE_DIR"] = str(capture_dir)
        step_env["WARPLENS_CAPTURE_MODE"] = args.mode
        tutorial = int(name[-1]) if name.startswith("e2e/cutlass_tutorials.") else None
        if tutorial is not None:
            raw_dir = out / "e2e" / f"cutlass_raw_0{tutorial}"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True)
            step_env["WARPLENS_CAPTURE_RAW_DIR"] = str(raw_dir)
        begun = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if command is None:
            status, seconds, stdout, stderr = f"skipped ({why})", 0.0, b"", b""
        else:
            status, seconds, stdout, stderr = run_process(command, limit, env=step_env,
                                                          cwd=REPO)
            if status == "exit 2" and stdout.startswith(b"SKIPPED:"):
                # The program found that it cannot run here (no PyTorch, no GPU) and said why.
                reason = stdout.decode(errors="replace").splitlines()[0][len("SKIPPED:"):].strip()
                status = f"skipped ({reason[:200]})"
        if status == "ok" and tutorial is not None:
            finalize = [sys.executable, str(capture_tool), "finalize", "--raw", str(raw_dir),
                        "--build", str(capture_build), "--tutorial", str(tutorial),
                        "--out", str(capture_dir), "--mode", args.mode]
            finalized, _, final_stdout, final_stderr = run_process(finalize, 120, env=step_env, cwd=REPO)
            (out / "e2e" / f"cutlass_finalize_0{tutorial}.log").write_bytes(final_stdout + final_stderr)
            if finalized != "ok":
                status = "capture-failed"
                stderr += b"\nCUTLASS capture finalization failed\n" + final_stderr
        errors = completion_errors(name, status, stdout.decode(errors="replace"))
        if status == "ok":
            errors.extend(validate_capture_tree(capture_dir, EXPECTED_CASES[name]))
        if errors:
            infrastructure_failures += 1
            if status == "ok":
                status = "verification-failed"
            stderr += ("\ncollection: " + "; ".join(errors) + "\n").encode()
        (out / (name + ".stdout")).write_bytes(stdout)
        (out / (name + ".stderr")).write_bytes(stderr)
        shown = " ".join(command) if command else "-"
        with status_file.open("a") as f:
            f.write(f"{name}\t{status}\t{seconds:.1f}\t{shown}\t{begun}\n")
        print(f"{name:42} {status:10} {seconds:6.1f} s", flush=True)
        if args.predicted and sim and command is not None:
            target = Path(args.predicted) / (name + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Committed files: no path of this machine in them (warnings name site-packages).
            def portable(data: bytes) -> str:
                return data.decode(errors="replace").replace(sys.prefix, "<env>").replace(str(REPO), "<repo>")
            target.write_text(portable(stdout))
            target.with_suffix(".sim.txt").write_text(f"status: {status}\n" + portable(stderr))
    return 1 if infrastructure_failures else 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
