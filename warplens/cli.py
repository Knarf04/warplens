"""The `warplens` command.

    warplens run [options] program [arguments...]

runs a program with its GPU work executed by the simulator. `program` is either a
compiled CUDA executable, or a Python script that uses Triton (a `.py` file).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parent / "shim"

BUILD_HINT = """\
The program must carry PTX and use the shared CUDA runtime. Build it like this:
  nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog"""


def shim_libraries() -> Path:
    """The directory holding the shim libraries, building them on first use."""
    def complete(lib):
        return (lib / "libcuda.so.1").is_file() and any(p.is_file() for p in lib.glob("libcudart.so.*"))

    # A manual build remains usable without a compiler or installed CUDA headers.
    if complete(SHIM_DIR / "lib"):
        return SHIM_DIR / "lib"
    digest = hashlib.sha256()
    for source in sorted(SHIM_DIR.iterdir()):
        if source.suffix in {".c", ".h"} or source.name == "build.sh":
            digest.update(source.name.encode() + b"\0" + source.read_bytes())
    # The toolkit changes the runtime SONAME and exported ABI, even when our
    # sources have not changed. Never reuse another toolkit's cached build.
    if nvcc := shutil.which("nvcc"):
        root = Path(nvcc).resolve().parent.parent
        digest.update(str(root).encode())
        for include in (root / "targets/x86_64-linux/include", root / "include"):
            if (include / "cuda_runtime_api.h").is_file():
                for name in ("cuda_runtime_api.h", "cuda.h"):
                    header = include / name
                    if header.is_file():
                        digest.update(header.read_bytes())
                break
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    lib = (cache / "warplens" / "shim" / digest.hexdigest()).absolute()
    try:
        lib.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent first runs share the build, rather than loading half of it.
        with lib.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not complete(lib):
                print("warplens: building the shim libraries (first run)", file=sys.stderr)
                done = subprocess.run([str(SHIM_DIR / "build.sh"), "", str(lib)],
                                      capture_output=True, text=True)
                if done.returncode or not complete(lib):
                    sys.exit(f"warplens: could not build the shim libraries:\n{done.stdout}{done.stderr}\n"
                             f"Run {SHIM_DIR / 'build.sh'} <CUDA include directory> by hand.")
    except OSError as error:
        sys.exit(f"warplens: could not build the shim libraries: {error}")
    return lib


def cute_dsl_runtime(shims: Path) -> Path | None:
    """A runtime library for the CuTe DSL that calls the shims, built on first use from the
    wrapper archive in the installed `nvidia-cutlass-dsl`. None when the DSL is not installed."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return None
    archives = sorted(Path(root).glob("cu*/lib/libcuda_dialect_runtime_static.a")
                      for root in spec.submodule_search_locations)
    archives = [a for found in archives for a in found]
    if not archives:
        return None
    archive = archives[-1]
    digest = hashlib.sha256((SHIM_DIR / "build_cute_dsl.sh").read_bytes())
    for part in (archive, *sorted(shims.glob("libcudart.so.*"))):
        stat = part.stat()
        digest.update(f"{part}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    out = (cache / "warplens" / "cute_dsl" / digest.hexdigest()).absolute()
    library = out / "libcute_dsl_runtime.so"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.is_file():
                done = subprocess.run([str(SHIM_DIR / "build_cute_dsl.sh"), str(archive), str(shims), str(out)],
                                      capture_output=True, text=True)
                if done.returncode or not library.is_file():
                    print(f"warplens: could not build the CuTe DSL runtime; its launches will not be "
                          f"simulated:\n{done.stdout}{done.stderr}", file=sys.stderr)
                    return None
    except OSError as error:
        print(f"warplens: could not build the CuTe DSL runtime: {error}", file=sys.stderr)
        return None
    return library


def frontend_environment(env: dict, model: str, shims: Path, kept: Path) -> Path:
    """Point the compilers that run inside the program at the simulator: the architecture to
    compile for, and, for the CuTe DSL, the runtime library to launch through and the request
    to keep its PTX. Settings the user made win. Returns the directory the PTX is kept in."""
    env.setdefault("CUTE_DSL_ARCH", "sm_103a" if model == "B300" else "sm_100a")
    if "CUTE_DSL_LIBS" not in env and (runtime := cute_dsl_runtime(shims)):
        env["CUTE_DSL_LIBS"] = str(runtime)
    tokens = [t.strip() for t in env.get("CUTE_DSL_KEEP", "").split(",") if t.strip()]
    if not {"ptx", "all"} & {t.lower() for t in tokens}:
        env["CUTE_DSL_KEEP"] = ",".join([*tokens, "ptx"])
    kept = Path(env.setdefault("CUTE_DSL_DUMP_DIR", str(kept)))
    # A Python program gets warplens/shim/python/sitecustomize.py.
    env["PYTHONPATH"] = os.pathsep.join([str(SHIM_DIR / "python"), *filter(None, [env.get("PYTHONPATH")])])
    return kept


def short(name: str, width: int = 64) -> str:
    return name if len(name) <= width else name[:width - 3] + "..."


def summary(launches, model: str) -> None:
    out = sys.stderr
    findings = sum(len(s.findings) for _, s in launches)
    print(f"warplens: {len(launches)} kernel launch(es) simulated on {model}, "
          f"{findings} finding(s)", file=out)
    for name, stats in launches:
        line = f"  {short(name)}: {stats.ctas} CTA(s), {stats.instructions} warp-instructions"
        if stats.timing is not None:
            line += f", {stats.timing.time_ns:,.0f} ns simulated ({stats.timing.bottleneck}-bound)"
        print(line, file=out)
        for finding in stats.findings:
            print("    " + str(finding).replace("\n", "\n    "), file=out)


def run_script(args) -> int:
    """A Python script: make Triton target the simulator, then run the script in this process.
    The script creates its tensors with `triton.runtime.driver.active.tensor(...)`."""
    import runpy
    try:
        from . import triton_driver
    except ImportError as exc:
        sys.exit(f"warplens: {exc}")
    driver = triton_driver.install(args.model, **({"timing": True} if args.timing else {}))
    previous_argv, previous_path = sys.argv, sys.path[:]
    sys.argv = [args.program, *args.arguments]
    # run_path does not add a plain script's directory to the import path as
    # `python script.py` does. Preserve the surrounding interpreter afterwards.
    sys.path.insert(0, str(Path(args.program).absolute().parent))
    code = 0
    try:
        runpy.run_path(args.program, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        triton_driver.uninstall()
    if not args.quiet:
        if driver.launches:
            summary(driver.launches, args.model)
        else:
            print("warplens: the script launched no Triton kernels", file=sys.stderr)
    return code


def run(args) -> int:
    from .host.shim_server import ShimServer

    program = shutil.which(args.program) or args.program
    if not Path(program).exists():
        sys.exit(f"warplens: no such program: {args.program}")
    program = str(Path(program).absolute())
    if program.endswith(".py"):
        args.program = program
        return run_script(args)
    lib = shim_libraries()
    ours, theirs = socket.socketpair()
    env = dict(os.environ, WARPLENS_SHIM_FD=str(theirs.fileno()), WARPLENS_MODEL=args.model)
    scratch = tempfile.TemporaryDirectory(prefix="warplens_kept_ptx_")
    server = ShimServer(args.model, timing=args.timing,
                        kept_ptx=frontend_environment(env, args.model, lib, Path(scratch.name)))
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    # Preload as well: a program built with an RPATH would otherwise find the real runtime first.
    shims = [str(p) for p in sorted(lib.glob("libcudart.so.*")) + [lib / "libcuda.so.1"]]
    env["LD_PRELOAD"] = " ".join(shims + env.get("LD_PRELOAD", "").split())
    child = None
    try:
        try:
            child = subprocess.Popen([program, *args.arguments], env=env, pass_fds=[theirs.fileno()])
        except OSError as error:
            sys.exit(f"warplens: cannot execute {args.program}: {error.strerror}")
        theirs.close()
        server.serve(ours)
        ours.close()
        code = child.wait()
    finally:
        theirs.close()
        ours.close()
        # A server exception or interruption must not orphan the CUDA program,
        # which may otherwise remain blocked waiting for its next RPC response.
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait()
        scratch.cleanup()

    if not args.quiet:
        out = sys.stderr
        if not server.launches:
            print("warplens: the program launched no kernels", file=out)
            if not server.machine_code and (server.needs_ptx or not server.modules):
                print(BUILD_HINT, file=out)
        else:
            summary(server.launches, args.model)
    return code if code >= 0 else 128 - code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="warplens", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run a compiled CUDA program or a Triton script in the simulator",
                                 epilog=BUILD_HINT, formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_argument("--model", default="B200", choices=["B200", "B300"], help="GPU to simulate")
    runner.add_argument("--timing", action="store_true",
                        help="also run the approximate timing model (uncalibrated)")
    runner.add_argument("--quiet", action="store_true", help="do not print the summary")
    runner.add_argument("program")
    runner.add_argument("arguments", nargs=argparse.REMAINDER)
    runner.set_defaults(handler=run)
    args = parser.parse_args(argv)
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
