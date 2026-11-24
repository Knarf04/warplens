"""The `warplens` command.

    warplens run [options] program [arguments...]

runs a program with its GPU work executed by the simulator. `program` is either a
compiled CUDA executable, or a Python script (a `.py` file) that uses Triton, TIRx or the
CuTe DSL; `warplens run script.py` is `warplens run python script.py` with this interpreter.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parent / "shim"
# What `warplens run` tells the Python programs it starts, as JSON: the model, the launch
# options, and the file to report the launches of a Triton driver in. See `child_options`.
RUN_OPTIONS = "WARPLENS_RUN_OPTIONS"

BUILD_HINT = """\
The program must carry PTX and use the shared CUDA runtime. Build it like this:
  nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog"""


SHIM_STAMP = "warplens-shim.stamp"  # written by warplens/shim/build.sh next to the libraries
SHIM_HEADERS = ("cuda_runtime_api.h", "cuda.h")


def _digest(directory: Path, names) -> str:
    """A digest of files of one directory: of the lines `sha256sum` prints for them, which
    is how warplens/shim/build.sh computes the same number."""
    lines = "".join(f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}\n" for name in names)
    return hashlib.sha256(lines.encode()).hexdigest()


def shim_stamp(include: Path | None = None) -> dict:
    """What build.sh records about a build made now: `sources`, a digest of the shim's C
    sources, protocol.h and build.sh; and with `include` (a CUDA include directory) a digest
    of the CUDA headers there that the shim is compiled against."""
    names = sorted(p.name for p in SHIM_DIR.iterdir() if p.suffix in {".c", ".h"} or p.name == "build.sh")
    stamp = {"sources": _digest(SHIM_DIR, names)}
    if include is not None:
        stamp["headers"] = _digest(include, [name for name in SHIM_HEADERS if (include / name).is_file()])
    return stamp


def shim_complete(lib: Path) -> bool:
    return (lib / "libcuda.so.1").is_file() and any(p.is_file() for p in lib.glob("libcudart.so.*"))


def _recorded_stamp(lib: Path) -> dict | None:
    """The stamp build.sh left in `lib`: `sources`, `headers` and `include`. None without one."""
    try:
        lines = (lib / SHIM_STAMP).read_text().splitlines()
    except OSError:
        return None
    return dict(line.split(" ", 1) for line in lines if " " in line)


def shim_staleness(lib: Path, sources: str | None = None) -> str | None:
    """Why the libraries in `lib` must not be used with the shim sources of this package, or
    None when they were built from exactly these sources (and from the CUDA headers that are
    still where they were). Costs a hash of the sources and two headers: called once for
    every `warplens run`, never for a launch."""
    if not shim_complete(lib):
        return "holds no complete pair of shim libraries"
    recorded = _recorded_stamp(lib)
    if recorded is None:
        return (f"has no {SHIM_STAMP}: the libraries were built by an older build.sh, or a build was "
                "interrupted, and nothing says which sources they are from")
    if recorded.get("sources") != (sources or shim_stamp()["sources"]):
        return "was built from other shim sources (cudart_shim.c, cuda_shim.c, protocol.h or build.sh changed since)"
    # A build made by hand stays usable where the CUDA headers are not installed any more.
    include = Path(recorded.get("include", ""))
    if recorded.get("include") and (include / SHIM_HEADERS[0]).is_file() \
            and shim_stamp(include)["headers"] != recorded.get("headers"):
        return f"was built against the CUDA headers in {include}, which have changed since"
    return None


def _rebuild_local_shim(lib: Path, sources: str) -> None:
    """Rebuild a build made by hand (warplens/shim/lib) whose sources changed, or stop: a
    stale library must never be loaded as if it were current."""
    try:
        with (lib / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            why = shim_staleness(lib, sources)
            if why is None:  # another run rebuilt it meanwhile
                return
            print(f"warplens: {lib} {why}; rebuilding it", file=sys.stderr)
            # Against the headers it was built with, if they are still there; else nvcc's.
            include = (_recorded_stamp(lib) or {}).get("include", "")
            if not include or not (Path(include) / SHIM_HEADERS[0]).is_file():
                include = ""
            done = subprocess.run([str(SHIM_DIR / "build.sh"), include, str(lib)], capture_output=True, text=True)
            failure = (f"{done.stdout}{done.stderr}".strip() or f"build.sh exited with status {done.returncode}"
                       if done.returncode else shim_staleness(lib, sources))
    except OSError as error:
        why, failure = shim_staleness(lib, sources), str(error)
    if failure is not None:
        sys.exit(f"warplens: {lib} {why}, and it could not be rebuilt:\n{failure}\n"
                 f"The stale libraries are not used. Run {SHIM_DIR / 'build.sh'} <CUDA include directory> by "
                 f"hand, or remove {lib} to have them built in the user cache.")


def shim_libraries() -> Path:
    """The directory holding the shim libraries, building them on first use, and again when
    the sources they were built from are not the ones here (`shim_staleness`)."""
    sources = shim_stamp()["sources"]
    # A manual build remains usable without a compiler or installed CUDA headers, as long as
    # it is a build of these sources.
    local = SHIM_DIR / "lib"
    if shim_complete(local):
        if shim_staleness(local, sources) is not None:
            _rebuild_local_shim(local, sources)
        return local
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
            # The cache directory is named after the sources and the toolkit; the stamp says
            # that what is in it is a finished build of them.
            if shim_staleness(lib, sources) is not None:
                print("warplens: building the shim libraries (first run)", file=sys.stderr)
                done = subprocess.run([str(SHIM_DIR / "build.sh"), "", str(lib)],
                                      capture_output=True, text=True)
                if done.returncode or shim_staleness(lib, sources) is not None:
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


def launch_lines(name: str, stats) -> list[str]:
    """The lines of the summary for one launch: what ran, then its findings."""
    line = f"  {short(name)}: {stats.ctas} CTA(s), {stats.instructions} warp-instructions"
    if stats.canceled_ctas:  # cluster launch control: their work was done by CTAs that did start
        line = line.replace(" CTA(s)", f" CTA(s) of which {stats.canceled_ctas} canceled before starting")
    if stats.timing is not None:
        line += f", {stats.timing.time_ns:,.0f} ns simulated ({stats.timing.bottleneck}-bound)"
    return [line, *("    " + str(finding).replace("\n", "\n    ") for finding in stats.findings)]


def summary(launches, model: str, reported=()) -> None:
    """`launches` are (kernel name, LaunchStats) of this process. `reported` are the launches
    of a Triton driver in the program, as it wrote them: {"lines": [...], "findings": count}."""
    out = sys.stderr
    findings = sum(len(s.findings) for _, s in launches) + sum(r["findings"] for r in reported)
    print(f"warplens: {len(launches) + len(reported)} kernel launch(es) simulated on {model}, "
          f"{findings} finding(s)", file=out)
    for name, stats in launches:
        print(*launch_lines(name, stats), sep="\n", file=out)
    for launch in reported:
        print(*launch["lines"], sep="\n", file=out)


def launch_options(timing=False, timing_config=None, active_sms=None, clc_limit=None) -> dict:
    """The keyword arguments of `Device.launch` that the options of `warplens run` stand for.
    A timing config or an SM count asks for the timing model by itself."""
    options = {}
    if timing_config is not None:
        from .timing.config import GpuConfig
        options["timing"] = GpuConfig.load(timing_config)
    elif timing or active_sms is not None:
        options["timing"] = True
    if active_sms is not None:
        options["active_sms"] = active_sms
    if clc_limit is not None:
        options["clc_limit"] = clc_limit
    return options


def child_options(args) -> dict:
    """The options of this run in the form that crosses to a Python program (RUN_OPTIONS)."""
    config = getattr(args, "timing_config", None)
    return {"model": args.model, "timing": bool(args.timing),
            "timing_config": None if config is None else str(config),
            "active_sms": getattr(args, "active_sms", None), "clc_limit": getattr(args, "clc_limit", None)}


def install_triton_driver() -> None:
    """In a Python program under `warplens run`, once it has imported Triton (called by
    warplens/shim/python/sitecustomize.py): make Triton target the simulator with the options
    of the run, and report the launches to `warplens run` when the program ends."""
    from . import triton_driver
    options = json.loads(os.environ.get(RUN_OPTIONS) or "{}")
    report = options.pop("report", None)
    options.pop("package", None)
    driver = triton_driver.install(options.pop("model", "B200"), **launch_options(**options))

    def tell() -> None:
        launches = [{"lines": launch_lines(name, stats), "findings": len(stats.findings)}
                    for name, stats in driver.launches]
        if report and launches:
            try:
                with open(report, "a") as file:  # one line per process that launched
                    file.write(json.dumps(launches) + "\n")
            except OSError as error:
                print(f"warplens: could not report the Triton launches: {error}", file=sys.stderr)

    atexit.register(tell)


def run_script(args) -> int:
    """A Python script, where the shim libraries cannot be had: make Triton target the
    simulator, then run the script in this process. Only its Triton kernels are simulated.
    The script creates its tensors with `triton.runtime.driver.active.tensor(...)`."""
    import runpy
    try:
        from . import triton_driver
    except ImportError as exc:
        sys.exit(f"warplens: {exc}")
    options = child_options(args)
    driver = triton_driver.install(options.pop("model"), **launch_options(**options))
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
    # A Python script is run by this interpreter, like `warplens run python script.py`: under
    # the shim, which serves TIRx and the CuTe DSL, and with the start-up hook that installs
    # the Triton driver if the script imports Triton. One command for the three of them.
    script = program.endswith(".py")
    try:
        lib = shim_libraries()
    except SystemExit as failure:
        if not script:
            raise
        # Triton needs neither a C compiler nor the CUDA headers, and must not start to.
        print(f"{failure}\nwarplens: running {args.program} without the shim libraries: its Triton "
              "kernels are simulated, but a TIRx or CuTe DSL script will not find a device",
              file=sys.stderr)
        lib = None
    if lib is None:  # outside the handler, so that the script's own errors are not chained to it
        args.program = program
        return run_script(args)
    command = [sys.executable, program] if script else [program]
    options = child_options(args)
    launch = launch_options(**{k: v for k, v in options.items() if k != "model"})
    ours, theirs = socket.socketpair()
    env = dict(os.environ, WARPLENS_SHIM_FD=str(theirs.fileno()), WARPLENS_MODEL=args.model)
    scratch = tempfile.TemporaryDirectory(prefix="warplens_kept_ptx_")
    handle, report = tempfile.mkstemp(prefix="warplens_triton_launches_")
    os.close(handle)
    env[RUN_OPTIONS] = json.dumps({**options, "report": report,
                                   "package": str(Path(__file__).resolve().parent.parent)})
    server = ShimServer(args.model, kept_ptx=frontend_environment(env, args.model, lib, Path(scratch.name)),
                        **{"timing": False, **launch})
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    # Preload as well: a program built with an RPATH would otherwise find the real runtime first.
    shims = [str(p) for p in sorted(lib.glob("libcudart.so.*")) + [lib / "libcuda.so.1"]]
    env["LD_PRELOAD"] = " ".join(shims + env.get("LD_PRELOAD", "").split())
    child = None
    try:
        try:
            child = subprocess.Popen([*command, *args.arguments], env=env, pass_fds=[theirs.fileno()])
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
        reported = []  # by the Triton driver of a Python program: see install_triton_driver
        try:
            for line in Path(report).read_text().splitlines():
                reported += json.loads(line)
        except (OSError, ValueError):
            pass
        Path(report).unlink(missing_ok=True)

    if not args.quiet:
        out = sys.stderr
        if reported:
            summary(server.launches, args.model, reported)
        elif server.launches:  # two arguments: validation/blackwell_timing/predict.py replaces it
            summary(server.launches, args.model)
        elif script:
            print("warplens: the script launched no kernels", file=out)
        else:
            print("warplens: the program launched no kernels", file=out)
            if not server.machine_code and (server.needs_ptx or not server.modules):
                print(BUILD_HINT, file=out)
    return code if code >= 0 else 128 - code


def count(least: int):
    """An argparse type: an integer that is at least `least`."""
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            value = None
        if value is None or value < least:
            raise argparse.ArgumentTypeError(f"must be an integer that is at least {least}, got {text!r}")
        return value
    return parse


def check_timing(runner, args) -> None:
    """Refuse a timing config or an SM count that no launch could use, before anything runs."""
    if args.timing_config is None and args.active_sms is None:
        return
    from .errors import LaunchError
    from .timing.config import GpuConfig
    try:
        config = GpuConfig.load(args.model if args.timing_config is None else args.timing_config)
    except LaunchError as error:
        runner.error(f"argument --timing-config: {error}")
    if args.active_sms is not None and args.active_sms > config["sm_count"]:
        runner.error(f"argument --active-sms: must be from 1 to {int(config['sm_count'])}, the SM count "
                     f"of timing config `{config.name}`, got {args.active_sms}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="warplens", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run a compiled CUDA program, or a Triton, TIRx or CuTe DSL "
                                             "script, in the simulator",
                                 epilog=BUILD_HINT, formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_argument("--model", default="B200", choices=["B200", "B300"], help="GPU to simulate")
    runner.add_argument("--timing", action="store_true",
                        help="also run the approximate timing model (uncalibrated)")
    runner.add_argument("--timing-config", metavar="CONFIG",
                        help="the parameters of the timing model: a YAML file like warplens/configs/b200.yaml, "
                             "or the name of a built-in config (b200, b300). Implies --timing. "
                             "Default: the config of --model")
    runner.add_argument("--active-sms", type=count(1), metavar="N",
                        help="how many SMs the timing model assumes to share the memory bandwidth, from 1 to "
                             "the SM count of the timing config. Implies --timing. Default: all of them")
    runner.add_argument("--clc-limit", type=count(0), metavar="N",
                        help="cluster launch control: how many cancellations each cluster that starts is "
                             "granted before its requests fail; 0 makes every cluster start by itself. "
                             "Default: every request is granted while clusters are left")
    runner.add_argument("--quiet", action="store_true", help="do not print the summary")
    runner.add_argument("program", help="a compiled CUDA executable, or a Python script (a .py file), "
                                        "which is run by this Python interpreter")
    runner.add_argument("arguments", nargs=argparse.REMAINDER)
    runner.set_defaults(handler=run)
    args = parser.parse_args(argv)
    check_timing(runner, args)
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
