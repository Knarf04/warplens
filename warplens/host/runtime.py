"""What a program needs around it to run with its GPU work done by the simulator.

Two entry points start such a program: the `warplens` command (warplens/cli.py), which prints
a summary of the launches, and `run_program` (warplens/host/program.py), which returns them
to Python. Both use what is here and neither uses the other:

* the shim libraries that stand in for the CUDA runtime and the driver: where they are, when
  they are built and when a build is stale (`shim_libraries`, `shim_staleness`);
* the environment of the program: the channel to the server, the shim libraries ahead of the
  real runtime, and the settings of the compilers that run inside it (`program_environment`,
  `frontend_environment`, `cute_dsl_runtime`);
* the program as a child process: started with its end of the channel, served until it
  closes that end, and never left running (`start_program`, `serve_program`).

Nothing here parses a command line or formats a report. A failure to build the shim libraries
is a `SystemExit` with the message for the user, as it has always been for `warplens run`;
`run_program` turns it into a `WarplensError`.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parent.parent / "shim"

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


def program_environment(model: str, shims: Path, channel, kept: Path, extra: dict | None = None) -> tuple[dict, Path]:
    """The environment of a program whose CUDA calls go to the simulator, and the directory
    its compilers keep their PTX in. It is this process's environment and `extra`, then:
    `channel`, the program's end of the socket pair, and the model, for the shim libraries to
    find; the frontend settings (`frontend_environment`, with `kept` as the default
    directory); and the shim libraries in `shims` ahead of the real CUDA runtime."""
    env = dict(os.environ, **(extra or {}), WARPLENS_SHIM_FD=str(channel.fileno()), WARPLENS_MODEL=model)
    kept = frontend_environment(env, model, shims, kept)
    env["LD_LIBRARY_PATH"] = f"{shims}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    # Preload as well: a program built with an RPATH would otherwise find the real runtime first.
    preload = [str(p) for p in sorted(shims.glob("libcudart.so.*")) + [shims / "libcuda.so.1"]]
    env["LD_PRELOAD"] = " ".join(preload + env.get("LD_PRELOAD", "").split())
    return env, kept


def start_program(command, env: dict, ours, theirs, **options) -> subprocess.Popen:
    """Start the program with `theirs`, its end of the socket pair, open in it. `options` are
    those of `subprocess.Popen`. If it cannot be started, both ends are closed and the error
    (an OSError, for a program that cannot be executed) is the caller's."""
    try:
        return subprocess.Popen(command, env=env, pass_fds=[theirs.fileno()], **options)
    except BaseException:
        ours.close(), theirs.close()
        raise


def serve_program(server, child: subprocess.Popen, ours, theirs, grace: float | None = None) -> int:
    """Serve a started program until it closes its end of the socket pair, and return its
    exit status. Whatever happens, both ends are closed and the program is not left running:
    a server exception or an interruption must not orphan it, as it may otherwise remain
    blocked waiting for its next RPC response. With `grace` it is asked to terminate and
    given that many seconds before it is killed; without, it is killed."""
    try:
        theirs.close()
        server.serve(ours)  # returns when the child closes its end
        ours.close()
        return child.wait()
    finally:
        theirs.close()
        ours.close()
        if child.poll() is None:
            if grace is None:
                child.kill()
            else:
                child.terminate()
                try:
                    child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    child.kill()
        child.wait()
