"""The `warplens` command.

    warplens run [options] program [arguments...]

runs a compiled CUDA program with its GPU work executed by the simulator.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parent / "shim"

BUILD_HINT = """\
The program must carry PTX and use the shared CUDA runtime. Build it like this:
  nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog"""


def shim_libraries() -> Path:
    """The directory holding the shim libraries, building them on first use."""
    lib = SHIM_DIR / "lib"
    if not any(lib.glob("libcudart.so.*")):
        print("warplens: building the shim libraries (first run)", file=sys.stderr)
        done = subprocess.run([str(SHIM_DIR / "build.sh")], capture_output=True, text=True)
        if done.returncode:
            sys.exit(f"warplens: could not build the shim libraries:\n{done.stdout}{done.stderr}\n"
                     f"Run {SHIM_DIR / 'build.sh'} <CUDA include directory> by hand.")
    return lib


def short(name: str, width: int = 64) -> str:
    return name if len(name) <= width else name[:width - 3] + "..."


def run(args) -> int:
    from .host.shim_server import ShimServer

    program = shutil.which(args.program) or args.program
    if not Path(program).exists():
        sys.exit(f"warplens: no such program: {args.program}")
    lib = shim_libraries()
    ours, theirs = socket.socketpair()
    env = dict(os.environ, WARPLENS_SHIM_FD=str(theirs.fileno()), WARPLENS_MODEL=args.model)
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    # Preload as well: a program built with an RPATH would otherwise find the real runtime first.
    shims = [str(p) for p in sorted(lib.glob("libcudart.so.*")) + [lib / "libcuda.so.1"]]
    env["LD_PRELOAD"] = " ".join(shims + env.get("LD_PRELOAD", "").split())
    child = subprocess.Popen([program, *args.arguments], env=env, pass_fds=[theirs.fileno()])
    theirs.close()
    server = ShimServer(args.model, timing=args.timing)
    try:
        server.serve(ours)
    finally:
        ours.close()
    code = child.wait()

    if not args.quiet:
        out = sys.stderr
        if not server.launches:
            print("warplens: the program launched no kernels", file=out)
            if server.needs_ptx or not server.modules:
                print(BUILD_HINT, file=out)
        else:
            findings = sum(len(s.findings) for _, s in server.launches)
            print(f"warplens: {len(server.launches)} kernel launch(es) simulated on {args.model}, "
                  f"{findings} finding(s)", file=out)
            for name, stats in server.launches:
                line = f"  {short(name)}: {stats.ctas} CTA(s), {stats.instructions} warp-instructions"
                if stats.timing is not None:
                    line += f", {stats.timing.time_ns:,.0f} ns simulated ({stats.timing.bottleneck}-bound)"
                print(line, file=out)
                for finding in stats.findings:
                    print("    " + str(finding).replace("\n", "\n    "), file=out)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="warplens", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run a compiled CUDA program in the simulator",
                                 epilog=BUILD_HINT, formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_argument("--model", default="B200", choices=["B200", "B300"], help="GPU to simulate")
    runner.add_argument("--timing", action="store_true",
                        help="also run the approximate timing model (uncalibrated)")
    runner.add_argument("--quiet", action="store_true", help="do not print the summary")
    runner.add_argument("program")
    runner.add_argument("arguments", nargs=argparse.REMAINDER)
    runner.set_defaults(handler=run)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
