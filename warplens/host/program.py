"""Run a whole program in the simulator from Python, and keep what each launch produced.

`warplens run program` does this from the shell: it starts the program with its CUDA runtime
replaced, executes the runtime calls it makes in the simulator, prints a summary and exits.
`run_program` does the same from Python and returns the launch statistics, findings and
timing reports instead of printing them:

    import sys, warplens
    done = warplens.run_program([sys.executable, "examples/07_tirx/run.py"], timing=True)
    for launch in done.launches:
        print(launch.kernel, launch.stats.ctas, launch.stats.timing.time_ns)

The program runs in a child process, so this process never loads a CUDA runtime, and a crash
in the program's host code cannot take the caller (a notebook, say) down. Linux only; the
shim libraries are compiled on first use, which needs a C compiler and the CUDA headers.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import WarplensError
from ..frontend.ir import Module
from .launch import LaunchStats

__all__ = ["ProgramLaunch", "ProgramRun", "run_program"]


@dataclass
class ProgramLaunch:
    """One kernel launch a program made, as the simulator saw it."""
    kernel: str  #: the entry point's name in the PTX
    module: Module  #: the parsed PTX the kernel came from
    grid: tuple
    block: tuple
    cluster: tuple | None  #: the cluster shape in CTAs, or None for a launch without clusters
    smem: int  #: dynamic shared memory requested, in bytes
    stats: LaunchStats  #: what `Device.launch` returned

    @property
    def instructions(self) -> list:
        """The kernel's own PTX instructions, in program order. The device functions it calls
        are not included."""
        return self.module.functions[self.kernel].instrs


@dataclass
class ProgramRun:
    """What `run_program` returns: how the program ended, and every launch it made."""
    returncode: int
    stdout: str
    stderr: str
    seconds: float  #: wall-clock time of the run on this machine, not simulated time
    launches: list[ProgramLaunch] = field(default_factory=list)

    @property
    def launch(self) -> ProgramLaunch:
        """The only launch of a program that made exactly one; RuntimeError otherwise."""
        if len(self.launches) != 1:
            raise RuntimeError(f"the program made {len(self.launches)} launches, not one\n"
                               f"{self.stdout[-2000:]}{self.stderr[-2000:]}")
        return self.launches[0]


def run_program(command, *, model: str = "B200", timing=False, active_sms: int | None = None,
                trace=None, env: dict | None = None) -> ProgramRun:
    """Run `command` (a list, as for `subprocess.run`) with its GPU work done by the simulator.

    The program is a compiled CUDA executable that carries PTX, or a Python interpreter
    running a TIRx or CuTe DSL script: whatever `warplens run` accepts, except a Triton
    script (`warplens.triton_driver.install` is the way in for those).

    `model` is the GPU to simulate. `timing` (True, or a `GpuConfig`), `active_sms` and
    `trace` are passed to every `Device.launch` the program causes. `env` is added to the
    environment the program inherits from this process.

    The program's output is captured, not shown. A program that fails is reported through
    `returncode` and `stderr`, not by an exception; `WarplensError` is raised only when the
    shim libraries cannot be built.
    """
    from .. import cli  # here, not at the top: cli is the command and imports nothing from host
    from .shim_server import ShimServer

    try:
        shims = cli.shim_libraries()
    except SystemExit as stop:
        raise WarplensError(str(stop.code)) from None
    ours, theirs = socket.socketpair()
    child_env = dict(os.environ, **(env or {}), WARPLENS_SHIM_FD=str(theirs.fileno()), WARPLENS_MODEL=model)
    with tempfile.TemporaryDirectory(prefix="warplens_program_") as scratch:
        kept = cli.frontend_environment(child_env, model, shims, Path(scratch))
        server = ShimServer(model, kept_ptx=kept)
        launches: list[ProgramLaunch] = []
        device_launch = server.device.launch

        def launch(module, entry, **options):  # every launch the program makes passes through here
            if timing:
                options.update(timing=timing, active_sms=active_sms)
            stats = device_launch(module, entry, trace=trace, **options)
            launches.append(ProgramLaunch(entry, module, options["grid"], options["block"],
                                          options.get("cluster"), options.get("smem", 0), stats))
            return stats

        server.device.launch = launch
        child_env["LD_LIBRARY_PATH"] = f"{shims}:{child_env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        # Preload as well: a program built with an RPATH would otherwise find the real runtime first.
        preload = [str(p) for p in sorted(shims.glob("libcudart.so.*")) + [shims / "libcuda.so.1"]]
        child_env["LD_PRELOAD"] = " ".join(preload + child_env.get("LD_PRELOAD", "").split())
        start = time.perf_counter()
        # Output goes to files, not pipes: this process is busy serving the child and would
        # not drain a pipe.
        with open(Path(scratch, "stdout"), "w+") as out, open(Path(scratch, "stderr"), "w+") as err:
            try:
                child = subprocess.Popen([str(c) for c in command], env=child_env, pass_fds=[theirs.fileno()],
                                         stdout=out, stderr=err, stdin=subprocess.DEVNULL)
            except BaseException:
                ours.close(), theirs.close()
                raise
            try:
                theirs.close()
                server.serve(ours)  # returns when the child closes its end
                code = child.wait()
            finally:
                ours.close()
                if child.poll() is None:
                    child.kill()
                    child.wait()
            out.seek(0), err.seek(0)
            return ProgramRun(code, out.read(), err.read(), time.perf_counter() - start, launches)
