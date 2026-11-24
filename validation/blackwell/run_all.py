#!/usr/bin/env python3
"""Run every Blackwell probe unattended and pack the results.

    validation/blackwell/run_all.py                    on the rented B200: build, run, archive
    validation/blackwell/run_all.py --mode sim         dry run: the same probes in the simulator
    validation/blackwell/run_all.py --mode predict     rewrite predicted/ from the simulator

Every probe runs in its own process, under a timeout, with its standard output, standard
error and exit status saved; a probe that hangs, faults or crashes does not stop the rest.

GPU mode
    1. builds the probes (build.sh --native) unless --no-build
    2. records the environment: nvidia-smi, driver and toolkit versions, the device
    3. chooses the binaries: the PTX-carrying ones in bin/ (the files the simulator ran) if
       the driver compiles them, else the machine-code ones in bin-native/ (--binaries)
    4. runs the probes in manifest order
    5. runs the end-to-end programs (e2e/run_e2e.py) unless --no-e2e
    6. writes <out>.tar.gz

Simulator modes need the `warplens` package and its Python environment:
    PYTHONPATH=$PWD conda run -n warplens python validation/blackwell/run_all.py --mode sim

Options
    --out DIR         results directory (default: results/<mode>-<host>-<time>)
    --only PATTERN    run only probes whose name matches this shell pattern; repeatable
    --no-build        do not run build.sh first
    --no-e2e          skip the end-to-end programs
    --book PATH       clone of the MLSys book, for e2e (see e2e/run_e2e.py)
    --binaries WHICH  gpu mode: auto (default), ptx or native; or ampere, for the dry run on an
                      Ampere GPU of the probes `build.sh --ampere` builds (use with --no-build)
    --jobs N          simulator modes: probes to run at once (default: 4)
    --timeout-scale X multiply every timeout (default 1)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import os
import socket
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from manifest import PROBES, Probe  # noqa: E402
from probe_process import forward_interrupts, run_process  # noqa: E402

SIM_TIMEOUT = 600  # seconds for one probe in the simulator


def simulator_command(binary: Path, args) -> list[str]:
    return [sys.executable, "-m", "warplens.cli", "run", str(binary), *args]


def simulator_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["CUDA_VISIBLE_DEVICES"] = ""  # nothing may reach a real GPU by accident
    return env


def save(out: Path, name: str, stdout: bytes, stderr: bytes) -> None:
    path = out / (name + ".stdout")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stdout)
    (out / (name + ".stderr")).write_bytes(stderr)


def record_environment(out: Path, binaries: Path) -> None:
    """Everything needed later to know what machine this was. Failures are recorded, not fatal."""
    env_dir = out / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    nvcc = os.environ.get("NVCC", "nvcc")
    commands = {
        "nvidia-smi.txt": ["nvidia-smi"],
        "nvidia-smi-q.txt": ["nvidia-smi", "-q"],
        "gpu.csv": ["nvidia-smi", "--format=csv",
                    "--query-gpu=index,name,driver_version,memory.total,compute_cap,pci.device_id,vbios_version"],
        "nvcc.txt": [nvcc, "--version"],
        "driver.txt": ["cat", "/proc/driver/nvidia/version"],
        "uname.txt": ["uname", "-a"],
        "os-release.txt": ["cat", "/etc/os-release"],
        "cpu.txt": ["lscpu"],
        "ldd.txt": ["ldd", str(binaries / "e2e" / "device_attributes")],
        "git.txt": ["git", "-C", str(REPO), "log", "-1", "--format=%H %cI %s"],
        "git-status.txt": ["git", "-C", str(REPO), "status", "--short"],
        "python.txt": [sys.executable, "--version"],
        "pip.txt": [sys.executable, "-m", "pip", "freeze"],
        "date.txt": ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
    }
    for filename, command in commands.items():
        status, _, stdout, stderr = run_process(command, 60)
        (env_dir / filename).write_bytes(stdout + (b"" if status == "ok" else
                                                   f"[{status}]\n".encode() + stderr))


def choose_binaries(which: str, out: Path) -> Path:
    """bin/ carries PTX, which the driver must be new enough to compile. If the first probe
    (which launches nothing risky) fails there and works from bin-native/, use that."""
    ptx, native = HERE / "bin", HERE / "bin-native"
    if which == "ptx":
        return ptx
    if which == "native":
        return native
    if which == "ampere":  # the local dry run: build.sh --ampere, on a GPU without TMA or tcgen05
        return HERE / "bin-ampere"
    canary = Path("mbarrier") / "script"
    status, _, stdout, stderr = run_process([str(ptx / canary), "basic"], 120)
    note = f"canary bin/{canary} basic: {status}\n"
    chosen = ptx
    if status != "ok" and (native / canary).exists():
        native_status, _, _, native_err = run_process([str(native / canary), "basic"], 120)
        note += (stdout + stderr).decode(errors="replace")
        note += f"canary bin-native/{canary} basic: {native_status}\n" + native_err.decode(errors="replace")
        if native_status == "ok":
            chosen = native
    note += f"using {chosen.name}/\n"
    (out / "env").mkdir(parents=True, exist_ok=True)
    (out / "env" / "binaries.txt").write_text(note)
    print(note, end="", flush=True)
    return chosen


def run_probe(probe: Probe, mode: str, binaries: Path, scale: float):
    binary = binaries / probe.program
    if mode == "gpu":
        return run_process([str(binary), *probe.args], probe.timeout * scale)
    return run_process(simulator_command(binary, probe.args), SIM_TIMEOUT * scale, env=simulator_env())


def numerics_prediction(probe: Probe) -> tuple[str, bytes, bytes]:
    status, _, stdout, stderr = run_process(
        [sys.executable, str(HERE / "formats" / "predict_small_formats.py"), *probe.args], SIM_TIMEOUT,
        env=simulator_env())
    return status, stdout, stderr


def write_prediction(probe: Probe, status: str, stdout: bytes, stderr: bytes) -> None:
    """predicted/<name>.txt is what the program printed; predicted/<name>.sim.txt is how the
    simulator's run ended and what the simulator said (findings, or why it refused)."""
    target = HERE / "predicted" / (probe.name + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if probe.predict == "none":
        return
    if probe.predict == "numerics":
        model_status, model_out, model_err = numerics_prediction(probe)
        if model_status != "ok":
            raise SystemExit(f"{probe.name}: predict_small_formats.py failed: {model_err.decode()}")
        target.write_bytes(model_out)
        note = ("prediction: written by formats/predict_small_formats.py from warplens.numerics\n"
                f"the probe itself under `warplens run`: {status}\n")
    else:
        target.write_bytes(stdout)
        note = f"status: {status}\n"
    target.with_suffix(".sim.txt").write_text(note + stderr.decode(errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gpu", "sim", "predict"], default="gpu")
    parser.add_argument("--out")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--no-e2e", action="store_true")
    parser.add_argument("--book", default=os.environ.get("WARPLENS_BOOK", ""))
    parser.add_argument("--binaries", choices=["auto", "ptx", "native", "ampere"], default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()

    probes = [p for p in PROBES if not args.only or any(fnmatch.fnmatch(p.name, pattern) for pattern in args.only)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else HERE / "results" / f"{args.mode}-{socket.gethostname()}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    infrastructure_failures = 0

    if not args.no_build:
        command = [str(HERE / "build.sh")] + (["--native"] if args.mode == "gpu" else [])
        status, seconds, stdout, stderr = run_process(command, 3600)
        (out / "build.log").write_bytes(stdout + stderr)
        print(f"build: {status} ({seconds:.0f} s)", flush=True)
        if status != "ok":
            infrastructure_failures += 1
            # Keep going: whatever did build still runs, and the log says what did not.
            print((stdout + stderr).decode(errors="replace")[-3000:], flush=True)

    if args.mode == "gpu":
        binaries = choose_binaries(args.binaries, out)
        record_environment(out, binaries)
    else:
        binaries = HERE / "bin"

    rows = []

    def one(probe: Probe):
        if args.mode != "gpu" and probe.predict == "none":
            return probe, "skipped (no kernel; nothing to simulate)", 0.0, b"", b""
        if not (binaries / probe.program).exists():
            return probe, "not built", 0.0, b"", b""
        return (probe, *run_probe(probe, args.mode, binaries, args.timeout_scale))

    workers = 1 if args.mode == "gpu" else max(1, args.jobs)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for probe, status, seconds, stdout, stderr in pool.map(one, probes):
            if status.startswith(("not built", "not run")):
                infrastructure_failures += 1
            save(out, probe.name, stdout, stderr)
            command = " ".join([probe.program, *probe.args])
            rows.append(f"{probe.name}\t{status}\t{seconds:.1f}\t{command}")
            print(f"{probe.name:42} {status:10} {seconds:6.1f} s", flush=True)
            if args.mode == "predict" and not status.startswith(("skipped", "not built")):
                write_prediction(probe, status, stdout, stderr)
            # status.tsv is rewritten after every probe, so a run that is cut short still has it.
            (out / "status.tsv").write_text("name\tstatus\tseconds\tcommand\n" + "\n".join(rows) + "\n")

    if not args.no_e2e and not args.only:
        command = [sys.executable, str(HERE / "e2e" / "run_e2e.py"), "--mode", "gpu" if args.mode == "gpu" else "sim",
                   "--out", str(out), "--timeout-scale", str(args.timeout_scale)]
        if args.book:
            command += ["--book", args.book]
        if args.mode == "predict":
            command += ["--predicted", str(HERE / "predicted")]
        status, _, stdout, stderr = run_process(command, 7 * 3600 * args.timeout_scale)
        (out / "e2e-runner.log").write_bytes(stdout + stderr)
        print(f"end-to-end runner: {status}", flush=True)
        infrastructure_failures += status != "ok"

    (out / "manifest.py").write_bytes((HERE / "manifest.py").read_bytes())
    (out / "elapsed.txt").write_text(f"{time.monotonic() - started:.0f} seconds\n")
    archive = out.with_suffix(".tar.gz") if out.suffix else Path(str(out) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    counts: dict[str, int] = {}
    for row in rows:
        key = row.split("\t")[1].split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"\n{len(rows)} probe(s): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    print(f"results in {out}\narchive    {archive}")
    if args.mode == "gpu":
        print(f"compare with: python {HERE / 'compare.py'} {archive}")
    return 1 if infrastructure_failures else 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
