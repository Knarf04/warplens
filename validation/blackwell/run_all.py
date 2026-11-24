#!/usr/bin/env python3
"""Run every Blackwell probe unattended and pack the results.

    validation/blackwell/run_all.py                    on the rented B200: build, run, archive
    validation/blackwell/run_all.py --mode sim         dry run: the same probes in the simulator
    validation/blackwell/run_all.py --mode predict     rewrite predicted/ from the simulator

Every probe runs in its own process, under a timeout, with its standard output, standard
error and exit status saved; a probe that hangs, faults or crashes does not stop the rest.

Statuses in status.tsv: ok; exit 3 (the program printed KERNEL FAILED: the kernel was refused
or faulted, which for some probes is the predicted outcome); launch-failed (the program
printed LAUNCH FAILED: the driver could not load its kernels, e.g. PTX newer than the
driver; nothing was tested); host-error (a host-side CUDA call failed); timeout; exit N or
signal N (the program itself died); not built. In GPU mode the exit status is 1 when any
probe is launch-failed, host-error or not built, and the summary says so in capitals.

The toolkit that built the programs is compared with the one the committed predictions
were made with (predicted/provenance.json). When they differ the run is flagged: a banner
here, env/toolkit.txt and TOOLKIT-MISMATCH.txt in the archive, and a banner from compare.py.

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
    --predicted DIR   predict mode: write the predictions there instead of predicted/ (how
                      tests/test_predict_path.py runs the whole path without touching the
                      committed files)
    --binaries WHICH  gpu mode: auto (default), ptx or native; or ampere, for the dry run on an
                      Ampere GPU of the probes `build.sh --ampere` builds (use with --no-build)
    --jobs N          simulator modes: probes to run at once (default: 4)
    --timeout-scale X multiply every timeout (default 1)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from manifest import PROBES, Probe  # noqa: E402

SIM_TIMEOUT = 600  # seconds for one probe in the simulator
PREDICTED = HERE / "predicted"  # --predicted replaces it


def run_process(command, timeout, env=None, cwd=None):
    """Run one command in its own process group. Returns (status, seconds, stdout, stderr);
    status is "ok", "exit N", "signal N" or "timeout"."""
    start = time.monotonic()
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd,
                                 start_new_session=True)
    except OSError as error:
        return f"not run ({error.strerror})", 0.0, b"", str(error).encode()
    try:
        out, err = child.communicate(timeout=timeout)
        code = child.returncode
        status = "ok" if code == 0 else f"exit {code}" if code > 0 else f"signal {-code}"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)  # the program and anything it started
        except ProcessLookupError:
            pass
        out, err = child.communicate()
        status = "timeout"
    return status, time.monotonic() - start, out, err


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


def read_keys(path: Path) -> dict[str, str]:
    """key=value lines of a file; {} when it does not exist."""
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def predictions_toolkit() -> str:
    """The nvcc version whose PTX the simulator ran to make predicted/ ("" if unrecorded)."""
    path = PREDICTED / "provenance.json"
    return json.loads(path.read_text()).get("nvcc_version", "") if path.is_file() else ""


def toolkit_report(binaries: Path, smi_text: str = "", driver_version: str = "") -> dict[str, str]:
    """What built the programs in `binaries`, what the driver is, and whether the committed
    predictions can be held against this run. The caller supplies nvidia-smi's text."""
    built = read_keys(binaries / "toolkit.txt").get("nvcc_version", "") or "unknown"
    predicted = predictions_toolkit() or "unknown"
    cuda = re.search(r"CUDA Version: *([0-9]+\.[0-9]+)", smi_text)
    report = {"binaries": binaries.name, "built_nvcc_version": built, "predictions_nvcc_version": predicted,
              "driver_version": driver_version or "unknown", "driver_cuda": cuda.group(1) if cuda else "unknown"}
    if binaries.name == "bin-ampere":
        comparable, why = "no", ("NOT THE PROGRAMS THE PREDICTIONS WERE MADE FROM: the Ampere dry-run build "
                                 f"(nvcc {built}, another target, -DPROBE_AMPERE)")
    elif "unknown" in (built, predicted):
        comparable, why = "unknown", (f"TOOLKIT NOT RECORDED: programs built by nvcc {built}, predictions made with "
                                      f"nvcc {predicted} (bin/toolkit.txt is written by build.sh, "
                                      "predicted/provenance.json by predict.sh); cannot tell whether the "
                                      "predictions are comparable")
    elif built != predicted:
        comparable, why = "no", (f"BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: the programs were built "
                                 f"by nvcc {built}, the committed predictions are the simulator's run of PTX from "
                                 f"nvcc {predicted}. A difference from a prediction may be the compiler's")
    elif binaries.name == "bin-native":
        comparable, why = "yes", (f"nvcc {built} as for the predictions; machine-code programs (the driver did not "
                                  "take this toolkit's PTX), so ptxas of the toolkit compiled the kernels, not the driver")
    else:
        comparable, why = "yes", f"nvcc {built} as for the predictions; the driver compiled the PTX the simulator ran"
    report["predictions_comparable"] = comparable
    report["reason"] = why
    return report


def record_toolkit(out: Path, binaries: Path) -> dict[str, str]:
    status, _, smi, _ = run_process(["nvidia-smi"], 60)
    _, _, driver, _ = run_process(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 60)
    report = toolkit_report(binaries, smi.decode(errors="replace") if status == "ok" else "",
                            driver.decode(errors="replace").strip().splitlines()[0] if driver.strip() else "")
    (out / "env").mkdir(parents=True, exist_ok=True)
    (out / "env" / "toolkit.txt").write_text("".join(f"{k}={v}\n" for k, v in report.items()))
    if report["predictions_comparable"] != "yes":
        (out / "TOOLKIT-MISMATCH.txt").write_text(report["reason"] + "\n")
        bar = "!" * 100
        print(f"{bar}\n{report['reason']}\n{bar}", flush=True)
    else:
        print(f"toolkit: {report['reason']}", flush=True)
    return report


def classify(status: str, stdout: bytes) -> str:
    """Name the two failures that are not results of a test: the program said it could not
    load its kernels, or a host-side CUDA call failed."""
    if status.startswith("exit") and (stdout.startswith(b"LAUNCH FAILED") or b"\nLAUNCH FAILED" in stdout):
        return "launch-failed"
    if status.startswith("exit") and (stdout.startswith(b"HOST ERROR") or b"\nHOST ERROR" in stdout):
        return "host-error"
    return status


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
        else:
            note += ("NEITHER BUILD RUNS: the canary failed from bin/ and from bin-native/. Every probe that "
                     "launches a kernel will be recorded as launch-failed or host-error\n")
    elif status != "ok":
        note += (stdout + stderr).decode(errors="replace")
        note += "THE CANARY FAILED and there is no bin-native/ to fall back to (build.sh --native)\n"
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
    target = PREDICTED / (probe.name + ".txt")
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


def write_provenance(partial: bool) -> None:
    """predicted/provenance.json: the compiler whose PTX the simulator ran. A partial
    regeneration (--only) with another compiler than the recorded one is refused, because
    the file can name only one."""
    built = read_keys(HERE / "bin" / "toolkit.txt").get("nvcc_version", "")
    if not built or built == "unknown":
        raise SystemExit("predict: bin/toolkit.txt does not say which nvcc built the programs; run build.sh")
    recorded = predictions_toolkit()
    if partial and recorded and recorded != built:
        raise SystemExit(f"predict: the programs were built by nvcc {built} but predicted/provenance.json says the "
                         f"other predictions were made with nvcc {recorded}. Regenerate all of them (no --only)")
    text = json.dumps({"nvcc_version": built,
                       "what": "the nvcc that built validation/blackwell/bin, whose PTX the simulator ran to "
                               "produce these predictions; run_all.py and compare.py flag a GPU run built by "
                               "another version"}, indent=1) + "\n"
    path = PREDICTED / "provenance.json"
    if not path.is_file() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gpu", "sim", "predict"], default="gpu")
    parser.add_argument("--out")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--no-e2e", action="store_true")
    parser.add_argument("--book", default=os.environ.get("WARPLENS_BOOK", ""))
    parser.add_argument("--predicted")
    parser.add_argument("--binaries", choices=["auto", "ptx", "native", "ampere"], default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    args = parser.parse_args()
    global PREDICTED
    if args.predicted:
        if args.mode != "predict":
            parser.error("--predicted goes with --mode predict")
        PREDICTED = Path(args.predicted).resolve()

    probes = [p for p in PROBES if not args.only or any(fnmatch.fnmatch(p.name, pattern) for pattern in args.only)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else HERE / "results" / f"{args.mode}-{socket.gethostname()}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    if not args.no_build:
        command = [str(HERE / "build.sh")] + (["--native"] if args.mode == "gpu" else [])
        status, seconds, stdout, stderr = run_process(command, 3600)
        (out / "build.log").write_bytes(stdout + stderr)
        print(f"build: {status} ({seconds:.0f} s)", flush=True)
        if status != "ok":
            # Keep going: whatever did build still runs, and the log says what did not.
            print((stdout + stderr).decode(errors="replace")[-3000:], flush=True)

    toolkit = None
    if args.mode == "gpu":
        binaries = choose_binaries(args.binaries, out)
        record_environment(out, binaries)
        toolkit = record_toolkit(out, binaries)
    else:
        binaries = HERE / "bin"
    if args.mode == "predict":
        write_provenance(partial=bool(args.only))

    rows = []

    def one(probe: Probe):
        if args.mode != "gpu" and probe.predict == "none":
            return probe, "skipped (no kernel; nothing to simulate)", 0.0, b"", b""
        if not (binaries / probe.program).exists():
            return probe, "not built", 0.0, b"", b""
        status, seconds, stdout, stderr = run_probe(probe, args.mode, binaries, args.timeout_scale)
        return probe, classify(status, stdout), seconds, stdout, stderr

    workers = 1 if args.mode == "gpu" else max(1, args.jobs)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for probe, status, seconds, stdout, stderr in pool.map(one, probes):
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
                   "--out", str(out)]
        if args.book:
            command += ["--book", args.book]
        if args.mode == "predict":
            command += ["--predicted", str(PREDICTED)]
        subprocess.run(command)

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
        broken = {key: counts.get(key, 0) for key in ("launch-failed", "host-error", "not")}
        if any(broken.values()):
            bar = "!" * 100
            print(f"{bar}\n{broken['launch-failed']} PROBE(S) FAILED TO LAUNCH (the driver could not load their "
                  f"kernels), {broken['host-error']} HAD A HOST ERROR, {broken['not']} WERE NOT BUILT OR NOT RUN.\n"
                  "These tested nothing. See env/binaries.txt and the .stdout files in the results, and "
                  f"'When probes do not launch' in validation/blackwell/README.md.\n{bar}")
        if toolkit and toolkit["predictions_comparable"] != "yes":
            print(toolkit["reason"])
        print(f"compare with: python {HERE / 'compare.py'} {archive}")
        return 1 if any(broken.values()) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
