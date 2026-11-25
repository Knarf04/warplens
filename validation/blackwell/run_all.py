#!/usr/bin/env python3
"""Run every Blackwell probe unattended and pack the results.

    validation/blackwell/run_all.py                    on the rented B200: build, run, archive
    validation/blackwell/run_all.py --mode sim         dry run: the same probes in the simulator
    validation/blackwell/run_all.py --mode predict     rewrite predicted/ from the simulator

Every probe runs in its own process, under a timeout, with its standard output, standard
error and exit status saved; a probe that hangs, faults or crashes does not stop the rest,
unless it leaves the GPU unable to run anything (below, "GPU health").

status.tsv has one line per run: name, status, seconds, command, and the UTC time the run
started. Statuses: ok; exit 3 (the program printed KERNEL FAILED: the kernel was refused
or faulted, which for some probes is the predicted outcome); launch-failed (the program
printed LAUNCH FAILED: the driver could not load its kernels, e.g. PTX newer than the
driver; nothing was tested); host-error (a host-side CUDA call failed); timeout; `timeout
(process did not exit after SIGKILL)`; exit N or signal N (the program itself died); not
built; `not run (GPU unhealthy after <run>)`. In GPU mode the exit status is 1 when any
required run fails, times out unexpectedly, or lacks build/capture artifacts, or when the GPU
was found unhealthy. Faults and hangs explicitly declared in the manifest remain
observations, not collection failures: that includes a run that may hang by design and timed
out although the simulator predicts that it finishes, which is printed here as `TIMED OUT,
PREDICTED TO FINISH` and counted by compare.py as a mismatch, not by this script as a failed
collection.

The toolkit that built the programs is compared with the one the committed predictions
were made with (predicted/provenance.json). When they differ the run is flagged: a banner
here, env/toolkit.txt and TOOLKIT-MISMATCH.txt in the archive, and a banner from compare.py.

GPU mode
    1. builds the probes (build.sh --native) unless --no-build
    2. records the environment: nvidia-smi, driver and toolkit versions, the device
    3. chooses the binaries: the PTX-carrying ones in bin/ (the files the simulator ran) if
       the driver compiles them, else the machine-code ones in bin-native/ (--binaries)
    4. runs the probes the manifest gives no risk (manifest order): a fault or a hang there
       is a failure of the probe, so these are the runs least likely to disturb the GPU
    5. runs the end-to-end programs (e2e/run_e2e.py) unless --no-e2e
    6. runs the matrix rows that are not probes (run_rows.py: RT-11, NUM-05, RT-10) into
       <out>/rows/. NUM-05 and RT-10 are informational: a failure there is recorded and never
       fails the run. RT-11's capture is required on a GPU of compute capability 10.x: without
       it the exit status is 1 and <out>/RT-11-NOT-CAPTURED.txt says why (--rt11 optional
       to do without; --no-e2e and --only leave the rows out and are not failures). Where it
       is required the file is written before anything runs ("not captured yet") and removed
       when the row ends as ok, so a run that is killed before it gets there leaves it too
    7. runs the probes that may fault by design (risk "fault"), then those that may hang
       (risk "hang"): whatever these do to the GPU, everything above is already on disk
    8. writes <out>.tar.gz

GPU health (GPU mode). A known-good kernel, the canary (mbarrier/script basic from the
chosen binaries, at most CANARY_TIMEOUT seconds), is run once before the probes and again
after every run that did not end as ok, after every run with a risk, after the end-to-end
programs and after the rows. Each check is a line of <out>/health.tsv (after which run, how
the canary ended, seconds, attempts, when, a note), and the new lines of `dmesg` that name
Xid or NVRM go to <out>/xid.log under the run's name (once, that dmesg cannot be read, if
it cannot). A canary that fails is tried again after 5, 10, 15 and 30 seconds, with
`nvidia-smi` saved to <out>/health/ before the first and after the last retry. If it passes,
the event stays in health.tsv and the run goes on. If it does not, nothing more is launched:
every remaining run is recorded as `not run (GPU unhealthy after <run>)`,
<out>/GPU-UNHEALTHY.txt names the run, and the exit status is 1. The breaker is armed only
once the canary has passed in this run: on a machine where it never runs (the development
machine with bin/: sm_100a programs on an RTX 3080 Ti) its failures are recorded and stop
nothing. Never tripped by a real GPU so far; tests/test_probe_health.py trips it with
stand-in programs.

Simulator modes need the `warplens` package and its Python environment:
    PYTHONPATH=$PWD conda run -n warplens python validation/blackwell/run_all.py --mode sim

Options
    --out DIR         results directory (default: results/<mode>-<host>-<time>)
    --only PATTERN    run only probes whose name matches this shell pattern; repeatable
                      (GPU mode: still without risk first, then fault, then hang)
    --resume          gpu mode, with the --out of an earlier run: keep every run that has a
                      status there other than `not run`, and run the rest. For the runs a
                      tripped breaker left out, after the GPU was reset. The run after which
                      the GPU was unhealthy has a status and is not repeated; the end-to-end
                      programs are repeated unless all of them have one; the rows are left
                      alone once RT-11 has a line in rows/status.tsv (run_rows.py --only
                      repeats one by hand)
    --no-build        do not run build.sh first
    --no-e2e          skip the end-to-end programs
    --rt11 WHEN       auto (default), required or optional: whether a missing RT-11 capture fails
                      the run (run_rows.py)
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
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from provenance import archive_probes
from manifest import E2E, PROBES, Probe  # noqa: E402
from probe_process import forward_interrupts, run_process  # noqa: E402

SIM_TIMEOUT = 600  # seconds for one probe in the simulator
ROWS_LIMIT = 4200  # seconds for run_rows.py as a whole; its steps' own limits add up to 3900 (run_rows.py, LIMITS)
PREDICTED = HERE / "predicted"  # --predicted replaces it
STATUS_HEADER = "name\tstatus\tseconds\tcommand\tstarted_utc\n"

CANARY = Path("mbarrier") / "script"       # with the argument "basic": one CTA, mbarrier.init/arrive/test_wait
CANARY_TIMEOUT = 30                        # seconds; the program takes well under one on the development machine
CANARY_BACKOFF = (5, 10, 15, 30)           # seconds waited before each retry of a failed canary: 60 in all
KERNEL_LOG_LINE = re.compile(r"Xid|NVRM")
RT11_NOT_YET = "MATRIX ROW RT-11 IS NOT CAPTURED YET"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    if (stdout.startswith(b"LAUNCH FAILED:") or b"\nLAUNCH FAILED:" in stdout):
        return "launch-failed"
    if status == "exit 4" and b"LAUNCH FAILED:" in stdout:
        # The program had begun a line when its first launch failed (e2e/ampere_replay vote
        # prints the inputs first), so the words are not at the start of one. Status 4 is
        # only ever that (common.h).
        return "launch-failed"
    if (stdout.startswith(b"HOST ERROR:") or b"\nHOST ERROR:" in stdout):
        return "host-error"
    return status


def expected_outcome(probe: Probe, status: str, stdout: bytes) -> bool:
    """Only the manifest may allow a hardware fault or timeout as an observation. Whether the
    observation is what the simulator predicts is another question: predicted_to_finish()."""
    if status == "ok":
        return b"KERNEL FAILED" not in stdout
    if status == "timeout":
        return probe.risk == "hang"
    return status == "exit 3" and b"KERNEL FAILED" in stdout and bool(probe.risk)


def predicted_to_finish(probe: Probe) -> bool:
    """Does the committed prediction say this run's kernels all return (`status: ok` in
    predicted/<name>.sim.txt)? False without a prediction."""
    note = PREDICTED / (probe.name + ".sim.txt")
    return note.is_file() and note.read_text(errors="replace").splitlines()[:1] == ["status: ok"]


def execution_phases(probes: list, mode: str, e2e: bool, rows: bool) -> list[tuple[str, list]]:
    """What runs, in order: ("probes", [...]), ("e2e", []), ("rows", []).

    The simulator modes keep the manifest's order with the end-to-end programs last, as they
    always were (the predictions do not depend on the order, and nothing there can be hurt).
    On a GPU the runs the manifest gives no risk go first, then the end-to-end programs and
    the rows without a probe, and only then the runs that may fault and the runs that may
    hang: what is most likely to leave the GPU unusable is last, when the measurements that
    cannot be repeated without it (RT-11's capture among them) are on disk."""
    if mode != "gpu":
        return [("probes", list(probes))] + ([("e2e", [])] if e2e else [])
    by_risk = {risk: [p for p in probes if p.risk == risk] for risk in ("", "fault", "hang")}
    other = [p for p in probes if p.risk not in by_risk]          # none now; never dropped
    return ([("probes", by_risk[""])] + ([("e2e", [])] if e2e else []) + ([("rows", [])] if rows else [])
            + [("probes", by_risk["fault"]), ("probes", by_risk["hang"] + other)])


def compute_capability() -> str:
    """Of the first GPU nvidia-smi lists ("" if it cannot say)."""
    status, _, stdout, _ = run_process(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], 60)
    lines = stdout.decode(errors="replace").strip().splitlines()
    return lines[0].strip() if status == "ok" and lines else ""


class Health:
    """The canary, health.tsv, xid.log and the circuit breaker (the module's docstring)."""

    def __init__(self, out: Path, binaries: Path, scale: float):
        self.out, self.binaries, self.scale = out, binaries, scale
        self.armed = False            # has the canary passed in this run?
        self.unhealthy_after = ""     # the run after which the GPU did not come back
        self.recovered: list[str] = []
        self.lines: list[str] = []
        self.seen: set[str] | None = None   # kernel log lines already written; None before the first reading
        self.log_unreadable = False
        (out / "GPU-UNHEALTHY.txt").unlink(missing_ok=True)
        with (out / "xid.log").open("a") as log:
            log.write(f"# run started {utc_now()}: lines of `dmesg` matching {KERNEL_LOG_LINE.pattern}, under the "
                      "run after which each was first seen\n")
        self.kernel_log("(before the run)")
        self.check("(start)")

    def canary(self) -> tuple[str, float]:
        program = self.binaries / CANARY
        if not program.exists():
            return "not built", 0.0
        status, seconds, stdout, _ = run_process([str(program), "basic"], CANARY_TIMEOUT * self.scale)
        return classify(status, stdout), seconds

    def kernel_log(self, after: str) -> None:
        """Append to xid.log what the kernel said about the GPU since the last look. Never fails."""
        if self.log_unreadable:
            return
        try:
            status, _, stdout, stderr = run_process(["dmesg"], 20)
            if status != "ok":             # kernel.dmesg_restrict; the journal may still be readable
                why = (stderr or stdout).decode(errors="replace").strip()[:200]
                status, _, stdout, _ = run_process(["journalctl", "-k", "-b", "-q", "--no-pager"], 20)
            with (self.out / "xid.log").open("a") as log:
                if status != "ok":
                    self.log_unreadable = True
                    log.write(f"# dmesg is unreadable here ({why or 'no reason given'}; `journalctl -k`: "
                              f"{status}): no kernel log line of this run is recorded\n")
                    return
                lines = [line for line in stdout.decode(errors="replace").splitlines() if KERNEL_LOG_LINE.search(line)]
                new = [line for line in lines if self.seen is None or line not in self.seen]
                self.seen = (self.seen or set()) | set(lines)
                log.writelines(f"{after}\t{line}\n" for line in new)
        except Exception as error:          # the kernel log is an extra; the run is not
            self.log_unreadable = True
            try:
                with (self.out / "xid.log").open("a") as log:
                    log.write(f"# reading the kernel log failed ({error!r}); nothing more is recorded\n")
            except OSError:
                pass

    def snapshot(self, after: str, when: str) -> None:
        directory = self.out / "health"
        directory.mkdir(parents=True, exist_ok=True)
        status, _, stdout, stderr = run_process(["nvidia-smi"], 30)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", after).strip("_")
        (directory / f"nvidia-smi.{name}.{when}.txt").write_bytes(
            stdout + (b"" if status == "ok" else f"[{status}]\n".encode() + stderr))

    def record(self, after: str, status: str, seconds: float, attempts: int, started: str, note: str) -> None:
        self.lines.append(f"{after}\t{status}\t{seconds:.1f}\t{attempts}\t{started}\t{note}")
        (self.out / "health.tsv").write_text("after\tcanary\tseconds\tattempts\tstarted_utc\tnote\n"
                                             + "\n".join(self.lines) + "\n")

    def check(self, after: str) -> bool:
        """Run the canary after `after`. False when the GPU did not come back: the caller stops."""
        if self.unhealthy_after:
            return False
        started, begun = utc_now(), time.monotonic()
        status, _ = self.canary()
        if after != "(start)":
            self.kernel_log(after)
        if status == "ok":
            self.armed = True
            self.record(after, status, time.monotonic() - begun, 1, started, "")
            return True
        if not self.armed:
            self.record(after, status, time.monotonic() - begun, 1, started,
                        "the canary has not passed in this run, so the breaker is not armed; nothing is concluded")
            return True
        first, attempts = status, 1
        print(f"  canary after {after}: {first}; retrying for up to {sum(CANARY_BACKOFF) * self.scale:.0f} s",
              flush=True)
        self.snapshot(after, "first-failure")
        for wait in CANARY_BACKOFF:
            time.sleep(wait * self.scale)
            status, _ = self.canary()
            attempts += 1
            if status == "ok":
                break
        self.kernel_log(after)
        seconds = time.monotonic() - begun
        if status == "ok":
            self.recovered.append(after)
            self.record(after, status, seconds, attempts, started,
                        f"FAILED THEN RECOVERED: first {first}; passed at attempt {attempts}")
            print(f"  canary after {after}: passed at attempt {attempts} ({seconds:.0f} s); going on", flush=True)
            return True
        self.snapshot(after, "unhealthy")
        self.unhealthy_after = after
        self.record(after, status, seconds, attempts, started, f"GPU UNHEALTHY: first {first}; no attempt passed")
        (self.out / "GPU-UNHEALTHY.txt").write_text(
            f"THE GPU WAS UNHEALTHY AFTER {after}\n\n"
            f"The canary ({CANARY} basic, a kernel that had run on this GPU earlier in this run) ended as: {first}\n"
            f"and did not pass in {attempts - 1} more attempt(s) over {seconds:.0f} s (last: {status}). Nothing was "
            "launched after that:\nevery remaining run is `not run (GPU unhealthy after ...)` in status.tsv, and "
            "the end-to-end\nprograms and the rows without a probe were left out if they had not run yet.\n"
            "See health.tsv, xid.log and health/nvidia-smi.*.txt here. After the GPU is back (a reset, or a\n"
            "new instance), `run_all.py --no-build --resume --out <this directory>` runs what was left out\n"
            f"and does not repeat {after}.\n")
        return False

    def banner(self) -> str:
        return (self.out / "GPU-UNHEALTHY.txt").read_text() if self.unhealthy_after else ""


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rt11", choices=["auto", "required", "optional"], default="auto")
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

    if args.resume and args.mode != "gpu":
        parser.error("--resume goes with --mode gpu")
    probes = [p for p in PROBES if not args.only or any(fnmatch.fnmatch(p.name, pattern) for pattern in args.only)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else HERE / "results" / f"{args.mode}-{socket.gethostname()}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    infrastructure_failures = 0
    gpu = args.mode == "gpu"
    # --resume: what an earlier run into this directory recorded, by name.
    previous = {}
    if args.resume and (out / "status.tsv").is_file():
        previous = {line.split("\t")[0]: line for line in (out / "status.tsv").read_text().splitlines()[1:]
                    if len(line.split("\t")) >= 2 and not line.split("\t")[1].startswith("not run")}

    if not args.no_build:
        command = [str(HERE / "build.sh")] + (["--native"] if gpu else [])
        status, seconds, stdout, stderr = run_process(command, 3600)
        (out / "build.log").write_bytes(stdout + stderr)
        print(f"build: {status} ({seconds:.0f} s)", flush=True)
        if status != "ok":
            infrastructure_failures += 1
            # Keep going: whatever did build still runs, and the log says what did not.
            print((stdout + stderr).decode(errors="replace")[-3000:], flush=True)

    toolkit = health = None
    rt11_marker = out / "RT-11-NOT-CAPTURED.txt"
    with_e2e = not args.no_e2e and not args.only
    with_rows = gpu and not args.only
    # --resume: the rows are left as they are once RT-11 has a line (run_rows.py --only repeats one).
    recorded_rows = out / "rows" / "status.tsv"
    rows_kept = args.resume and recorded_rows.is_file() and any(
        line.startswith("e2e/cute_dsl_tma\t") for line in recorded_rows.read_text().splitlines())
    if gpu:
        binaries = choose_binaries(args.binaries, out)
        record_environment(out, binaries)
        toolkit = record_toolkit(out, binaries)
        if with_rows and not args.no_e2e and not rows_kept and not (args.resume and rt11_marker.is_file()) and (
                args.rt11 == "required" or (args.rt11 == "auto" and compute_capability().startswith("10."))):
            # Said first, unsaid by run_rows.py when the row ends as ok: a run that is killed
            # (the stage's limit, the instance) before it reaches the row must still leave the
            # file env/remote/run.sh and compare.py look for.
            rt11_marker.write_text(
                f"{RT11_NOT_YET}: the run has not reached the row (written at {utc_now()}, before the first "
                "probe; run_rows.py removes this file when e2e/cute_dsl_tma ends as ok).\nIf this is what a "
                "finished or killed run left: the capture is missing and is required on this GPU. "
                "env/remote/RUNBOOK.md, section 3, \"The rows without a probe\", has the commands to run it "
                "by hand before the instance is released.\n")
        health = Health(out, binaries, args.timeout_scale)
    else:
        binaries = HERE / "bin"
    if args.mode == "predict":
        write_provenance(partial=bool(args.only))

    artifact_errors = archive_probes(out, HERE, binaries, probes, mode=args.mode)
    if artifact_errors:
        infrastructure_failures += 1
        (out / "PROVENANCE-ERRORS.txt").write_text("\n".join(artifact_errors) + "\n")
        print("Incomplete probe provenance: " + "; ".join(artifact_errors), flush=True)

    rows = []       # one line of status.tsv per probe, in the order they ran
    table = []      # status.tsv: those, and the lines e2e/run_e2e.py appended, where it ran
    surprises = []  # runs that may hang by design, timed out, and are predicted to finish

    def write_status() -> None:
        # Rewritten after every probe, so a run that is cut short still has it.
        (out / "status.tsv").write_text(STATUS_HEADER + "".join(line + "\n" for line in table))

    def one(probe: Probe):
        begun = utc_now()
        if not gpu and probe.predict == "none":
            return probe, "skipped (no kernel; nothing to simulate)", 0.0, b"", b"", begun
        if not (binaries / probe.program).exists():
            return probe, "not built", 0.0, b"", b"", begun
        status, seconds, stdout, stderr = run_probe(probe, args.mode, binaries, args.timeout_scale)
        return probe, classify(status, stdout), seconds, stdout, stderr, begun

    def not_run(names, why: str) -> None:
        for name, command in names:
            line = f"{name}\tnot run ({why})\t0.0\t{command}\t{utc_now()}"
            table.append(line)
            if name not in E2E:
                rows.append(line)
            print(f"{name:42} not run ({why})", flush=True)
        write_status()

    def run_probes(batch: list) -> None:
        nonlocal infrastructure_failures
        workers = 1 if gpu else max(1, args.jobs)
        fresh = [probe for probe in batch if probe.name not in previous]
        for probe in batch:
            if probe.name in previous:      # --resume: as the earlier run left it, outputs included
                rows.append(previous[probe.name])
                table.append(previous[probe.name])
        write_status()
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            results = pool.map(one, fresh) if not gpu else None
            for index, probe in enumerate(fresh):
                if health is not None and health.unhealthy_after:
                    not_run([(p.name, " ".join([p.program, *p.args])) for p in fresh[index:]],
                            f"GPU unhealthy after {health.unhealthy_after}")
                    infrastructure_failures += 1
                    return
                probe, status, seconds, stdout, stderr, begun = next(results) if results is not None else one(probe)
                if status.startswith(("not built", "not run")):
                    infrastructure_failures += 1
                if gpu and not expected_outcome(probe, status, stdout):
                    infrastructure_failures += 1
                save(out, probe.name, stdout, stderr)
                command = " ".join([probe.program, *probe.args])
                rows.append(f"{probe.name}\t{status}\t{seconds:.1f}\t{command}\t{begun}")
                table.append(rows[-1])
                print(f"{probe.name:42} {status:10} {seconds:6.1f} s", flush=True)
                if gpu and status.startswith("timeout") and probe.risk == "hang" and predicted_to_finish(probe):
                    surprises.append(probe.name)
                    print(f"  TIMED OUT, PREDICTED TO FINISH: {probe.name} may hang by design, but the simulator's "
                          "prediction is a kernel that returns. compare.py counts this as a mismatch", flush=True)
                if args.mode == "predict" and not status.startswith(("skipped", "not built")):
                    write_prediction(probe, status, stdout, stderr)
                write_status()
                if health is not None and (status != "ok" or probe.risk) and not status.startswith("not built"):
                    health.check(probe.name)

    def run_e2e() -> None:
        nonlocal infrastructure_failures
        if previous and all(name in previous for name in E2E):
            table.extend(previous[name] for name in E2E)
            write_status()
            return
        if health is not None and health.unhealthy_after:
            not_run([(name, "-") for name in E2E], f"GPU unhealthy after {health.unhealthy_after}")
            infrastructure_failures += 1
            return
        command = [sys.executable, str(HERE / "e2e" / "run_e2e.py"), "--mode", "gpu" if gpu else "sim",
                   "--out", str(out), "--timeout-scale", str(args.timeout_scale)]
        if args.book:
            command += ["--book", args.book]
        if args.mode == "predict":
            command += ["--predicted", str(PREDICTED)]
        write_status()
        status, _, stdout, stderr = run_process(command, 7 * 3600 * args.timeout_scale)
        (out / "e2e-runner.log").write_bytes(stdout + stderr)
        print(f"end-to-end runner: {status}", flush=True)
        infrastructure_failures += status != "ok"
        # run_e2e.py appended its lines to status.tsv; keep them through the rewrites to come.
        if (out / "status.tsv").is_file():
            table.extend((out / "status.tsv").read_text().splitlines()[1 + len(table):])
        if health is not None:
            health.check("(end-to-end programs)")

    def run_rows() -> None:
        nonlocal infrastructure_failures
        # The rows without a probe (RT-11, NUM-05, RT-10). No prediction, no verdict: what
        # happens to NUM-05 and RT-10 is in rows/status.tsv and does not count against the run.
        # run_rows.py exits non-zero only when RT-11's capture is required (a B200) and is not
        # there, or when it broke before it could say; either way the capture is missing.
        if rows_kept:
            print("rows without a probe (run_rows.py): kept from the earlier run (--resume)", flush=True)
            infrastructure_failures += rt11_marker.is_file()
            return
        if health is not None and health.unhealthy_after:
            print(f"rows without a probe (run_rows.py): not run (GPU unhealthy after {health.unhealthy_after})",
                  flush=True)
            if rt11_marker.is_file():
                rt11_marker.write_text(
                    f"MATRIX ROW RT-11 WAS NOT CAPTURED: the GPU was unhealthy after {health.unhealthy_after} "
                    "and run_rows.py was not run. See GPU-UNHEALTHY.txt; env/remote/RUNBOOK.md, section 3, "
                    "\"The rows without a probe\", has the commands to run it by hand once the GPU is back.\n")
            return
        command = [sys.executable, str(HERE / "run_rows.py"), "--out", str(out), "--rt11", args.rt11,
                   "--timeout-scale", str(args.timeout_scale)] + (["--no-e2e"] if args.no_e2e else [])
        status, _, stdout, stderr = run_process(command, ROWS_LIMIT * args.timeout_scale)
        (out / "rows-runner.log").write_bytes(stdout + stderr)
        print(stdout.decode(errors="replace"), end="")
        print(f"rows without a probe (run_rows.py): {status}", flush=True)
        if status != "ok":
            infrastructure_failures += 1
            if not rt11_marker.is_file() or rt11_marker.read_text().startswith(RT11_NOT_YET):
                rt11_marker.write_text(
                    f"run_rows.py ended with: {status}, before it could say whether matrix row RT-11 was "
                    "captured. See rows-runner.log and rows/status.tsv; env/remote/RUNBOOK.md, section 3, "
                    "\"The rows without a probe\", has the commands to run it by hand.\n")
        if health is not None:
            health.check("(rows without a probe)")

    for kind, batch in execution_phases(probes, args.mode, with_e2e, with_rows):
        {"probes": lambda: run_probes(batch), "e2e": run_e2e, "rows": run_rows}[kind]()
    if health is not None and health.unhealthy_after:
        infrastructure_failures += 1

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
        if surprises:
            print(f"{len(surprises)} run(s) TIMED OUT, PREDICTED TO FINISH (may hang by design, so not a failed "
                  f"collection; compare.py counts each as a mismatch): {', '.join(surprises)}")
        if health.recovered:
            print(f"the canary failed and then passed after: {', '.join(health.recovered)} (health.tsv)")
        print(f"compare with: python {HERE / 'compare.py'} {archive}")
        infrastructure_failures += any(broken.values())
        if health.unhealthy_after:
            bar = "!" * 100
            print(f"{bar}\n{health.banner()}{bar}")
        if (out / "RT-11-NOT-CAPTURED.txt").is_file():      # run_rows.py's banner, again as the last thing said
            bar = "!" * 100
            print(f"{bar}\n{(out / 'RT-11-NOT-CAPTURED.txt').read_text()}{bar}")
    return 1 if infrastructure_failures else 0


if __name__ == "__main__":
    with forward_interrupts():
        sys.exit(main())
