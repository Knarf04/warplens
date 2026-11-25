#!/usr/bin/env python3
"""Compare a run of the probe suite on a GPU with the simulator's predictions.

    validation/blackwell/compare.py RESULTS [--predicted DIR] [--sim-captures DIR] [--lines N]

RESULTS is the archive run_all.py wrote (or its unpacked directory), or the archive of a
whole rental from env/remote/run.sh (or its unpacked directory): there the probes' results
are out/blackwell/. A status.tsv under predicted-toolkit/ (a simulator's run on the rented
machine) or in a rows/ directory (run_rows.py's) is never taken for the GPU's; if more than
one remains and none is out/blackwell/status.tsv, the candidates are listed and nothing is
compared. Each probe gets one verdict:

    identical       its output is the committed prediction, line for line. When that output
                    is a failed kernel, the detail names the GPU's error and where it was
                    reported, and the simulator's side is a rule it models (it names the rule)
                    or a deadlock (the detail then says that a GPU reports no deadlock)
    unverified      only finite float32 words differ; the ULP distance is reported, but
                    no tolerance or hardware-equivalence claim is inferred
    DIFFERS         it is not; the first differing lines are shown below the table
    informational   there is no prediction to hold it against: the probe has none, or the
                    simulator refuses the kernel and the GPU ran it
    kernel failed   the GPU faulted or rejected the kernel (the program reported it); the
                    detail has the CUDA error and whether it came at the launch or at the
                    synchronize (from the program's standard error; "place not recorded" if
                    that is not in the results)
    GPU faulted, not modelled
                    the same lines on both sides, but the simulator's are "unsupported
                    instruction": it does not run this kernel at all, so the GPU's fault
                    confirms no model. A failed kernel prints only its label on the compared
                    lines, so ANY error of the GPU matches such a prediction
    FAILED TO LAUNCH  the driver could not load the program's kernels (PTX it does not
                    accept, no code for this GPU, ...). Nothing was tested. Never by design,
                    and never counted as "the GPU refused the kernel"
    HOST ERROR      a host-side CUDA call failed before or after the kernel. Nothing was
                    tested, or the GPU was left unusable by an earlier fault
    timed out       killed by the timeout. "(as predicted)": the manifest lets the run hang
                    and the simulator predicts a deadlock. "(by design)": the manifest lets
                    it hang and there is no prediction of a kernel that returns
    TIMED OUT, predicted to finish
                    the manifest lets the run hang, but the simulator's prediction is a
                    kernel that returns: a mismatch, counted, with an entry under Details
    crashed         the program itself died, or could not be started
    not run         no result in the archive
    NOT RUN, GPU unhealthy
                    run_all.py's canary stopped passing and it launched nothing more
                    (GPU-UNHEALTHY.txt in the results, repeated at the top). Not a finding
                    about the probe; the run is incomplete, so the exit status is 1

The end-to-end runs (validation matrix RT-06 to RT-09) are judged on their tensors, not on
their printed lines: every element of each output the GPU produced is compared with the
simulator's, with the tolerance its matrix row states (e2e_compare.py has the table):

    bit-identical     every output of the run equals the simulator's, bit for bit
    within tolerance  every element is inside its row's tolerance; the largest distance is shown
    DIFFERS           an element is outside it, or is a NaN or an infinity, on either side
    NOT COMPARED      there is no simulator result to hold a tensor against. Counted as a failure
    verification failed  the program did not pass its own check, or a capture is missing,
                      incomplete or altered
    NOT HARDWARE      with --require-hardware: the captures are labelled as a simulator's

The simulator's tensors come from --sim-captures DIR (the output directory of
`e2e/run_e2e.py --mode sim`, or its captures/), for each case in which the simulator ran the
PTX the GPU ran, with the same launch and bit-identical initial buffers. For any other case,
and for all of them without --sim-captures, the GPU's capture is replayed in the simulator
(its PTX, its buffers, its launch) and that result is used; this needs no compiler. --no-replay
forbids that, and such a case is then NOT COMPARED. --replay-out DIR keeps the replayed
tensors. --require-hardware fails a run whose captures are labelled as a simulator's.

"(by design)" marks an outcome the manifest allows for that probe. Lines that start with
"info:" carry machine-dependent values and are never compared. The exit status is 1 if any
probe DIFFERS, failed to launch, had a host error, crashed, was not run, timed out where it
is predicted to finish, or failed or timed out without that being by design, or if any
end-to-end run has a verdict other than "bit-identical" or "within tolerance".

If the programs were not built by the toolkit the predictions were made with
(env/toolkit.txt in the results, written by run_all.py), a banner says so before and after
the table: every verdict then compares two different compilations of the probe.
--require-same-toolkit makes that an error (exit status 3, no table).
Exit 2 means only unverified numeric differences remain; human assessment is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import E2E, PROBES  # noqa: E402
from e2e_checks import EXPECTED_CASES, completion_errors
from provenance import validate_probe_archive
from capture import validate_capture_tree  # noqa: E402
from e2e_compare import compare_run  # noqa: E402

HEX_WORD = re.compile(r"^[0-9a-f]{8}$")
E2E_KEY = re.compile(r"Relative error|Execution is|max abs error|matches NumPy|FAILED|SKIPPED|rror")
SECONDS = re.compile(r", \d+\.\d+ s\)")
# "matches NumPy (max error 0.0156)": the number is the distance to NumPy's result on the
# machine that ran the script, and NumPy's float32 matrix product is its BLAS's, whose order
# of summation depends on the CPU and the thread count. Rounded to float16 the reference
# moves by one unit in the last place on some elements, so the same kernel output is 0.0078
# from it on one machine and 0.0156 on another. The claim is "matches NumPy" (the script
# asserts its tolerance before it prints that); the number is not part of it.
# Normalize only the known successful book output, with a finite nonnegative
# distance. A failed program, another format, NaN or infinity remains visible.
MAX_ERROR = re.compile(
    r"^(hgemm_v\d+ \d+x\d+x\d+: matches NumPy) "
    r"\(max error (\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\)$")


def key_lines(text: str) -> list[str]:
    """The result lines of an end-to-end program, without the times in them."""
    return [SECONDS.sub(")", line.strip()) for line in text.splitlines() if E2E_KEY.search(line)]


def claims(lines: list[str]) -> list[str]:
    """`key_lines` without what depends on the machine: "hgemm_v6 128x128x256: matches NumPy"."""
    result = []
    for line in lines:
        match = MAX_ERROR.fullmatch(line)
        result.append(match[1] if match and math.isfinite(float(match[2])) else line)
    return result


def compared_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if not line.startswith("info:")]


def float_order(bits: int) -> int:
    """float32 bit patterns onto a line of integers on which neighbours differ by one."""
    return -(bits & 0x7FFFFFFF) if bits >> 31 else bits


def is_nan(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) > 0x7F800000


def ulp_distance(got: list[str], want: list[str]):
    """If the outputs differ only in float32 hex words, (largest distance, words differing,
    words compared); else None."""
    if len(got) != len(want):
        return None
    worst = differing = total = 0
    for a, b in zip(got, want):
        if a == b:
            total += sum(1 for token in a.split() if HEX_WORD.match(token))
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return None
        for x, y in zip(ta, tb):
            if HEX_WORD.match(x) and HEX_WORD.match(y):
                total += 1
                if x != y:
                    i, j = int(x, 16), int(y, 16)
                    if (i & 0x7FFFFFFF) >= 0x7F800000 or (j & 0x7FFFFFFF) >= 0x7F800000:
                        return None
                    differing += 1
                    worst = max(worst, abs(float_order(i) - float_order(j)))
            elif x != y:
                return None
    return worst, differing, total


def first_differences(got: list[str], want: list[str], limit: int) -> list[str]:
    shown = []
    for index in range(max(len(got), len(want))):
        a = got[index] if index < len(got) else "<end of output>"
        b = want[index] if index < len(want) else "<end of output>"
        if a != b:
            shown += [f"    line {index + 1}", f"      predicted: {b[:200]}", f"      GPU:       {a[:200]}"]
            if len(shown) >= 3 * limit:
                break
    count = sum(1 for index in range(max(len(got), len(want)))
                if (got[index] if index < len(got) else None) != (want[index] if index < len(want) else None))
    shown.append(f"    {count} line(s) differ; predicted {len(want)} line(s), GPU {len(got)}")
    return shown


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def launch_failed(status: str | None, text: str) -> bool:
    return (status == "launch-failed" or text.startswith("LAUNCH FAILED") or "\nLAUNCH FAILED" in text
            or (status == "exit 4" and "LAUNCH FAILED:" in text))     # the words in the middle of a begun line


def host_error(status: str | None, text: str) -> bool:
    return status == "host-error" or text.startswith("HOST ERROR") or "\nHOST ERROR" in text


FAILED_AT = re.compile(r"^KERNEL FAILED: .*: (cudaError\w+) (at launch|at synchronize|after synchronize) \(", re.M)


def gpu_error(stdout: str, stderr: str) -> str:
    """The CUDA error of a failed kernel and where the program got it: "cudaErrorIllegalAddress
    at synchronize". The place is on standard error only (common.h, probe_report_failure)."""
    found = FAILED_AT.search(stderr)
    if found:
        return f"{found[1]} {found[2]}"
    name = re.search(r"^info: error (\w+)", stdout, re.M)
    return f"{name[1]} (place not recorded)" if name else "error not recorded"


def simulator_says(note: list[str]) -> tuple[str, str]:
    """What predicted/<name>.sim.txt says of the simulator's run: (kind, the simulator's words).

        none         there is no prediction
        finishes     every kernel returned
        deadlock     the simulator reports a kernel that can never return
        unsupported  it does not run the kernel: an instruction or operand it has no model of
        refuses      it stops the kernel for a rule it models, and names the rule"""
    if not note or not note[0].startswith("status:"):
        return "none", ""
    if note[0] == "status: ok":
        return "finishes", ""
    reason = next((line.removeprefix("warplens: ") for line in note[1:] if line.startswith("warplens: ")), "")
    if reason.startswith("deadlock"):
        return "deadlock", reason
    if reason.startswith("unsupported instruction"):
        return "unsupported", reason
    return "refuses", reason


def find_results(root: Path, given: Path) -> Path:
    """The directory with the GPU's status.tsv under `root`, or exit saying why there is not one."""
    if (root / "status.tsv").is_file():
        return root
    candidates = sorted(path for path in root.rglob("status.tsv")
                        if "predicted-toolkit" not in path.relative_to(root).parts and path.parent.name != "rows")
    if not candidates:
        sys.exit(f"{given}: no status.tsv of a GPU run inside (one under predicted-toolkit/ is a simulator's, "
                 "one in rows/ is run_rows.py's)")
    preferred = [path for path in candidates if path.parts[-3:] == ("out", "blackwell", "status.tsv")]
    if len(preferred) == 1:
        return preferred[0].parent
    if len(candidates) == 1:
        return candidates[0].parent
    sys.exit(f"{given}: more than one status.tsv could be the GPU run's; give the directory of one:\n"
             + "\n".join(f"    {path.parent.relative_to(root)}" for path in (preferred or candidates)))


def toolkit_banner(results: Path, predicted: Path) -> str:
    """"" when the programs of this run were built by the nvcc the predictions in `predicted`
    were made with, else what to print. The run says what built it (env/toolkit.txt, from
    run_all.py); the predictions say what they were made from (provenance.json). Comparing
    the two here, not trusting the verdict recorded at run time, is what lets a run with a
    fallback toolkit be held against predictions regenerated with that same toolkit."""
    keys = dict(line.split("=", 1) for line in read(results / "env" / "toolkit.txt").splitlines() if "=" in line)
    if not keys:
        if not (results / "env").is_dir():
            return ""  # a simulator run: it ran the very programs the predictions came from
        return ("TOOLKIT NOT RECORDED: the results have no env/toolkit.txt, so it is not known whether the "
                "programs were built by the toolkit the predictions were made with")
    built = keys.get("built_nvcc_version", "unknown")
    provenance = predicted / "provenance.json"
    made_with = json.loads(provenance.read_text()).get("nvcc_version", "") if provenance.is_file() else ""
    if keys.get("binaries") == "bin-ampere":
        return keys.get("reason", "NOT THE PROGRAMS THE PREDICTIONS WERE MADE FROM: the Ampere dry-run build")
    if built == "unknown" or not made_with:
        return (f"TOOLKIT NOT RECORDED: programs built by nvcc {built}, predictions in {predicted} made with nvcc "
                f"{made_with or 'unknown'}; cannot tell whether they are comparable")
    if built != made_with:
        return (f"BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: the programs were built by nvcc {built}, "
                f"the predictions in {predicted} are the simulator's run of PTX from nvcc {made_with}. A difference "
                f"from a prediction may be the compiler's. Predictions for nvcc {built} can be made on any machine "
                "(env/remote/RUNBOOK.md, \"Driver older than the toolkit\") and given with --predicted")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--predicted", default=str(HERE / "predicted"))
    parser.add_argument("--lines", type=int, default=3, help="differing lines to show per probe")
    parser.add_argument("--require-same-toolkit", action="store_true",
                        help="refuse to compare a run that was not built by the predictions' toolkit")
    parser.add_argument("--sim-captures", help="output of `e2e/run_e2e.py --mode sim` (or its captures/): the "
                        "simulator's tensors for the end-to-end runs")
    parser.add_argument("--no-replay", action="store_true",
                        help="never replay a GPU capture in the simulator; a case without a matching simulator "
                             "capture is then NOT COMPARED, which is a failure")
    parser.add_argument("--replay-out", help="keep the tensors of simulator replays here")
    parser.add_argument("--require-hardware", action="store_true",
                        help="fail an end-to-end run whose captures are not labelled as made on a GPU")
    args = parser.parse_args()

    sim_captures = None
    if args.sim_captures:
        sim_captures = Path(args.sim_captures)
        if (sim_captures / "captures").is_dir():
            sim_captures = sim_captures / "captures"
        if not sim_captures.is_dir():
            sys.exit(f"{args.sim_captures}: no such directory of simulator captures")
    results = Path(args.results)
    scratch = None
    if results.is_file():
        scratch = tempfile.TemporaryDirectory()
        with tarfile.open(results) as tar:
            tar.extractall(scratch.name, filter="data")
        results = find_results(Path(scratch.name), results)
    elif results.is_dir():
        results = find_results(results, results)
    predicted = Path(args.predicted)

    statuses = {}
    for line in read(results / "status.tsv").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 2:
            statuses[fields[0]] = fields[1]

    banner = toolkit_banner(results, predicted)
    if banner:
        bar = "!" * 100
        print(f"{bar}\n{banner}\n{bar}\n")
        if args.require_same_toolkit:
            return 3

    unhealthy = read(results / "GPU-UNHEALTHY.txt")
    if unhealthy:
        bar = "!" * 100
        print(f"{bar}\n{unhealthy}{bar}\n")

    if (results / "RT-11-NOT-CAPTURED.txt").is_file():
        # No verdict here depends on it; a reader of this report must still not miss it.
        bar = "!" * 100
        print(f"{bar}\n{read(results / 'RT-11-NOT-CAPTURED.txt')}{bar}\n")

    artifact_errors = validate_probe_archive(results, {p.program for p in PROBES})
    rows, details, bad, unverified = [], artifact_errors, int(bool(artifact_errors)), 0
    for probe in PROBES:
        status = statuses.get(probe.name)
        got_text = read(results / (probe.name + ".stdout"))
        got = compared_lines(got_text)
        want = compared_lines(read(predicted / (probe.name + ".txt")))
        note = read(predicted / (probe.name + ".sim.txt")).splitlines()
        says, reason = simulator_says(note)
        refused = says in ("deadlock", "unsupported", "refuses")
        design = " (by design)" if probe.risk else ""
        gpu_failed = "KERNEL FAILED" in got_text
        error_line = next((l for l in got_text.splitlines() if l.startswith("info: error")), "")[6:]
        stderr_text = read(results / (probe.name + ".stderr"))
        failure = gpu_error(got_text, stderr_text) if gpu_failed else ""
        # Results that are themselves a simulator's run (run_all.py --mode sim or predict, held
        # against predictions to see that they are current): its messages are on standard error.
        simulated = any(line.startswith("warplens: ") for line in stderr_text.splitlines())

        if status is not None and status.startswith("not run (GPU unhealthy"):
            # Not a finding about this probe: the runner stopped launching (GPU-UNHEALTHY.txt).
            verdict, detail = "NOT RUN, GPU unhealthy", status[len("not run ("):].rstrip(")")
            bad += 1
        elif status is None or status.startswith(("not built", "not run", "skipped")):
            verdict, detail = "not run", status or "no entry in status.tsv"
            bad += 1
        elif launch_failed(status, got_text):
            # Before anything else: a program whose kernels the driver could not load says
            # nothing about the claim, whatever the manifest allows this probe to do.
            verdict, detail = "FAILED TO LAUNCH", error_line[:110]
            bad += 1
        elif host_error(status, got_text):
            verdict = "HOST ERROR"
            detail = next((l for l in got_text.splitlines() if l.startswith("HOST ERROR")), "")[12:110]
            bad += 1
        elif status.startswith("timeout"):
            unreaped = "; " + status[len("timeout ("):].rstrip(")") if status != "timeout" else ""
            if probe.risk != "hang":
                verdict, detail = "timed out", (f"simulator: {reason[:90]}" if refused else "") + unreaped
                bad += 1
            elif says == "finishes":
                # The manifest lets it hang; the simulator says it does not. That is a prediction
                # the GPU did not confirm, whatever the manifest allows.
                verdict, detail = "TIMED OUT, predicted to finish", "the simulator's kernel returns" + unreaped
                bad += 1
                details += [f"{probe.name}: {', '.join(probe.ids) or '-'}: {probe.claim}",
                            f"    timed out on the GPU ({status}); the simulator predicts that it finishes, with "
                            f"{len(want)} line(s) of output. The GPU printed {len(got)} before it was killed:"] + \
                           ["    " + l[:200] for l in got[:12]]
            elif says == "deadlock":
                verdict, detail = "timed out (as predicted)", f"simulator: {reason[:90]}{unreaped}"
            else:
                verdict = "timed out (by design)"
                detail = (f"simulator: {reason[:90]}" if refused else "no prediction") + unreaped
        elif status != "ok" and not gpu_failed:
            verdict, detail = "crashed", status + ": " + (read(results / (probe.name + ".stderr")).strip()[-120:])
            bad += 1
        elif probe.predict == "none" and gpu_failed:
            verdict = "kernel failed" + design
            detail = f"no prediction; GPU: {failure}; {len(got_text.splitlines())} line(s) recorded"
            bad += not probe.risk
        elif probe.predict == "none":
            verdict, detail = "informational", f"no prediction; {len(got_text.splitlines())} line(s) recorded"
        elif refused and not gpu_failed:
            verdict, detail = "informational", f"the GPU ran what the simulator refuses ({reason[:70]})"
            details += [f"{probe.name}: GPU output (the simulator refuses this kernel)"] + \
                       ["    " + l[:200] for l in got[:12]]
        elif got == want and gpu_failed and says == "unsupported" and not simulated:
            # The compared lines of a failed kernel are its label and nothing else, so this
            # "match" holds for any error at all, and the simulator's side is not a model.
            verdict = "GPU faulted, not modelled"
            detail = f"GPU: {failure}; the simulator does not model this ({reason[:80]})"
            bad += not probe.risk
        elif got == want and gpu_failed and says == "deadlock":
            # The same lines: the simulator reports its deadlock as a failed kernel. A GPU has
            # no such report (a kernel that deadlocks is a timeout there), so its error is
            # something else that ended the kernel; the detail says so and the reader judges.
            verdict = "identical"
            detail = f"GPU: {failure}, where the simulator predicts a deadlock, which a GPU does not report ({reason[:50]})"
        elif got == want:
            verdict = "identical"
            detail = f"both refuse the kernel; GPU: {failure}; simulator: {reason[:80]}" if gpu_failed else ""
            if gpu_failed and simulated:
                detail = f"a simulator's run, refusing the kernel as predicted ({reason[:80]})"
            if gpu_failed and not refused:
                verdict, detail = "kernel failed" + design, f"as predicted; GPU: {failure}"
        elif gpu_failed:
            verdict = "kernel failed" + design
            detail = f"GPU: {failure}" + (f"; simulator: {reason[:80]}" if refused else "; the simulator's kernel returns")
            bad += not probe.risk
            details += [f"{probe.name}: the GPU reported a failed kernel; outputs differ"] + \
                first_differences(got, want, args.lines)
        else:
            distance = ulp_distance(got, want) if probe.compare == "ulp" else None
            if distance is not None:
                worst, differing, total = distance
                verdict = "unverified"
                unverified += 1
                detail = f"{differing} of {total} float32 words differ, by at most {worst} ulp"
            else:
                verdict, detail = "DIFFERS", ""
                bad += 1
            details += [f"{probe.name}: {', '.join(probe.ids)}: {probe.claim}"] + \
                first_differences(got, want, args.lines)
        rows.append((probe.name, verdict, detail))

    for name in E2E:
        status = statuses.get(name)
        text = read(results / (name + ".stdout"))
        errors = completion_errors(name, status, text)
        captures = results / "captures" / name.removeprefix("e2e/")
        if status == "ok":
            errors.extend(validate_capture_tree(captures, EXPECTED_CASES[name]))
        verdict, numeric = "verification failed", []
        if status == "ok" and not errors:
            # The program passed its own check and left complete captures: hold its tensors
            # against the simulator's (matrix rows RT-06 to RT-09).
            sim_run = sim_captures / name.removeprefix("e2e/") if sim_captures is not None else None
            errors, numeric = compare_run(
                captures, EXPECTED_CASES[name], sim_run, replay=not args.no_replay,
                replay_out=Path(args.replay_out) / name.removeprefix("e2e/") if args.replay_out else None,
                require_hardware=args.require_hardware)
            if errors:
                verdict = "NOT HARDWARE" if all("not hardware" in e for e in errors) else \
                    "DIFFERS" if any("NOT COMPARED" not in e for e in errors) else "NOT COMPARED"
            elif len(numeric) != len(EXPECTED_CASES[name]):  # nothing passes by not being looked at
                errors, verdict = [f"{len(numeric)} of {len(EXPECTED_CASES[name])} case(s) compared"], "NOT COMPARED"
            else:
                verdict = "bit-identical" if all(": bit-identical [" in line for line in numeric) else "within tolerance"
        bad += bool(errors)
        if status is None:
            rows.append((name, "not run", "missing required end-to-end run"))
            continue
        got, want = key_lines(read(results / (name + ".stdout"))), key_lines(read(predicted / (name + ".txt")))
        if status.startswith("not run (GPU unhealthy"):
            verdict = "NOT RUN, GPU unhealthy"
        elif status.startswith(("skipped", "not run")):
            verdict = "not run"
        elif status.startswith("timeout"):
            verdict = "timed out"
        elif status != "ok":
            verdict = "crashed"
        rows.append((name, verdict, "; ".join(errors)[:400] if errors else
                     f"{len(numeric)} tensor(s) compared with the simulator's, below"))
        if errors:
            details += [f"{name}: failed"] + ["    " + e for e in errors]
        details += [f"{name}: tensors, GPU against simulator"] + ["    " + line for line in numeric]
        details += [f"{name}: result lines, GPU then simulator"] + ["    GPU: " + l[:160] for l in got[:12]] + \
                   ["    sim: " + l[:160] for l in want[:12]]
        if got != want and claims(got) == claims(want):
            details += ["    the same claims: a max error is the distance to NumPy's result on the machine that "
                        "ran the script, and that depends on its BLAS"]

    width = max(len(name) for name, _, _ in rows)
    print(f"{'probe':{width}}  {'verdict':30}  detail")
    for name, verdict, detail in rows:
        print(f"{name:{width}}  {verdict:30}  {detail}")
    counts: dict[str, int] = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\n" + ", ".join(f"{n} {v}" for v, n in sorted(counts.items(), key=lambda item: -item[1])))
    if counts.get("FAILED TO LAUNCH"):
        print(f"\n{counts['FAILED TO LAUNCH']} probe(s) FAILED TO LAUNCH: the driver could not load their kernels, "
              "so they tested nothing.\nIf the error is cudaErrorUnsupportedPtxVersion the driver is older than the "
              "toolkit: rerun with `run_all.py --binaries native` (see README, 'When probes do not launch').")
    # health.tsv, from run_all.py: every check of the canary. A failure that went away is
    # not a verdict on any probe, but the runs after it were made on a GPU that had just
    # failed, and a reader should know.
    events = [line.split("\t") for line in read(results / "health.tsv").splitlines()[1:]]
    events = [fields for fields in events if len(fields) >= 6 and fields[5].startswith(("FAILED THEN", "GPU UNHEALTHY"))]
    if events:
        print("\nGPU health (health.tsv): the canary failed after")
        for fields in events:
            print(f"    {fields[0]}: {fields[5]} ({fields[2]} s, {fields[3]} attempt(s), from {fields[4]})")
    kernel_lines = [line for line in read(results / "xid.log").splitlines()
                    if line and not line.startswith(("#", "(before the run)\t"))]
    if kernel_lines:
        # NVRM also logs what is no fault (on the development machine: "User does not have
        # permission to access register offset ..."); an Xid is the driver reporting a GPU
        # error, so those are the ones shown.
        xids = [line for line in kernel_lines if "Xid" in line]
        print(f"\n{len(kernel_lines)} kernel log line(s) naming Xid or NVRM appeared during the run (xid.log, under "
              f"the run after which each was first seen); {len(xids)} name an Xid" + (", the first:" if xids else ""))
        print("\n".join("    " + line[:200] for line in xids[:8]), end="\n" if xids else "")
    if banner:
        print(f"\n{banner}")
    if details:
        print("\nDetails\n" + "\n".join(details))
    if scratch:
        scratch.cleanup()
    return 1 if bad else 2 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
