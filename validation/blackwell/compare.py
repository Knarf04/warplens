#!/usr/bin/env python3
"""Compare a run of the probe suite on a GPU with the simulator's predictions.

    validation/blackwell/compare.py RESULTS [--predicted DIR] [--lines N]

RESULTS is the archive run_all.py wrote (or its unpacked directory). Each probe gets one
verdict:

    identical       its output is the committed prediction, line for line
    rounding        only float32 words differ (probes compared with a tolerance); the largest
                    distance in units in the last place is given
    DIFFERS         it is not; the first differing lines are shown below the table
    informational   there is no prediction to hold it against: the probe has none, or the
                    simulator refuses the kernel and the GPU ran it
    kernel failed   the GPU faulted or rejected the kernel (the program reported it)
    timed out       killed by the timeout
    crashed         the program itself died, or could not be started
    not run         no result in the archive

"(by design)" marks an outcome the manifest allows for that probe. Lines that start with
"info:" carry machine-dependent values and are never compared. The exit status is 1 if any
probe DIFFERS, crashed, was not run, or failed or timed out without that being by design.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import E2E, PROBES  # noqa: E402

HEX_WORD = re.compile(r"^[0-9a-f]{8}$")
E2E_KEY = re.compile(r"Relative error|Execution is|max abs error|matches NumPy|FAILED|SKIPPED|rror")
SECONDS = re.compile(r", \d+\.\d+ s\)")


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
                    if is_nan(i) or is_nan(j):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--predicted", default=str(HERE / "predicted"))
    parser.add_argument("--lines", type=int, default=3, help="differing lines to show per probe")
    args = parser.parse_args()

    results = Path(args.results)
    scratch = None
    if results.is_file():
        scratch = tempfile.TemporaryDirectory()
        with tarfile.open(results) as tar:
            tar.extractall(scratch.name, filter="data")
        found = list(Path(scratch.name).rglob("status.tsv"))
        if not found:
            sys.exit(f"{results}: no status.tsv inside")
        results = found[0].parent
    predicted = Path(args.predicted)

    statuses = {}
    for line in read(results / "status.tsv").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 2:
            statuses[fields[0]] = fields[1]

    rows, details, bad = [], [], 0
    for probe in PROBES:
        status = statuses.get(probe.name)
        got_text = read(results / (probe.name + ".stdout"))
        got = compared_lines(got_text)
        want = compared_lines(read(predicted / (probe.name + ".txt")))
        note = read(predicted / (probe.name + ".sim.txt")).splitlines()
        refused = bool(note) and note[0].startswith("status:") and note[0] != "status: ok"
        reason = next((l.removeprefix("warplens: ") for l in note[1:] if l.startswith("warplens: ")), "")
        design = " (by design)" if probe.risk else ""
        gpu_failed = "KERNEL FAILED" in got_text or "HOST ERROR" in got_text

        if status is None or status.startswith(("not built", "not run", "skipped")):
            verdict, detail = "not run", status or "no entry in status.tsv"
            bad += 1
        elif status == "timeout":
            verdict, detail = "timed out" + (design if probe.risk == "hang" else ""), ""
            bad += probe.risk != "hang"
            if refused:
                detail = f"simulator: {reason[:90]}"
        elif status != "ok" and not gpu_failed:
            verdict, detail = "crashed", status + ": " + (read(results / (probe.name + ".stderr")).strip()[-120:])
            bad += 1
        elif probe.predict == "none":
            verdict, detail = "informational", f"no prediction; {len(got_text.splitlines())} line(s) recorded"
        elif refused and not gpu_failed:
            verdict, detail = "informational", f"the GPU ran what the simulator refuses ({reason[:70]})"
            details += [f"{probe.name}: GPU output (the simulator refuses this kernel)"] + \
                       ["    " + l[:200] for l in got[:12]]
        elif got == want:
            verdict = "identical"
            detail = "both refuse the kernel" if gpu_failed else ""
            if gpu_failed and not refused:
                verdict, detail = "kernel failed" + design, "as predicted"
        elif gpu_failed:
            verdict = "kernel failed" + design
            detail = next((l for l in got_text.splitlines() if l.startswith("info: error")), "")[6:]
            bad += not probe.risk
            details += [f"{probe.name}: the GPU reported a failed kernel; outputs differ"] + \
                first_differences(got, want, args.lines)
        else:
            distance = ulp_distance(got, want) if probe.compare == "ulp" else None
            if distance is not None:
                worst, differing, total = distance
                verdict = "rounding"
                detail = f"{differing} of {total} float32 words differ, by at most {worst} ulp"
            else:
                verdict, detail = "DIFFERS", ""
                bad += 1
            details += [f"{probe.name}: {', '.join(probe.ids)}: {probe.claim}"] + \
                first_differences(got, want, args.lines)
        rows.append((probe.name, verdict, detail))

    for name in E2E:
        status = statuses.get(name)
        if status is None:
            rows.append((name, "not run", ""))
            continue
        keys = lambda text: [SECONDS.sub(")", l.strip()) for l in text.splitlines() if E2E_KEY.search(l)]
        got, want = keys(read(results / (name + ".stdout"))), keys(read(predicted / (name + ".txt")))
        verdict = {"ok": "informational", "timeout": "timed out"}.get(status, "crashed")
        if status.startswith("skipped"):
            verdict = "not run"
        rows.append((name, verdict, status if status != "ok" else f"{len(got)} result line(s), below"))
        details += [f"{name}: result lines, GPU then simulator"] + ["    GPU: " + l[:160] for l in got[:12]] + \
                   ["    sim: " + l[:160] for l in want[:12]]

    width = max(len(name) for name, _, _ in rows)
    print(f"{'probe':{width}}  {'verdict':26}  detail")
    for name, verdict, detail in rows:
        print(f"{name:{width}}  {verdict:26}  {detail}")
    counts: dict[str, int] = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\n" + ", ".join(f"{n} {v}" for v, n in sorted(counts.items(), key=lambda item: -item[1])))
    if details:
        print("\nDetails\n" + "\n".join(details))
    if scratch:
        scratch.cleanup()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
