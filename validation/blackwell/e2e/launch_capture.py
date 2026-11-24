#!/usr/bin/env python3
"""Is a traced launch's capture complete enough to replay the kernel without the GPU?
(matrix row RT-11; e2e/cute_dsl_tma.py runs this after its capture, and it can be run again
on the unpacked archive, with no GPU and no gdb)

    validation/blackwell/e2e/launch_capture.py verify DIR

DIR is what e2e/cute_dsl_tma.py wrote (rows/e2e/cute_dsl_tma in a run's results). Its
capture/ holds what e2e/gdb_driver_trace.py saves with WARPLENS_TRACE_CAPTURE=module,memory,
and capture/expect.json what this run was expected to contain. Checked:

  - the traced program ended with exit 0, and, where expected, its output and the plain
    run's end with the expected last line (PASS for the tutorial)
  - every module the driver was handed was saved: its file has the recorded size, which is
    not zero, and the recorded SHA-256 (the modules of PyTorch's own kernels are listed and
    not kept; the launch in question must find its kernel in one that was)
  - the launch in question (every launch with a parameter of 128 bytes or more, the size of
    a tensor map; every launch when no descriptor is expected; never one whose memory the
    tracer was told to skip, as PyTorch's own kernels are) names its kernel, and that
    name was looked up in a saved module or is in one's bytes
  - where expected, PTX with that kernel's `.entry` is there (the CuTe DSL hands the driver
    machine code, so its PTX comes from its own dump: capture/dsl_dump/*.ptx)
  - the launch call and the cuCtxSynchronize after it returned 0
  - at least one parameter points into device memory; every such pointer's allocation was
    read before and after the kernel, each dump with the allocation's size and its SHA-256
  - every parameter of 128 bytes or more has a pointer found in it, or else every live
    allocation was read (a warning: the address is then in a form the scan does not know)
  - the kernel changed some memory: an output that equals its input is no expected result

Written to DIR: summary.txt (the verdict, each problem and warning, each pointer with its
dumps) and capture.sha256 (`sha256sum -c capture.sha256` in DIR checks every file).
Exit status: 0 complete, 1 incomplete (summary.txt says what is missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DESCRIPTOR_BYTES = 128      # a CUtensorMap
EXPECT = {"descriptor": True, "ptx": True, "last_line": "PASS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, problems: list[str]):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        problems.append(f"{path.name}: cannot be read ({error})")
        return None


def saved_file(directory: Path, record, size, what: str, problems: list[str]) -> bool:
    """A {"file", "size", "sha256"} record of the tracer against the file itself."""
    if not isinstance(record, dict) or "file" not in record:
        why = record.get("error", "no record") if isinstance(record, dict) else "not read"
        problems.append(f"{what}: not saved ({why})")
        return False
    path = directory / record["file"]
    if not path.is_file():
        problems.append(f"{what}: {record['file']} is missing")
        return False
    actual = path.stat().st_size
    if actual == 0 or actual != record.get("size") or (size is not None and actual != size):
        problems.append(f"{what}: {record['file']} has {actual} bytes, recorded {record.get('size')}"
                        + (f", expected {size}" if size is not None else ""))
        return False
    if sha256(path) != record.get("sha256"):
        problems.append(f"{what}: {record['file']} does not have its recorded SHA-256")
        return False
    return True


def last_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[-1].strip()


def check_launch(directory: Path, launch: dict, capture: dict, expect: dict, module_bytes: dict[str, bytes],
                 ptx: dict[str, str], problems: list[str], warnings: list[str], lines: list[str]) -> None:
    name = f"launch {launch.get('index')}"
    kernel = launch.get("kernel")
    lines.append(f"{name}: {launch.get('call')} kernel={kernel or '?'} grid={launch.get('grid')} "
                 f"block={launch.get('block')} parameter sizes={[p['size'] for p in launch.get('parameters', [])]}")
    if "note" in launch:
        problems.append(f"{name}: {launch['note']}")
    if not kernel:
        problems.append(f"{name}: the kernel's name is not known")
    else:
        handles = {m.get("handle") for m in capture.get("modules", []) if m.get("file") in module_bytes} - {None}
        # cuLibraryGetModule: a library's kernels may be looked up through its module handle.
        handles |= {h.get("handle") for h in capture.get("handles", []) if h.get("from_handle") in handles} - {None}
        looked_up = any(l.get("name") == kernel and l.get("result") == 0 and l.get("module_handle") in handles
                        for l in capture.get("lookups", []))
        inside = [file for file, data in module_bytes.items() if kernel.encode() in data]
        if not looked_up and not inside:
            problems.append(f"{name}: kernel {kernel} was not looked up in a saved module and is in none's bytes")
        else:
            lines.append(f"  module: " + ("looked up in a saved module" if looked_up else "not seen looked up")
                         + (f"; name in {', '.join(inside)}" if inside else ""))
        if expect["ptx"]:
            sources = [file for file, text in ptx.items() if f".entry {kernel}(" in text or f".entry {kernel} " in text]
            if not sources:
                problems.append(f"{name}: no PTX with `.entry {kernel}` (capture/dsl_dump/*.ptx, or a module in text)")
            else:
                lines.append(f"  PTX: {', '.join(sources)}")
    for p in launch.get("parameters", []):
        raw = directory / f"launch{launch['index']:02d}.param{p['index']:02d}.bin"
        if p["size"] >= 64 and (not raw.is_file() or raw.read_bytes().hex() != p["hex"]):
            problems.append(f"{name}: {raw.name} is missing or is not the bytes in launches.json")

    memory = launch.get("memory")
    if not isinstance(memory, dict):
        problems.append(f"{name}: no memory was captured (the tracer ran without WARPLENS_TRACE_CAPTURE=memory, "
                        "or failed before it)")
        return
    problems.extend(f"{name}: {note}" for note in memory.get("notes", []))
    if memory.get("driver_questions_exhausted"):
        warnings.append(f"{name}: more candidate values than the driver was asked about; a pointer may be missing")
    for key, what in (("launch_result", "the launch call"), ("synchronize_result", "cuCtxSynchronize after the launch")):
        if memory.get(key) != 0:
            problems.append(f"{name}: {what} returned {memory.get(key, 'nothing (not reached)')}")
    allocations = {a["base"]: a for a in memory.get("allocations", [])}
    pointers = memory.get("pointers", [])
    if not pointers:
        problems.append(f"{name}: no parameter points into device memory")
    good = {}
    for base, allocation in allocations.items():
        what = f"{name}: allocation {base:#x} ({allocation['size']} bytes)"
        before = saved_file(directory, allocation.get("before"), allocation["size"], what + " before the kernel",
                            problems if allocation.get("referenced") else warnings)
        after = saved_file(directory, allocation.get("after"), allocation["size"], what + " after the kernel",
                           problems if allocation.get("referenced") else warnings)
        good[base] = before and after
    for pointer in pointers:
        allocation = allocations.get(pointer["allocation"])
        where = f"parameter {pointer['parameter']} offset {pointer['offset']}"
        if allocation is None:
            problems.append(f"{name}: {where} points at {pointer['value']:#x}, whose allocation is not listed")
            continue
        lines.append(f"  {where}: {pointer['value']:#x} = allocation {allocation['base']:#x} + "
                     f"{pointer['offset_in_allocation']} ({allocation['size']} bytes, known from: "
                     f"{allocation.get('known_from')}); before {allocation.get('before', {}).get('file', 'NOT SAVED')}, "
                     f"after {allocation.get('after', {}).get('file', 'NOT SAVED')}")
    for p in launch.get("parameters", []):
        if p["size"] < DESCRIPTOR_BYTES or any(pointer["parameter"] == p["index"] for pointer in pointers):
            continue
        if memory.get("all_live_allocations_read") and memory.get("live_allocations", 0) > 0:
            warnings.append(f"{name}: no pointer was found in parameter {p['index']} ({p['size']} bytes); its "
                            f"address is in a form the scan does not know. All {memory['live_allocations']} live "
                            "allocation(s) were read, so the memory it names is in the capture")
        else:
            problems.append(f"{name}: no pointer was found in parameter {p['index']} ({p['size']} bytes), and not "
                            "every live allocation was read")
    changed = [base for base, a in allocations.items()
               if good.get(base) and a["before"]["sha256"] != a["after"]["sha256"]]
    if not changed:
        problems.append(f"{name}: no captured memory differs after the kernel; there is no output to compare with")
    else:
        lines.append(f"  changed by the kernel: {', '.join(f'{base:#x}' for base in changed)}")


def verify(directory: Path) -> tuple[list[str], list[str], list[str]]:
    """(problems, warnings, lines for the summary) of the capture in `directory`."""
    problems: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    cap = directory / "capture"
    expect = dict(EXPECT)
    if (cap / "expect.json").is_file():
        expect.update(load(cap / "expect.json", problems) or {})
    capture = load(cap / "capture.json", problems)
    launches = load(cap / "launches.json", problems)
    if capture is None or launches is None:
        return problems, warnings, lines
    problems.extend(f"the tracer: {problem}" for problem in capture.get("problems", []))
    if capture.get("unseen_during_calls"):
        warnings.append(f"{len(capture['unseen_during_calls'])} driver call(s) by other threads were not followed "
                        f"({', '.join(sorted(set(capture['unseen_during_calls'])))}): an allocation may be unknown")

    ended = (cap / "exit.txt").read_text().strip() if (cap / "exit.txt").is_file() else "no exit.txt"
    if ended != "exit 0":
        problems.append(f"the traced program ended with: {ended}")
    if expect["last_line"]:
        plain = last_line((directory / "tutorial.stdout").read_text(errors="replace")) \
            if (directory / "tutorial.stdout").is_file() else "(no tutorial.stdout)"
        if plain != expect["last_line"]:
            problems.append(f"the plain run's output ends with '{plain[:80]}', not '{expect['last_line']}'")
        traced = (cap / "gdb.stdout").read_text(errors="replace").splitlines() if (cap / "gdb.stdout").is_file() else []
        if expect["last_line"] not in [line.strip() for line in traced]:
            problems.append(f"the traced run did not print '{expect['last_line']}'")

    module_bytes: dict[str, bytes] = {}
    ptx: dict[str, str] = {}
    if not capture.get("modules"):
        problems.append("no module was handed to the driver where the tracer could see it (capture.json: modules)")
    for index, module in enumerate(capture.get("modules", [])):
        what = f"module {index} ({module.get('call')}, {module.get('format', '?')})"
        if module.get("result") != 0:
            warnings.append(f"{what}: the call returned {module.get('result')}")
        if "not_kept" in module:        # a library's own kernels, left out on request
            lines.append(f"{what}: not kept ({module['not_kept']}), {module.get('size')} bytes")
        elif saved_file(cap, module, None, what, problems):
            module_bytes[module["file"]] = (cap / module["file"]).read_bytes()
            lines.append(f"{what}: {module['file']}, {module['size']} bytes, sha256 {module['sha256']}")
            if module.get("format") == "text":
                ptx[module["file"]] = module_bytes[module["file"]].decode(errors="replace")
    for path in sorted((cap / "dsl_dump").glob("*.ptx")) if (cap / "dsl_dump").is_dir() else []:
        ptx[f"dsl_dump/{path.name}"] = path.read_text(errors="replace")
    for path in sorted((cap / "dsl_dump").glob("*.cubin")) if (cap / "dsl_dump").is_dir() else []:
        inside = [file for file, data in module_bytes.items() if path.read_bytes() in data]
        lines.append(f"dsl_dump/{path.name}: " + (f"its bytes are inside {', '.join(inside)}: the DSL's dump is the "
                                                  "code the driver was handed" if inside else
                                                  "NOT inside any module the driver was handed"))
        if not inside:
            warnings.append(f"dsl_dump/{path.name} is not inside any saved module: the DSL's PTX dump may not be "
                            "the code that was launched")

    # A launch whose memory the tracer was told to leave alone (PyTorch's own kernels) is not in question.
    asked = [l for l in launches if not (isinstance(l.get("memory"), dict) and "skipped" in l["memory"])]
    chosen = [l for l in asked if any(p["size"] >= DESCRIPTOR_BYTES for p in l.get("parameters", []))] \
        if expect["descriptor"] else asked
    if not launches:
        problems.append("no kernel launch reached the driver")
    elif not chosen:
        problems.append(f"no launch has a parameter of {DESCRIPTOR_BYTES} bytes or more: no tensor map descriptor "
                        f"was captured ({len(launches)} launch(es) seen)")
    for launch in chosen:
        check_launch(cap, launch, capture, expect, module_bytes, ptx, problems, warnings, lines)
    return problems, warnings, lines


def write_summary(directory: Path) -> list[str]:
    """Verify, write summary.txt and capture.sha256, and return the problems."""
    problems, warnings, lines = verify(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in ("summary.txt", "capture.sha256"))
    total = sum(p.stat().st_size for p in files)
    text = [f"RT-11 capture: {'INCOMPLETE' if problems else 'COMPLETE'}",
            f"{len(files)} file(s), {total} bytes; checksums in capture.sha256", ""]
    text += [f"PROBLEM: {problem}" for problem in problems]
    text += [f"WARNING: {warning}" for warning in warnings]
    text += ([""] if problems or warnings else []) + lines
    (directory / "summary.txt").write_text("\n".join(text) + "\n")
    (directory / "capture.sha256").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(directory).as_posix()}\n" for p in files + [directory / "summary.txt"]))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["verify"])
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    if not (directory / "capture").is_dir():
        print(f"INCOMPLETE: {directory} has no capture/ (e2e/cute_dsl_tma.py did not get that far)")
        return 1
    problems = write_summary(directory)
    print((directory / "summary.txt").read_text(), end="")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
