"""The capture of matrix row RT-11: what e2e/gdb_driver_trace.py saves of a launch with
WARPLENS_TRACE_CAPTURE=module,memory, and e2e/launch_capture.py, which says whether that is
complete.

The row itself needs a B200. What can be shown without one:
  - with no GPU at all: the completeness check, on capture directories written here, for a
    complete capture and for each thing that can be missing; and that a missing capture
    fails run_rows.py exactly when it is required
  - on any CUDA GPU, with gdb (skipped otherwise): the tracer's whole path for real. The
    CuTe DSL's SAXPY through e2e/cute_dsl_tma.py --script, held against the frontend
    capture of the same script; and a CUDA C++ kernel that takes 128-byte, 64-byte-aligned
    structs by value with device addresses inside, as a kernel with tensor maps does.
    With a CUDA PyTorch named by WARPLENS_TORCH_PYTHON as well: a CuTe DSL kernel over PyTorch
    tensors, traced with the settings the tutorial is traced with.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
SUITE = REPO / "validation" / "blackwell"
E2E = SUITE / "e2e"
KERNEL = "kernel_cutlass_kernel_0"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(SUITE), str(E2E)]
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[:2]
    return module


launch_capture = load("launch_capture_under_test", E2E / "launch_capture.py")


# ---------------------------------------------------------------------------------------------
# The completeness check, on captures written here
# ---------------------------------------------------------------------------------------------

def saved(directory: Path, name: str, data: bytes) -> dict:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def dump(directory: Path, data: bytes) -> dict:
    return saved(directory, f"mem/{hashlib.sha256(data).hexdigest()}.bin", data)


def write(directory: Path, capture: dict, launches: list, expect: dict | None = None) -> None:
    cap = directory / "capture"
    (cap / "capture.json").write_text(json.dumps(capture))
    (cap / "launches.json").write_text(json.dumps(launches))
    if expect is not None:
        (cap / "expect.json").write_text(json.dumps(expect))


def tutorial_like(directory: Path):
    """A capture shaped like the tutorial's: one launch with two 128-byte descriptors and a
    pointer, three tensors inside one allocation of 4096 bytes, the third written by the kernel."""
    cap = directory / "capture"
    cap.mkdir(parents=True)
    base = 0x7F0000000000
    before, after = bytes(range(256)) * 16, bytes(range(256)) * 12 + b"\x07" * 1024
    descriptor = lambda address: (b"\x01" * 8 + address.to_bytes(8, "little")).ljust(128, b"\xA5")
    parameters = [(0, descriptor(base)), (128, descriptor(base + 1024)), (256, (base + 3072).to_bytes(8, "little"))]
    module = saved(cap, "module00.bin", b"\x50\xed\x55\xba" + b"machine code of " + KERNEL.encode())
    capture = {"capture": ["memory", "module"], "problems": [],
               "modules": [{"call": "cuLibraryLoadData", "format": "fatbin", "result": 0, "handle": 11, **module}],
               "lookups": [{"call": "cuLibraryGetKernel", "module_handle": 11, "name": KERNEL, "result": 0, "handle": 12}],
               "handles": [{"call": "cuKernelGetFunction", "from_handle": 12, "result": 0, "handle": 13}],
               "allocations": [{"call": "cuMemAlloc_v2", "event": "allocate", "size": 4096, "result": 0, "base": base}]}
    launch = {"index": 0, "call": "cuLaunchKernelEx", "kernel": KERNEL, "grid": [2, 2, 1], "block": [128, 1, 1],
              "parameters": [{"index": i, "offset": offset, "size": len(data), "hex": data.hex()}
                             for i, (offset, data) in enumerate(parameters)],
              "memory": {"notes": [], "launch_result": 0, "synchronize_result": 0, "live_allocations": 1,
                         "all_live_allocations_read": True,
                         "pointers": [{"parameter": 0, "offset": 8, "value": base, "allocation": base,
                                       "offset_in_allocation": 0},
                                      {"parameter": 1, "offset": 8, "value": base + 1024, "allocation": base,
                                       "offset_in_allocation": 1024},
                                      {"parameter": 2, "offset": 0, "value": base + 3072, "allocation": base,
                                       "offset_in_allocation": 3072}],
                         "allocations": [{"base": base, "size": 4096, "known_from": "followed", "referenced": True,
                                          "before": dump(cap, before), "after": dump(cap, after)}]}}
    for i, (_, data) in enumerate(parameters[:2]):
        (cap / f"launch00.param{i:02d}.bin").write_bytes(data)
    (cap / "dsl_dump").mkdir()
    (cap / "dsl_dump" / "host.sm_100a.ptx").write_text(f".version 8.8\n.visible .entry {KERNEL}(\n)\n{{\n}}\n")
    (cap / "exit.txt").write_text("exit 0\n")
    (cap / "gdb.stdout").write_text("Thread 1 hit Breakpoint\nPASS\n[Inferior 1 exited normally]\n")
    (directory / "tutorial.stdout").write_text("Running Blackwell fp16 GEMM example 0\nPASS\n")
    write(directory, capture, [launch])
    return capture, launch


def problems_after(directory: Path, change) -> list[str]:
    """The problems of a tutorial-like capture after `change(cap, capture, launch)`."""
    capture, launch = tutorial_like(directory)
    change(directory / "capture", capture, launch)
    write(directory, capture, [launch])
    return launch_capture.verify(directory)[0]


def test_a_complete_capture_is_complete_and_its_files_are_listed(tmp_path):
    tutorial_like(tmp_path)
    problems, warnings, lines = launch_capture.verify(tmp_path)
    assert problems == [] and warnings == []
    text = "\n".join(lines)
    assert "parameter 1 offset 8: 0x7f0000000400 = allocation 0x7f0000000000 + 1024" in text
    assert "looked up in a saved module" in text and "PTX: dsl_dump/host.sm_100a.ptx" in text

    assert launch_capture.main(["verify", str(tmp_path)]) == 0
    assert (tmp_path / "summary.txt").read_text().startswith("RT-11 capture: COMPLETE\n")
    listed = dict(reversed(line.split("  ", 1)) for line in (tmp_path / "capture.sha256").read_text().splitlines())
    on_disk = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} - {"capture.sha256"}
    assert set(listed) == on_disk
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest for name, digest in listed.items())
    if shutil.which("sha256sum"):
        assert subprocess.run(["sha256sum", "--quiet", "-c", "capture.sha256"], cwd=tmp_path).returncode == 0


def first_allocation(launch):
    return launch["memory"]["allocations"][0]


MISSING = {
    "a dump that is not there":
        (lambda cap, c, l: (cap / first_allocation(l)["after"]["file"]).unlink(), "is missing"),
    "a dump that was not read":
        (lambda cap, c, l: first_allocation(l).update(after={"error": "cuMemcpyDtoH_v2 returned 1"}),
         "after the kernel: not saved (cuMemcpyDtoH_v2 returned 1)"),
    "no dump after the kernel at all":
        (lambda cap, c, l: first_allocation(l).pop("after"), "after the kernel: not saved"),
    "a dump shorter than its allocation":
        (lambda cap, c, l: first_allocation(l).update(size=8192), "expected 8192"),
    "a dump that was altered":
        (lambda cap, c, l: (cap / first_allocation(l)["before"]["file"]).write_bytes(b"\0" * 4096),
         "does not have its recorded SHA-256"),
    "a module of no bytes":
        (lambda cap, c, l: ((cap / "module00.bin").write_bytes(b""), c["modules"][0].update(size=0)),
         "module00.bin has 0 bytes"),
    "a module that could not be read":
        (lambda cap, c, l: [c["modules"][0].pop(k) for k in ("file", "size", "sha256")]
         and c["modules"][0].update(error="not saved: Cannot access memory"), "not saved (not saved: Cannot access memory)"),
    "no module":
        (lambda cap, c, l: c.update(modules=[]), "no module was handed to the driver"),
    "no descriptor":
        (lambda cap, c, l: l.update(parameters=l["parameters"][2:]), "no tensor map descriptor was captured"),
    "no launch":
        (lambda cap, c, l: l.clear() or l.update(index=0, call="x", parameters=[]), "no tensor map descriptor"),
    "a pointer whose allocation is not listed":
        (lambda cap, c, l: l["memory"]["pointers"][0].update(allocation=1), "whose allocation is not listed"),
    "no pointer at all":
        (lambda cap, c, l: l["memory"].update(pointers=[], all_live_allocations_read=False),
         "no parameter points into device memory"),
    "memory not captured":
        (lambda cap, c, l: l.pop("memory"), "no memory was captured"),
    "the launch failed":
        (lambda cap, c, l: l["memory"].update(launch_result=700), "the launch call returned 700"),
    "the kernel faulted":
        (lambda cap, c, l: l["memory"].update(synchronize_result=700), "cuCtxSynchronize after the launch returned 700"),
    "the launch never returned":
        (lambda cap, c, l: (l["memory"].pop("synchronize_result"),
                            l["memory"]["notes"].append("the launch call did not return; nothing was read after it")),
         "the launch call did not return"),
    "nothing changed":
        (lambda cap, c, l: first_allocation(l).update(after=first_allocation(l)["before"]),
         "no captured memory differs after the kernel"),
    "a kernel that is in no module":
        (lambda cap, c, l: l.update(kernel="another_kernel"), "was not looked up in a saved module"),
    "no kernel name":
        (lambda cap, c, l: l.pop("kernel"), "the kernel's name is not known"),
    "no PTX":
        (lambda cap, c, l: (cap / "dsl_dump" / "host.sm_100a.ptx").unlink(), "no PTX with `.entry"),
    "PTX of another kernel":
        (lambda cap, c, l: (cap / "dsl_dump" / "host.sm_100a.ptx").write_text(".entry other(\n"), "no PTX with `.entry"),
    "the program failed":
        (lambda cap, c, l: (cap / "exit.txt").write_text("exit 1\n"), "the traced program ended with: exit 1"),
    "the traced run did not pass":
        (lambda cap, c, l: (cap / "gdb.stdout").write_text("FAILED\n"), "the traced run did not print 'PASS'"),
    "the plain run did not pass":
        (lambda cap, c, l: (cap.parent / "tutorial.stdout").write_text("Traceback\nAssertionError\n"),
         "the plain run's output ends with 'AssertionError'"),
    "the tracer had a problem":
        (lambda cap, c, l: c["problems"].append("launch_memory_before: Cannot access memory"), "the tracer: launch_memory"),
    "a parameter file that is not there":
        (lambda cap, c, l: (cap / "launch00.param01.bin").unlink(), "launch00.param01.bin is missing"),
    "the parameters could not be read":
        (lambda cap, c, l: l.update(note="could not be read: Cannot access memory"), "could not be read"),
}


@pytest.mark.parametrize("name", MISSING)
def test_what_is_missing_is_named(tmp_path, name):
    change, words = MISSING[name]
    problems = problems_after(tmp_path, change)
    assert any(words in problem for problem in problems), problems
    assert launch_capture.main(["verify", str(tmp_path)]) == 1
    summary = (tmp_path / "summary.txt").read_text()
    assert summary.startswith("RT-11 capture: INCOMPLETE\n") and "PROBLEM: " in summary


def test_a_descriptor_with_no_visible_address_needs_every_live_allocation(tmp_path):
    """The address inside a tensor map may be packed. Then the capture is complete only if
    every live allocation was read, and it says so."""
    def no_pointer_in_descriptor_1(cap, capture, launch):
        launch["memory"]["pointers"] = [p for p in launch["memory"]["pointers"] if p["parameter"] != 1]

    capture, launch = tutorial_like(tmp_path / "all")
    no_pointer_in_descriptor_1(None, capture, launch)
    write(tmp_path / "all", capture, [launch])
    problems, warnings, _ = launch_capture.verify(tmp_path / "all")
    assert problems == []
    assert len(warnings) == 1 and "no pointer was found in parameter 1" in warnings[0] and "All 1 live" in warnings[0]
    assert launch_capture.write_summary(tmp_path / "all") == []
    summary = (tmp_path / "all" / "summary.txt").read_text()
    assert summary.startswith("RT-11 capture: COMPLETE\n") and "WARNING: launch 0: no pointer was found" in summary

    def and_not_all_read(cap, capture, launch):
        no_pointer_in_descriptor_1(cap, capture, launch)
        launch["memory"]["all_live_allocations_read"] = False
    problems = problems_after(tmp_path / "some", and_not_all_read)
    assert any("no pointer was found in parameter 1" in p and "not every live allocation" in p for p in problems)


def test_launches_and_modules_left_out_on_request_are_not_missing(tmp_path):
    """PyTorch's own kernels: their launches are listed with no memory, their modules with no
    bytes. Neither is a hole in the capture; an unreadable allocation nobody points into is a warning."""
    capture, launch = tutorial_like(tmp_path)
    torch_launch = {"index": 1, "call": "cuLaunchKernel", "kernel": "_ZN2at6native4copyE", "grid": [1, 1, 1],
                    "block": [128, 1, 1], "parameters": [{"index": 0, "offset": 0, "size": 160, "hex": "00" * 160}],
                    "memory": {"skipped": "the kernel's name matches WARPLENS_TRACE_MEMORY_SKIP (^_Z)"}}
    unread = {"index": 2, "call": "cuLaunchKernel", "kernel": "_ZN2at6native3addE", "grid": [1, 1, 1],
              "block": [128, 1, 1], "parameters": [], "note": "cuFuncGetParamInfo returned 400 for parameter 0",
              "memory": {"pointers": [], "allocations": [], "notes": ["no parameter of the launch was read"]}}
    capture["modules"].append({"call": "cuLibraryLoadData", "format": "wrapped fatbin", "size": 1034552,
                               "sha256": "00" * 32, "not_kept": "linked into a library", "result": 0, "handle": 21})
    launch["memory"]["allocations"].append({"base": 0x7F0000100000, "size": 512, "known_from": "followed",
                                            "referenced": False, "before": {"error": "over the limit; not read"},
                                            "after": {"error": "over the limit; not read"}})
    write(tmp_path, capture, [launch, torch_launch, unread])
    (tmp_path / "capture" / "launch01.param00.bin").write_bytes(b"\0" * 160)
    problems, warnings, lines = launch_capture.verify(tmp_path)
    assert problems == []
    assert len(warnings) == 2 and all("over the limit" in warning for warning in warnings)
    assert any("not kept (linked into a library), 1034552 bytes" in line for line in lines)


def test_a_script_without_descriptors_is_checked_for_what_it_has(tmp_path):
    """--script (the SAXPY dry run): no descriptor and no PASS are expected, every launch is checked."""
    capture, launch = tutorial_like(tmp_path)
    launch["parameters"] = launch["parameters"][2:]
    launch["memory"]["pointers"] = launch["memory"]["pointers"][2:]
    (tmp_path / "tutorial.stdout").write_text("saxpy over 256 elements: matches NumPy\n")
    (tmp_path / "capture" / "gdb.stdout").write_text("saxpy over 256 elements: matches NumPy\n")
    write(tmp_path, capture, [launch], expect={"descriptor": False, "ptx": True, "last_line": ""})
    assert launch_capture.verify(tmp_path)[0] == []
    # Held to the tutorial's expectations, it is not complete.
    write(tmp_path, capture, [launch], expect=launch_capture.EXPECT)
    assert len(launch_capture.verify(tmp_path)[0]) == 3


def test_a_directory_without_a_capture_is_incomplete(tmp_path, capsys):
    assert launch_capture.main(["verify", str(tmp_path)]) == 1
    assert capsys.readouterr().out.startswith("INCOMPLETE: ")
    (tmp_path / "capture").mkdir()
    assert launch_capture.main(["verify", str(tmp_path)]) == 1
    assert "PROBLEM: capture.json: cannot be read" in (tmp_path / "summary.txt").read_text()


# ---------------------------------------------------------------------------------------------
# A missing capture fails the run exactly when it is required
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def rows():
    return load("run_rows_under_test", SUITE / "run_rows.py")


@pytest.mark.parametrize("rt11, capability, status, verdict", [
    ("auto", "10.0", "ok", 0),
    ("auto", "10.0", "skipped (gdb not found)", 1),            # a B200 without the capture
    ("auto", "10.0", "failed (the capture is incomplete: ...)", 1),
    ("auto", "10.0", "timeout", 1),
    ("auto", "8.6", "skipped (needs sm_100 ...)", 0),           # the development machine
    ("auto", "", "skipped (gdb not found)", 0),                 # no GPU: CI
    ("auto", "10.0", "skipped (--no-e2e)", 0),                  # asked to leave it out
    ("auto", "10.0", None, 0),                                  # --only chose other steps
    ("optional", "10.0", "skipped (gdb not found)", 0),
    ("required", "8.6", "skipped (needs sm_100 ...)", 1),
    ("required", "8.6", "ok", 0),
])
def test_rt11_is_required_on_a_b200_and_nowhere_else(rows, tmp_path, monkeypatch, capsys, rt11, capability, status, verdict):
    monkeypatch.setitem(sys.modules, "cute_dsl_tma", SimpleNamespace(compute_capability=lambda: capability))
    marker = tmp_path / "RT-11-NOT-CAPTURED.txt"
    statuses = {} if status is None else {rows.RT11: status}
    assert rows.rt11_verdict(SimpleNamespace(rt11=rt11), tmp_path, statuses) == verdict
    out = capsys.readouterr().out
    assert marker.is_file() == bool(verdict)
    if verdict:
        assert "MATRIX ROW RT-11 WAS NOT CAPTURED" in out and status in marker.read_text()
        assert out.startswith("!" * 100) and "RUNBOOK.md" in marker.read_text()
    else:
        assert out == ""


def test_a_capture_made_later_clears_the_flag_and_only_that(rows, tmp_path):
    marker = tmp_path / "RT-11-NOT-CAPTURED.txt"
    marker.write_text("from the first attempt\n")
    args = SimpleNamespace(rt11="required")
    assert rows.rt11_verdict(args, tmp_path, {}) == 0 and marker.is_file()             # the step did not run
    assert rows.rt11_verdict(args, tmp_path, {rows.RT11: "skipped (--no-e2e)"}) == 0 and marker.is_file()
    assert rows.rt11_verdict(args, tmp_path, {rows.RT11: "ok"}) == 0 and not marker.exists()


def run_rows(*arguments, env=None):
    return subprocess.run([sys.executable, str(SUITE / "run_rows.py"), *map(str, arguments)],
                          capture_output=True, text=True, env={**os.environ, **(env or {})})


def test_run_rows_fails_for_a_required_capture_that_is_skipped(tmp_path):
    """The whole script, with no gdb: RT-11 skips itself. Required, that is exit status 1, a
    banner and the flag file; RT-10, skipped for the same reason, never is."""
    no_gdb = {"WARPLENS_GDB": str(tmp_path / "no-gdb")}
    out = tmp_path / "results"
    done = run_rows("--out", out, "--only", "e2e/*", "--rt11", "required", env=no_gdb)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "MATRIX ROW RT-11 WAS NOT CAPTURED: e2e/cute_dsl_tma ended as: skipped (gdb not found" in done.stdout
    assert (out / "RT-11-NOT-CAPTURED.txt").read_text() in done.stdout
    assert done.stdout.count("skipped (gdb not found") == 4          # three steps, and the banner

    for arguments in (["--rt11", "optional"], ["--rt11", "required", "--no-e2e"],
                      ["--rt11", "required", "--only", "e2e/driver_call_trace.*"]):
        other = tmp_path / "-".join(arguments).replace("/", "_").replace("*", "all")
        done = run_rows("--out", other, *(["--only", "e2e/*"] if "--only" not in arguments else []), *arguments, env=no_gdb)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "NOT CAPTURED" not in done.stdout and not (other / "RT-11-NOT-CAPTURED.txt").exists()


def test_running_one_step_again_keeps_the_other_lines(tmp_path):
    """The runbook's repair: `run_rows.py --only e2e/cute_dsl_tma` after the cause is fixed
    must not throw away what status.tsv says about the other rows."""
    no_gdb = {"WARPLENS_GDB": str(tmp_path / "no-gdb")}
    assert run_rows("--out", tmp_path, "--only", "e2e/*", "--rt11", "optional", env=no_gdb).returncode == 0
    status = tmp_path / "rows" / "status.tsv"
    status.write_text(status.read_text().replace("skipped (gdb not found", "FIRST RUN (gdb not found"))
    assert run_rows("--out", tmp_path, "--only", "e2e/cute_dsl_tma", "--rt11", "optional", env=no_gdb).returncode == 0
    lines = status.read_text().splitlines()
    # RT-11 runs, and is listed, first (it used to be last).
    assert [line.split("\t")[0] for line in lines[1:]] == ["e2e/cute_dsl_tma", "e2e/driver_call_trace.tirx",
                                                          "e2e/driver_call_trace.cute_dsl"]
    assert ["FIRST RUN" in line for line in lines[1:]] == [False, True, True]


def test_run_all_fails_without_a_required_capture_and_says_so_at_the_end():
    """run_all.py hands --rt11 on, counts a run_rows.py that did not exit 0 as a failure of
    the run, and repeats the flag file after the summary."""
    source = (SUITE / "run_all.py").read_text()
    block = source[source.index("    def run_rows() -> None:"):source.index("    for kind, batch in execution_phases(")]
    assert '"--rt11", args.rt11' in block
    assert 'if status != "ok":\n            infrastructure_failures += 1' in block
    ending = source[source.index('print(f"compare with: '):]
    assert 'RT-11-NOT-CAPTURED.txt' in ending and "return 1 if infrastructure_failures else 0" in ending


@pytest.mark.parametrize("rows_exit, flagged, verdict", [
    ("ok", False, 0),
    ("exit 1", True, 1),            # run_rows.py: required and missing
    ("timeout", False, 1),          # run_rows.py did not get to say
])
def test_run_all_ends_as_run_rows_does(tmp_path, monkeypatch, capsys, rows_exit, flagged, verdict):
    """run_all.py in GPU mode with no probes and the rest of the machine stood in for: its
    exit status and its last lines follow run_rows.py's, and --rt11 reaches it."""
    run_all = load("run_all_under_test", SUITE / "run_all.py")
    out = tmp_path / "results"
    commands = []

    def run_process(command, timeout, **_):
        commands.append([str(word) for word in command])
        if any(word.endswith("run_rows.py") for word in commands[-1]):
            if flagged:
                (out / "RT-11-NOT-CAPTURED.txt").write_text("MATRIX ROW RT-11 WAS NOT CAPTURED: for this test\n")
            return rows_exit, 1.0, b"e2e/cute_dsl_tma    RT-11   the status\n", b""
        return "ok", 0.0, b"", b""

    monkeypatch.setattr(run_all, "run_process", run_process)
    monkeypatch.setattr(run_all, "PROBES", [])
    monkeypatch.setattr(run_all, "archive_probes", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_all, "record_toolkit", lambda out, binaries: {"predictions_comparable": "yes", "reason": ""})
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--mode", "gpu", "--no-build", "--no-e2e", "--binaries", "ampere",
                                      "--rt11", "required", "--out", str(out)])
    assert run_all.main() == verdict
    (rows_command,) = [command for command in commands if any(word.endswith("run_rows.py") for word in command)]
    assert rows_command[rows_command.index("--rt11") + 1] == "required" and "--no-e2e" in rows_command
    printed = capsys.readouterr().out
    assert (out / "RT-11-NOT-CAPTURED.txt").is_file() == bool(verdict)
    if verdict:
        text = (out / "RT-11-NOT-CAPTURED.txt").read_text()
        assert printed.rstrip().endswith("!" * 100) and text in printed[printed.index("results in "):]
        assert ("for this test" in text) if flagged else ("run_rows.py ended with: timeout" in text)
        with tarfile.open(str(out) + ".tar.gz") as tar:      # and it is in the archive
            assert "results/RT-11-NOT-CAPTURED.txt" in tar.getnames()
    else:
        assert "NOT CAPTURED" not in printed


def test_compare_repeats_the_flag(tmp_path):
    """compare.py's report starts with the missing capture, whatever the probes say."""
    (tmp_path / "status.tsv").write_text("name\tstatus\tseconds\tcommand\n")
    (tmp_path / "RT-11-NOT-CAPTURED.txt").write_text("MATRIX ROW RT-11 WAS NOT CAPTURED: for this test\n")
    done = subprocess.run([sys.executable, str(SUITE / "compare.py"), str(tmp_path)], capture_output=True, text=True)
    assert "MATRIX ROW RT-11 WAS NOT CAPTURED: for this test" in done.stdout, done.stdout[-2000:] + done.stderr
    (tmp_path / "RT-11-NOT-CAPTURED.txt").unlink()
    done = subprocess.run([sys.executable, str(SUITE / "compare.py"), str(tmp_path)], capture_output=True, text=True)
    assert "NOT CAPTURED" not in done.stdout


# ---------------------------------------------------------------------------------------------
# The tracer itself, on whatever CUDA GPU this machine has
# ---------------------------------------------------------------------------------------------

def gpu_tracer():
    """The tracer module, or a skip with the reason: no GPU in reach, no gdb."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CUDA_VISIBLE_DEVICES is empty: this run is to stay off the GPU")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("needs a CUDA GPU (no nvidia-smi)")
    if smi.returncode != 0 or not smi.stdout.strip():
        pytest.skip("needs a CUDA GPU (nvidia-smi found none)")
    tracer = load("driver_call_trace_for_capture", E2E / "driver_call_trace.py")
    if not tracer.find_gdb()[0]:
        pytest.skip("needs gdb with Python support")
    return tracer


def skip_unless_traceable(text: str) -> None:
    if "ptrace is not permitted" in text or "SKIPPED:" in text:
        pytest.skip("this machine does not let gdb trace a process")


@pytest.mark.gpu
def test_the_whole_capture_of_a_cute_dsl_launch_on_this_gpu(tmp_path):
    """e2e/cute_dsl_tma.py --script with the CuTe DSL's SAXPY, for real: module, PTX,
    allocations, the memory before and after. Held against what the frontend capture of the
    same script saves from inside the program (its PTX, its inputs, its output)."""
    gpu_tracer()
    pytest.importorskip("cutlass")
    numpy = pytest.importorskip("numpy")
    script = REPO / "examples" / "06_cute_dsl" / "saxpy.py"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = tmp_path / "rt11"
    done = subprocess.run([sys.executable, str(E2E / "cute_dsl_tma.py"), "--out", str(out), "--script", str(script)],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    skip_unless_traceable(done.stdout)
    if "plain run of saxpy.py: ok" not in done.stdout:
        pytest.skip("the CuTe DSL's SAXPY does not run on this GPU: " + done.stdout[-300:])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "capture: complete" in done.stdout
    assert (out / "summary.txt").read_text().startswith("RT-11 capture: COMPLETE\n")
    assert launch_capture.main(["verify", str(out)]) == 0           # again, as it would be offline

    cap = out / "capture"
    capture = json.loads((cap / "capture.json").read_text())
    (launch,) = json.loads((cap / "launches.json").read_text())
    assert capture["problems"] == []
    # The code: the DSL hands the driver a fat binary; its own cubin dump is inside it.
    (module,) = capture["modules"]
    assert module["call"] == "cuLibraryLoadData" and module["format"] == "fatbin" and module["result"] == 0
    (cubin,) = (cap / "dsl_dump").glob("*.cubin")
    assert cubin.read_bytes() in (cap / module["file"]).read_bytes()
    (lookup,) = capture["lookups"]
    assert lookup["name"] == launch["kernel"] and lookup["module_handle"] == module["handle"]
    # Two allocations of 1024 bytes, followed from cuMemAlloc and confirmed by the driver.
    memory = launch["memory"]
    assert [p["parameter"] for p in memory["pointers"]] == [1, 2] and all(p["offset"] == 0 for p in memory["pointers"])
    assert all(a["size"] == a["driver_size"] == 1024 and a["known_from"] == "followed" for a in memory["allocations"])
    assert memory["launch_result"] == 0 and memory["synchronize_result"] == 0

    # The same script's frontend capture, made inside the program with no debugger.
    raw = tmp_path / "frontend"
    again = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=REPO,
                           env={**env, "WARPLENS_CAPTURE_DIR": str(raw), "WARPLENS_CAPTURE_MODE": "gpu"}, timeout=900)
    case = raw / "raw" / "cute_dsl_saxpy"
    assert (case / "y.output.npy").is_file(), again.stdout + again.stderr
    (ptx,) = (cap / "dsl_dump").glob("*.ptx")
    assert ptx.read_text() == (case / "kernel.ptx").read_text()
    by_parameter = {p["parameter"]: next(a for a in memory["allocations"] if a["base"] == p["allocation"])
                    for p in memory["pointers"]}
    contents = lambda record: numpy.fromfile(cap / record["file"], dtype=numpy.float32)[:256]
    x, y, result = (numpy.load(case / name) for name in ("x.input.npy", "y.input.npy", "y.output.npy"))
    assert (contents(by_parameter[1]["before"]) == x).all() and (contents(by_parameter[1]["after"]) == x).all()
    assert (contents(by_parameter[2]["before"]) == y).all()
    assert (contents(by_parameter[2]["after"]) == result).all()     # the expected result, bit for bit
    assert not (result == y).all()


@pytest.mark.gpu
def test_addresses_inside_128_byte_parameters_are_followed_on_this_gpu(tmp_path):
    """tests/data/launch_capture_descriptors.cu: three 128-byte, 64-byte-aligned structs by
    value. The address at offset 0 and the one at offset 3 are found; the packed one is not,
    and its memory is in the capture all the same, because every live allocation is read."""
    tracer = gpu_tracer()
    numpy = pytest.importorskip("numpy")
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        pytest.skip("needs nvcc")
    program = tmp_path / "descriptors"
    built = subprocess.run([nvcc, "-arch=native", "-o", str(program), str(REPO / "tests/data/launch_capture_descriptors.cu")],
                           capture_output=True, text=True, timeout=600)
    if built.returncode != 0:
        pytest.skip("nvcc cannot build for this GPU: " + built.stderr[-300:])
    plain = subprocess.run([str(program)], capture_output=True, text=True, timeout=300)
    if plain.stdout.strip() != "PASS":
        pytest.skip("the program does not run on this GPU: " + (plain.stdout + plain.stderr)[-300:])

    out = tmp_path / "rt11"
    (out / "capture").mkdir(parents=True)
    (out / "tutorial.stdout").write_text(plain.stdout)
    (out / "capture" / "expect.json").write_text(json.dumps({"descriptor": True, "ptx": False, "last_line": "PASS"}))
    status, facts = tracer.trace([str(program)], out / "capture", 600, capture=("module", "memory"))
    if status.startswith("skipped"):
        pytest.skip(status)
    assert status == "ok"
    (launch,) = facts["launches"]
    assert launch["kernel"] == "add_three" and [p["size"] for p in launch["parameters"]] == [4, 128, 128, 128, 8]

    problems, warnings, _ = launch_capture.verify(out)
    assert problems == []
    assert len(warnings) == 1 and "no pointer was found in parameter 3 (128 bytes)" in warnings[0]

    memory = launch["memory"]
    assert [(p["parameter"], p["offset"]) for p in memory["pointers"]] == [(1, 0), (2, 3), (4, 0)]
    assert memory["live_allocations"] == 4 and memory["all_live_allocations_read"] is True
    assert [a["referenced"] for a in memory["allocations"]] == [True, True, True, False]
    assert all(a["known_from"] == "followed" and a["size"] == a["driver_size"] == 4000 for a in memory["allocations"])
    contents = lambda record: numpy.fromfile(out / "capture" / record["file"], dtype=numpy.float32)
    by_parameter = {p["parameter"]: next(a for a in memory["allocations"] if a["base"] == p["allocation"])
                    for p in memory["pointers"]}
    index = numpy.arange(1000, dtype=numpy.float32)
    assert (contents(by_parameter[1]["before"]) == 1000 + index).all()
    assert (contents(by_parameter[2]["before"]) == 2000 + index).all()
    packed = memory["allocations"][3]                   # reached only by reading every live allocation
    assert (contents(packed["before"]) == 3000 + index).all() and packed["after"] == packed["before"]
    assert (contents(by_parameter[4]["before"]) == 0).all()
    assert (contents(by_parameter[4]["after"]) == 6000 + 3 * index).all()
    # The code: what nvcc links in is a wrapper around a fat binary; the fat binary is what is saved.
    capture = json.loads((out / "capture" / "capture.json").read_text())
    (module,) = capture["modules"]
    assert module["format"] == "wrapped fatbin"
    assert (out / "capture" / module["file"]).read_bytes()[:4] == bytes.fromhex("50ed55ba")
    assert capture["problems"] == []


@pytest.mark.gpu
def test_a_cute_dsl_launch_over_pytorch_tensors_on_this_gpu(tmp_path):
    """tests/data/cute_dsl_torch_saxpy.py has the tutorial's host side (tensors from PyTorch's
    caching allocator, from_dlpack, a PyTorch reference computed on the GPU afterwards) around a
    kernel any GPU runs. Traced as cute_dsl_tma.py traces the tutorial: PyTorch's own kernels
    keep their parameters to themselves, their modules are listed and not kept."""
    tracer = gpu_tracer()
    numpy = pytest.importorskip("numpy")
    python = os.environ.get("WARPLENS_TORCH_PYTHON")
    if not python:
        pytest.skip("needs WARPLENS_TORCH_PYTHON: an interpreter with a CUDA PyTorch and the CuTe DSL")
    script = REPO / "tests" / "data" / "cute_dsl_torch_saxpy.py"
    plain = subprocess.run([python, str(script)], capture_output=True, text=True, timeout=900)
    if plain.stdout.strip().splitlines()[-1:] != ["PASS"]:
        pytest.skip("the script does not run with that interpreter on this GPU: " + plain.stderr[-300:])

    out = tmp_path / "rt11"
    cap = out / "capture"
    (cap / "dsl_dump").mkdir(parents=True)
    (out / "tutorial.stdout").write_text(plain.stdout)
    (cap / "expect.json").write_text(json.dumps({"descriptor": False, "ptx": True, "last_line": "PASS"}))
    env = {**os.environ, "CUTE_DSL_KEEP": "cubin,ptx", "CUTE_DSL_DUMP_DIR": str(cap / "dsl_dump"),
           "WARPLENS_TRACE_MEMORY_SKIP": "^_Z", "WARPLENS_TRACE_MODULE_SKIP_WRAPPED": "1"}
    status, facts = tracer.trace([python, str(script)], cap, 900, env=env, capture=("module", "memory"))
    if status.startswith("skipped"):
        pytest.skip(status)
    assert status == "ok"
    problems, warnings, _ = launch_capture.verify(out)
    assert problems == [] and warnings == []

    capture = json.loads((cap / "capture.json").read_text())
    ours = [l for l in facts["launches"] if not l["kernel"].startswith("_Z")]
    theirs = [l for l in facts["launches"] if l["kernel"].startswith("_Z")]
    assert len(ours) == 1 and theirs, [l.get("kernel") for l in facts["launches"]]
    assert all("skipped" in l["memory"] for l in theirs)
    kept = [m for m in capture["modules"] if "file" in m]
    assert len(kept) == 1 and kept[0]["format"] == "fatbin"
    assert all(m["format"] == "wrapped fatbin" and m["size"] > 0 for m in capture["modules"] if "not_kept" in m)
    # Both tensors are inside one allocation of PyTorch's allocator, which the tracer followed
    # from cuMemAlloc (through cudaMalloc) and the driver confirms.
    memory = ours[0]["memory"]
    (allocation,) = memory["allocations"]
    assert allocation["known_from"] == "followed" and allocation["driver_size"] == allocation["size"] >= 2 * 4096 * 4
    x_at, y_at = (next(p["offset_in_allocation"] for p in memory["pointers"] if p["parameter"] == i) for i in (1, 2))
    assert x_at != y_at
    tensor = lambda record, at: numpy.frombuffer((cap / record["file"]).read_bytes(), dtype=numpy.float32,
                                                 count=4096, offset=at)
    x, y = tensor(allocation["before"], x_at), tensor(allocation["before"], y_at)
    assert set(numpy.unique(x)) <= {-2.0, -1.0, 0.0, 1.0} and len(numpy.unique(x)) > 1      # random_(-2, 2)
    assert (tensor(allocation["after"], x_at) == x).all()
    assert (tensor(allocation["after"], y_at) == numpy.float32(2.5) * x + y).all()           # exact in float32
