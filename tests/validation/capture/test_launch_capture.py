"""The capture of matrix row RT-11: what e2e/gdb_driver_trace.py saves of a launch with
WARPLENS_TRACE_CAPTURE=module,memory, and e2e/launch_capture.py, which says whether that is
complete.

The row itself needs a B200. What can be shown without one:
  - with no GPU at all: the completeness check, on capture directories written here, for a
    complete capture and for each thing that can be missing; and that a missing capture
    fails run_rows.py exactly when it is required
  - on any CUDA GPU, with gdb: the tracer's whole path for real. Those tests need a live GPU
    and are in tests/validation/live_gpu/test_launch_capture_on_gpu.py.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import BLACKWELL as SUITE
from tests.support.probes import E2E, load_e2e

KERNEL = "kernel_cutlass_kernel_0"


launch_capture = load_e2e("launch_capture_under_test", E2E / "launch_capture.py")


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


@pytest.mark.parametrize("extra_kernel, remove_dsl_ptx, expected", [
    ("cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu", False, 0),
    ("another_large_parameter_kernel", False, 1),
    ("cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu", True, 1),
])
def test_tutorial_reference_gemm_is_excluded_but_other_capture_checks_remain(
        tmp_path, monkeypatch, extra_kernel, remove_dsl_ptx, expected):
    """B200's reference GEMM has an unmangled name and a 384-byte ordinary struct.
    Its absent PTX is irrelevant; an unexpected launch or missing DSL PTX is still fatal."""
    tutorial = load_e2e("cute_dsl_tma_reference_test", E2E / "cute_dsl_tma.py")
    fixture = tmp_path / "fixture"
    capture, launch = tutorial_like(fixture)
    extra = {"index": 1, "call": "cuLaunchKernelEx", "kernel": extra_kernel,
             "grid": [16, 8, 1], "block": [64, 1, 1],
             "parameters": [{"index": 0, "offset": 0, "size": 384, "hex": "00" * 384}]}
    out = tmp_path / "run"
    program = tmp_path / "fp16_gemm_0.py"
    program.touch()

    def trace(command, directory, timeout, env, **options):
        if options.get("capture"):
            # The tracer records a skip only when the actual harness configuration
            # matches the launch; keep the reference's launch and parameter bytes.
            if re.search(env["WARPLENS_TRACE_MEMORY_SKIP"], extra_kernel):
                extra["memory"] = {"skipped": "the kernel's name matches WARPLENS_TRACE_MEMORY_SKIP"}
            shutil.copytree(fixture / "capture", directory, dirs_exist_ok=True)
            (directory / "launch01.param00.bin").write_bytes(bytes(384))
            write(out, capture, [launch, extra])
            if remove_dsl_ptx:
                (directory / "dsl_dump" / "host.sm_100a.ptx").unlink()
        return "ok", {"launches": [launch]}

    monkeypatch.setattr(tutorial, "find_gdb", lambda: ("gdb", ""))
    monkeypatch.setattr(tutorial, "compute_capability", lambda: "10.0")
    monkeypatch.setattr(tutorial.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(tutorial, "run_process", lambda *args, **kwargs: ("ok", 0, b"PASS\n", b""))
    monkeypatch.setattr(tutorial, "trace", trace)
    monkeypatch.setattr(sys, "argv", ["cute_dsl_tma.py", "--out", str(out), "--tutorial", str(program)])
    assert tutorial.main() == expected
    summary = (out / "summary.txt").read_text()
    assert ("RT-11 capture: COMPLETE\n" in summary) == (expected == 0)
    if expected:
        assert "no PTX with `.entry" in summary
    else:
        assert launch["kernel"] in summary and "changed by the kernel" in summary


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
    return load_e2e("run_rows_under_test", SUITE / "run_rows.py")


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
    run_all = load_e2e("run_all_under_test", SUITE / "run_all.py")
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
