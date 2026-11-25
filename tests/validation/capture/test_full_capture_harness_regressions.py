"""Missing native work/results/artifacts must never produce a successful full capture."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

from tests.support import REPO as ROOT

SUITE = ROOT / "validation" / "blackwell"
sys.path.insert(0, str(SUITE))
from validation.blackwell.e2e_checks import EXPECTED_CASES, completion_errors
from validation.blackwell.manifest import Probe
from validation.blackwell import provenance


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status", [None, "exit 1", "timeout", "skipped (no GPU)", "not built"])
def test_e2e_bad_status_never_passes(status):
    assert completion_errors("e2e/triton", status, "")


@pytest.mark.parametrize("text", ["Execution is failed.\nRelative error: 0\n",
    "Execution is successful.\nRelative error: nan\n", "Execution is successful.\n",
    "Execution is successful.\nRelative error: 0\nFAILED numerical result\n"])
def test_cutlass_exit_zero_requires_successful_finite_result(text):
    assert completion_errors("e2e/cutlass_tutorials.tutorial_01", "ok", text)


def test_all_completion_contracts_have_known_cases():
    assert not completion_errors("e2e/cutlass_tutorials.tutorial_01", "ok",
                                 "Relative error: 0.00002\nExecution is successful.\n")
    assert not completion_errors("e2e/triton", "ok", "vector add: max abs error vs NumPy 0\nmatmul: max abs error vs NumPy 0.02\n")
    assert not completion_errors("e2e/unchanged_scripts.tirx", "ok", "saxpy over 1024 elements: matches NumPy\n")
    book = "\n".join(f"hgemm_v{n} 128x128x256: matches NumPy (max error 0.0156, 1.0 s)" for n in range(1, 10))
    assert not completion_errors("e2e/book_hgemm", "ok", book)
    assert completion_errors("e2e/book_hgemm", "ok", book.splitlines()[0])
    assert len(EXPECTED_CASES["e2e/book_hgemm"]) == 9


@pytest.mark.parametrize("risk,status,output,allowed", [
    ("", "timeout", b"", False), ("hang", "timeout", b"", True),
    ("fault", "timeout", b"", False), ("fault", "exit 3", b"KERNEL FAILED: x", True),
    ("", "exit 3", b"KERNEL FAILED: x", False), ("fault", "exit 1", b"", False),
    ("fault", "launch-failed", b"LAUNCH FAILED: x", False), ("", "ok", b"KERNEL FAILED: x", False),
])
def test_manifest_only_allows_declared_faults(risk, status, output, allowed):
    runner = load("capture_runner_policy", SUITE / "run_all.py")
    assert runner.expected_outcome(Probe("p", "p", risk=risk), status, output) is allowed


def elf(ptx=b".version 9.0\n.target sm_100a\n.address_size 64\n", *, compressed=0):
    body = ptx + b"\0" * ((-len(ptx)) % 8)
    entry = struct.pack("<HHIQI", 1, 0x101, 24, len(body), compressed) + bytes(4) + body
    fatbin = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entry)) + entry
    names = b"\0.shstrtab\0.nv_fatbin\0"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<HHH", header, 58, 64, 3, 1)
    sections = bytearray(3 * 64)
    struct.pack_into("<II", sections, 64, 1, 3)
    struct.pack_into("<QQ", sections, 64 + 24, 256, len(names))
    struct.pack_into("<II", sections, 128, 11, 1)
    struct.pack_into("<QQ", sections, 128 + 24, 256 + len(names), len(fatbin))
    return bytes(header + sections) + names + fatbin


def record(binary, source=b"// frozen source\n", *, ptx=None):
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(elf() if ptx is None else elf(ptx))
    key = provenance.digest(source)
    sources = binary.parent / ".build-sources"
    sources.mkdir(exist_ok=True)
    (sources / key).write_bytes(source)
    meta = {"schema": provenance.SCHEMA, "argv": ["nvcc", "--no-compress", "probe.cu"],
            "nvcc_version": "CUDA 13.5", "sources": [{"path": "/frozen/probe.cu", "sha256": key,
            "bytes": len(source), "asset": f".build-sources/{key}"}],
            "executable_sha256": provenance.digest(binary.read_bytes())}
    Path(str(binary) + ".build.json").write_text(json.dumps(meta))


def test_extracts_exact_embedded_ptx_and_rejects_truncation(tmp_path):
    program = tmp_path / "program"
    program.write_bytes(elf())
    assert provenance.extracted_ptx(program) == [".version 9.0\n.target sm_100a\n.address_size 64\n"]
    program.write_bytes(elf()[:-3])
    with pytest.raises(ValueError, match="extent"):
        provenance.extracted_ptx(program)
    program.write_bytes(elf(compressed=1))
    with pytest.raises(ValueError, match="compressed"):
        provenance.extracted_ptx(program)


def test_archive_keeps_binary_ptx_sources_manifest_and_case_arguments(tmp_path):
    suite, out = tmp_path / "suite", tmp_path / "results"
    suite.mkdir(); out.mkdir()
    (suite / "manifest.py").write_text("# exact manifest\n")
    record(suite / "bin" / "p")
    probes = [Probe("p.basic", "p", args=("basic",)), Probe("p.edge", "p", args=("edge",), risk="fault")]
    assert provenance.archive_probes(out, suite, suite / "bin", probes, mode="gpu") == []
    assert provenance.validate_probe_archive(out, {"p"}) == []
    index = json.loads((out / "probe-artifacts" / "index.json").read_text())
    assert [p["args"] for p in index["cases"]] == [["basic"], ["edge"]]
    frozen = out / "probe-artifacts" / "p" / "executed"
    assert (frozen / "program").read_bytes() == (suite / "bin" / "p").read_bytes()
    assert (frozen / "module-0.ptx").read_text().startswith(".version 9.0")
    (frozen / "module-0.ptx").write_text("changed")
    assert any("integrity" in e for e in provenance.validate_probe_archive(out, {"p"}))


def test_missing_stale_provenance_and_changed_native_companion_fail(tmp_path):
    suite, out = tmp_path / "suite", tmp_path / "results"
    suite.mkdir(); out.mkdir(); (suite / "manifest.py").write_text("# manifest")
    probes = [Probe("p", "p")]
    assert provenance.archive_probes(out, suite, suite / "bin", probes, mode="gpu")
    record(suite / "bin" / "p")
    record(suite / "bin-native" / "p", source=b"different source")
    assert "different sources" in provenance.archive_probes(out, suite, suite / "bin-native", probes, mode="gpu")[0]
    (suite / "bin" / "p").write_bytes(elf(b".version 8.0\n"))
    assert "stale" in provenance.archive_probes(out, suite, suite / "bin", probes, mode="gpu")[0]


def test_compare_missing_e2e_capture_fails_even_with_success_stdout(tmp_path, monkeypatch):
    compare = load("capture_compare_missing", SUITE / "compare.py")
    name = "e2e/cutlass_tutorials.tutorial_01"
    monkeypatch.setattr(compare, "PROBES", [])
    monkeypatch.setattr(compare, "E2E", [name])
    monkeypatch.setattr(compare, "validate_probe_archive", lambda *args: [])
    (tmp_path / "e2e").mkdir()
    (tmp_path / "status.tsv").write_text(f"name\tstatus\n{name}\tok\n")
    (tmp_path / (name + ".stdout")).write_text("Relative error: 0\nExecution is successful.\n")
    monkeypatch.setattr(sys, "argv", ["compare", str(tmp_path)])
    assert compare.main() == 1


@pytest.mark.parametrize("failed_status", ["timeout", "exit 1", "ok"])
def test_e2e_collector_rejects_failure_or_missing_capture_and_records_every_step(tmp_path, monkeypatch, failed_status):
    runner = load("full_capture_e2e_runner", SUITE / "e2e" / "run_e2e.py")
    book = tmp_path / "book" / "chapter_gemm_basics"
    book.mkdir(parents=True); (book / "index.md").touch()
    out = tmp_path / "out"
    (out / "e2e" / "cutlass_capture_build" / "bin").mkdir(parents=True)
    for n in range(1, 5):
        (out / "e2e" / "cutlass_capture_build" / "bin" / f"tutorial_0{n}").touch()
        (tmp_path / f"tutorial_0{n}").touch()
    seen = []
    def process(command, timeout, **kwargs):
        if "build" in command or "finalize" in command:
            return "ok", 0, b"", b""
        if str(command[0]).endswith("build.sh"):
            return "ok", 0, b"", b""
        env = kwargs["env"]
        seen.append(env)
        if "tutorial_0" in command[-1]:
            return "ok", 0, b"Relative error: 0\nExecution is successful.\n", b""
        if command[-1].endswith("triton_gpu.py"):
            return failed_status, 0, b"vector add: max abs error vs NumPy 0\nmatmul: max abs error vs NumPy 0\n", b""
        if command[-1] == str(book.parent):
            text = "\n".join(f"hgemm_v{n} 128x128x256: matches NumPy (max error 0)" for n in range(1, 10))
            return "ok", 0, text.encode(), b""
        return "ok", 0, b"saxpy over 1024 elements: matches NumPy\n", b""
    monkeypatch.setattr(runner, "run_process", process)
    monkeypatch.setattr(runner, "cudart_directory", lambda: "")
    # The nominally successful Triton step produced no capture. Other fake steps model
    # correctly completed fixture writers so this test isolates the aggregate gate.
    monkeypatch.setattr(runner, "validate_capture_tree", lambda path, cases: ["missing capture"] if path.name == "triton" else [])
    monkeypatch.setattr(sys, "argv", ["run_e2e", "--out", str(out), "--book", str(book.parent), "--cutlass-bin", str(tmp_path)])
    assert runner.main() == 1
    rows = (out / "status.tsv").read_text().splitlines()[1:]
    assert len(rows) == 8
    triton = next(line for line in rows if line.startswith("e2e/triton\t"))
    assert ("verification-failed" if failed_status == "ok" else failed_status) in triton
    assert all(env["WARPLENS_CAPTURE_MODE"] == "gpu" and env["WARPLENS_CAPTURE_DIR"].startswith(str(out)) for env in seen)


@pytest.mark.parametrize("available", [0, 1])
def test_custom_cutlass_binary_availability_is_honored_without_substituting_missing_tutorials(
        tmp_path, monkeypatch, available):
    runner = load("capture_requested_cutlass", SUITE / "e2e" / "run_e2e.py")
    requested = tmp_path / "requested-bin"
    requested.mkdir()
    if available:
        (requested / "tutorial_01").touch()
    source, out = tmp_path / "chosen-source", tmp_path / "out"
    calls = []
    def process(command, timeout, **kwargs):
        calls.append(command)
        if "build" in command:
            assert command[command.index("--cutlass") + 1] == str(source)
            binaries = out / "e2e" / "cutlass_capture_build" / "bin"
            binaries.mkdir(parents=True)
            for n in range(1, 5):
                (binaries / f"tutorial_0{n}").touch()
            return "ok", 0, b"", b""
        if "finalize" in command:
            return "ok", 0, b"", b""
        if command[-1].endswith("tutorial_01"):
            assert command[-1] != str(requested / "tutorial_01")
            return "ok", 0, b"Relative error: 0\nExecution is successful.\n", b""
        return "exit 2", 0, b"SKIPPED: unavailable test frontend\n", b""
    monkeypatch.setattr(runner, "run_process", process)
    monkeypatch.setattr(runner, "cudart_directory", lambda: "")
    monkeypatch.setattr(runner, "validate_capture_tree", lambda *args: [])
    monkeypatch.setattr(sys, "argv", ["run_e2e", "--out", str(out), "--cutlass-bin", str(requested),
                                     "--cutlass-source", str(source), "--book", str(tmp_path / "no-book")])
    assert runner.main() == 1
    statuses = dict(line.split("\t")[:2] for line in (out / "status.tsv").read_text().splitlines()[1:])
    for n in range(1, 5):
        status = statuses[f"e2e/cutlass_tutorials.tutorial_0{n}"]
        assert status == "ok" if n <= available else status == f"skipped ({requested / f'tutorial_0{n}'} is not built)"
    assert sum("build" in command for command in calls) == available
    assert not any(str(runner.CUTLASS / "build.sh") in command for command in calls)
    if available:
        log = (out / "e2e" / "cutlass_capture_build.log").read_text()
        assert f"rebuilt from {source}" in log and "those binaries are not executed" in log
