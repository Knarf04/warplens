"""Frontend records preserve executed PTX and can replay without their compiler."""
from collections import namedtuple
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from validation.blackwell import capture, frontend_capture as adapters

from tests.support import REPO as ROOT

PTX = """.version 8.8
.target sm_100a
.address_size 64
.visible .entry kernel(.param .f32 scale, .param .align 8 .b8 x[8],
                       .param .align 8 .b8 y[8]) { ret; }
"""


@pytest.fixture
def recording(monkeypatch, tmp_path):
    monkeypatch.setenv("WARPLENS_CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("WARPLENS_CAPTURE_MODE", "sim")
    monkeypatch.setattr(capture, "provenance", lambda: {"test": True})
    return tmp_path / "capture"


def test_capture_is_inert_without_opt_in(monkeypatch):
    monkeypatch.delenv("WARPLENS_CAPTURE_DIR", raising=False)
    assert adapters.triton_case("unused", None, values=None, pointers=None, grid=None,
                               inputs=None, outputs=None, comparison=None) is None
    assert adapters.tvm_case("unused", None, inputs=None, outputs=None, comparison=None) is None
    assert adapters.cute_saxpy_case(None, scalar=None, grid=None, block=None,
                                   inputs=None, output=None) is None


def test_standalone_gpu_runner_does_not_need_capture_package(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "WARPLENS_CAPTURE_DIR")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run([sys.executable, str(ROOT / "validation/blackwell/e2e/triton_gpu.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 2 and "SKIPPED:" in done.stdout, done.stdout + done.stderr


def test_cute_aggregate_pointers_are_relocatable_and_scalar_exact(recording):
    compiled = SimpleNamespace(kernel_info={"kernel": {}}, __ptx__=PTX)
    initial = np.arange(256, dtype=np.float32)
    directory = adapters.cute_saxpy_case(compiled, scalar=2.5, grid=(4, 1, 1), block=(64, 1, 1),
                                        inputs={"x": initial, "y": initial}, output=initial)
    case = capture.load_case(directory)
    assert case["launch"]["args"] == [
        {"hex": "00002040"},
        {"hex": "00" * 8, "relocations": [{"offset": 0, "buffer": "x"}]},
        {"hex": "00" * 8, "relocations": [{"offset": 0, "buffer": "y"}]}]
    assert (directory / "kernel.ptx").read_text() == PTX


def test_failed_cute_abi_preserves_native_evidence(recording):
    ptx = PTX.replace("x[8]", "x[16]")
    compiled = SimpleNamespace(kernel_info={"kernel": {}}, __ptx__=ptx)
    initial = np.arange(256, dtype=np.float32)
    with pytest.raises(ValueError, match="ABI changed"):
        adapters.cute_saxpy_case(compiled, scalar=2.5, grid=(4, 1, 1), block=(64, 1, 1),
                                inputs={"x": initial, "y": initial}, output=initial)
    raw = recording / "raw" / "cute_dsl_saxpy"
    assert (raw / "kernel.ptx").read_text() == ptx
    assert np.array_equal(np.load(raw / "y.output.npy"), initial)
    assert not (recording / "cute_dsl_saxpy").exists()


def test_triton_records_specialization_cluster_geometry_and_zero_scratch(recording):
    ptx = """.version 8.8
.target sm_100a
.address_size 64
.visible .entry add(.param .u64 x, .param .u32 n, .param .u64 scratch,
                   .param .u64 profile) { ret; }
"""
    Metadata = namedtuple("Metadata", "name num_ctas num_warps shared global_scratch_size profile_scratch_size")
    kernel = SimpleNamespace(metadata=Metadata("add", 2, 4, 128, 0, 0), asm={"ptx": ptx},
                             src=SimpleNamespace(signature={"x": "*fp32", "n": "i32", "BLOCK": "constexpr"},
                                                 constants={(2,): 128}, attrs={}))
    x = np.arange(8, dtype=np.float32)
    directory = adapters.triton_case("triton", kernel, values={"n": 8}, pointers={"x": "x"},
                                     grid=(3, 2), inputs={"x": x}, outputs={"x": x}, comparison={"kind": "exact"})
    case = capture.load_case(directory)
    assert case["launch"] == {"entry": "add", "grid": [6, 2, 1], "block": [128, 1, 1],
                              "cluster": [2, 1, 1], "smem": 128,
                              "args": [{"buffer": "x", "offset": 0}, {"hex": "08000000"},
                                       {"hex": "00" * 8}, {"hex": "00" * 8}]}
    kernel.metadata = kernel.metadata._replace(global_scratch_size=16)
    with pytest.raises(ValueError, match="scratch"):
        adapters.triton_case("unsupported", kernel, values={}, pointers={}, grid=(),
                            inputs={"x": x}, outputs={"x": x + 1}, comparison={})
    raw = recording / "raw" / "unsupported"
    assert (raw / "kernel.ptx").read_text() == ptx
    assert np.array_equal(np.load(raw / "x.output.npy"), x + 1)
    assert not (recording / "unsupported").exists()


def test_tvm_preserves_raw_outputs_before_recipe_failure(recording, monkeypatch):
    from validation.blackwell import record_host
    cuda = SimpleNamespace(kind="cuda", imports=[], inspect_source=lambda fmt: PTX)
    module = SimpleNamespace(kind="llvm", imports=[cuda])
    executable = SimpleNamespace(mod=module, export_library=lambda path: Path(path).write_bytes(b"exported module"))
    x = np.arange(8, dtype=np.float32)
    def fail(path, inputs, ptx):
        assert Path(path).read_bytes() == b"exported module"
        assert ptx == PTX and np.array_equal(inputs["x"], x)
        raise ValueError("different PTX")
    monkeypatch.setattr(record_host, "capture_exported", fail)
    with pytest.raises(ValueError, match="different PTX"):
        adapters.tvm_case("failed", executable, inputs={"x": x}, outputs={"x": x + 1},
                          comparison={"kind": "exact"})
    raw = recording / "raw" / "failed"
    assert np.array_equal(np.load(raw / "x.input.npy"), x)
    assert np.array_equal(np.load(raw / "x.output.npy"), x + 1)
    assert (raw / "kernel.ptx").read_text() == PTX
    assert not (recording / "failed" / "capture.json").exists()


def test_tvm_requires_one_actual_ptx_module():
    empty = SimpleNamespace(kind="llvm", imports=[])
    with pytest.raises(ValueError, match="got 0"):
        adapters.tvm_ptx(empty)
    cuda = SimpleNamespace(kind="cuda", imports=[], inspect_source=lambda fmt: PTX)
    with pytest.raises(ValueError, match="got 2"):
        adapters.tvm_ptx(SimpleNamespace(kind="llvm", imports=[cuda, cuda]))


def test_real_triton_captures_replay_without_frontend(recording, monkeypatch, tmp_path):
    pytest.importorskip("triton")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    import warplens.triton_driver as wt
    spec = importlib.util.spec_from_file_location("frontend_capture_triton", ROOT / "examples/05_triton/run.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    gpu = wt.install("B200")
    try:
        example.vector_add(gpu, n=129)
        example.matmul(gpu, M=128, N=64, K=32)
    finally:
        wt.uninstall()
    for name in ("triton_vector_add", "triton_matmul"):
        case = capture.load_case(recording / name)
        assert case["metadata"]["frontend"] == "triton"
        assert capture.replay_case(recording / name, tmp_path / "replay" / name) == []


@pytest.mark.parametrize("frontend,script,case", [
    ("tvm_tirx", "examples/07_tirx/run.py", "tirx_saxpy"),
    ("cutlass", "examples/06_cute_dsl/saxpy.py", "cute_dsl_saxpy"),
])
def test_real_shim_frontend_capture_replays_without_frontend(frontend, script, case, tmp_path):
    pytest.importorskip(frontend)
    if frontend == "tvm_tirx":
        pytest.importorskip("cuda.bindings")
    recording = tmp_path / "capture"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "WARPLENS_CAPTURE_DIR": str(recording), "WARPLENS_CAPTURE_MODE": "sim",
           "XDG_CACHE_HOME": str(tmp_path / "cache")}
    done = subprocess.run([sys.executable, "-m", "warplens.cli", "run", sys.executable, str(ROOT / script)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    capture.load_case(recording / case)
    assert capture.replay_case(recording / case, tmp_path / "replay") == []
    if frontend == "tvm_tirx":
        assert (recording / "raw" / case / "host.so").is_file()
        assert capture.load_case(recording / case)["metadata"]["recipe"] == "replayed exported host module"


def test_book_descriptor_capture_replays_without_tvm(tmp_path):
    pytest.importorskip("tvm_tirx")
    pytest.importorskip("cuda.bindings")
    book = os.environ.get("WARPLENS_BOOK", "")
    if not book or not (Path(book) / "chapter_gemm_basics/index.md").is_file():
        pytest.skip("WARPLENS_BOOK is not a clone of the book")
    recording = tmp_path / "capture"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "WARPLENS_CAPTURE_DIR": str(recording), "WARPLENS_CAPTURE_MODE": "sim",
           "XDG_CACHE_HOME": str(tmp_path / "cache")}
    done = subprocess.run([sys.executable, "-m", "warplens.cli", "run", sys.executable,
                           str(ROOT / "examples/08_book/run.py"), book, "v8"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    case = capture.load_case(recording / "book_hgemm_v8")
    assert case["launch"]["cluster"] == [2, 1, 1]
    assert len(case["launch"]["args"]) == 3
    assert all("tensor_map" in arg["relocations"][0] for arg in case["launch"]["args"])
    assert capture.replay_case(recording / "book_hgemm_v8", tmp_path / "replay") == []
