"""Reject portable recipes that silently lose allocation state or change pointer meaning."""
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from validation.blackwell import capture, record_host


def make_case(tmp_path, monkeypatch, *, tma=False, scalar=False, annotated=False):
    monkeypatch.setattr(capture, "provenance", lambda: {"test": True})
    parameter = ".param .align 64 .b8 arg[128]" if tma else ".param .u64 " + (".ptr .global " if annotated else "") + "arg"
    ptx = f".version 9.5\n.target sm_100a\n.address_size 64\n.visible .entry k({parameter}) {{ ret; }}"
    array = np.array(1, dtype="<u4") if scalar else np.zeros((4, 8), dtype="<u4")
    args = [{"buffer": "a", "offset": 0}]
    if tma:
        args = [{"hex": "00" * 128, "relocations": [{"offset": 0, "tensor_map": {
            "buffer": "a", "data_type": 2, "rank": 2, "dims": [8, 4], "strides": [32],
            "box": [8, 4], "steps": [1, 1], "swizzle": 0, "nan_fill": False, "interleave": 0}}]}]
    return capture.save_case(tmp_path, "test", inputs={"a": array}, outputs={"a": array}, ptx=ptx,
                             launch={"entry": "k", "grid": [1, 1, 1], "block": [1, 1, 1],
                                     "cluster": [1, 1, 1], "smem": 0, "args": args},
                             comparison={"kind": "exact"}, mode="sim")


def mutate_case(directory, change):
    path = directory / "capture.json"
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize("extra", [{"buffer_offset": 4}, {"hex": "00" * 8}, {"relocations": []}])
def test_direct_pointer_cannot_have_ignored_or_conflicting_fields(tmp_path, monkeypatch, extra):
    case = make_case(tmp_path, monkeypatch)
    mutate_case(case, lambda d: d["launch"]["args"][0].update(extra))
    assert capture.validate_capture_tree(case)


@pytest.mark.parametrize("change", [
    lambda d: d.update(dims=[8, 5]),
    lambda d: d.update(strides=[48]),
    lambda d: d.update(buffer_offset=1),
    lambda d: d.update(buffer_offset=16),
])
def test_tensor_map_must_fit_and_align_within_actual_captured_allocation(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch, tma=True)
    mutate_case(case, lambda d: change(d["launch"]["args"][0]["relocations"][0]["tensor_map"]))
    assert capture.validate_capture_tree(case)


def test_pointer_annotation_requires_relocation_except_null(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, annotated=True)
    mutate_case(case, lambda d: d["launch"].update(args=[{"hex": struct.pack("<Q", 0x200001000).hex()}]))
    assert capture.validate_capture_tree(case)
    mutate_case(case, lambda d: d["launch"].update(args=[{"hex": "00" * 8}]))
    assert capture.validate_capture_tree(case) == []


def test_scalar_buffer_preserves_shape_and_replays(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, scalar=True)
    assert capture.load_case(case)["buffers"]["a"]["shape"] == []
    assert capture.replay_case(case, tmp_path / "replay") == []


@pytest.mark.parametrize("scratch", [False, True])
def test_exported_host_does_not_mistake_uncaptured_allocations_for_scalars(tmp_path, monkeypatch, scratch):
    from warplens import Device, cli
    from warplens.host import shim_server
    array = np.array([17, 19], "<u4")
    (tmp_path / "input.bin").write_bytes(array.tobytes())
    (tmp_path / "spec.json").write_text(json.dumps({"module": "unused", "inputs": {
        "a": {"file": "input.bin", "dtype": "<u4", "shape": [2]}}}))
    ptx = ".version 9.5\n.target sm_100a\n.address_size 64\n.visible .entry k(.param .u64 a, .param .u64 second) { ret; }"
    (tmp_path / "expected.ptx").write_text(ptx)
    def run(_):
        module = shim_server.parse_ptx(ptx)
        device = Device("B200")
        a = device.from_numpy(array)
        other = device.from_numpy(np.array([29, 31], "<u4"))
        server = SimpleNamespace(device=device, modules={0: [module]}, functions={1: (0, 0, "k")})
        raw = struct.pack("<10IQ", 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)
        raw += struct.pack("<QQ", a.addr, other.addr if scratch else 12345)
        assert shim_server.ShimServer.handle(server, shim_server.LAUNCH, raw) == b""
        return 0
    monkeypatch.setattr(cli, "main", run)
    if scratch:
        with pytest.raises(ValueError, match="uncaptured allocation pointer"):
            record_host._record(tmp_path)
        assert not (tmp_path / "launch.json").exists()
    else:
        assert record_host._record(tmp_path) == 0
        args = json.loads((tmp_path / "launch.json").read_text())["args"]
        assert args == [{"buffer": "a", "offset": 0}, {"hex": struct.pack("<Q", 12345).hex()}]


def test_source_snapshot_rejects_dangling_tracked_symlink(tmp_path):
    import subprocess
    from env.remote.freeze_source import freeze
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init")
    (repo / "broken.py").symlink_to("missing.py")
    git("add", "broken.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-m", "symlink")
    with pytest.raises(ValueError, match="regular tracked files"):
        freeze(repo, tmp_path / "snapshot")


def test_host_recipe_timeout_uses_process_tree_cleanup(monkeypatch):
    def run(command, timeout, env):
        assert command[1:3] == ["-m", "validation.blackwell.record_host"]
        assert timeout == 120
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        return "timeout", 120, b"partial progress", b"worker hung"
    monkeypatch.setattr(record_host, "run_process", run)
    with pytest.raises(RuntimeError, match=r"failed \(timeout\)") as error:
        record_host.capture_exported("unused.so", {"a": np.zeros(1, "<u4")}, "test PTX")
    assert "partial progress" in str(error.value) and "worker hung" in str(error.value)
