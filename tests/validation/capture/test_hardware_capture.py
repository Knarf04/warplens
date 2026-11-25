"""An offline fixture survives moving the bundle and rejects incomplete evidence."""
import json
import shutil

import numpy as np
import pytest

from validation.blackwell import capture

PTX = """.version 9.5
.target sm_100a
.address_size 64
.visible .entry copy(.param .u64 src, .param .u64 dst) {
.reg .b64 %a, %b;
.reg .b32 %v;
ld.param.u64 %a, [src];
ld.param.u64 %b, [dst];
ld.global.u32 %v, [%a];
st.global.u32 [%b], %v;
ret;
}
"""


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "provenance", lambda: {"test": True})
    return capture.save_case(tmp_path, "copy", inputs={"x": np.array([0x80000000], "<u4"),
                                                     "y": np.zeros(1, "<u4")},
                             outputs={"y": np.array([0x80000000], "<u4")}, ptx=PTX,
                             launch={"entry": "copy", "grid": [1, 1, 1], "block": [1, 1, 1],
                                     "cluster": [1, 1, 1], "smem": 0,
                                     "args": [{"buffer": "x"}, {"buffer": "y"}]},
                             comparison={"kind": "exact"}, mode="sim")


def test_moved_capture_replays_without_source_or_frontend(case, tmp_path):
    moved = tmp_path / "moved"
    shutil.move(case, moved)
    assert capture.validate_capture_tree(moved, ["copy"]) == []
    assert capture.replay_case(moved, tmp_path / "candidate") == []
    assert capture.compare_outputs(moved, tmp_path / "candidate") == []
    assert (tmp_path / "candidate/y.output.bin").read_bytes() == b"\0\0\0\x80"


@pytest.mark.parametrize("filename", ["x.input.bin", "y.output.bin", "kernel.ptx"])
def test_corrupt_and_missing_artifacts_fail(case, filename):
    path = case / filename
    path.write_bytes(path.read_bytes() + b"broken")
    assert capture.validate_capture_tree(case)
    path.unlink()
    assert capture.validate_capture_tree(case)


def test_missing_case_and_duplicate_cases_fail(case, tmp_path):
    assert capture.validate_capture_tree(tmp_path / "absent")
    assert capture.validate_capture_tree(case, ["copy", "missing"])
    shutil.copytree(case, tmp_path / "another")
    assert "duplicate capture case names" in capture.validate_capture_tree(tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="unknown"),
    lambda d: d.update(mode="unknown"),
    lambda d: d["buffers"]["x"].update(shape=[-1]),
    lambda d: d["buffers"]["x"]["initial"].update(file="../escape"),
    lambda d: d["launch"].update(entry="absent"),
    lambda d: d["launch"].update(cluster=[2, 1, 1]),
    lambda d: d["launch"].update(args=[{"buffer": "absent"}]),
    lambda d: d["launch"].update(args=[{"hex": "00", "relocations": [{"offset": 0, "buffer": "x"}]}]),
    lambda d: d["buffers"]["y"].update(comparison={"kind": "allclose", "rtol": 1, "atol": 1}),
])
def test_invalid_manifest_fails_closed(case, mutate):
    path = case / "capture.json"
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))
    assert capture.validate_capture_tree(case)


def test_candidate_missing_and_wrong_bytes_fail(case, tmp_path):
    assert capture.compare_outputs(case, tmp_path / "missing")
    (tmp_path / "y.output.bin").write_bytes(b"\0" * 4)
    assert capture.compare_outputs(case, tmp_path)


def test_numerical_policies_do_not_accept_nonfinite_values():
    raw = lambda x: np.array(x, "<f4").tobytes()
    policy = {"kind": "allclose", "rtol": 1e-5, "atol": 1e-6}
    assert capture.compare_bytes(raw([1.000001]), raw([1]), "<f4", policy) is None
    for value in [np.nan, np.inf, -np.inf]:
        assert capture.compare_bytes(raw([value]), raw([value]), "<f4", policy)
    assert capture.compare_bytes(raw([-0.0]), raw([0.0]), "<f4", {"kind": "exact"})


def test_ulp_both_signs_and_zero():
    policy = {"kind": "ulp", "max_ulp": 1}
    for dtype in ["<f2", "<f4", "<f8"]:
        a = np.array([-1, -0., 0., 1], dtype)
        b = np.nextafter(a, np.array(np.inf, dtype))
        assert capture.compare_bytes(a.tobytes(), b.tobytes(), dtype, policy) is None
        c = np.nextafter(b, np.array(np.inf, dtype))
        assert capture.compare_bytes(a.tobytes(), c.tobytes(), dtype, policy)


def test_hardware_origin_required(case):
    assert capture.main(["verify", str(case)]) == 0
    assert capture.main(["verify", str(case), "--require-hardware"]) == 1


def test_duplicate_json_key_rejected(case):
    path = case / "capture.json"
    path.write_text(path.read_text().replace('"schema":', '"schema": "wrong", "schema":', 1))
    assert capture.validate_capture_tree(case)


def test_symlink_asset_rejected(case, tmp_path):
    path = case / "x.input.bin"
    target = tmp_path / "external"
    path.rename(target)
    path.symlink_to(target)
    assert capture.validate_capture_tree(case)


def test_frozen_source_contains_tracked_working_bytes(tmp_path):
    import hashlib
    import subprocess
    import tarfile
    from env.remote.freeze_source import freeze
    repo, out = tmp_path / "repo", tmp_path / "snapshot"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init")
    (repo / "kernel.py").write_text("original\n")
    git("add", "kernel.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial")
    (repo / "kernel.py").write_text("edited\n")
    manifest = freeze(repo, out)
    assert manifest["tracked_status"]
    with tarfile.open(out / "source.tar.gz") as archive:
        assert archive.extractfile("warplens/kernel.py").read() == b"edited\n"
    assert manifest["files"]["kernel.py"]["sha256"] == hashlib.sha256(b"edited\n").hexdigest()
    assert b"+edited" in (out / "tracked.diff").read_bytes()
