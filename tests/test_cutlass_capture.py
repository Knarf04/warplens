"""CUTLASS capture recipes must preserve the actual ABI, not assume struct offsets."""
import json
import struct

import numpy as np
import pytest

from validation.blackwell.e2e.capture_cutlass import (
    digest, finalize, instrument_source, instrument_tma_header, launch_recipe, unique_offset,
)


def raw_launch(tmp_path, tutorial):
    pointers = dict(zip("abcd", (0x111100000, 0x222200000, 0x333300000, 0x444400000)))
    args = [{"hex": "00"} for _ in range(9 if tutorial == 1 else 11)]
    for index, buffer in ({0: "a", 1: "b", 2: "c", 3: "d"} if tutorial == 1 else {2: "c", 3: "d"}).items():
        args[index] = {"hex": (bytes(16) + struct.pack("<Q", pointers[buffer])).hex()}
    maps, bindings = [], []
    if tutorial != 1:
        for index, buffer in ((7, "a"), (8, "b")):
            desc = bytes([index]) * 128
            # A nonzero offset verifies that capture uses get_tma_descriptor(), not an ABI guess.
            args[index] = {"hex": (bytes(16) + desc + bytes(112)).hex()}
            bindings.append({"argument": index, "offset": 16, "hex": desc.hex()})
            maps.append({"hex": desc.hex(), "address": pointers[buffer], "data_type": 6,
                         "rank": 2, "dims": [64, 128], "strides": [128], "box": [64, 32],
                         "steps": [1, 1], "swizzle": 3, "nan_fill": False, "interleave": 0,
                         "l2_promotion": 2})
        (tmp_path / "tensor_maps.jsonl").write_text("\n".join(map(json.dumps, maps)))
        (tmp_path / "bindings.jsonl").write_text("\n".join(map(json.dumps, bindings)))
    launch = {"grid": [4, 4, 1], "block": [128, 1, 1], "cluster": [4, 4, 1], "smem": 49152, "args": args}
    (tmp_path / "launch.json").write_text(json.dumps(launch))
    return pointers, launch, maps, bindings


@pytest.mark.parametrize("tutorial", [1, 2, 3, 4])
def test_launch_recipe_preserves_pack_and_explicit_relocations(tmp_path, tutorial):
    pointers, original, maps, _ = raw_launch(tmp_path, tutorial)
    launch, captured_maps = launch_recipe(tmp_path, tutorial, pointers)
    assert [arg["hex"] for arg in launch["args"]] == [arg["hex"] for arg in original["args"]]
    assert launch["args"][2]["relocations"] == [{"offset": 16, "buffer": "c"}]
    assert captured_maps == maps
    if tutorial != 1:
        rel = launch["args"][7]["relocations"][0]
        assert rel["offset"] == 16
        assert rel["tensor_map"] == {"buffer": "a", "data_type": 6, "rank": 2, "dims": [64, 128],
                                     "strides": [128], "box": [64, 32], "steps": [1, 1],
                                     "swizzle": 3, "nan_fill": False, "interleave": 0}


@pytest.mark.parametrize("defect", ["wrong_offset", "wrong_pointer", "missing_recipe", "duplicate_binding", "duplicate_recipe"])
def test_tensor_map_binding_must_match_actual_argument(tmp_path, defect):
    pointers, _, maps, bindings = raw_launch(tmp_path, 4)
    if defect == "wrong_offset":
        bindings[0]["offset"] = 0
    elif defect == "wrong_pointer":
        maps[0]["address"] = pointers["c"]
    elif defect == "missing_recipe":
        maps[0]["hex"] = "00" * 128
    elif defect == "duplicate_binding":
        bindings[1] = bindings[0]
    else:
        maps[1] = maps[0]
    (tmp_path / "tensor_maps.jsonl").write_text("\n".join(map(json.dumps, maps)))
    (tmp_path / "bindings.jsonl").write_text("\n".join(map(json.dumps, bindings)))
    with pytest.raises(ValueError):
        launch_recipe(tmp_path, 4, pointers)


def test_pointer_relocation_rejects_ambiguous_or_absent_values():
    pointer = struct.pack("<Q", 0x1234567890)
    assert unique_offset(b"padding" + pointer, pointer) == 7
    with pytest.raises(ValueError, match="found 2"):
        unique_offset(pointer + pointer, pointer)
    with pytest.raises(ValueError, match="found 0"):
        unique_offset(bytes(24), pointer)


@pytest.mark.parametrize("tutorial", [1, 2, 3, 4])
def test_instrumentation_only_changes_known_host_anchors(tutorial):
    kernel = "__global__ void kernel() { /* unchanged device body */ }\n"
    source = kernel + """  // Setup input and output tensors, and the kernel parameters; and execute the kernel on device
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, ptr, a);
  bool success = relative_error <= 0.0;
"""
    changed = instrument_source(source, tutorial)
    assert kernel in changed
    assert "capture_initial_D = device_D" in changed
    assert "wl_capture::result" in changed
    assert "if (!success) return 1" in changed
    assert ("wl_capture::binding(7" in changed) == (tutorial != 1)
    with pytest.raises(ValueError, match="anchor changed"):
        instrument_source(source + source, tutorial)


def test_descriptor_hook_runs_after_driver_and_cutlass_descriptor_adjustments():
    source = "driver_encode();\nadjust_descriptor();\n  #endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)\n"
    changed = instrument_tma_header(source)
    assert changed.index("adjust_descriptor") < changed.index("wl_capture::tensor_map")
    with pytest.raises(ValueError, match="anchor changed"):
        instrument_tma_header("unknown revision")


def finalization_fixture(tmp_path, monkeypatch):
    from validation.blackwell import provenance
    raw, build = tmp_path / "raw", tmp_path / "build"
    raw.mkdir()
    (build / "bin").mkdir(parents=True)
    pointers, launch, _, _ = raw_launch(raw, 1)
    (raw / "buffers.json").write_text(json.dumps({"m": 2, "n": 3, "k": 4, "pointers": pointers}))
    for name, shape in (("a", (2, 4)), ("b", (3, 4)), ("c", (2, 3)), ("d", (2, 3))):
        value = np.arange(np.prod(shape), dtype="<f2" if name in "ab" else "<f4")
        (raw / f"{name}.input.bin").write_bytes(value.tobytes())
    expected = np.arange(6, dtype="<f4").tobytes()
    (raw / "d.output.bin").write_bytes(expected)
    (raw / "d.reference.bin").write_bytes(expected)
    (raw / "result.json").write_text('{"reference_passed":true}')
    (build / "bin/tutorial_01").write_bytes(b"mock executable")
    (build / "build.json").write_text(json.dumps({"tutorials": {
        "tutorial_01": {"binary_sha256": digest(b"mock executable")}}}))
    params = ",\n".join(f".param .align 1 .b8 p{i}[{len(bytes.fromhex(arg['hex']))}]"
                        for i, arg in enumerate(launch["args"]))
    ptx = f".version 8.7\n.target sm_100a\n.address_size 64\n.visible .entry gemm_device_capture({params}) {{ ret; }}\n"
    monkeypatch.setattr(provenance, "extracted_ptx", lambda _: [ptx])
    return raw, build, ptx


def test_finalize_saves_full_tensors_and_executable_abi(tmp_path, monkeypatch):
    from validation.blackwell.capture import load_case
    raw, build, ptx = finalization_fixture(tmp_path, monkeypatch)
    output = finalize(raw, build, 1, tmp_path / "captures", "sim")
    case = load_case(output)
    assert case["mode"] == "sim"
    assert case["case"] == "cutlass_tutorial_01"
    assert (output / "kernel.ptx").read_text() == ptx
    for name in "abcd":
        assert (output / f"{name}.input.bin").read_bytes() == (raw / f"{name}.input.bin").read_bytes()
    assert (output / "d.output.bin").read_bytes() == (raw / "d.output.bin").read_bytes()
    assert case["launch"]["args"][3]["relocations"] == [{"offset": 16, "buffer": "d"}]


@pytest.mark.parametrize("defect", ["missing_initial_d", "short_output", "reference_failed", "wrong_output", "abi_mismatch", "binary_changed"])
def test_finalize_refuses_incomplete_or_unverified_capture(tmp_path, monkeypatch, defect):
    from validation.blackwell import provenance
    raw, build, ptx = finalization_fixture(tmp_path, monkeypatch)
    if defect == "missing_initial_d":
        (raw / "d.input.bin").unlink()
    elif defect == "short_output":
        (raw / "d.output.bin").write_bytes(b"bad")
    elif defect == "reference_failed":
        (raw / "result.json").write_text('{"reference_passed":false}')
    elif defect == "wrong_output":
        (raw / "d.output.bin").write_bytes(bytes(24))
    elif defect == "abi_mismatch":
        monkeypatch.setattr(provenance, "extracted_ptx", lambda _: [ptx.replace("p0[24]", "p0[32]")])
    else:
        (build / "bin/tutorial_01").write_bytes(b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize(raw, build, 1, tmp_path / "captures", "sim")
    assert not (tmp_path / "captures/cutlass_tutorial_01").exists()
