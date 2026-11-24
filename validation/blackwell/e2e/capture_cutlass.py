#!/usr/bin/env python3
"""Build host-instrumented copies of the four pinned CUTLASS tutorials and finalize captures.

build --cutlass CLONE --out BUILD creates BUILD/bin/tutorial_01..04. Run each executable
with WARPLENS_CAPTURE_RAW_DIR set to its own empty directory, then call finalize. Device
kernel bodies and the original CUTLASS checkout are untouched. Only the GEMM launch is
recorded; the buffers include the effects of preceding Thrust initialization kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
PIN = "f8acf1a"
TUTORIALS = ("01_mma_sm100", "02_mma_tma_sm100", "03_mma_tma_multicast_sm100", "04_mma_tma_2sm_sm100")
TMA_HEADER = Path("include/cute/atom/copy_traits_sm90_tma.hpp")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"pinned CUTLASS capture anchor changed: {old[:80]!r}")
    return source.replace(old, new, 1)


def instrument_source(source: str, tutorial: int) -> str:
    source = '#include "capture_cutlass.hpp"\n' + source
    anchor = "  // Setup input and output tensors, and the kernel parameters; and execute the kernel on device"
    source = replace_once(source, anchor, """  thrust::host_vector<TypeD> capture_initial_D = device_D;
  wl_capture::buffers(Gemm_M, Gemm_N, Gemm_K, host_A.data(), host_B.data(), host_C.data(),
                      capture_initial_D.data(), device_A.data().get(), device_B.data().get(),
                      device_C.data().get(), device_D.data().get());
""" + anchor)
    anchor = "  cutlass::Status status = cutlass::launch_kernel_on_cluster("
    bindings = "" if tutorial == 1 else "  wl_capture::binding(7, tma_atom_A);\n  wl_capture::binding(8, tma_atom_B);\n"
    source = replace_once(source, anchor, bindings + "  cutlass::Status status = wl_capture::launch(")
    anchor = "  bool success = relative_error <= 0.0;"
    return replace_once(source, anchor, anchor + """
  wl_capture::result(host_D.data(), host_reference_D.data(), host_D.size(), success);
  if (!success) return 1;""")


def instrument_tma_header(source: str) -> str:
    anchor = "  #endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)"
    hook = """    wl_capture::tensor_map(&tma_desc, int(tma_format), tma_dim, gmem_address,
        gmem_prob_shape.data(), gmem_prob_stride.data() + 1, smem_box_shape.data(),
        smem_box_stride.data(), int(tma_interleave), int(smem_swizzle), int(tma_l2Promotion),
        int(tma_oobFill), int(result));
"""
    return replace_once(source, anchor, hook + anchor)


def prepare_sources(cutlass: Path, out: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip()
    if not commit.startswith(PIN):
        raise ValueError(f"capture requires CUTLASS {PIN}, found {commit}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bin").mkdir(exist_ok=True)
    shutil.copyfile(HERE / "capture_cutlass.hpp", out / "capture_cutlass.hpp")
    originals = {}
    for n, name in enumerate(TUTORIALS, 1):
        path = cutlass / "examples/cute/tutorial/blackwell" / f"{name}.cu"
        raw = path.read_bytes()
        originals[str(path.relative_to(cutlass))] = digest(raw)
        (out / f"tutorial_0{n}.cu").write_text(instrument_source(raw.decode(), n))
    raw = (cutlass / TMA_HEADER).read_bytes()
    originals[str(TMA_HEADER)] = digest(raw)
    overlay = out / "overlay" / TMA_HEADER.relative_to("include")
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(instrument_tma_header(raw.decode()))
    return {"cutlass_commit": commit, "upstream_source_sha256": originals,
            "capture_header_sha256": digest((out / "capture_cutlass.hpp").read_bytes()),
            "tma_overlay_sha256": digest(overlay.read_bytes())}


def build(cutlass: Path, out: Path, nvcc: str) -> None:
    cutlass, out = cutlass.resolve(), out.resolve()
    metadata = prepare_sources(cutlass, out)
    metadata["nvcc_version"] = subprocess.check_output([nvcc, "--version"], text=True)
    includes = [out / "overlay", out, cutlass / "include", cutlass / "tools/util/include",
                cutlass / "examples/common", cutlass / "examples/cute/tutorial/blackwell"]
    flags = ["-gencode", "arch=compute_100a,code=compute_100a", "--no-compress", "--cudart", "shared",
             "-std=c++17", "-O3", "-DNDEBUG", "-w"]
    metadata["compile_flags"] = flags
    metadata["tutorials"] = {}
    for n in range(1, 5):
        name = f"tutorial_0{n}"
        source, binary = out / f"{name}.cu", out / "bin" / name
        command = [nvcc, *flags, *(f"-I{x}" for x in includes), str(source), "-o", str(binary)]
        subprocess.run(command, check=True, timeout=900)
        metadata["tutorials"][name] = {"instrumented_source_sha256": digest(source.read_bytes()),
                                        "binary_sha256": digest(binary.read_bytes())}
    (out / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def unique_offset(haystack: bytes, needle: bytes) -> int:
    offsets = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i:i + len(needle)] == needle]
    if len(offsets) != 1:
        raise ValueError(f"expected one pointer relocation, found {len(offsets)}")
    return offsets[0]


def launch_recipe(raw: Path, tutorial: int, pointers: dict[str, int]) -> tuple[dict, list[dict]]:
    launch = json.loads((raw / "launch.json").read_text())
    args = launch["args"]
    if len(args) != (9 if tutorial == 1 else 11):
        raise ValueError("unexpected CUTLASS argument pack")
    pointer_args = {0: "a", 1: "b", 2: "c", 3: "d"} if tutorial == 1 else {2: "c", 3: "d"}
    for idx, buffer in pointer_args.items():
        offset = unique_offset(bytes.fromhex(args[idx]["hex"]), struct.pack("<Q", pointers[buffer]))
        args[idx]["relocations"] = [{"offset": offset, "buffer": buffer}]
    maps, bindings = json_lines(raw / "tensor_maps.jsonl"), json_lines(raw / "bindings.jsonl")
    if tutorial == 1:
        if maps or bindings:
            raise ValueError("tutorial 01 unexpectedly captured tensor maps")
    else:
        if len(maps) != 2 or len(bindings) != 2 or {b["argument"] for b in bindings} != {7, 8}:
            raise ValueError("expected exactly two tensor-map captures and argument bindings")
        for binding in bindings:
            idx, offset = binding["argument"], binding["offset"]
            desc = bytes.fromhex(binding["hex"])
            pack = bytes.fromhex(args[idx]["hex"])
            if len(desc) != 128 or offset < 0 or pack[offset:offset + 128] != desc:
                raise ValueError("tensor-map binding does not match the launch argument bytes")
            matches = [m for m in maps if m["hex"] == binding["hex"]]
            if len(matches) != 1:
                raise ValueError("tensor-map argument has no unique encoder recipe")
            encoded = matches[0]
            buffer = "a" if idx == 7 else "b"
            if encoded["address"] != pointers[buffer]:
                raise ValueError("tensor-map pointer does not match its input buffer")
            fields = ("data_type", "rank", "dims", "strides", "box", "steps", "swizzle", "nan_fill", "interleave")
            recipe = {"buffer": buffer, **{k: encoded[k] for k in fields}}
            args[idx]["relocations"] = [{"offset": offset, "tensor_map": recipe}]
    return launch, maps


def finalize(raw: Path, build_dir: Path, tutorial: int, out: Path, mode: str = "gpu") -> Path:
    from validation.blackwell.capture import save_case
    from validation.blackwell.provenance import extracted_ptx
    from warplens.frontend.ptx_parser import parse_ptx

    name = f"tutorial_0{tutorial}"
    metadata = json.loads((build_dir / "build.json").read_text())
    binary = build_dir / "bin" / name
    if digest(binary.read_bytes()) != metadata["tutorials"][name]["binary_sha256"]:
        raise ValueError("capture executable changed since build")
    buffers = json.loads((raw / "buffers.json").read_text())
    m, n, k = (buffers[key] for key in ("m", "n", "k"))
    if min(m, n, k) <= 0:
        raise ValueError("captured matrix dimensions must be positive")
    shapes = {"a": (m, k), "b": (n, k), "c": (m, n), "d": (m, n)}
    def load(key: str, suffix: str) -> np.ndarray:
        dtype = np.dtype("<f2" if key in "ab" else "<f4")
        data = (raw / f"{key}.{suffix}.bin").read_bytes()
        if len(data) != int(np.prod(shapes[key])) * dtype.itemsize:
            raise ValueError(f"truncated or oversized {key}.{suffix} tensor")
        return np.frombuffer(data, dtype=dtype).reshape(shapes[key])
    inputs = {key: load(key, "input") for key in shapes}
    result, reference = load("d", "output"), load("d", "reference")
    if not json.loads((raw / "result.json").read_text())["reference_passed"]:
        raise ValueError("CUTLASS tutorial failed its own reference comparison")
    if not np.array_equal(result, reference):
        raise ValueError("CUTLASS full output does not exactly match its saved integer-input reference")
    launch, maps = launch_recipe(raw, tutorial, buffers["pointers"])
    ptx_records = extracted_ptx(binary)
    entries = [(ptx, f) for ptx in ptx_records for f in parse_ptx(ptx).functions.values()
               if "gemm_device" in f.name]
    if len(entries) != 1:
        raise ValueError("expected one GEMM PTX entry in the captured executable")
    ptx, function = entries[0]
    if [p.size for p in function.params] != [len(bytes.fromhex(a["hex"])) for a in launch["args"]]:
        raise ValueError("captured launch argument sizes do not match the executable PTX ABI")
    launch["entry"] = function.name
    metadata.update({"frontend": "cutlass", "tutorial": tutorial, "capture_mode": mode,
                     "embedded_ptx_sha256": [digest(p.encode()) for p in ptx_records],
                     "original_device_pointers": buffers["pointers"], "tensor_map_encodings": maps,
                     "reference_sha256": digest((raw / "d.reference.bin").read_bytes()),
                     "scope": "one GEMM launch; exact buffer prestate includes Thrust initialization"})
    return save_case(out, f"cutlass_{name}", inputs=inputs, outputs={"d": result}, ptx=ptx,
                     launch=launch, comparison={"kind": "exact"}, metadata=metadata, mode=mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    b = subs.add_parser("build")
    b.add_argument("--cutlass", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    f = subs.add_parser("finalize")
    f.add_argument("--raw", type=Path, required=True)
    f.add_argument("--build", type=Path, required=True)
    f.add_argument("--tutorial", type=int, choices=range(1, 5), required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--mode", choices=("gpu", "sim"), default=os.environ.get("WARPLENS_CAPTURE_MODE", "gpu"))
    args = parser.parse_args()
    if args.command == "build":
        build(args.cutlass, args.out, args.nvcc)
    else:
        finalize(args.raw, args.build, args.tutorial, args.out, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
