"""Portable, content-checked kernel fixtures. See docs/hardware-capture.md.

No frontend or compiler is needed to verify a bundle or compare a candidate's raw
outputs. Replay uses the saved PTX and launch ABI, never recompiles the source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

import numpy as np

SCHEMA = "warplens.capture.v1"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _name(name):
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name in (".", ".."):
        raise ValueError(f"unsafe capture name: {name!r}")
    return name


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected integer >= {minimum}, got {value!r}")
    return value


def _fields(value, required, optional=()):
    if (not isinstance(value, dict) or not set(required) <= value.keys()
            or value.keys() - set(required) - set(optional)):
        raise ValueError(f"invalid capture fields: expected {sorted(required)} with optional {sorted(optional)}")


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON number {x}")))


def _asset(directory, filename, data):
    path = directory / filename
    path.write_bytes(data)
    return {"file": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _read(directory, asset):
    filename = asset["file"]
    _name(filename)
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked artifact: {path}")
    data = path.read_bytes()
    if len(data) != asset["bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"artifact integrity mismatch: {path}")
    return data


def provenance():
    versions = {}
    for package in ("numpy", "triton", "torch", "apache-tvm-tirx", "nvidia-cutlass-dsl", "cuda-bindings"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[2]
    def command(argv):
        try:
            p = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=10)
            return {"argv": argv, "exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"argv": argv, "error": str(error)}
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "git": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--porcelain"]),
            "nvcc": command([os.environ.get("NVCC", "nvcc"), "--version"]),
            "gpu": command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap", "--format=csv"])}


def save_case(root, name, *, inputs, outputs, ptx, launch, comparison, metadata=None, mode=None):
    """Write a complete case atomically; inputs include initial bytes of output buffers.

    Arrays describe contiguous memory in C order; their original strides are recorded
    separately in metadata if relevant. Launch arguments preserve the actual byte ABI.
    """
    name = _name(name)
    mode = mode or os.environ.get("WARPLENS_CAPTURE_MODE")
    if mode not in ("gpu", "sim"):
        raise ValueError("capture mode must explicitly be gpu or sim")
    if not inputs or not outputs or not outputs.keys() <= inputs.keys():
        raise ValueError("every output must have captured initial buffer bytes")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise ValueError(f"capture already exists (use a fresh run directory): {target}")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        directory = Path(temporary)
        buffers = {}
        for key, original in inputs.items():
            _name(key)
            array = np.asarray(original)
            if array.dtype.kind not in "biuf" or array.dtype.itemsize not in (1, 2, 4, 8):
                raise ValueError(f"unsupported portable dtype: {array.dtype}")
            dtype = array.dtype.newbyteorder("<")
            shape = array.shape
            array = np.ascontiguousarray(array.astype(dtype, copy=False)).reshape(shape)
            item = {"dtype": dtype.str, "shape": list(array.shape),
                    "initial": _asset(directory, f"{key}.input.bin", array.tobytes())}
            if key in outputs:
                expected = np.asarray(outputs[key])
                if expected.shape != array.shape or expected.dtype.newbyteorder("<") != dtype:
                    raise ValueError(f"output buffer shape/dtype changed: {key}")
                item["expected"] = _asset(directory, f"{key}.output.bin",
                                           np.ascontiguousarray(expected.astype(dtype, copy=False)).tobytes())
                item["comparison"] = comparison
            buffers[key] = item
        document = {"schema": SCHEMA, "case": name, "mode": mode,
                    "byte_order": "little", "buffers": buffers,
                    "ptx": _asset(directory, "kernel.ptx", ptx.encode("utf-8")),
                    "launch": launch, "metadata": metadata or {}, "provenance": provenance()}
        (directory / "capture.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        load_case(directory)
        directory.rename(target)
    return target


def capture_case(name, **kwargs):
    root = os.environ.get("WARPLENS_CAPTURE_DIR")
    return save_case(root, name, **kwargs) if root else None


def _policy(policy, dtype):
    kind = policy["kind"]
    if kind == "exact":
        return
    if dtype.kind != "f":
        raise ValueError("numerical comparisons require floating-point buffers")
    if kind == "allclose":
        for key in ("rtol", "atol"):
            value = policy[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid comparison {key}: {value}")
    elif kind == "ulp":
        _integer(policy["max_ulp"])
    else:
        raise ValueError(f"unknown comparison policy: {kind}")


def load_case(directory):
    directory = Path(directory)
    document = _json(directory / "capture.json")
    if document["schema"] != SCHEMA or document["byte_order"] != "little":
        raise ValueError("unsupported capture schema or byte order")
    _name(document["case"])
    if document["mode"] not in ("gpu", "sim") or not document.get("provenance"):
        raise ValueError("missing capture origin")
    ptx = _read(directory, document["ptx"]).decode("utf-8")
    if ".version" not in ptx or ".entry" not in ptx:
        raise ValueError("missing executable PTX")
    buffers = document["buffers"]
    if not buffers or not any("expected" in b for b in buffers.values()):
        raise ValueError("capture has no input/output buffers")
    for name, item in buffers.items():
        _name(name)
        dtype = np.dtype(item["dtype"])
        if dtype.kind not in "biuf" or dtype.itemsize not in (1, 2, 4, 8) or dtype.str[0] == ">":
            raise ValueError(f"unsupported dtype: {dtype}")
        size = math.prod(_integer(n) for n in item["shape"]) * dtype.itemsize
        if len(_read(directory, item["initial"])) != size:
            raise ValueError(f"initial buffer size does not match shape: {name}")
        if "expected" in item:
            if len(_read(directory, item["expected"])) != size:
                raise ValueError(f"output buffer size does not match shape: {name}")
            _policy(item["comparison"], dtype)
    launch = document["launch"]
    if not isinstance(launch["entry"], str) or not re.search(r"\.entry\s+" + re.escape(launch["entry"]) + r"\s*\(", ptx):
        raise ValueError("launch entry not present in captured PTX")
    for kind in ("grid", "block", "cluster"):
        if not isinstance(launch[kind], list) or len(launch[kind]) != 3:
            raise ValueError(f"{kind} must have three dimensions")
        for n in launch[kind]:
            _integer(n, 1)
    if any(g % c for g, c in zip(launch["grid"], launch["cluster"])):
        raise ValueError("grid does not contain whole clusters")
    _integer(launch["smem"])
    def reference(ref):
        if ref["buffer"] not in buffers:
            raise ValueError(f"unknown buffer {ref['buffer']}")
        offset = _integer(ref.get("buffer_offset", ref.get("offset", 0)))
        if offset >= buffers[ref["buffer"]]["initial"]["bytes"]:
            raise ValueError("pointer is outside its captured buffer")
        return offset
    from warplens.frontend.ptx_parser import parse_ptx
    from warplens.errors import WarplensError
    try:
        params = parse_ptx(ptx).functions[launch["entry"]].params
    except WarplensError as error:
        raise ValueError(f"invalid captured PTX: {error}") from error
    if len(params) != len(launch["args"]):
        raise ValueError("launch argument count differs from captured PTX ABI")
    for param, arg in zip(params, launch["args"]):
        if "buffer" in arg:
            _fields(arg, ("buffer",), ("offset",))
        else:
            _fields(arg, ("hex",), ("relocations",))
        size = 8 if "buffer" in arg else len(bytes.fromhex(arg["hex"]))
        if size != param.size:
            raise ValueError(f"launch argument size differs from captured PTX ABI: {param.name}")
        if "buffer" in arg:
            reference(arg)
        else:
            raw = bytes.fromhex(arg["hex"])
            occupied = set()
            for fixup in arg.get("relocations", []):
                if "tensor_map" in fixup:
                    _fields(fixup, ("offset", "tensor_map"))
                    _fields(fixup["tensor_map"], ("buffer", "data_type", "rank", "dims", "strides", "box",
                                                  "steps", "swizzle", "nan_fill", "interleave"), ("buffer_offset",))
                else:
                    _fields(fixup, ("offset", "buffer"), ("buffer_offset",))
                offset = _integer(fixup["offset"])
                size = 128 if "tensor_map" in fixup else 8
                if offset + size > len(raw) or occupied.intersection(range(offset, offset + size)):
                    raise ValueError("overlapping or out-of-range argument relocation")
                occupied.update(range(offset, offset + size))
                ref = fixup.get("tensor_map", fixup)
                buffer_offset = reference({**ref, "offset": ref.get("buffer_offset", 0)})
                if "tensor_map" in fixup:
                    from warplens.host.tensor_map import from_driver
                    kwargs = {k: v for k, v in ref.items() if k not in ("buffer", "buffer_offset")}
                    try:
                        info = from_driver(address=buffer_offset, **kwargs)
                    except WarplensError as error:
                        raise ValueError(f"invalid tensor-map recipe: {error}") from error
                    extent = info.elem_size + sum((dim - 1) * stride for dim, stride in
                                                  zip(info.dims, (info.elem_size, *info.strides)))
                    if buffer_offset + extent > buffers[ref["buffer"]]["initial"]["bytes"]:
                        raise ValueError("tensor-map geometry exceeds its captured buffer")
            if param.is_ptr and any(raw) and not any(
                    f.get("offset") == 0 and "buffer" in f for f in arg.get("relocations", [])):
                raise ValueError(f"PTX pointer argument has no relocation: {param.name}")
    return document


def validate_capture_tree(root, expected_cases=None):
    root = Path(root)
    errors = []
    files = sorted(root.rglob("capture.json")) if root.exists() else []
    if not files:
        return [f"no complete captures under {root}"]
    names = []
    for file in files:
        try:
            case = load_case(file.parent)
            names.append(case["case"])
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f"{file}: {error}")
    if len(set(names)) != len(names):
        errors.append("duplicate capture case names")
    if expected_cases is not None and set(names) != set(expected_cases):
        errors.append(f"capture cases differ: expected {sorted(expected_cases)}, got {sorted(names)}")
    return errors


def compare_bytes(actual, expected, dtype, policy):
    """None on a match, otherwise an error string. No implicit NaN equality."""
    if len(actual) != len(expected):
        return f"byte count differs: {len(actual)} != {len(expected)}"
    if policy["kind"] == "exact":
        return None if actual == expected else "bit patterns differ"
    a, b = (np.frombuffer(raw, dtype=dtype) for raw in (actual, expected))
    # Non-finite arithmetic is a separate bit-pattern probe, not a tolerance pass.
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return "non-finite value in numerical comparison"
    if policy["kind"] == "allclose":
        matched = np.isclose(a.astype(np.float64), b.astype(np.float64),
                             rtol=policy["rtol"], atol=policy["atol"])
    else:
        bits = np.dtype(dtype).itemsize * 8
        unsigned = np.dtype(f"<u{bits // 8}")
        def ordered(raw):
            words = np.frombuffer(raw, dtype=unsigned).astype(object)
            sign = 1 << (bits - 1)
            return np.array([sign - (int(x) & (sign - 1)) if int(x) & sign else sign + int(x)
                             for x in words], dtype=object)
        matched = np.abs(ordered(actual) - ordered(expected)) <= policy["max_ulp"]
    if bool(np.all(matched)):
        return None
    error = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) if a.size else 0
    return f"{np.count_nonzero(~matched)}/{a.size} elements differ; max absolute error {error:.9g}"


def compare_outputs(directory, candidate):
    directory, candidate = Path(directory), Path(candidate)
    case = load_case(directory)
    errors = []
    for name, item in case["buffers"].items():
        if "expected" not in item:
            continue
        path = candidate / item["expected"]["file"]
        if not path.is_file():
            errors.append(f"{name}: missing candidate {path}")
            continue
        error = compare_bytes(path.read_bytes(), _read(directory, item["expected"]),
                              item["dtype"], item["comparison"])
        if error:
            errors.append(f"{name}: {error}")
    return errors


def replay_case(directory, output, *, model="B200", max_steps=10_000_000):
    """Execute a captured launch without importing its original frontend/compiler."""
    from warplens import Device
    from warplens.host import tensor_map
    directory, output = Path(directory), Path(output)
    case = load_case(directory)
    device = Device(model)
    arrays = {}
    for name, item in case["buffers"].items():
        value = np.frombuffer(_read(directory, item["initial"]), dtype=item["dtype"]).reshape(item["shape"])
        arrays[name] = device.from_numpy(value)
    arguments = []
    for arg in case["launch"]["args"]:
        if "buffer" in arg:
            arguments.append((arrays[arg["buffer"]].addr + arg.get("offset", 0)).to_bytes(8, "little"))
            continue
        raw = bytearray.fromhex(arg["hex"])
        for fixup in arg.get("relocations", []):
            offset = fixup["offset"]
            if "tensor_map" in fixup:
                spec = dict(fixup["tensor_map"])
                address = arrays[spec.pop("buffer")].addr + spec.pop("buffer_offset", 0)
                value = tensor_map.encode(tensor_map.from_driver(address=address, **spec))
            else:
                value = (arrays[fixup["buffer"]].addr + fixup.get("buffer_offset", 0)).to_bytes(8, "little")
            raw[offset:offset + len(value)] = value
        arguments.append(bytes(raw))
    launch = case["launch"]
    stats = device.launch(directory / case["ptx"]["file"], launch["entry"], args=arguments,
                          grid=launch["grid"], block=launch["block"], cluster=launch["cluster"],
                          smem=launch["smem"], max_steps=max_steps)
    output.mkdir(parents=True, exist_ok=True)
    for name, item in case["buffers"].items():
        if "expected" in item:
            (output / item["expected"]["file"]).write_bytes(arrays[name].numpy().tobytes())
    (output / "replay.txt").write_text(str(stats) + "\n")
    return compare_outputs(directory, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "replay", "compare"))
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, help="replay output or candidate output directory")
    parser.add_argument("--require-hardware", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_capture_tree(args.capture)
    files = sorted(args.capture.rglob("capture.json"))
    if args.require_hardware:
        for file in files:
            try:
                if load_case(file.parent)["mode"] != "gpu":
                    errors.append(f"{file}: simulator capture, not hardware evidence")
            except (ValueError, KeyError, TypeError, OSError):
                pass  # reported above
    if args.action != "verify" and args.out is None:
        parser.error("replay/compare require --out")
    if not errors and args.action != "verify":
        for file in files:
            relative = file.parent.relative_to(args.capture)
            try:
                action = replay_case if args.action == "replay" else compare_outputs
                errors.extend(f"{relative}: {error}" for error in action(file.parent, args.out / relative))
            except Exception as error:
                errors.append(f"{relative}: {type(error).__name__}: {error}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{args.action}: {len(files)} case(s), {len(errors)} error(s)")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
