"""Recover a TIRx host module's launch ABI on the CPU, without recompilation.

The module is the same exported module used for execution. Only CUDA launches are
intercepted; their PTX and input allocations must match the captured hardware case.
This is recipe extraction, never a source of hardware expected outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

from validation.blackwell.probe_process import run_process


def capture_exported(module_path, inputs, ptx):
    with tempfile.TemporaryDirectory(prefix="warplens-host-recipe-") as temporary:
        directory = Path(temporary)
        spec = {"module": str(Path(module_path).resolve()), "inputs": {}}
        for index, (name, array) in enumerate(inputs.items()):
            array = np.ascontiguousarray(array)
            filename = f"input-{index}.bin"
            (directory / filename).write_bytes(array.tobytes())
            spec["inputs"][name] = {"file": filename, "dtype": array.dtype.str, "shape": list(array.shape)}
        (directory / "spec.json").write_text(json.dumps(spec))
        (directory / "expected.ptx").write_text(ptx)
        env = {key: value for key, value in os.environ.items()
               if key not in ("WARPLENS_CAPTURE_DIR", "WARPLENS_SHIM_FD", "LD_PRELOAD")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        # A sim frontend child inherits shim library search paths, but the new CLI owns
        # a fresh socket and installs its own shims for the replay child.
        command = [sys.executable, "-m", "validation.blackwell.record_host", "record", str(directory)]
        status, _, stdout, stderr = run_process(command, 120, env=env)
        if status != "ok":
            raise RuntimeError(f"exported host recipe extraction failed ({status}):\n"
                               f"{stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}")
        return json.loads((directory / "launch.json").read_text())


def _inputs(directory, spec):
    return {name: np.frombuffer((directory / item["file"]).read_bytes(), dtype=item["dtype"]).reshape(item["shape"])
            for name, item in spec["inputs"].items()}


def _record(directory):
    from warplens import cli
    from warplens.host import shim_server as shim
    spec = json.loads((directory / "spec.json").read_text())
    inputs = _inputs(directory, spec)
    expected_ptx = (directory / "expected.ptx").read_text()
    original = shim.ShimServer.handle
    original_parse = shim.parse_ptx
    texts = {}
    descriptors = {}
    launches = []
    def parse(text, *args, **kwargs):
        result = original_parse(text, *args, **kwargs)
        texts[id(result)] = text
        return result
    def handle(server, op, data):
        if op == shim.TENSOR_MAP:
            encoded = original(server, op, data)
            kind, rank, swizzle, fill, interleave, address = struct.unpack_from("<5IQ", data)
            descriptors[encoded] = {"data_type": kind, "rank": rank, "swizzle": swizzle,
                                    "nan_fill": fill, "interleave": interleave, "address": address,
                                    "dims": list(struct.unpack_from("<5Q", data, 28)[:rank]),
                                    "strides": list(struct.unpack_from("<4Q", data, 68)[:rank - 1]),
                                    "box": list(struct.unpack_from("<5I", data, 100)[:rank]),
                                    "steps": list(struct.unpack_from("<5I", data, 120)[:rank])}
            return encoded
        if op != shim.LAUNCH:
            return original(server, op, data)
        function, gx, gy, gz, bx, by, bz, kx, ky, kz, smem = struct.unpack_from("<10IQ", data)
        module, index, entry = server.functions[function]
        parsed = server.modules[module][index]
        if texts[id(parsed)].rstrip("\x00\n") != expected_ptx.rstrip("\x00\n"):
            raise ValueError("exported host launched PTX different from the captured PTX")
        memory = server.device.memory
        bindings = {}
        for name, array in inputs.items():
            matches = [int(base) for base, buf in zip(memory._bases, memory._bufs)
                       if buf.tobytes() == array.tobytes()]
            if len(matches) != 1 or matches[0] in bindings:
                raise ValueError(f"cannot uniquely identify initialized host allocation for {name}")
            bindings[matches[0]] = name
        def pointer(address):
            matches = [(name, address - base) for base, name in bindings.items()
                       if 0 <= address - base < inputs[name].nbytes]
            if len(matches) != 1:
                raise ValueError(f"unmapped pointer {address:#x} in exported host recipe")
            name, offset = matches[0]
            return {"buffer": name, "buffer_offset": offset}
        args, at = [], 48
        for param in parsed.functions[entry].params:
            raw = data[at:at + param.size]
            at += param.size
            if len(raw) != param.size:
                raise ValueError("truncated exported launch arguments")
            if len(raw) == 8 and not param.by_value:
                address = int.from_bytes(raw, "little")
                if any(0 <= address - base < inputs[name].nbytes for base, name in bindings.items()):
                    ref = pointer(address)
                    args.append({"buffer": ref["buffer"], "offset": ref["buffer_offset"]})
                    continue
                # A runtime allocation outside the supplied arrays is scratch state, not
                # a portable scalar. Keeping its transient address would silently make
                # an incomplete recipe that cannot be replayed on another runtime.
                if (param.is_ptr and address) or any(
                        0 <= address - int(base) < len(buf) for base, buf in zip(memory._bases, memory._bufs)):
                    raise ValueError(f"uncaptured allocation pointer {address:#x} in host argument {param.name}")
            arg = {"hex": raw.hex()}
            if param.by_value:
                # These fixtures pass tensor maps directly, not arbitrary C++ objects.
                # Reject an unknown aggregate instead of inventing pointer relocations.
                if raw not in descriptors:
                    raise ValueError(f"unknown {len(raw)}-byte host aggregate {param.name}")
                desc = dict(descriptors[raw])
                desc.update(pointer(desc.pop("address")))
                arg["relocations"] = [{"offset": 0, "tensor_map": desc}]
            args.append(arg)
        if at != len(data):
            raise ValueError("extra exported launch arguments")
        launches.append({"entry": entry, "grid": [gx, gy, gz], "block": [bx, by, bz],
                         "cluster": [kx, ky, kz] if kx else [1, 1, 1], "smem": smem, "args": args})
        return b""  # Recipe extraction: deliberately execute no device instructions.
    shim.parse_ptx = parse
    shim.ShimServer.handle = handle
    try:
        status = cli.main(["run", sys.executable, str(Path(__file__).resolve()), "execute", str(directory)])
    finally:
        shim.parse_ptx = original_parse
        shim.ShimServer.handle = original
    if status or len(launches) != 1:
        raise ValueError(f"expected one successful host launch, got status={status}, launches={len(launches)}")
    (directory / "launch.json").write_text(json.dumps(launches[0], indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "execute"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "record":
        return _record(args.directory)
    import tvm_tirx
    spec = json.loads((args.directory / "spec.json").read_text())
    module = tvm_tirx.runtime.load_module(spec["module"])
    arrays = [tvm_tirx.runtime.tensor(array, device=tvm_tirx.cuda(0))
              for array in _inputs(args.directory, spec).values()]
    module(*arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
