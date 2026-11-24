"""Opt-in capture adapters for the fixed Python frontend validation examples.

The compiled objects, not regenerated source, supply PTX and launch metadata.
No optional frontend is imported unless its adapter is actually used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from .capture import capture_case


def enabled():
    return bool(os.environ.get("WARPLENS_CAPTURE_DIR"))


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _raw_case(name, *, inputs, outputs, ptx, metadata):
    """Preserve execution evidence before attempting portable ABI interpretation."""
    from .capture import provenance
    directory = Path(os.environ["WARPLENS_CAPTURE_DIR"]) / "raw" / name
    directory.mkdir(parents=True, exist_ok=False)
    if isinstance(ptx, str):
        (directory / "kernel.ptx").write_text(ptx)
    for key, value in inputs.items():
        np.save(directory / f"{key}.input.npy", value, allow_pickle=False)
    for key, value in outputs.items():
        np.save(directory / f"{key}.output.npy", value, allow_pickle=False)
    details = {"mode": os.environ.get("WARPLENS_CAPTURE_MODE"),
               "metadata": _jsonable(metadata), "provenance": provenance()}
    (directory / "metadata.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    return directory


def _params(ptx, entry):
    from warplens import parse_ptx
    return parse_ptx(ptx).functions[entry].params


def _scalar(value, param):
    if param.by_value:
        raise ValueError("unexpected aggregate in scalar frontend launch ABI")
    if param.type.startswith("f"):
        return {"hex": struct.pack({2: "<e", 4: "<f", 8: "<d"}[param.size], value).hex()}
    if not isinstance(value, (int, np.integer, bool)):
        raise ValueError(f"expected integer for {param.name}: {value!r}")
    return {"hex": (int(value) % (1 << (8 * param.size))).to_bytes(param.size, "little").hex()}


def triton_case(name, compiled, *, values, pointers, grid, inputs, outputs, comparison):
    """Record the actual Triton compiled ABI; pointer names refer to captured buffers.

    This adapter handles the scalar/pointer kernels in examples/05_triton. Tensor
    descriptors and scratch allocations must get an explicit adapter before use.
    """
    if not enabled():
        return None
    metadata = compiled.metadata
    ptx = compiled.asm.get("ptx")
    details = {"frontend": "triton", "recipe": "compiled kernel metadata",
               "compiler_metadata": _jsonable(metadata._asdict()),
               "signature": _jsonable(compiled.src.signature),
               "constants": _jsonable(compiled.src.constants),
               "attributes": _jsonable(compiled.src.attrs)}
    _raw_case(name, inputs=inputs, outputs=outputs, ptx=ptx, metadata=details)
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise ValueError("capture of Triton scratch allocations is not implemented")
    if not isinstance(ptx, str):
        raise ValueError("compiled Triton kernel contains no PTX")
    params = _params(ptx, metadata.name)
    arguments = []
    for key, kind in compiled.src.signature.items():
        if kind == "constexpr":
            continue
        if not isinstance(kind, str) or kind.startswith(("tensordesc", "nvTmaDesc")):
            raise ValueError(f"unsupported Triton capture argument: {key}={kind}")
        if kind.startswith("*"):
            arguments.append({"buffer": pointers[key], "offset": 0})
        else:
            arguments.append(values[key])
    arguments += [0, 0]  # Triton 3.9 global/profile scratch pointers, both unused.
    if len(arguments) != len(params):
        raise ValueError("compiled Triton PTX parameter count differs from launch signature")
    args = []
    for arg, param in zip(arguments, params):
        if isinstance(arg, dict):
            if param.size != 8 or param.by_value:
                raise ValueError("compiled Triton pointer has an unexpected ABI")
            args.append(arg)
        else:
            args.append(_scalar(arg, param))
    grid = list(grid) + [1] * (3 - len(grid))
    grid[0] *= metadata.num_ctas
    return capture_case(name, inputs=inputs, outputs=outputs, ptx=ptx,
                        launch={"entry": metadata.name, "grid": grid,
                                "block": [metadata.num_warps * 32, 1, 1],
                                "cluster": [metadata.num_ctas, 1, 1],
                                "smem": metadata.shared, "args": args},
                        comparison=comparison,
                        metadata=details)


def tvm_ptx(module):
    """Find the one embedded CUDA PTX module used by these single-kernel examples."""
    found = []
    def visit(current):
        if current.kind == "cuda":
            text = current.inspect_source("ptx")
            if not text or ".entry" not in text:
                raise ValueError("compiled TVM CUDA module contains no PTX")
            found.append(text.rstrip("\x00"))
        for child in current.imports:
            visit(child)
    visit(module)
    if len(found) != 1:
        raise ValueError(f"expected one compiled TVM CUDA module, got {len(found)}")
    return found[0]


def tvm_case(name, executable, *, inputs, outputs, comparison, metadata=None):
    if not enabled():
        return None
    from .record_host import capture_exported
    ptx = tvm_ptx(executable.mod)
    # Keep native results even if portable recipe extraction fails. The complete
    # capture is published separately only after that extraction is validated.
    details = {"frontend": "tirx", "recipe": "replayed exported host module",
               **(metadata or {})}
    directory = _raw_case(name, inputs=inputs, outputs=outputs, ptx=ptx, metadata=details)
    library = directory / "host.so"
    executable.export_library(str(library))
    details["host_module_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    launch = capture_exported(library, inputs, ptx)
    return capture_case(name, inputs=inputs, outputs=outputs, ptx=ptx,
                        launch=launch, comparison=comparison, metadata=details)


def cute_saxpy_case(compiled, *, scalar, grid, block, inputs, output):
    """The fixed CuTe SAXPY ABI: f32, one-pointer tensor, one-pointer tensor.

    Shapes are compile-time constants in this example. Reject a compiler ABI change
    instead of assuming arbitrary CuTe tensor structures have the same layout.
    """
    if not enabled():
        return None
    ptx = compiled.__ptx__
    details = {"frontend": "cute_dsl", "recipe": "fixed SAXPY host launch",
               "kernel_info": _jsonable(compiled.kernel_info), "compiler_options": "--keep-ptx"}
    _raw_case("cute_dsl_saxpy", inputs=inputs, outputs={"y": output}, ptx=ptx, metadata=details)
    entries = list(compiled.kernel_info)
    if len(entries) != 1:
        raise ValueError("CuTe SAXPY capture requires exactly one compiled kernel")
    entry = entries[0]
    if not isinstance(ptx, str):
        raise ValueError("CuTe SAXPY capture needs the compiled PTX (--keep-ptx)")
    params = _params(ptx, entry)
    if (len(params) != 3 or params[0].type != "f32" or params[0].by_value
            or any(p.size != 8 or not p.by_value for p in params[1:])):
        raise ValueError("compiled CuTe SAXPY ABI changed")
    args = [_scalar(scalar, params[0])]
    args += [{"hex": "00" * 8, "relocations": [{"offset": 0, "buffer": name}]}
             for name in ("x", "y")]
    return capture_case("cute_dsl_saxpy", inputs=inputs, outputs={"y": output}, ptx=ptx,
                        launch={"entry": entry, "grid": list(grid), "block": list(block),
                                "cluster": [1, 1, 1], "smem": 0, "args": args},
                        comparison={"kind": "allclose", "rtol": 1e-6, "atol": 0},
                        metadata=details)
