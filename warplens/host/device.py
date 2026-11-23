"""The simulated GPU as seen from the host: memory and kernel launches."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.program import Program
from ..errors import LaunchError
from ..frontend.ir import Module, NP_TYPES
from ..frontend.ptx_parser import parse_ptx_file
from ..units.memory import GlobalMemory
from .launch import LaunchStats, run_kernel

MODELS = ("B200", "B300")


def _dtype(dtype) -> np.dtype:
    if isinstance(dtype, str) and dtype in NP_TYPES and dtype != "pred":
        return NP_TYPES[dtype]  # PTX-style names: "f32", "u64", ...
    return np.dtype(dtype)


class DeviceArray:
    """An array in simulated global memory."""

    def __init__(self, device: "Device", addr: int, shape: tuple[int, ...], dtype: np.dtype):
        self.device, self.addr, self.shape, self.dtype = device, addr, shape, dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def numpy(self) -> np.ndarray:
        """Copy the current contents back to the host."""
        buf = self.device.memory.buffer(self.addr)
        return buf.view(self.dtype).reshape(self.shape).copy()

    def __repr__(self) -> str:
        return f"DeviceArray({self.shape}, {self.dtype}, addr={self.addr:#x})"


class Device:
    def __init__(self, model: str = "B200"):
        if model not in MODELS:
            raise LaunchError(f"unknown GPU model {model!r}; choose one of {', '.join(MODELS)}")
        self.model = model
        self.memory = GlobalMemory()
        self._modules: dict[Path, Module] = {}
        self._programs: dict[tuple[int, str], Program] = {}

    # -- memory --------------------------------------------------------------
    def empty(self, shape, dtype="f32") -> DeviceArray:
        """Allocate an array. Like real device memory it is not meaningfully
        initialized; here it happens to be zero."""
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        arr = DeviceArray(self, 0, shape, _dtype(dtype))
        arr.addr = self.memory.alloc(arr.nbytes)
        return arr

    zeros = empty

    def from_numpy(self, array) -> DeviceArray:
        """Copy a host array to the device (C order)."""
        array = np.ascontiguousarray(array)
        out = self.empty(array.shape, array.dtype)
        self.memory.buffer(out.addr)[:] = array.reshape(-1).view(np.uint8)
        return out

    # -- kernels -------------------------------------------------------------
    def load_ptx(self, path) -> Module:
        path = Path(path).resolve()
        if path not in self._modules:
            self._modules[path] = parse_ptx_file(path)
        return self._modules[path]

    def launch(self, ptx, entry: str, *, grid, block, args=(), trace=None,
               max_steps: int | None = None) -> LaunchStats:
        """Run kernel `entry` from `ptx` (a path or a parsed Module) to completion.

        `trace`, if given, is called as trace(warp, pc, instr, active_mask)
        before every instruction.
        """
        module = ptx if isinstance(ptx, Module) else self.load_ptx(ptx)
        if entry not in module.functions:
            known = ", ".join(sorted(module.functions)) or "none"
            raise LaunchError(f"no kernel named `{entry}` in {module.name} (found: {known})")
        key = (id(module), entry)
        if key not in self._programs:
            self._programs[key] = Program(module, module.functions[entry])
        return run_kernel(self._programs[key], self.memory, grid=grid, block=block,
                          args=args, trace=trace, max_steps=max_steps)
