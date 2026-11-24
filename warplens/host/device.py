"""The simulated GPU as seen from the host: memory and kernel launches."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.program import Program
from ..errors import LaunchError
from ..frontend.ir import Module, NP_TYPES
from ..frontend.ptx_parser import parse_ptx_file
from . import tensor_map as _tensor_map
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


class TensorMap(DeviceArray):
    """A tensor map in device memory. Pass it to a kernel like a pointer."""

    def __init__(self, device: "Device", addr: int, info):
        super().__init__(device, addr, (_tensor_map.SIZE,), np.dtype(np.uint8))
        self.info = info

    def tobytes(self) -> bytes:
        """The 128-byte tensor map, for kernels that take it by value inside a struct."""
        return _tensor_map.encode(self.info)

    def __repr__(self) -> str:
        return f"TensorMap({self.info}, addr={self.addr:#x})"


class Device:
    def __init__(self, model: str = "B200"):
        if model not in MODELS:
            raise LaunchError(f"unknown GPU model {model!r}; choose one of {', '.join(MODELS)}")
        self.model = model
        self.memory = GlobalMemory()
        self._modules: dict[Path, Module] = {}
        self._programs: dict[tuple[int, str], Program] = {}
        self._globals: dict[int, dict[str, int]] = {}  # module -> its .global variables' addresses

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

    def tensor_map_tiled(self, array: DeviceArray, box, *, swizzle=None, element_strides=None,
                         oob_fill: str = "zero") -> TensorMap:
        """Describe `array` to TMA, like cuTensorMapEncodeTiled.

        `box` is the tile a TMA instruction moves, one entry per array axis in
        the array's own (NumPy) axis order. `swizzle` is None, "32B", "64B" or
        "128B". Inside a kernel, TMA coordinates are in the opposite order:
        coordinate 0 indexes the last array axis.
        """
        if array.device is not self:
            raise LaunchError("the array belongs to a different device")
        info = _tensor_map.make_tiled(address=array.addr, shape=array.shape, dtype=array.dtype,
                                      box=box, swizzle=swizzle, element_strides=element_strides,
                                      oob_fill=oob_fill)
        addr = self.memory.alloc(_tensor_map.SIZE, align=64)
        self.memory.buffer(addr)[:] = np.frombuffer(_tensor_map.encode(info), dtype=np.uint8)
        return TensorMap(self, addr, info)

    # -- kernels -------------------------------------------------------------
    def load_ptx(self, path) -> Module:
        path = Path(path).resolve()
        if path not in self._modules:
            self._modules[path] = parse_ptx_file(path)
        return self._modules[path]

    def launch(self, ptx, entry: str, *, grid, block, args=(), smem: int = 0, trace=None,
               max_steps: int | None = None, async_delay: int = 4, timing=False,
               active_sms: int | None = None, cluster=None,
               clc_limit: int | None = None) -> LaunchStats:
        """Run kernel `entry` from `ptx` (a path or a parsed Module) to completion.

        `smem` is the dynamic shared memory per CTA in bytes, as passed to a
        real launch; it sizes `extern __shared__` arrays. `cluster` is the
        cluster shape in CTAs, as passed to a cluster launch; the grid must be
        a whole number of clusters.

        With `timing=True` the launch also runs an approximate timing model of
        this device (`stats.timing`); pass a `timing.config.GpuConfig` to use
        other parameters. `active_sms` is how many SMs are assumed to share the
        memory bandwidth (default: all of them). Without timing, `async_delay` is how
        many scheduling rounds a TMA copy takes to complete after it is issued.

        Clusters start one at a time. A kernel that uses cluster launch control
        (`clusterlaunchcontrol.try_cancel`) can cancel those that have not started and do
        their work itself. `clc_limit` is how many cancellations each cluster that starts is
        granted before its requests fail: None for as many as there are clusters left, 0 for
        none, so that every cluster starts by itself. See docs/clc.md.

        `trace`, if given, is called as trace(warp, pc, instr, active_mask)
        before every instruction.
        """
        return run_kernel(self.program(ptx, entry), self.memory, grid=grid, block=block,
                          args=args, smem=smem, trace=trace, max_steps=max_steps,
                          async_delay=async_delay, timing=self._timing(timing, active_sms),
                          cluster=cluster, clc_limit=clc_limit)

    def program(self, ptx, entry: str) -> Program:
        """Kernel `entry` of `ptx` (a path or a parsed Module) prepared for launching, which
        happens once: what `launch` runs. It knows the kernel's static shared memory."""
        module = ptx if isinstance(ptx, Module) else self.load_ptx(ptx)
        if entry not in module.functions:
            known = ", ".join(sorted(module.functions)) or "none"
            raise LaunchError(f"no kernel named `{entry}` in {module.name} (found: {known})")
        key = (id(module), entry)
        if key not in self._programs:
            if id(module) not in self._globals:  # place the module's .global data once
                placed = self._globals[id(module)] = {}
                for var in module.globals:
                    placed[var.name] = self.memory.alloc(max(len(var.data), 1), align=max(var.align, 16))
                    self.memory.buffer(placed[var.name])[:len(var.data)] = np.frombuffer(var.data, np.uint8)
            self._programs[key] = Program(module, module.functions[entry], self._globals[id(module)])
        return self._programs[key]

    def _timing(self, timing, active_sms):
        if timing is None or timing is False:
            return None
        from ..timing.config import GpuConfig
        from ..timing.model import Timing
        if timing is not True and not isinstance(timing, GpuConfig):
            raise LaunchError("timing must be True, False, None or a GpuConfig, "
                              f"got {type(timing).__name__}")
        config = timing if isinstance(timing, GpuConfig) else GpuConfig.load(self.model)
        return Timing(config, active_sms)
