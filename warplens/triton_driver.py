"""Run Triton kernels in warplens: a Triton driver that launches into the simulator.

    import numpy as np, triton, triton.language as tl
    import warplens.triton_driver as wt

    gpu = wt.install("B200")              # Triton now targets a simulated Blackwell GPU
    x, y = gpu.tensor(a), gpu.tensor(b)
    out = gpu.empty(a.shape, "float32")
    add_kernel[(grid,)](x, y, out, n, BLOCK=128)
    result, stats = out.numpy(), gpu.last_stats

Triton compiles for `sm_100a` with its own bundled toolchain; no GPU or CUDA driver is
involved. Tensors are `gpu.tensor(...)` objects rather than torch tensors. Needs the
`triton` package.
"""

from __future__ import annotations

import struct
import weakref

import numpy as np

from .errors import LaunchError
from .frontend.ptx_parser import parse_ptx
from .host.device import Device, DeviceArray

try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.driver import DriverBase
except ImportError as exc:  # pragma: no cover
    raise ImportError("warplens.triton_driver needs the `triton` package (pip install triton)") from exc


# JITFunction caches binders, targets and launchers by the driver's device key.
# A simulator must never reuse a real CUDA device's cache (usually device 0).
_DEVICE = object()
_installed: list = []
_loaded_kernels = weakref.WeakSet()


class Tensor:
    """An array in simulated device memory that Triton accepts as a pointer argument."""

    def __init__(self, array: DeviceArray):
        self.array = array
        self.dtype = array.dtype  # Triton reads the element type from str(dtype)
        self.shape = array.shape

    def data_ptr(self) -> int:
        return self.array.addr

    def numpy(self) -> np.ndarray:
        return self.array.numpy()

    def __repr__(self) -> str:
        return f"warplens Tensor({self.shape}, {self.dtype})"


class _Utils:
    def __init__(self, driver):
        self.driver = driver

    def load_binary(self, name, kernel, shared, device):
        # Triton hands over the machine code it assembled; warplens runs the PTX instead,
        # which the launcher looks up through this handle.
        handle = object()
        return object(), handle, 0, 0, 1024

    def unload_module(self, module):
        pass

    def get_device_properties(self, device):
        return {"max_shared_mem": 227 * 1024, "max_num_regs": 65536, "multiprocessor_count": 148,
                "warpSize": 32, "sm_clock_rate": 1500000, "mem_clock_rate": 0, "mem_bus_width": 0}


def _encode(value, ptx_type: str) -> bytes:
    if hasattr(value, "data_ptr"):
        value = value.data_ptr()
    size = int(ptx_type[1:]) // 8
    if ptx_type[0] == "f":
        return struct.pack({2: "<e", 4: "<f", 8: "<d"}[size], float(value))
    if isinstance(value, (float, np.floating)):
        raise LaunchError(f"a float was passed for an integer kernel parameter (.{ptx_type})")
    return (int(value) % (1 << (8 * size))).to_bytes(size, "little")


def _arguments(value, kind, device):
    """Flatten Triton's tuple ABI and remove constexpr leaves."""
    if isinstance(kind, (tuple, list)):
        if not isinstance(value, (tuple, list)) or len(value) != len(kind):
            raise LaunchError("Triton tuple argument does not match its compiled signature")
        return [item for v, k in zip(value, kind) for item in _arguments(v, k, device)]
    if kind == "constexpr":
        return []
    if kind.startswith("*"):
        if isinstance(value, Tensor):
            if value.array.device is not device:
                raise LaunchError("the Triton tensor belongs to a different simulated device")
        elif hasattr(value, "data_ptr"):
            raise LaunchError("use a warplens tensor for simulated pointer arguments")
        elif value is None:
            value = 0
    elif kind.startswith(("tensordesc", "nvTmaDesc")):
        raise LaunchError("Triton tensor descriptor arguments are not supported")
    return [value]


def _make_launcher():
    class Launcher:
        """What Triton calls to launch one compiled kernel. Triton caches compiled kernels,
        launcher included, so the launcher looks up the driver that is active at each call."""

        def __init__(self, src, metadata):
            self.src, self.metadata = src, metadata
            self.module = None
            for kernel in src.fn.device_caches[_DEVICE][0].values():
                if getattr(kernel, "src", None) is src:
                    _loaded_kernels.add(kernel)

        def _ptx(self, function):
            if self.module is None:
                kernels = self.src.fn.device_caches[_DEVICE][0]
                for kernel in kernels.values():
                    if getattr(kernel, "function", None) is function:
                        # Triton's destructor uses the currently active driver,
                        # but this fake module is not a hardware resource.
                        kernel._module_pid = None
                        self.module = parse_ptx(kernel.asm["ptx"], name=f"{self.metadata.name}.ptx")
                        break
                if self.module is None:
                    raise LaunchError(f"could not find the PTX of Triton kernel `{self.metadata.name}`")
            return self.module

        def __call__(self, grid_x, grid_y, grid_z, stream, function, packed_metadata,
                     launch_metadata, enter_hook, exit_hook, *args):
            driver = active_driver()
            if driver is None:
                raise LaunchError("this kernel was compiled for the warplens driver, which is no "
                                  "longer active; call warplens.triton_driver.install() again")
            module = self._ptx(function)
            params = module.functions[self.metadata.name].params
            if len(args) != len(self.src.signature):
                raise LaunchError("Triton arguments do not match the compiled signature")
            values = [v for a, kind in zip(args, self.src.signature.values())
                      for v in _arguments(a, kind, driver.device)]
            # Triton 3.9 appends global and profile scratch pointers to the ABI.
            values += [0, 0]
            if len(values) != len(params):
                raise LaunchError(f"Triton kernel `{self.metadata.name}` takes {len(params)} "
                                  f"parameters in PTX but {len(values)} were supplied")
            if getattr(self.metadata, "global_scratch_size", 0):
                raise LaunchError("Triton kernels that need a global scratch buffer are not supported")
            if getattr(self.metadata, "profile_scratch_size", 0):
                raise LaunchError("Triton kernels that need a profile scratch buffer are not supported")
            if enter_hook is not None:
                enter_hook(launch_metadata)
            # Triton's native launcher treats any empty grid as a no-op.
            if 0 in (grid_x, grid_y, grid_z):
                if exit_hook is not None:
                    exit_hook(launch_metadata)
                return
            # Triton's grid counts programs (clusters), not physical CTAs. Its
            # CUDA launcher groups num_ctas consecutive x coordinates per program.
            num_ctas = getattr(self.metadata, "num_ctas", 1)
            cluster = (num_ctas, 1, 1) if num_ctas > 1 else None
            driver.last_stats = driver.device.launch(
                module, self.metadata.name, grid=(grid_x * num_ctas, grid_y, grid_z),
                block=self.metadata.num_warps * 32, smem=self.metadata.shared, cluster=cluster,
                args=[_encode(v, p.type) for v, p in zip(values, params)], **driver.launch_options)
            driver.launches.append((self.metadata.name, driver.last_stats))
            if driver.last_stats.output:
                print(driver.last_stats.output, end="")
            if exit_hook is not None:
                exit_hook(launch_metadata)

    return Launcher


class WarplensDriver(DriverBase):
    """A Triton driver whose device is a warplens simulator."""

    def __init__(self, model: str = "B200", **launch_options):
        self.device = Device(model)
        self.launch_options = launch_options  # e.g. timing=True
        self.last_stats = None
        self.launches: list = []
        self.utils = _Utils(self)
        self.launcher_cls = _make_launcher()
        self.get_current_stream = lambda device: 0
        self.get_current_device = lambda: _DEVICE
        super().__init__()

    # -- what Triton asks of a driver ------------------------------------------------------
    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return GPUTarget("cuda", 100, 32)  # sm_100a

    def get_active_torch_device(self):
        raise NotImplementedError("warplens tensors are not torch tensors; use driver.tensor()")

    def get_device_interface(self):
        raise NotImplementedError("warplens has no torch device interface")

    def get_benchmarker(self):
        raise NotImplementedError("benchmarking a simulator is meaningless; use timing=True for "
                                  "warplens's approximate timing model")

    def get_empty_cache_for_benchmark(self):
        raise NotImplementedError

    def clear_cache(self, cache):
        pass

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    # -- tensors ---------------------------------------------------------------------------
    def tensor(self, array) -> Tensor:
        """Copy a NumPy array to the simulated device."""
        return Tensor(self.device.from_numpy(np.asarray(array)))

    def empty(self, shape, dtype="float32") -> Tensor:
        return Tensor(self.device.empty(shape, np.dtype(dtype)))


def install(model: str = "B200", **launch_options) -> WarplensDriver:
    """Make Triton compile for, and launch on, a simulated GPU. Returns the driver.
    `launch_options` are passed to every `Device.launch`, e.g. `timing=True`."""
    driver = WarplensDriver(model, **launch_options)
    # Reading .active would initialize the default hardware driver on CPU-only
    # machines. Triton 3.9 keeps the uninitialized state as _active=None.
    _installed.append(triton.runtime.driver._active)
    triton.runtime.driver.set_active(driver)
    return driver


def uninstall() -> None:
    """Restore the state preceding install(), without initializing hardware."""
    if _installed:
        # Include kernels whose handles were initialized but never launched.
        # Keep their module marker for Triton's _init_handles() fast path.
        for kernel in _loaded_kernels:
            kernel._module_pid = None
        triton.runtime.driver.set_active(_installed.pop())


def active_driver() -> WarplensDriver | None:
    """The installed simulator, or None; never probes for a hardware driver."""
    driver = triton.runtime.driver._active
    return driver if isinstance(driver, WarplensDriver) else None
