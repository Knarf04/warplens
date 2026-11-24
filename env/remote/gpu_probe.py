#!/usr/bin/env python3
"""Talk to the GPU through the CUDA driver API, for env/remote/preflight.sh.

Standard library only: libcuda.so.1 comes with the driver, and ctypes calls it
directly, so nothing here depends on which toolkit or Python packages are
installed.

    gpu_probe.py info                 driver API version, device name, compute capability
    gpu_probe.py vecadd MODULE        load MODULE (a cubin or PTX text), run pf_vecadd, check
    gpu_probe.py tmem CUBIN           run pf_tmem (tcgen05.alloc / dealloc)
    gpu_probe.py tma CUBIN            encode a tensor map, run pf_tma, check the box
    gpu_probe.py tensormap            cuTensorMapEncodeTiled alone

Output is `key=value` lines. Exit status: 0 the check passed, 1 it failed (the
reason is in `error=`), 2 the driver library could not be loaded or initialized.
"""

import ctypes
import struct
import sys
from ctypes import byref, c_char_p, c_int, c_size_t, c_uint, c_uint32, c_uint64, c_void_p

ERROR_NAMES_FALLBACK = {
    100: "CUDA_ERROR_NO_DEVICE",
    209: "CUDA_ERROR_NO_BINARY_FOR_GPU",
    218: "CUDA_ERROR_INVALID_PTX",
    222: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
    300: "CUDA_ERROR_INVALID_SOURCE",
    801: "CUDA_ERROR_NOT_SUPPORTED",
}
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7

BOX_W, BOX_H = 16, 8          # must match kernels/pf_blackwell.cu
TENSOR_W, TENSOR_H = 64, 32


class CudaError(Exception):
    def __init__(self, call, code, name, detail=""):
        super().__init__(f"{call}: {name} ({code})" + (f": {detail}" if detail else ""))
        self.call, self.code, self.name, self.detail = call, code, name, detail


class Driver:
    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            print(f"error=cannot load libcuda.so.1: {e}")
            sys.exit(2)
        self.ctx = None

    def error_name(self, code):
        name = c_char_p()
        try:
            if self.lib.cuGetErrorName(code, byref(name)) == 0 and name.value:
                return name.value.decode()
        except AttributeError:
            pass
        return ERROR_NAMES_FALLBACK.get(code, "CUDA_ERROR")

    def call(self, fn, *args, detail=""):
        try:
            f = getattr(self.lib, fn)
        except AttributeError:
            raise CudaError(fn, -1, "MISSING_SYMBOL", "this libcuda.so.1 does not export it")
        f.restype = c_int
        code = f(*args)
        if code != 0:
            raise CudaError(fn, code, self.error_name(code), detail)

    def init(self):
        try:
            self.call("cuInit", c_uint(0))
        except CudaError as e:
            print(f"error={e}")
            sys.exit(2)

    def info(self):
        version, device, major, minor = c_int(), c_int(), c_int(), c_int()
        count, total = c_int(), c_size_t()
        name = ctypes.create_string_buffer(256)
        self.call("cuDriverGetVersion", byref(version))
        self.call("cuDeviceGetCount", byref(count))
        self.call("cuDeviceGet", byref(device), c_int(0))
        self.call("cuDeviceGetName", name, c_int(256), device)
        self.call("cuDeviceGetAttribute", byref(major),
                  c_int(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), device)
        self.call("cuDeviceGetAttribute", byref(minor),
                  c_int(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR), device)
        self.call("cuDeviceTotalMem_v2", byref(total), device)
        return {
            "driver_api": f"{version.value // 1000}.{version.value % 1000 // 10}",
            "device_count": count.value,
            "name": name.value.decode(),
            "compute_cap": f"{major.value}.{minor.value}",
            "total_mem_mib": total.value >> 20,
        }

    def create_context(self):
        device, ctx = c_int(), c_void_p()
        self.call("cuDeviceGet", byref(device), c_int(0))
        self.call("cuCtxCreate_v2", byref(ctx), c_uint(0), device)
        self.ctx = ctx

    def close(self):
        if self.ctx is not None:
            self.lib.cuCtxDestroy_v2(self.ctx)
            self.ctx = None

    def load_module(self, path):
        with open(path, "rb") as f:
            image = f.read() + b"\0"   # PTX text must be NUL-terminated; harmless for a cubin
        log = ctypes.create_string_buffer(8192)
        options = (c_int * 2)(CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        values = (c_void_p * 2)(ctypes.cast(log, c_void_p), c_void_p(len(log)))
        module = c_void_p()
        try:
            self.call("cuModuleLoadDataEx", byref(module), image, c_uint(2), options, values)
        except CudaError as e:
            text = " | ".join(log.value.decode(errors="replace").split("\n")).strip(" |")
            raise CudaError(e.call, e.code, e.name, text) from None
        return module

    def function(self, module, name):
        fn = c_void_p()
        self.call("cuModuleGetFunction", byref(fn), module, name.encode())
        return fn

    def alloc(self, nbytes):
        ptr = c_uint64()
        self.call("cuMemAlloc_v2", byref(ptr), c_size_t(nbytes))
        return ptr

    def upload(self, data):
        ptr = self.alloc(len(data))
        self.call("cuMemcpyHtoD_v2", ptr, data, c_size_t(len(data)))
        return ptr

    def download(self, ptr, nbytes):
        out = ctypes.create_string_buffer(nbytes)
        self.call("cuMemcpyDtoH_v2", out, ptr, c_size_t(nbytes))
        return out.raw

    def launch(self, fn, grid, block, args):
        """`args` are ctypes objects; each is passed by value, as the driver expects."""
        params = (c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), c_void_p) for a in args])
        self.call("cuLaunchKernel", fn, c_uint(grid), c_uint(1), c_uint(1),
                  c_uint(block), c_uint(1), c_uint(1), c_uint(0), c_void_p(0), params, c_void_p(0))
        self.call("cuCtxSynchronize")

    def tensor_map(self, address):
        """A tensor map for the TENSOR_H x TENSOR_W float32 tensor at `address`."""
        tmap = (c_uint64 * 24)()   # 128 bytes, with room to align to 64
        base = ctypes.addressof(tmap)
        aligned = (base + 63) & ~63
        global_dim = (c_uint64 * 2)(TENSOR_W, TENSOR_H)
        global_strides = (c_uint64 * 1)(TENSOR_W * 4)
        box = (c_uint32 * 2)(BOX_W, BOX_H)
        element_strides = (c_uint32 * 2)(1, 1)
        self.call("cuTensorMapEncodeTiled", c_void_p(aligned),
                  c_int(CU_TENSOR_MAP_DATA_TYPE_FLOAT32), c_uint32(2), c_void_p(address.value),
                  global_dim, global_strides, box, element_strides,
                  c_int(0), c_int(0), c_int(0), c_int(0))
        return ctypes.string_at(aligned, 128)


def floats(values):
    return struct.pack(f"<{len(values)}f", *values)


def check_vecadd(drv, path):
    n = 1000
    a = [float(i) for i in range(n)]
    b = [0.5 * i for i in range(n)]
    module = drv.load_module(path)
    fn = drv.function(module, "pf_vecadd")
    pa, pb, pc = drv.upload(floats(a)), drv.upload(floats(b)), drv.upload(bytes(4 * n))
    drv.launch(fn, 4, 256, [pa, pb, pc, c_int(n)])
    got = struct.unpack(f"<{n}f", drv.download(pc, 4 * n))
    wrong = sum(1 for i in range(n) if got[i] != a[i] + b[i])
    print(f"elements={n}")
    print(f"mismatches={wrong}")
    if wrong:
        raise CudaError("pf_vecadd", 0, "WRONG_RESULT", f"{wrong} of {n} elements differ")


def check_tmem(drv, path):
    module = drv.load_module(path)
    fn = drv.function(module, "pf_tmem")
    out = drv.upload(bytes(8))
    drv.launch(fn, 1, 32, [out])
    magic, address = struct.unpack("<2I", drv.download(out, 8))
    print(f"magic={magic:#x}")
    print(f"tmem_address={address:#x}")
    if magic != 0xB200B200:
        raise CudaError("pf_tmem", 0, "WRONG_RESULT", "the kernel did not write its marker")


def check_tensormap(drv):
    tensor = drv.alloc(TENSOR_W * TENSOR_H * 4)
    tmap = drv.tensor_map(tensor)
    print(f"tensor_map_bytes={tmap.hex()}")
    if not any(tmap):
        raise CudaError("cuTensorMapEncodeTiled", 0, "WRONG_RESULT", "the tensor map is all zero")


def check_tma(drv, path):
    module = drv.load_module(path)
    fn = drv.function(module, "pf_tma")
    data = [float(i) for i in range(TENSOR_W * TENSOR_H)]
    tensor = drv.upload(floats(data))
    tmap = (ctypes.c_ubyte * 128).from_buffer_copy(drv.tensor_map(tensor))
    out, status = drv.upload(bytes(BOX_W * BOX_H * 4)), drv.upload(bytes(4))
    c0, c1 = 2 * BOX_W, 1 * BOX_H
    drv.launch(fn, 1, 64, [tmap, out, status, c_int(c0), c_int(c1)])
    (state,) = struct.unpack("<I", drv.download(status, 4))
    got = struct.unpack(f"<{BOX_W * BOX_H}f", drv.download(out, BOX_W * BOX_H * 4))
    want = [data[(c1 + r) * TENSOR_W + c0 + c] for r in range(BOX_H) for c in range(BOX_W)]
    wrong = sum(1 for g, w in zip(got, want) if g != w)
    print(f"wait_state={state}")
    print(f"mismatches={wrong}")
    if state == 2:
        raise CudaError("pf_tma", 0, "TIMEOUT", "the mbarrier never completed")
    if state != 1 or wrong:
        raise CudaError("pf_tma", 0, "WRONG_RESULT", f"{wrong} of {len(want)} elements differ")


def main(argv):
    if len(argv) < 2 or argv[1] not in ("info", "vecadd", "tmem", "tma", "tensormap"):
        print(__doc__, file=sys.stderr)
        return 2
    command = argv[1]
    if command in ("vecadd", "tmem", "tma") and len(argv) != 3:
        print(f"usage: gpu_probe.py {command} MODULE", file=sys.stderr)
        return 2
    drv = Driver()
    drv.init()
    try:
        if command == "info":
            for key, value in drv.info().items():
                print(f"{key}={value}")
            return 0
        drv.create_context()
        if command == "vecadd":
            check_vecadd(drv, argv[2])
        elif command == "tmem":
            check_tmem(drv, argv[2])
        elif command == "tma":
            check_tma(drv, argv[2])
        else:
            check_tensormap(drv)
    except CudaError as e:
        print(f"error_code={e.code}")
        print(f"error_name={e.name}")
        print(f"error={e}")
        return 1
    finally:
        drv.close()
    print("ok=1")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
