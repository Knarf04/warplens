"""The `warplens run` side of the shim: executes the CUDA runtime calls of a compiled
program in the simulator. See warplens/shim/protocol.h for the wire format."""

from __future__ import annotations

import re
import struct

import numpy as np

from ..errors import WarplensError
from ..frontend.ptx_parser import parse_ptx
from . import tensor_map
from .device import Device

(REGISTER_FATBIN, REGISTER_FUNCTION, MALLOC, FREE, WRITE, READ, COPY, FILL, LAUNCH,
 TENSOR_MAP, REGISTER_MODULE) = range(1, 12)
LAUNCH_FAILURE = 719  # cudaErrorLaunchFailure
INVALID_VALUE = 1  # cudaErrorInvalidValue
MEMORY_ALLOCATION = 2  # cudaErrorMemoryAllocation
MACHINE_CODE = ("the module is machine code for a real GPU, and the simulator runs PTX. Ask the "
                'compiler for PTX (TIRx: backend_config={"cuda": {"target_format": "ptx"}})')


def _read_exact(stream, size: int) -> bytes:
    parts, remaining = [], size
    while remaining:
        part = stream.read(remaining)
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _write_all(stream, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise ConnectionError("lost the connection to the CUDA program")
        remaining = remaining[written:]


def ptx_from_fatbin(blob: bytes) -> list[str]:
    """The PTX texts in a fat binary (what `nvcc -gencode ...,code=compute_XX` embeds)."""
    if len(blob) < 16:
        raise WarplensError("truncated CUDA fat binary header")
    magic, _, header_size, size = struct.unpack_from("<IHHQ", blob, 0)
    if magic != 0xBA55ED50:
        raise WarplensError("not a CUDA fat binary")
    if header_size < 16 or header_size > len(blob) or size > len(blob) - header_size:
        raise WarplensError("invalid CUDA fat binary size")
    texts, at, end = [], header_size, header_size + size
    while at < end:
        if end - at < 20:
            raise WarplensError("truncated CUDA fat binary entry")
        kind, _, entry_header, payload = struct.unpack_from("<HHIQ", blob, at)
        if entry_header < 20 or entry_header > end - at or payload > end - at - entry_header:
            raise WarplensError("invalid CUDA fat binary entry size")
        (compressed,) = struct.unpack_from("<I", blob, at + 16)
        body = blob[at + entry_header:at + entry_header + payload]
        if kind == 1:
            if compressed or b".version" not in body[:4096]:
                raise WarplensError("the program's PTX is compressed; rebuild it with nvcc --no-compress")
            texts.append(body.rstrip(b"\0").decode("utf-8", errors="replace"))
        at += entry_header + payload
    return texts


class ShimServer:
    """Serves one program. `kernels` records every launch for the summary."""

    def __init__(self, model: str = "B200", timing: bool = False, kept_ptx=None):
        self.device = Device(model)
        self.timing = timing
        self.modules: list = []
        self.functions: list[tuple[int, int, str]] = []
        self.launches: list[tuple[str, object]] = []  # (kernel name, LaunchStats)
        self.needs_ptx = False
        self.machine_code = False  # a runtime handed over a compiled module instead of PTX
        self.kept_ptx = kept_ptx  # the directory where a compiler keeps the PTX of such modules
        self.kept: set[int] = set()  # the modules whose kernels are looked up there
        import os
        self.debug = bool(os.environ.get("WARPLENS_SHIM_DEBUG"))  # log every request

    def serve(self, sock) -> None:
        with sock.makefile("rwb", buffering=0) as stream:
            while True:
                head = _read_exact(stream, 8)
                if len(head) < 8:
                    return
                op, size = struct.unpack("<II", head)
                payload = _read_exact(stream, size)
                if len(payload) != size:
                    return
                try:
                    status, reply = 0, self.handle(op, payload)
                except WarplensError as e:
                    status, reply = (LAUNCH_FAILURE if op == LAUNCH else INVALID_VALUE), str(e).encode()
                except MemoryError:
                    status, reply = MEMORY_ALLOCATION, b"host memory exhausted by the simulated CUDA request"
                if self.debug:
                    import sys
                    print(f"warplens shim: op {op}, {size} byte(s) in -> status {status}, "
                          f"{len(reply)} byte(s) out" + (f": {reply[:300].decode(errors='replace')}" if status else ""),
                          file=sys.stderr)
                _write_all(stream, struct.pack("<iI", status, len(reply)) + reply)

    def load_module(self, data: bytes) -> list:
        """A module a program loads itself: PTX text, a fat binary, or machine code. Machine
        code cannot be simulated; its module starts empty and takes each kernel from the PTX
        that the compiler kept in `kept_ptx`, when that is known."""
        name = f"module{len(self.modules)}.ptx"
        texts = None
        if data.startswith(b"\x50\xed\x55\xba"):
            texts = ptx_from_fatbin(data)
        elif not data.startswith(b"\x7fELF"):
            try:
                texts = [data.decode()]
            except UnicodeDecodeError:
                raise WarplensError("the module is not PTX text") from None
            if ".version" not in texts[0][:4096]:
                raise WarplensError("the module is not PTX text (no .version directive)")
        if texts:
            return [parse_ptx(text, name=name) for text in texts]
        if self.kept_ptx is None:
            self.machine_code = True
            raise WarplensError(MACHINE_CODE)
        self.kept.add(len(self.modules))
        return []

    def find_kept(self, kernel: str):
        """The newest PTX file in `kept_ptx` that defines `kernel`, parsed."""
        entry = re.compile(rf"\.entry\s+{re.escape(kernel)}\s*\(")
        files = sorted(self.kept_ptx.glob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            text = path.read_text(errors="replace")
            if entry.search(text):
                return parse_ptx(text, name=path.name)
        self.machine_code = True
        raise WarplensError(f"kernel `{kernel}`: {MACHINE_CODE}")

    def handle(self, op: int, data: bytes) -> bytes:
        sizes = {MALLOC: 8, FREE: 8, READ: 16, COPY: 24, FILL: 20, TENSOR_MAP: 140}
        minimum = {REGISTER_FUNCTION: 4, WRITE: 8, LAUNCH: 48}
        if ((op in sizes and len(data) != sizes[op]) or
                (op in minimum and len(data) < minimum[op])):
            raise WarplensError(f"invalid payload size {len(data)} for shim request {op}")
        memory = self.device.memory
        if op == REGISTER_FATBIN:
            texts = ptx_from_fatbin(data)
            if not texts:
                self.needs_ptx = True
                raise WarplensError(
                    "the program carries no PTX, only machine code for a real GPU. Build it with "
                    "nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared")
            self.modules.append([parse_ptx(t, name=f"module{len(self.modules)}.ptx") for t in texts])
            return struct.pack("<I", len(self.modules) - 1)
        if op == REGISTER_MODULE:
            self.modules.append(self.load_module(data))
            return struct.pack("<I", len(self.modules) - 1)
        if op == REGISTER_FUNCTION:
            (module,) = struct.unpack_from("<I", data)
            if module >= len(self.modules):
                raise WarplensError(f"unknown module {module}")
            try:
                name = data[4:].decode()
            except UnicodeDecodeError:
                raise WarplensError("kernel name is not UTF-8") from None
            if module in self.kept and not any(name in p.functions for p in self.modules[module]):
                self.modules[module].append(self.find_kept(name))
            for index, parsed in enumerate(self.modules[module]):
                if name in parsed.functions:
                    self.functions.append((module, index, name))
                    sizes = [p.size for p in parsed.functions[name].params]
                    return struct.pack(f"<II{len(sizes)}I", len(self.functions) - 1, len(sizes), *sizes)
            raise WarplensError(f"kernel `{name}` is not in the program's PTX")
        if op == MALLOC:
            (size,) = struct.unpack("<Q", data)
            if size > np.iinfo(np.intp).max:
                raise MemoryError
            return struct.pack("<Q", memory.alloc(max(size, 1)))
        if op == FREE:
            (address,) = struct.unpack("<Q", data)
            memory.free(address)
            return b""
        if op == WRITE:
            (address,) = struct.unpack_from("<Q", data)
            body = np.frombuffer(data, dtype=np.uint8, offset=8)
            memory.validate_range(address, len(body), "write")
            memory.write(np.uint64(address) + np.arange(len(body), dtype=np.uint64), body)
            return b""
        if op == READ:
            address, size = struct.unpack("<QQ", data)
            memory.validate_range(address, size, "read")
            if size > 0xffffffff:
                raise WarplensError("shim read exceeds the 32-bit response length; split the transfer")
            return memory.read(np.uint64(address) + np.arange(size, dtype=np.uint64),
                               np.dtype(np.uint8)).tobytes()
        if op == COPY:
            dst, src, size = struct.unpack("<QQQ", data)
            memory.validate_range(src, size, "read")
            memory.validate_range(dst, size, "write")
            span = np.arange(size, dtype=np.uint64)
            memory.write(np.uint64(dst) + span, memory.read(np.uint64(src) + span, np.dtype(np.uint8)))
            return b""
        if op == FILL:
            address, size, value = struct.unpack("<QQI", data)
            memory.validate_range(address, size, "write")
            memory.write(np.uint64(address) + np.arange(size, dtype=np.uint64),
                         np.full(size, value & 0xff, dtype=np.uint8))
            return b""
        if op == LAUNCH:
            function, gx, gy, gz, bx, by, bz, kx, ky, kz, smem = struct.unpack_from("<10IQ", data)
            if function >= len(self.functions):
                raise WarplensError(f"unknown kernel function {function}")
            module, index, name = self.functions[function]
            parsed = self.modules[module][index]
            expected = 48 + sum(p.size for p in parsed.functions[name].params)
            if len(data) != expected:
                raise WarplensError(f"kernel `{name}` launch needs {expected - 48} argument bytes, "
                                    f"got {len(data) - 48}")
            args, at = [], 48
            for p in parsed.functions[name].params:
                args.append(data[at:at + p.size])
                at += p.size
            stats = self.device.launch(parsed, name, grid=(gx, gy, gz), block=(bx, by, bz),
                                       smem=smem, args=args, timing=self.timing,
                                       cluster=(kx, ky, kz) if kx else None)
            self.launches.append((name, stats))
            return stats.output.encode()
        if op == TENSOR_MAP:
            data_type, rank, swizzle, fill, interleave, address = struct.unpack_from("<5IQ", data)
            dims = struct.unpack_from("<5Q", data, 28)
            strides = struct.unpack_from("<4Q", data, 68)
            box = struct.unpack_from("<5I", data, 100)
            steps = struct.unpack_from("<5I", data, 120)
            info = tensor_map.from_driver(data_type=data_type, rank=rank, address=address, dims=dims,
                                          strides=strides, box=box, steps=steps, swizzle=swizzle,
                                          nan_fill=fill, interleave=interleave)
            return tensor_map.encode(info)
        raise WarplensError(f"unknown shim request {op}")
