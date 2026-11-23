"""The `warplens run` side of the shim: executes the CUDA runtime calls of a compiled
program in the simulator. See warplens/shim/protocol.h for the wire format."""

from __future__ import annotations

import struct

import numpy as np

from ..errors import WarplensError
from ..frontend.ptx_parser import parse_ptx
from . import tensor_map
from .device import Device

(REGISTER_FATBIN, REGISTER_FUNCTION, MALLOC, FREE, WRITE, READ, COPY, FILL, LAUNCH,
 TENSOR_MAP) = range(1, 11)
LAUNCH_FAILURE = 719  # cudaErrorLaunchFailure
INVALID_VALUE = 1  # cudaErrorInvalidValue


def ptx_from_fatbin(blob: bytes) -> list[str]:
    """The PTX texts in a fat binary (what `nvcc -gencode ...,code=compute_XX` embeds)."""
    magic, _, header_size, size = struct.unpack_from("<IHHQ", blob, 0)
    if magic != 0xBA55ED50:
        raise WarplensError("not a CUDA fat binary")
    texts, at, end = [], header_size, header_size + size
    while at < end:
        kind, _, entry_header, payload = struct.unpack_from("<HHIQ", blob, at)
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

    def __init__(self, model: str = "B200", timing: bool = False):
        self.device = Device(model)
        self.timing = timing
        self.modules: list = []
        self.functions: list[tuple[int, str]] = []
        self.launches: list[tuple[str, object]] = []  # (kernel name, LaunchStats)
        self.needs_ptx = False
        import os
        self.debug = bool(os.environ.get("WARPLENS_SHIM_DEBUG"))  # log every request

    def serve(self, sock) -> None:
        stream = sock.makefile("rwb", buffering=0)
        while True:
            head = stream.read(8)
            if len(head) < 8:
                return
            op, size = struct.unpack("<II", head)
            payload = b""
            while len(payload) < size:
                chunk = stream.read(size - len(payload))
                if not chunk:
                    return
                payload += chunk
            try:
                status, reply = 0, self.handle(op, payload)
            except WarplensError as e:
                status, reply = (LAUNCH_FAILURE if op == LAUNCH else INVALID_VALUE), str(e).encode()
            if self.debug:
                import sys
                print(f"warplens shim: op {op}, {size} byte(s) in -> status {status}, "
                      f"{len(reply)} byte(s) out" + (f": {reply[:300].decode(errors='replace')}" if status else ""),
                      file=sys.stderr)
            stream.write(struct.pack("<iI", status, len(reply)) + reply)

    def handle(self, op: int, data: bytes) -> bytes:
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
        if op == REGISTER_FUNCTION:
            (module,) = struct.unpack_from("<I", data)
            name = data[4:].decode()
            for index, parsed in enumerate(self.modules[module]):
                if name in parsed.functions:
                    self.functions.append((module, index, name))
                    sizes = [p.size for p in parsed.functions[name].params]
                    return struct.pack(f"<II{len(sizes)}I", len(self.functions) - 1, len(sizes), *sizes)
            raise WarplensError(f"kernel `{name}` is not in the program's PTX")
        if op == MALLOC:
            (size,) = struct.unpack("<Q", data)
            return struct.pack("<Q", memory.alloc(max(size, 1)))
        if op == FREE:
            return b""  # addresses are never reused, so stale pointers keep faulting cleanly
        if op == WRITE:
            (address,) = struct.unpack_from("<Q", data)
            body = np.frombuffer(data, dtype=np.uint8, offset=8)
            memory.write(np.uint64(address) + np.arange(len(body), dtype=np.uint64), body)
            return b""
        if op == READ:
            address, size = struct.unpack("<QQ", data)
            return memory.read(np.uint64(address) + np.arange(size, dtype=np.uint64),
                               np.dtype(np.uint8)).tobytes()
        if op == COPY:
            dst, src, size = struct.unpack("<QQQ", data)
            span = np.arange(size, dtype=np.uint64)
            memory.write(np.uint64(dst) + span, memory.read(np.uint64(src) + span, np.dtype(np.uint8)))
            return b""
        if op == FILL:
            address, size, value = struct.unpack("<QQI", data)
            memory.write(np.uint64(address) + np.arange(size, dtype=np.uint64),
                         np.full(size, value, dtype=np.uint8))
            return b""
        if op == LAUNCH:
            function, gx, gy, gz, bx, by, bz, kx, ky, kz, smem = struct.unpack_from("<10IQ", data)
            module, index, name = self.functions[function]
            parsed = self.modules[module][index]
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
                                          nan_fill=bool(fill), interleave=interleave)
            return tensor_map.encode(info)
        raise WarplensError(f"unknown shim request {op}")
