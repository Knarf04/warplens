"""`warplens run`: the server that stands in for the CUDA runtime, and the command itself."""

import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.host import shim_server as shim
from warplens.host.shim_server import ShimServer, ptx_from_fatbin

from conftest import EXAMPLES, HEADER

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = EXAMPLES / "03_cutlass_mma" / "bin"
VECADD = (EXAMPLES / "00_vecadd" / "vecadd.ptx").read_text()


def fatbin(*texts: str, compressed: int = 0, kind: int = 1) -> bytes:
    """A fat binary like the one nvcc embeds with code=compute_XX and --no-compress."""
    entries = b""
    for text in texts:
        body = b"\n\n" + text.encode() + b"\0"
        body += b"\0" * (-len(body) % 8)
        entries += struct.pack("<HHIQI", kind, 0x0101, 80, len(body), compressed).ljust(80, b"\0") + body
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entries)) + entries


class Client:
    """Speaks the shim protocol to a ShimServer running in a thread."""

    def __init__(self, **options):
        self.server = ShimServer(**options)
        ours, theirs = socket.socketpair()
        self.sock = ours
        self.thread = threading.Thread(target=self.server.serve, args=(theirs,), daemon=True)
        self.thread.start()

    def call(self, op: int, payload: bytes = b""):
        self.sock.sendall(struct.pack("<II", op, len(payload)) + payload)
        head = self._read(8)
        status, size = struct.unpack("<iI", head)
        return status, self._read(size)

    def _read(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            data += self.sock.recv(n - len(data))
        return data

    def ok(self, op: int, payload: bytes = b"") -> bytes:
        status, reply = self.call(op, payload)
        assert status == 0, reply.decode()
        return reply


def test_ptx_is_extracted_from_a_fat_binary():
    assert ptx_from_fatbin(fatbin(VECADD))[0].strip() == VECADD.strip()
    assert len(ptx_from_fatbin(fatbin(VECADD, VECADD))) == 2
    assert ptx_from_fatbin(fatbin(VECADD, kind=2)) == []  # machine code only
    with pytest.raises(warplens.WarplensError, match="compressed; rebuild it with nvcc --no-compress"):
        ptx_from_fatbin(fatbin(VECADD, compressed=100))
    with pytest.raises(warplens.WarplensError, match="not a CUDA fat binary"):
        ptx_from_fatbin(b"\0" * 32)


def test_a_host_program_session():
    """What a compiled program does: register its module and kernel, allocate, copy in,
    launch with raw argument bytes, copy out."""
    c = Client()
    (module,) = struct.unpack("<I", c.ok(shim.REGISTER_FATBIN, fatbin(VECADD)))
    function, count, *sizes = struct.unpack("<6I", c.ok(shim.REGISTER_FUNCTION, struct.pack("<I", module) + b"vecadd"))
    assert (count, sizes) == (4, [8, 8, 8, 4])
    n = 1000
    a = np.arange(n, dtype=np.float32)
    b = np.full(n, 0.5, dtype=np.float32)
    pa, pb, pc = (struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 4 * n)))[0] for _ in range(3))
    c.ok(shim.WRITE, struct.pack("<Q", pa) + a.tobytes())
    c.ok(shim.WRITE, struct.pack("<Q", pb) + b.tobytes())
    args = struct.pack("<QQQi", pa, pb, pc, n)
    printed = c.ok(shim.LAUNCH, struct.pack("<10IQ", function, 4, 1, 1, 256, 1, 1, 0, 0, 0, 0) + args)
    assert printed == b""
    out = np.frombuffer(c.ok(shim.READ, struct.pack("<QQ", pc, 4 * n)), dtype=np.float32)
    np.testing.assert_array_equal(out, a + b)
    (name, stats), = c.server.launches
    assert name == "vecadd" and stats.ctas == 4 and not stats.findings

    c.ok(shim.COPY, struct.pack("<QQQ", pa, pc, 4 * n))  # device to device
    c.ok(shim.FILL, struct.pack("<QQI", pb, 4 * n, 0))
    np.testing.assert_array_equal(np.frombuffer(c.ok(shim.READ, struct.pack("<QQ", pa, 4 * n)), np.float32), a + b)
    assert not any(c.ok(shim.READ, struct.pack("<QQ", pb, 4 * n)))


def test_errors_reach_the_program_as_a_status_and_a_message():
    c = Client()
    status, text = c.call(shim.REGISTER_FATBIN, fatbin(VECADD, kind=2))
    assert status != 0 and b"carries no PTX" in text and b"code=compute_100a" in text
    (module,) = struct.unpack("<I", c.ok(shim.REGISTER_FATBIN, fatbin(VECADD)))
    status, text = c.call(shim.REGISTER_FUNCTION, struct.pack("<I", module) + b"missing")
    assert status != 0 and b"`missing` is not in the program's PTX" in text
    function = struct.unpack_from("<I", c.ok(shim.REGISTER_FUNCTION, struct.pack("<I", module) + b"vecadd"))[0]
    (p,) = struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 64)))
    # 16 floats allocated, 32 threads told to use them: the kernel runs off the end.
    args = struct.pack("<QQQi", p, p, p, 32)
    status, text = c.call(shim.LAUNCH, struct.pack("<10IQ", function, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0) + args)
    assert status == shim.LAUNCH_FAILURE and b"outside every allocation" in text and b"vecadd" in text
    status, text = c.call(99)
    assert status != 0 and b"unknown shim request" in text


def test_tensor_map_request_builds_a_usable_tensor_map():
    c = Client()
    (p,) = struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 64 * 128 * 2)))
    request = struct.pack("<5IQ5Q4Q5I5I", 6, 2, 3, 0, 0, p,       # float16, rank 2, 128B swizzle
                          64, 128, 0, 0, 0,  128, 0, 0, 0,         # dims (innermost first), stride
                          64, 32, 0, 0, 0,  1, 1, 0, 0, 0)         # box, element strides
    raw = c.ok(shim.TENSOR_MAP, request)
    info = warplens.host.tensor_map.decode(raw)
    assert len(raw) == 128 and info.dims == (64, 128) and info.box == (64, 32) and info.strides == (128,)
    assert (info.elem_size, info.is_float, info.swizzle, info.address) == (2, True, 3, p)
    bad = struct.pack("<5IQ5Q4Q5I5I", 6, 2, 1, 0, 0, p, 64, 128, 0, 0, 0, 128, 0, 0, 0, 64, 32, 0, 0, 0, 1, 1, 0, 0, 0)
    status, text = c.call(shim.TENSOR_MAP, bad)
    assert status != 0 and b"fit the swizzle span" in text  # 128-byte rows with 32B swizzle


def test_launch_arguments_may_be_raw_bytes(sim):
    n = 64
    a = sim.from_numpy(np.arange(n, dtype=np.float32))
    c = sim.empty(n, "f32")
    sim.launch(EXAMPLES / "00_vecadd" / "vecadd.ptx", "vecadd", grid=1, block=64,
               args=[struct.pack("<Q", a.addr), a, struct.pack("<Q", c.addr), struct.pack("<i", n)])
    np.testing.assert_array_equal(c.numpy(), 2 * np.arange(n, dtype=np.float32))
    with pytest.raises(warplens.LaunchError, match="must be 4 bytes, got 8"):
        sim.launch(EXAMPLES / "00_vecadd" / "vecadd.ptx", "vecadd", grid=1, block=64, args=[a, a, c, b"\0" * 8])


# -- the command, on real programs ---------------------------------------------------------------

def run(*arguments):
    return subprocess.run([sys.executable, "-m", "warplens.cli", "run", *arguments],
                          capture_output=True, text=True, cwd=ROOT, timeout=900)


def test_cli_reports_a_missing_program():
    done = run("no/such/program")
    assert done.returncode != 0 and "no such program" in done.stderr


@pytest.mark.parametrize("tutorial", [1, 2, 3, 4])
def test_unmodified_cutlass_program_runs_and_passes_its_own_check(tutorial):
    """Built by `examples/03_cutlass_mma/build.sh --programs`: the tutorial's own host code
    allocates, encodes tensor maps, launches (in clusters for 3 and 4) and verifies."""
    program = PROGRAMS / f"tutorial_0{tutorial}"
    if not program.exists() or not any((ROOT / "warplens" / "shim" / "lib").glob("libcudart.so.*")):
        pytest.skip("CUTLASS programs or the shim libraries are not built")
    done = run(str(program))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Execution is successful." in done.stdout
    assert "Relative error: 0.000000e+00" in done.stdout
    assert "kernel launch(es) simulated on B200, 0 finding(s)" in done.stderr
    assert "gemm_device" in done.stderr and "16 CTA(s)" in done.stderr
