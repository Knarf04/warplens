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


def test_a_refused_launch_has_the_runtime_code_for_that_refusal():
    """A launch refused before the kernel runs is not a kernel fault. The codes are what
    CUDA 12.4 returned on an RTX 3080 Ti for too much shared memory (invalid value), a block
    or grid it cannot run (invalid configuration) and a cluster it does not accept (invalid
    cluster size; there for any cluster, since Ampere has none)."""
    c = Client()
    (module,) = struct.unpack("<I", c.ok(shim.REGISTER_FATBIN, fatbin(VECADD)))
    function = struct.unpack_from("<I", c.ok(shim.REGISTER_FUNCTION, struct.pack("<I", module) + b"vecadd"))[0]
    (p,) = struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 4096)))

    def launch(grid=(1, 1, 1), block=(32, 1, 1), cluster=(0, 0, 0), smem=0, n=32):
        return c.call(shim.LAUNCH, struct.pack("<10IQ", function, *grid, *block, *cluster, smem)
                      + struct.pack("<QQQi", p, p, p, n))

    assert launch()[0] == 0
    assert launch(cluster=(2, 1, 1), grid=(2, 1, 1))[0] == 0
    status, text = launch(cluster=(32, 1, 1), grid=(32, 1, 1))
    assert status == 912 and b"more than 16 CTAs" in text  # cudaErrorInvalidClusterSize
    status, text = launch(cluster=(2, 1, 1), grid=(3, 1, 1))
    assert status == 912 and b"not a whole number of clusters" in text
    status, text = launch(smem=227 * 1024 + 16)
    assert status == 1 and b"the limit is 232448" in text  # cudaErrorInvalidValue
    status, text = launch(smem=227 * 1024)  # within the limit, but not opted in to
    assert status == 1 and b"without opt-in it may use 49152" in text and b"cudaFuncSetAttribute" in text
    c.ok(shim.SET_ATTRIBUTE, struct.pack("<Iii", function, shim.MAX_DYNAMIC_SMEM, 227 * 1024))
    assert launch(smem=227 * 1024)[0] == 0
    status, text = launch(block=(1025, 1, 1))
    assert status == 9 and b"1025 threads" in text  # cudaErrorInvalidConfiguration
    assert launch(block=(0, 1, 1))[0] == 9 and launch(grid=(0, 1, 1))[0] == 9
    assert launch(block=(1, 1, 65))[0] == 9
    for grid in ((2**31, 1, 1), (1, 65536, 1), (1, 1, 65536)):
        status, text = launch(grid=grid)
        assert status == 9 and b"dimension" in text  # refused before constructing the grid
    assert launch(grid=(64, 1, 1), n=2048)[0] == shim.LAUNCH_FAILURE == 719  # runs off the allocation


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


STATIC_KERNELS = HEADER + "".join(f"""
.visible .entry static_{size}(.param .u64 out)
{{
    .shared .align 1 .b8 fixed[{size}];
    .reg .b64 %p; .reg .b32 %v;
    ld.param.u64 %p, [out];
    mov.u32 %v, %tid.x;
    st.shared.u8 [fixed+{max(size - 1, 0)}], %v;
    ret;
}}
""" for size in (16, 40960, 49152, 49168, 61440, 227 * 1024))


def test_a_launch_is_held_to_the_shared_memory_the_kernel_was_opted_in_to():
    """The rule an RTX 3080 Ti follows (validation/ampere/launch_errors_3080ti.txt), at the
    simulated device's limits: 48 KB by default, 227 KB with opt-in. A kernel may use the
    default less its static shared memory in dynamic shared memory; setting its maximum
    replaces that number (a smaller one too) and is refused above what the device could
    give the kernel; a kernel whose static shared memory alone is over the default launches
    without opt-in and may use all the rest. A refusal is cudaErrorInvalidValue."""
    default, most = 48 * 1024, 227 * 1024
    c = Client()
    (module,) = struct.unpack("<I", c.ok(shim.REGISTER_FATBIN, fatbin(STATIC_KERNELS)))
    (out,) = struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 64)))

    def kernel(size):
        return struct.unpack_from("<I", c.ok(shim.REGISTER_FUNCTION, struct.pack("<I", module) + f"static_{size}".encode()))[0]

    def launch(function, smem):
        return c.call(shim.LAUNCH, struct.pack("<10IQ", function, 1, 1, 1, 32, 1, 1, 0, 0, 0, smem) + struct.pack("<Q", out))[0]

    def attributes(function):
        return struct.unpack("<II", c.ok(shim.GET_ATTRIBUTES, struct.pack("<I", function)))

    def opt_in(function, value):
        return c.call(shim.SET_ATTRIBUTE, struct.pack("<Iii", function, shim.MAX_DYNAMIC_SMEM, value))[0]

    small = kernel(16)
    assert attributes(small) == (16, default - 16)
    assert launch(small, default - 16) == 0 and launch(small, default - 15) == 1 and launch(small, default) == 1
    assert opt_in(small, 60000) == 0 and attributes(small) == (16, 60000)
    assert launch(small, 60000) == 0 and launch(small, 60001) == 1
    assert opt_in(small, 1024) == 0  # a maximum under the default lowers what a launch may use
    assert launch(small, 1024) == 0 and launch(small, 1040) == 1 and launch(small, default - 16) == 1
    assert opt_in(small, most - 16) == 0 and opt_in(small, most - 15) == 1 and opt_in(small, -1) == 1
    assert attributes(small) == (16, most - 16)  # a refused value changes nothing
    assert launch(small, most - 16) == 0 and launch(small, most) == 1

    medium = kernel(40960)  # static and dynamic together against the default
    assert attributes(medium) == (40960, 8192)
    assert launch(medium, 8192) == 0 and launch(medium, 8208) == 1
    assert opt_in(medium, 8208) == 0 and launch(medium, 8208) == 0 and launch(medium, 8224) == 1
    another = kernel(49152)  # each kernel has its own maximum: this one is not opted in
    assert attributes(another) == (default, 0) and launch(another, 0) == 0 and launch(another, 16) == 1

    # Static shared memory over the default: no opt-in is needed at all. What such a kernel
    # reports as its maximum is 0 all the same, as a B200 does (the 3080 Ti reported the
    # rest of the opt-in maximum): validation/blackwell/measured/b200/e2e/smem_static.txt.
    over = kernel(49168)
    assert attributes(over) == (49168, 0)
    assert launch(over, 0) == 0 and launch(over, most - 49168) == 0 and launch(over, most - 49168 + 16) == 1
    large = kernel(61440)
    assert attributes(large) == (61440, 0) and launch(large, 16) == 0
    assert opt_in(large, most - 61440 + 1) == 1 and opt_in(large, 0) == 0
    assert launch(large, 0) == 0 and launch(large, 16) == 1
    full = kernel(most)
    assert attributes(full) == (most, 0) and launch(full, 0) == 0 and launch(full, 16) == 1
    assert [name for name, _ in c.server.launches] == (
        ["static_16"] * 4 + ["static_40960"] * 2 + ["static_49152"] + ["static_49168"] * 2 + ["static_61440"] * 2
        + [f"static_{most}"])  # no refused launch ran


def test_a_cluster_of_more_than_eight_ctas_needs_the_non_portable_attribute(sim):
    """As a B200 does (validation/blackwell/measured/b200/cluster/max_size.txt and
    cluster-optin/optin.txt): 8 CTAs launch, 16 are refused with cudaErrorInvalidClusterSize
    until the program sets cudaFuncAttributeNonPortableClusterSizeAllowed for the kernel, and
    32 are refused either way. Each kernel has its own setting, and it can be taken back."""
    c = Client()
    (module,) = struct.unpack("<I", c.ok(shim.REGISTER_FATBIN, fatbin(STATIC_KERNELS)))
    (out,) = struct.unpack("<Q", c.ok(shim.MALLOC, struct.pack("<Q", 64)))
    one, other = (struct.unpack_from("<I", c.ok(shim.REGISTER_FUNCTION,
                                                 struct.pack("<I", module) + name.encode()))[0]
                  for name in ("static_16", "static_40960"))

    def refusal(function, ctas):
        return c.call(shim.LAUNCH, struct.pack("<10IQ", function, ctas, 1, 1, 32, 1, 1, ctas, 1, 1, 0)
                      + struct.pack("<Q", out))

    def launch(function, ctas):
        return refusal(function, ctas)[0]

    def allow(function, value):
        return c.call(shim.SET_ATTRIBUTE, struct.pack("<Iii", function, shim.NONPORTABLE_CLUSTER, value))[0]

    invalid_cluster_size = 912
    assert launch(one, 8) == 0 and launch(one, 16) == invalid_cluster_size and launch(one, 32) == invalid_cluster_size
    assert "cudaFuncAttributeNonPortableClusterSizeAllowed, 1)" in refusal(one, 16)[1].decode()
    assert allow(one, 1) == 0
    assert launch(one, 16) == 0 and launch(one, 32) == invalid_cluster_size
    assert launch(other, 16) == invalid_cluster_size  # another kernel was not allowed them
    assert allow(one, 0) == 0 and launch(one, 16) == invalid_cluster_size and launch(one, 8) == 0
    # The Python API has no attribute to set and takes any cluster up to the device's 16.
    module = warplens.parse_ptx(STATIC_KERNELS)
    sim.launch(module, "static_16", grid=16, block=32, cluster=(16, 1, 1), args=[sim.empty(16, "u32")])


def test_the_python_api_has_no_opt_in_step(sim):
    """`Device.launch` takes the shared memory of the launch and nothing else: the opt-in is
    a step of the CUDA runtime, which only programs under the shim go through."""
    module = warplens.parse_ptx(STATIC_KERNELS)
    stats = sim.launch(module, "static_16", grid=1, block=32, smem=200 * 1024, args=[sim.empty(16, "u32")])
    assert stats.smem_bytes == 16 + 200 * 1024
