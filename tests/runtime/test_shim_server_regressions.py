"""Runtime protocol boundaries and allocation lifetimes."""

import struct

import pytest

from warplens.errors import MemoryFault, WarplensError
from warplens.host import shim_server as shim

from tests.support.shim import VECADD, fatbin


class FragmentedSocket:
    """A stream transport may split any header, payload, or reply write."""

    def __init__(self, data):
        self.data = data
        self.written = bytearray()
        self.closed = False

    def makefile(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, size):
        part, self.data = self.data[:min(size, 3)], self.data[min(size, 3):]
        return part

    def write(self, data):
        part = data[:5]
        self.written.extend(part)
        return len(part)


def frame(op, data=b""):
    return struct.pack("<II", op, len(data)) + data


def replies(data):
    out = []
    while data:
        status, size = struct.unpack_from("<iI", data)
        assert len(data) >= size + 8
        out.append((status, bytes(data[8:8 + size])))
        data = data[8 + size:]
    return out


def test_partial_reads_and_writes_preserve_framing_and_close_stream():
    sock = FragmentedSocket(frame(shim.MALLOC, struct.pack("<Q", 16)) + frame(99))
    shim.ShimServer().serve(sock)
    first, second = replies(sock.written)
    assert first[0] == 0 and struct.unpack("<Q", first[1])[0] > 0
    assert second[0] == shim.INVALID_VALUE and b"unknown shim request" in second[1]
    assert sock.closed


@pytest.mark.parametrize("op,payload", [
    (shim.MALLOC, b""), (shim.FREE, b""), (shim.READ, b""), (shim.WRITE, b""),
    (shim.COPY, b""), (shim.FILL, b""), (shim.TENSOR_MAP, b""),
    (shim.REGISTER_FUNCTION, b""),
    (shim.REGISTER_FUNCTION, struct.pack("<I", 123) + b"kernel"),
    (shim.LAUNCH, b""), (shim.LAUNCH, bytes(48)),
    (shim.REGISTER_FATBIN, b"bad header"),
])
def test_bad_request_returns_error_and_session_can_continue(op, payload):
    sock = FragmentedSocket(frame(op, payload) + frame(shim.MALLOC, struct.pack("<Q", 4)))
    shim.ShimServer().serve(sock)
    bad, good = replies(sock.written)
    assert bad[0] == (shim.LAUNCH_FAILURE if op == shim.LAUNCH else shim.INVALID_VALUE)
    assert bad[1]
    assert good[0] == 0 and len(good[1]) == 8


@pytest.mark.parametrize("blob", [
    b"", b"\0" * 12,
    struct.pack("<IHHQ", 0xBA55ED50, 1, 0, 16),
    struct.pack("<IHHQ", 0xBA55ED50, 1, 16, 100),
    struct.pack("<IHHQ", 0xBA55ED50, 1, 16, 1) + b"x",
    struct.pack("<IHHQ", 0xBA55ED50, 1, 16, 20) + bytes(20),
    struct.pack("<IHHQ", 0xBA55ED50, 1, 16, 20) + struct.pack("<HHIQI", 1, 1, 20, 100, 0),
])
def test_malformed_fatbin_is_a_reported_error(blob):
    with pytest.raises(WarplensError):
        shim.ptx_from_fatbin(blob)


def test_free_reclaims_memory_and_does_not_reuse_addresses():
    server = shim.ShimServer()
    pointers = [struct.unpack("<Q", server.handle(shim.MALLOC, struct.pack("<Q", 4)))[0]
                for _ in range(3)]
    for i, p in enumerate(pointers):
        server.handle(shim.WRITE, struct.pack("<Q", p) + bytes([i + 1]) * 4)
    server.handle(shim.FREE, struct.pack("<Q", pointers[1]))
    assert len(server.device.memory._bufs) == 2
    with pytest.raises(MemoryFault, match="outside every allocation"):
        server.handle(shim.READ, struct.pack("<QQ", pointers[1], 4))
    for i in (0, 2):
        assert server.handle(shim.READ, struct.pack("<QQ", pointers[i], 4)) == bytes([i + 1]) * 4
    for bad in (pointers[1], pointers[0] + 1, 123):
        with pytest.raises(WarplensError, match="cannot free"):
            server.handle(shim.FREE, struct.pack("<Q", bad))
    assert server.handle(shim.FREE, bytes(8)) == b""
    new = struct.unpack("<Q", server.handle(shim.MALLOC, struct.pack("<Q", 4)))[0]
    assert new > pointers[-1]


def registered_vecadd():
    server = shim.ShimServer()
    module = server.handle(shim.REGISTER_FATBIN, fatbin(VECADD))
    function = struct.unpack_from("<I", server.handle(shim.REGISTER_FUNCTION, module + b"vecadd"))[0]
    address = struct.unpack("<Q", server.handle(shim.MALLOC, struct.pack("<Q", 128)))[0]
    head = struct.pack("<10IQ", function, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0)
    args = struct.pack("<QQQi", address, address, address, 32)
    return server, address, head, args


def test_kernel_access_after_free_faults():
    server, address, head, args = registered_vecadd()
    server.handle(shim.FREE, struct.pack("<Q", address))
    with pytest.raises(MemoryFault, match="outside every allocation"):
        server.handle(shim.LAUNCH, head + args)


@pytest.mark.parametrize("extra", [-1, 1])
def test_launch_rejects_wrong_argument_length_before_execution(extra):
    server, _, head, args = registered_vecadd()
    args = args[:-1] if extra < 0 else args + b"x"
    with pytest.raises(WarplensError, match="argument bytes"):
        server.handle(shim.LAUNCH, head + args)
    assert not server.launches


def test_impossible_allocation_reports_cuda_status():
    sock = FragmentedSocket(frame(shim.MALLOC, struct.pack("<Q", 2**64 - 1)))
    shim.ShimServer().serve(sock)
    (status, message), = replies(sock.written)
    assert status == shim.MEMORY_ALLOCATION and b"memory exhausted" in message


@pytest.mark.parametrize("op", [shim.READ, shim.COPY, shim.FILL])
def test_impossible_transfer_count_returns_error_before_allocating_indices(op):
    server = shim.ShimServer()
    p = struct.unpack("<Q", server.handle(shim.MALLOC, struct.pack("<Q", 16)))[0]
    request = {shim.READ: struct.pack("<QQ", p, 2**64 - 1),
               shim.COPY: struct.pack("<QQQ", p, p, 2**64 - 1),
               shim.FILL: struct.pack("<QQI", p, 2**64 - 1, 0)}[op]
    sock = FragmentedSocket(frame(op, request) + frame(shim.READ, struct.pack("<QQ", p, 16)))
    server.serve(sock)
    bad, good = replies(sock.written)
    assert bad[0] == shim.INVALID_VALUE and b"outside every allocation" in bad[1]
    assert good == (0, bytes(16))


def test_fill_uses_the_low_byte_of_the_value():
    server = shim.ShimServer()
    p = struct.unpack("<Q", server.handle(shim.MALLOC, struct.pack("<Q", 4)))[0]
    server.handle(shim.FILL, struct.pack("<QQI", p, 4, 0x123))
    assert server.handle(shim.READ, struct.pack("<QQ", p, 4)) == b"####"


def test_oversized_reply_is_rejected_before_materializing_data(monkeypatch):
    server = shim.ShimServer()
    # Model an existing large allocation without allocating gigabytes in this test.
    monkeypatch.setattr(server.device.memory, "validate_range", lambda *args: None)
    with pytest.raises(WarplensError, match="32-bit response length"):
        server.handle(shim.READ, struct.pack("<QQ", 0x200000000, 2**32))
