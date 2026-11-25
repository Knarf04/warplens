"""The shim protocol without a program: a fat binary like the one nvcc embeds, and a client that
speaks to a ShimServer running in a thread."""

import socket
import struct
import threading

from warplens.host.shim_server import ShimServer

from tests.support.paths import EXAMPLES

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
