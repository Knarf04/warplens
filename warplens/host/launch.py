"""Kernel launch: turn (grid, block, args) into CTAs and warps, then run them."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..check.report import Finding, Findings
from ..core.cta import Cta
from ..core.engine import Engine
from ..core.program import Program
from ..core.warp import WARP_SIZE, Warp
from ..errors import LaunchError
from ..frontend.ir import NP_TYPES, Function
from ..units.smem import MAX_SMEM_BYTES
from ..units.smem_banks import SmemReport

MAX_THREADS_PER_BLOCK = 1024


@dataclass(slots=True)
class LaunchStats:
    ctas: int = 0
    warps: int = 0
    threads: int = 0
    instructions: int = 0  # instructions issued, counted once per warp
    lane_instructions: int = 0  # the same, counted once per active lane
    divergent_branches: int = 0
    max_simt_depth: int = 0
    smem_bytes: int = 0  # shared memory per CTA
    smem: SmemReport = field(default_factory=SmemReport, repr=False)  # bank conflicts
    findings: list[Finding] = field(default_factory=list, repr=False)

    def report(self) -> str:
        """Bank-conflict table and findings as text."""
        parts = [self.smem.format()]
        if self.findings:
            parts.append(f"{len(self.findings)} finding(s):\n" + "\n".join(str(f) for f in self.findings))
        return "\n\n".join(parts)


def dim3(value, what: str) -> tuple[int, int, int]:
    dims = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
    if not 1 <= len(dims) <= 3 or any(int(d) != d or d < 1 for d in dims):
        raise LaunchError(f"{what} must be 1 to 3 positive integers, got {value!r}")
    return tuple(int(d) for d in dims) + (1,) * (3 - len(dims))


def encode_args(fn: Function, args, *, mem) -> dict[str, tuple[int, int]]:
    """Convert host arguments to raw parameter bits: name -> (bits, size in bytes)."""
    from .device import DeviceArray

    args = list(args)
    if len(args) != len(fn.params):
        raise LaunchError(f"kernel `{fn.name}` takes {len(fn.params)} argument(s), got {len(args)}")
    out = {}
    for k, (param, arg) in enumerate(zip(fn.params, args)):
        dt = NP_TYPES[param.type]
        what = f"argument {k} (`{param.name}`, .{param.type})"
        if isinstance(arg, DeviceArray):
            if arg.device.memory is not mem:
                raise LaunchError(f"{what} belongs to a different device")
            if dt.itemsize != 8 or dt.kind == "f":
                raise LaunchError(f"{what} cannot hold a device pointer")
            bits = arg.addr
        elif isinstance(arg, (bool, np.bool_)):
            raise LaunchError(f"{what}: pass an int, not a bool")
        elif isinstance(arg, (int, np.integer)):
            if dt.kind == "f":
                raise LaunchError(f"{what} is floating-point but got the integer {arg!r}")
            bits = int(arg) % (1 << (8 * dt.itemsize))
        elif isinstance(arg, (float, np.floating)):
            if dt.kind != "f":
                raise LaunchError(f"{what} is an integer but got the float {arg!r}")
            fmt = {2: "<e", 4: "<f", 8: "<d"}[dt.itemsize]
            bits = int.from_bytes(struct.pack(fmt, float(arg)), "little")
        else:
            raise LaunchError(f"{what}: unsupported argument type {type(arg).__name__}")
        out[param.name] = (bits, dt.itemsize)
    return out


def run_kernel(program: Program, mem, *, grid, block, args, smem: int = 0, trace=None,
               max_steps: int | None = None) -> LaunchStats:
    gx, gy, gz = dim3(grid, "grid")
    bx, by, bz = dim3(block, "block")
    threads = bx * by * bz
    if threads > MAX_THREADS_PER_BLOCK:
        raise LaunchError(f"block {(bx, by, bz)} has {threads} threads; the limit is {MAX_THREADS_PER_BLOCK}")
    params = encode_args(program.fn, args, mem=mem)
    if int(smem) != smem or smem < 0:
        raise LaunchError(f"smem must be a non-negative byte count, got {smem!r}")
    smem_bytes = program.smem_bytes(int(smem))
    if smem_bytes > MAX_SMEM_BYTES:
        raise LaunchError(f"kernel `{program.fn.name}` needs {smem_bytes} bytes of shared memory per CTA "
                          f"({program.smem_static} static + {int(smem)} dynamic); the limit is {MAX_SMEM_BYTES}")

    # Per-lane thread coordinates for each warp of a CTA are the same in every CTA.
    nwarps = -(-threads // WARP_SIZE)
    linear = np.arange(nwarps * WARP_SIZE, dtype=np.uint32).reshape(nwarps, WARP_SIZE)
    valid = linear < threads
    tid = (linear % bx, (linear // bx) % by, linear // (bx * by))
    lane = np.arange(WARP_SIZE, dtype=np.uint32)

    def const(v: int) -> np.ndarray:
        return np.full(WARP_SIZE, v, dtype=np.uint32)

    shared = {"ntid.x": const(bx), "ntid.y": const(by), "ntid.z": const(bz),
              "nctaid.x": const(gx), "nctaid.y": const(gy), "nctaid.z": const(gz),
              "laneid": lane, "nwarpid": const(nwarps)}

    stats = LaunchStats(ctas=gx * gy * gz, warps=gx * gy * gz * nwarps,
                        threads=gx * gy * gz * threads, smem_bytes=smem_bytes)
    findings = Findings()
    engine = Engine(max_steps)
    with np.errstate(all="ignore"):
        for cz in range(gz):
            for cy in range(gy):
                for cx in range(gx):
                    ids = {"ctaid.x": const(cx), "ctaid.y": const(cy), "ctaid.z": const(cz)}
                    cta = Cta((cx, cy, cz), smem_bytes, stats.smem, findings)
                    warps = cta.warps
                    for wi in range(nwarps):
                        t = tuple(c[wi] for c in tid)
                        specials = {**shared, **ids, "warpid": const(wi),
                                    "tid.x": t[0], "tid.y": t[1], "tid.z": t[2]}
                        warps.append(Warp(program, cta=cta, warp_id=wi, tid=t,
                                          mask=valid[wi], specials=specials, params=params,
                                          mem=mem, trace=trace))
                    engine.run(cta)
                    for w in warps:
                        stats.instructions += w.instructions
                        stats.lane_instructions += w.lane_instructions
                        stats.divergent_branches += w.divergences
                        stats.max_simt_depth = max(stats.max_simt_depth, w.stack.max_depth)
    stats.findings = findings.items
    return stats
