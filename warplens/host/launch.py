"""Kernel launch: turn (grid, block, args) into CTAs and warps, then run them."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..core.engine import Engine
from ..core.program import Program
from ..core.warp import WARP_SIZE, Warp
from ..errors import LaunchError
from ..frontend.ir import NP_TYPES, Function

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


def run_kernel(program: Program, mem, *, grid, block, args, trace=None,
               max_steps: int | None = None) -> LaunchStats:
    gx, gy, gz = dim3(grid, "grid")
    bx, by, bz = dim3(block, "block")
    threads = bx * by * bz
    if threads > MAX_THREADS_PER_BLOCK:
        raise LaunchError(f"block {(bx, by, bz)} has {threads} threads; the limit is {MAX_THREADS_PER_BLOCK}")
    params = encode_args(program.fn, args, mem=mem)

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
                        threads=gx * gy * gz * threads)
    engine = Engine(max_steps)
    with np.errstate(all="ignore"):
        for cz in range(gz):
            for cy in range(gy):
                for cx in range(gx):
                    cta = {"ctaid.x": const(cx), "ctaid.y": const(cy), "ctaid.z": const(cz)}
                    warps = []
                    for wi in range(nwarps):
                        t = tuple(c[wi] for c in tid)
                        specials = {**shared, **cta, "warpid": const(wi),
                                    "tid.x": t[0], "tid.y": t[1], "tid.z": t[2]}
                        warps.append(Warp(program, cta=(cx, cy, cz), warp_id=wi, tid=t,
                                          mask=valid[wi], specials=specials, params=params,
                                          mem=mem, trace=trace))
                    engine.run(warps)
                    for w in warps:
                        stats.instructions += w.instructions
                        stats.lane_instructions += w.lane_instructions
                        stats.divergent_branches += w.divergences
                        stats.max_simt_depth = max(stats.max_simt_depth, w.stack.max_depth)
    return stats
