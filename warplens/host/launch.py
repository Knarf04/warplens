"""Kernel launch: turn (grid, block, args) into CTAs and warps, then run them."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..check.report import Finding, Findings
from ..core.cluster import Cluster
from ..core.cta import Cta
from ..core.engine import Engine
from ..core.launch_queue import LaunchQueue
from ..core.program import Program
from ..core.warp import WARP_SIZE, Warp
from ..errors import LaunchError
from ..frontend.ir import NP_TYPES, Function
from ..units.smem import MAX_SMEM_BYTES
from ..units.smem_banks import SmemReport

MAX_THREADS_PER_BLOCK = 1024
MAX_CLUSTER_CTAS = 16  # a multicast mask has 16 bits


@dataclass(slots=True)
class LaunchStats:
    ctas: int = 0  # of the grid; `canceled_ctas` of them never started
    canceled_ctas: int = 0  # in clusters canceled by clusterlaunchcontrol.try_cancel
    warps: int = 0
    threads: int = 0
    instructions: int = 0  # instructions issued, counted once per warp
    lane_instructions: int = 0  # the same, counted once per active lane
    divergent_branches: int = 0
    max_simt_depth: int = 0
    smem_bytes: int = 0  # shared memory per CTA
    smem: SmemReport = field(default_factory=SmemReport, repr=False)  # bank conflicts
    findings: list[Finding] = field(default_factory=list, repr=False)
    output: str = field(default="", repr=False)  # what the kernel printed with printf
    timing: object = field(default=None, repr=False)  # timing.model.TimingReport of a timed launch

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


def _struct_bytes(arg) -> bytes | None:
    """Raw bytes of a by-value struct argument, or None if `arg` is not one."""
    import ctypes
    if isinstance(arg, (bytes, bytearray, memoryview)):
        return bytes(arg)
    if isinstance(arg, (np.ndarray, np.void)):
        return arg.tobytes()
    if isinstance(arg, (ctypes.Structure, ctypes.Union, ctypes.Array)):
        return bytes(arg)
    return None


def encode_args(fn: Function, args, *, mem) -> dict[str, bytes]:
    """Convert host arguments to the raw little-endian bytes of each parameter."""
    from .device import DeviceArray

    args = list(args)
    if len(args) != len(fn.params):
        raise LaunchError(f"kernel `{fn.name}` takes {len(fn.params)} argument(s), got {len(args)}")
    out = {}
    for k, (param, arg) in enumerate(zip(fn.params, args)):
        dt = NP_TYPES[param.type]
        what = f"argument {k} (`{param.name}`, .{param.type})"
        if param.by_value:  # a struct passed by value
            raw = _struct_bytes(arg)
            if raw is None:
                raise LaunchError(f"argument {k} (`{param.name}`) is a {param.size}-byte struct passed "
                                  f"by value; pass bytes, a NumPy record or a ctypes structure, "
                                  f"not {type(arg).__name__}")
            if len(raw) != param.size:
                raise LaunchError(f"argument {k} (`{param.name}`) must be {param.size} bytes, got {len(raw)}")
            out[param.name] = raw
            continue
        if isinstance(arg, (bytes, bytearray)):  # raw bytes, as a compiled host program passes them
            if len(arg) != dt.itemsize:
                raise LaunchError(f"{what} must be {dt.itemsize} bytes, got {len(arg)}")
            out[param.name] = bytes(arg)
            continue
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
        out[param.name] = bits.to_bytes(dt.itemsize, "little")
    return out


def run_kernel(program: Program, mem, *, grid, block, args, smem: int = 0, trace=None,
               max_steps: int | None = None, async_delay: int = 4, timing=None,
               cluster=None, clc_limit: int | None = None) -> LaunchStats:
    if clc_limit is not None and (not isinstance(clc_limit, (int, np.integer))
                                  or isinstance(clc_limit, (bool, np.bool_)) or clc_limit < 0):
        raise LaunchError(f"clc_limit must be None or a non-negative integer, got {clc_limit!r}")
    if (not isinstance(async_delay, (int, np.integer))
            or isinstance(async_delay, (bool, np.bool_)) or async_delay < 0):
        raise LaunchError(f"async_delay must be a non-negative integer, got {async_delay!r}")
    async_delay = int(async_delay)
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
    explicit_cluster = cluster is not None
    cluster = dim3(cluster, "cluster") if explicit_cluster else (1, 1, 1)
    if cluster[0] * cluster[1] * cluster[2] > MAX_CLUSTER_CTAS:
        raise LaunchError(f"cluster {cluster} has more than {MAX_CLUSTER_CTAS} CTAs")
    if gx % cluster[0] or gy % cluster[1] or gz % cluster[2]:
        raise LaunchError(f"grid {(gx, gy, gz)} is not a whole number of clusters of shape {cluster}")
    output: list[str] = []
    if timing is None:
        engine = Engine(max_steps)
    else:
        from ..timing.engine import TimedEngine
        engine = TimedEngine(max_steps)
    kx, ky, kz = cluster
    cluster_specials = {"cluster_nctaid.x": const(kx), "cluster_nctaid.y": const(ky),
                        "cluster_nctaid.z": const(kz), "cluster_nctarank": const(kx * ky * kz),
                        "nclusterid.x": const(gx // kx), "nclusterid.y": const(gy // ky),
                        "nclusterid.z": const(gz // kz),
                        "is_explicit_cluster": const(int(explicit_cluster))}
    # Clusters start one at a time, in this order. A running cluster can cancel those still
    # in the queue with clusterlaunchcontrol.try_cancel; they then never start.
    queue = LaunchQueue([(qx, qy, qz) for qz, qy, qx in np.ndindex(gz // kz, gy // ky, gx // kx)],
                        cluster, None if clc_limit is None else int(clc_limit))
    with np.errstate(all="ignore"):
        while (started := queue.start_next()) is not None:
            qx, qy, qz = started
            group = Cluster(started, cluster)
            group.launch = queue
            # CTAs within a cluster overlap in time, so they must agree when
            # fetched lines become readable in L2. Earlier clusters have
            # finished and remain warm through Timing.l2_lines.
            cluster_l2_ready: dict[int, float] = {}
            for rz, ry, rx in np.ndindex(kz, ky, kx):  # its CTAs in rank order
                cx, cy, cz = qx * kx + rx, qy * ky + ry, qz * kz + rz
                ids = {"ctaid.x": const(cx), "ctaid.y": const(cy), "ctaid.z": const(cz),
                       "clusterid.x": const(qx), "clusterid.y": const(qy), "clusterid.z": const(qz),
                       "cluster_ctaid.x": const(rx), "cluster_ctaid.y": const(ry),
                       "cluster_ctaid.z": const(rz),
                       "cluster_ctarank": const(rx + kx * (ry + ky * rz))}
                cta = Cta((cx, cy, cz), smem_bytes, stats.smem, findings, async_delay)
                cta.output = output
                group.add(cta)
                if timing is not None:
                    cta.timeline = timing.timeline(cta.id)
                    cta.timeline.l2_ready = cluster_l2_ready
                for wi in range(nwarps):
                    t = tuple(c[wi] for c in tid)
                    specials = {**shared, **cluster_specials, **ids, "warpid": const(wi),
                                "tid.x": t[0], "tid.y": t[1], "tid.z": t[2]}
                    cta.warps.append(Warp(program, cta=cta, warp_id=wi, tid=t,
                                          mask=valid[wi], specials=specials, params=params,
                                          mem=mem, trace=trace))
                group.warps.extend(cta.warps)
            engine.run(group)
            queue.close(group)
            for w in group.warps:
                stats.instructions += w.instructions
                stats.lane_instructions += w.lane_instructions
                stats.divergent_branches += w.divergences
                stats.max_simt_depth = max(stats.max_simt_depth, w.stack.max_depth)
    stats.canceled_ctas = queue.canceled * kx * ky * kz
    stats.findings = findings.items
    if timing is not None:
        stats.timing = timing.report()
    stats.output = "".join(output)
    return stats
