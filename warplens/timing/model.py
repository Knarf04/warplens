"""An approximate timing model.

Time is in nanoseconds. Each CTA is modeled as running alone on one SM, with
these resources, each a server that handles one piece of work at a time:

    issue slots   4 per SM; warp w uses slot w % 4; one instruction per clock
    SMEM          one shared-memory wavefront per `smem_wavefront_clk`
    TMA engine    `tma_bytes_per_clk`, plus a fixed `tma_latency_ns` per copy
    HBM           this SM's share of the device bandwidth, for bytes not yet in L2
    L2            `l2_bandwidth_per_sm_gb_s`, for bytes some CTA already fetched or stored
    Tensor Core   this SM's share of the device FLOP rate for the MMA's type

A warp's instruction starts when the warp and its issue slot are free. A TMA
copy or an MMA is asynchronous: it queues on its servers and completes later,
and warps waiting on its mbarrier resume at that time. A cluster launch control
request queues on nothing; its response lands a fixed `clc_latency_ns` later.

TMA loads, TMA stores and `cp.async` copies share one L2 record. The global lines
a copy reads or writes are charged to L2 if an earlier copy left them there and
to HBM otherwise; either way they are in L2 once the copy completes.

This explains where time goes; it does not predict it. The microarchitecture
is not public, several parameters are assumptions (the config files say which),
and nothing here has been calibrated against hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .config import GpuConfig
from ..errors import LaunchError

SATURATED = 0.85  # a resource busy at least this fraction of the time is called the bottleneck
LINE = 128  # bytes; granularity of the "already in L2" bookkeeping


@dataclass(slots=True)
class Slice:
    track: str
    name: str
    start: float
    end: float
    cta: tuple = (0, 0, 0)
    args: dict = field(default_factory=dict)


class Server:
    __slots__ = ("name", "free", "busy", "work", "first", "last")

    def __init__(self, name: str):
        self.name = name
        self.free = 0.0  # time the server next becomes available
        self.busy = 0.0  # total time spent working
        self.work = 0.0  # total amount of work (bytes, FLOPs, instructions, wavefronts)
        self.first = None  # start of the first job
        self.last = 0.0  # end of the last job

    def reserve(self, ready: float, duration: float, amount: float) -> tuple[float, float]:
        start = max(ready, self.free)
        end = start + duration
        self.free = end
        self.busy += duration
        self.work += amount
        if self.first is None:
            self.first = start
        self.last = end
        return start, end


# Which config entries each resource's numbers are derived from.
RESOURCE_SOURCES = {
    "issue slots": ("clock_ghz",),
    "SMEM": ("smem_wavefront_clk", "clock_ghz"),
    "TMA engine": ("tma_bytes_per_clk", "tma_latency_ns", "clock_ghz"),
    "HBM": ("hbm_bandwidth_tb_s",),
    "L2": ("l2_bandwidth_per_sm_gb_s",),
    "Tensor Core": ("tensor_core_tflops_f16", "tensor_core_tflops_tf32", "sm_count",
                    "mma_latency_clk", "clock_ghz"),
}


class Timeline:
    """The timing state of one CTA on its SM."""

    def __init__(self, timing: "Timing", cta_id):
        self.timing = timing
        self.cta_id = cta_id
        cfg = timing.config
        self.clk = 1.0 / cfg["clock_ghz"]  # ns per clock
        self.now = 0.0
        self.end = 0.0
        self.slots = [Server("issue slots") for _ in range(4)]
        self.smem = Server("SMEM")
        self.tma = Server("TMA engine")
        self.hbm = Server("HBM")
        self.l2 = Server("L2")
        self.tensor_core = Server("Tensor Core")
        self.tma_rate = cfg["tma_bytes_per_clk"] / self.clk  # bytes per ns
        self.hbm_rate = cfg["hbm_bandwidth_tb_s"] * 1e3 / timing.active_sms  # TB/s -> bytes/ns, shared
        self.l2_rate = cfg["l2_bandwidth_per_sm_gb_s"]  # GB/s == bytes/ns
        self.flop_rate = {kind: cfg[f"tensor_core_tflops_{kind}"] * 1e3 / cfg["sm_count"]
                          for kind in ("f16", "tf32")}  # FLOP per ns for this SM
        self.flops = 0.0
        self.flops_by_kind = {kind: 0.0 for kind in self.flop_rate}
        self.bytes_loaded = 0.0
        # Lines fetched or stored by earlier CTAs are warm at this CTA's time zero.
        # Within a CTA, issuing a copy does not make its lines available yet.
        self.l2_ready: dict[int, float] = {}
        self.async_first = None  # window in which asynchronous work was in flight
        self.async_last = 0.0
        self.mma_done: dict[tuple[int, int], float] = {}  # (warp, lane) -> latest MMA completion
        self.wait_start: dict[int, tuple[float, object]] = {}
        self.run_start: dict[int, float] = {}

    # -- warps ---------------------------------------------------------------------
    def charge(self, warp, ins) -> None:
        """Advance `warp.time` past the instruction it just executed."""
        cfg = self.timing.config
        slot = self.slots[warp.warp_id % 4]
        t = warp.time
        waves, waits = warp.smem_waves, warp.global_accesses
        warp.smem_waves = warp.global_accesses = 0
        # An `atom` or `red` is charged as the load and then the store it stands for: an issue slot
        # each, the wait of a global load, and for shared memory its wavefronts twice
        # (units/atomics.py:_atomic records them doubled). Assumed, not measured.
        parts = ((waves // 2, waits), (waves // 2, 0)) if ins.opcode in ("atom", "red") else ((waves, waits),)
        for waves, waits in parts:
            _, t = slot.reserve(t, self.clk, 1)
            if waves:
                _, t = self.smem.reserve(t, waves * cfg["smem_wavefront_clk"] * self.clk, waves)
            if waits:
                t += cfg["global_access_latency_clk"] * self.clk
        if ins.opcode == "tcgen05" and ins.mods[:1] in (("ld",), ("st",)):
            t += cfg["tmem_access_clk"] * self.clk
        warp.time = t
        self.end = max(self.end, t)

    def slice(self, track: str, name: str, start: float, end: float, **args) -> None:
        self.timing.slices.append(Slice(track, name, start, end, self.cta_id, args))

    def blocked(self, warp, when: float) -> None:
        self.slice(f"warp {warp.warp_id}", "run", self.run_start.get(warp.warp_id, 0.0), when)
        self.wait_start[warp.warp_id] = (when, warp.blocked)

    def resumed(self, warp, when: float) -> None:
        start, wait = self.wait_start.pop(warp.warp_id, (when, None))
        warp.time = max(warp.time, when)
        if wait is not None:
            text = " ".join(wait.ins.text.split())
            self.slice(f"warp {warp.warp_id}", f"wait: {wait.kind}", start, warp.time,
                       line=wait.ins.line, instruction=text)
        self.run_start[warp.warp_id] = warp.time

    def finished(self, warp) -> None:
        self.slice(f"warp {warp.warp_id}", "run", self.run_start.get(warp.warp_id, 0.0), warp.time)

    # -- asynchronous operations -----------------------------------------------------
    def schedule(self, operation) -> float:
        """Completion time of a TMA copy, MMA or commit issued now by its warp."""
        cfg = self.timing.config
        issued = operation.warp.time
        work = operation.work or {}
        thread = (operation.warp.warp_id, operation.lane)
        if operation.kind in ("load", "store", "copy"):  # "copy": cp.async, charged like TMA
            nbytes = work.get("bytes", 0)
            global_bytes = work.get("global_bytes", nbytes)
            # The global lines this copy reads (load, cp.async) or writes (store). A line
            # an earlier copy already brought into L2 moves at the L2 rate. Any other
            # goes through this SM's share of HBM: fetched for a load, and for a store
            # written to a line L2 does not hold. Without lines, all of it is HBM.
            lines = work.get("lines")
            fresh = global_bytes
            new = []
            if lines is not None and len(lines):
                seen = self.timing.l2_lines
                new = [int(x) for x in lines if int(x) not in seen or
                       self.l2_ready.get(int(x), 0.0) > issued]
                seen.update(new)
                fresh = global_bytes * len(new) / len(lines)
            start, end = self.tma.reserve(issued, nbytes / self.tma_rate, nbytes)
            if fresh:
                _, hbm_end = self.hbm.reserve(issued, fresh / self.hbm_rate, fresh)
                end = max(end, hbm_end)
            if global_bytes - fresh:
                _, l2_end = self.l2.reserve(issued, (global_bytes - fresh) / self.l2_rate,
                                          global_bytes - fresh)
                end = max(end, l2_end)
            done = end + cfg["tma_latency_ns"]
            for address in new:
                self.l2_ready[address] = min(self.l2_ready.get(address, done), done)
            self.bytes_loaded += nbytes
            label = "cp.async copy" if operation.kind == "copy" else f"TMA {operation.kind}"
            # Bytes charged to HBM: fetched from it, or for a store, written to it.
            hbm_key = "to_hbm" if operation.kind == "store" else "from_hbm"
            self.slice("TMA engine", label, start, done, bytes=nbytes,
                       issued_ns=issued, line=operation.ins.line, **{hbm_key: int(fresh)})
        elif operation.kind == "clc":
            # A clusterlaunchcontrol.try_cancel response lands a fixed time after the request.
            # How long a launch takes to answer is not known: `clc_latency_ns` is a placeholder.
            # The request uses none of the SM's servers, so it is not part of their busy window.
            done = issued + cfg["clc_latency_ns"]
            self.end = max(self.end, done)
            return done
        elif operation.kind == "mma":
            flops = work.get("flops", 0.0)
            rate = self.flop_rate[work.get("kind", "f16")]
            start, end = self.tensor_core.reserve(issued, flops / rate, flops)
            done = end + cfg["mma_latency_clk"] * self.clk
            self.flops += flops
            self.flops_by_kind[work.get("kind", "f16")] += flops
            self.mma_done[thread] = done
            self.slice("Tensor Core", f"MMA {work.get('shape', '')}", start, done,
                       flops=flops, line=operation.ins.line)
        else:  # tcgen05.commit tracks earlier MMAs issued by this thread only
            done = max(issued, self.mma_done.get(thread, issued))
        if self.async_first is None:
            self.async_first = issued
        self.async_last = max(self.async_last, done)
        self.end = max(self.end, done)
        return done

    # -- summary -------------------------------------------------------------------
    @property
    def did_work(self) -> bool:
        """Whether this CTA used anything beyond issuing instructions: shared memory,
        a TMA or cp.async copy, or the Tensor Core. A CTA of a persistent kernel that
        finds nothing to do and returns has not."""
        return any(server.work for server in
                   (self.smem, self.tma, self.hbm, self.l2, self.tensor_core))

    def utilization(self) -> dict[str, float]:
        total = max(self.end, 1e-12)
        window = max(self.async_last - (self.async_first or 0.0), 1e-12)
        return {
            "issue slots": sum(s.busy for s in self.slots) / (4 * total),
            "SMEM": self.smem.busy / total,
            "TMA engine": self.tma.busy / window,
            "HBM": self.hbm.busy / window,
            "L2": self.l2.busy / window,
            "Tensor Core": self.tensor_core.busy / window,
        }


class Timing:
    """Timing of one launch: a Timeline per CTA plus what they share."""

    def __init__(self, config: GpuConfig, active_sms: int | None = None):
        self.config = config
        # How many SMs draw on HBM at once. Each CTA is simulated alone, so this
        # decides its share of the bandwidth. Default: the whole device is busy.
        if active_sms is None:
            active_sms = int(config["sm_count"])
        if (isinstance(active_sms, (bool, np.bool_)) or not isinstance(active_sms, Integral)
                or not 1 <= active_sms <= config["sm_count"]):
            raise LaunchError(f"active_sms must be an integer from 1 to {int(config['sm_count'])}")
        self.active_sms = int(active_sms)
        self.l2_lines: set[int] = set()  # global-memory lines some CTA has already fetched
        self.timelines: list[Timeline] = []
        self.slices: list[Slice] = []
        self.instants: list[tuple[str, float, tuple]] = []

    def timeline(self, cta_id) -> Timeline:
        self.timelines.append(Timeline(self, cta_id))
        return self.timelines[-1]

    def report(self) -> "TimingReport":
        return TimingReport(self)


class TimingReport:
    def __init__(self, timing: Timing):
        self.config = timing.config
        self.active_sms = timing.active_sms
        self.slices = timing.slices
        self.instants = timing.instants
        lines = timing.timelines
        self.ctas = len(lines)
        self.time_ns = max((t.end for t in lines), default=0.0)
        self.clocks = self.time_ns * self.config["clock_ghz"]
        names = list(RESOURCE_SOURCES)
        # Per-CTA figures are averaged over the CTAs that did work, so that CTAs which
        # start, find nothing to do and return do not dilute them. If no CTA used more
        # than its issue slots (a kernel of plain arithmetic), all of them count.
        working = [t for t in lines if t.did_work] or lines
        self.working_ctas = len(working)

        def mean(group) -> dict[str, float]:
            per_cta = [t.utilization() for t in group]
            return {n: float(np.mean([u[n] for u in per_cta])) if per_cta else 0.0 for n in names}

        self.utilization = mean(working)
        self.utilization_all_ctas = mean(lines)  # the diluted figure, for comparison
        self.flops = sum(t.flops for t in lines)
        self.flops_by_kind = {kind: sum(t.flops_by_kind[kind] for t in lines)
                              for kind in ("f16", "tf32")}
        self.compute_source_keys = tuple(f"tensor_core_tflops_{kind}"
                                         for kind, flops in self.flops_by_kind.items() if flops)
        if not self.compute_source_keys:
            self.compute_source_keys = ("tensor_core_tflops_f16",)
        self.bytes_loaded = sum(t.bytes_loaded for t in lines)
        self.pipeline_ns = float(np.mean([t.async_last - (t.async_first or 0.0) for t in working])) \
            if working else 0.0

    @property
    def bottleneck(self) -> str:
        """The busiest resource if it is saturated, otherwise "latency"."""
        name, value = max(self.utilization.items(), key=lambda kv: kv[1])
        return name if value >= SATURATED else "latency"

    @property
    def verdict(self) -> str:
        name, value = max(self.utilization.items(), key=lambda kv: kv[1])
        if value >= SATURATED:
            return f"{name}-bound: the {name} is busy {value:.0%} of the time"
        return (f"latency-bound: no resource is saturated (the busiest, {name}, is at {value:.0%}); "
                "time goes to waiting for round trips")

    def roofline(self) -> dict[str, float]:
        """Achieved rate against this SM's compute and memory ceilings, per CTA."""
        seconds = max(self.pipeline_ns, 1e-12) * 1e-9
        per_cta_flops = self.flops / max(self.working_ctas, 1)  # as pipeline_ns: working CTAs
        per_cta_bytes = self.bytes_loaded / max(self.working_ctas, 1)
        intensity = per_cta_flops / per_cta_bytes if per_cta_bytes else float("inf")
        # A mixed workload's ceiling weights time at each dtype's peak rate.
        peak_seconds = sum(flops / (self.config[f"tensor_core_tflops_{kind}"] * 1e12 /
                                     self.config["sm_count"])
                           for kind, flops in self.flops_by_kind.items())
        peak = self.flops / peak_seconds if peak_seconds else \
            self.config["tensor_core_tflops_f16"] * 1e12 / self.config["sm_count"]
        memory = self.config["hbm_bandwidth_tb_s"] * 1e12 / self.active_sms
        return {"achieved_flops_per_s": per_cta_flops / seconds,
                "compute_ceiling_flops_per_s": peak,
                "memory_ceiling_flops_per_s": intensity * memory,
                "flops_per_byte": intensity}

    def summary(self) -> str:
        cfg = self.config
        lines = [f"Approximate timing on {cfg.name} (model, not a measurement)",
                 f"  {self.ctas} CTA(s), each alone on one SM; {self.active_sms} SMs assumed to share HBM",
                 f"  kernel time {self.time_ns:,.0f} ns ({self.clocks:,.0f} clocks at "
                 f"{cfg['clock_ghz']:g} GHz, an assumed clock)",
                 f"  verdict: {self.verdict}"]
        if self.working_ctas < self.ctas:
            lines.append(f"  busy is averaged over the {self.working_ctas} CTA(s) that did work; "
                         f"{self.ctas - self.working_ctas} did none and are left out")
        lines.append("  resource       busy   derived from")
        for name, value in self.utilization.items():
            keys = ", ".join(k + ("*" if cfg.params[k].is_assumption else "")
                             for k in RESOURCE_SOURCES[name])
            if name == "HBM":
                keys += ", active_sms (launch parameter)"
            lines.append(f"  {name:<13} {value:>5.0%}   {keys}")
        r = self.roofline()
        if self.flops:
            lines.append(f"  roofline: {r['achieved_flops_per_s'] / 1e12:.2f} TFLOP/s pipeline rate per SM; "
                         f"compute ceiling {r['compute_ceiling_flops_per_s'] / 1e12:.2f}, memory "
                         f"ceiling {r['memory_ceiling_flops_per_s'] / 1e12:.2f} at "
                         f"{r['flops_per_byte']:.1f} FLOP/byte")
        lines.append("  * assumed value in this model; see the config file for every source")
        return "\n".join(lines)

    def write_trace(self, path) -> None:
        from ..viz.trace_perfetto import write_trace
        write_trace(self, path)

    def write_html(self, path, title: str = "warplens timing report") -> None:
        from ..viz.html_report import write_html
        write_html(self, path, title)
