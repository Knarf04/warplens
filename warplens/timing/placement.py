"""How many SMs a launch keeps busy: an assumption derived from the launch's shape.

The timing model simulates each CTA alone on one SM and divides the HBM bandwidth by
`active_sms`, a launch argument that defaults to every SM of the device. That default is
right for a launch that fills the device and wrong for a launch of one CTA. `place` gives
the number a launch's own shape implies:

    CTAs per SM   the most CTAs of this kernel one SM holds at once: the smallest of
                  shared memory per SM / shared memory per CTA, warps per SM / warps per
                  CTA, and the limit on resident CTAs
    resident      CTAs running at once: the grid if it fits, otherwise every SM full
                  (a whole number of clusters)
    active SMs    SMs holding at least one resident CTA

Two things about the scheduler are assumed, because nothing public states them:

1. CTAs are spread: no SM gets a second CTA while another SM has none. So a grid of N CTAs
   keeps min(N, SM count) SMs busy. (A scheduler that filled one SM before the next would
   keep ceil(N / CTAs per SM) busy; `Placement.packed_sms` is that number.)
2. The CTAs of a cluster run on different SMs, and a cluster starts only as a whole.

Not modeled: registers per thread as an occupancy limit (the PTX does not say how many
ptxas allots), the kilobyte of shared memory the driver keeps per CTA, how SMs are grouped
into GPCs (a cluster must fit in one), and other work on the GPU.

Only the standard library is used, so the calibration scripts can import this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Per-SM limits of compute capability 10.x, from the table of technical specifications in
# the CUDA C++ Programming Guide. The same numbers the shim reports to a program
# (cudaGetDeviceProperties), and the timing benchmarks print what a GPU reports in their
# DEVICE line (smem_per_sm_bytes, max_threads_per_sm, max_blocks_per_sm).
SMEM_PER_SM_BYTES = 228 * 1024
THREADS_PER_SM = 2048
CTAS_PER_SM = 32
WARP_SIZE = 32
LIMITS = {"smem_per_sm_bytes": SMEM_PER_SM_BYTES, "max_threads_per_sm": THREADS_PER_SM,
          "max_blocks_per_sm": CTAS_PER_SM}


@dataclass(frozen=True, slots=True)
class Placement:
    grid_ctas: int       # CTAs of the launch
    cluster_ctas: int    # CTAs per cluster (1 without a cluster shape)
    block_threads: int
    smem_bytes: int      # shared memory per CTA, static plus dynamic
    sm_count: int
    ctas_per_sm: int     # occupancy: the most CTAs of this kernel on one SM
    limited_by: str      # which limit gives ctas_per_sm: "shared memory", "threads" or "CTA count"
    resident_ctas: int   # CTAs running at once
    active_sms: int      # SMs with at least one of them (CTAs spread over the SMs)
    packed_sms: int      # the same if each SM were filled before the next one

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def derivation(self) -> str:
        """One line that says where `active_sms` comes from."""
        clusters = f" in clusters of {self.cluster_ctas}" if self.cluster_ctas > 1 else ""
        fits = "all resident" if self.resident_ctas == self.grid_ctas else f"{self.resident_ctas} resident at once"
        return (f"grid of {self.grid_ctas} CTA(s){clusters}, {self.block_threads} threads and "
                f"{self.smem_bytes} bytes of shared memory each: at most {self.ctas_per_sm} per SM "
                f"({self.limited_by}), {fits} on {self.active_sms} of {self.sm_count} SMs")


def place(grid_ctas: int, *, sm_count: int, cluster_ctas: int = 1, block_threads: int = 32,
          smem_bytes: int = 0, limits: dict | None = None) -> Placement:
    """The placement of a launch of `grid_ctas` CTAs on a device of `sm_count` SMs."""
    limits = {**LIMITS, **(limits or {})}
    grid_ctas, cluster_ctas, sm_count = int(grid_ctas), int(cluster_ctas), int(sm_count)
    block_threads, smem_bytes = int(block_threads), int(smem_bytes)
    if grid_ctas < 1 or cluster_ctas < 1 or sm_count < 1 or block_threads < 1 or smem_bytes < 0:
        raise ValueError("a launch has at least one CTA, one thread and one SM, and no negative size")
    if grid_ctas % cluster_ctas:
        raise ValueError(f"a grid of {grid_ctas} CTAs is not a whole number of clusters of {cluster_ctas}")
    if cluster_ctas > sm_count:
        raise ValueError(f"a cluster of {cluster_ctas} CTAs does not fit on {sm_count} SMs")
    warps = -(-block_threads // WARP_SIZE)
    candidates = {"CTA count": int(limits["max_blocks_per_sm"]),
                  "threads": int(limits["max_threads_per_sm"]) // (warps * WARP_SIZE)}
    if smem_bytes:
        candidates["shared memory"] = int(limits["smem_per_sm_bytes"]) // smem_bytes
    # On a tie the first of shared memory, threads, CTA count is named.
    limited_by = min(("shared memory", "threads", "CTA count"),
                     key=lambda name: candidates.get(name, float("inf")))
    ctas_per_sm = candidates[limited_by]
    if ctas_per_sm < 1:
        raise ValueError(f"one CTA of {block_threads} threads and {smem_bytes} bytes of shared memory "
                         f"does not fit on an SM ({limited_by})")
    capacity = sm_count * ctas_per_sm // cluster_ctas * cluster_ctas  # whole clusters
    resident = min(grid_ctas, capacity)
    return Placement(grid_ctas=grid_ctas, cluster_ctas=cluster_ctas, block_threads=block_threads,
                     smem_bytes=smem_bytes, sm_count=sm_count, ctas_per_sm=ctas_per_sm,
                     limited_by=limited_by, resident_ctas=resident,
                     active_sms=min(sm_count, resident),
                     packed_sms=min(sm_count, max(cluster_ctas, -(-resident // ctas_per_sm))))
