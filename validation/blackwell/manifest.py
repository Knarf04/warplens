"""Every run of the Blackwell probe suite: which program, which arguments, what it tests.

One entry is one process. Its name is `<area>/<short name>` as in docs/validation-matrix.md
where a row of the matrix names it, with a `.case` suffix for further cases of the same
program; the prediction for it is `predicted/<name>.txt`.

    python validation/blackwell/manifest.py          # print the table
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str  # <area>/<short name>[.case]
    program: str  # source without .cu, relative to validation/blackwell
    args: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()  # rows of docs/validation-matrix.md
    claim: str = ""  # one line: what the simulator claims, which this run tests
    # What may happen on hardware BY DESIGN, besides running to the end:
    #   ""       nothing: a timeout or a crash is a failure of the probe
    #   "hang"   the kernel may never return (always run under the timeout)
    #   "fault"  the driver or the hardware may reject or fault the kernel
    risk: str = ""
    # Where the prediction comes from:
    #   "sim"        the program's standard output under `warplens run`
    #   "numerics"   formats/predict_small_formats.py (the simulator cannot run the kernel)
    #   "none"       no prediction: the output is new information
    predict: str = "sim"
    # How compare.py judges a difference: "exact", or "ulp" (float32 hex words may differ;
    # the distance in units in the last place is reported).
    compare: str = "exact"
    timeout: int = 120  # seconds, on the GPU


def _tma() -> list[Probe]:
    return [
        Probe("tma/swizzle_aligned", "tma/swizzle", ("aligned",), ("TMA-01",),
              "a swizzled load XORs the 16-byte chunk index with the low 1/2/3 bits of the 128-byte row index"),
        Probe("tma/swizzle_misaligned", "tma/swizzle", ("misaligned",), ("TMA-02",),
              "the swizzle is a function of the shared ADDRESS, not of the offset into the buffer"),
        Probe("tma/swizzle_misaligned.sixteen", "tma/swizzle", ("sixteen",), ("TMA-02",),
              "the same with a destination 16 bytes into a 128-byte row", risk="fault"),
        Probe("tma/ranks", "tma/ranks", (), ("TMA-03", "TMA-15"),
              "rank 1-5 boxes land densely, dimension 0 fastest; complete_tx reports box bytes; "
              "the tensor map is a by-value kernel parameter"),
        Probe("tma/misaligned_requests", "tma/ranks", ("unaligned",), ("TMA-10",),
              "an innermost coordinate that is not a multiple of 16 bytes is refused", risk="fault"),
        Probe("tma/oob_fill", "tma/oob_fill", (), ("TMA-04", "TMA-05", "TMA-06"),
              "out-of-bounds elements are zero or the default quiet NaN; complete_tx counts the whole box"),
        Probe("tma/oob_complete_tx", "tma/oob_fill", ("inbounds",), ("TMA-06",),
              "expect_tx of only the in-bounds bytes does NOT complete the phase"),
        Probe("tma/element_strides", "tma/element_strides", (), ("TMA-07",),
              "element stride s moves ceil(box / s) elements, packed densely, and reports those bytes"),
        Probe("tma/element_strides.inner", "tma/element_strides", ("inner",), ("TMA-07",),
              "an element stride in dimension 0 is ignored", risk="fault"),
        Probe("tma/store_wait_group", "tma/store", (), ("TMA-09", "TMA-01", "TMA-03"),
              "a store is in global memory after commit_group + wait_group 0; it reads through the load's swizzle"),
        Probe("tma/store_wait_group.edge", "tma/store", ("edge",), ("TMA-04",),
              "a store writes only the elements of the box that are inside the tensor", risk="fault"),
        Probe("tma/store_wait_group.outside", "tma/store", ("outside",), ("TMA-04",),
              "a store wholly outside the tensor writes nothing", risk="fault"),
        Probe("tma/tensormap_encode_sweep", "tma/tensormap_encode_sweep", (), ("TMA-11", "TMA-12"),
              "none: records the driver's 128-byte tensor maps over a parameter sweep", predict="none"),
    ]


def _mbarrier() -> list[Probe]:
    return [
        Probe("mbarrier/phase_parity", "mbarrier/script", ("basic",), ("MB-01",),
              "a new mbarrier is in phase 0; test_wait.parity p is true when the phase's parity is not p"),
        Probe("mbarrier/phase_parity.counts", "mbarrier/script", ("count",), ("MB-01", "MB-02"),
              "the arrival count is reloaded at each completion; arrive with a count; init restarts"),
        Probe("mbarrier/arrivals_and_bytes", "mbarrier/script", ("tx",), ("MB-02",),
              "a phase completes when pending arrivals and the byte count are both zero, in either order"),
        Probe("mbarrier/lane_arrivals", "mbarrier/lane_arrivals", (), ("MB-09",),
              "every lane that executes an arrive arrives once, also when only part of a warp does"),
        Probe("mbarrier/complete_before_expect", "mbarrier/script", ("negative",), ("MB-07",),
              "the byte count is signed: complete_tx before expect_tx completes once balanced", risk="fault"),
        Probe("mbarrier/over_arrival", "mbarrier/script", ("over",), ("MB-04",),
              "arrivals beyond the pending count complete the phase and carry into the next", risk="fault"),
        Probe("mbarrier/expect_tx_small", "mbarrier/expect_tx", (), ("MB-05", "MB-06", "MB-07", "MB-08"),
              "undersized, oversized and missing expect_tx against real TMA loads, with bounded polls"),
        Probe("mbarrier/expect_tx_small.wait", "mbarrier/expect_tx", ("wait-small",), ("MB-05", "MB-11"),
              "an undersized expect_tx with one load never completes: the usual wait loop deadlocks",
              risk="hang", timeout=30),
        Probe("mbarrier/expect_tx_large", "mbarrier/expect_tx", ("wait-large",), ("MB-08", "MB-11"),
              "an oversized expect_tx never completes: the usual wait loop deadlocks", risk="hang", timeout=30),
    ]


def _tmem() -> list[Probe]:
    probes = [
        Probe("tmem/lane_window", "tmem/lane_window", (), ("TM-01", "TM-02"),
              "a warp reaches only lanes 32 (w % 4) .. + 31; a lane field naming another window reads its own"),
    ]
    probes += [Probe(f"tmem/lane_window.{lane}", "tmem/lane_window", (str(lane),), ("TM-03",),
                     f"lane field {lane}: a 32x32b transfer running past the warp's window is refused",
                     risk="fault") for lane in (16, 1, 48)]
    probes += [
        Probe("tmem/layout_f", "tmem/layout_f", (), ("TM-04", "TM-01"),
              "M = 128: row m on lane m. M = 64: row m on lane 32 (m / 16) + m % 16 + a, a = 0 or 16"),
        Probe("tmem/layout_f.64_8", "tmem/layout_f", ("64", "8"), ("TM-04",),
              "M = 64 with lane offset 8: the simulator just adds the offset", risk="fault"),
        Probe("tmem/transfer_shapes", "tmem/transfer_shapes", (), ("TM-05",),
              "thread and register to (lane, column) for 32x32b, 16x64b, 16x128b, 16x256b, load and store"),
        Probe("tmem/alloc_addresses", "tmem/alloc", (), ("TM-07",),
              "alloc returns the lowest free block aligned to its size, as (lane 0, column)"),
        Probe("tmem/alloc_increasing", "tmem/alloc", ("increasing",), ("TM-08",),
              "a later allocation may not be larger than an earlier one", risk="fault"),
    ]
    probes += [Probe(f"tmem/alloc_addresses.size_{n}", "tmem/alloc", (f"size-{n}",), ("TM-07",),
                     f"an allocation of {n} columns is refused", risk="fault") for n in (48, 16, 0, 1024)]
    probes += [
        Probe("tmem/alloc_after_relinquish", "tmem/alloc", ("after-relinquish",), ("TM-09",),
              "no allocation after relinquish_alloc_permit", risk="fault"),
        Probe("tmem/alloc_blocks", "tmem/alloc", ("exhaust",), ("TM-10",),
              "an allocation that does not fit waits (here: forever)", risk="hang", timeout=30),
        Probe("tmem/dealloc_repeated", "tmem/alloc", ("dealloc-twice",), ("TM-11",),
              "a second dealloc of the same columns is ignored", risk="hang", timeout=30),
        Probe("tmem/dealloc_mismatch", "tmem/alloc", ("dealloc-wrong-size",), ("TM-12",),
              "a dealloc with another size than the allocation's is refused", risk="hang", timeout=30),
        Probe("tmem/alloc_leak", "tmem/alloc", ("leak",), (),
              "columns never released are reported and do not affect the next launch", risk="hang", timeout=30),
    ]
    return probes


def _mma() -> list[Probe]:
    return [
        Probe("mma/operand_layouts", "mma/operand_layouts", (), ("MMA-01", "MMA-02", "MMA-03"),
              "K-major (f16, tf32) and MN-major (f16) operand addressing in all four swizzle modes; exact results"),
        Probe("mma/operand_layouts.tf32_mn", "mma/operand_layouts", ("tf32-mn",), ("MMA-03",),
              "MN-major tf32 is refused (needs a 128B swizzle with 32-byte atomicity)", risk="fault"),
        Probe("mma/accumulation_order", "mma/accumulation_order", (), ("MMA-04", "MMA-05"),
              "none exactly: the simulator sums in float32 in NumPy's order; hardware may differ in the last bits",
              compare="ulp"),
        Probe("mma/negate", "mma/negate", (), ("MMA-07", "MMA-05"),
              "each negate bit flips the product's sign, both leave it; a second MMA accumulates"),
        Probe("mma/tf32_low_bits", "mma/tf32_low_bits", (), ("MMA-08",),
              "kind::tf32 multiplies the operands as stored float32 values; low bits take part"),
    ]


def _cluster() -> list[Probe]:
    probes = [
        Probe("cluster/barrier", "cluster/barrier", (), ("CL-01", "CL-03", "CL-04"),
              "the cluster barrier holds every thread until all have arrived; mapa + ld.shared::cluster"),
        Probe("cluster/multicast", "cluster/multicast", (), ("CL-07",),
              "a multicast load writes the box in every CTA of the mask and completes each one's mbarrier"),
        Probe("cluster/multicast.outside", "cluster/multicast", ("outside",), ("CL-07",),
              "a mask naming a rank outside the cluster is refused", risk="fault"),
        Probe("cluster/pair_accounting", "cluster/pair_accounting", (), ("CL-08",),
              "with .cta_group::2 a pair's completions go to the member with the mbarrier address's parity"),
    ]
    for m, issuer, ids in ((256, 0, ("CL-10", "CL-13", "CL-09")), (128, 0, ("CL-11", "CL-09")),
                           (256, 1, ("CL-12",)), (128, 1, ("CL-12",))):
        probes.append(Probe(
            f"cluster/pair_mma.m{m}_issuer{issuer}", "cluster/pair_mma", (str(m), str(issuer)), ids,
            f"two-CTA MMA, M = {m}, issued by rank {issuer}: " +
            ("each CTA holds 128 rows on 128 lanes" if m == 256 else
             "Layout B: 64 rows per CTA, columns folded onto lanes 64-127"),
            risk="hang", timeout=30))
    return probes


def _formats() -> list[Probe]:
    return [
        Probe("formats/fp8_codes", "formats/small_formats", ("fp8",), ("NUM-01", "NUM-02"),
              "E4M3 and E5M2: every code decodes as modeled; encoding rounds to nearest even and saturates",
              predict="numerics"),
        Probe("formats/fp4_codes", "formats/small_formats", ("fp4",), ("NUM-03",),
              "E2M1: 16 codes; ties to even, saturating at 6", predict="numerics"),
        Probe("formats/e8m0_ties", "formats/small_formats", ("e8m0",), ("NUM-04",),
              "UE8M0: code c is 2^(c - 127); a midpoint between two powers of two goes to the larger",
              predict="numerics"),
        Probe("formats/special_inputs", "formats/small_formats", ("special",), ("NUM-06",),
              "NaN, infinities, zeros and out-of-range inputs for the four formats", predict="numerics"),
        Probe("formats/tf32_modes", "formats/tf32_modes", (), ("NUM-07", "NUM-08"),
              "cvt.rn/.rz.tf32.f32 round the magnitude bits like .rna with their own tie rule; .rna as on Ampere"),
    ]


def _device() -> list[Probe]:
    return [
        Probe("e2e/device_attributes", "e2e/device_attributes", (), ("RT-03",),
              "the device reports compute capability 10.0, 148 SMs and the shared-memory limits the shim gives"),
    ]


PROBES: list[Probe] = _device() + _tma() + _mbarrier() + _tmem() + _mma() + _cluster() + _formats()
BY_NAME = {probe.name: probe for probe in PROBES}
assert len(BY_NAME) == len(PROBES), "duplicate probe names"

# The end-to-end runs are driven by e2e/run_e2e.py; these are the names it records.
E2E = {
    "e2e/cutlass_tutorials.tutorial_01": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_02": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_03": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_04": ("RT-06",),
    "e2e/triton": ("RT-07",),
    "e2e/book_hgemm": ("RT-08",),
}


if __name__ == "__main__":
    for probe in PROBES:
        flags = ", ".join(x for x in (probe.risk and f"may {probe.risk}",
                                      probe.predict != "sim" and f"prediction: {probe.predict}",
                                      probe.compare != "exact" and f"compare: {probe.compare}") if x)
        print(f"{probe.name:40} {probe.program} {' '.join(probe.args)}".rstrip())
        print(f"{'':40} {', '.join(probe.ids) or '-'}: {probe.claim}" + (f" [{flags}]" if flags else ""))
