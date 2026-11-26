"""Every run of the Blackwell probe suite: which program, which arguments, what it tests.

One entry is one process. Its name is `<area>/<short name>` as in docs/validation-matrix.md
where a row of the matrix names it, with a `.case` suffix for further cases of the same
program; the prediction for it is `predicted/<name>.txt`.

    python validation/blackwell/manifest.py          # print the table
    python validation/blackwell/manifest.py --tables # the tables of README.md, as Markdown
    python validation/blackwell/manifest.py --update-readme   # write them into README.md
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
    #   "numerics"   formats/predict_small_formats.py (for a kernel the simulator cannot run; none now)
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
              "a destination 16 bytes into a 128-byte row is refused", risk="fault"),
        Probe("tma/swizzle_misaligned.thirtytwo", "tma/swizzle", ("thirtytwo",), ("TMA-02",),
              "a destination 32 bytes into a 128-byte row is refused: a tensor copy needs a shared address "
              "aligned to 128 bytes", risk="fault"),
        Probe("tma/swizzle_misaligned.sixtyfour", "tma/swizzle", ("sixtyfour",), ("TMA-02",),
              "a destination 64 bytes into a 128-byte row is refused", risk="fault"),
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
              "a store that hangs over the last row writes the elements inside; one that starts before the first "
              "row ends the kernel", risk="fault"),
        Probe("tma/store_wait_group.outside", "tma/store", ("outside",), ("TMA-04",),
              "a store wholly outside the tensor writes nothing", risk="fault"),
        Probe("tma/tensormap_encode_sweep", "tma/tensormap_encode_sweep", (), ("TMA-11", "TMA-12"),
              "none: records the driver's 128-byte tensor maps over a parameter sweep", predict="none"),
        Probe("tma/raw_bulk", "tma/raw_bulk", (), ("TMA-08",),
              "cp.async.bulk without a tensor map copies 16, 64 and 4096 bytes verbatim in both directions "
              "and reports the size to the mbarrier"),
        Probe("tma/hints", "tma/hints", (), ("TMA-14",),
              "a cache-hint operand, prefetch.tensormap and cp.async.bulk.prefetch do not change what a load loads"),
        Probe("tma/missing_fence", "tma/missing_fence", (), ("TMA-13",),
              "a TMA store right after ordinary stores carries the current bytes with and without "
              "fence.proxy.async (2000 stores per variant)"),
        Probe("tma/missing_fence.long", "tma/missing_fence", ("100000",), ("TMA-13",),
              "none: the same with 10^5 stores per variant, too slow to simulate; the prediction of "
              "tma/missing_fence applies line for line", predict="none"),
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
        Probe("mbarrier/arrive_token", "mbarrier/arrive_token", (), ("MB-03",),
              "test_wait with an arrive token is false while the token's phase is open and true once it "
              "completed, also during the next phase"),
        Probe("mbarrier/arrive_token.stale", "mbarrier/arrive_token", ("stale",), ("MB-03",),
              "a token two phases old is invalid PTX use and is diagnosed; its predicate is informational only"),
        Probe("mbarrier/ampere_token", "mbarrier/ampere_token", (), ("MB-03",),
              "as on an RTX 3080 Ti (validation/ampere/mbarrier_token_golden.txt): test_wait answers from bit 63 "
              "of the token, the parity of the arrival's phase. For tokens two or more phases old and made-up "
              "tokens (invalid PTX use) a difference is Blackwell's, not a simulator error"),
        Probe("mbarrier/try_wait", "mbarrier/try_wait", (), ("MB-01", "MB-03"),
              "one try_wait, in the parity and the token form, is true once the phase it names has completed"),
        Probe("mbarrier/try_wait.incomplete", "mbarrier/try_wait", ("incomplete",), ("MB-11",),
              "none: the simulator's try_wait blocks (a deadlock report here). Records whether one try_wait "
              "on a phase that never completes returns false, and after how many clock ticks",
              risk="hang", predict="none", timeout=30),
        Probe("mbarrier/init_count", "mbarrier/init_count", (), ("MB-10",),
              "mbarrier.init accepts 2^20 - 1 arrivals, and an arrive with that count completes the phase"),
        Probe("mbarrier/init_count.over", "mbarrier/init_count", ("over",), ("MB-10",),
              "an arrival count of 2^20 is refused", risk="fault"),
        Probe("mbarrier/init_count.zero", "mbarrier/init_count", ("zero",), ("MB-10",),
              "an arrival count of 0 is refused", risk="fault"),
        Probe("mbarrier/complete_before_expect", "mbarrier/script", ("negative",), ("MB-07",),
              "the byte count is signed: complete_tx before expect_tx completes once balanced", risk="fault"),
        Probe("mbarrier/over_arrival", "mbarrier/script", ("over",), ("MB-04",),
              "arrivals beyond the pending count end the kernel", risk="fault"),
        Probe("mbarrier/expect_tx_small", "mbarrier/expect_tx", (), ("MB-06", "MB-07", "MB-08"),
              "exact, oversized and missing expect_tx against real TMA loads, with bounded polls"),
        Probe("mbarrier/expect_tx_small.too_small", "mbarrier/expect_tx", ("too-small",), ("MB-05",),
              "an undersized expect_tx with one load completes the phase part of the way through it in some "
              "runs and not in others (measured; no single output to predict)", predict="none"),
        Probe("mbarrier/expect_tx_small.wait", "mbarrier/expect_tx", ("wait-small",), ("MB-05", "MB-11"),
              "the same with the usual wait loop, which returns in some runs and hangs in others (measured; "
              "no single output to predict)", risk="hang", predict="none", timeout=30),
        Probe("mbarrier/expect_tx_small.pieces", "mbarrier/expect_tx", ("pieces",), ("MB-05",),
              "a load reports its bytes in pieces that vary from run to run: an undersized expect_tx completes "
              "the phase only at a multiple of 32 bytes, and there in some runs and not in others (measured; "
              "no single output to predict)", predict="none"),
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
              "M = 64 with lane offset 8, neither of Layout F's two positions: refused", risk="fault"),
        Probe("tmem/transfer_shapes", "tmem/transfer_shapes", (), ("TM-05",),
              "thread and register to (lane, column) for 32x32b, 16x64b, 16x128b, 16x256b, load and store"),
        Probe("tmem/repeat_counts", "tmem/repeat_counts", (), ("TM-06", "TM-05"),
              "every power-of-two repeat count up to 128 registers per thread exists (the build assembles "
              "them) and loads the cells the mapping names"),
        Probe("tmem/fresh_contents", "tmem/fresh_contents", (), ("TM-13",),
              "tensor memory a launch wrote reads back; what a launch that only reads finds is recorded and "
              "not compared (a B200 kept the last launch's values)"),
        Probe("tmem/cp_record", "tmem/cp", (), (),
              "none: tcgen05.cp is outside v1 and refused. Records which shared bytes the four one-CTA "
              "shapes CUTLASS uses put in which (lane, column)", risk="fault", predict="none"),
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
              "a second dealloc of the same columns is ignored (a B200 ended this kernel, and ran the book's, which "
              "repeat it from other warps: open)", risk="hang", timeout=30),
        Probe("tmem/dealloc_who.two_warps", "tmem/dealloc_who", ("two-warps",), ("TM-11",),
              "two warps release the same allocation once each: the second release is ignored and reported",
              risk="fault", timeout=30),
        Probe("tmem/dealloc_who.two_warps_order", "tmem/dealloc_who", ("two-warps-order",), ("TM-11",),
              "the same with a barrier between the two releases", risk="fault", timeout=30),
        Probe("tmem/dealloc_who.relinquish_two", "tmem/dealloc_who", ("relinquish-two",), ("TM-11",),
              "two warps each relinquish the permit and then release the same allocation, as the book's kernels "
              "do: runs to the end", risk="fault", timeout=30),
        Probe("tmem/dealloc_who.relinquish_once", "tmem/dealloc_who", ("relinquish-once",), ("TM-11",),
              "one warp relinquishes the permit and then releases the same allocation twice", risk="fault", timeout=30),
        Probe("tmem/dealloc_who.once", "tmem/dealloc_who", ("once",), ("TM-11",),
              "one warp of two releases the allocation once", timeout=30),
        Probe("tmem/dealloc_mismatch", "tmem/alloc", ("dealloc-wrong-size",), ("TM-12",),
              "a dealloc with another size than the allocation's is refused", risk="hang", timeout=30),
        Probe("tmem/alloc_leak", "tmem/alloc", ("leak",), (),
              "columns never released fail the launch (cudaErrorTensorMemoryLeak)", risk="hang", timeout=30),
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
        Probe("mma/subnormals", "mma/subnormals", (), ("MMA-04",),
              "half-precision subnormal operands are multiplied like any other number, and lined up and cut in "
              "a sum like any other term", compare="ulp"),
        Probe("mma/negate", "mma/negate", (), ("MMA-07", "MMA-05"),
              "each negate bit flips the product's sign, both leave it; a second MMA accumulates"),
        Probe("mma/tf32_low_bits", "mma/tf32_low_bits", (), ("MMA-08",),
              "kind::tf32 multiplies the operands as stored float32 values; low bits take part"),
        Probe("mma/scale_input_d", "mma/scale_input_d", (), ("MMA-06",),
              "scale-input-d = s multiplies the old accumulator by 2^-s before the addition, for s = 0, 1, 7, 15"),
        Probe("mma/disable_lanes", "mma/disable_lanes", (), ("MMA-09",),
              "bit l of word i of the disable-output-lane operand leaves TMEM lane 32 i + l unwritten"),
        Probe("mma/commit_arrival", "mma/commit_arrival", (), ("MMA-10",),
              "tcgen05.commit makes exactly one arrival; data loaded after waiting on it is the sum of all "
              "eight chained MMAs"),
        Probe("mma/swizzle_mismatch", "mma/swizzle_mismatch", (), ("MMA-11", "TMA-01", "MMA-02"),
              "an operand loaded with a 128B swizzle and described as 64B is read by the descriptor's formula"),
        Probe("mma/base_offset", "mma/base_offset", (), ("MMA-12",),
              "with the base-offset field 0 (all CUTLASS emits) an operand that starts off a swizzle period "
              "is read through the swizzle of its absolute shared address"),
    ] + [
        # One process per case: the program stops at its first failed kernel, and after a fault
        # the context is gone, so `base_offset field` alone would lose every case after it.
        Probe(f"mma/base_offset.field.{index}", "mma/base_offset", ("field", str(index)), ("MMA-12",),
              f"none: a non-zero base-offset field is outside v1 and refused. Records what field {field} "
              f"selects for a {layout} operand starting at byte {start}, against three candidate rules",
              risk="fault", predict="none")
        for index, (layout, start, field) in enumerate((
            ("128B", 128, 1), ("128B", 384, 3), ("128B", 896, 7), ("64B", 128, 1), ("64B", 384, 3),
            ("32B", 128, 1), ("128B", 0, 1), ("128B", 128, 7)))   # WITH_FIELD in mma/base_offset.cu
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
        Probe("cluster/max_size", "cluster/max_size", (), ("CL-02",),
              "without the non-portable attribute a cluster of 8 CTAs launches and those of 16 and 32 are "
              "refused, with 1 KiB and with the most shared memory per CTA"),
        Probe("cluster/max_size.nonportable", "cluster/max_size", ("nonportable",), ("CL-02",),
              "with cudaFuncAttributeNonPortableClusterSizeAllowed clusters of 8 and 16 CTAs launch and one "
              "of 32 is refused"),
        Probe("cluster/address_dump", "cluster/address_dump", (), ("CL-05",),
              "mapa addresses each rank's copy of a shared symbol; in the simulator's encoding clearing "
              "bit 24 names the even member of a pair"),
        Probe("cluster/remote_arrive", "cluster/remote_arrive", (), ("CL-06",),
              "mbarrier.arrive.shared::cluster on a mapa address counts on the target CTA's mbarrier"),
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


def _clc() -> list[Probe]:
    """Cluster launch control. Under `warplens run` every request is granted, so the
    predictions are the extreme where the first cluster takes over the grid; what a GPU's mix
    of started and canceled clusters changes is printed on `info:` lines only."""
    return [
        Probe("cluster/clc_tx_bytes", "cluster/clc_tx_bytes", (), ("CL-20",),
              "a request completes its mbarrier with complete_tx of 16 bytes in each receiving CTA: "
              "expect_tx 16 completes, 8 and 32 do not"),
        Probe("cluster/clc_multicast", "cluster/clc_multicast", (), ("CL-19",),
              "the plain form writes the response and completes the mbarrier in the requesting CTA only; "
              ".multicast::cluster::all does both in every CTA of the cluster"),
        Probe("cluster/clc_failed_response", "cluster/clc_failed_response", (), ("CL-17",),
              "with nothing to cancel the request completes and is_canceled is false; "
              "get_first_ctaid of it (undefined in PTX) is recorded"),
        Probe("cluster/clc_per_thread", "cluster/clc_per_thread", (), ("CL-16",),
              "four lanes executing try_cancel make four requests: 64 bytes complete the phase, "
              "and the successful ones name different clusters that did not start"),
        Probe("cluster/clc_order", "cluster/clc_order", (), ("CL-14", "CL-15"),
              "every cluster either starts or is handed to exactly one running cluster; "
              "the order of the coordinates handed out is recorded"),
        Probe("cluster/clc_first_ctaid", "cluster/clc_first_ctaid", (), ("CL-21",),
              "the coordinate returned is the canceled cluster's first CTA in grid coordinates, for cluster "
              "shapes (2,1,1), (2,2,1), (1,2,2) and for a launch with no cluster shape"),
        Probe("cluster/clc_failures", "cluster/clc_failures", (), ("CL-15",),
              "examples/09_clc is right for any number of requests granted; how many CTAs start at each "
              "grid size is recorded"),
        Probe("cluster/clc_failed_response.again", "cluster/clc_failed_response", ("again",), ("CL-18",),
              "three more requests after an observed failure complete and fail again (undefined in PTX)",
              risk="hang", timeout=30),
        Probe("cluster/clc_failed_response.again_grid", "cluster/clc_failed_response", ("again-grid",), ("CL-18",),
              "on a large grid, one more request after the first failure a worker observes: "
              "the launch's invariants still hold; how many such requests succeed is recorded",
              risk="hang", timeout=30),
    ]


def _formats() -> list[Probe]:
    return [
        Probe("formats/fp8_codes", "formats/small_formats", ("fp8",), ("NUM-01", "NUM-02"),
              "E4M3 and E5M2: every code decodes as modeled; encoding rounds to nearest even and saturates"),
        Probe("formats/fp4_codes", "formats/small_formats", ("fp4",), ("NUM-03",),
              "E2M1: 16 codes; ties to even, saturating at 6"),
        Probe("formats/e8m0_ties", "formats/small_formats", ("e8m0",), ("NUM-04",),
              "UE8M0: code c is 2^(c - 127); a midpoint between two powers of two goes to the larger"),
        Probe("formats/special_inputs", "formats/small_formats", ("special",), ("NUM-06",),
              "NaN, infinities, zeros and out-of-range inputs for the four formats"),
        Probe("formats/tf32_modes", "formats/tf32_modes", (), ("NUM-07", "NUM-08"),
              "cvt.rn/.rz.tf32.f32 round the magnitude bits like .rna with their own tie rule; .rna as on Ampere"),
        Probe("formats/ftz", "formats/ftz", (), ("NUM-09",),
              ".ftz add, sub, mul, div, fma and sqrt around the subnormal range, as recorded on an RTX 3080 Ti "
              "(validation/ampere/ftz_golden.txt; NaN results compared as NaNs)"),
        Probe("formats/fma", "formats/fma", (), ("NUM-10",),
              "fma and mad round the exact product-sum once, f32 and f64: 1000 cases in full and checksums "
              "of 10^4 each"),
        Probe("formats/fma.all", "formats/fma", ("all",), ("NUM-10",),
              "none: all 2 x 10^4 fma cases in full, to locate a checksum of formats/fma that differs",
              predict="none"),
        Probe("formats/scalar_edges", "formats/scalar_edges", (), ("NUM-11", "NUM-12"),
              "min/max with NaNs and signed zeros; float-to-integer conversion at the limits; integer div, "
              "rem, mul.hi and shifts at theirs"),
        Probe("formats/ampere_replay.minmax_cvt", "formats/ampere_replay", ("minmax_cvt",), ("NUM-11",),
              "as on an RTX 3080 Ti (validation/ampere/minmax_cvt_golden.txt): min/max over zeros, infinities "
              "and NaNs, and cvt to s32, u32, s64, u64 in four rounding modes, with the bits of NaN results"),
        Probe("formats/ampere_replay.intdiv", "formats/ampere_replay", ("intdiv",), ("NUM-12",),
              "as on an RTX 3080 Ti (validation/ampere/intdiv_golden.txt): div and rem on 16, 32 and 64 bits; "
              "by zero (unspecified in PTX) both give all ones, the minimum over -1 gives the minimum and 0"),
        Probe("formats/ampere_replay.bitfield", "formats/ampere_replay", ("bitfield",), ("NUM-12",),
              "as on an RTX 3080 Ti (validation/ampere/bitfield_golden.txt): bfi.b32, bfi.b64 and "
              "shf.{l,r}.{wrap,clamp}.b32 over positions and lengths up to and past the register width"),
        Probe("formats/ampere_replay.nan", "formats/ampere_replay", ("nan",), ("NUM-13",),
              "as on an RTX 3080 Ti (validation/ampere/nan_golden.txt): the bits of every NaN result of add, "
              "sub, mul, div, fma, sqrt, neg, abs, mov and cvt (with .ftz too), f32 and f64"),
        Probe("formats/ampere_replay.cvt_ftz", "formats/ampere_replay", ("cvt_ftz",), ("NUM-09",),
              "as on an RTX 3080 Ti (validation/ampere/cvt_ftz_golden.txt): cvt with .ftz between f16, f32 and "
              "f64, to the integer types and to an integer in f32, over the subnormal range of source and result"),
    ]


def _device() -> list[Probe]:
    return [
        Probe("e2e/device_attributes", "e2e/device_attributes", (), ("RT-03",),
              "the device reports compute capability 10.0, 148 SMs, the shared-memory limits the shim gives "
              "and cluster launch support (cudaDevAttrClusterLaunch = 1)"),
        Probe("e2e/smem_limit", "e2e/smem_limit", (), ("RT-04",),
              "a CTA launches with 227 KiB, 227 KiB + 16 and 228 KiB of dynamic shared memory"),
        Probe("e2e/smem_optin", "e2e/smem_optin", (), ("RT-15",),
              "a launch may use 48 KiB less the kernel's static shared memory in dynamic shared memory, or what "
              "cudaFuncSetAttribute set for that kernel"),
        Probe("e2e/smem_static", "e2e/smem_static", (), ("RT-15",),
              "a kernel with more than 48 KiB of static shared memory launches without opt-in and may use the "
              "rest of 227 KiB", risk="fault"),
        Probe("e2e/elect", "e2e/elect", (), ("RT-05",),
              "elect.sync elects the lowest lane of the member mask; every executing lane gets its id"),
        Probe("e2e/elect.all", "e2e/elect", ("all",), ("RT-05",),
              "a lane outside the member mask that executes elect.sync ends the kernel",
              risk="fault"),
        Probe("e2e/elect.partial", "e2e/elect", ("partial",), ("RT-05",),
              "the member mask names lanes that do not execute elect.sync: the lowest executing lane is elected",
              risk="fault"),
        Probe("e2e/elect.other_branch", "e2e/elect", ("other-branch",), ("RT-05",),
              "elect.sync does not wait for lanes of its mask that are on another branch: the flags read 0 "
              "before and after it on a B200. No prediction: which of two branches runs first is not a rule, and "
              "the simulator runs the other one first", risk="hang", predict="none", timeout=30),
        Probe("e2e/elect.other_callers", "e2e/elect", ("other-callers",), ("RT-05",),
              "the control: with a mask that names only its callers elect.sync has nothing to wait for",
              risk="hang", predict="none", timeout=30),
        Probe("e2e/elect.other_sites", "e2e/elect", ("other-sites",), ("RT-05",),
              "two groups at two call sites of elect.sync, each naming the other in its mask: on a B200 one "
              "lane is elected for both, which is the compiler's layout (no prediction)", risk="hang",
              predict="none", timeout=30),
        Probe("e2e/launch_errors", "e2e/launch_errors", (), ("RT-04", "RT-12"),
              "the code of every launch and attribute request the runtime refuses, at this device's own limits"),
    ] + [
        Probe(f"e2e/ampere_replay.{case}", "e2e/ampere_replay", (case,), ids,
              f"as on an RTX 3080 Ti (validation/ampere/{golden}): {what}")
        for case, ids, golden, what in (
            ("shfl", ("RT-01",), "shfl_golden.txt", "shfl.sync in four modes over widths and distances"),
            ("printf", ("RT-01",), "printf_golden.txt, lines in another order",
             "device printf with a by-value struct and a local array"),
            ("ldmatrix", ("RT-01",), "ldmatrix_golden.txt", "ldmatrix in six variants, and bfe"),
            ("prmt", ("RT-01",), "prmt_golden.txt", "prmt.b32"),
            ("cpasync", ("RT-01",), "cpasync_golden.txt", "cp.async with a source size"),
            ("barrier_exit", ("RT-01",), "barrier_exit_hw.cu prints 'output ok'",
             "__syncthreads() completes when some threads have exited"),
            ("addrwrap", ("RT-01",), "addrwrap_golden.txt",
             "[reg+offset] on shared memory with a 32-bit register that has wrapped below zero is summed "
             "in 32 bits, for a reversed index as nvcc 13.5 compiles it and as inline PTX"),
            ("vote", ("RT-16",), "vote_golden.txt",
             "vote.sync all, any, uni and ballot in full and partial warps, with lanes that returned or are "
             "on another branch, and member masks that name lanes which do not vote"),
            ("call", ("RT-02",), "call_golden.txt", "calls to device functions"),
            ("callforms", ("RT-02",), "callforms_golden.txt",
             "structs passed and returned by value, calls through function pointers, recursion"),
        )
    ] + [
        Probe("e2e/ampere_atom", "e2e/ampere_atom", (), ("RT-13",),
              "as on an RTX 3080 Ti (validation/ampere/atom_golden.txt): every atomic function on global and "
              "shared words, by 4 CTAs of 128 threads; only results that do not depend on the threads' order"),
        Probe("e2e/ampere_reconvergence", "e2e/ampere_reconvergence", (), ("RT-14",),
              "as on an RTX 3080 Ti (validation/ampere/reconvergence_golden.txt): lanes that reach a join by "
              "different paths pass it together with lane 0, in 20 PTX kernels (loaded with cuModuleLoadData), "
              "with bar.sync at the join and, as Ampere happens to, without"),
    ]


PROBES: list[Probe] = _device() + _tma() + _mbarrier() + _tmem() + _mma() + _cluster() + _clc() + _formats()
BY_NAME = {probe.name: probe for probe in PROBES}
assert len(BY_NAME) == len(PROBES), "duplicate probe names"

# The end-to-end runs are driven by e2e/run_e2e.py; these are the names it records.
E2E = {
    "e2e/cutlass_tutorials.tutorial_01": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_02": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_03": ("RT-06",),
    "e2e/cutlass_tutorials.tutorial_04": ("RT-06",),
    "e2e/triton": ("RT-07",),
    "e2e/unchanged_scripts.tirx": ("RT-09",),
    "e2e/unchanged_scripts.cute_dsl": ("RT-09",),
    "e2e/book_hgemm": ("RT-08",),
}


def simulator_cell(probe: Probe, predicted) -> str:
    """What the README says about a run in the simulator, from predicted/<name>.sim.txt."""
    if probe.predict == "none":
        return "not run (no prediction)"
    if probe.predict == "numerics":
        return "refused; predicted from `warplens.numerics`"
    note = (predicted / (probe.name + ".sim.txt")).read_text().splitlines()
    if note[0] == "status: ok":
        import re
        findings = re.search(r"(\d+) finding\(s\)", "\n".join(note[1:3]))
        return "runs" + (f" ({findings[1]} finding(s))" if findings and findings[1] != "0" else "")
    reason = next((line.removeprefix("warplens: ") for line in note[1:] if line.startswith("warplens: ")), note[0])
    return "**refuses**: " + reason[:120].replace("|", "/")


def tables() -> str:
    """The per-area tables of README.md ("What each run tests"), in manifest order."""
    from pathlib import Path
    predicted = Path(__file__).resolve().parent / "predicted"
    areas: dict[str, list[Probe]] = {}
    for probe in PROBES:
        areas.setdefault(probe.name.split("/")[0], []).append(probe)
    text = []
    for area, probes in areas.items():
        text += [f"### `{area}/`", "",
                 "| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for probe in probes:
            command = " ".join([probe.program.split("/", 1)[1], *probe.args])
            text.append(f"| `{probe.name.split('/', 1)[1]}` | `{command}` | {', '.join(probe.ids) or '-'} | "
                        f"{probe.claim.replace('|', '/')} | {simulator_cell(probe, predicted)} | {probe.risk or '-'} |")
        text.append("")
    return "\n".join(text)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--tables"]:   # what belongs between the markers in README.md; a test compares
        print(tables(), end="")
        sys.exit(0)
    if sys.argv[1:] == ["--update-readme"]:   # rewrite that part of README.md in place
        from pathlib import Path
        readme = Path(__file__).resolve().parent / "README.md"
        text = readme.read_text()
        start = text.index("\n\n", text.index("<!-- tables:")) + 2
        end = text.index("<!-- end of tables -->")
        readme.write_text(text[:start] + tables() + text[end:])
        sys.exit(0)
    for probe in PROBES:
        flags = ", ".join(x for x in (probe.risk and f"may {probe.risk}",
                                      probe.predict != "sim" and f"prediction: {probe.predict}",
                                      probe.compare != "exact" and f"compare: {probe.compare}") if x)
        print(f"{probe.name:40} {probe.program} {' '.join(probe.args)}".rstrip())
        print(f"{'':40} {', '.join(probe.ids) or '-'}: {probe.claim}" + (f" [{flags}]" if flags else ""))
