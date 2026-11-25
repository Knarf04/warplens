# The first B200 rental (2026-11-25): what was measured

The first run of the probe suites on Blackwell hardware (plan steps E19 to E22,
`PLAN.md` section 11). This file is the record: what ran, what agreed with the
simulator, what did not, and what each difference asks of the simulator. The
fixes are at the end.

What a statement rests on is marked where it matters: **measured** is in the
archives below; **reading** is an interpretation of the output that no further
experiment has checked.

## The machine and the runs

| | |
| --- | --- |
| GPU | one NVIDIA B200, compute capability 10.0, 148 SMs, 183359 MiB |
| Host | a VM (not a container), Ubuntu 24.04.5, kernel 6.11.0-1016-nvidia, 20 CPUs (Xeon Platinum 8580) |
| Driver | 610.57.04, which supports CUDA 13.3: older than the pinned toolkit 13.5 |
| `ncu` | read counters as a normal user (`RmProfilingAdminOnly=0`) |
| Commit | `e488e35cf4be` with five files changed (below) |

R610's `nvidia-smi` prints `KMD Version:` and `CUDA UMD Version:` where earlier
drivers print `Driver Version:` and `CUDA Version:`; `env/remote/lib.sh` did not
read that and was patched on the machine before the first run.

Three runs, each an archive (kept outside the repository; SHA-256 of the
`.tar.gz`):

| Run | Compiler | Archive | SHA-256 |
| --- | --- | --- | --- |
| Main, first attempt | nvcc 13.3.73 (the fallback of `RUNBOOK.md` section 2) | `warplens-b200-20261125T012055Z-e488e35cf4.tar.gz` (64 MB) | `8d142ae2bf5e…` |
| Main, final | the same, after the RT-11 fix and with the self-test | the same name (62 MB) | `25824c8d90e4…` |
| Pinned | nvcc 13.5.37, machine code (`--toolkit pinned`, `--no-e2e`) | `warplens-b200-20261125T022247Z-e488e35cf4.tar.gz` (21 MB) | `40513ebe20ac…` |

The pinned run is "the one deviation worth its time" of the runbook: the
predictions made again with nvcc 13.3 differed from the committed ones for two
probes (`formats/e8m0_ties`, `formats/special_inputs`), so the probes were run
a second time with the predictions' own compiler. Its verdicts are the ones to
hold against `validation/blackwell/predicted/`. The end-to-end programs ran in
the main run only (the driver refuses PTX `.version 9.5`).

In the repository:

| Path | What |
| --- | --- |
| `validation/blackwell/measured/b200/` | the standard output of every run of the pinned run whose verdict was not `identical` (38 runs), `status.tsv`, the kernel's Xid lines, both comparison tables, the rows without a probe, and the cluster-size diagnostic |
| `validation/blackwell_timing/measured/b200/` | the benchmarks' output and the fit of the pinned run; the fit and the `ncu` counter tables of the main run under `fallback-nvcc-13.3.73/` |

Not in the repository: the 17 end-to-end captures (37 MB) and the RT-11 capture
(42 MB). They are in the main run's final archive.

### What went wrong on the day

- **RT-11 failed in the first attempt.** The tutorial's `torch.einsum` reference
  launches a cuBLAS kernel with an unmangled name
  (`cutlass3x_sm100_simt_sgemm_…`) and a 384-byte argument; the capture check
  took it for a DSL kernel and asked for its PTX. Fixed in
  `validation/blackwell/e2e/cute_dsl_tma.py` (its memory is skipped like that of
  PyTorch's own kernels); the rerun ended `RT-11 capture: COMPLETE`.
- **`run-blackwell` is `failed` in the main run's manifest** for a second
  reason, which stands: with nvcc 13.3 `formats/e8m0_ties` and
  `formats/special_inputs` stop at a device-side assert in that toolkit's
  `cuda_fp8.hpp` (`__nv_cvt_float_to_e8m0` accepts only round-toward-zero and
  round-up there). With nvcc 13.5 both run and are identical to the prediction.
- **The self-test** (`selftest`, fallback toolkit): 20 failed, 4 errors, 3083
  passed. Those read are the fallback compiler (the two E8M0 predictions, PTX
  `.version 9.5` fixtures refused by ptxas 13.3, timing instruction counts) or
  tests that assume no real `ncu` is installed. Not all 24 were read.
- **The pinned run's `ncu`** profiled 10 of 11 benchmarks; `transpose` failed
  (`ncu` exit 1). The main run has all 11.
- **The VM was stopped from outside at least three times** (a power-button
  event from the hypervisor in the two cases whose log was read), once in the
  middle of a self-test and once in the pinned run's timing stage. `run.sh`
  resumed each time; no measurement was lost. The rental took about the two
  hours budgeted for it.

## Probes: 126 runs against the committed predictions

The pinned run (`measured/b200/compare-pinned-nvcc-13.5.37.txt`):

| Verdict | Runs |
| --- | --- |
| identical | 88 |
| informational (no prediction, or the GPU ran what the simulator refuses) | 18 |
| **DIFFERS** | **8** |
| kernel failed by design (the GPU faults, the simulator's kernel returns) | 6 |
| GPU faulted, not modelled | 3 |
| timed out, as predicted | 2 |
| unverified | 1 |

The fallback run gave the same verdict for every probe but the two E8M0 runs
above.

### Confirmed

Identical to the prediction, on hardware, for the first time:

- **mbarrier**: phases and parity, arrival counts, `expect_tx` with bytes,
  arrive tokens, `try_wait` on completed phases, completion before
  `expect_tx`, the refused `init` counts.
- **TMA**: the swizzle modes on aligned and misaligned buffers, ranks 1 to 5,
  element strides, zero fill out of bounds, stores and `wait_group`, raw bulk
  copies, hints, the missing fence.
- **TMEM**: allocation addresses and order, the lane window, the layout of
  `tcgen05.mma` results, every transfer shape and repeat count.
- **MMA**: operand layouts for f16 and i8, negation, `scale-input-d`, disabled
  lanes, `commit` arrivals, the swizzle mismatch, the base offset with the
  field 0 (by address: MMA-12).
- **Clusters**: the cluster barrier, multicast, remote arrive, the two-SM MMA
  in all four forms, the address dump, and every cluster-launch-control run
  (CL-11 to CL-21).
- **Numerics**: FP8 and FP4 codes, `.ftz`, `fma`, the scalar edges; E8M0 ties
  and the special inputs (pinned run).
- **Every Ampere golden file replayed on the B200** is reproduced line for
  line (RT-01, RT-02, RT-13, RT-14, RT-16, MB-03, NUM-09, NUM-11 to NUM-13).

### Differences

Each row is a claim of `docs/validation-matrix.md` that hardware contradicts.

| Row | Criterion | Probe | Predicted | B200 (measured) |
| --- | --- | --- | --- | --- |
| TMA-06 | bit-exact | `tma/oob_complete_tx` | `expect_tx` of only the in-bounds bytes of a box that is half outside the tensor does not complete the phase | it completes; and in that case the out-of-bounds rows of the destination keep their old bytes |
| MB-05, MB-06 | bit-exact | `mbarrier/expect_tx_small` | `expect_tx 128` then a 256-byte load: the phase does not complete | the phase completes. `mbarrier/expect_tx_small.wait` agrees: the unbounded wait returns |
| TMA-05 | bit-exact | `tma/oob_fill` | NaN fill is NumPy's quiet NaN (`7fc00000` for f32) | `7ff77ff7` for f32; 112 lines differ, f16 and f64 among them |
| NUM-07 | bit-exact | `formats/tf32_modes` | `cvt.rn/.rz.tf32.f32` of a NaN clears the low 13 bits and keeps the rest | `7fffe000` for every NaN input, whatever its sign and payload (7 lines) |
| CL-02 | bit-exact | `cluster/max_size` | a cluster of 16 CTAs launches | refused: `cudaErrorInvalidClusterSize`. 8 launch. With `cudaFuncAttributeNonPortableClusterSizeAllowed` set, 16 launch and 32 are refused (`measured/b200/cluster-optin/`) |
| RT-15 | bit-exact | `e2e/smem_static` | a kernel with more than 48 KiB of static shared memory reports `227 KiB - static` as its maximum of dynamic shared memory | it reports 0. The launches themselves are as predicted: up to `227 KiB - static` dynamic bytes launch without opt-in, one 16-byte step more is refused |
| MMA-08 | informational | `mma/tf32_low_bits` | `kind::tf32` multiplies the stored float32 values, low bits included | the low 13 bits of one operand do not take part (223 of 258 lines) |
| TM-13 | informational | `tmem/fresh_contents` | tensor memory never written reads as zero | zero in a first launch; after a launch that wrote it, the next launch reads that launch's values, each warp those another warp's lanes had stored (warp 0 read lane 64's, warp 2 lane 0's) |

One more is outside its tolerance, and one could not be judged:

| Row | Criterion | Run | B200 (measured) |
| --- | --- | --- | --- |
| RT-08 | tolerance: 1 unit in the last place of the simulator's f16 output | `e2e/book_hgemm` | `hgemm_v1` to `v5` (K = 64) within 1 ulp. `v6`, `v7` (K = 256): 2 of 16384 elements, up to 5 ulp. `v8`: 10 of 65536, up to 30 ulp. `v9`: 18 of 131072, up to 27 ulp. All nine still match NumPy within the book's own tolerance |
| MMA-04, MMA-05 | tolerance | `mma/accumulation_order` | 646 of 3016 float32 words differ. Where the simulator loses a 2^-10 term against 2^14 and gets 0, hardware keeps it: the sum is not a chain of float32 additions |

TMA-06 and MB-05/06 are one rule, and RT-08 and MMA-04/05 one cause; both are
worked out under "What this changed" below. The first is not "a phase
completes when the bytes delivered reach the bytes expected": the instruction
`mbarrier.complete_tx 64` against `expect_tx 32` completes nothing on the same
GPU (`mbarrier/complete_before_expect`, identical to its prediction). A TMA
copy reports its bytes in pieces, and the count passes through zero on the
way.

### Where the simulator is more permissive than hardware

Six runs marked as allowed to fault did fault, while the simulator's kernel
returns:

| Run | B200 |
| --- | --- |
| `e2e/elect.all` (member mask `0xffff0000` executed by all lanes) | `cudaErrorIllegalInstruction`; Xid 13 "Illegal Instruction Parameter" |
| `tma/swizzle_misaligned.sixteen` | `cudaErrorMisalignedAddress` |
| `tma/store_wait_group.edge` | `cudaErrorIllegalInstruction` |
| `mbarrier/over_arrival` | `cudaErrorLaunchFailure` |
| `tmem/dealloc_repeated` | `cudaErrorLaunchFailure` |
| `tmem/alloc_leak` | `cudaErrorTensorMemoryLeak` |

And three fault where the simulator has no model: `tmem/lane_window.16`, `.1`
and `.48` (`cudaErrorMisalignedAddress`; the simulator stops at an unsupported
`tcgen05.ld` form).

### Where the simulator is stricter than hardware

| Run | The simulator refuses | B200 |
| --- | --- | --- |
| `tmem/alloc_addresses.size_48` | an allocation of 48 columns (not a power of two) | allocated at column 0, released |
| `tmem/alloc_increasing` | a larger allocation after a smaller one | 64 columns after 32: at column 64 |
| `tmem/alloc_after_relinquish` | `alloc` after `relinquish_alloc_permit` | allocated at column 0 |
| `mma/operand_layouts.tf32_mn` | `kind::tf32` with an MN-major operand | runs; 179 of 8192 elements equal `A B^T`, so the layout it reads is not the one the probe assumed |
| `mbarrier/expect_tx_small.wait` | reports a deadlock | the wait returns (MB-05 above) |

`mbarrier/try_wait.incomplete` (MB-11): `try_wait` on a phase that never
completes returns false, each of 17 times. The simulator blocks, by the
decision recorded in `PLAN.md` section 3.5.

### Rows without a probe

| Row | Result |
| --- | --- |
| RT-11 | captured, `COMPLETE`: 34 files, 43.9 MB. The tensor maps of a CuTe DSL kernel, its PTX and the memory before and after |
| RT-10 | both traces ran; the call lists are in `measured/b200/rows/` |
| NUM-05 | no UE4M3 conversion in ptxas 13.3 or 13.5: the row stays open |
| TMA-12 | the `cuTensorMapEncodeTiled` sweep: 1382 lines recorded |

## End to end (main run, nvcc 13.3; simulator replay of the GPU's own PTX)

| Program | Row | Result |
| --- | --- | --- |
| CUTLASS tutorials 1 to 4 | RT-06 | bit-identical |
| Triton vector add | RT-07 | bit-identical |
| Triton matmul | RT-07 | within tolerance (3.8e-6 against a limit of 1.35e-4) |
| TIRx saxpy, CuTe DSL saxpy | RT-09 | bit-identical |
| The book's `hgemm_v1` to `v5` | RT-08 | within 1 ulp |
| The book's `hgemm_v6` to `v9` | RT-08 | **outside**: up to 30 ulp (above) |

All 17 captures pass `validation.blackwell.capture verify --require-hardware`.

## Timing

Fourteen benchmarks ran in both runs. The fit of the pinned run is in
`validation/blackwell_timing/measured/b200/report.md`; the numbers below are
from it (the main run's are within a few percent).

| Parameter | Fitted | Config before | Ratio |
| --- | --- | --- | --- |
| `clock_ghz` (effective issue clock of one warp) | 0.513 | 1.5 | 0.34 |
| `mma_latency_clk` | about 96 | 11.3 | 8.5 |
| `global_access_latency_clk` | 91 | 30 | 3.0 |
| `tma_bytes_per_clk` | 439 | 128 | 3.4 |
| `tma_latency_ns` | 147 | 100 | 1.5 |
| `clc_latency_ns` | 236 | 100 | 2.4 |
| `smem_wavefront_clk` | 0.52 | 1 | 0.52 |
| `tmem_access_clk` | about 3.6 | 4 | 0.9 |
| `hbm_bandwidth_tb_s` | 6.16 | 4.14 | 1.5 |
| `tensor_core_tflops_f16` | about 2240 | 1930 | 1.16 |

- The cluster-launch-control latency is not one number: 236 ns for a failed
  request in one cluster, 423 to 493 ns for the other forms. The model has one.
- An mbarrier arrive and wait costs 33 ns; the model charges about 3.
- **The bank-conflict counters agree exactly.** `ncu`'s
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` equals the model's count in
  all cases, and the wavefront counter is above the model's by the same 4160
  the RTX 3080 Ti showed (`validation/README.md`, "The wavefront offset"): the
  rule found on Ampere holds on Blackwell.

## What this changed

Every difference above was taken to the simulator on the branch
`work/b200-rental-1`. "Reproduces" means the simulator now prints, for the
probe as it ran on the B200, the recorded output line for line
(`validation/blackwell/measured/b200/`).

| # | Row | Change | Check |
| --- | --- | --- | --- |
| 1 | NUM-07 | `cvt.rn/.rz.tf32.f32` of any NaN gives `7fffe000` | reproduces `formats/tf32_modes` |
| 2 | TMA-05 | NaN fill is `0x7ff7` in every 16 bits of the element | reproduces `tma/oob_fill` |
| 3 | CL-02 | under the shim a cluster of more than 8 CTAs needs `cudaFuncAttributeNonPortableClusterSizeAllowed`; a new run `cluster/max_size.nonportable` sets it | reproduces `cluster/max_size` and the diagnostic with the attribute |
| 4 | RT-15 | `maxDynamicSharedSizeBytes` reads 0 for a kernel with more than 48 KiB static until the program sets it; what a launch may use is unchanged | reproduces `e2e/smem_static` |
| 5 | TMA-06, MB-05, MB-06 | a TMA copy reports its bytes in pieces, so a count smaller than the copy passes through zero and completes the phase early; the instruction `mbarrier.complete_tx` still subtracts at once | reproduces `mbarrier/expect_tx_small` and `.wait`; `tma/oob_complete_tx` but for the one row the GPU had not yet written, for which the probe now settles before it reads |
| 6 | MMA-08 | `kind::tf32` clears the low 13 bits of both operands | reproduces `mma/tf32_low_bits` (256 products) |
| 7 | MMA-04, MMA-05, RT-08 | one MMA is one sum: terms lined up under the largest (by the operands' exponents), cut toward zero 25 bits down, added exactly, cut toward zero to float32; the old accumulator is one of the terms | reproduces `mma/accumulation_order` (3016 words); the book's nine kernels replay **bit for bit** against the day's captures (311,296 elements; before: up to 30 ulp off) |
| 8 | TM-13 | no model change: the two read-only launches of the probe are no longer compared | - |
| 9 | RT-05, TMA-10, TMA-04, MB-04, TM-11 | four of the six faults are errors now: `elect.sync` by a lane outside its mask, a tensor copy at an address that is not a multiple of 32, a store at a negative coordinate, more arrivals than the phase needs. A leak of tensor memory fails the launch under the shim (`cudaErrorTensorMemoryLeak`). The repeated dealloc stays a finding: the book's kernels do it and ran | reproduces `e2e/elect.all`, `tma/swizzle_misaligned.sixteen`, `tma/store_wait_group.edge`, `mbarrier/over_arrival`, `tmem/alloc_leak` |
| 10 | TM-08 | a larger allocation after a smaller one is made and reported, not refused. 48 columns and an allocation after the relinquish stay refused: the PTX contract rules them out | reproduces `tmem/alloc_increasing` |
| 11 | timing | **not applied.** The fit is recorded (`validation/blackwell_timing/measured/b200/calibrated.yaml`, usable today with `--timing-config`); `warplens/configs/b200.yaml` still has the assumed values. See below | `validation/blackwell_timing/measured/b200/report.md` |

**The timing config is a decision, not a fix.** Tried on this branch and taken
back: with `calibrated.yaml` in place of `b200.yaml` the predictions move
toward the measurements (45 of 152 benchmark configurations within 10 %, from
5; 104 within a factor of two, from 53), and three things break that are not
mechanical to repair:

- one property the model is meant to keep fails by a hair:
  `tests/test_timing.py::test_more_stages_never_make_it_slower` (TIM-18's
  ordering, 10888 ns against 10905 ns);
- the worked examples of `docs/guide.md` print other numbers and another
  verdict ("TMA engine-bound, 91 %" was written for the assumed config), so
  the text around them has to be read again, not regenerated;
- about twenty tests state the assumed values (`clc_latency_ns == 100`,
  `clock_ghz == 1.5`) and several check `fit.py` against them.

And the fit says of itself where one parameter per mechanism is not enough:
`clc_latency_ns` is 236 ns for one form of the request and 423 to 493 ns for
the others, `clock_ghz` is an effective issue rate of one warp (0.51) and not
a clock, an mbarrier arrive and wait costs 33 ns against 3. Replacing the
numbers without deciding those would trade assumed values for fitted ones
that are known to be wrong elsewhere.

Three numbers in these fixes are **assumptions**, each the weakest that the
measurements allow, and each has a new probe run that asks a GPU:

| Assumption | Where | Asked by |
| --- | --- | --- |
| A TMA copy reports its bytes in pieces of 64 (measured: the size divides 64) | `TX_PIECE_BYTES` | `mbarrier/expect_tx_small.pieces` |
| A tensor copy needs a shared address aligned to 32 bytes (measured: 16 faults, 128 runs) | `TENSOR_COPY_ALIGNMENT` | `tma/swizzle_misaligned.thirtytwo`, `.sixtyfour` |
| In an MMA a negative term is cut toward zero, a product is lined up by its operands' exponents, and the old accumulator is cut like a product (from the book's kernels, not from a probe) | `mma_sum` | `mma/accumulation_order`, parts 4 to 6 |

## What the second rental has to confirm

Fixes 1 to 7 and 9 changed predictions of bit-exact and tolerance rows, which
is trigger 1 of `env/remote/RUNBOOK.md`, section 5. The run is the same
command; what to look for afterwards:

- every run of the table above is `identical`, and `e2e/book_hgemm` is
  bit-identical or within 1 ulp;
- the three assumptions: `mbarrier/expect_tx_small.pieces`,
  `tma/swizzle_misaligned.thirtytwo` and `.sixtyfour`, and parts 4 to 6 of
  `mma/accumulation_order`. A difference there is an answer, and changes one
  constant or one rule;
- `tma/oob_complete_tx` with the settle: the whole box written;
- the timing suite once more, for a second set of measurements to hold the
  first fit against (the config is unchanged, so the predictions are too).

Open, and not for a GPU to settle: what separates the repeated dealloc that
faulted from the ones in the book's kernels that did not (TM-11); `kind::i8`
arithmetic; half-precision subnormals in an MMA.

## The tensor-map decoder (done after the rental, without a GPU)

The layout of the driver's 128-byte tensor map is read from the TMA-12 sweep
(`warplens/host/tensor_map.py`, "the driver's own layout"): every one of the
1087 maps the B200's driver encoded reads back as the parameters that made it,
and the four maps the CuTe DSL wrote for `fp16_gemm_0.py` read back as the
tutorial's tensors (`tests/test_tensor_map_driver_layout.py`). The DSL fills
in the first 64 bytes only and sets one flag bit the driver does not.

With that, the kernel captured for RT-11 gets past its TMA loads in the
simulator and stops at two instruction forms **outside the v1 scope**:
`tcgen05.mma` with `.collector::a::discard` (12 uses) and `cvt.rn.f16x2.f32`
(32). As an experiment, with both rewritten in the PTX text
(`validation/blackwell/e2e/replay_launch.py --rewrite-out-of-scope`: the
modifier dropped, the packed conversion written as two scalar ones), the
launch replays in 3 seconds and **no byte of the 2 MiB allocation differs
from the B200's after the kernel**: 255,768 bytes the GPU changed, the 131,072
half-precision elements of C among them. That is the first kernel compared
with hardware whose TMA, MMA and epilogue all ran from a frontend's own
descriptors, and it holds the MMA sum above on a tenth kernel.

Two things it showed besides:

- The kernel's tiles sit 128 bytes off the swizzle period (`tma-swizzle-misaligned`,
  64 times) and the result is right: TMA and the MMA both swizzle by the
  absolute address (TMA-02, MMA-12), now on a real kernel. The finding's text
  warns about code that indexes by offset, which this kernel does not; for
  such a kernel it is noise.
- The driver's map carries the bytes of one box as `box // element stride`,
  rounded down, dimension 0 included; `TensorMapInfo.counts` rounds up and
  takes dimension 0's stride as 1. They agree when the strides divide the box,
  which is all that has run on a GPU. Not changed; not measured.

**A decision for the owner:** whether those two forms join the v1 surface.
`PLAN.md` freezes it ("nothing is added to it before v1.0"), so they were not
added. Both are small (`discard` is what an MMA without the modifier does; the
conversion is two existing ones packed), and with them `fp16_gemm_0.py` would
run under `warplens run` and its capture could be a committed fixture.
