# The first B200 rental (2026-11-25): what was measured

The first run of the probe suites on Blackwell hardware (plan steps E19 to E22,
`PLAN.md` section 11). This file is the record: what ran, what agreed with the
simulator, what did not, and what each difference asks of the simulator. The
fixes are tracked at the end.

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
| TM-13 | informational | `tmem/fresh_contents` | tensor memory never written reads as zero | zero in a first launch; after a launch that wrote it, the next launch reads that launch's values, with the warps' quarters exchanged |

One more is outside its tolerance, and one could not be judged:

| Row | Criterion | Run | B200 (measured) |
| --- | --- | --- | --- |
| RT-08 | tolerance: 1 unit in the last place of the simulator's f16 output | `e2e/book_hgemm` | `hgemm_v1` to `v5` (K = 64) within 1 ulp. `v6`, `v7` (K = 256): 2 of 16384 elements, up to 5 ulp. `v8`: 10 of 65536, up to 30 ulp. `v9`: 18 of 131072, up to 27 ulp. All nine still match NumPy within the book's own tolerance |
| MMA-04, MMA-05 | tolerance | `mma/accumulation_order` | 646 of 3016 float32 words differ. Where the simulator loses a 2^-10 term against 2^14 and gets 0, hardware keeps it: the sum is not a chain of float32 additions |

**Reading**: TMA-06 and MB-05/06 look like one rule. The simulator completes a
phase when the pending byte count is exactly zero; both results fit a phase
that completes when the bytes delivered reach or pass the bytes expected.
RT-08 and MMA-04/05 are probably one cause too (how the K products are
summed), and it grows with K as the book's kernels show.

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

## What this changes, and the state of each fix

| # | Change | State |
| --- | --- | --- |
| 1 | NUM-07: `cvt.rn/.rz.tf32.f32` of a NaN gives `7fffe000` | open |
| 2 | TMA-05: the NaN fill constants | open |
| 3 | CL-02: a cluster above 8 CTAs needs the non-portable attribute | open |
| 4 | RT-15: the reported maximum of dynamic shared memory above 48 KiB static | open |
| 5 | TMA-06, MB-05, MB-06: when a phase with a byte count completes | open |
| 6 | MMA-08: `kind::tf32` converts an operand | open |
| 7 | MMA-04, MMA-05, RT-08: how the K products are summed | open |
| 8 | TM-13: tensor memory across launches | open (the finding stays; the predicted value changes) |
| 9 | The six faults the simulator does not report | open |
| 10 | The three TMEM allocations the simulator refuses | open |
| 11 | The timing config from the fit | open |

Fixes 1 to 7 change a prediction that a bit-exact or tolerance row needs
hardware to confirm: they are trigger 1 for the second rental
(`env/remote/RUNBOOK.md`, section 5).

Still to do without a GPU: the decoder of driver-format tensor maps, from the
RT-11 capture and the TMA-12 sweep (`docs/hardware-capture.md`).
