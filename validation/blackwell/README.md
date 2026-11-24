# Blackwell probe suite

Small kernels that test, on a real B200, what the simulator claims about TMA, mbarrier,
tensor memory, `tcgen05.mma`, clusters and the small number formats; and the harness that
runs them unattended and compares the results with predictions made beforehand.

**Nothing in this directory has run on Blackwell hardware.** The development machine has an
RTX 3080 Ti (Ampere). Every prediction is the simulator's output, committed so that a
mismatch on the rented machine is unambiguous. What was run where is under
[Dry run](#dry-run).

## How it works

Every probe is one `.cu` file: a kernel and a host program that prints deterministic text
(bit patterns in hex, inputs next to outputs, no times, no addresses). It is built once, as
a program that carries PTX and uses the shared CUDA runtime:

```
nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared probe.cu -o probe
```

That one file runs on a B200, where the driver compiles the PTX, and under `warplens run`,
where the simulator executes it (`docs/m6.md`). The prediction of a probe is what it printed
under `warplens run`.

| File | What it is |
| --- | --- |
| `manifest.py` | every run: name, program, arguments, matrix ids, claim, risk, timeout. `python manifest.py` prints it |
| `build.sh` | builds every probe; needs only nvcc. Also assembles each one to `sm_100a` machine code, which is what proves ptxas accepts it |
| `run_all.py` | runs everything, one process per run, each under a timeout; records the environment; packs an archive |
| `predict.sh` | rewrites `predicted/` from the simulator |
| `../regenerate_predictions.sh` | everything that is predicted, here and in `validation/blackwell_timing`, in one command: `predict.sh`, the tables of this file, the repeat-count table, the timing predictions |
| `predicted/<name>.txt` | the predicted standard output of a run |
| `predicted/<name>.sim.txt` | how the simulator's run ended and what the simulator said: findings, or why it refused |
| `predicted/provenance.json` | the nvcc whose PTX the simulator ran to make the predictions (written by `predict.sh` from `bin/toolkit.txt`, which `build.sh` writes) |
| `compare.py` | compares an archive with `predicted/` and prints one verdict per run |
| `common.h`, `tcgen05_common.h`, `tma/tma_dump.h` | the PTX wrappers and the shared kernels |
| `e2e/run_e2e.py`, `e2e/triton_gpu.py` | the end-to-end programs |
| `run_rows.py` | the three matrix rows that are not probes, run at the end of a GPU run into `rows/`: `formats/ue4m3_ptxas.py` (NUM-05), `e2e/driver_call_trace.py` with `e2e/gdb_driver_trace.py` (RT-10), `e2e/cute_dsl_tma.py` (RT-11). See [Rows without a probe](#rows-without-a-probe) |
| `tests/test_blackwell_probes.py` (in the repository's `tests/`) | fails when a committed prediction is no longer what the simulator prints |
| `tests/test_predict_path.py` | runs `run_all.py --mode predict` whole (probes, small formats, end to end) into a temporary directory and holds it against `predicted/` |
| `tests/test_rental_harness.py` | a probe that does not launch is never a result; a run built by another toolkit is flagged |
| `tests/test_rows_without_probes.py` | the rows without a probe say `skipped` with a reason where they cannot run, and never fail a run |

Conventions the probes share:

- A line that starts with `info:` carries something machine-dependent (a device name, an
  address, an error name) and is never compared.
- A wait whose outcome is the thing under test is a bounded poll (`POLL_BUDGET` = 20000
  `test_wait`s, on the order of a millisecond on a GPU), so the kernel returns either way and
  prints `phase completed 0` or `1`. The assumption that any copy a probe issues lands within
  that budget is the suite's; a `0` that should be a `1` would point at it.
- A kernel that is refused or faults: the program prints `KERNEL FAILED: ...` and exits with
  status 3. On a GPU nothing after a fault can be trusted, so anything risky is its own run.
- A program whose kernels the driver could not load at all (PTX newer than the driver, no
  code for this GPU, no device) prints `LAUNCH FAILED: ...` and exits with status 4. That is
  not an outcome of the test and is never read as one; see
  [When probes do not launch](#when-probes-do-not-launch). A failed host-side CUDA call
  prints `HOST ERROR: ...`, status 2. `probe_sync()` in `common.h` checks the error of the
  launch itself, of the synchronize, and the last error after it.
- Names follow `docs/validation-matrix.md` (branch `work/validation-matrix`) where a row of
  the matrix names a probe; `.case` suffixes are further runs of the same program.

## On the rented machine

Requirements: a B200 (`sm_100a`; the first run, `e2e/device_attributes`, prints the compute
capability, which must be 10.0), a CUDA toolkit with `nvcc` that knows `sm_100a` (12.8 or
newer; the probes were written against 13.5), Python 3.9 or newer for the runner (standard
library only). For the end-to-end part also this repository installed (`pip install -e .`),
network access to fetch CUTLASS, PyTorch and Triton, `apache-tvm-tirx` and `cuda-bindings`,
and a clone of the book.

```bash
git clone <this repository> warplens && cd warplens
validation/blackwell/run_all.py --book /path/to/modern-gpu-programming-for-mlsys
# ... prints one line per run, then the name of the archive
validation/blackwell/compare.py validation/blackwell/results/gpu-<host>-<time>.tar.gz
```

One command, no interaction. `run_all.py` builds (`build.sh --native`), records
`nvidia-smi`, driver, toolkit and device, runs the 117 probe runs in manifest order, then the
end-to-end programs, and writes `results/gpu-<host>-<time>.tar.gz` with every run's standard
output, standard error and status (`status.tsv`), the build log and the environment. Bring
the archive back; `compare.py` needs nothing but Python and this directory.

Useful options: `--no-e2e` (probes only), `--only 'tmem/*'` (repeatable), `--no-build`,
`--timeout-scale 2`, `--binaries native`.

**PTX or machine code.** `bin/` holds the PTX-carrying programs, the very files the
simulator ran. The driver can compile them only if it is at least as new as the toolkit that
wrote the PTX. `run_all.py` tries one harmless run from `bin/` first and falls back to
`bin-native/` (the same sources as `sm_100a` machine code) if that fails; which one was used
is in `env/binaries.txt` in the archive.

### When probes do not launch

A driver older than the toolkit refuses the toolkit's PTX. Measured on the development
machine (toolkit 13.5, driver 580.173.02 which supports CUDA 13.0): the first launch returns
`cudaErrorUnsupportedPtxVersion`, and the same source built to machine code by the same nvcc
runs. Nothing about this is silent:

| Where | What you see |
| --- | --- |
| the probe | `LAUNCH FAILED: <what>`, the error name and text on an `info:` line, exit status 4 |
| `run_all.py` | status `launch-failed` in `status.tsv`; a block in capitals at the end counting the probes that failed to launch, had a host error or were not built; exit status 1, so the stage fails in `env/remote/run.sh` |
| `env/binaries.txt` | the canary's output from `bin/` and from `bin-native/`, and which was used; `NEITHER BUILD RUNS` if both failed |
| `compare.py` | verdict `FAILED TO LAUNCH` (or `HOST ERROR`), counted as a failure whatever the manifest allows the probe to do: a probe that may fault "by design", and whose prediction is a refused kernel, does not pass by failing to load |

What to do: `run_all.py --binaries native` (after `build.sh --native`) runs the machine-code
programs. They come from the same nvcc, so the predictions still apply.

### Which toolkit built the programs

The predictions are the simulator's run of PTX from one nvcc (`predicted/provenance.json`:
13.5.37, the version `env/remote/pins.env` installs). If the rented machine's driver is
older, `env/remote/setup.sh` by default installs the toolkit that matches the driver, and
the probes are then compiled by another compiler than the one behind the predictions. Most
probes are inline PTX and should come out the same, but that is an expectation, not a fact,
so the run is flagged rather than trusted:

- `build.sh` writes `bin/toolkit.txt` (and `bin-native/`, `bin-ampere/`): the nvcc version.
- `run_all.py` writes `env/toolkit.txt` (driver, the CUDA release it supports, the nvcc that
  built the programs, the nvcc of the predictions, `predictions_comparable=yes|no|unknown`),
  prints `BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE` between rows of `!` and
  writes `TOOLKIT-MISMATCH.txt` at the top of the results.
- `compare.py` prints the same line before and after its table. `--require-same-toolkit`
  refuses to compare at all (exit status 3).
- `env/remote/setup.sh`, `preflight.sh` (check `predictions`) and `run.sh` say it too, and
  `run.sh` puts `PREDICTIONS-NOT-COMPARABLE.txt` at the top of its archive
  (`env/remote/RUNBOOK.md`, "Driver older than the toolkit").

To keep the predictions' compiler on such a machine: `env/remote/run.sh --toolkit pinned`.
The driver then refuses the PTX and the machine-code programs run (`bin-native/`).

**Expected run time** (estimated; nothing here has been timed on a B200): the build takes a
few minutes (60 programs, three compilations each with `--native`). Most runs take well
under a second of GPU time plus process start-up. Runs that hang are killed at their
timeout, 30 s each; 12 runs are allowed to hang, and two of them (`mbarrier/expect_tx_large`,
`tmem/alloc_blocks`) are expected to. So the probes take between 3 and 10 minutes. The
end-to-end part is dominated by fetching and compiling CUTLASS and by Triton's, PyTorch's
and TVM's start-up and compilers: allow 15 minutes. In the simulator the whole suite takes a
few minutes on the development machine (the book's nine kernels are about one of them).

**After a hang or fault.** Each run is a separate process, killed with its process group on
timeout, so the next run starts with a fresh CUDA context. If a fault leaves the GPU itself
unusable (an Xid that needs a reset), every later run fails at its first CUDA call and says
`HOST ERROR`; `status.tsv` shows where that began. The runs most likely to do that are at the
end of each area in the manifest, and can be left out with `--only`.

## What each run tests

`python manifest.py` prints this from the source of truth. "Refuses" means the simulator
stops the kernel with the message shown; the committed prediction for such a run is the
program's `KERNEL FAILED` output, and `compare.py` reports what the GPU did as
*informational* (there is nothing to hold it against). "May hang" and "may fault" are what
the manifest allows on hardware **by design**: the behaviour is unknown or undefined, and the
run is isolated for that reason.

<!-- tables: written by `python manifest.py --tables`; tests/test_blackwell_probes.py compares -->

### `e2e/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `device_attributes` | `device_attributes` | RT-03 | the device reports compute capability 10.0, 148 SMs, the shared-memory limits the shim gives and cluster launch support (cudaDevAttrClusterLaunch = 1) | runs | - |
| `smem_limit` | `smem_limit` | RT-04 | a CTA launches with 227 KiB, 227 KiB + 16 and 228 KiB of dynamic shared memory | runs | - |
| `smem_optin` | `smem_optin` | RT-15 | a launch may use 48 KiB less the kernel's static shared memory in dynamic shared memory, or what cudaFuncSetAttribute set for that kernel | runs | - |
| `smem_static` | `smem_static` | RT-15 | a kernel with more than 48 KiB of static shared memory launches without opt-in and may use the rest of 227 KiB | runs | fault |
| `elect` | `elect` | RT-05 | elect.sync elects the lowest lane of the member mask; every executing lane gets its id | runs | - |
| `elect.all` | `elect all` | RT-05 | lanes outside the member mask execute elect.sync too: the lowest lane of the mask is elected | runs | fault |
| `elect.partial` | `elect partial` | RT-05 | the member mask names lanes that do not execute elect.sync: the lowest executing lane is elected | runs | fault |
| `ampere_replay.shfl` | `ampere_replay shfl` | RT-01 | as on an RTX 3080 Ti (validation/ampere/shfl_golden.txt): shfl.sync in four modes over widths and distances | runs | - |
| `ampere_replay.printf` | `ampere_replay printf` | RT-01 | as on an RTX 3080 Ti (validation/ampere/printf_golden.txt, lines in another order): device printf with a by-value struct and a local array | runs | - |
| `ampere_replay.ldmatrix` | `ampere_replay ldmatrix` | RT-01 | as on an RTX 3080 Ti (validation/ampere/ldmatrix_golden.txt): ldmatrix in six variants, and bfe | runs | - |
| `ampere_replay.prmt` | `ampere_replay prmt` | RT-01 | as on an RTX 3080 Ti (validation/ampere/prmt_golden.txt): prmt.b32 | runs | - |
| `ampere_replay.cpasync` | `ampere_replay cpasync` | RT-01 | as on an RTX 3080 Ti (validation/ampere/cpasync_golden.txt): cp.async with a source size | runs | - |
| `ampere_replay.barrier_exit` | `ampere_replay barrier_exit` | RT-01 | as on an RTX 3080 Ti (validation/ampere/barrier_exit_hw.cu prints 'output ok'): __syncthreads() completes when some threads have exited | runs (1 finding(s)) | - |
| `ampere_replay.vote` | `ampere_replay vote` | RT-16 | as on an RTX 3080 Ti (validation/ampere/vote_golden.txt): vote.sync all, any, uni and ballot in full and partial warps, with lanes that returned or are on another branch, and member masks that name lanes which do not vote | runs | - |
| `ampere_replay.call` | `ampere_replay call` | RT-02 | as on an RTX 3080 Ti (validation/ampere/call_golden.txt): calls to device functions | runs | - |
| `ampere_replay.callforms` | `ampere_replay callforms` | RT-02 | as on an RTX 3080 Ti (validation/ampere/callforms_golden.txt): structs passed and returned by value, calls through function pointers, recursion | runs | - |
| `ampere_atom` | `ampere_atom` | RT-13 | as on an RTX 3080 Ti (validation/ampere/atom_golden.txt): every atomic function on global and shared words, by 4 CTAs of 128 threads; only results that do not depend on the threads' order | runs | - |
| `ampere_reconvergence` | `ampere_reconvergence` | RT-14 | as on an RTX 3080 Ti (validation/ampere/reconvergence_golden.txt): lanes that reach a join by different paths pass it together with lane 0, in 20 PTX kernels (loaded with cuModuleLoadData), with bar.sync at the join and, as Ampere happens to, without | runs | - |

### `tma/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `swizzle_aligned` | `swizzle aligned` | TMA-01 | a swizzled load XORs the 16-byte chunk index with the low 1/2/3 bits of the 128-byte row index | runs | - |
| `swizzle_misaligned` | `swizzle misaligned` | TMA-02 | the swizzle is a function of the shared ADDRESS, not of the offset into the buffer | runs (6 finding(s)) | - |
| `swizzle_misaligned.sixteen` | `swizzle sixteen` | TMA-02 | the same with a destination 16 bytes into a 128-byte row | runs (3 finding(s)) | fault |
| `ranks` | `ranks` | TMA-03, TMA-15 | rank 1-5 boxes land densely, dimension 0 fastest; complete_tx reports box bytes; the tensor map is a by-value kernel parameter | runs | - |
| `misaligned_requests` | `ranks unaligned` | TMA-10 | an innermost coordinate that is not a multiple of 16 bytes is refused | **refuses**: TMA innermost coordinate must give a 16-byte aligned bounding box (coordinate 1, element size 4) | fault |
| `oob_fill` | `oob_fill` | TMA-04, TMA-05, TMA-06 | out-of-bounds elements are zero or the default quiet NaN; complete_tx counts the whole box | runs | - |
| `oob_complete_tx` | `oob_fill inbounds` | TMA-06 | expect_tx of only the in-bounds bytes does NOT complete the phase | runs (6 finding(s)) | - |
| `element_strides` | `element_strides` | TMA-07 | element stride s moves ceil(box / s) elements, packed densely, and reports those bytes | runs | - |
| `element_strides.inner` | `element_strides inner` | TMA-07 | an element stride in dimension 0 is ignored | runs | fault |
| `store_wait_group` | `store` | TMA-09, TMA-01, TMA-03 | a store is in global memory after commit_group + wait_group 0; it reads through the load's swizzle | runs | - |
| `store_wait_group.edge` | `store edge` | TMA-04 | a store writes only the elements of the box that are inside the tensor | runs | fault |
| `store_wait_group.outside` | `store outside` | TMA-04 | a store wholly outside the tensor writes nothing | runs | fault |
| `tensormap_encode_sweep` | `tensormap_encode_sweep` | TMA-11, TMA-12 | none: records the driver's 128-byte tensor maps over a parameter sweep | not run (no prediction) | - |
| `raw_bulk` | `raw_bulk` | TMA-08 | cp.async.bulk without a tensor map copies 16, 64 and 4096 bytes verbatim in both directions and reports the size to the mbarrier | runs | - |
| `hints` | `hints` | TMA-14 | a cache-hint operand, prefetch.tensormap and cp.async.bulk.prefetch do not change what a load loads | runs | - |
| `missing_fence` | `missing_fence` | TMA-13 | a TMA store right after ordinary stores carries the current bytes with and without fence.proxy.async (2000 stores per variant) | runs (4 finding(s)) | - |
| `missing_fence.long` | `missing_fence 100000` | TMA-13 | none: the same with 10^5 stores per variant, too slow to simulate; the prediction of tma/missing_fence applies line for line | not run (no prediction) | - |

### `mbarrier/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `phase_parity` | `script basic` | MB-01 | a new mbarrier is in phase 0; test_wait.parity p is true when the phase's parity is not p | runs | - |
| `phase_parity.counts` | `script count` | MB-01, MB-02 | the arrival count is reloaded at each completion; arrive with a count; init restarts | runs | - |
| `arrivals_and_bytes` | `script tx` | MB-02 | a phase completes when pending arrivals and the byte count are both zero, in either order | runs | - |
| `lane_arrivals` | `lane_arrivals` | MB-09 | every lane that executes an arrive arrives once, also when only part of a warp does | runs | - |
| `arrive_token` | `arrive_token` | MB-03 | test_wait with an arrive token is false while the token's phase is open and true once it completed, also during the next phase | runs | - |
| `arrive_token.stale` | `arrive_token stale` | MB-03 | a token two phases old is invalid PTX use and is diagnosed; its predicate is informational only | runs (2 finding(s)) | - |
| `ampere_token` | `ampere_token` | MB-03 | as on an RTX 3080 Ti (validation/ampere/mbarrier_token_golden.txt): test_wait answers from bit 63 of the token, the parity of the arrival's phase. For tokens two or more phases old and made-up tokens (invalid PTX use) a difference is Blackwell's, not a simulator error | runs (333 finding(s)) | - |
| `try_wait` | `try_wait` | MB-01, MB-03 | one try_wait, in the parity and the token form, is true once the phase it names has completed | runs | - |
| `try_wait.incomplete` | `try_wait incomplete` | MB-11 | none: the simulator's try_wait blocks (a deadlock report here). Records whether one try_wait on a phase that never completes returns false, and after how many clock ticks | not run (no prediction) | hang |
| `init_count` | `init_count` | MB-10 | mbarrier.init accepts 2^20 - 1 arrivals, and an arrive with that count completes the phase | runs | - |
| `init_count.over` | `init_count over` | MB-10 | an arrival count of 2^20 is refused | **refuses**: mbarrier.init with an arrival count of 1048576; it must be 1 to 2^20-1 | fault |
| `init_count.zero` | `init_count zero` | MB-10 | an arrival count of 0 is refused | **refuses**: mbarrier.init with an arrival count of 0; it must be 1 to 2^20-1 | fault |
| `complete_before_expect` | `script negative` | MB-07 | the byte count is signed: complete_tx before expect_tx completes once balanced | runs | fault |
| `over_arrival` | `script over` | MB-04 | arrivals beyond the pending count complete the phase and carry into the next | runs (2 finding(s)) | fault |
| `expect_tx_small` | `expect_tx` | MB-05, MB-06, MB-07, MB-08 | undersized, oversized and missing expect_tx against real TMA loads, with bounded polls | runs | - |
| `expect_tx_small.wait` | `expect_tx wait-small` | MB-05, MB-11 | an undersized expect_tx with one load never completes: the usual wait loop deadlocks | **refuses**: deadlock in CTA (0, 0, 0): every unfinished warp is blocked and nothing can release them | hang |
| `expect_tx_large` | `expect_tx wait-large` | MB-08, MB-11 | an oversized expect_tx never completes: the usual wait loop deadlocks | **refuses**: deadlock in CTA (0, 0, 0): every unfinished warp is blocked and nothing can release them | hang |

### `tmem/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `lane_window` | `lane_window` | TM-01, TM-02 | a warp reaches only lanes 32 (w % 4) .. + 31; a lane field naming another window reads its own | runs | - |
| `lane_window.16` | `lane_window 16` | TM-03 | lane field 16: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-memory wind | fault |
| `lane_window.1` | `lane_window 1` | TM-03 | lane field 1: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-memory wind | fault |
| `lane_window.48` | `lane_window 48` | TM-03 | lane field 48: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-memory wind | fault |
| `layout_f` | `layout_f` | TM-04, TM-01 | M = 128: row m on lane m. M = 64: row m on lane 32 (m / 16) + m % 16 + a, a = 0 or 16 | runs | - |
| `layout_f.64_8` | `layout_f 64 8` | TM-04 | M = 64 with lane offset 8, neither of Layout F's two positions: refused | **refuses**: tcgen05.mma result with M=64 at lane 8; the lane of the destination must be 0 or 16 | fault |
| `transfer_shapes` | `transfer_shapes` | TM-05 | thread and register to (lane, column) for 32x32b, 16x64b, 16x128b, 16x256b, load and store | runs | - |
| `repeat_counts` | `repeat_counts` | TM-06, TM-05 | every power-of-two repeat count up to 128 registers per thread exists (the build assembles them) and loads the cells the mapping names | runs | - |
| `fresh_contents` | `fresh_contents` | TM-13 | tensor memory that was never written reads as zero, in a first launch and after a launch that wrote it | runs (64 finding(s)) | - |
| `cp_record` | `cp` | - | none: tcgen05.cp is outside v1 and refused. Records which shared bytes the four one-CTA shapes CUTLASS uses put in which (lane, column) | not run (no prediction) | fault |
| `alloc_addresses` | `alloc` | TM-07 | alloc returns the lowest free block aligned to its size, as (lane 0, column) | runs | - |
| `alloc_increasing` | `alloc increasing` | TM-08 | a later allocation may not be larger than an earlier one | **refuses**: tcgen05.alloc of 64 columns increases the previous allocation size of 32 columns | fault |
| `alloc_addresses.size_48` | `alloc size-48` | TM-07 | an allocation of 48 columns is refused | **refuses**: tcgen05.alloc of 48 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_16` | `alloc size-16` | TM-07 | an allocation of 16 columns is refused | **refuses**: tcgen05.alloc of 16 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_0` | `alloc size-0` | TM-07 | an allocation of 0 columns is refused | **refuses**: tcgen05.alloc of 0 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_1024` | `alloc size-1024` | TM-07 | an allocation of 1024 columns is refused | **refuses**: tcgen05.alloc of 1024 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_after_relinquish` | `alloc after-relinquish` | TM-09 | no allocation after relinquish_alloc_permit | **refuses**: tcgen05.alloc after this CTA relinquished its allocation permit | fault |
| `alloc_blocks` | `alloc exhaust` | TM-10 | an allocation that does not fit waits (here: forever) | **refuses**: deadlock in CTA (0, 0, 0): every unfinished warp is blocked and nothing can release them | hang |
| `dealloc_repeated` | `alloc dealloc-twice` | TM-11 | a second dealloc of the same columns is ignored | runs (1 finding(s)) | hang |
| `dealloc_mismatch` | `alloc dealloc-wrong-size` | TM-12 | a dealloc with another size than the allocation's is refused | **refuses**: tcgen05.dealloc of 32 columns at column 0 does not match an allocation | hang |
| `alloc_leak` | `alloc leak` | - | columns never released are reported and do not affect the next launch | runs (1 finding(s)) | hang |

### `mma/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `operand_layouts` | `operand_layouts` | MMA-01, MMA-02, MMA-03 | K-major (f16, tf32) and MN-major (f16) operand addressing in all four swizzle modes; exact results | runs | - |
| `operand_layouts.tf32_mn` | `operand_layouts tf32-mn` | MMA-03 | MN-major tf32 is refused (needs a 128B swizzle with 32-byte atomicity) | **refuses**: unsupported instruction `tcgen05.mma.cta_group::1.kind::tf32` (MN-major TF32 requires 128B swizzling with 32-byte atomic | fault |
| `accumulation_order` | `accumulation_order` | MMA-04, MMA-05 | none exactly: the simulator sums in float32 in NumPy's order; hardware may differ in the last bits | runs | - |
| `negate` | `negate` | MMA-07, MMA-05 | each negate bit flips the product's sign, both leave it; a second MMA accumulates | runs | - |
| `tf32_low_bits` | `tf32_low_bits` | MMA-08 | kind::tf32 multiplies the operands as stored float32 values; low bits take part | runs (1 finding(s)) | - |
| `scale_input_d` | `scale_input_d` | MMA-06 | scale-input-d = s multiplies the old accumulator by 2^-s before the addition, for s = 0, 1, 7, 15 | runs | - |
| `disable_lanes` | `disable_lanes` | MMA-09 | bit l of word i of the disable-output-lane operand leaves TMEM lane 32 i + l unwritten | runs | - |
| `commit_arrival` | `commit_arrival` | MMA-10 | tcgen05.commit makes exactly one arrival; data loaded after waiting on it is the sum of all eight chained MMAs | runs | - |
| `swizzle_mismatch` | `swizzle_mismatch` | MMA-11, TMA-01, MMA-02 | an operand loaded with a 128B swizzle and described as 64B is read by the descriptor's formula | runs (2 finding(s)) | - |
| `base_offset` | `base_offset` | MMA-12 | with the base-offset field 0 (all CUTLASS emits) an operand that starts off a swizzle period is read through the swizzle of its absolute shared address | runs | - |
| `base_offset.field` | `base_offset field` | MMA-12 | none: a non-zero base-offset field is outside v1 and refused. Records what it selects, against three candidate rules | not run (no prediction) | fault |

### `cluster/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `barrier` | `barrier` | CL-01, CL-03, CL-04 | the cluster barrier holds every thread until all have arrived; mapa + ld.shared::cluster | runs | - |
| `multicast` | `multicast` | CL-07 | a multicast load writes the box in every CTA of the mask and completes each one's mbarrier | runs | - |
| `multicast.outside` | `multicast outside` | CL-07 | a mask naming a rank outside the cluster is refused | **refuses**: CTA rank 4 does not exist in a cluster of 4 CTA(s) | fault |
| `pair_accounting` | `pair_accounting` | CL-08 | with .cta_group::2 a pair's completions go to the member with the mbarrier address's parity | runs (3 finding(s)) | - |
| `max_size` | `max_size` | CL-02 | clusters of 8 and 16 CTAs launch and one of 32 is refused, with 1 KiB and with the most shared memory per CTA | runs | - |
| `address_dump` | `address_dump` | CL-05 | mapa addresses each rank's copy of a shared symbol; in the simulator's encoding clearing bit 24 names the even member of a pair | runs | - |
| `remote_arrive` | `remote_arrive` | CL-06 | mbarrier.arrive.shared::cluster on a mapa address counts on the target CTA's mbarrier | runs | - |
| `pair_mma.m256_issuer0` | `pair_mma 256 0` | CL-10, CL-13, CL-09 | two-CTA MMA, M = 256, issued by rank 0: each CTA holds 128 rows on 128 lanes | runs | hang |
| `pair_mma.m128_issuer0` | `pair_mma 128 0` | CL-11, CL-09 | two-CTA MMA, M = 128, issued by rank 0: Layout B: 64 rows per CTA, columns folded onto lanes 64-127 | runs | hang |
| `pair_mma.m256_issuer1` | `pair_mma 256 1` | CL-12 | two-CTA MMA, M = 256, issued by rank 1: each CTA holds 128 rows on 128 lanes | runs | hang |
| `pair_mma.m128_issuer1` | `pair_mma 128 1` | CL-12 | two-CTA MMA, M = 128, issued by rank 1: Layout B: 64 rows per CTA, columns folded onto lanes 64-127 | runs | hang |
| `clc_tx_bytes` | `clc_tx_bytes` | CL-20 | a request completes its mbarrier with complete_tx of 16 bytes in each receiving CTA: expect_tx 16 completes, 8 and 32 do not | runs (9 finding(s)) | - |
| `clc_multicast` | `clc_multicast` | CL-19 | the plain form writes the response and completes the mbarrier in the requesting CTA only; .multicast::cluster::all does both in every CTA of the cluster | runs | - |
| `clc_failed_response` | `clc_failed_response` | CL-17 | with nothing to cancel the request completes and is_canceled is false; get_first_ctaid of it (undefined in PTX) is recorded | runs (4 finding(s)) | - |
| `clc_per_thread` | `clc_per_thread` | CL-16 | four lanes executing try_cancel make four requests: 64 bytes complete the phase, and the successful ones name different clusters that did not start | runs | - |
| `clc_order` | `clc_order` | CL-14, CL-15 | every cluster either starts or is handed to exactly one running cluster; the order of the coordinates handed out is recorded | runs | - |
| `clc_first_ctaid` | `clc_first_ctaid` | CL-21 | the coordinate returned is the canceled cluster's first CTA in grid coordinates, for cluster shapes (2,1,1), (2,2,1), (1,2,2) and for a launch with no cluster shape | runs | - |
| `clc_failures` | `clc_failures` | CL-15 | examples/09_clc is right for any number of requests granted; how many CTAs start at each grid size is recorded | runs | - |
| `clc_failed_response.again` | `clc_failed_response again` | CL-18 | three more requests after an observed failure complete and fail again (undefined in PTX) | runs (5 finding(s)) | hang |
| `clc_failed_response.again_grid` | `clc_failed_response again-grid` | CL-18 | on a large grid, one more request after the first failure a worker observes: the launch's invariants still hold; how many such requests succeed is recorded | runs (1 finding(s)) | hang |

### `formats/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `fp8_codes` | `small_formats fp8` | NUM-01, NUM-02 | E4M3 and E5M2: every code decodes as modeled; encoding rounds to nearest even and saturates | runs | - |
| `fp4_codes` | `small_formats fp4` | NUM-03 | E2M1: 16 codes; ties to even, saturating at 6 | runs | - |
| `e8m0_ties` | `small_formats e8m0` | NUM-04 | UE8M0: code c is 2^(c - 127); a midpoint between two powers of two goes to the larger | runs | - |
| `special_inputs` | `small_formats special` | NUM-06 | NaN, infinities, zeros and out-of-range inputs for the four formats | runs | - |
| `tf32_modes` | `tf32_modes` | NUM-07, NUM-08 | cvt.rn/.rz.tf32.f32 round the magnitude bits like .rna with their own tie rule; .rna as on Ampere | runs | - |
| `ftz` | `ftz` | NUM-09 | .ftz add, sub, mul, div, fma and sqrt around the subnormal range, as recorded on an RTX 3080 Ti (validation/ampere/ftz_golden.txt; NaN results compared as NaNs) | runs | - |
| `fma` | `fma` | NUM-10 | fma and mad round the exact product-sum once, f32 and f64: 1000 cases in full and checksums of 10^4 each | runs | - |
| `fma.all` | `fma all` | NUM-10 | none: all 2 x 10^4 fma cases in full, to locate a checksum of formats/fma that differs | not run (no prediction) | - |
| `scalar_edges` | `scalar_edges` | NUM-11, NUM-12 | min/max with NaNs and signed zeros; float-to-integer conversion at the limits; integer div, rem, mul.hi and shifts at theirs | runs (8 finding(s)) | - |
| `ampere_replay.minmax_cvt` | `ampere_replay minmax_cvt` | NUM-11 | as on an RTX 3080 Ti (validation/ampere/minmax_cvt_golden.txt): min/max over zeros, infinities and NaNs, and cvt to s32, u32, s64, u64 in four rounding modes, with the bits of NaN results | runs | - |
| `ampere_replay.intdiv` | `ampere_replay intdiv` | NUM-12 | as on an RTX 3080 Ti (validation/ampere/intdiv_golden.txt): div and rem on 16, 32 and 64 bits; by zero (unspecified in PTX) both give all ones, the minimum over -1 gives the minimum and 0 | runs (12 finding(s)) | - |
| `ampere_replay.bitfield` | `ampere_replay bitfield` | NUM-12 | as on an RTX 3080 Ti (validation/ampere/bitfield_golden.txt): bfi.b32, bfi.b64 and shf.{l,r}.{wrap,clamp}.b32 over positions and lengths up to and past the register width | runs | - |
| `ampere_replay.nan` | `ampere_replay nan` | NUM-13 | as on an RTX 3080 Ti (validation/ampere/nan_golden.txt): the bits of every NaN result of add, sub, mul, div, fma, sqrt, neg, abs, mov and cvt (with .ftz too), f32 and f64 | runs | - |
| `ampere_replay.cvt_ftz` | `ampere_replay cvt_ftz` | NUM-09 | as on an RTX 3080 Ti (validation/ampere/cvt_ftz_golden.txt): cvt with .ftz between f16, f32 and f64, to the integer types and to an integer in f32, over the subnormal range of source and result | runs | - |
<!-- end of tables -->

### End to end (`e2e/run_e2e.py`)

| Run | What | Matrix | Prediction |
| --- | --- | --- | --- |
| `cutlass_tutorials.tutorial_01` .. `_04` | the four CUTLASS Blackwell tutorials, unmodified, built by `examples/03_cutlass_mma/build.sh --programs`; the same binaries on the GPU and in the simulator | RT-06 | committed (`predicted/e2e/`), from the simulator |
| `triton` | the vector add and the matmul of `examples/05_triton`. On the GPU through `e2e/triton_gpu.py`, which imports the kernels from the example and gives them PyTorch tensors, because the example itself always installs the simulator's driver. Run by the interpreter `WARPLENS_TORCH_PYTHON` names: PyTorch lives in an env of its own (`env/remote/pins.env` says why), pinned to torch 2.15.0 with Triton 3.9.0. Without PyTorch the step is recorded as `skipped (...)`, not as a crash | RT-07 | committed, from `warplens run examples/05_triton/run.py` |
| `book_hgemm` | `hgemm_v1` to `v9` of *Modern GPU Programming for MLSys* through `examples/08_book/run.py <clone>`. The book has no license; its code is never copied here, and the clone is an argument (`--book`, or `WARPLENS_BOOK`). Skipped without it | RT-08 | committed (`predicted/e2e/book_hgemm.txt`: nine "matches NumPy" lines), from the simulator against the book's commit `3a24144d`, the one `tests/test_book.py` names. Regenerate with `predict.sh --book <clone>` |

These print error norms and "matches NumPy" lines, not bit patterns, and their outputs
contain times. `compare.py` treats them as informational and shows the result lines from the
GPU next to the simulator's. The `max error` of a "matches NumPy" line is measured against
NumPy on the machine that ran the script, whose BLAS decides the last float16 digit of the
reference (0.0078 on one CPU, 0.0156 on another, for the same kernel output): the claim is
the name, the size and "matches NumPy", and `compare.py` says so when only that number differs.

## Reading the comparison

```
probe                              verdict                     detail
tma/swizzle_misaligned             identical
mma/accumulation_order             unverified                    812 of 2968 float32 words differ, by at most 2 ulp
tmem/lane_window.16                informational               the GPU ran what the simulator refuses (...)
mbarrier/expect_tx_large           timed out (by design)
tmem/alloc_increasing              kernel failed (by design)   ...
tma/oob_fill                       DIFFERS
```

(The lines above illustrate the format; they are not results.) `FAILED TO LAUNCH` and
`HOST ERROR` are not in the illustration: they mean the probe tested nothing (above).
`identical`: the claim held.
`DIFFERS`: it did not; the first differing lines follow the table, with the claim. `unverified`:
finite values differ in `mma/accumulation_order`, which the simulator does not claim to match
bit for bit (`docs/numerics.md`). The reported ULP distance is an observation, not an accepted
tolerance or proof that rounding explains the difference. Unequal infinities or NaNs are
`DIFFERS`. The comparison exits 1 for failures, 2 for unverified numeric differences without
other failures, and 0 otherwise. `informational`: no prediction; read the output. A run that is refused
in the simulator and *also* fails on the GPU counts as `identical`: both print the same
`KERNEL FAILED` line. The error names differ (`info:` lines) and are in the archive.

`tma/tensormap_encode_sweep` has no prediction and never will: it prints, for 1380
parameter sets, the 128 bytes the driver's `cuTensorMapEncodeTiled` wrote. That is the data
a decoder of real tensor maps needs (`docs/m6.md`, "Not working: kernels that use TMA").
Under `warplens run` it is skipped; the program itself was run once against the shim only
to see that it runs to the end.

## Matrix coverage

Every row of `docs/validation-matrix.md` outside the timing section has at least one run
here, except four: TMA-01 to -15; MB-01 to -11; TM-01 to -13; MMA-01 to -12; CL-01 to -21;
NUM-01 to -04 and -06 to -13; RT-01 to -09, -13 and -14.

Partly covered, and how: MMA-01 (only the M, N, format and major-ness fields are varied);
MMA-05 (repeated accumulation of the same product; no case that isolates "one float32
addition"); MB-11 (`mbarrier/try_wait.incomplete` prints what a single `try_wait` on an open phase
returns, without a prediction: the simulator's `try_wait` blocks by design); MMA-12 (the field 0
is predicted; `mma/base_offset.field` records a non-zero field without a prediction); CL-01 (ranks along x only); CL-09 and CL-13 (exercised
inside `pair_mma`, not isolated); TMA-10 (the unaligned coordinate; an unaligned shared
address is `tma/swizzle_misaligned.sixteen`, which is 16-byte aligned, so a shared address
that is not a multiple of 16 is not tried); TMA-13 (2000 stores per variant in the predicted
run; `tma/missing_fence.long` does the row's 10^5 on hardware only); CL-14 to CL-21 (the
prediction is one extreme, every request granted, because `warplens run` cannot limit
cancellations; on hardware how many requests succeed depends on how many clusters start at
once, so the compared lines carry only what the rows' criteria name, and counts and order
are on `info:` lines; CL-19 and CL-20 are tested on failed requests, in launches of one
cluster).

Where a probe differs from what its matrix row planned: TM-06 is `tmem/repeat_counts` (every
claimed shape and count in one program, so the `sm_100a` build is the assembler's answer)
plus `tmem/repeat_counts_ptxas.py`, which also tries the counts that must be rejected and
writes `tmem/repeat_counts_ptxas.txt`; it is not a run of the manifest because it needs
ptxas, not a GPU. NUM-09 compares with `validation/ampere/ftz_golden.txt` except that NaN
results are printed as `nan` (the payload differs between the simulator and the 3080 Ti).
RT-01 and RT-02 are `e2e/ampere_replay`, one run per Ampere program.

### Reruns of the Ampere programs

Where the simulator does what an RTX 3080 Ti did because PTX gives no rule, or gives one
that was only ever checked on Ampere, the program that recorded the golden file in
`validation/ampere/` is rerun on the B200. Each run includes that program's source (the
kernels and the host code that prints; nothing is copied) and prints the lines of the golden
file. The simulator reproduces the golden files, so the committed prediction of such a run
*is* the golden file; `tests/test_ampere_replay.py` checks that, and that no golden file is
without a run. A line that differs on the B200 is then a difference between Blackwell and
Ampere, on a compared line and not an `info:` one.

| Golden file (`validation/ampere/`) | Run | Matrix | Lines compared |
| --- | --- | --- | --- |
| `shfl_golden.txt`, `ldmatrix_golden.txt`, `prmt_golden.txt`, `cpasync_golden.txt` | `e2e/ampere_replay.shfl`, `.ldmatrix`, `.prmt`, `.cpasync` | RT-01 | all |
| `printf_golden.txt` | `e2e/ampere_replay.printf` | RT-01 | all, as a set (device and host lines reach the output in an undefined order) |
| `call_golden.txt`, `callforms_golden.txt` | `e2e/ampere_replay.call`, `.callforms` | RT-02 | all |
| `vote_golden.txt` | `e2e/ampere_replay.vote` | RT-16 | all 2220 launches. `vote_legacy_golden.txt` (the forms without `.sync`) has no run: ptxas refuses them for `sm_100a` |
| `atom_golden.txt` | `e2e/ampere_atom` | RT-13 | all |
| `reconvergence_golden.txt` | `e2e/ampere_reconvergence` | RT-14 | all; each kernel launched 100 times, as for the recording |
| `mbarrier_token_golden.txt` | `mbarrier/ampere_token` | MB-03 | all, the answers for stale and made-up tokens included |
| `minmax_cvt_golden.txt` | `formats/ampere_replay.minmax_cvt` | NUM-11 | all, NaN results as bits |
| `intdiv_golden.txt` | `formats/ampere_replay.intdiv` | NUM-12 | all, division by zero included |
| `nan_golden.txt` | `formats/ampere_replay.nan` | NUM-13 | all, the 44 lines of `cvt.ftz.f64.f32` and `cvt.rn.ftz.f32.f64` included (finding 17) |
| `cvt_ftz_golden.txt` | `formats/ampere_replay.cvt_ftz` | NUM-09 | all, NaN results as bits |
| `ftz_golden.txt` | `formats/ftz` | NUM-09 | all, but a NaN result as the word `nan` (its bits are the f32 pattern `formats/ampere_replay.nan` compares) |
| `tf32_golden.txt` | `formats/tf32_modes` | NUM-08 | all, as the `.rna` column beside `.rn` and `.rz` (the inputs are written out again in that probe; `tests/test_blackwell_probes.py` holds the column to the golden file) |

The whole of each file is printed, also the three large ones (`nan`, `minmax_cvt` and
`intdiv`, 36 000 lines and 2.5 MB of predictions together): what would differ is unknown,
a NaN policy shows only in particular operand positions, and each run is one launch that
takes well under a second on a GPU and in the simulator.

Why these are four programs beside `e2e/ampere_replay` and not more cases of it: one
program is one PTX module, and a module the driver or the simulator will not take stops
every case in it. The mbarrier token kernel uses the `sm_80` spellings (`.shared` without
`::cta`), the atomics are spelled by each toolkit in its own way, the reconvergence kernels
are PTX text loaded at run time, and the NaN program has two instructions the simulator
refuses; each is kept from the cases that have nothing unusual in them. `callforms` is
ordinary CUDA C++ like `call` and is a case of `e2e/ampere_replay`, whose other cases print
what they printed before.

`e2e/ampere_reconvergence` is the one probe that does not carry its kernels as the
compiler's PTX: the layouts of the branches are the point, so the kernels are the PTX text
of `validation/ampere/reconvergence_golden.py` (`.target sm_86`), in a header that script
writes (`reconvergence_kernels.h`; a test keeps it current). The program finds
`cuModuleLoadData`, `cuModuleGetFunction` and `cuLaunchKernel` through
`cudaGetDriverEntryPointByVersion`, as `common.h` finds the tensor-map encoder, and uses
them in the runtime's context, so it links nothing more than the other probes. `build.sh`
cannot show that ptxas for `sm_100a` accepts these kernels (they are not compiled until
the driver loads them); they use nothing but `bra`, `bar.sync`, `ld` and `st`. A module the
driver will not load is reported as `LAUNCH FAILED`.

Not rerun: the launch-error recordings (`launch_errors_3080ti.txt`,
`driver_errors_3080ti.txt`) are taken at the 3080 Ti's own limits, so their lines cannot be
a B200's; RT-04 and RT-12 test the same refusals at Blackwell's limits
(`e2e/smem_limit`, `cluster/max_size`). The bank-conflict counters (`transpose_ncu_*.csv`,
`wavefront_probe_*.csv`) are Nsight Compute measurements and belong to
`validation/blackwell_timing/`.

**No probe, but a scripted run:** NUM-05, RT-10 and RT-11; see
[Rows without a probe](#rows-without-a-probe). TIM-* belong to `validation/blackwell_timing/`.

### Rows without a probe

Three rows are not a kernel with a predicted output, so they are not in the manifest and
`compare.py` does not read them. `run_all.py` ends a GPU run (not `--only`) with
`run_rows.py`, which runs each as a step under a timeout and writes `rows/status.tsv`
(`ok`, `skipped (why)`, `news (...)`, or what failed) and the outputs under `rows/`. Nothing
there fails the run.

| Row | Step | Script | What it does | Without a B200 |
| --- | --- | --- | --- | --- |
| NUM-05 | `formats/ue4m3_ptxas` | `formats/ue4m3_ptxas.py` | CUDA has no conversion to or from UE4M3, so the row cannot be tested bit for bit. Rather than take that from a document, the script asks the installed ptxas: one `cvt` kernel per format and direction, with the four formats that do convert as controls. With ptxas 13.5 (`formats/ue4m3_ptxas.txt`): all controls accepted, `.ue4m3x2` and `.ue4m3x4` are unknown modifiers, `.ue4m3` is a known type that `cvt` does not take. Status `news` if a toolkit ever accepts one | runs (needs ptxas only) |
| RT-10 | `e2e/driver_call_trace.tirx`, `.cute_dsl` | `e2e/driver_call_trace.py` | Runs `examples/07_tirx/run.py` and `examples/06_cute_dsl/saxpy.py` under gdb with `e2e/gdb_driver_trace.py`: a breakpoint on the address of every export of `libcuda.so` (674 on driver 580.173.02) and of the shared `libcudart.so`, each call logged in order (`calls.txt`, `summary.txt`). No symbols, no root | runs on any GPU the scripts run on. On the 3080 Ti: the CuTe DSL trace is complete (26 distinct calls, one `cuLaunchKernelEx`); TIRx stops at `cuModuleLoadData`, because the 13.0 driver refuses NVRTC 13.5's PTX |
| RT-11 | `e2e/cute_dsl_tma` | `e2e/cute_dsl_tma.py` | Runs CUTLASS's `fp16_gemm_0.py` (from the clone `examples/03_cutlass_mma/build.sh` makes; PyTorch from `WARPLENS_TORCH_PYTHON`) plain, then under the same tracer. At `cuLaunchKernel` / `cuLaunchKernelEx` the tracer asks the driver for every parameter's offset and size (`cuFuncGetParamInfo`) and saves the bytes: `trace/launches.txt`, `launches.json`, and `launchNN.paramMM.bin` for each parameter of 64 bytes or more. The 128-byte tensor maps the DSL wrote are among them | `skipped (needs sm_100 ...)`. The capture itself was checked on the 3080 Ti with a CUDA C++ kernel that takes a 128-byte struct by value (bytes read back identical) and with `--script examples/06_cute_dsl/saxpy.py` |

Needs gdb with Python support (`WARPLENS_GDB`, else `gdb` on `PATH`; `env/remote/setup.sh`
provides one). What to do on the day when a step is skipped, and the one part of RT-11
that stays manual (`create_tensor_map_tiled`): `env/remote/RUNBOOK.md`, section 3.

## Dry run

What was done on the development machine (RTX 3080 Ti Laptop GPU, `sm_86`, driver
580.173.02; nvcc 13.5.37 from the `warplens` conda env, compile-only), on 2026-11-23:

| Step | Result |
| --- | --- |
| `build.sh`: all 51 programs built as PTX-carrying programs **and assembled by ptxas to `sm_100a` machine code** | all 51 pass |
| `build.sh --native`: the same as `sm_100a` machine-code programs | all 51 link (13 s on 20 cores); none can run here |
| `run_all.py --mode predict`, the whole path into a temporary directory (`tests/test_predict_path.py`), then `compare.py` on its results | all 101 runs recorded, 98 executed (three have no prediction and are skipped); the 94 that the simulator predicts are identical to `predicted/`, the 4 small-format runs fail as predicted (the simulator has no such instructions); the four CUTLASS tutorials, Triton, the TIRx and CuTe DSL scripts and the book's nine kernels (from a clone at commit `3a24144d`) all pass their own checks |
| the timeout path: `--mode sim --timeout-scale 0.004` | the run is killed and recorded as `timeout`; the next run proceeds; `compare.py` reports `timed out` |
| **on the GPU**: `build.sh --ampere`, `run_all.py --binaries ampere --no-build --only ...` | `mbarrier/phase_parity`, `mbarrier/phase_parity.counts` and `mbarrier/lane_arrivals` ran on the 3080 Ti and are **identical** to the predictions. `e2e/device_attributes` ran and differs, as it should (8.6, 58 SMs): that is the check that catches a wrong machine. The environment capture (`nvidia-smi`, `compute_cap`, driver, ...) was exercised by the same run |
| **the driver refusing PTX, on the GPU**: `mbarrier/script` built by nvcc 13.5 for `compute_86` as a PTX-carrying program and as machine code; the driver supports CUDA 13.0 | the PTX program prints `LAUNCH FAILED` (`cudaErrorUnsupportedPtxVersion`, exit 4); `run_all.py` records `launch-failed`, exits 1, and `compare.py` says `FAILED TO LAUNCH`. With `--binaries auto` the canary fails from `bin/`, passes from `bin-native/`, and the three mbarrier runs are identical to the predictions. With `toolkit.txt` naming another nvcc the banner appears in `run_all.py` and `compare.py` |
| **on the GPU**, the probes added later that need nothing newer than Ampere: `formats/fma`, `fma.all`, `scalar_edges`, `ftz`, `mbarrier/arrive_token`, the seven `e2e/ampere_replay` cases | identical to the predictions. `mbarrier/arrive_token.stale` records an **invalid-use observation**: the model now reproduces the 3080 Ti's parity fallback, but stale results and opaque token bits remain informational, not portable PTX expectations (finding 10 below) |
| **on the GPU**, 2026-11-24, the reruns of the Ampere programs: `build.sh --ampere`, then `run_all.py --binaries ampere --no-build --no-e2e --only 'e2e/ampere_*' --only 'formats/ampere_*' --only 'mbarrier/ampere_*'`, and `compare.py` on the archive | all 14 predicted runs **identical** to the predictions, which are the golden files: the eight cases of `e2e/ampere_replay` (`callforms` is new), `e2e/ampere_atom`, `e2e/ampere_reconvergence` (20 kernels, 100 launches each, none unstable), `mbarrier/ampere_token`, `formats/ampere_replay.minmax_cvt`, `.intdiv` and `.nan`. `formats/ampere_replay.nan_cvt_ftz` (no prediction) printed the 44 lines the golden file has for those two instructions. The three Ampere programs whose sources changed for this (`nan_golden`, `minmax_cvt_golden`, `callforms_golden`: shared error checks; the two `.ftz` conversions in a kernel of their own) were rebuilt with nvcc 12.4 and print their golden files unchanged. The four new programs also build as PTX programs, assemble to `sm_100a` and link with `--native`, which makes 55 programs |
| **on the GPU**, 2026-11-24, after `cvt` with `.ftz` and `vote` were added: `build.sh --ampere`, then `run_all.py --binaries ampere --no-build --no-e2e --only 'e2e/ampere_replay.vote' --only 'formats/ampere_replay.*'`, and `compare.py` on the archive | all 6 runs **identical** to the predictions: `e2e/ampere_replay.vote` (2220 launches), `formats/ampere_replay.nan` (now all 14270 lines), `.cvt_ftz` (8120 lines), `.minmax_cvt`, `.intdiv` and `.bitfield` |
| **another compiler**: every probe built by nvcc 13.0.88 (the toolkit `env/remote/setup.sh` falls back to on this machine's driver), then `run_all.py --mode predict` on those programs | all 98 predictions byte-identical to the committed ones, which come from nvcc 13.5.37 code. Before the barriers of finding 16 were added, 16 differed |
| **the whole rental, rehearsed**: `env/remote/run.sh --keep-going` on the 3080 Ti with `PREFLIGHT_EXPECT_CAP=8.6`, env from `setup.sh` (fallback toolkit 13.0, PyTorch env) | 13 min 52 s. Preflight: the new `predictions` check warns, `torch` passes, the three Blackwell checks fail as they must. Both suites build. `run-blackwell`: the canary fails from `bin/` and `bin-native/` (`cudaErrorNoKernelImageForDevice`: `sm_100a` code on `sm_86`), `NEITHER BUILD RUNS`; 71 runs `launch-failed`, 28 `host-error` (no tensor maps on Ampere), 2 `ok`; the stage fails with the count in capitals. `compare.py`: 71 `FAILED TO LAUNCH`, 28 `HOST ERROR`, none "by design". `run-timing`: every benchmark `LAUNCH FAILED`. The archive has `PREDICTIONS-NOT-COMPARABLE.txt` at its top and the banner is the last thing printed. An interrupt (Ctrl-C) during the self-test stopped the stage and still wrote the archive |
| end to end in the simulator | tutorials 1-4 pass their own check; Triton, the TIRx script and the CuTe DSL script match NumPy; the book's `hgemm_v1` to `v9` match NumPy |
| **end to end on the GPU** (`run_e2e.py --mode gpu`, in an env made by `env/remote/setup.sh`: toolkit 13.0 to match the driver, PyTorch 2.15.0+cu132 in its own venv) | `triton`: ran whole, vector add error 0, matmul error 5.72e-06 (on `sm_86` Triton emits `mma.sync`, not `tcgen05`). `unchanged_scripts.tirx` and `.cute_dsl`: ran, match NumPy. `book_hgemm`: every kernel was compiled by NVRTC and reached the driver, then failed as it must on Ampere (`CUDA_ERROR_INVALID_PTX` for v1 to v3, `cuTensorMapEncodeTiled: operation not supported` for v4 to v9), each with its own `FAILED` line. The CUTLASS tutorials were not tried on the GPU (`sm_100a` only) |

Only compiled and assembled, never executed on any GPU: every other run. That includes all
of TMA, tensor memory, `tcgen05`, clusters and cluster launch control,
`expect_tx`/`complete_tx`, `elect.sync`, `cvt.rn/.rz.tf32` and the small-format conversions.

With the development machine's own toolkit (`/usr/local/cuda`, nvcc 12.4, which has no
`sm_100a`): nothing here compiles as it is, because `common.h` uses
`cudaGetDriverEntryPointByVersion`, which 12.4 does not have. With that one call mapped to
`cudaGetDriverEntryPoint`, the probes that need nothing newer than `sm_90a` compile
(`tma/raw_bulk`, `hints`, `missing_fence`, `mbarrier/arrive_token`, `init_count`,
`cluster/max_size`, `address_dump`, `remote_arrive`, `formats/ftz`, `fma`, `scalar_edges`,
`e2e/smem_limit`, `elect`, `ampere_replay`); everything with `tcgen05` or
`clusterlaunchcontrol` does not ("Unknown modifier '.cta_group::1'"), which is the toolkit's
age, not the probes. The Ampere builds (`build.sh --ampere`) do compile with 12.4 as they are.

The Ampere build exists because `mbarrier.init`, `arrive` and `test_wait` date from `sm_80`.
`-DPROBE_AMPERE` leaves out `expect_tx`, `complete_tx` and the init fence, and turns `arrive`
with a count (which `sm_86` rejects without `.noComplete`) into single arrivals. So
`over_arrival` was not checked on Ampere.

## Found while writing the probes

Simulator gaps and oddities. None was fixed here; this suite does not touch the simulator.
Those marked "fixed since" were fixed on other branches.

1. **`__syncwarp()` did not run** (fixed since). nvcc emits `bar.warp.sync -1;`, which the
   simulator refused as an unsupported instruction. It runs now; `tmem/alloc.cu` still uses
   `__syncthreads()` in a one-warp block.
2. **`volatile` shared memory did not run** (fixed since). A store to
   `__shared__ volatile uint32_t x` compiles to `st.volatile.shared.b32`, which was refused.
   It runs now; `cluster/barrier.cu` still spins with an inline `add.u32`.
3. **The small-format conversion instructions were not implemented** (fixed since)
   (`cvt.rn.satfinite.e4m3x2.f32`, `cvt.rn.f16x2.e2m1x2`, ... as `cuda_fp8.h` and
   `cuda_fp4.h` emit them). `formats/small_formats` is now predicted by running it, like
   every other probe. `formats/predict_small_formats.py`, which computes the same lines from
   the `warplens.numerics` tables, is kept as a second implementation that a test compares
   with the predictions.
4. **Over-arrival did not carry the way its finding says** (fixed since, on branch
   `work/sim-bugs`). With one arrival expected, an `arrive` with count 5 completed one phase
   and left the pending count at -3; every later single arrival then completed one more
   phase and the count stayed at -3 for ever. `Mbarrier._check` now completes as many phases
   as the surplus covers (five here), so the count is positive again afterwards. What
   `mbarrier/over_arrival` prints is the same either way, because the probe prints parities
   and one phase and five have the same one; only the findings in
   `predicted/mbarrier/over_arrival.sim.txt` changed, from four to two. Hardware behaviour
   here is unknown and possibly undefined.
5. **An M = 64 MMA accepted any destination lane offset** (fixed since, on branch
   `work/sim-bugs`). `docs/m3.md` gives Layout F for offsets 0 and 16; the simulator added
   whatever lane field the destination had (offset 8 put the rows on lanes 8-23 of each
   window). It now refuses any other lane, and `tmem/layout_f.64_8` predicts a failed
   kernel.
6. **`arrive.expect_tx 0` before a load completes the phase at once**, and the load's bytes
   are then charged to the next phase. That is the signed-count model working as designed,
   but it means a box wholly outside the tensor cannot be used to ask whether `complete_tx`
   counts filled bytes; `tma/oob_complete_tx` uses partly-outside boxes only.
7. **Every kernel failure reaches the program as `cudaErrorLaunchFailure`.** A GPU
   distinguishes illegal instructions, illegal addresses and so on, so the probes print the
   name on an `info:` line. (The shim's `cudaGetErrorName` knew six names; it now knows the
   eleven codes the shim can return. A launch refused before the kernel runs is no longer
   among the failures: finding 13.)
8. Not the simulator: **`nvcc -arch=sm_100a -c` assembles for `.target sm_100`** with this
   toolkit and rejects `tcgen05` and `.cta_group` ("not supported on .target 'sm_100'").
   `-gencode arch=compute_100a,code=sm_100a` is what `build.sh` uses. The first version of
   `build.sh` had the short form and passed everything that did not use `tcgen05`.
9. Not the simulator: `warplens.numerics` rejects a NaN for E2M1 and negative values for
   UE8M0, where CUDA's conversions return something. Those inputs are printed on `info:`
   lines; what the GPU returns for them will be in the archive.

10. **Stale mbarrier tokens are invalid use.** PTX permits state waits only for the
    current or immediately preceding phase. Older or malformed tokens report
    `mbarrier-invalid-token` and grant no synchronization credit. The model's
    parity fallback now matches recorded RTX 3080 Ti results, including invalid
    inputs, without turning those observations into a Blackwell requirement.
    Token bits and stale predicates remain on `info:` lines. See `docs/m2.md`
    and matrix row MB-03.
11. **Atomic updates are implemented.** The forms nvcc 13.5 emits for the
    atomic functions run and match the recorded RTX 3080 Ti results
    (`validation/ampere/atom_golden.txt`), as do the forms nvcc 12.4 emits: the
    spellings without ordering or scope (`atom.global.add.u32`) and `red` for
    every operation but `exch` and `cas`. `add.s32` is accepted too. Review checks
    enforce natural alignment, global f32 FTZ and shared race/async diagnostics.
    Acquire/release, cluster scope and packed/half/vector forms remain
    unsupported; see `docs/atomics.md`. The CLC probes still use single-writer
    words.
12. **NaN bit patterns and integer zero divisors now match recorded Ampere results.**
    The model uses the RTX 3080 Ti's f32 NaN pattern and measured f64 payload/
    quieting rules (`validation/ampere/nan_golden.txt`). PTX leaves f32 NaN
    encodings unspecified but supports f64 payloads; the complete measured policy
    is not guaranteed on Blackwell. Integer `div`/`rem` by zero continue to report
    `integer-divide-by-zero` and now use Ampere's all-ones fallback. The probes
    compare NaN classification and record zero-divisor results on `info:` lines.
    See `docs/numerics.md` and matrix rows NUM-12 and NUM-13.
13. **Refused launches now carry their specific CUDA error codes.** Too much shared
    memory returns `cudaErrorInvalidValue`; invalid block/grid dimensions return
    `cudaErrorInvalidConfiguration`; invalid cluster shapes return
    `cudaErrorInvalidClusterSize`. Block dimensions are limited to 1024 x 1024 x 64
    (1024 threads total), and grid dimensions to 2147483647 x 65535 x 65535.
    `cudaGetLastError` reports the refusal once; synchronization does not repeat
    it. Dynamic shared-memory attributes above the reported 227 KiB are refused.
    The shared-memory, block and grid error codes were measured on Ampere at its
    own limits; the B200 cluster code follows CUDA headers and is not measured.
    Real kernel faults still become `cudaErrorLaunchFailure`. Clearing that fault
    with `cudaGetLastError` is a simulator recovery policy, not CUDA context recovery
    (`docs/m6.md`). Programs run under the shim now need the shared-memory opt-in a
    GPU needs: a launch with more dynamic shared memory than the 48 KiB default
    leaves after the kernel's static shared memory is refused with
    `cudaErrorInvalidValue` unless `cudaFuncSetAttribute` (`cuFuncSetAttribute`)
    raised the kernel's maximum first. The rule, including that static shared
    memory above 48 KiB needs no opt-in, was measured on Ampere
    (`validation/README.md`, "The shared-memory opt-in") and is assumed for the
    B200; `e2e/smem_limit` opts in before each launch, so it does not test it.
14. Not the simulator: **TVM's CUDA runtime does not find `libcudart.so.13` under plain
    `python`** (its library has no run path to the env's `lib`), and TIRx then reports "no
    CUDA device is available" on a machine with a working GPU. `e2e/run_e2e.py` sets
    `LD_LIBRARY_PATH` for the two steps that use TVM; `examples/07_tirx/run.py` and
    `examples/08_book/run.py` run by hand on a GPU need the same.
15. Not the simulator: **installing PyTorch into the project env changes which NVRTC
    `cuda-bindings` loads** (the CUDA 13.2 wheel torch requires, instead of the toolkit's
    13.5), and with it the PTX of every TIRx kernel. PyTorch therefore lives in a venv of
    its own (`env/remote/pins.env`).

16. **Fixed: warp rendezvous beyond the static reconvergence point.** Two layouts of
    `if (tid == 0) shared = 7; __syncthreads(); out[tid] = shared;` let lanes 1–31
    read before lane 0 wrote, because `bar.sync` counted a partial SIMT path as a
    whole warp. Named CTA barriers now run pending paths until every non-exited
    lane reaches the instruction. `tcgen05` `.sync.aligned` operations also collect
    paths before their full-warp check, fixing the compiler-merged allocation
    refusal. All four synchronized layouts in `findings/reconvergence.py` now
    return 32 correct lanes; tests also cover multiple warps, `bar.arrive`, exits,
    repeated phases and timing. Unsynchronized reads have no implicit warp-wide
    communication guarantee and remain dependent on scheduling. Rendezvous at
    different instruction addresses or across divergent recursive calls remains
    unsupported. The probes retain their explicit synchronization between setup
    and allocation, keeping the generated predictions easy to compare.

    The implementation's early grouping policy (`core/warp.py:_regroup`) also
    joins paths that meet before their static postdominator. This matches the
    recorded RTX 3080 Ti runs in `validation/ampere/reconvergence_golden.py`:
    all four layouts read the expected word even without the barrier. That
    observation is specific to those compiled programs; it does not make
    unsynchronized communication portable. The explicit rendezvous above is
    still needed when paths meet inside a control-flow cycle, which the early
    grouping policy deliberately leaves alone.

17. **`cvt` with `.ftz` was not implemented** (fixed), and the simulator refused such a kernel
    at its launch, before any of it ran: the rerun of `validation/ampere/nan_golden.cu`, which
    has `cvt.ftz.f64.f32` and `cvt.rn.ftz.f32.f64`, ended with `warplens: unsupported
    instruction 'cvt.rn.ftz.f32.f64' (modifier .rn)`, `KERNEL FAILED`. For a while the two
    instructions were a run of their own without a prediction
    (`formats/ampere_replay.nan_cvt_ftz`). The simulator now implements `cvt` with `.ftz` as
    the 3080 Ti does it (`warplens/units/alu.py:_float_to_float`,
    `validation/ampere/cvt_ftz_golden.txt`, 8120 lines; `docs/numerics.md`), the 44 lines are
    back in `formats/ampere_replay.nan`, whose prediction is the whole golden file, and
    `formats/ampere_replay.cvt_ftz` reruns the new golden program. The same change added
    `cvt.{rni,rzi,rmi,rpi}.f32.f32` and `.f64.f64` (`floorf`, `ceil`, ...), which the
    simulator also refused.

## Adding a probe

Write `<area>/<name>.cu` including `../common.h`; print inputs beside outputs; isolate one
claim; put anything that may fault or hang behind a command-line case. Add the runs to
`manifest.py`, run `./predict.sh --only '<area>/<name>*'`, read the prediction, run
`python manifest.py --update-readme`, and commit all of it with the source. `tests/test_blackwell_probes.py` checks that every source is in the manifest
and every run has its prediction.

The collectors finish and archive results even when a probe faults or times out. A collector
returns nonzero for missing required binaries, build failures, or an E2E runner that could
not complete; this reports incomplete collection rather than a hardware verdict. SIGINT
and SIGTERM stop the detached probe processes before the collector exits.

Do not let a test on one thread (`if (tid == 0) {...}`) run straight into a warp-wide
instruction (`tcgen05.alloc`, anything `.sync.aligned`) or into code that depends on what
that thread did: put a `__syncthreads()` between them. Finding 16 says why.
