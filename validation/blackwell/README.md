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
| `predicted/<name>.txt` | the predicted standard output of a run |
| `predicted/<name>.sim.txt` | how the simulator's run ended and what the simulator said: findings, or why it refused |
| `compare.py` | compares an archive with `predicted/` and prints one verdict per run |
| `common.h`, `tcgen05_common.h`, `tma/tma_dump.h` | the PTX wrappers and the shared kernels |
| `e2e/run_e2e.py`, `e2e/triton_gpu.py` | the end-to-end programs |
| `tests/test_blackwell_probes.py` (in the repository's `tests/`) | fails when a committed prediction is no longer what the simulator prints |

Conventions the probes share:

- A line that starts with `info:` carries something machine-dependent (a device name, an
  address, an error name) and is never compared.
- A wait whose outcome is the thing under test is a bounded poll (`POLL_BUDGET` = 20000
  `test_wait`s, on the order of a millisecond on a GPU), so the kernel returns either way and
  prints `phase completed 0` or `1`. The assumption that any copy a probe issues lands within
  that budget is the suite's; a `0` that should be a `1` would point at it.
- After a failed launch the program prints `KERNEL FAILED: ...` and exits with status 3.
  On a GPU nothing after a fault can be trusted, so anything risky is its own run.
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
`nvidia-smi`, driver, toolkit and device, runs the 59 probe runs in manifest order, then the
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

**Expected run time** (estimated; nothing here has been timed on a B200): the build takes a
minute or two. Most runs take well under a second of GPU time plus process start-up. Runs
that hang are killed at their timeout, 30 s each; at most 10 runs are allowed to hang, and two
of them (`mbarrier/expect_tx_large`, `tmem/alloc_blocks`) are expected to. So the probes
take between 2 and 8 minutes. The end-to-end part is dominated by fetching and compiling
CUTLASS and by Triton's and TVM's compilers: allow 15 minutes. In the simulator the whole
suite takes about one minute on the development machine.

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

### `e2e/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `device_attributes` | `device_attributes` | RT-03 | the device reports compute capability 10.0, 148 SMs and the shared-memory limits the shim gives | runs | - |

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

### `mbarrier/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `phase_parity` | `script basic` | MB-01 | a new mbarrier is in phase 0; test_wait.parity p is true when the phase's parity is not p | runs | - |
| `phase_parity.counts` | `script count` | MB-01, MB-02 | the arrival count is reloaded at each completion; arrive with a count; init restarts | runs | - |
| `arrivals_and_bytes` | `script tx` | MB-02 | a phase completes when pending arrivals and the byte count are both zero, in either order | runs | - |
| `lane_arrivals` | `lane_arrivals` | MB-09 | every lane that executes an arrive arrives once, also when only part of a warp does | runs | - |
| `complete_before_expect` | `script negative` | MB-07 | the byte count is signed: complete_tx before expect_tx completes once balanced | runs | fault |
| `over_arrival` | `script over` | MB-04 | arrivals beyond the pending count complete the phase and carry into the next | runs (2 finding(s)) | fault |
| `expect_tx_small` | `expect_tx` | MB-05, MB-06, MB-07, MB-08 | undersized, oversized and missing expect_tx against real TMA loads, with bounded polls | runs | - |
| `expect_tx_small.wait` | `expect_tx wait-small` | MB-05, MB-11 | an undersized expect_tx with one load never completes: the usual wait loop deadlocks | **refuses**: deadlock in CTA (0, 0, 0) | hang |
| `expect_tx_large` | `expect_tx wait-large` | MB-08, MB-11 | an oversized expect_tx never completes: the usual wait loop deadlocks | **refuses**: deadlock in CTA (0, 0, 0) | hang |

### `tmem/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `lane_window` | `lane_window` | TM-01, TM-02 | a warp reaches only lanes 32 (w % 4) .. + 31; a lane field naming another window reads its own | runs | - |
| `lane_window.16` | `lane_window 16` | TM-03 | lane field 16: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-m | fault |
| `lane_window.1` | `lane_window 1` | TM-03 | lane field 1: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-m | fault |
| `lane_window.48` | `lane_window 48` | TM-03 | lane field 48: a 32x32b transfer running past the warp's window is refused | **refuses**: unsupported instruction `tcgen05.ld.sync.aligned.32x32b.x1.b32` (a transfer spanning the local 32-row tensor-m | fault |
| `layout_f` | `layout_f` | TM-04, TM-01 | M = 128: row m on lane m. M = 64: row m on lane 32 (m / 16) + m % 16 + a, a = 0 or 16 | runs | - |
| `layout_f.64_8` | `layout_f 64 8` | TM-04 | M = 64 with lane offset 8, neither of Layout F's two positions: refused | refused | fault |
| `transfer_shapes` | `transfer_shapes` | TM-05 | thread and register to (lane, column) for 32x32b, 16x64b, 16x128b, 16x256b, load and store | runs | - |
| `alloc_addresses` | `alloc` | TM-07 | alloc returns the lowest free block aligned to its size, as (lane 0, column) | runs | - |
| `alloc_increasing` | `alloc increasing` | TM-08 | a later allocation may not be larger than an earlier one | **refuses**: tcgen05.alloc of 64 columns increases the previous allocation size of 32 columns | fault |
| `alloc_addresses.size_48` | `alloc size-48` | TM-07 | an allocation of 48 columns is refused | **refuses**: tcgen05.alloc of 48 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_16` | `alloc size-16` | TM-07 | an allocation of 16 columns is refused | **refuses**: tcgen05.alloc of 16 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_0` | `alloc size-0` | TM-07 | an allocation of 0 columns is refused | **refuses**: tcgen05.alloc of 0 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_addresses.size_1024` | `alloc size-1024` | TM-07 | an allocation of 1024 columns is refused | **refuses**: tcgen05.alloc of 1024 columns; the count must be a power of two from 32 to 512 | fault |
| `alloc_after_relinquish` | `alloc after-relinquish` | TM-09 | no allocation after relinquish_alloc_permit | **refuses**: tcgen05.alloc after this CTA relinquished its allocation permit | fault |
| `alloc_blocks` | `alloc exhaust` | TM-10 | an allocation that does not fit waits (here: forever) | **refuses**: deadlock in CTA (0, 0, 0) | hang |
| `dealloc_repeated` | `alloc dealloc-twice` | TM-11 | a second dealloc of the same columns is ignored | runs (1 finding(s)) | hang |
| `dealloc_mismatch` | `alloc dealloc-wrong-size` | TM-12 | a dealloc with another size than the allocation's is refused | **refuses**: tcgen05.dealloc of 32 columns at column 0 does not match an allocation | hang |
| `alloc_leak` | `alloc leak` | - | columns never released are reported and do not affect the next launch | runs (1 finding(s)) | hang |

### `mma/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `operand_layouts` | `operand_layouts` | MMA-01, MMA-02, MMA-03 | K-major (f16, tf32) and MN-major (f16) operand addressing in all four swizzle modes; exact results | runs | - |
| `operand_layouts.tf32_mn` | `operand_layouts tf32-mn` | MMA-03 | MN-major tf32 is refused (needs a 128B swizzle with 32-byte atomicity) | **refuses**: unsupported instruction `tcgen05.mma.cta_group::1.kind::tf32` (MN-major TF32 requires 128B swizzling with 32-b | fault |
| `accumulation_order` | `accumulation_order` | MMA-04, MMA-05 | none exactly: the simulator sums in float32 in NumPy's order; hardware may differ in the last bits | runs | - |
| `negate` | `negate` | MMA-07, MMA-05 | each negate bit flips the product's sign, both leave it; a second MMA accumulates | runs | - |
| `tf32_low_bits` | `tf32_low_bits` | MMA-08 | kind::tf32 multiplies the operands as stored float32 values; low bits take part | runs (1 finding(s)) | - |

### `cluster/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `barrier` | `barrier` | CL-01, CL-03, CL-04 | the cluster barrier holds every thread until all have arrived; mapa + ld.shared::cluster | runs | - |
| `multicast` | `multicast` | CL-07 | a multicast load writes the box in every CTA of the mask and completes each one's mbarrier | runs | - |
| `multicast.outside` | `multicast outside` | CL-07 | a mask naming a rank outside the cluster is refused | **refuses**: CTA rank 4 does not exist in a cluster of 4 CTA(s) | fault |
| `pair_accounting` | `pair_accounting` | CL-08 | with .cta_group::2 a pair's completions go to the member with the mbarrier address's parity | runs (3 finding(s)) | - |
| `pair_mma.m256_issuer0` | `pair_mma 256 0` | CL-10, CL-13, CL-09 | two-CTA MMA, M = 256, issued by rank 0: each CTA holds 128 rows on 128 lanes | runs | hang |
| `pair_mma.m128_issuer0` | `pair_mma 128 0` | CL-11, CL-09 | two-CTA MMA, M = 128, issued by rank 0: Layout B: 64 rows per CTA, columns folded onto lanes 64-127 | runs | hang |
| `pair_mma.m256_issuer1` | `pair_mma 256 1` | CL-12 | two-CTA MMA, M = 256, issued by rank 1: each CTA holds 128 rows on 128 lanes | runs | hang |
| `pair_mma.m128_issuer1` | `pair_mma 128 1` | CL-12 | two-CTA MMA, M = 128, issued by rank 1: Layout B: 64 rows per CTA, columns folded onto lanes 64-127 | runs | hang |

### `formats/`

| Run | Program and arguments | Matrix | Claim tested | In the simulator | On hardware may |
| --- | --- | --- | --- | --- | --- |
| `fp8_codes` | `small_formats fp8` | NUM-01, NUM-02 | E4M3 and E5M2: every code decodes as modeled; encoding rounds to nearest even and saturates | refused; predicted from `warplens.numerics` | - |
| `fp4_codes` | `small_formats fp4` | NUM-03 | E2M1: 16 codes; ties to even, saturating at 6 | refused; predicted from `warplens.numerics` | - |
| `e8m0_ties` | `small_formats e8m0` | NUM-04 | UE8M0: code c is 2^(c - 127); a midpoint between two powers of two goes to the larger | refused; predicted from `warplens.numerics` | - |
| `special_inputs` | `small_formats special` | NUM-06 | NaN, infinities, zeros and out-of-range inputs for the four formats | refused; predicted from `warplens.numerics` | - |
| `tf32_modes` | `tf32_modes` | NUM-07, NUM-08 | cvt.rn/.rz.tf32.f32 round the magnitude bits like .rna with their own tie rule; .rna as on Ampere | runs | - |

### End to end (`e2e/run_e2e.py`)

| Run | What | Matrix | Prediction |
| --- | --- | --- | --- |
| `cutlass_tutorials.tutorial_01` .. `_04` | the four CUTLASS Blackwell tutorials, unmodified, built by `examples/03_cutlass_mma/build.sh --programs`; the same binaries on the GPU and in the simulator | RT-06 | committed (`predicted/e2e/`), from the simulator |
| `triton` | the vector add and the matmul of `examples/05_triton`. On the GPU through `e2e/triton_gpu.py`, which imports the kernels from the example and gives them PyTorch tensors, because the example itself always installs the simulator's driver | RT-07 | committed, from `warplens run examples/05_triton/run.py` |
| `book_hgemm` | `hgemm_v1` to `v9` of *Modern GPU Programming for MLSys* through `examples/08_book/run.py <clone>`. The book has no license; its code is never copied here, and the clone is an argument (`--book`, or `WARPLENS_BOOK`). Skipped without it | RT-08 | **not committed**: no clone on the development machine. `tests/test_book.py` runs the same script in the simulator in CI |

These print error norms and "matches NumPy" lines, not bit patterns, and their outputs
contain times. `compare.py` treats them as informational and shows the result lines from the
GPU next to the simulator's.

## Reading the comparison

```
probe                              verdict                     detail
tma/swizzle_misaligned             identical
mma/accumulation_order             rounding                    812 of 2968 float32 words differ, by at most 2 ulp
tmem/lane_window.16                informational               the GPU ran what the simulator refuses (...)
mbarrier/expect_tx_large           timed out (by design)
tmem/alloc_increasing              kernel failed (by design)   ...
tma/oob_fill                       DIFFERS
```

(The lines above illustrate the format; they are not results.) `identical`: the claim held.
`DIFFERS`: it did not; the first differing lines follow the table, with the claim. `rounding`:
only for `mma/accumulation_order`, which the simulator does not claim to match bit for bit
(`docs/numerics.md`). `informational`: no prediction; read the output. A run that is refused
in the simulator and *also* fails on the GPU counts as `identical`: both print the same
`KERNEL FAILED` line. The error names differ (`info:` lines) and are in the archive.

`tma/tensormap_encode_sweep` has no prediction and never will: it prints, for 1380
parameter sets, the 128 bytes the driver's `cuTensorMapEncodeTiled` wrote. That is the data
a decoder of real tensor maps needs (`docs/m6.md`, "Not working: kernels that use TMA").
Under `warplens run` it is skipped; the program itself was run once against the shim only
to see that it runs to the end.

## Matrix coverage

Rows of `docs/validation-matrix.md` with at least one run here: TMA-01 to -07, -09 to -12,
-15; MB-01, -02, -04 to -09, -11; TM-01 to -05, -07 to -12; MMA-01 to -05, -07, -08; CL-01,
-03, -04, -07 to -13; NUM-01 to -04, -06 to -08; RT-03, -06 to -08.

Partly covered, and how: MMA-01 (only the M, N, format and major-ness fields are varied);
MMA-05 (repeated accumulation of the same product; no case that isolates "one float32
addition"); MB-11 (`try_wait` is only used in a loop; whether a single `try_wait` returns
false on hardware is not printed); CL-01 (ranks along x only); CL-09 and CL-13 (exercised
inside `pair_mma`, not isolated); TMA-10 (the unaligned coordinate; an unaligned shared
address is `tma/swizzle_misaligned.sixteen`, which is 16-byte aligned, so a shared address
that is not a multiple of 16 is not tried).

**No probe yet:** TMA-08 (raw `cp.async.bulk`), TMA-13 (missing fence), TMA-14 (hints);
MB-03 (arrive token), MB-10 (init count limits); TM-06 (repeat counts; needs only ptxas),
TM-13 (fresh contents); MMA-06 (`scale-input-d`), MMA-09 (disable-output-lane), MMA-10
(commit arrival, isolated), MMA-11 (swizzle mismatch); CL-02, CL-05, CL-06; NUM-05 (UE4M3:
CUDA has no conversion for it), NUM-09 to -12 (`.ftz`, `fma`, scalar edges: Ampere-checked
already, not replayed); RT-01, -02, -04, -05, -09 to -11. TIM-* belong to
`validation/blackwell_timing/`.

## Dry run

What was done on the development machine (RTX 3080 Ti Laptop GPU, `sm_86`, driver
580.173.02; nvcc 13.5.37 from the `warplens` conda env, compile-only), on 2026-11-23:

| Step | Result |
| --- | --- |
| `build.sh`: all 24 programs built as PTX-carrying programs **and assembled by ptxas to `sm_100a` machine code** | all 24 pass |
| `build.sh --native`: the same as `sm_100a` machine-code programs | all 24 link; none can run here |
| `run_all.py --mode sim`: the whole harness, each probe under `warplens run`, archive, then `compare.py` on the archive | 58 of 59 runs executed (the sweep is skipped); 54 identical to `predicted/`, the 4 small-format runs fail as predicted (the simulator has no such instructions), 61 s in total including the four CUTLASS tutorials and Triton |
| the timeout path: `--mode sim --timeout-scale 0.004` | the run is killed and recorded as `timeout`; the next run proceeds; `compare.py` reports `timed out` |
| **on the GPU**: `build.sh --ampere`, `run_all.py --binaries ampere --no-build --only ...` | `mbarrier/phase_parity`, `mbarrier/phase_parity.counts` and `mbarrier/lane_arrivals` ran on the 3080 Ti and are **identical** to the predictions. `e2e/device_attributes` ran and differs, as it should (8.6, 58 SMs): that is the check that catches a wrong machine. The environment capture (`nvidia-smi`, `compute_cap`, driver, ...) was exercised by the same run |
| end to end in the simulator | tutorials 1-4 pass their own check; Triton matches NumPy. The book was **not** run (no clone here) |

Only compiled and assembled, never executed on any GPU: every other run. That includes all
of TMA, tensor memory, `tcgen05`, clusters, `expect_tx`/`complete_tx`, `cvt.rn/.rz.tf32`
and the small-format conversions. `e2e/triton_gpu.py` has not been run at all (no PyTorch in
the project environment); it is the least tested file here.

The Ampere build exists because `mbarrier.init`, `arrive` and `test_wait` date from `sm_80`.
`-DPROBE_AMPERE` leaves out `expect_tx`, `complete_tx` and the init fence, and turns `arrive`
with a count (which `sm_86` rejects without `.noComplete`) into single arrivals. So
`over_arrival` was not checked on Ampere.

## Found while writing the probes

Simulator gaps and oddities. None was fixed here; this suite does not touch the simulator.

1. **`__syncwarp()` does not run.** nvcc emits `bar.warp.sync -1;`, and the simulator stops
   with `unsupported instruction 'bar.warp.sync'`. Reproduction: a kernel whose body is
   `__syncwarp();`. `tmem/alloc.cu` uses `__syncthreads()` in a one-warp block instead.
2. **`volatile` shared memory does not run.** A store to `__shared__ volatile uint32_t x`
   compiles to `st.volatile.shared.b32`, refused as `unsupported instruction
   'st.volatile.shared.b32' (modifier .volatile)`. Reproduction:
   `__shared__ volatile unsigned x; x = 1;`. `cluster/barrier.cu` spins with an inline
   `add.u32` instead.
3. **The small-format conversion instructions are not implemented**
   (`cvt.rn.satfinite.e4m3x2.f32`, `cvt.rn.f16x2.e2m1x2`, ... as `cuda_fp8.h` and
   `cuda_fp4.h` emit them). `docs/numerics.md` says the formats are host-side only; the
   consequence here is that `formats/small_formats` cannot be predicted by running it, so
   its predictions come from `formats/predict_small_formats.py`, a second implementation of
   the probe's input generation that has to be kept in step by hand.
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
7. **Every kernel failure reaches the program as `cudaErrorLaunchFailure`** (the shim's
   `cudaGetErrorName` knows six names). A GPU distinguishes illegal instructions, illegal
   addresses and so on, so the probes print the name on an `info:` line.
8. Not the simulator: **`nvcc -arch=sm_100a -c` assembles for `.target sm_100`** with this
   toolkit and rejects `tcgen05` and `.cta_group` ("not supported on .target 'sm_100'").
   `-gencode arch=compute_100a,code=sm_100a` is what `build.sh` uses. The first version of
   `build.sh` had the short form and passed everything that did not use `tcgen05`.
9. Not the simulator: `warplens.numerics` rejects a NaN for E2M1 and negative values for
   UE8M0, where CUDA's conversions return something. Those inputs are printed on `info:`
   lines; what the GPU returns for them will be in the archive.

## Adding a probe

Write `<area>/<name>.cu` including `../common.h`; print inputs beside outputs; isolate one
claim; put anything that may fault or hang behind a command-line case. Add the runs to
`manifest.py`, run `./predict.sh --only '<area>/<name>*'`, read the prediction, and commit it
with the source. `tests/test_blackwell_probes.py` checks that every source is in the manifest
and every run has its prediction.
