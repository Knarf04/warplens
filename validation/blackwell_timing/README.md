# Timing calibration (plan step C14)

Microbenchmarks that each isolate one parameter, or one group of parameters, of
the approximate timing model (`warplens/timing/model.py`,
`warplens/configs/b200.yaml`), for one unattended run on a rented B200.

**Nothing here has run on a Blackwell GPU.** What exists today:

- every benchmark compiles and assembles for `sm_100a` (`build.sh`);
- every benchmark runs in the simulator, and what the uncalibrated model predicts
  for each configuration is committed in `predicted/`;
- the benchmarks that are valid on Ampere ran on an RTX 3080 Ti Laptop GPU through
  the same `run_all.sh` and `fit.py` (`dryrun_3080ti/`); that shows the measuring
  method is stable, and says nothing about Blackwell;
- `fit.py` recovers known parameters from synthetic results (`fixture_synthetic/`).

## On the rented machine

```bash
cd validation/blackwell_timing
NVCC=nvcc ./build.sh hw          # needs CUDA 12.8 or newer (sm_100a); or copy bin/hw from here
./run_all.sh                     # no interaction; 2 to 5 minutes plus ncu (estimate, below)
# -> results_<host>_<time>/ and results_<host>_<time>.tar.gz: bring the archive back
```

Then, anywhere with Python and PyYAML (the simulator is not needed):

```bash
python fit.py results_<host>_<time>          # report.md, derived.yaml, calibrated.yaml, ...
```

`run_all.sh` runs every benchmark in its own process under a timeout (900 s each,
`--timeout`), records the exit status and goes on after a failure or a hang. It
saves `env.txt` (driver, `nvidia-smi`, other processes on the GPU, compiler,
checksums of the programs), runs `run_ncu.sh`, runs `fit.py` if it can, and packs
the directory. `--scale 4` makes every launch four times longer if the spread in
`report.md` is too large; `--trials N` changes the number of trials (default 5).

Build with the same compiler as the predictions if possible (nvcc 13.5 here). The
loops are written so that the compiler has little freedom (the timed accesses are
inline PTX), but `clock_ghz` is derived from an instruction count taken from the
PTX compiled here. The safest route is to copy `bin/hw` from this machine: the
programs link the CUDA runtime statically.

Before paying for time, check: `nvidia-smi` shows a B200 (compute capability
10.0, not a consumer card), nobody else is on the GPU, and whether `ncu` works
(`./run_ncu.sh` says so in a few seconds and is harmless if it does not).

## How a benchmark measures

Each program takes `--trials`, `--scale`, `--only SUBSTRING` and `--sim`, and prints
one `DEVICE key=value ...` line and one `BENCH key=value ...` line per kernel launch:

```
BENCH id=tma/load/warm/bytes=4096/ctas=1 bytes=4096 nctas=1 ... reps=20000 trial=0 event_ns=... wall_ns=... clk=... check=...
```

- A configuration (`id`) is launched with two repetition counts, R and 2R, in each
  of 5 trials, after one warm-up launch (`trial=-1`, ignored). The kernel repeats
  the operation `reps` times in one loop.
- **Everything reported is the slope of time against repetitions.** That removes
  launch overhead, kernel setup and the first cold iteration, which is how effects
  below a microsecond are measured with a timer that resolves about half a
  microsecond: a launch lasts tens of milliseconds.
- `event_ns` is `cudaEvent` time around the launch, `wall_ns` host time from launch
  to `cudaDeviceSynchronize`, `clk` the difference of two `%clock64` reads around the
  loop in thread 0 of CTA 0 (not affected by launch overhead at all). `fit.py` uses
  `event_ns` and reports `clk` per repetition next to it.
- `fit.py` reports, per configuration, the slope through the fastest launch at each
  repetition count ("min"), the median of the per-trial slopes, and the spread
  (max - min) / median of the per-trial slopes. Parameters use "min"
  (`--stat median` for the other).
- `check` is a value the kernel computed, to show it did the work (for `mma_rate` it
  must equal K times the batch size).
- Before measuring, each program keeps every SM busy for about 0.2 s so that a GPU
  that lowered its clocks while idle is back at speed.

Two builds of the same source: `bin/hw` reads `%clock64`; `bin/sim` does not,
because the simulator rejects that register, and carries PTX for `warplens run`.
With `--sim` the programs use tiny repetition counts, one trial and no warm-up.

## The benchmarks

`TIM-nn` are the rows of `docs/validation-matrix.md` (branch
`work/validation-matrix`). "delta" means: parameter = config value + (measured -
predicted) in clocks, i.e. the value that makes the model reproduce the measured
loop, whose other instructions the simulator counted (`predicted/`).
`f` is `clock_ghz` as fitted by `alu_chain`.

| Program | Calibrates | From measured quantity to parameter | Matrix |
| --- | --- | --- | --- |
| `launch` | nothing (the model has no launch cost); prints the device facts `sm_count`, `hbm_capacity_gb` | SM count from `cudaGetDeviceProperties`; memory = `cudaMemGetInfo` total / 2^30. Launch time min/median reported | TIM-13 |
| `alu_chain` | `clock_ghz` | `f = clock_ghz(config) * predicted / measured` ns per repetition, one warp. Equivalently PTX instructions per repetition / measured ns. Also `%clock64` ticks per ns. `warps=1..32` tests the four issue slots (hard-coded, no parameter) | TIM-01, TIM-02 |
| `smem_wavefronts` | `smem_wavefront_clk` | slope of ns per access against conflict degree (1 to 32 wavefronts, `load_chain`) times `f`. `load_indep`, `store` and `warps=4` are structure checks | TIM-03 |
| `global_load` | `global_access_latency_clk` | delta, from `load_chain/footprint=l2` (one dependent load per repetition). `l1` and `dram` footprints and `store` are reported | TIM-08 |
| `gstream` | `hbm_bandwidth_tb_s` | STREAM Triad over arrays far larger than L2: 24 bytes / (slope of launch time against elements) / 1000 | TIM-07 (the config's cited figure is a Triad) |
| `cp_async` | nothing of its own (the model charges `cp.async` at the TMA engine's rate and latency, and its source to L2 or HBM as for a TMA load) | measured against predicted round trip of a 32 x 16-byte group, warm and cold source | TIM-14 |
| `mbar` | nothing (mbarrier instructions cost one clock each in the model) | ns per arrive + wait round trip; the overhead floor inside the `tma` and `mma_rate` intercepts | (none) |
| `tma` | `tma_latency_ns`, `tma_bytes_per_clk`, `l2_bandwidth_per_sm_gb_s`; cross-check of `hbm_bandwidth_tb_s` | Round trip T of one copy against tile bytes (64 B to 32 KiB), straight-line fit. `tma_bytes_per_clk` = 1 / slope / `f`, from the faster of warm loads and stores. `l2_bandwidth_per_sm_gb_s` = 1 / slope of warm loads. `tma_latency_ns` = intercept of warm loads - the instruction time the simulator counts (delta on the intercept). Cold loads on 8, 37 and all SMs: CTAs x bytes / (T - fixed cost per copy) | TIM-04, TIM-05, TIM-06, TIM-07 |
| `mma_rate` | `tensor_core_tflops_f16`, `tensor_core_tflops_tf32`, `mma_latency_clk` | Round trip of a batch of J MMAs + commit + wait, against J = 1, 2, 4, 8. TFLOPS = 2 M N K / slope x SM count / 1000 (M 128, N 256; the all-SM variant if present). `mma_latency_clk` = delta on the intercept (f16, M 128, N 256, one CTA). M = 64/128 x N = 32/128/256 x f16/tf32 are reported; `kind=i8` is hardware-only | TIM-10, TIM-11, TIM-12 |
| `tmem_access` | `tmem_access_clk` | delta, from `ld/x1` (one `tcgen05.ld` + `wait::ld` per repetition). `st/x1`, `.x8` and the loops without a wait per access are reported | TIM-09, TIM-19 |

Valid on Ampere, and run on the 3080 Ti: `launch`, `alu_chain`, `smem_wavefronts`,
`global_load`, `gstream`, `cp_async`. Blackwell only (simulator and `sm_100a`
assembly here): `mbar`, `tma`, `mma_rate`, `tmem_access`.

Per benchmark: method details are in the comment at the top of each `.cu` file.

### Expected run time

Estimated from the predicted time per repetition (16 R repetitions per
configuration) and from the Ampere run; not measured on a B200.

| Program | Configurations | Estimate on B200 | 3080 Ti dry run |
| --- | --- | --- | --- |
| `launch` | 2 (200 launches each) | 1 s | 1 s |
| `alu_chain` | 7 | 5 s | 4 s |
| `smem_wavefronts` | 20 | 10 s | 13 s |
| `global_load` | 6 | 10 s | 7 s |
| `gstream` | 1 | 5 s (6 GiB allocated) | 1 s |
| `cp_async` | 2 | 2 s | 1 s |
| `mbar` | 2 | 2 s | |
| `tma` | 22 | 20 s (16 GiB allocated and filled) | |
| `mma_rate` | 104, plus 4 for i8 | 30 s | |
| `tmem_access` | 6 | 2 s | |
| `run_ncu.sh` | 9 profiled programs | 5 to 15 min if counters are readable, seconds if not | refused (no permission) |

### Confounders

- **Timer resolution.** `cudaEvent` resolves about 0.5 us and a launch costs a few
  microseconds (3080 Ti: 3.9 us minimum, 4.7 us median for an empty kernel). Both
  cancel in the slope; what remains is their jitter against launches of 10 ms or
  more. If spreads are large, use `--scale`.
- **Frequency and power state.** On the 3080 Ti the first launch after start ran
  at 1.45 `%clock64` ticks per ns and later ones at 1.88: the GPU had clocked down.
  The spin-up at program start and the warm-up launch address this; `clk` per
  repetition is the check (it is in clock ticks, so it does not move with
  frequency, while ns do). `env.txt` records the clocks `nvidia-smi` reports.
  Thermal throttling during a run shows up as trials drifting in one direction.
- **L2 state.** "Warm" variants touch the same bytes every repetition and are
  warmed by the warm-up launch. "Cold" variants never reuse a region until the
  whole buffer (16 GiB on the B200, if a quarter of free memory allows) has been
  walked; that only makes them cold if the buffer is much larger than L2. The
  tensor maps use no L2 promotion. `global_load`'s footprints (16 KiB, 2 MiB,
  1 GiB) assume L1 > 16 KiB and L2 > 2 MiB per SM.
- **Other tenants.** A shared GPU or a desktop on it adds launches between ours
  and takes memory bandwidth. `env.txt` lists other compute processes. Slopes from
  the fastest launches ("min") are the robust figure; a large gap between min and
  median means interference. The bandwidth benchmarks (`gstream`, cold `tma` on
  all SMs) are the most exposed.
- **The compiler and the assembler.** With a fixed address, ptxas removed the
  repeated shared-memory loads and stores on the 3080 Ti; the independent accesses
  therefore move between two adjacent words. A different toolkit may optimize
  differently: compare `clk` per repetition and `check` with `dryrun_3080ti/` and
  the instruction counts in `predicted/predictions.md` if a number looks impossibly
  small.
- **JIT.** `bin/hw` holds machine code. `bin/sim` holds PTX only; if that is run on
  a GPU the driver compiles it at the first launch, which the warm-up absorbs.
- **What a round trip contains.** The `tma` and `mma_rate` intercepts include the
  mbarrier wake-up and the wait loop, which the model does not charge beyond their
  instructions. `mbar` measures that floor separately.

## ncu counters

`run_ncu.sh [BIN] [OUT]` first profiles one empty kernel with one counter. If
`ncu` is missing, lacks permission (`ERR_NVGPUCTRPERM`; it needs root or a driver
set up for non-admin profiling) or prints no counter table, it says why, writes
`STATUS.txt` and exits 0; `run_all.sh` continues. It never uses sudo. Otherwise
it saves raw `ncu --csv` output per benchmark (instructions executed, cycles,
shared-memory wavefronts and bank conflicts, cache hit rates, DRAM bytes), run
with one trial and 1% of the repetitions because every launch is profiled. If a
counter name is unknown to that GPU or ncu version, the run is repeated with
`smsp__inst_executed.sum` alone. It also profiles `validation/ampere/transpose_hw`
(TIM-16) and compares with the simulator when Python and warplens are present.
On this machine it stops at the permission check, so the profiling branch is
untested.

## Predictions and fitting

```bash
./build.sh sim
PYTHONPATH=<repo> python predict.py                  # rewrites predicted/ (about 3 minutes)
PYTHONPATH=<repo> python predict.py --out again --config results_x/fit/calibrated.yaml
```

`predicted/<bench>.out` holds each launch's BENCH line with the simulator's
numbers appended (`sim_ns`, `sim_instr`, `sim_bound`); `predictions.md` is the
table; `meta.json` has the config used and what was not simulated.
The second command re-predicts with a calibrated config; `fit.py --predicted
again --config .../calibrated.yaml` then shows ratios near 1 where the model can
reproduce the measurements, and the remaining ratios are what no parameter fixes.

`fit.py` writes into `<results>/fit/`:

| File | Content |
| --- | --- |
| `report.md` | (a) measured values, (b) derived parameters with formula and inputs, other measured quantities, (c) measured against predicted with the ratio |
| `measured.csv`, `comparison.csv` | (a) and (c) as tables |
| `derived.yaml` | the parameters measured, in the schema of `warplens/configs/*.yaml` |
| `calibrated.yaml` | `b200.yaml` with those entries replaced; `GpuConfig.load(path)` reads it, or copy it over `warplens/configs/b200.yaml` |

Results that did not come from compute capability 10.x, or that are synthetic, are
labelled in every file, and their `source` starts with `assumption` so the
simulator's reports never show them as measured.

The simulator cannot run:

- `mma_rate/kind=i8`: only `kind::f16` and `kind::tf32` are implemented, and the
  timing model has no rate for other kinds. It runs on hardware only, in its own
  process (`mma_rate_i8.out`).
- the `%clock64` reads: not implemented, hence the two builds.
- `gstream` runs, but its simulated time is not a prediction (below).

### Synthetic check

`make_synthetic.py OUT` runs the simulator with a different, known config (its
`TRUTH`) and writes what the hardware build would have printed: times
extrapolated to 1000 times the repetitions, a 20 us launch overhead and up to
0.4% noise. `fit.py` on that directory, with the committed predictions of the
unmodified config, returns `TRUTH` (exactly for the latencies and the clock,
within 0.1% for the Tensor Core rates). `fixture_synthetic/` is a small committed
copy used by `tests/test_timing_calibration.py`. Every file in it says SYNTHETIC.

### Dry run on the RTX 3080 Ti (Ampere, 2026-11-23)

`./build.sh ampere && ./run_all.sh --ampere`: 28 s, all six programs exit 0, ncu
refused. `dryrun_3080ti/` has the raw outputs and the report. Over the 36
configurations, the spread of per-trial slopes was 0.02% to 5.9% (median 1.7%;
only `cp_async/cold` above 5%), and min and median slopes agree within 1.3%.
`%clock64` gave 1.88 ticks per ns.

These are Ampere numbers, measured with a program compiled by a different
toolkit (CUDA 12.4) than the predictions. They do not calibrate anything, but
they already contradict the model's structure in ways worth knowing before the
rental (see "Problems" below).

## Model parameters and what calibrates them

Every entry of `b200.yaml` has a benchmark. Three are weaker than the rest:

- `l2_bandwidth_per_sm_gb_s` is identifiable only if warm loads are slower than
  stores. The model charges a copy max(TMA engine time, memory time), so a slope
  gives the slower of the two rates; if warm loads run at the engine rate, L2 is
  only known to be at least that fast, and `fit.py` says "LOWER BOUND".
- `clock_ghz` is an effective issue rate for one loop of register instructions.
  Every other clock-denominated parameter is converted with it, so a wrong clock
  does not make them wrong in nanoseconds.
- `hbm_capacity_gb` is what the driver reports as usable.

Not calibrated by any benchmark, and why:

| What | Why |
| --- | --- |
| The number of issue slots (4) and the slot assignment `w % 4` | Hard-coded in `model.py`, not a config entry. `alu_chain/warps=N` measures the shape; there is nowhere to write the result |
| `active_sms` | A launch argument, not a config entry; `warplens run` always uses the SM count. The cold `tma` variants on 1, 8, 37 and all SMs measure how the fetch rate really divides |
| L2 line size (128 B), L2 capacity and eviction | The line size is a constant; capacity and eviction are not modeled |
| The 85% "saturated" threshold | A reporting choice, not a property of the hardware |
| Tensor Core rates for kinds other than f16 and tf32 | No parameter exists. `mma_rate/kind=i8` records the number |
| A latency per MMA shape or kind | One `mma_latency_clk` for all; the per-shape table shows the error |
| Global store cost, `tcgen05.wait` cost, mbarrier cost, launch overhead | No parameter exists; each is measured and reported |
| A `cp.async` rate of its own | It borrows the TMA parameters (TIM-14) |
| Everything in `b300.yaml` | No B300 |
| Multicast and two-CTA MMA charging (TIM-15), `ncu` wavefront offset by block shape (TIM-17), the pipelined GEMM end to end (TIM-18) | No benchmark here: they need cluster launches, a different host program, and a host program for `examples/04_pipelined_gemm` |

Also not covered from the matrix: TIM-09 asks for `.x16`, `.x128` and `16x256b`
transfers; only `32x32b.x1` and `.x8` are here.

## Problems found in the timing model

Reported here; 2 and 3 have since been fixed in the model, the rest have not.

1. **`warplens run` cannot set `active_sms` or the config.** A single CTA is
   charged HBM at 1/148 of the device bandwidth (28 GB/s), so a 32 KiB cold load
   is predicted at 1276 ns against 275 ns warm, and 8 SMs loading cold are
   predicted to move 8 x 28 GB/s. `predict.py --config` swaps the config by
   replacing `warplens.timing.config.CONFIG_DIR`; a `--config` and `--active-sms`
   option on `warplens run` would be the proper way.
2. **TMA stores were always charged to HBM**, at that same 1/active_sms share,
   never to L2. Fixed: a store's destination lines use the same L2 record as a
   load's source. The `tma/store` benchmark writes the same bytes every
   repetition, so its slope is now a warm one: a 32 KiB store is predicted at
   274 ns, where it was 1274 ns.
3. **`cp.async` likewise** had no line addresses, so warm and cold were both
   charged to HBM and predicted the same (122 ns per group). Fixed: `cp_async/warm`
   is now predicted at 106.5 ns per group and `cp_async/cold` at 122 ns.
4. **TMA engine rate and L2 rate are not separately identifiable** from copies,
   because of the max() above (see the lower bound).
5. **One global-load latency.** On the 3080 Ti a dependent load took 44 ns from
   a 16 KiB footprint, 163 ns from 2 MiB and 364 ns from 1 GiB. Stores, which the
   model charges nothing, took 32 to 34 ns per warp store there.
6. **Latency and throughput are one number.** On the 3080 Ti a dependent
   shared-memory load took 14.7 ns (28 clock ticks) but independent ones 4.8 ns;
   each extra bank-conflict wavefront cost 2 ticks in both. The model charges
   one cost per access.
7. **One shared-memory server per SM.** Four warps doing 32-way-conflicting
   loads took 1.42 times as long as one warp on the 3080 Ti; the model says 4.
8. **Four issue slots.** On the 3080 Ti the time per iteration was flat up to 8
   warps, 1.23x at 16 and 2.4x at 32; the model doubles from 4 to 5 warps.
9. **Time is printed in whole nanoseconds** by `warplens run --timing`, and
   elapsed event time is zero, so a program cannot read its simulated time;
   `predict.py` reads the launch records in-process instead.
10. `%clock64` and `ld.volatile` are rejected by the simulator (found while
    writing the kernels).
11. `docs/m4.md` says `tcgen05.wait` has a fixed cost; the code charges
    `tmem_access_clk` to `ld` and `st` only (noted by the matrix's author, TIM-19).

Items 5 to 8 are Ampere observations. They may not hold on Blackwell; the same
configurations run there.

## Files

| | |
| --- | --- |
| `common.h`, `blackwell.h` | harness (arguments, timing, output lines); PTX wrappers and the tensor-map call |
| `*.cu` | one benchmark each: kernel, host program, and a header comment with method and formula |
| `build.sh` | `hw` (sm_100a), `sim` (PTX for the simulator), `ampere` (local GPU) |
| `run_all.sh`, `run_ncu.sh` | the unattended run; the counter run |
| `predict.py`, `predicted/` | simulator predictions |
| `fit.py` | results directory to parameters |
| `make_synthetic.py`, `fixture_synthetic/` | synthetic results for testing `fit.py` |
| `dryrun_3080ti/` | raw outputs and report of the Ampere dry run |
