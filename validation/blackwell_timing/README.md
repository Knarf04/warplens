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
                                 # ncu at most 20 minutes; a benchmark that hangs costs 3 minutes
# -> results_<host>_<time>/ and results_<host>_<time>.tar.gz: bring the archive back
```

Then, anywhere with Python and PyYAML (the simulator is not needed):

```bash
python fit.py results_<host>_<time>          # report.md, derived.yaml, calibrated.yaml, ...
```

`run_all.sh` runs every benchmark in its own process under a timeout, records the
exit status and goes on after a failure or a hang. It saves `env.txt` (driver,
`nvidia-smi`, other processes on the GPU, compiler, checksums of the programs),
runs `run_ncu.sh`, runs `fit.py` if it can, and packs the directory. `--scale 4`
makes every launch four times longer if the spread in `report.md` is too large;
`--trials N` changes the number of trials (default 5).

The order puts what is proven first, so that a run cut short (the 60-minute
stage limit of `env/remote/run.sh`, or Ctrl-C) has lost the unproven part:

1. the seven benchmarks that are valid on Ampere, all of which have run on the
   RTX 3080 Ti (`launch`, `alu_chain`, `smem_wavefronts`, `global_load`,
   `gstream`, `cp_async`, `ncu_banks`);
2. `run_ncu.sh --group ampere`: counters for those programs, the two comparisons
   with the simulator (TIM-16, TIM-17) first;
3. the seven Blackwell-only benchmarks, which have never run on a GPU (`mbar`,
   `tma`, `mma_rate`, `tmem_access`, `clc_latency`, `cluster_costs`,
   `pipelined_gemm`), then `mma_rate_i8`;
4. `run_ncu.sh --group blackwell --continue`: counters for the Blackwell-only
   programs, added to the same `ncu/STATUS.txt`;
5. `fit.py` and the archive. `fit.py` reads the `<bench>.out` files by name, so
   the order does not matter to it.

**The limit of a benchmark is 180 s** (`--timeout SECONDS`; it was 900 s). Where
the number comes from: the slowest benchmark measured takes 13 s
(`smem_wavefronts` on the 3080 Ti; the seven Ampere-valid ones take 1 to 13 s,
29 s together), and the longest estimate for the B200 is 30 s (`mma_rate`, not
measured; table below). 180 s is six times that estimate. With `--scale X` above
1 the default limit is 180 s times X, because the run was asked to take X times
as long. The limit matters because the Blackwell-only kernels wait on an
mbarrier in a loop without a bound (`mbarrier_wait` in `blackwell.h`): a wrong
descriptor or tensor map is a hang, not an error. Worst cases, by arithmetic and
not measured: all eight Blackwell-only runs hang, 8 x 190 s = 25 minutes, plus
the ncu budget of 20 minutes, is inside the 60-minute stage limit; all fifteen
runs hang, 48 minutes plus ncu, is not. A benchmark that reaches its limit is
`TIMED OUT after 180s` in `status.txt`, with the number of BENCH lines it had
printed (`lines=`); those lines are in its `.out` file, because the programs
write stdout line by line (`setvbuf` in `common.h`; the bytes are unchanged, and
the predictions regenerated from the rebuilt `bin/sim` are identical to the
committed ones).

**Stopping.** `timeout` puts its command in a process group of its own, so a
signal to `run_all.sh` or to its group does not reach the benchmark. Until the
pre-rental review that left the benchmark (or ncu and the program under it)
running on the GPU after the stage limit or Ctrl-C, next to the rerun and
writing the same output file. Now every command runs in the background and
`run_all.sh` and `run_ncu.sh` trap SIGTERM, SIGINT, SIGHUP and exit: they send
SIGTERM to the command's process group, SIGKILL after 8 s (`run_ncu.sh`: 3 s),
write what was stopped and what was not run to `status.txt`, and exit with 143,
130 or 129 without making an archive. Measured on the 3080 Ti: SIGTERM to
`run_all.sh --ampere` during `smem_wavefronts` ended it at once with status 143
and no benchmark process left, the 52 lines printed until then in the output
file. Not tried: a benchmark that hangs inside the driver; whether SIGTERM ends
it or the SIGKILL after 8 s is needed is unknown, and a process that ignores
SIGTERM was only a stand-in (`tests/test_timing_run_all.py`).

**Counters.** `<out>/ncu/STATUS.txt` exists at the end of every run, and its
first line starts with exactly one of

| First line starts with | Meaning |
| --- | --- |
| `ncu ok:` | every profile that was attempted produced a counter table |
| `ncu partial:` | some did; the line names each one that did not, with the reason (timed out, ncu's exit status, budget used up, stopped) |
| `ncu failed:` | ncu was run and no counter table was read (the probe gave none, or no profile did) |
| `ncu not run:` | nothing was attempted: ncu not installed, no permission, the probe program could not launch, nothing built, or `--no-ncu` (`not requested`) |

The lines after the first are for people. `status.txt` repeats the first line
(after step 2 it also has a line `ncu, first part (...)` with the state at that
point). With the outer runner that file is `out/timing/ncu/STATUS.txt` in the
run directory.

**Exit status**: the number of benchmarks that failed, plus 1 if
`WARPLENS_NCU=ok` is in the environment (`env/remote/run.sh` sets it when
preflight read a counter with ncu) and the verdict is nevertheless `ncu
failed:` or `ncu not run:`. Then counters were known to be readable and none
was read, which is a failure of the run and not a property of the machine.
`ncu partial:` adds nothing (read `STATUS.txt`), and neither does any verdict
under `--no-ncu` or without `WARPLENS_NCU=ok`. Either way every benchmark and
`fit.py` have run before the script exits. `run_ncu.sh` itself exits 0 whatever
its verdict.

Build with the same compiler as the predictions if possible (nvcc 13.5 here). The
loops are written so that the compiler has little freedom (the timed accesses are
inline PTX), but `clock_ghz` is derived from an instruction count taken from the
PTX compiled here. The safest route is to copy `bin/hw` from this machine: the
programs link the CUDA runtime statically.

Which compiler built a run is recorded and compared, not left to memory:
`build.sh` writes `bin/<target>/toolkit.txt`, `predict.py` writes the nvcc version
into `predicted/meta.json`, and `run_all.sh` writes `TOOLKIT.txt` (driver, the
CUDA release it supports, both nvcc versions, `predictions_comparable`). When
they differ, `TOOLKIT.txt` and the first line of `status.txt` say `BUILT WITH
FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE`. On a machine whose driver is
older than toolkit 13.5 that is what `env/remote/setup.sh` leads to by default;
`env/remote/RUNBOOK.md` ("Driver older than the toolkit") has the alternative.

How much the compiler matters, measured: built by nvcc 13.0.88 (what
`env/remote/setup.sh` falls back to on the development machine's driver), all
14 benchmarks still run in the simulator, and 13 of them have other instruction
counts and predicted times than the committed predictions from nvcc 13.5.37
(only `ncu_banks` is the same). So after a run with a fallback toolkit, predict
again with that toolkit before fitting: `build.sh sim`, then `predict.py --out
DIR` and `fit.py RESULTS --predicted DIR`, in an env made by `env/remote/setup.sh
--toolkit X.Y` (the runbook has the commands). Every prediction here, the committed
ones included, is regenerated by `validation/regenerate_predictions.sh`.

A benchmark whose kernels the driver could not load at all (for example machine
code for another GPU) prints `BENCH_ERROR ... class=launch-failed`, exits with
status 4, and is `LAUNCH FAILED` in `status.txt`; one that ran and failed is
`class=failed`, status 1, `FAILED`. The harness checks the error of every launch
(`cudaGetLastError` after the launch and after the synchronize).

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
| `clc_latency` | `clc_latency_ns` | ns per request (arm, `try_cancel`, wait). `clc_latency_ns` = round trip of `fail/plain/ctas=1` (one cluster of one CTA, every request fails) - the instruction time the simulator counts for the same loop. The other four are reported with the latency each implies: `fail/plain/ctas=2` and `fail/multicast/ctas=2` from their slopes, `succeed/*` (a grid of `reps` + 1 clusters) from `clk` / (`check` + 1), the first cluster's own ticks per request, because many clusters start at once there and the slope is not a round trip. See "Cluster launch control latency" below | TIM-20 |
| `cluster_costs` | nothing (two charging rules) | `load/multicast/ctas=N` against `load/separate/ctas=N` (N = 1, 2, 4), ns per repetition of a 16 KiB warm load to every CTA of a cluster: the model predicts the same. `mma/pair` (one `cta_group::2` MMA, M = 256) against `mma/single` (two CTAs, M = 128 each), slope against the batch size: the model predicts the pair takes twice as long per MMA, because only the issuing SM's Tensor Core is charged | TIM-15 |
| `pipelined_gemm` | nothing; end-to-end check | ns per K tile of `examples/04_pipelined_gemm/gemm.cu` (included unchanged) with 1, 2 and 4 stages, on 8 CTAs and on one per SM; the result is checked. The ordering (2 stages no slower than 1, 4 no slower than 2) is what a calibrated model must keep | TIM-18 |
| `ncu_banks` | nothing; input for the bank model's counters | the transpose of `examples/01_smem_transpose` with blocks of 32 x 32, 32 x 8 and 16 x 16 threads, grid 64, one launch per kernel, to be profiled under `ncu` and compared by `ncu_banks_compare.py` (below). Its launch times are reported and mean little | TIM-16, TIM-17 |

Valid on Ampere, and run on the 3080 Ti: `launch`, `alu_chain`, `smem_wavefronts`,
`global_load`, `gstream`, `cp_async`, `ncu_banks` (the last one run by hand, not part of the
committed dry run). Blackwell only (simulator and `sm_100a`
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
| `clc_latency` | 5 | 5 s (guess: the request latency is what it measures; the `succeed` grids have 10^4 and 2 x 10^4 clusters) | |
| `cluster_costs` | 10 | 5 s | |
| `pipelined_gemm` | 6 | 10 s (up to 2.5 GiB allocated and filled per CTA count) | |
| `ncu_banks` | 9 (one launch each per trial) | 1 s | 1 s |
| `run_ncu.sh` | 11 profiled programs | 5 to 15 min if counters are readable (guess), never more than its budget of 20 min; seconds if not | refused (no permission) |

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

`run_ncu.sh [options] [BIN] [OUT]` first profiles one kernel with one counter
(the probe). If `ncu` is missing, lacks permission (`ERR_NVGPUCTRPERM`; it needs
root or a driver set up for non-admin profiling) or prints no counter table, it
says why in the first line of `OUT/STATUS.txt` (the four beginnings are in the
table above) and exits 0; `run_all.sh` continues. It never uses sudo. Otherwise
it saves raw `ncu --csv` output per benchmark (instructions executed, cycles,
shared-memory wavefronts and bank conflicts, cache hit rates, DRAM bytes), run
with one trial and 1% of the repetitions because every launch is profiled.
`OUT/profiles.tsv` has one row per program (name, state, detail), and
`STATUS.txt` is rewritten after every profile, so it is true also for a run
that was killed.

The probe. `launch --trials 1 --only ctas=1` starts 42 kernels (the spin-up on
every SM of `common.h`, a warm-up launch and 40 empty kernels), and the probe
used to profile every one of them inside 60 s. It now passes `--launch-count 1`, so ncu profiles the first kernel
and lets the rest run unprofiled, and it has 300 s (`--probe-timeout`,
`NCU_PROBE_TIMEOUT`), the limit `env/remote/preflight.sh` gives its single
kernel. What is known about that option: the machine's ncu 2024.1.1 and
`ncu --help` accept `--launch-count 1` (as a normal user the run then stops at
the permission error, as before). **It has not been run where counters can be
read.** So if the probe ends without a table for a reason other than permission
or its time limit, it is asked once more without the option, the form that read
counters as root on the 3080 Ti, and `STATUS.txt` says so if only that form
worked. A probe that reaches its limit is `ncu failed:` and is not repeated.

Order and limits (changed in the pre-rental review; before it the two proven
comparisons ran last and a profile could take three times 20 minutes):

- **Order**: `ncu_banks` and the transpose (`validation/ampere/transpose_hw`,
  TIM-16, TIM-17) with their comparisons against the simulator, then
  `alu_chain`, `smem_wavefronts`, `global_load`, `cp_async`: the six programs
  that have been profiled on the 3080 Ti. Then `mbar`, `tma`, `mma_rate`,
  `tmem_access`, `clc_latency`, which have never been profiled. `--group
  ampere` and `--group blackwell` select one half; `--continue` adds to an
  earlier call for the same directory without a second probe.
- **Limit of one ncu run**: 300 s (`--timeout`, `NCU_PROFILE_TIMEOUT`;
  `run_all.sh --ncu-timeout`). A choice, not a measurement: how long a profile
  takes was not recorded in the runs as root on the 3080 Ti, and the estimate
  of 5 to 15 minutes for all eleven on a B200 is a guess. 300 s is the whole of
  the lower estimate for a single program.
- **No retry after a timeout.** A run that reaches its limit (`timeout` returns
  124 or 137) is recorded as timed out and the script goes to the next program.
  Only a run that ncu ended itself without a table is repeated: without the
  names the device does not list (`ncu --query-metrics`), and only then with
  `smsp__inst_executed.sum` alone; `STATUS.txt` names every benchmark whose
  list was reduced.
- **Budget**: 1200 s for everything the script does for one output directory,
  both calls of `run_all.sh` together (`--budget`, `NCU_BUDGET`; `run_all.sh
  --ncu-budget`). When it is used up, the running profile is cut at the budget,
  the remaining programs are not started, and `STATUS.txt` names them (`ncu
  partial:`). The script ends within the budget plus the 10 s of `timeout
  --kill-after`. A comparison with the simulator (about 9 s each, measured)
  counts against the budget and has 120 s.

On this machine the real ncu stops at the permission check, with the installed
2024.1.1.0 and with the pinned 2026.4.0.0 alike, so **no counter has been read
through this script**. What was done instead:

- `./run_ncu.sh --dry-run [BIN] [OUT]` prints every ncu command line it would
  run and whether its program is built; nothing is executed or written.
- `NCU_BIN=<ncu> ./run_ncu.sh --check-metrics gb100` asks ncu which of the
  counter names it lists for the B200's chip. It needs neither a GPU nor
  permission. With the pinned ncu 2026.4: 18 of the 19 names are listed; the one
  that is not, `l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal`, is not listed
  for the 3080 Ti's chip either, where ncu accepted it and reported `n/a`.
- `tests/test_ncu_scripts.py` runs the script from end to end against a stand-in
  for ncu that replays recorded output (`fixture_ncu/`, whose README says what is
  recorded and what is invented): the permission error, a full run, a rejected
  counter list and its retry, and `compare_ncu.py` on the table recorded from
  the 3080 Ti; and, with a stand-in that hangs, fails or is slow: no second run
  after a timeout, the budget, the four first lines of `STATUS.txt`, the two
  calls of `run_all.sh`, and that SIGTERM or SIGINT to the script leaves neither
  ncu nor the profiled program. `tests/test_timing_run_all.py` does the same for
  `run_all.sh` with stand-in benchmarks: the order, the three endings of a
  benchmark, stopping, and the exit status with `WARPLENS_NCU=ok`.

The two runs below were made with the script as it was before the pre-rental
review: the same counter lists and programs, in the old order, with the old
probe. **The script as it is now has not been run where counters can be read**
(no root in the session that changed it); as a normal user on the 3080 Ti it
ends in a second with `ncu not run: no permission ...`, inside `run_all.sh
--ampere` (31 s, seven programs, `fit.py` ok), and with the `sm_100a` programs,
none of which loads there, with `ncu not run: bin/hw/launch could not launch its
kernels on this GPU (cudaErrorNoKernelImageForDevice)`. Seen with the real ncu
2024.1.1 as that user: ncu, its `TreeLauncherSubreaper` and the profiled program
are all in the process group of `timeout`, which is the group the scripts
signal when they are stopped.

Run for real on 2026-11-24, as root on the RTX 3080 Ti (driver 580.173.02, the machine's
ncu 2024.1.1, not the pinned one): `sudo env NCU_BIN=/usr/local/cuda/bin/ncu ./run_ncu.sh
bin/ampere OUT` read counters and profiled all six programs built for Ampere and the
transpose, none with a reduced counter list; the five Blackwell programs were reported as
not built. `ncu_banks_compare.py` on those counters: instruction and conflict counters
equal the model's for all nine kernels, and the wavefront surplus is 4160 for the 32 x 32
kernels and 1088 for 32 x 8 and 16 x 16, that is 65 and 17 per CTA: one per warp and one
per CTA more. `../ampere/compare_ncu.py`: all counters match. Under `sudo` the script's
`python3` is the system's, without NumPy, so both comparisons failed inside the run and
were made again as the user; `run.sh` runs the suite inside the env, where they have it.

The same again that day with the pinned ncu 2026.4.0 (conda-forge's `nsight-compute` in an
env of its own, as `NCU_BIN`) and `PYTHON` set to the project env's: counters read on this
driver, the same six programs and the transpose, as many rows per program as with 2024.1,
no reduced counter list, both comparisons made inside the run and with the same result.
`l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal` is `n/a` on this GPU with both.

Still untested: what ncu prints for a rejected counter name, and whether the B200's ncu
knows each counter at run time.

`ncu_banks` is the program for TIM-17: nine kernels, the three layouts in three block
shapes, launched once each with `--trials 1`. `ncu_banks_compare.py NCU.csv` simulates the
same nine kernels (from the committed `ncu_banks.ptx`), prints `ncu`'s counters next to the
model's, and tabulates the wavefront surplus per kernel against its CTAs, warps and
shared-memory instructions; without an argument it prints the model's counters alone, and
`--self-check` confirms that the 32 x 32 kernels count exactly like the example's. The
model's totals are in the comment at the top of `ncu_banks.cu`. The program ran on the
3080 Ti (results checked); it has not been profiled, for the same lack of permission.

## Predictions and fitting

```bash
./build.sh sim
PYTHONPATH=<repo> python predict.py                  # rewrites predicted/ (about 3 minutes)
PYTHONPATH=<repo> python predict.py --out again --config results_x/fit/calibrated.yaml
```

`predicted/<bench>.out` holds each launch's BENCH line with the simulator's
numbers appended (`sim_ns`, `sim_instr`, `sim_bound`, and the placement the
launch was timed with: `sim_active_sms`, `sim_ctas_per_sm`, `sim_resident_ctas`);
`predictions.md` is the table; `meta.json` has the config used, the active-SM
assumption of every configuration and what was not simulated. `--model B300`
selects the matching B300 config unless `--config` explicitly overrides it; the child
process and recorded metadata use that same selection. Prediction subprocesses are
bounded and cleaned up on SIGINT/SIGTERM, including detached native children.
The second command re-predicts with a calibrated config; `fit.py --predicted
again --config .../calibrated.yaml` then shows ratios near 1 where the model can
reproduce the measurements, and the remaining ratios are what no parameter fixes.

`fit.py` writes into `<results>/fit/`:

| File | Content |
| --- | --- |
| `report.md` | (a) measured values, (b) derived parameters with formula and inputs, other measured quantities, (c) measured against predicted with the ratio and the SMs the prediction assumed active. "Problems in the results" lists failed benchmarks and where the run contradicts that assumption |
| `measured.csv`, `comparison.csv` | (a) and (c) as tables |
| `derived.yaml` | the parameters measured, in the schema of `warplens/configs/*.yaml` |
| `calibrated.yaml` | `b200.yaml` with those entries replaced; `GpuConfig.load(path)` reads it, or copy it over `warplens/configs/b200.yaml` |

Results that did not come from compute capability 10.x, or that are synthetic, are
labelled in every file, and their `source` starts with `assumption` so the
simulator's reports never show them as measured.

### Active SMs

The model simulates each CTA alone on one SM and gives it 1/`active_sms` of the
HBM bandwidth. `warplens run --active-sms N` sets one number for a whole program
(default: every SM), and the benchmarks launch several shapes each, so
`predict.py` sets it per launch, from the launch's shape
(`warplens/timing/placement.py`, applied by `install_placement` in `predict.py`):

- CTAs per SM = the smallest of 228 KB / shared memory per CTA (static plus
  dynamic), 2048 threads / threads per CTA (whole warps), and 32 CTAs;
- CTAs resident at once = min(grid, SM count x CTAs per SM), a whole number of
  clusters;
- active SMs = min(SM count, resident CTAs): CTAs are assumed to be spread, no SM
  taking a second one while another has none, and a cluster's CTAs to be on
  different SMs.

With the spread rule the occupancy limit decides how many CTAs are resident, not
how many SMs are active, except that only whole clusters start. It is recorded
because the alternative rule (fill an SM before the next: `packed_sms`) depends
on it.

On a B200 (148 SMs), per benchmark:

| Benchmark | Launch (hardware build) | CTAs per SM (limit) | Active SMs |
| --- | --- | --- | --- |
| `launch` | 1 CTA; one CTA per SM (`allsm`) | 32 | 1; 148 |
| `alu_chain` | 1 CTA of 1 to 32 warps | 32 down to 2 (threads) | 1 |
| `smem_wavefronts`, `global_load`, `cp_async`, `mbar`, `tmem_access` | 1 CTA of one warp (four in two `smem_wavefronts` configurations) | 32 (16 with four warps) | 1 |
| `gstream` | 2^19 and 2^20 CTAs of 256 threads (8 and 16 with `--sim`) | 8 (threads) | 148 |
| `tma` | 1 CTA; 8, 37 and 148 CTAs for the cold 16 KiB loads | 7 (shared memory) | 1; 8, 37, 148 |
| `mma_rate` | 1 CTA; one per SM (`allsm`) | 7 (shared memory) | 1; 148 |
| `clc_latency/fail/*` | one cluster of 1 or 2 CTAs | 32 | 1; 2 |
| `clc_latency/succeed/*` | 10001 and 20001 clusters of 1 or 2 CTAs, 100 KB each (17 and 33 clusters with `--sim`) | 2 (shared memory): 296 CTAs resident | 148 |
| `cluster_costs/load/*` | one cluster of 1, 2 or 4 CTAs | 14 (shared memory) | 1, 2, 4 |
| `cluster_costs/mma/*` | 2 CTAs, as one cluster or as two | 7 (shared memory) | 2 |
| `pipelined_gemm` | 8 CTAs; one per SM | 2 (shared memory, 96 KB each) | 8; 148 |
| `ncu_banks` | 64 CTAs of 1024, 256 and 256 threads | 2, 8, 8 (threads) | 64 |

The exact numbers of every configuration are in `predicted/predictions.md`
(columns "SMs active (assumed)" and "CTAs per SM (limit)"), on every line of
`predicted/<bench>.out`, and in `predicted/meta.json` under
`benchmarks.<bench>.active_sms.<id>` with the launch shape, the limit that binds
and a one-line derivation; `meta.json` `placement` has the rule and the per-SM
limits. `tests/test_timing_placement.py` holds the table above against the
committed predictions.

Where the hardware build launches a larger grid than the simulator build
(`gstream`, `clc_latency/succeed/*`), the assumption is made for the hardware
grid (`HARDWARE_CLUSTERS` in `predict.py`), since that is the launch the
prediction is compared with; `meta.json` has both grids.

What it changes in the predictions: only what draws on HBM, since the Tensor
Core rate is per SM whatever the launch. A cold 16 KiB TMA load is now predicted
at 190 ns on one SM and on 8 (HBM offers each SM more than its TMA engine moves),
251 ns on 37 and 690 ns on 148, where it was 690 ns for all four; the pipelined
GEMM on 8 CTAs at 331 / 166 / 128 ns per K tile for 1 / 2 / 4 stages, where it
was 1081 / 878 / 878 as on the whole device; `cp_async/cold` (one CTA) at 106.5 ns
per group, the same as warm, where it was 122 ns. The one-CTA cold `tma` loads
are likewise predicted like warm ones now. Twelve of the 153 configurations
changed; no other prediction did. `predict.py --placement all-sms` makes the old
predictions, to hold both against a measurement.

Against a run, `fit.py`:

- shows "SMs assumed" for every configuration in section (c) of `report.md` and
  as `predicted_active_sms` in `comparison.csv`;
- reports as problems: a device with another SM count than the predictions'
  config; per-SM limits in the DEVICE line (`smem_per_sm_bytes`,
  `max_threads_per_sm`, `max_blocks_per_sm`, from `cudaGetDeviceProperties`) that
  differ from the ones the placement used; a configuration whose `nctas` on
  hardware implies another number of active SMs than its prediction; predictions
  made without the assumption.

Nothing in a run measures the number of busy SMs directly: a kernel cannot read
which SM it is on. What a B200 run does show is the consequence. Under the
spread rule, cold loads on 8 CTAs move 8 times what one CTA moves; if the
scheduler packed SMs instead (`packed_sms` in `meta.json`: 2 SMs for those 8
CTAs, 4 for the GEMM's 8) and CTAs on one SM share its TMA engine, they move
less. "cold TMA fetch rate, ctas=1 / 8 / 37 / allsm" in `report.md` are the
numbers to compare. Not modeled, and stated in `placement.py`: registers as an
occupancy limit, the shared memory the driver keeps for itself, GPC boundaries
for clusters, and anything else running on the GPU.

The limits of the timing model itself are unchanged: resident CTAs are still
each simulated alone on an SM, also where the placement says an SM holds
several (`clc_latency/succeed/*`, `gstream`), and a grid larger than the device
holds is not run in waves.

### Cluster launch control latency

`docs/clc.md` and `warplens/timing/model.py` (kind `"clc"`) model a
`clusterlaunchcontrol.try_cancel` as a request that uses none of the SM's
resources and whose response lands `clc_latency_ns` after it is issued: one
number for a request that succeeds or fails, with or without multicast. `fit.py`
fits that one number:

    clc_latency_ns = measured round trip - (predicted round trip - config clc_latency_ns) x f_config / f

from `clc_latency/fail/plain/ctas=1`: the slope of launch time against requests,
minus the time of the loop's other instructions (arm the mbarrier, issue, wait),
which the simulator counted for the same loop and which is rescaled to the fitted
clock `f`. This is the rule `tma_latency_ns` follows. Like that parameter it
includes whatever the mbarrier wake-up costs on hardware beyond its instructions.

- If `fail/plain/ctas=1` is missing or has no positive slope, `fail/plain/ctas=2`
  and then `fail/multicast/ctas=2` are used, and the parameter's note says so
  (their loop also has a cluster barrier per request).
- Every form measured is listed under "Other measured quantities" as "CLC
  request, ..." with its round trip and the latency it implies. For
  `succeed/*` the round trip is `clk` / (`check` + 1) of the first cluster,
  converted with the `%clock64` ticks per ns that `alu_chain` measures; it is
  reported and never used for the parameter, because it rests on that conversion.
- If another form implies a latency more than 20% from the fitted one, the note
  of `clc_latency_ns` says "NOT ONE NUMBER" and names it: the model would then
  need a latency per form, which it does not have.
- If `clc_latency` is missing from the results, the parameter is left out of
  `derived.yaml`, `calibrated.yaml` keeps the config's 100 ns assumption, and
  `report.md` lists it under "Not derived". The same holds when the predictions
  have no `clc_latency` configuration or were made with a config without the
  entry; then the measured round trip is still reported. A round trip shorter
  than the counted instructions gives 0, the model's lower limit, with a note.

What only a B200 can settle: the value; whether failing and successful requests
take the same time, and multicast the same as plain; and whether a successful
request's latency depends on how many clusters are pending, which the benchmark
does not vary.

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
unmodified config, returns `TRUTH` (exactly for the latencies, `clc_latency_ns`
among them, and the clock, within 0.1% for the Tensor Core rates).
`fixture_synthetic/` is a small committed
copy used by `tests/test_timing_calibration.py`. Every file in it says SYNTHETIC.

### Dry run on the RTX 3080 Ti (Ampere, 2026-11-23)

`./build.sh ampere && ./run_all.sh --ampere`: 28 s, all six programs exit 0, ncu
refused. `dryrun_3080ti/` has the raw outputs and the report (its `status.txt`
is in the format of that day: no `ncu_banks`, and the ncu line begins `ncu: ncu
not run:`). Again on 2026-11-24 with the runner as it is now and `ncu_banks`
among the programs: 31 s, seven programs exit 0, `ncu not run: no permission`;
not committed. Over the 36
configurations, the spread of per-trial slopes was 0.02% to 5.9% (median 1.7%;
only `cp_async/cold` above 5%), and min and median slopes agree within 1.3%.
`%clock64` gave 1.88 ticks per ns.

These are Ampere numbers, measured with a program compiled by a different
toolkit (CUDA 12.4) than the predictions. They do not calibrate anything, but
they already contradict the model's structure in ways worth knowing before the
rental (see "Problems" below).

## Model parameters and what calibrates them

Every entry of `b200.yaml` has a benchmark. Four are weaker than the rest:

- `clc_latency_ns` is one number for five forms of the request, fitted from the
  one that is cleanest to measure (a request that fails, alone in its launch).

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
| `active_sms` | A launch argument, not a config entry. `predict.py` derives it per launch from the launch's shape ("Active SMs" above); it is an assumption about the scheduler, not a fitted number. The cold `tma` variants on 1, 8, 37 and all SMs measure how the fetch rate really divides |
| L2 line size (128 B), L2 capacity and eviction | The line size is a constant; capacity and eviction are not modeled |
| The 85% "saturated" threshold | A reporting choice, not a property of the hardware |
| Tensor Core rates for kinds other than f16 and tf32 | No parameter exists. `mma_rate/kind=i8` records the number |
| A latency per MMA shape or kind | One `mma_latency_clk` for all; the per-shape table shows the error |
| Global store cost, `tcgen05.wait` cost, mbarrier cost, launch overhead | No parameter exists; each is measured and reported |
| A `cp.async` rate of its own | It borrows the TMA parameters (TIM-14) |
| Everything in `b300.yaml` | No B300 |
| Multicast and two-CTA MMA charging (TIM-15), the pipelined GEMM end to end (TIM-18) | Measured by `cluster_costs` and `pipelined_gemm` and reported; `fit.py` derives no parameter from them, because the model has none for them. Their rows are in `measured.csv` and `comparison.csv` like every configuration's |
| A latency per form of a cluster-launch-control request (TIM-20) | One `clc_latency_ns` for all, fitted from requests that fail; `fit.py` reports what each of the five forms implies |
| `ncu` wavefront offset by block shape (TIM-17) | `ncu_banks` and `ncu_banks_compare.py`; needs `ncu` with permission to read counters |

Also not covered from the matrix: TIM-09 asks for `.x16`, `.x128` and `16x256b`
transfers; only `32x32b.x1` and `.x8` are here.

## Problems found in the timing model

Reported here; 1, 2, 3 and 13 have since been fixed, the rest have not.

1. **`warplens run` could not set `active_sms` or the config.** A single CTA was
   charged HBM at 1/148 of the device bandwidth (28 GB/s), so a 32 KiB cold load
   was predicted at 1276 ns against 275 ns warm, and 8 SMs loading cold were
   predicted to move 8 x 28 GB/s. `warplens run` has `--timing-config` and
   `--active-sms` now, but one number per program; `predict.py` sets the active
   SMs per launch ("Active SMs" above), and a cold load on one SM or on 8 is
   predicted at the TMA engine's rate. `predict.py --config` still swaps the
   config by replacing `warplens.timing.config.CONFIG_DIR`.
2. **TMA stores were always charged to HBM**, at that same 1/active_sms share,
   never to L2. Fixed: a store's destination lines use the same L2 record as a
   load's source. The `tma/store` benchmark writes the same bytes every
   repetition, so its slope is now a warm one: a 32 KiB store is predicted at
   274 ns, where it was 1274 ns.
3. **`cp.async` likewise** had no line addresses, so warm and cold were both
   charged to HBM and predicted the same (122 ns per group). Fixed: `cp_async/warm`
   is now predicted at 106.5 ns per group, and `cp_async/cold` was at 122 ns while
   every SM was assumed active. With the one active SM of its launch, the cold
   source no longer limits it, and it is predicted at 106.5 ns too.
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
12. `atom` (what `atomicAdd` compiles to: `atom.relaxed.gpu.global.add.u32`) was rejected
    by the simulator when this was written, so `clc_latency` does not sum over the clusters
    that start; it reports the first cluster's ticks and request count instead. The
    simulator runs `atom` now (branch `work/atom`); the kernel was left as it is.
13. The simulator's `pipelined_gemm/*/ctas=8` rows were predicted as if all 148 SMs
    shared HBM. Fixed in `predict.py`: they are predicted with 8 active SMs (331 /
    166 / 128 ns per K tile), which is the "8 SMs share HBM" case of `docs/m4.md`
    for the code nvcc 13.5 makes of the benchmark.

Items 5 to 8 are Ampere observations. They may not hold on Blackwell; the same
configurations run there.

## Files

| | |
| --- | --- |
| `common.h`, `blackwell.h` | harness (arguments, timing, output lines); PTX wrappers and the tensor-map call |
| `*.cu` | one benchmark each: kernel, host program, and a header comment with method and formula |
| `build.sh` | `hw` (sm_100a), `sim` (PTX for the simulator), `ampere` (local GPU) |
| `run_all.sh`, `run_ncu.sh` | the unattended run; the counter run |
| `fixture_ncu/` | recorded ncu output and a stand-in for ncu, for `tests/test_ncu_scripts.py` |
| `ncu_banks_compare.py`, `ncu_banks.ptx` | the simulator's bank counters for `ncu_banks`, next to `ncu`'s; the PTX it simulates |
| `predict.py`, `predicted/` | simulator predictions, each launch timed with the active SMs of its shape (`warplens/timing/placement.py`) |
| `fit.py` | results directory to parameters |
| `make_synthetic.py`, `fixture_synthetic/` | synthetic results for testing `fit.py` |
| `dryrun_3080ti/` | raw outputs and report of the Ampere dry run |
