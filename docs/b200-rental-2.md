# The second B200 rental (2026-11-25): what was confirmed, and two answers

The run that [b200-rental-1.md](b200-rental-1.md) asked for under "What the
second rental has to confirm": the same probe suites, on the same machine,
with the simulator as it was after the first rental's fixes. This file is the
record of what ran, what agreed, and what the two differences changed.

**Measured** is in the archives below; **reading** is an interpretation that
no further experiment has checked.

## The machine and the runs

The instance of the first rental, started again: one NVIDIA B200 on a VM,
Ubuntu 24.04.5, driver 610.57.04 (which supports CUDA 13.3, older than the
pinned toolkit 13.5), 20 CPUs. Commit `312fe4836d`, nothing changed on the
machine. The two installs of the first rental were still on its disk and were
used as they were (`setup.sh` verified them in under two minutes each).

Two runs, as `env/remote/RUNBOOK.md` plans for such a driver, and one set of
extra measurements by hand:

| Run | Compiler | Archive | SHA-256 |
| --- | --- | --- | --- |
| Main | nvcc 13.3.73 (the fallback), predictions made again for it on the machine | `warplens-b200-20261125T012055Z-312fe4836d.tar.gz` (66 MB) | `817eab7e7d57…` |
| Pinned | nvcc 13.5.37, machine code (`--toolkit pinned`, `--no-e2e`) | `warplens-b200-20261125T022247Z-312fe4836d.tar.gz` (23 MB) | `241fb5bd48ff…` |
| Sweeps | nvcc 13.3.73, two probes with one more script each, run by hand | `validation/blackwell/measured/b200/rental-2-sweeps.raw.txt` (every output, one after another) | in the repository |

The archives carry the first rental's timestamps in their names because the
run directories on the machine were reused; the commit at the end tells them
apart. They are kept outside the repository.

The machine was up for about 45 minutes. The provider stopped it once, during
the pinned run's timing stage; after the restart `run.sh` skipped the stages
that had finished and ran that one again.

In the repository:

| Path | What |
| --- | --- |
| `validation/blackwell/measured/b200/compare-rental-2-*.txt` | both comparison tables, as made against the predictions the runs were held to, and the pinned run once more against the predictions after the two changes below |
| `validation/blackwell/measured/b200/tma/swizzle_misaligned.thirtytwo.txt`, `.sixtyfour.txt` (and `.stderr.txt`) | the two runs that ended the kernel where the simulator predicted a result |
| `validation/blackwell/measured/b200/tma/swizzle_offsets.txt` | every offset tried by hand, and what happened |
| `validation/blackwell/measured/b200/mbarrier/expect_tx_small.pieces.fallback.txt`, `.pinned.txt` | the same probe in the two runs: two different outputs |
| `validation/blackwell/measured/b200/mbarrier/expect_tx_sweep.txt` | how often each count completed the phase, over 80 runs |

## What agreed

| | Main (nvcc 13.3) | Pinned (nvcc 13.5) |
| --- | --- | --- |
| identical | 103 | 105 |
| end-to-end, bit-identical | 8 | not run |
| informational (no prediction, or the GPU ran what the simulator refuses) | 16 | 16 |
| the kernel failed where the probe allows a fault | 3 | 3 |
| the GPU faulted where the simulator has no model | 5 | 3 |
| timed out, as predicted | 2 | 2 |
| **differs** | **1** | **1** |

Two of the three kernels that failed are the first answer below: the
comparison files a failure of a probe that may fault under "by design",
whatever was predicted.

- **Every prediction that the first rental's fixes changed is identical**, in
  both runs: the MMA sum (`mma/accumulation_order`, all parts, which settles
  the third assumption of the first write-up), the TF32 conversions, the NaN
  fill of TMA, `tma/oob_complete_tx`, the faults a GPU ends a kernel for, the
  shared-memory limits.
- **The end-to-end programs are bit-identical** to the simulator: the four
  CUTLASS tutorials, the Triton kernels, the TIRx and CuTe DSL scripts, and
  all nine of the book's GEMM kernels (five were within tolerance and four
  outside it in the first rental, before the MMA arithmetic was changed).
- **RT-11** was captured in the first attempt (`RT-11 capture: COMPLETE`),
  and NUM-05 and both RT-10 traces ran.
- The two runs the main run cannot make with nvcc 13.3 (`formats/e8m0_ties`,
  `formats/special_inputs`, which stop at an assert of that release's
  headers) are identical in the pinned run. That is also why `run-blackwell`
  is `failed` in the main run's manifest, as in the first rental.

## The two answers

Both are in rows the first write-up listed as assumptions with a probe run
that asks a GPU.

### A tensor copy needs a shared address aligned to 128 bytes

**Measured.** `tma/swizzle_misaligned.thirtytwo` and `.sixtyfour` ended the
kernel with `cudaErrorMisalignedAddress`, as offset 16 had in the first
rental; the simulator, which refused only what is not a multiple of 32,
predicted that both run. By hand afterwards: offsets 48, 80, 96, 112, 144 and
192 end the kernel the same way, and 128 and 256 run for every swizzle mode
(`measured/b200/tma/swizzle_offsets.txt`).

**Changed.** `TENSOR_COPY_ALIGNMENT` is 128 (`warplens/units/tma.py`). The
simulator now ends the kernel for each of those offsets, with the error the
GPU gives.

### An undersized `expect_tx`: one diagnosis, more than one outcome

**Measured.** `mbarrier/expect_tx_small.pieces` gave different outputs in the
two runs of the same day. Repeated 80 times by hand
(`measured/b200/mbarrier/expect_tx_sweep.txt`), for one TMA load against an
`expect_tx` of fewer bytes:

- a count that is not a multiple of 32 never completed the phase (none of
  the 226 such counts in the sweeps over every multiple of 4, and none in
  the 80 repeats);
- a count that is a multiple of 32 completed it in some runs and not in
  others: for a 256-byte load, 32 in 39 of 40 runs, 128 in 30, 64 in 23, 224
  in 11, 160 in 8, 192 in 5, 96 in 1;
- the full count always completed it, and a count above the load never did.

**Reading.** The copy reports its bytes in pieces whose sizes are multiples
of 32 and vary from run to run; the phase completes when a piece boundary
falls exactly on the count. The first rental's two measurements (128 of 256,
64 of 128) were of counts that complete often, and the first reading, pieces
of a fixed size that divides 64, does not survive: 64 of 256 fails in 17 runs
of 40.

**Changed.** Two things that the simulator keeps apart:

- *The diagnosis is deterministic.* The phase accounts for fewer bytes than
  the copy delivers, so a wait on it cannot establish that the whole copy is
  there. That is a statement about the program, not about a schedule, and
  `mbarrier-tx-mismatch` says it first.
- *The outcome shown is one of those a GPU has.* The simulator has one
  schedule per run, and takes the one in which the count passes through
  every multiple of 32 (`TX_PIECE_BYTES` is 32, where it was 64): the phase
  completes early wherever a B200 was seen to complete it, and the wait
  returns before the rest of the data. That is the dangerous outcome, and the
  finding says that a GPU does this in some runs and in others never
  completes the phase. A count that is not a multiple of 32 is passed over,
  as on the GPU; the unbalanced count is then reported when the kernel ends
  or deadlocks.

Matching how often a GPU takes each outcome is not attempted. The probe run
has no prediction any more (`predict="none"` in the manifest): no single
output can be held against it.

## The timing measurements

Both runs ran the timing suite again: 14 benchmarks, 161 configurations. The
tables are in `validation/blackwell_timing/measured/b200/rental-2/`; the first
rental's are one directory up. Nothing in the simulator or its configs was
changed by what follows.

**The measurements repeat** (measured; pinned compiler, first rental against
second). Of 161 configurations 127 are within 1 % and 147 within 5 %; the
median difference is 0.24 %. By benchmark:

| Benchmark | Configurations | Median difference | Largest |
| --- | --- | --- | --- |
| `mma_rate` | 64 | 0.23 % | 0.66 % |
| `smem_wavefronts` | 20 | 0.22 % | 0.25 % |
| `alu_chain`, `mbar`, `tmem_access`, `gstream` | 16 | 0.2 to 0.4 % | 0.35 % |
| `pipelined_gemm` | 6 | 0.74 % | 1.5 % |
| `cluster_costs` | 10 | 0.55 % | 1.7 % |
| `tma` | 21 | 1.4 % | 6.1 % |
| `cp_async` | 2 | 2.4 % | 2.4 % |
| `clc_latency` | 5 | 1.8 % | 5.2 % |
| `global_load` | 6 | 0.29 % | 5.9 % |

What moved is one thing: latency on the path through L2. Warm TMA loads,
`cp.async`, the load chain with an L2-sized footprint and the requests of
cluster launch control were 2 to 6 % slower in the second run; register
arithmetic, shared memory, tensor memory, mbarriers and every MMA
configuration were within 0.7 %. With the fallback compiler the same
configurations moved further, up to 14 %. **Reading:** the state of the
machine between two boots, not the programs; nothing here says which state.

**The parameters a fit derives repeat with them.** Ten of the fourteen are
within 0.4 % of the first rental's (the SM count and memory size, the issue
rate of one warp, the shared-memory wavefront, tensor-memory access, the MMA
latency and both tensor-core rates, HBM bandwidth, the latency of a
cluster-launch-control request). The four that moved are the ones read from
the L2 path: global access latency +6.0 %, TMA latency +3.6 %, and TMA bytes
per clock and L2 bandwidth per SM about -3 %.

**What the compiler changes is stable.** A TMA store takes 15 to 35 % longer
per repetition when built by nvcc 13.5 than by 13.3 (38.8 ns against 25.1 ns
at 1024 bytes), and each of the four runs gives its compiler's number to
within 0.3 %. That is code around the instruction, measured twice; it is not
noise, and a fit made with one compiler's build carries it.

**The first rental's fit, held against the second rental's measurements**
(`rental-2/first-fit-held-against-it/`). This is the question the first
write-up left for this run: a fit agrees with the measurements it was made
from, and that says little. Predicting every benchmark with
`calibrated.yaml` of the first rental and comparing with what the second
measured:

| | Within 10 % | Within a factor of two | Of |
| --- | --- | --- | --- |
| Shipped config, first rental's measurements | 6 | 53 | 153 |
| Shipped config, second rental's measurements | 5 | 53 | 153 |
| First rental's fit, its own measurements | 46 | 104 | 153 |
| First rental's fit, second rental's measurements | 47 | 105 | 153 |

(The first write-up counted 45 and 104 of 152 for the third line; these were
counted again for this one, with the simulator as it is now.) The fit is as
good out of sample as in: no configuration that it had within 10 % left that
band. And it is no better: where it was wrong it is wrong by the same factor.
With the fit, all 56 `mma_rate` configurations are within a factor of two
and 30 within 10 %; `smem_wavefronts` has none of 20 within 10 %, `mbar` is
off by 3.7 times, a failed cluster-launch-control request on two CTAs by 1.8
times. Those are the places the first write-up named, where one number per
mechanism is not enough.

**What this changes: nothing in v1.** The decision recorded in `PLAN.md`
stands and has more under it: the measurements a fit would be made from are
repeatable to a fraction of a percent for most mechanisms, so the remaining
error is the model's shape and not the data. That is the work listed there
for after v1.

`ncu` profiled all 11 benchmarks in the main run and 10 of 11 in the pinned
run, where it failed on `transpose` as it had in the first rental (not looked
into, in either).

## What was not done

- **The two changes above were not run on hardware again.** For the
  alignment the new predictions are what this rental recorded (the kernel
  ends). Held offline against the pinned run's outputs, the predictions as
  they are now give 107 identical and none that differs
  (`validation/blackwell/measured/b200/compare-rental-2-pinned-after-the-two-changes.txt`).
  For the undersized `expect_tx` there is no prediction to confirm.
- The self-test on the machine was skipped in both runs, as planned.
- The open questions of the first write-up that this run was not asked about
  stand as they were (`PLAN.md`, "Open and recorded, not blocking").

## What the day showed about the runbook

- A home directory that is reused keeps the first run's timestamp in the
  archive name. Harmless, and confusing: the commit suffix is what
  distinguishes the archives.
- The provider stopped the instance again without being asked. The first
  connection after a start can be closed within a minute; wait for the second
  boot before copying anything.
- `git bundle create … main` leaves a clone with no branch checked out
  (`git checkout main` after it); `… HEAD`, as the runbook writes it, does not.
- Copying each archive home as soon as `run.sh` printed it worked as the
  first rental's lesson said it would.
