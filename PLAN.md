# warplens: what is left

**Understand a Blackwell GPU by watching it run, on your CPU.**

This file is the release checklist for v1.0 and the roadmap after it. What
warplens is and does is in the manual, [docs/guide.md](docs/guide.md); how it
works is in [docs/architecture.md](docs/architecture.md).

The plan this project was built from (goals, the survey of related work, the
architecture as first sketched, milestones M0 to M6, and the path to the
first run on a rented B200, step by step) is kept as it stood on 2026-11-25
in [docs/history/plan.md](docs/history/plan.md). Section and step numbers
that other files cite ("section 3.5", "step A3", "section 11") are that
file's.

## Where the project stands (2026-11-25)

- Milestones M0 to M6 are built: the PTX executor, shared memory, TMA and
  mbarriers, `tcgen05` and tensor memory, the timing model, clusters, and the
  front ends (`warplens run` for compiled programs, Triton, TIRx, the CuTe
  DSL). The write-ups of each are under [docs/history/](docs/history/README.md).
- The probe suites ran once on a rented B200, on 2026-11-25
  ([docs/b200-rental-1.md](docs/b200-rental-1.md)). Of 126 probe runs 88 were
  identical to the simulator's prediction; the differences were taken into
  the simulator, which now reproduces the recorded outputs. Those changes
  have not been run on hardware again.
- The timing model's shipped configs hold the values assumed before that
  run, and v1 ships them: a fit of the run is recorded and can be chosen,
  and is not the default (decided below).

The guide's section 2 has the evidence area by area.

## The v1 scope

Frozen on 2026-11-24, amended once by the owner on 2026-11-25
([docs/history/plan.md](docs/history/plan.md), section 3.5, has the decision
and its reasons). The surface is what [docs/coverage.md](docs/coverage.md)
lists form by form. Nothing is added before v1.0: a result from hardware can
change how an instruction in scope behaves, not which instructions are in
it. Anything outside is a hard error that names the instruction. Three
points of the freeze that other documents rely on:

- the matrix base offset of a shared-memory descriptor must be 0;
- `mbarrier.try_wait` blocks and never returns false;
- `tcgen05.cp` is refused by name.

The amendment added `tcgen05.mma` with `.collector::a::discard` and
`cvt.rn.f16x2.f32`, for the kernel of CUTLASS's `fp16_gemm_0.py`.

Non-goals for v1: cycle accuracy; multi-GPU, NVLink, NVSHMEM, NCCL; SASS
input; graphics, texture units, consumer Blackwell (`sm_120`).

The project has two version numbers and no others: `0.0.1`, a placeholder to
reserve the PyPI name, and `1.0`, the public release
([RELEASING.md](RELEASING.md)).

## Still owed for v1.0

1. **A second rental.** 4 GPU-hours are in reserve (the budget and the
   triggers are in `env/remote/RUNBOOK.md`, section 5). The fixes made after
   the first run changed predictions of bit-exact and tolerance rows, which
   hardware has to confirm, and three constants in those fixes are
   assumptions that new probe runs measure: the piece size in which a TMA
   copy reports its bytes, the alignment a tensor copy needs, and three rules
   of the MMA sum. What to look for is in
   [docs/b200-rental-1.md](docs/b200-rental-1.md), "What the second rental
   has to confirm".
2. **Reserve the PyPI name** with the `0.0.1` placeholder.
3. What [RELEASING.md](RELEASING.md) lists under "Before the release".

Decided by the owner on 2026-11-25:

- **The timing config stays as it is for v1.** `b200.yaml` and `b300.yaml`
  ship the assumed values and say so; the fit of the first run
  (`validation/blackwell_timing/measured/b200/calibrated.yaml`) is recorded
  and usable with `--timing-config`, and is not the default. No prediction
  changes, so the second rental's timing run is compared with the same
  predictions as the first. Giving the model what the fit says it lacks is
  after v1 (below; [docs/b200-rental-1.md](docs/b200-rental-1.md), "The
  timing config is a decision, not a fix").
- **The RT-11 capture stays out of the repository.** The kernel of
  `fp16_gemm_0.py` replays byte for byte against the B200, given the capture
  (`tests/validation/golden_replay/test_rt11_replay.py` with
  `WARPLENS_RT11_CAPTURE` naming it in an unpacked rental archive); without
  the variable the test is left out of the run. The capture is 42 MB, most of
  it the modules of other kernels in the process. The tutorial as a script
  under `warplens run` is not done: it holds its arrays in PyTorch, which
  needs a GPU.

Open and recorded, not blocking:

- What separates the repeated `tcgen05.dealloc` that a B200 ended a kernel
  for from the ones in the book's kernels, which ran (TM-11).
- NUM-05: ptxas 13.3 and 13.5 have no UE4M3 conversion, so that format has
  nothing on a GPU to be compared with.
- The self-test on the rented machine under the fallback toolkit: 20 failed,
  4 errors, of which not all were read.
- `shfl.sync`, `bar.warp.sync` and `elect.sync` do not wait in the simulator
  for lanes their masks name that are elsewhere; what a GPU does then was not
  measured (`vote.sync` was, and waits).
- The launch-error recordings of the RTX 3080 Ti are at Ampere's own limits
  and were not rerun verbatim on the B200; RT-04 and RT-12 cover them.
- `kind::i8` arithmetic and half-precision subnormals in an MMA.

## After v1

- Multi-GPU: NVLink / NVL72 domain, NVSHMEM and NCCL-style collectives, copy
  engines, multimem / NVLS.
- Structured sparsity, block-scaled formats (MXFP8/FP4, NVFP4) and the
  `tcgen05.cp` scale-factor paths; MMA kinds other than `f16` and `tf32`;
  operand A from tensor memory.
- A non-zero descriptor base offset, and `mbarrier.try_wait` returning false
  (a non-blocking fallback). The B200 recordings both waited for exist now
  (`validation/blackwell/measured/b200/mma/base_offset.field.*.txt`,
  `mbarrier/try_wait.incomplete.txt`).
- An interactive viewer that steps through a pipeline stage by stage; Jupyter
  widgets. The views are static SVG today.
- A `warplens check` command (the first plan's CLI named one; it is not
  built).
- Hopper (`sm_90a`) mode: `wgmma` on the same TMA and mbarrier engine.
- Optional SASS-level input via `nvdisasm` for timing studies.
- The timing model given what the fit of the first B200 run says it lacks (a
  cluster-launch-control latency per form of the request, the cost of an
  mbarrier wake-up, an issue rate that is not called a clock), fitted again,
  and the fit made the default: about twenty tests and the guide's worked
  examples state the assumed values and are rewritten with it.
- A calibration run on a B300: nothing in `b300.yaml` was measured on one.
- If profiling shows the interpreter is the bottleneck, the core (PTX IR
  execution) in Rust behind the same Python API.
  [docs/hardware-capture.md](docs/hardware-capture.md) describes the capture
  format a second implementation would be checked against.
