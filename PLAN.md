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
  the simulator, which now reproduces the recorded outputs.
- They ran a second time the same day, on the same machine
  ([docs/b200-rental-2.md](docs/b200-rental-2.md); about 45 minutes of the 4
  GPU-hours in reserve). Every prediction the first run had changed was
  identical, and the whole programs were bit-identical. Two assumptions were
  answered and taken into the simulator: a tensor copy needs a shared address
  aligned to 128 bytes, and an undersized `expect_tx` completes its phase
  early only at a multiple of 32 bytes, in some runs and not in others.
- A third time on 2026-11-26, for the questions that were open
  ([docs/b200-rental-3.md](docs/b200-rental-3.md); about 40 minutes more of
  the reserve). The alignment was a prediction and held; the answers are
  listed below under "Closed by the third rental". What that run changed in
  the simulator reproduces what it recorded.
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

The tree is at `1.0.0`, the first release ([RELEASING.md](RELEASING.md)). It
carried `0.0.1` until then; the placeholder upload that number was meant for
was skipped (decided by the owner on 2026-11-26).

## Still owed for v1.0

1. **The release itself**, which is the owner's to make: the tag `v1.0.0`, the
   upload to PyPI (which also claims the name) and the GitHub release
   ([RELEASING.md](RELEASING.md), steps 7 to 9). The version, the changelog
   section, the classifier (`4 - Beta`, the owner's choice) and the install
   instructions are in the tree, and the files of that commit were built and
   checked as steps 5 and 6 say.

Decided by the owner on 2026-11-25:

- **The timing config stays as it is for v1.** `b200.yaml` and `b300.yaml`
  ship the assumed values and say so; the fit of the first run
  (`validation/blackwell_timing/measured/b200/calibrated.yaml`) is recorded
  and usable with `--timing-config`, and is not the default. No prediction
  changes, so the second rental's timing run was compared with the same
  predictions as the first: the measurements repeat (127 of 161 configurations
  within 1 %), and the first run's fit predicts the second run as well as its
  own, 45 of 152 within 10 % ([docs/b200-rental-2.md](docs/b200-rental-2.md),
  "The timing measurements"). What is left is the model's shape. Giving the model what the fit says it lacks is
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

- NUM-05: ptxas 13.3 and 13.5 have no UE4M3 conversion, so that format has
  nothing on a GPU to be compared with.
- One `vote.sync.ballot` on a B200 did not wait for lanes of its mask that
  were on another branch, while `vote.sync.all/any/uni` wait on both GPUs
  that were asked; the simulator makes every `vote.sync` wait
  ([docs/b200-rental-3.md](docs/b200-rental-3.md)).
- The launch-error recordings of the RTX 3080 Ti were made with the runtime
  of CUDA 12.4. The same program on the B200, with the runtime of CUDA 13,
  gave two other codes, which the shim now returns; whether the release or
  the GPU makes the difference was not separated.

Closed by the third rental (2026-11-26,
[docs/b200-rental-3.md](docs/b200-rental-3.md)): what separates the repeated
`tcgen05.dealloc` that ends a kernel from the one that runs (order);
`elect.sync` and `shfl.sync` do not wait for lanes that are elsewhere and
`bar.warp.sync` does; half-precision subnormals in an MMA; the launch errors
at a B200's limits. The first rental's self-test failures were all read.

## After v1

- Multi-GPU: NVLink / NVL72 domain, NVSHMEM and NCCL-style collectives, copy
  engines, multimem / NVLS.
- Structured sparsity, block-scaled formats (MXFP8/FP4, NVFP4) and the
  `tcgen05.cp` scale-factor paths; MMA kinds other than `f16` and `tf32`;
  operand A from tensor memory. `kind::i8` among them: its arithmetic was
  never compared with a GPU, and the simulator refuses the kind.
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
