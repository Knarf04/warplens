# Changelog

What changed in each release of warplens, newest first. A release is a version on PyPI and
the `v<version>` tag it was built from; `RELEASING.md` is how one is made. The section for the
version in `warplens/__init__.py` must exist (`tests/packaging/test_packaging.py` checks), so the
heading is written when the version is bumped, and the date when the tag is.

## 0.0.1 (not released)

The version the repository has carried since its first commit. `PLAN.md` reserves it for a
placeholder upload that claims the PyPI name; no upload has been made.

What the tree holds at this version (milestones M0 to M6 of `docs/history/plan.md`; the user
guide is `docs/guide.md`, and what is and is not modeled is in `docs/coverage.md`):

- A PTX parser and a SIMT executor for kernels compiled for `sm_100a` and `sm_103a`, with
  global and shared memory, named barriers and a bank-conflict report.
- TMA and tensor maps, mbarriers, `tcgen05` and tensor memory, clusters, distributed shared
  memory and cluster launch control, with checkers for misuse of these protocols.
- An approximate, uncalibrated timing model with B200 and B300 parameter files, a Perfetto
  trace and an HTML report.
- `warplens run`: compiled CUDA programs through `libcudart`/`libcuda` shim libraries that
  are compiled on first use, and Triton, TIRx and CuTe DSL scripts.
- Validation programs and the simulator's predictions for them (`validation/`), compared
  with an RTX 3080 Ti for what Ampere shares with Blackwell.

Compared with a B200 for the first time (2026-11-25; `docs/b200-rental-1.md` is the record,
`validation/blackwell/measured/b200/` and `validation/blackwell_timing/measured/b200/` hold
the outputs that differed from a prediction and the timing measurements):

- Of 126 probe runs, 88 were identical to the simulator's prediction and 8 differed. The
  differences were taken into the simulator, which now reproduces the recorded outputs: one
  `tcgen05.mma` is one sum cut toward zero and `kind::tf32` drops operand bits; a TMA copy
  larger than the bytes expected completes the mbarrier phase; the NaN of the TMA fill and of
  `cvt.rn/.rz.tf32.f32`; clusters of more than 8 CTAs need the non-portable attribute; what
  `maxDynamicSharedSizeBytes` reports above 48 KiB static; four faults that end the kernel,
  and a tensor-memory leak that fails the launch.
- A second run the same day (`docs/b200-rental-2.md`) held those changes against the GPU:
  every probe run whose prediction had changed was identical, and the whole programs were
  bit-identical. Two answers from it changed the simulator again, and have not been run on
  hardware since: a TMA tensor copy at a shared address that is not a multiple of 128 bytes
  ends the kernel (it was 32, an assumption); and a TMA copy larger than the bytes expected
  completes the phase early wherever those bytes are a multiple of 32 (it was 64), with a
  finding that states the accounting first and says that a B200 does this in some runs and
  hangs in others.
- A third run (`docs/b200-rental-3.md`, 2026-11-26) answered the open questions that needed
  a B200. A second `tcgen05.dealloc` of the same columns ends the kernel when it comes after
  the first and is reported when nothing orders the two, which is what tells the book's
  kernels from the probe that failed. `bar.warp.sync` waits for the lanes its mask names
  that are elsewhere, on an RTX 3080 Ti and on a B200; `shfl.sync` and `elect.sync` do not.
  A refused block or grid is `cudaErrorInvalidValue` and an attribute of a function that is
  not a kernel `cudaErrorInvalidResourceHandle`, as the runtime of CUDA 13 has them.
  Half-precision subnormals in an MMA were identical to the prediction.
- Tensor maps in the driver's own layout are read, from that run's encoder sweep.
- `tcgen05.mma` with `.collector::a::discard` and `cvt.rn.f16x2.f32` are implemented, for the
  kernel of CUTLASS's `fp16_gemm_0.py`, which replays byte for byte from its B200 capture.
- The timing model's configs are unchanged: the fit of that run is recorded
  (`validation/blackwell_timing/measured/b200/calibrated.yaml`) and not applied.

The pictures, sharpened after the first B200 run:

- The timeline has a row per mbarrier. Every phase completion carries the barrier's shared
  offset and its phase number and is joined to the waits it ended; a wait on an mbarrier
  names the barrier and phase it waits for, a TMA load the barrier it reports to.
  `LaunchStats.timing.instants` holds `Instant` records (`name`, `when`, `cta`, `args`) where
  it held tuples.
- The timeline and the trace have tracks for the busy time of HBM and of L2.
- The bank heatmap is coloured by conflict (words in a bank relative to the ideal wavefronts
  of the access), so a wide conflict-free access no longer looks like a conflict.
- `warplens.viz.smem.layout_svg(..., mark=(row, word))` and `--mark` follow one element of a
  tile into shared memory.
- Every timing output says whether its parameters are the illustrative defaults or an
  experimental fit to hardware (`GpuConfig.basis`); both are labelled simulated.
- `docs/gallery.md`: three annotated examples, regenerated by `tools/make_gallery.py`.

Documentation, reorganized before v1:

- `docs/guide.md` is the manual. `docs/architecture.md` is new and describes how the
  simulator works; the milestone write-ups it was drawn from (`m0.md` to `m6.md`) and the
  plan as first written are under `docs/history/`. `PLAN.md` is the remaining release
  checklist and the roadmap.
- The book's notebook and its helper moved from `notebooks/` to `examples/08_book/`, beside
  the runner they use.

Packaging, done ahead of the first upload:

- The wheel and the sdist are built and installed in CI (the `dist` job), and the suite runs
  against each in place of the checkout. `tests/packaging/test_packaging.py` builds both and checks
  their file lists.
- The sdist is the whole checked-in tree (`MANIFEST.in`), so the tests run from it.
- The version is written once, in `warplens/__init__.py`.
- Project metadata: classifiers, URLs, keywords, the license file.
