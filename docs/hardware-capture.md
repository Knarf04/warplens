# Hardware captures and offline replay

A rental must leave an independently usable reference corpus, not just a passing
console log. `env/remote/run.sh` collects it automatically. No B200 capture has been
made yet; simulator captures exercise the collection path and are labelled `sim`.

## What is retained

* Every functional microprobe: exact executable, embedded PTX, invocation, build
  arguments and compiler version, build-time local sources, stdout, stderr, exit
  status and timeout. A native-only executable has a separately identified PTX
  companion. It is never described as having executed that companion's PTX.
* End-to-end kernels: complete initial and resulting buffer bytes, exact compiled
  PTX, entry point, grid, block, cluster shape, dynamic shared memory, argument bytes,
  pointer relocations and semantic tensor-map encoder arguments. Shapes, dtypes,
  comparison rules and hashes are explicit. No tensor is reduced to a few printed
  elements or an error summary.
* Compiler/package versions, GPU UUID/model, driver, toolkit and repository revision.
  The outer archive includes the exact tracked Python source tree, tracked edits,
  pinned/resolved environment information, and checksums.
* Timing: raw trial output and profiling CSVs, tool/environment information, and the
  benchmark executables and invocations. Numerical replay does not validate timing.

The end-to-end inventory is four CUTLASS tutorials, two Triton kernels, TIRx SAXPY,
CuTe DSL SAXPY, and all nine book GEMMs (the book is cloned by `env/remote/setup.sh` at
the commit in `env/remote/pins.env` and used where it is; its files are never copied into
the archive). Missing dependencies, skipped cases, failed
programs, incomplete captures and corrupt artifacts fail collection. Successful
collection means the evidence is complete and the programs passed their own checks.
`validation/blackwell/compare.py` then compares every captured output tensor with the
simulator's, with the tolerances of validation matrix rows RT-06 to RT-09
(`validation/blackwell/README.md`, "End to end"); the replay below is the same
comparison under each fixture's own, self-declared policy.

CUTLASS capture builds use generated **host instrumentation** of the pinned tutorial
sources and TMA encoder wrapper; original device code is retained. Original tutorial
acceptance tests remain separate. `--cutlass-bin` checks that the requested original
tutorials are available; the captured executables are rebuilt from `--cutlass-source`,
with their source and binary identity recorded. TIRx captures export the exact host module used
for the run. A CPU-only pass through that module recovers its launch ABI, verifies
its PTX and initialized allocations, and intercepts launches without executing device
instructions. Its hardware expected outputs always come from the hardware run.
Raw outputs and the exported module are retained even when recipe extraction fails.

These are fixtures for the current validation programs, not a general CUDA application
recorder. Unsupported argument layouts, extra launches or scratch allocations fail
explicitly. Undefined-behavior probes retain observations and their manifest rules;
their arbitrary outputs do not become a universal hardware contract.

## Before releasing the rental

1. Run `env/remote/run.sh` and inspect its exit status, `stages.tsv`, and all failure
   logs. Do not treat an archive existing as proof of a complete run.
2. Copy the archive **and its `.sha256` file** to durable local storage. Check the
   archive checksum, unpack it, then run `sha256sum -c SHA256SUMS` inside it.
3. Verify the functional capture corpus, using the path inside the extracted archive:

   ```bash
   python -m validation.blackwell.capture verify /path/to/run/out/blackwell/captures --require-hardware
   python validation/blackwell/compare.py /path/to/run/out/blackwell
   ```

   `compare.py` replays each end-to-end capture in the simulator and fails on any
   element outside its matrix row's tolerance, a NaN, or a missing artifact; add
   `--require-hardware` there too. Give it `--sim-captures DIR` to use the tensors of a
   simulator run of the same programs where the PTX and inputs are the same.
   The individual suite archive also contains `captures/`. The verifier rejects
   simulator fixtures when `--require-hardware` is set; this label is provenance,
   not cryptographic attestation of the machine.
4. Check that the expected 17 end-to-end cases are present, all required probe
   statuses exist, the encoder sweep is populated, and required timing/counter data
   was collected. For the counters: `manifest.txt` must say `ncu=ok` and the archive
   must have no `NCU-COUNTERS-NOT-READ.txt`. A full run stops at preflight when `ncu`
   cannot read a counter, so that file exists only after `run.sh --no-ncu`,
   `--keep-going` or a run without the timing suite (`env/remote/RUNBOOK.md`, section 7).
   A partial/custom `--only` run cannot establish full-suite coverage.
5. Keep an immutable copy of the original archive. Preserve mismatches and failed
   attempts as well as successful results. Keep at least two independent copies.
6. Check the capture of the CuTe DSL kernel with TMA (matrix row RT-11, next section):

   ```bash
   python validation/blackwell/e2e/launch_capture.py verify /path/to/run/out/blackwell/rows/e2e/cute_dsl_tma
   ```

   It must print `RT-11 capture: COMPLETE`. `run.sh` exits 1 and ends with a banner when
   it is not there; `out/blackwell/RT-11-NOT-CAPTURED.txt` exists exactly then.

Replay can run locally after downloading. It is also useful to replay a small case
before ending the rental, so missing artifacts are discovered while access remains.

## The CuTe DSL kernel with TMA (RT-11): captured, and kept in the archive

CUTLASS's `fp16_gemm_0.py` cannot be one of the cases above. A case's tensor maps are
recipes, the arguments of `cuTensorMapEncodeTiled`, and the CuTe DSL never calls the
encoder: its compiled host code writes the 128-byte descriptors itself, in the driver's
layout, which the simulator could not read when the capture was made (it can now: below).
So its launch is captured from outside, by a debugger, and kept raw. It is in
`out/blackwell/rows/e2e/cute_dsl_tma/` of the archive, not in `captures/`; the capture
verifier above does not look at it and `compare.py` does not replay it.
`validation/blackwell/e2e/cute_dsl_tma.py` writes it, `validation/blackwell/e2e/launch_capture.py verify`
checks it, with no GPU.

| Path under `rows/e2e/cute_dsl_tma/` | What |
| --- | --- |
| `summary.txt`, `capture.sha256` | the verdict (`RT-11 capture: COMPLETE`), every pointer with its allocation and dumps; a checksum per file |
| `tutorial.stdout` | the unmodified tutorial at its own size; ends with `PASS` |
| `trace/launches.json`, `trace/launchNN.paramMM.bin` | the launch parameters at the tutorial's own size (8192 x 8192 x 8192): two more tensor maps for the decoder |
| `capture/dsl_dump/*.ptx` | the kernel's PTX, from the DSL's own dump (`CUTE_DSL_KEEP=ptx,cubin`). This is what a replay runs |
| `capture/moduleNN.bin`, `capture/dsl_dump/*.cubin` | what the driver was handed at `cuLibraryLoadData`: a fat binary around machine code, not PTX. The dumped cubin is checked to be inside it, which ties the PTX to the code that ran |
| `capture/launches.json` | the launch at `--mnk 256,512,512`: kernel name, grid, block, every parameter's offset, size and bytes; and under `memory`, every address found in a parameter (at any byte offset, so also inside a tensor map), the allocation it points into, the offset in it, and that allocation's dump before the kernel and after `cuCtxSynchronize` |
| `capture/mem/<sha256>.bin` | the dumps. A dump is a whole driver allocation: PyTorch keeps the three tensors in one allocation of its caching allocator, and where each starts is the pointer's `offset_in_allocation`. Their extents are the tutorial's (`a` is M x K, `b` N x K, `c` M x N, float16, row-major) |
| `capture/capture.json` | modules, kernel lookups, every allocation and free the tracer followed, and anything it could not do |

Done since, without a GPU: the decoder of driver-format tensor maps
(`warplens/host/tensor_map.py:decode_driver`, from the encoder sweep of TMA-12; the
descriptors here read back as the tutorial's tensors), and a replay of this directory:

```bash
python validation/blackwell/e2e/replay_launch.py /path/to/rows/e2e/cute_dsl_tma/capture
WARPLENS_RT11_CAPTURE=/path/to/rows/e2e/cute_dsl_tma/capture python -m pytest -q tests/validation/golden_replay/test_rt11_replay.py
```

The capture of 2026-11-25 replays with no byte of the allocation different from the
`after` dump, the 131,072 elements of `c` among them. The kernel needed two instruction
forms that were added for it (`tcgen05.mma` with `.collector::a::discard`,
`cvt.rn.f16x2.f32`: `docs/history/plan.md`, section 3.5). It is not a `warplens.capture.v1` case in the
repository: the directory is 42 MB, and its PTX is CUTLASS's tutorial compiled.

The address field of a driver-format tensor map is the first eight bytes, as the decoder
established. When the tracer was written it was not known. The tracer therefore
looks for addresses at every byte offset of every parameter, and when a launch has a
parameter of 64 bytes or more it also saves every live allocation, so that an address in a
form the scan does not see still has its memory in the capture (`summary.txt` then carries
a warning naming the parameter).

## Replay without a GPU or original compiler

From the frozen source snapshot (or a later Python implementation):

```bash
python -m validation.blackwell.capture replay /path/to/captures --out /tmp/python-replay
```

This loads the saved PTX, restores initial buffers, rebases pointers, reconstructs
simulator tensor maps from their encoder arguments, and runs each launch. It never
recompiles through Triton, TIRx, CuTe DSL or nvcc. Results and simulator diagnostics
are saved separately; a numerical mismatch exits nonzero. Kernel outputs alone do
not prove the race/synchronization checkers are complete.

A future Rust runner can consume the same fixtures and write the same output files
under an equivalent directory tree. Compare its bytes independently:

```bash
python -m validation.blackwell.capture compare /path/to/captures --out /tmp/rust-results
```

Use the frozen Python tree as a second oracle, alongside the hardware outputs. Do
not replace hardware expectations with Python output when they disagree. New
instructions, previously untested cases and timing calibration may still need a
future hardware run.

## Portable format: `warplens.capture.v1`

Each case has `capture.json`, `kernel.ptx`, and binary buffer files. JSON strings
and PTX use UTF-8. Tensor bytes are little-endian, contiguous C order, with dtype
strings such as `<f2`, `<f4`, `<u4`, or `|u1`, and an explicit shape. Initial bytes
exist for **every** buffer, including outputs. A buffer with `expected` is checked
in its entirety; aliases are represented by argument offsets into one buffer.

An artifact reference contains `file`, `bytes`, and its lowercase SHA-256. Filenames
are local basenames. `launch.args` follows PTX parameter order:

* `{"buffer": "A", "offset": 0}`: a 64-bit pointer into the named allocation.
* `{"hex": "01000000"}`: literal little-endian parameter bytes.
* `{"hex": "...", "relocations": [...]}`: a by-value parameter with substitutions.
  A pointer relocation is `{"offset": 8, "buffer": "A", "buffer_offset": 0}`.
  A tensor-map relocation replaces 128 bytes and contains `offset` and `tensor_map`.

`tensor_map` contains `buffer`, optional `buffer_offset`, and the encoder's
`data_type`, `rank`, `dims`, `strides`, `box`, `steps`, `swizzle`, `nan_fill`, and
`interleave`. Dimensions follow CUDA driver order (innermost first); strides are
bytes. Native descriptor bits remain evidence in the raw capture, but portable
replay reconstructs the descriptor for its own implementation.

Comparison policies are explicit per output: `exact` compares all bits;
`allclose` uses `abs(actual - expected) <= atol + rtol * abs(expected)`;
`ulp` bounds distance in the ordered IEEE encoding. Numerical policies reject
non-finite values and treat the two signed zeros as equal; use `exact` for NaN
payloads or signed-zero requirements. The book's hardware replay policy is at most
one half-precision ULP, separate from its looser NumPy reference self-check.

The format has one kernel launch per case. A new recorder that needs multiple
launches, pointer-containing memory, mutable module globals or another ABI must
extend and version the contract; it must not silently omit that state.
