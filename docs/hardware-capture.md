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

Replay can run locally after downloading. It is also useful to replay a small case
before ending the rental, so missing artifacts are discovered while access remains.

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
