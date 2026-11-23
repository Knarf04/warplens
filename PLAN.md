# warplens

**Understand a Blackwell GPU by watching it run, on your CPU.**

> Status: M0 to M2 done, M3 partly done (2026-11-23): CUTLASS tutorial 01 runs correctly; tutorial 02 and the TMEM layout view are open (docs/m3.md). M1's `ncu` cross-check matches on bank conflicts but not on the total wavefront count (validation/README.md). M2 and M3 are not validated on Blackwell hardware.
> GitHub description: *Understand a Blackwell GPU by watching it run, on your CPU.*
> Topics: `cuda` `gpu` `blackwell` `simulator` `ptx` `cutlass` `triton` `tensor-cores` `gpu-programming`

## 1. Goal

Take real kernel code written for datacenter Blackwell (sm_100a / sm_103a): CUDA C++, CUTLASS/CuTe, Triton, CuTe DSL.
Compile it to PTX on an ordinary machine and **execute it on the CPU** under a configurable GPU model (B200, B300).
Then show:

1. **Correctness:** output tensors, compared against a NumPy reference.
2. **Mechanisms:** how every byte moves (GMEM → TMA → swizzled SMEM → Tensor Core → TMEM → registers → SMEM → TMA store → GMEM), which mbarrier phase each step waits on, which banks each access hits.
3. **Misuse:** races, hangs, wrong `expect_tx` byte counts, TMEM lane-rule violations, missing proxy fences, bank conflicts. Each one is reported with the PTX line that caused it.
4. **Approximate performance:** a timeline and a utilization report for each resource (HBM, L2, TMA, SMEM, Tensor Core, issue slots). This answers questions like "is this pipeline TMA-bound or MMA-bound with 2 stages vs 4?"

### Non-goals for v1

- Cycle accuracy. The microarchitecture is not public, and even trace-driven research simulators miss by 5–13%. warplens aims for *explanatory* timing, clearly labeled as approximate.
- Multi-GPU, NVLink, NVSHMEM, NCCL. These are planned for v2 (see §10).
- SASS input. PTX is documented and SASS is not.
- Graphics, texture units, consumer Blackwell (sm_120).

## 2. Why this doesn't exist yet (survey, Nov 2026)

| Work | Data values | Timing | Blackwell tcgen05/TMEM | Runs without a Blackwell GPU end to end |
| --- | --- | --- | --- | --- |
| [ptxv](https://github.com/meta-pytorch/ptxv) (Meta, BSD, WIP) | yes (numerics) | no | yes | **no**: kernels are captured from a launch on a real Blackwell GPU |
| Accel-Sim 2.0 ([arXiv 2608.22602](https://arxiv.org/abs/2608.22602)) | no | cycle-level | yes (UTCMMA) | no: NVBit traces from real HW; code not found |
| Sim-FA ([arXiv 2605.00555](https://arxiv.org/html/2605.00555)) | no | cycle-level | no (Hopper TMA/WGMMA only) | traces from real HW; code not found |
| GPU-Tile-Sim ([arXiv 2607.11262](https://arxiv.org/abs/2607.11262)) | not stated | yes | preliminary | tile graphs, not real code; code not found |
| GPGPU-Sim / Accel-Sim 1.x | partial | yes | no (pre-Hopper) | yes |

**warplens's niche:** *real code in, no Blackwell hardware anywhere in the loop, functional + mechanism-checked + visual + approximate timing.*
The key enabler: warplens emulates the **host side** too (tensor-map encoding, launch with cluster dims, SMEM opt-in), so nothing has to be captured from hardware.

Target users:
- learners of Blackwell kernel programming, like this course's readers;
- kernel developers without B200 access;
- **LLM kernel-generation agents**, which need a cheap, no-GPU correctness and protocol checker in the loop.

## 3. Architecture

```
 CUDA C++ / CUTLASS ──nvcc -arch=sm_100a -ptx──┐
 Triton ──compile(target=cuda:100)────────────┤
 CuTe DSL ──export PTX (to investigate)───────┤
 hand-written PTX ────────────────────────────┤
                                              ▼
                                  ┌────────────────────┐
 host program / Python harness ──►│ host runtime emu   │  malloc, memcpy, cuTensorMapEncode*,
                                  │                    │  launch (grid, cluster, smem bytes)
                                  └─────────┬──────────┘
                                            ▼
                       ┌──────────────────────────────────────┐
                       │ PTX frontend: parse → IR, CFG, IPDOM │
                       └─────────────────┬────────────────────┘
                                         ▼
 ┌────────────────────────── discrete-event engine ─────────────────────────┐
 │  warps (SIMT, 32-lane vectors + active mask)   async units:              │
 │                                                  TMA engine, Tensor Core │
 │  state: GMEM | L2 (timing) | SMEM/CTA (+banks) | TMEM/CTA 128×512×32b    │
 │         registers/lane | mbarriers | bulk-async groups | tensor maps     │
 │  ── functional semantics ──┬── checkers ──┬── timing model (config) ──   │
 └────────────────────────────┼──────────────┼──────────────────────────────┘
                              ▼              ▼
              outputs + diff vs NumPy   reports: hazards, bank conflicts,
                                        Perfetto trace, utilization, layout views
```

### 3.1 Frontends (everything lowers to PTX)

- **CUDA C++ / CUTLASS:** the user runs `nvcc -arch=sm_100a -ptx` (CUDA ≥ 12.8, no GPU needed). warplens never redistributes NVIDIA headers or tools.
- **Triton:** compile for `GPUTarget("cuda", 100, 32)` without a GPU and take the PTX. Later, register a warplens "driver" so that `kernel[grid](...)` runs in the simulator.
- **CuTe DSL:** *open question.* Check whether `cute.compile` can emit PTX without a GPU or driver present, and how to extract the PTX and the launch parameters. This is a spike in M6.
- **Raw PTX:** first-class input, used for unit tests.

### 3.2 Host runtime emulation

- **v1:** a Python API.
  ```python
  sim = warplens.Device("B200")
  A = sim.from_numpy(a); C = sim.empty((M, N), "f32")
  tmap = sim.tensor_map_tiled(A, box=(64, 128), swizzle="128B")   # our own encoding of the 128-byte CUtensorMap
  sim.launch("gemm.ptx", "kernel", grid=(...), block=(...), cluster=(2, 1, 1), smem=..., args=[tmap, C, ...])
  ```
- **v1.x:** a drop-in `libcuda`/`libcudart` shim, so unmodified host programs (CUTLASS examples) run under `warplens run ./a.out`. This needs coverage of `cuTensorMapEncodeTiled`, `cudaLaunchKernelEx` (cluster dims), `cudaFuncSetAttribute`, memory APIs, and streams (serialized).
- **Tensor maps are opaque on real hardware.** Because warplens emulates the encode call, it defines its own 128-byte encoding and decodes it in the TMA unit.

### 3.3 PTX frontend

- A parser for the PTX grammar: directives, `.reg` / `.shared` / `.param`, predicates, vector operands, and inline-asm blocks that nvcc emits verbatim.
- IR → per-function CFG → immediate post-dominators, which give divergence reconvergence points.
- Coverage is driven by the target kernels (§6). Any unsupported instruction is a hard error that names the instruction, never silently skipped.

### 3.4 Execution model

- **Warp = unit of execution.** Each instruction runs on all 32 lanes at once as NumPy vectors, under an active mask; a SIMT stack handles divergence. This keeps Python fast enough for tile-sized problems.
- **Discrete-event engine.** Warps, the TMA engine, and the Tensor Core are actors. Async instructions (`cp.async.bulk.tensor`, `tcgen05.mma`, `tcgen05.commit`) enqueue work, and completion is an event (`complete_tx` on an mbarrier, `commit` arrival).
- **Scheduling:**
  - Functional mode uses a deterministic, configurable policy (round-robin or seeded random), with optional "adversarial" scheduling: delay async completions to expose races.
  - Timing mode orders events by the timing model.
- **Scale:** simulate a configurable subset of CTAs (default: one wave on k SMs). Shared resources (HBM, L2) are given each SM's share of bandwidth. Full 148-SM runs are allowed but slow.

### 3.5 State and semantics (v1 instruction surface)

| Area | PTX |
| --- | --- |
| Core | int/fp ALU, `cvt`, `setp`, `selp`, `bra`, `ret`, `ld/st.{global,shared,param,local}`, `atom`/`red` (global, shared), `shfl.sync`, `vote`, `elect.sync`, `bar.sync`/`bar.arrive` (named barriers), special registers (`%tid`, `%ctaid`, `%clusterid`, `%cluster_ctarank`, …) |
| mbarrier | `init`, `arrive`(`.expect_tx`, count), `expect_tx`, `complete_tx`, `test_wait`/`try_wait`(`.parity`), `inval` |
| TMA | `cp.async.bulk.tensor.{1..5}d` load (tile mode, OOB fill) and store, `cp.async.bulk` (non-tensor), `.multicast::cluster` (M5), `cp.async.bulk.commit_group`/`wait_group(.read)`, `prefetch.tensormap` |
| Fences / proxies | `fence.proxy.async`, `fence.mbarrier_init`, `fence.acq_rel` (modeled as checker events) |
| tcgen05 | `alloc`/`dealloc`/`relinquish_alloc_permit`, `mma` (`kind::f16`, `kind::tf32` first; `f8f6f4` and block-scaled later; `cta_group::1` first), `commit` (incl. `.multicast`), `ld`/`st` (32x32b, 16x64b, 16x128b, 16x256b shapes), `wait::ld/st`, `fence::before/after_thread_sync`, `cp` (M5) |
| Descriptors | SMEM matrix descriptor (start, LBO, SBO, swizzle mode, base offset), instruction descriptor (dtypes, M, N, major-ness, negate/transposes), decoded exactly as the PTX ISA specifies |
| Clusters (M5) | `barrier.cluster.arrive/wait`, `mapa`, DSMEM `ld/st.shared::cluster`, remote mbarrier arrive, `cta_group::2` (Layouts A/B for the M=256/128 pair), CLC `clusterlaunchcontrol.try_cancel` |

**Numerics:** the MMA computes in fp32 with documented input rounding; a bit-exact accumulation order is not promised. Report max abs/rel error vs NumPy.

### 3.6 Checkers (the main value for learning)

Each finding is reported with the PTX line, the warp/lane, and a short explanation of the rule.

- **Async-write hazards:** SMEM bytes written by TMA are tagged "pending (barrier B, phase p)". A generic-proxy read before that phase completes is a race.
- **TMA store source reuse:** overwriting SMEM before the matching `wait_group.read`. A TMA store of data written by `st.shared` without `fence.proxy.async` in between.
- **mbarrier:**
  - phase completes with bytes ≠ total box bytes (`expect_tx` mismatch);
  - an arrival that lands in the next phase;
  - waiting on a parity two phases behind (ABA);
  - use before `init` / fence.
- **Deadlock:** every warp is blocked, no async work is pending, and the kernel is unfinished. Report who waits on what.
- **TMEM:**
  - warp *w* accessing lanes outside `32·(w%4)…+31`;
  - reading accumulators before `commit` completes / `wait::ld`;
  - alloc not a power of 2 ≥ 32 columns;
  - leaking an allocation at exit.
- **SMEM banks:** wavefronts per access vs the minimum for its width; a heatmap per instruction.
- **Descriptors/layout:**
  - swizzle-mode alignment (1024 B for 128B mode);
  - box inner bytes > swizzle span;
  - an MMA descriptor whose layout disagrees with the TMA swizzle that wrote that buffer.

### 3.7 Timing model (approximate, config-driven)

- Every resource is modeled as a server with a latency and a throughput:
  - HBM (device-wide, shared);
  - L2 (hit/miss with a simple set-associative model, optional);
  - the TMA engine per SM (issue cost, bytes/clk);
  - SMEM per SM (128 B/clk of wavefronts);
  - the Tensor Core per SM (FLOP/clk per dtype, per MMA shape);
  - the TMEM ↔ RF path;
  - warp issue slots (4 sub-partitions per SM).
- **GPU configs** in YAML (`configs/b200.yaml`, `configs/b300.yaml`): SM count, clock, HBM bandwidth and capacity, L2 size, SMEM/TMEM sizes, Tensor Core rates. Values come from public specs and microbenchmark papers (e.g. [arXiv 2512.02189](https://arxiv.org/pdf/2512.02189)), with a `source:` field on every number.
- **Outputs:**
  - a Perfetto/Chrome trace with tracks per warp, TMA, Tensor Core, and HBM, plus mbarrier phase flips as instant events and counters for bytes in flight;
  - a summary table of utilization per resource and the bottleneck verdict;
  - roofline placement.

### 3.8 Visualization

- **v1 (static):**
  - SMEM layout view: logical tile → physical bytes, colored by bank;
  - TMEM map: which logical rows/columns are in which lane/column, for the four tcgen05 layouts;
  - mbarrier timeline;
  - all exported as SVG/HTML from `warplens report`.
- **Later:** an interactive web viewer that steps through a pipeline stage by stage.

## 4. Implementation choices

- **Language:** Python 3.11+ with NumPy for v1. This favors iteration speed, readability for learners, and contributions.
  - Hot loops are warp-vectorized.
  - If profiling shows the interpreter is the bottleneck, move the core (PTX IR execution) to Rust via PyO3, keeping the Python API.
- **Packaging:** `pip install warplens`; CLI `warplens run|report|check`. CUDA ≥ 12.8 is needed only to *compile* CUDA C++, never to run.
- **License:** Apache-2.0 (patent grant, friendly to companies). Do not vendor NVIDIA headers or PTX docs; link to them.

## 5. Repository layout

```
warplens/
  warplens/
    frontend/      ptx_parser.py, ir.py, cfg.py
    host/          device.py, tensor_map.py, launch.py, (shim/ later)
    core/          engine.py, warp.py, simt_stack.py
    units/         alu.py, memory.py, smem_banks.py, mbarrier.py, tma.py, tcgen05.py, tmem.py, cluster.py
    check/         hazards.py, deadlock.py, tmem_rules.py, report.py
    timing/        model.py, resources.py
    viz/           trace_perfetto.py, layout_svg.py, html_report.py
  configs/         b200.yaml, b300.yaml
  examples/        00_vecadd, 01_smem_transpose, 02_tma_copy, 03_mma_sm100, 04_pipelined_gemm, ...
  tests/           unit (per instruction), layout goldens, end-to-end kernels
  docs/            design.md, coverage.md (instruction support table), validation.md
```

## 6. Milestones and acceptance criteria

Each milestone ends with a runnable example, tests, and a short write-up in `docs/`.

| # | Scope | Done when |
| --- | --- | --- |
| **M0** skeleton | PTX parser (basic subset), warp SIMT executor, GMEM, Python launch API, CI | `vecadd` and a divergent-branch kernel, compiled by nvcc for sm_100a, run on CPU and match NumPy |
| **M1** SMEM | shared memory, named barriers, `shfl`, bank/wavefront model, conflict report | the transpose kernel in linear/padded/swizzled layouts reports 32-way / 0 / 0 conflicts. **Validated against `ncu` on a real Ampere GPU** (same counters) |
| **M2** TMA + mbarrier | tensor-map emulation, TMA load/store 1–5D tile mode, OOB fill, swizzle 32/64/128B, mbarrier full semantics, bulk groups, proxy fences, hazard + deadlock checkers | a TMA copy kernel round-trips GMEM→SMEM→GMEM for every swizzle mode. Deliberately broken variants (wrong expect_tx, missing fence, early reuse) are each caught |
| **M3** tcgen05 + TMEM | alloc/dealloc, SMEM + instruction descriptors, `mma` kind::f16/tf32 `cta_group::1` (M=64 Layout F, M=128), `commit`, `ld/st` shapes, lane-rule checker | the CUTLASS Blackwell CuTe tutorials (`examples/cute/tutorial/blackwell/01_mma_sm100.cu`, `02_mma_tma_sm100.cu`) produce correct GEMM output. The TMEM layout view matches the course's Layout F/B figures |
| **M4** timing + reports | timing model, B200/B300 configs, Perfetto trace, utilization summary, HTML report | a warp-specialized pipelined GEMM shows the expected trend over 1/2/4 stages (bottleneck moves from latency to TMA/HBM to Tensor Core). Every number in the report links to its config source |
| **M5** clusters | clusters, DSMEM, remote mbarrier, TMA multicast, `cta_group::2` (Layouts A/B), CLC | the CUTLASS tutorials with multicast and 2-SM MMA (`03_mma_tma_multicast_sm100.cu`, `04_mma_tma_2sm_sm100.cu`) are correct; Layout B worked example (C[10,11] → lane 74) is reproduced |
| **M6** frontends | Triton driver, CuTe DSL spike, libcudart/libcuda shim | a Triton matmul and an unmodified CUTLASS example binary run under `warplens run` |
| **v1.0** | docs, coverage table, tutorial notebook that follows the MLC course chapters | public release |

Suggested order of public visibility: announce after **M3**. "Run the CUTLASS Blackwell tutorial on your laptop and see every mbarrier phase" is the demo.

## 7. Validation strategy

1. **Spec-driven unit tests:** one test per instruction form, derived from the PTX ISA text and its examples.
2. **Golden layouts:** check swizzle and TMEM layouts against independent sources: the CuTe layout algebra evaluated on host (C++ `print_layout` / CuTe DSL), the course figures, and the PTX layout tables.
3. **Real-code end to end:** CUTLASS Blackwell tutorials, the course's examples, Triton tutorials. Compare outputs with NumPy.
4. **Hardware cross-checks where possible:**
   - the Ampere-level subset (SMEM, banks, `cp.async`, mbarrier without tx) against a real RTX 3080 Ti, using `ncu` counters;
   - later, community B200 runs to calibrate timing.
5. **Cross-check with ptxv** on shared kernels when someone with a Blackwell GPU can capture them.

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| PTX surface used by CUTLASS is large | coverage driven by target kernels; hard errors on unsupported ops; a `coverage.md` table so contributors can pick items |
| Undocumented behaviors (exact TMA/MMA timing, L2 policy, scheduling) | timing labeled approximate; every parameter is configurable and sourced; functional semantics stick to what the PTX ISA guarantees, and the checkers flag reliance on anything stronger |
| Python too slow | warp-level vectorization, simulate a subset of CTAs, Rust core later |
| CuTe DSL / Triton need a GPU or driver to compile | spike in M6; fall back to "export PTX + launch params" files |
| Tensor-map and descriptor encodings change across CUDA versions | own the host-side encoding; decode device-side descriptors exactly as the PTX ISA version declared in the PTX file |
| Overlap with ptxv / Accel-Sim 2.0 | position on *no hardware anywhere + education + visualization + timing*; credit and link them; consider interop (e.g. export our traces) |

## 9. Open questions

- ~~Project name~~ decided: **warplens** (GitHub and PyPI were free on 2026-11-23). Not related to NVIDIA Warp (`nvidia/warp`); say so in the README FAQ. Reserve the PyPI name early with a placeholder 0.0.1 release.
- How much of the PTX memory model to model in v1 (a checker for proxy and async ordering only, versus a full axiomatic model).
- Whether to support sm_103a (B300) specific instructions from day one, or only via config differences.

## 10. TODO for v2 and later

- Multi-GPU: NVLink / NVL72 domain, NVSHMEM and NCCL-style collectives, copy engines, multimem / NVLS.
- Structured sparsity, block-scaled formats (MXFP8/FP4, NVFP4) and `tcgen05.cp` scale-factor paths.
- Interactive web visualizer; Jupyter widgets.
- Hopper (sm_90a) mode: wgmma, the same TMA/mbarrier engine. This is cheap once M2 exists and broadens the audience.
- Optional SASS-level input via `nvdisasm` for timing studies.
- Calibration suite to run on real B200/B300 when available.
