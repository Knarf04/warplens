# warplens user guide

This is the manual: what warplens covers, how to install it, every way to get
code into it, how a kernel is executed and what its numbers are, how to read
the diagnostics and the pictures, what has been compared with a real GPU, and
what is not supported. The other documents are references that this one links
to (section 10 lists them).

warplens executes GPU kernels on the CPU. It reads PTX, the assembly language
that every CUDA compiler produces, and runs it under a model of a datacenter
Blackwell GPU (B200 or B300): warps, shared memory, TMA, mbarriers, the
`tcgen05` tensor core, tensor memory and clusters. No GPU is involved.

Every command and program from section 4 on was run to write this guide, and
the output shown is what it printed. `tests/documentation/test_guide.py` runs them again
with the test suite and compares the output, so a block that stops working
fails a test. The installation commands of section 3 were run by hand.

Contents:

1. [What warplens is, and what it is not](#1-what-warplens-is-and-what-it-is-not)
2. [What has been checked, and against what](#2-what-has-been-checked-and-against-what)
3. [Installation](#3-installation)
4. [A first run](#4-a-first-run)
5. [Getting code into the simulator](#5-getting-code-into-the-simulator)
   - [5.1 A PTX file through the Python API](#51-a-ptx-file-through-the-python-api)
   - [5.2 A compiled CUDA program: `warplens run`](#52-a-compiled-cuda-program-warplens-run)
   - [5.3 Triton](#53-triton)
   - [5.4 TIRx](#54-tirx)
   - [5.5 The CuTe DSL](#55-the-cute-dsl)
   - [5.6 The kernels of *Modern GPU Programming for MLSys*](#56-the-kernels-of-modern-gpu-programming-for-mlsys)
6. [Reading the results](#6-reading-the-results)
   - [6.1 Launch statistics](#61-launch-statistics)
   - [6.2 What the kernel printed](#62-what-the-kernel-printed)
   - [6.3 Errors that stop a launch](#63-errors-that-stop-a-launch)
   - [6.4 Findings](#64-findings)
   - [6.5 The bank-conflict report](#65-the-bank-conflict-report)
   - [6.6 Traces and views](#66-traces-and-views)
   - [6.7 The timing report, and how far to trust it](#67-the-timing-report-and-how-far-to-trust-it)
   - [6.8 Everything on one page: `warplens report`](#68-everything-on-one-page-warplens-report)
7. [Execution, numerics and the two models](#7-execution-numerics-and-the-two-models)
   - [7.1 How a kernel is executed](#71-how-a-kernel-is-executed)
   - [7.2 Numerical behaviour](#72-numerical-behaviour)
   - [7.3 Models: B200 and B300](#73-models-b200-and-b300)
8. [Limits](#8-limits)
9. [Troubleshooting](#9-troubleshooting)
10. [Where to read more](#10-where-to-read-more)

## 1. What warplens is, and what it is not

It is three things in one run:

- **A functional simulator.** A kernel launch runs to completion and leaves
  its results in simulated device memory, where you compare them with NumPy.
  A warp executes each PTX instruction on 32 lanes at once; divergence, shared
  memory, asynchronous copies and the tensor core follow the PTX
  specification as far as it goes (section 2 says how far that is).
- **A set of checkers.** While the kernel runs, warplens watches for misuse
  that a GPU would not report: shared-memory races, reads of data a TMA copy
  has not delivered yet, a missing proxy fence, tensor memory read before its
  MMA was committed, and others (section 6.4). A kernel can produce the right
  answer and still get a finding, because the checks do not depend on the
  order in which things happened to run.
- **An approximate timing model**, on request. It gives each launch a
  simulated duration, a utilization per resource and a verdict on what limits
  the kernel. Its parameters are **not calibrated**: the configs that ship
  hold assumed values for most of them. A B200 has since been measured and a
  fit of those measurements is recorded, but it is not what the model runs
  with (section 6.7).

It is not:

- **cycle-accurate, or a performance predictor.** The timing model explains
  where time goes under stated parameters. It does not tell you how fast a
  kernel is on hardware.
- **validated on Blackwell hardware in full.** Its probes ran on a B200 once
  (2026-11-25, [b200-rental-1.md](b200-rental-1.md)): most of what it does
  matched, and what did not was changed to what the GPU did. Those changes
  have not themselves been run on hardware yet. Section 2 lists what was
  checked and how.
- **a GPU emulator for whole applications.** It runs PTX. It does not run
  machine code (SASS), so a library that ships only compiled kernels (cuBLAS,
  cuDNN, PyTorch's own operators) cannot be simulated.
- **fast.** The executor is Python and NumPy. The largest kernel in this
  guide's examples executes 102,705 warp-instructions in about four seconds.
  Use problems of a few tiles.
- **complete.** PTX it does not implement stops the launch with an error that
  names the instruction (sections 8 and 9). Nothing is skipped silently.
- related to NVIDIA Warp.

**Scope.** The input is PTX compiled for datacenter Blackwell: `sm_100a`
(B200) or `sm_103a` (B300). The instructions that are in, form by form, are
listed in [coverage.md](coverage.md); that list was frozen for v1 on
2026-11-24 and amended once, by two forms a CUTLASS tutorial needs
([PLAN.md](../PLAN.md)). Outside it for v1: multiple GPUs, machine code,
consumer Blackwell (`sm_120`), and the tensor-core kinds other than `f16` and
`tf32`. Section 8 has the limits in full.

## 2. What has been checked, and against what

Two GPUs have been compared with the simulator:

- an **RTX 3080 Ti** (Ampere, `sm_86`), the development machine, for
  everything Ampere shares with Blackwell
  ([validation/README.md](../validation/README.md));
- a rented **B200**, on one day, 2026-11-25, for the rest: 126 runs of small
  probe programs whose output the simulator had predicted beforehand, 17
  whole programs, and 14 timing benchmarks. [b200-rental-1.md](b200-rental-1.md)
  is the record of that run, and the outputs that differed from a prediction
  are in `validation/blackwell/measured/b200/`. A second run on the same
  machine, later that day, held the simulator as changed after the first
  against the GPU again ([b200-rental-2.md](b200-rental-2.md)).

Of the 126 probe runs, 88 printed exactly what the simulator had predicted
and 8 did not. The other 30 have no pass or fail: 18 had no prediction to
hold or ran what the simulator refuses, in 6 the GPU ended a kernel that the
simulator let return, in 3 it faulted where the simulator has no model, 2
timed out as predicted, and 1 could not be judged.

**What "changed since" means below.** Every difference of the first run was
taken into the simulator after it, and for each one the simulator then
printed the recorded B200 output line for line. That is a model fitted to one
recording, which is weaker than a prediction. The second run made them
predictions: of 130 probe runs, every one whose prediction had changed came
back identical, in both of its passes, and the whole programs were
bit-identical. It also answered two questions the first run had left as
assumptions, and both answers changed the simulator once more: the alignment
a tensor copy needs (128 bytes, not 32), and what an undersized `expect_tx`
does (below, under mbarriers). Those two changes have not been run on a GPU
again; for the first, what the simulator now does is what the second run
recorded.

| Area | On the RTX 3080 Ti | On the B200 (2026-11-25) |
| --- | --- | --- |
| `shfl.sync`, `vote`, barriers with exited threads, device `printf`, by-value structs, local arrays, calls to device functions (also with struct arguments, through function pointers, and recursive), `[reg+offset]` on a wrapped register | recorded outputs, replayed in `tests/validation/golden_replay/test_hardware_golden.py`: match exactly | the same programs rerun: every recorded line reproduced. **Changed since:** `elect.sync` executed by a lane outside its member mask ends the kernel |
| `cvt.rna.tf32.f32`, `ldmatrix`, `bfe`, `bfi`, `shf`, `prmt`, `.ftz` arithmetic and `cvt`, `min`/`max`, float-to-integer `cvt`, NaN bit patterns, integer division by zero, `cp.async` zero fill | the same, bit for bit | the same programs rerun: every recorded line reproduced |
| Atomic updates (`atom`, `red`): every atomic function on 32- and 64-bit integers, `atomicAdd` on `float` and `double`, in global and shared memory, as compiled by nvcc 13.5 and by nvcc 12.4 | match exactly, for results that do not depend on the order of the threads | rerun: reproduced |
| Shared-memory bank conflicts | Nsight Compute counters on the transpose example: conflict counters match exactly; the total wavefront counter matches once two per warp and one per CTA per barrier, which hardware also counts, are added | the same comparison: conflict counters equal, and the wavefront counter above the model's by the same amount |
| Launch errors and the shared-memory opt-in under `warplens run` | codes and the opt-in rule measured at Ampere's limits | the limit, the opt-in and the device attributes identical to the prediction. **Changed since:** what `maxDynamicSharedSizeBytes` reports for a kernel with more than 48 KiB static (0), and that a cluster of more than 8 CTAs needs the non-portable attribute |
| mbarriers | arrival tokens and `test_wait` predicates (`try_wait` and byte counts need Hopper or newer) | identical: phases and parity, arrival counts, `expect_tx` with bytes, tokens, `try_wait` on completed phases, completion before `expect_tx`. **Changed since:** a TMA copy larger than the bytes expected completes the phase; over-arrival ends the kernel. The second run showed that the early completion is not the same from run to run: it happens only where the bytes expected are a multiple of 32, and there in some runs and not in others (80 runs counted). The simulator reports the accounting, which does not vary, and shows the early completion `try_wait` on a phase that never completes returned false; the simulator blocks, by decision (section 8) |
| TMA, tensor maps | not possible | identical: every swizzle mode on aligned and misaligned buffers, ranks 1 to 5, element strides, zero fill, stores and `wait_group`, raw bulk copies, the missing fence. **Changed since:** the NaN of the out-of-bounds fill; two cases the GPU ends the kernel for (a tensor copy at a shared address that is not a multiple of 128 bytes, measured at ten offsets over the two runs; a store at a negative coordinate). The driver's own tensor-map layout is read from this run's encoder sweep |
| Tensor memory | not possible | identical: allocation addresses and order, the lane window, the layout of MMA results, every transfer shape and repeat count. **Changed since:** a larger allocation after a smaller one is made, as the GPU made it; a leak fails the launch. Still stricter than the GPU, on purpose: 48 columns, and an allocation after the permit was relinquished, are refused. Not modeled: three transfers past a warp's lane window faulted on the GPU and are refused here; tensor memory never written held an earlier launch's values on the GPU |
| `tcgen05.mma`: where things go | not possible | identical: operand layouts, negation, `scale-input-d`, disabled lanes, `commit` arrivals, the swizzle mismatch, swizzle by absolute address |
| `tcgen05.mma`: the arithmetic | not possible | **did not match, changed since.** An element is one sum cut toward zero, not a chain of float32 additions, and `kind::tf32` drops the low 13 bits of its operands. The simulator reproduces both recordings word for word; three rules of the sum were inferred from the book's kernels, and the probe written for them was identical on the second run ([numerics.md](numerics.md)) |
| Clusters | not possible | identical: the cluster barrier, multicast, remote arrive, the two-CTA MMA in all four forms, the address dump, every cluster-launch-control run |
| `cvt.rn.tf32.f32`, `cvt.rz.tf32.f32` | not possible (they need Hopper or newer) | identical for all finite inputs and infinities. **Changed since:** every NaN gives `0x7fffe000` (7 of 715 lines) |
| FP8 and FP4 `cvt` conversions | not possible | identical: every code of E4M3, E5M2 and E2M1, the E8M0 ties, the special inputs. UE4M3 has no conversion in ptxas 13.3 or 13.5, so nothing to compare |
| Whole programs (below) | not possible | bit-identical or within tolerance, except the book's `hgemm_v6` to `v9` at the time; all nine replay bit for bit since the MMA change, and on the second run every program was bit-identical to the simulator |
| Timing model | seven benchmarks, to show the measuring method is stable | 14 benchmarks measured and fitted, and measured again on the second run: 127 of 161 configurations within 1 % of the first, and the first run's fit predicts the second run's measurements as well as its own. **The shipped config was not changed**: it still holds the assumed values (section 6.7) |

Whole programs, each run on the B200 and then replayed in the simulator from
the PTX the GPU ran:

- the first four CUTLASS Blackwell tutorials, built from unmodified sources
  (section 5.2): bit-identical;
- Triton (section 5.3): the vector add bit-identical, the matmul within its
  tolerance;
- the TIRx and CuTe DSL `saxpy` scripts (sections 5.4, 5.5): bit-identical;
- the nine GEMM kernels of the book *Modern GPU Programming for MLSys*
  (section 5.6): `hgemm_v1` to `v5` within one unit in the last place of the
  half-precision output; `v6` to `v9` (K = 256) outside it, by up to 30
  units. That difference is what showed how an MMA sums. With the changed
  arithmetic all nine replay bit for bit against that day's captures
  (311,296 elements);
- the kernel of CUTLASS's CuTe DSL tutorial with TMA (`fp16_gemm_0.py`),
  captured on the GPU with its parameters and memory: it replays with no byte
  of its output different, after the two instruction forms it needs were
  added.

The captures of these programs are in the run's archive and not in this
repository (they are 79 MB), so the replays against them cannot be repeated
from a checkout; `tests/validation/golden_replay/test_rt11_replay.py` runs the last one when given
the capture.

What this evidence does not cover:

- **Anything the probes and programs do not exercise.** The probes were
  written from the simulator's own claims
  ([validation-matrix.md](validation-matrix.md) has one row per claim), and a
  rule nobody thought to state has no probe.
- **The two changes made after the second run**, as said above: the 128-byte
  alignment of a tensor copy is what that run recorded, and has not been
  predicted and then measured; for an undersized `expect_tx` the simulator
  shows one of the outcomes a GPU has, and does not say how often a GPU
  takes each ([b200-rental-2.md](b200-rental-2.md)).
- **What only a race could show**: when exactly a copy reads its source,
  which synchronization a program may rely on. The checkers report these
  instead of modeling an outcome (the matrix, "Claims not turned into a
  probe").
- **B300.** Nothing was measured on one (section 7.3).
- **Timing**, beyond the label on every number: section 6.7.

Kernels producing right answers is evidence too, and it has been wrong
before: the rule for which tensor memory lanes a warp can reach was
implemented incorrectly until the book's kernels contradicted it, and the
MMA arithmetic passed every NumPy comparison while differing from hardware.
[numerics.md](numerics.md) states each numerical rule and what it rests on.

## 3. Installation

warplens is not on PyPI yet. Install it from a checkout:

```bash
git clone https://github.com/Knarf04/warplens
cd warplens
pip install -e ".[dev]"
```

That needs Python 3.11 or newer and installs NumPy, PyYAML and pytest. It is
enough for the Python API on PTX files (section 5.1), including every `.ptx`
file checked in under `examples/`. The other ways in need more:

| To do this | You also need |
| --- | --- |
| Compile your own CUDA C++ to PTX | `nvcc` from CUDA 12.8 or newer (it must know `sm_100a`). No GPU or driver. |
| `warplens run` on a compiled program, a TIRx script or a CuTe DSL script | Linux, a C compiler, and the CUDA toolkit headers. Two small shim libraries are compiled on first use (see below). |
| Triton | `pip install -e ".[triton]"` (Triton 3.9 is what is tested). No shim: where it cannot be built, `warplens run script.py` says so and runs the script in its own process, with only the Triton driver. |
| TIRx | `pip install -e ".[tirx]"` (`apache-tvm-tirx` and `cuda-bindings`; 0.1.0 and 13.5.0 are what is tested), and an NVRTC library that knows `sm_100a` (CUDA 12.8 or newer; the conda env has 13.5). |
| The CuTe DSL | `pip install -e ".[cute-dsl]"` (`nvidia-cutlass-dsl`; 4.8.1 is what is tested). |
| The tutorial notebook (`examples/08_book/book_tour.ipynb`) | The TIRx row above and a clone of the book. To open it in Jupyter, `pip install -e ".[jupyter]"` (Jupyter's `notebook`, `nbclient`, `nbformat` and `ipykernel`, and the TIRx packages); without Jupyter its cells run as a plain Python script ([examples/08_book/README.md](../examples/08_book/README.md)). |
| All of the above | `pip install -e ".[all]"`. Every package the extras name is on PyPI. |

The repository's conda env, created by `env/setup.sh`, has the compiler side
of CUDA 13.5 (`nvcc`, headers, NVRTC) and nothing that needs a GPU;
[env/README.md](../env/README.md) describes it. This guide was run in that env
with `pip install -e ".[dev,triton,tirx,cute-dsl]"` on top.

**The shim.** `warplens run` works by putting two libraries in front of the
CUDA runtime: `libcudart.so.N` and `libcuda.so.1`, written in C in
`warplens/shim/`. They forward each runtime call to the simulator. They are
compiled the first time `warplens run` needs them, with `cc` (or `$CC`),
against the CUDA headers found next to the `nvcc` on `PATH`, and cached under
`~/.cache/warplens/` (or `$XDG_CACHE_HOME`). The build takes a second or two
and prints `warplens: building the shim libraries (first run)`. To build them
by hand, for example with headers that are somewhere else, run
`warplens/shim/build.sh <CUDA include directory>` (section 9). Libraries are
never used with other shim sources than they were built from: a changed
source means a new build, by itself.

## 4. A first run

The examples directory has kernels already compiled to PTX. This one adds
two vectors:

<!-- test: run -->
```bash
python examples/00_vecadd/run.py
```

<!-- test: output -->
```text
matches NumPy: True
LaunchStats(ctas=4, canceled_ctas=0, warps=32, threads=1024, instructions=704, lane_instructions=21264, divergent_branches=1, max_simt_depth=2, smem_bytes=0)
```

The kernel was compiled once by `nvcc` for `sm_100a`
(`examples/00_vecadd/vecadd.cu`); the script loads the PTX, runs it and
compares with NumPy. The rest of `examples/` goes on from there:

| Example | What it shows |
| --- | --- |
| `00_vecadd`, `00_divergent` | the Python API; a warp's active mask as its lanes diverge |
| `01_smem_transpose`, `01_warp_reduce` | shared memory, bank conflicts, barriers, shuffles |
| `02_tma_copy` | TMA and mbarriers, then five deliberate bugs and their reports |
| `03_cutlass_mma` | CUTLASS's four Blackwell tutorials (`build.sh` fetches and compiles them) |
| `04_pipelined_gemm` | the timing model on a pipelined GEMM |
| `05_triton`, `06_cute_dsl`, `07_tirx` | the other front ends |
| `08_book` | the nine GEMM kernels of the book *Modern GPU Programming for MLSys*, from your clone of it, and a notebook that tours them |
| `09_clc` | cluster launch control |

## 5. Getting code into the simulator

Everything ends up as PTX. The ways in differ in who plays the host: your own
Python (5.1), or the program's own host code with its CUDA calls redirected
(5.2 to 5.6).

| You have | Use | Section |
| --- | --- | --- |
| a `.ptx` file, or CUDA C++ you can compile to one | `warplens.Device` in Python | 5.1 |
| a CUDA C++ program with its own `main` | `warplens run ./program` | 5.2 |
| a Triton script | `warplens run script.py` | 5.3 |
| a TIRx script | `warplens run script.py` | 5.4 |
| a CuTe DSL script | `warplens run script.py` | 5.5 |

The last three rows are one command. A program whose name ends in `.py` is
run by the Python interpreter that runs `warplens`, as a child process with
the shim in front of it: `warplens run script.py` is
`warplens run python script.py`, which is the way to pass options to the
interpreter or to choose another one. TIRx and the CuTe DSL launch through
the CUDA runtime, which the shim replaces. Triton launches through its own
driver, and warplens's is installed in the child when the script imports
Triton.

### 5.1 A PTX file through the Python API

<!-- test: run -->
```python
import numpy as np
import warplens

rng = np.random.default_rng(0)
a = rng.standard_normal(1000).astype(np.float32)
b = rng.standard_normal(1000).astype(np.float32)

sim = warplens.Device("B200")                  # or "B300"
A, B = sim.from_numpy(a), sim.from_numpy(b)    # copies into simulated global memory
C = sim.empty(1000, "f32")                     # a PTX type name or a NumPy dtype
stats = sim.launch("examples/00_vecadd/vecadd.ptx", "vecadd",
                   grid=4, block=256, args=[A, B, C, 1000])

assert np.array_equal(C.numpy(), a + b)        # .numpy() copies back to the host
print(stats)
```

<!-- test: output -->
```text
LaunchStats(ctas=4, canceled_ctas=0, warps=32, threads=1024, instructions=704, lane_instructions=21264, divergent_branches=1, max_simt_depth=2, smem_bytes=0)
```

`Device.launch(ptx, entry, *, grid, block, args=(), smem=0, cluster=None,
timing=False, active_sms=None, trace=None, max_steps=None, async_delay=4,
schedule=None)`:

- `ptx` is a path, or a module from `warplens.parse_ptx(text)`. `entry` is the
  kernel's name as it appears after `.entry`; for C++ that is the mangled name
  unless the kernel is declared `extern "C"`.
- `grid` and `block` are an integer or a tuple of up to three. A block has at
  most 1024 threads.
- `args` has one entry per kernel parameter: a `DeviceArray` for a pointer, a
  Python `int` or `float` for a scalar (an `int` for a `float` parameter is
  refused, and the reverse), and `bytes`, a NumPy record or a `ctypes`
  structure for a struct passed by value.
- `smem` is the dynamic shared memory per CTA in bytes, as in a real launch.
  It sizes `extern __shared__` arrays. The limit per CTA is 227 KB.
- `cluster` is the cluster shape in CTAs, at most 16 CTAs per cluster; the
  grid must be a whole number of clusters.
- `max_steps` stops a kernel that does not finish. `async_delay` is how many
  scheduling rounds a TMA copy or an MMA takes to complete in an untimed run;
  findings do not depend on it.
- `schedule` is the order in which runnable warps are stepped in an untimed
  run: round-robin by default, a seed for a pseudo-random order, or
  `"adversarial"`. A correct kernel computes the same under all of them
  ([scheduling.md](scheduling.md)).
- `timing`, `active_sms` and `trace` are covered in sections 6.6 and 6.7.

Memory is `sim.from_numpy(array)`, `sim.empty(shape, dtype)` and
`array.numpy()`. Allocations are separated by unmapped gaps, so a stray access
faults instead of reading a neighbour. An array belongs to the `Device` that
allocated it.

**Compiling your own kernel.** Ask `nvcc` for PTX instead of an object file:

<!-- test: file scale.cu -->
```cpp
#include <cstdio>

extern "C" __global__ void scale(float* x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
    if (i == n - 1) printf("the last element is now %f\n", x[i]);
}
```

<!-- test: run needs=nvcc -->
```bash
nvcc -arch=sm_100a -ptx scale.cu -o scale.ptx
grep -c "" scale.ptx
```

<!-- test: output unchecked -->
```text
74
```

<!-- test: run needs=nvcc -->
```python
import numpy as np
import warplens

sim = warplens.Device("B200")
x = sim.from_numpy(np.arange(100, dtype=np.float32))
stats = sim.launch("scale.ptx", "scale", grid=1, block=128, args=[x, 0.5, 100])
print(x.numpy()[:4], stats.output, end="")
```

<!-- test: output -->
```text
[0.  0.5 1.  1.5] the last element is now 49.500000
```

**Tensor maps.** A kernel that uses TMA takes tensor maps, which on real
hardware the driver writes with `cuTensorMapEncodeTiled`.
`Device.tensor_map_tiled` plays that role. `box` is the tile one TMA
instruction moves, in the array's own (NumPy) axis order; inside the kernel
TMA coordinates run the other way, innermost first.

<!-- test: run -->
```python
import numpy as np
import warplens

sim = warplens.Device("B200")
data = np.random.default_rng(0).standard_normal((48, 64)).astype(np.float32)
src = sim.from_numpy(data)
dst = sim.from_numpy(np.full(data.shape, np.nan, dtype=np.float32))
box = (16, 32)                                           # rows, columns of one tile
src_map = sim.tensor_map_tiled(src, box, swizzle="128B")  # None, "32B", "64B" or "128B"
dst_map = sim.tensor_map_tiled(dst, box, swizzle="128B")
print(src_map)

# The kernel loads one tile per CTA into shared memory with TMA, doubles it, and stores it
# with TMA. Its last argument selects a deliberate bug; 0 is none.
stats = sim.launch("examples/02_tma_copy/tma_copy.ptx", "tma_copy",
                   grid=(64 // box[1], 48 // box[0]), block=128,
                   args=[src_map, dst_map, box[1], box[0], 0])
print("correct:", np.array_equal(dst.numpy(), 2 * data), "findings:", len(stats.findings))
```

<!-- test: output -->
```text
TensorMap(TensorMapInfo(rank=2, elem_size=4, is_float=True, swizzle=3, nan_fill=False, address=8589934592, dims=(64, 48), strides=(256,), box=(32, 16), steps=(1, 1)), addr=0x200008000)
correct: True findings: 0
```

A tensor map is passed like a pointer. A kernel that takes it by value inside
a struct, as CUTLASS kernels do, gets `tensor_map.tobytes()`. The 128 bytes are
warplens's own encoding, not the driver's (section 8).

**Kernels with by-value arguments and clusters.**
`examples/03_cutlass_mma/run.py` runs the four CUTLASS tutorials this way,
building each kernel's CuTe argument structs by hand; tutorials 3 and 4 launch
clusters of 16 CTAs with `cluster=(4, 4, 1)`. `build.sh` clones CUTLASS at a
pinned commit and compiles the tutorials to PTX; it needs `nvcc` and network
access once.

<!-- test: run needs=cutlass -->
```bash
# examples/03_cutlass_mma/build.sh --programs     (once; also builds the programs for 5.2)
python examples/03_cutlass_mma/run.py 4
```

<!-- test: output unchecked -->
```text
tutorial 4, GEMM 512 x 1024 x 128: max abs error vs NumPy 8.58e-06 (relative 4.64e-07)
  16 CTA(s), 76432 warp-instructions, 0 finding(s), 25 line(s) printed by the kernel
```

Section 5.2 runs the same tutorials with no hand-written host side.

### 5.2 A compiled CUDA program: `warplens run`

An ordinary CUDA program can run unmodified, with its own `main`, its own
`cudaMalloc` and its own kernel launches. Two things have to be decided when
it is built, both by flags and neither in the source:

<!-- test: file saxpy.cu -->
```cpp
#include <cstdio>
#include <vector>

__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
  if (i == 0) printf("thread 0 sees n = %d\n", n);
}

int main() {
  const int n = 1000;
  std::vector<float> x(n, 1.0f), y(n, 2.0f);
  float *dx, *dy;
  cudaMalloc(&dx, n * sizeof(float));
  cudaMalloc(&dy, n * sizeof(float));
  cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  saxpy<<<(n + 255) / 256, 256>>>(3.0f, dx, dy, n);
  cudaError_t status = cudaDeviceSynchronize();
  cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("status %d, y[0] = %g, y[%d] = %g\n", (int)status, y[0], n - 1, y[n - 1]);
  return y[n - 1] == 5.0f ? 0 : 1;
}
```

<!-- test: run needs=nvcc -->
```bash
nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared saxpy.cu -o saxpy
warplens run ./saxpy
```

<!-- test: output -->
```text
thread 0 sees n = 1000
status 0, y[0] = 5, y[999] = 5
warplens: 1 kernel launch(es) simulated on B200, 0 finding(s)
  _Z5saxpyfPKfPfi: 4 CTA(s), 776 warp-instructions
```

- `code=compute_100a` makes the program carry PTX instead of machine code,
  and `--no-compress` keeps that PTX readable.
- `--cudart shared` links `libcudart.so` dynamically, so that `warplens run`
  can stand in for it. The default is a static runtime, which cannot be
  replaced.

`warplens run [options] program [arguments...]` starts the program with the
shim libraries preloaded and executes its runtime calls in the simulator.
The options come before the program; what follows it is the program's own.
They apply to every launch the program makes, whichever of 5.2 to 5.6 it is:

| Option | Meaning | `Device.launch` |
| --- | --- | --- |
| `--model B200\|B300` | the GPU to simulate (default B200) | `Device(model)` |
| `--timing` | also run the timing model (6.6) | `timing=True` |
| `--timing-config CONFIG` | the parameters of the timing model: a YAML file like `warplens/configs/b200.yaml`, or the name of a built-in config (`b200`, `b300`). Implies `--timing`. Default: the config of `--model` | `timing=GpuConfig.load(CONFIG)` |
| `--active-sms N` | how many SMs the timing model assumes to share the memory bandwidth, from 1 to the SM count of the timing config. Implies `--timing`. Default: all of them | `active_sms=N` |
| `--clc-limit N` | cluster launch control ([clc.md](clc.md)): how many cancellations each cluster that starts is granted; 0 makes every cluster start by itself. Default: every request is granted while clusters are left | `clc_limit=N` |
| `--quiet` | do not print the summary | |

A value that cannot be used (a negative count, more SMs than the config has, a
config file that is missing or incomplete) is refused before the program
starts, with exit status 2. This is what the program's calls become:

| What the program does | What happens |
| --- | --- |
| `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaFree` | simulated global memory; a freed address is never handed out again |
| `kernel<<<grid, block, smem>>>(args)`, `cudaLaunchKernelExC`, `cuLaunchKernel[Ex]` | a launch, with the cluster shape if one is given |
| `cuTensorMapEncodeTiled` | a warplens tensor map, validated like the real call |
| `printf` in a kernel | printed by the program, in order with its own output |
| device queries | a simulated B200: compute capability 10.0, 148 SMs (10.3 and 160 SMs for `--model B300`) |
| `cuModuleLoadData`, `cudaLibraryLoadData` and the lookups that follow | the PTX is parsed; a module of machine code is refused unless the compiler kept its PTX (5.5) |
| streams, events, `cudaMallocHost` | accepted; everything is synchronous and an elapsed time is zero |

When the program exits, `warplens run` prints one line per kernel launch and
every finding, on standard error. `--timing` adds the simulated duration and
the verdict of the timing model to each line; `--quiet` drops the summary. The
exit status is the program's. An error inside a kernel (a memory fault, a
deadlock, unsupported PTX) is printed and reaches the program as
`cudaErrorLaunchFailure`. A launch refused before the kernel runs has the code
a GPU gives it: `cudaErrorInvalidValue` (too much shared memory),
`cudaErrorInvalidConfiguration` (block or grid dimensions) or
`cudaErrorInvalidClusterSize` (a cluster the device does not accept).

"Too much shared memory" includes the opt-in a GPU requires. A kernel may use
48 KB of shared memory per CTA unless the program asks for more before it
launches:

```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);  // up to 227 KB
kernel<<<grid, block, bytes>>>(...);
```

(`cuFuncSetAttribute` with `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
in the driver API.) Without it the launch is refused here as on the GPU, and
`warplens run` says which kernel asked for how much:

```text
warplens: kernel `smem_touch` was launched with 65536 bytes of dynamic shared memory, but without opt-in it may use 49152. A GPU refuses this launch with cudaErrorInvalidValue. Before launching, call cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536), or with the driver API cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)
```

Static shared memory counts against the 48 KB; a kernel whose static shared
memory alone is larger needs no opt-in. The details are in
[architecture.md](architecture.md#9-warplens-run-the-shim-and-the-front-ends).
`Device.launch` (section 5) has no such step: its `smem` is
simply what the launch gets.

The CUTLASS tutorials are programs of this kind. Built with those flags, they
run as they are and pass their own check against a reference:

<!-- test: run needs=cutlass,shim -->
```bash
warplens run examples/03_cutlass_mma/bin/tutorial_04 2>&1 | tail -n 6
```

<!-- test: output -->
```text
Infinity norm of (C - C_ref): 0.000000e+00
Relative error: 0.000000e+00
Execution is successful.
warplens: 2 kernel launch(es) simulated on B200, 0 finding(s)
  _ZN3cub17_V_300500_SM_10006detail8for_each13static_kernelINS2...: 1024 CTA(s), 147456 warp-instructions
  _Z11gemm_deviceI13SharedStorageIN7cutlass6half_tES2_N4cute14C...: 16 CTA(s), 79207 warp-instructions
```

The first launch is a `for_each` kernel from the CUDA libraries that the
tutorial's host code uses on its device vectors; it is simulated like any
other.

### 5.3 Triton

Triton compiles for `sm_100a` with its own bundled toolchain and needs no GPU
for that. warplens supplies a Triton driver that launches into the simulator.
`warplens run script.py` runs the script and installs the driver when the
script imports Triton:

<!-- test: run needs=triton -->
```bash
warplens run examples/05_triton/run.py
```

<!-- test: output unchecked -->
```text
vector add: max abs error vs NumPy 0; 8 CTA(s), 672 warp-instructions, 0 finding(s)
matmul: max abs error vs NumPy 4.77e-06; 4 CTA(s), 8480 warp-instructions, 0 finding(s)
warplens: 2 kernel launch(es) simulated on B200, 0 finding(s)
  add_kernel: 8 CTA(s), 672 warp-instructions
  matmul_kernel: 4 CTA(s), 8480 warp-instructions
```

The matmul's `tl.dot` becomes a `tcgen05.mma`, so this exercises the same
tensor core as the CUTLASS examples through a different compiler.

A script can also install the driver itself. Tensors come from the driver,
`gpu.tensor(numpy_array)` and `gpu.empty(shape, dtype)`, not from PyTorch:

<!-- test: run needs=triton -->
```python
import warnings

import numpy as np
import triton
import triton.language as tl
import warplens.triton_driver as wt

warnings.filterwarnings("ignore", module="triton")

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
             mask=mask)


gpu = wt.install("B200")            # Triton now compiles for, and launches on, the simulator
try:
    a = np.arange(1000, dtype=np.float32)
    out = gpu.empty(1000, "float32")
    add_kernel[(triton.cdiv(1000, 128),)](gpu.tensor(a), gpu.tensor(a), out, 1000, BLOCK=128)
    stats = gpu.last_stats          # the LaunchStats of the most recent launch
    print(np.array_equal(out.numpy(), 2 * a), stats.ctas, "CTAs,", len(stats.findings), "findings")
finally:
    wt.uninstall()
```

<!-- test: output -->
```text
True 8 CTAs, 0 findings
```

`wt.install("B200", timing=True)` passes launch options to every launch.
`gpu.launches` is the list of `(kernel name, LaunchStats)`.

### 5.4 TIRx

TIRx (`apache-tvm-tirx`, the language of the book in 5.6) compiles with NVRTC
in process and launches through the CUDA driver, which `warplens run`
replaces:

<!-- test: run needs=tirx,shim -->
```bash
warplens run examples/07_tirx/run.py
```

<!-- test: output -->
```text
the last thread is 63 of block 3
saxpy over 256 elements on compute capability 10.0: matches NumPy
warplens: 1 kernel launch(es) simulated on B200, 0 finding(s)
  saxpy_kernel: 4 CTA(s), 159 warp-instructions
```

The script asks the device for its compute capability, is told 10.0, and
compiles for `sm_100a`. One thing in the script is there for the simulator:
it asks for PTX,

```text
T.device_entry(launch=..., backend_config={"cuda": {"target_format": "ptx"}})
```

because TIRx hands the driver machine code by default. A real GPU accepts PTX
too, so the same script runs on hardware. Arrays are TVM's own
(`tvm_tirx.runtime.tensor(numpy_array, device=tvm_tirx.cuda(0))`), which
allocate through the shared CUDA runtime and so land in simulated memory.

### 5.5 The CuTe DSL

The CuTe DSL (`nvidia-cutlass-dsl`, the Python front end of CUTLASS) compiles
host code that launches through the CUDA runtime. `warplens run`
arranges what that needs through the DSL's own environment variables: the
target architecture, a copy of the DSL's runtime library linked against the
shim (built on first use), and the PTX of each kernel, which the DSL is asked
to keep because the module it loads holds only machine code. The script is not
involved. Here `python` is spelled out to pass it an option:

<!-- test: run needs=cute_dsl,shim -->
```bash
warplens run python -W ignore examples/06_cute_dsl/saxpy.py
```

<!-- test: output -->
```text
saxpy over 256 elements: matches NumPy
warplens: 1 kernel launch(es) simulated on B200, 0 finding(s)
  kernel_cutlass_saxpy__tensorptrf32gmemo2561_tensorptrf32gmemo...: 4 CTA(s), 120 warp-instructions
```

(`-W ignore` hides a warning from the DSL that its compile cache is off,
which keeping the PTX causes.)

`saxpy.py` holds its arrays as device memory from the CUDA driver, through the
DSL's own helpers (`cutlass.base_dsl.runtime.cuda`), and passes them as
pointers. The kernel of CUTLASS's TMA tutorial (`fp16_gemm_0.py`) runs too,
from the tensor maps the DSL writes itself; the script does not, because it
keeps its arrays in PyTorch. See section 9.

### 5.6 The kernels of *Modern GPU Programming for MLSys*

The [book](https://mlc.ai/modern-gpu-programming-for-mlsys/) develops a GEMM
for Blackwell in nine steps, `hgemm_v1` to `hgemm_v9`, in TIRx. Its repository
has no license, so none of its code is in this one. `examples/08_book/run.py`
reads the kernel definitions from your clone and runs them:

<!-- test: run needs=book,tirx,shim -->
```bash
# git clone https://github.com/mlc-ai/modern-gpu-programming-for-mlsys book
# export WARPLENS_BOOK=$PWD/book
warplens run python examples/08_book/run.py "$WARPLENS_BOOK" v1 v7
```

<!-- test: output unchecked -->
```text
hgemm_v1 128x128x64: matches NumPy (max error 0.0156, 0.2 s)
hgemm_v7 128x128x256: matches NumPy (max error 0.0078, 1.6 s)
warplens: 2 kernel launch(es) simulated on B200, 1 finding(s)
  kernel_kernel: 1 CTA(s), 5010 warp-instructions
  kernel_kernel: 148 CTA(s), 48255 warp-instructions
    [tmem-dealloc-repeated] tensor memory at column 0 (512 columns) was already released by an earlier tcgen05.dealloc; this one is ignored (148 times)
      line 2009: tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r275, %r276;
      first in CTA (0, 0, 0), warp 4, lanes 0-31
```

Two things differ from the book's own cells: the arrays are TVM's instead of
PyTorch's, and the target asks for PTX. Without version arguments the runner
does all nine, each on the smallest problem it does any work on; that takes
about a minute and a half here, most of it compilation. `hgemm_v7` to `v9` get
one finding each, `tmem-dealloc-repeated`, explained in section 6.4.

The notebook beside the runner,
[examples/08_book/book_tour.ipynb](../examples/08_book/book_tour.ipynb), goes
through the nine kernels chapter by chapter and shows, for each, what the simulator can
see: the PTX the kernel became, which warp does what, the findings, the tensor
memory layout and the timing model's account. Jupyter is optional: the
notebook's code cells are plain Python and run as a script
([examples/08_book/README.md](../examples/08_book/README.md)).

The notebook gets those objects from `warplens.run_program`, which is
`warplens run` as a Python function: it runs a command with its GPU work done
by the simulator and returns the program's output and, for every launch, the
parsed PTX and the `LaunchStats` that the command only summarizes.

```python
import sys, warplens
from warplens.viz import timeline_svg

done = warplens.run_program([sys.executable, "examples/07_tirx/run.py"], timing=True)
launch = done.launch                       # the only launch; done.launches has them all
print(launch.kernel, launch.grid, launch.stats.timing.summary())
svg = timeline_svg(launch.stats.timing)    # the picture the HTML report draws
```

## 6. Reading the results

### 6.1 Launch statistics

`Device.launch` returns a `LaunchStats`. `warplens run` prints a line of it per
launch; under the Triton driver it is `gpu.last_stats`.

| Field | Meaning |
| --- | --- |
| `ctas`, `warps`, `threads` | what was launched |
| `instructions` | PTX instructions executed, counted once per warp ("warp-instructions") |
| `lane_instructions` | the same, counted once per active lane |
| `divergent_branches` | branches at which a warp's lanes disagreed |
| `max_simt_depth` | deepest nesting of divergence |
| `smem_bytes` | shared memory per CTA, static plus dynamic |
| `smem` | the bank-conflict record (6.5) |
| `findings` | what the checkers reported (6.4) |
| `output` | what the kernel printed (6.2) |
| `timing` | the timing report of a timed launch, otherwise `None` (6.7) |

`stats.report()` returns the bank-conflict table and the findings as text.
The count of warp-instructions can differ by a few between a timed and an
untimed run of the same kernel, because a warp that polls an mbarrier polls a
different number of times.

### 6.2 What the kernel printed

Device `printf` works. Through the Python API the text is in `stats.output`
(the `scale` example in 5.1). Under `warplens run` the program prints it
itself, as it would on a GPU. Formatting was compared with a real GPU for
`%d %ld %hd %f %5.2f %#x %s %c`.

### 6.3 Errors that stop a launch

All derive from `warplens.WarplensError`. Those raised while a kernel runs
name the PTX line and the CTA, warp and lanes.

| Exception | Raised when |
| --- | --- |
| `PtxSyntaxError` | the PTX text cannot be parsed |
| `UnsupportedError` | valid PTX that warplens does not implement |
| `LaunchError` | the launch is set up wrongly: argument count or types, block size, shared memory over the limit, a bad tensor map |
| `ExecutionError` | the kernel did something invalid: an mbarrier used before `init`, a tensor-map operand that is not a tensor map, `max_steps` exceeded |
| `MemoryFault` (an `ExecutionError`) | a load or store outside every allocation |
| `DeadlockError` (an `ExecutionError`) | every unfinished warp is blocked and nothing can release them |

<!-- test: run -->
```python
import numpy as np
import warplens

sim = warplens.Device("B200")
A, B, C = (sim.from_numpy(np.ones(100, dtype=np.float32)) for _ in range(3))
try:  # the kernel is told the arrays hold 200 elements
    sim.launch("examples/00_vecadd/vecadd.ptx", "vecadd", grid=1, block=256, args=[A, B, C, 200])
except warplens.MemoryFault as error:
    print(error)
```

<!-- test: output -->
```text
read of 4 byte(s) at 0x200000190 is outside every allocation (0 byte(s) past the end of the 400-byte allocation at 0x200000000)
  at vecadd.ptx:41: ld.global.b32 	%r6, [%rd8];
  in kernel `vecadd`, CTA (0, 0, 0), warp 3
  lane 4 (thread (100, 0, 0)) and 27 more lane(s)
```

A deadlock report says who waits where and what each mbarrier still needs.
`DeadlockError.findings` holds what the checkers had seen up to then, which is
often the cause. Here the kernel of 5.1 is told to expect twice the bytes its
TMA load delivers (its bug 2):

<!-- test: run -->
```python
import numpy as np
import warplens


def tma_copy(bug):
    sim = warplens.Device("B200")
    data = np.ones((16, 32), dtype=np.float32)
    maps = [sim.tensor_map_tiled(sim.from_numpy(data), (16, 32), swizzle="128B") for _ in range(2)]
    return sim.launch("examples/02_tma_copy/tma_copy.ptx", "tma_copy", grid=1, block=128,
                      args=[*maps, 32, 16, bug])


try:
    tma_copy(bug=2)
except warplens.DeadlockError as error:
    print(error)
```

<!-- test: output -->
```text
deadlock in CTA (0, 0, 0): every unfinished warp is blocked and nothing can release them
  warps 0-3 wait at line 84: mbarrier.try_wait.parity.shared::cta.b64 P1, [%r13], %r14;
    mbarrier at shared offset 8192 is in phase 0, still needs 0 arrival(s) and 2048 byte(s) of TMA data; no TMA load in flight will deliver them (expect_tx too large?)
```

### 6.4 Findings

A finding is something the kernel did that is wrong or suspicious but did not
stop it. Each has a kind, a message, the PTX line and instruction, and the
first CTA, warp and lanes where it happened; repeats at the same instruction
are counted, not listed again. Bug 4 of the same kernel leaves out the proxy
fence before its TMA store, and still produces the right output:

<!-- test: run -->
```python
import numpy as np
import warplens

sim = warplens.Device("B200")
data = np.ones((16, 32), dtype=np.float32)
maps = [sim.tensor_map_tiled(sim.from_numpy(data), (16, 32), swizzle="128B") for _ in range(2)]
stats = sim.launch("examples/02_tma_copy/tma_copy.ptx", "tma_copy", grid=1, block=128,
                   args=[*maps, 32, 16, 4])
for finding in stats.findings:
    print(finding)
    print("fields:", finding.kind, finding.line, finding.cta, finding.warp, finding.count)
```

<!-- test: output -->
```text
[tma-store-missing-fence] TMA store of shared memory written by ordinary stores with no fence.proxy.async in between; TMA may read stale data
  line 117: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd2, {%r21, %r22}], [%r20];
  first in CTA (0, 0, 0), warp 0, lanes 0
fields: tma-store-missing-fence 117 (0, 0, 0) 0 1
```

`python examples/02_tma_copy/run.py` shows five such bugs and their reports.

The checks do not depend on timing. Shared memory records, per byte, who last
wrote and read it and which asynchronous copy targets it; each warp keeps the
set of copies and MMAs it has synchronized with. Touching the bytes of an
operation outside that set is reported even when the data happened to arrive
in time.

Every kind, by area:

| Kind | Meaning |
| --- | --- |
| **Integer arithmetic** | |
| `integer-divide-by-zero` | an active integer `div` or `rem` lane has a zero divisor; its deterministic fallback is not a hardware prediction |
| **Shared memory** | |
| `smem-race` | two warps touched the same byte, at least one writing, with no barrier completing in between. Two atomic updates (`atom`) are not a race with each other; an atomic update and another warp's ordinary load or store are, which includes a compare-and-swap loop that starts with a plain load of the word |
| `smem-uninitialized-read` | a load, an atomic update, a TMA store or an MMA read shared memory this CTA never wrote |
| **Barriers and shuffles** | |
| `barrier-in-divergent-code` | an aligned CTA barrier has a divergent guard predicate, or a cluster barrier has partial warp participation |
| `shfl-inactive-source` | a shuffle read a lane that was not executing it or was outside the member mask |
| `shfl-membermask-mismatch` | a caller is absent from its own mask, or a named member is not executing with the same mask |
| `warp-sync-membermask-mismatch` | the same, for `bar.warp.sync` (`__syncwarp()`): lanes on separate divergent paths do not wait for each other |
| `vote-membermask-mismatch` | a `vote.sync` caller is absent from its own mask, or a member executes it with another mask. (A member that is elsewhere is waited for; if it cannot arrive, the launch is refused) |
| **mbarrier** | |
| `mbarrier-tx-mismatch` | completion bytes not balanced by `expect_tx` when the kernel returned or deadlocked |
| `mbarrier-over-arrival` | more arrivals than the phase needed. An error, not a finding: a B200 ends the kernel |
| `mbarrier-init-race` | an mbarrier used by another warp with no barrier since its `init` |
| `mbarrier-invalid-token` | a state token is outside the current or immediately preceding phase; it grants no synchronization credit |
| **TMA and `cp.async`** | |
| `tma-load-unsynchronized` | shared memory that a TMA load writes was read, written, stored or fed to an MMA without waiting on the load's mbarrier |
| `tma-store-missing-fence` | a TMA store of bytes written by ordinary stores with no `fence.proxy.async` in between |
| `tma-store-source-overwritten` | bytes a TMA store reads were overwritten before waiting for the store |
| `tma-load-not-waited`, `tma-store-not-waited` | the kernel returned with the copy still in flight |
| `tma-swizzle-misaligned` | a swizzled buffer not aligned to the swizzle period |
| `cp-async-overlapping-writes` | two `cp.async` copies in flight write the same shared bytes |
| `cp-async-not-waited` | the kernel returned with a `cp.async` copy in flight |
| **Tensor core and tensor memory** | |
| `tmem-read-before-commit` | accumulators read while their MMA was in flight, or without waiting on the commit's mbarrier |
| `tmem-write-before-commit` | a TMEM store overwrote an MMA's destination without synchronizing with it |
| `tmem-dealloc-before-commit` | an allocation released before its MMAs were synchronized with |
| `tmem-uninitialized-read` | a load or an accumulating MMA read TMEM cells never written |
| `tmem-unallocated` | an MMA, load or store on columns that are not allocated |
| `tmem-leak` | columns still allocated when the kernel returned. Under `warplens run` the launch then fails with `cudaErrorTensorMemoryLeak`, as on a B200 |
| `tmem-alloc-increasing` | a `tcgen05.alloc` larger than the one before it, which the PTX contract rules out and a B200 allocates |
| `tmem-dealloc-repeated` | a second `tcgen05.dealloc` of an allocation already released; the repeat is ignored |
| `clc-canceled-cluster-lost` | a cluster finished having canceled another whose coordinate no thread read with `get_first_ctaid`: nobody did that cluster's work |
| `clc-not-waited` | a CTA returned while its `clusterlaunchcontrol.try_cancel` request was in flight |
| `clc-response-unsynchronized` | a cancel response was read, or overwritten by another request, by a warp that had not waited on the request's mbarrier |
| `clc-query-not-a-response` | `query_cancel` was given a value that no `try_cancel` wrote; it is treated as a failed request |
| `clc-ctaid-of-failed-cancel` | `get_first_ctaid` of a failed response, which the PTX manual leaves undefined; it reads as zero. The CuTe DSL's `clc_response` does this once per kernel |
| `clc-try-cancel-after-failure` | `try_cancel` after a thread of the CTA saw a request fail; undefined in the manual, it fails again here |
| `clc-multicast-to-exited-cta` | a multicast request after a CTA of the cluster returned |
| `tcgen05-not-waited` | the kernel returned with an MMA or a commit in flight |
| `mma-operand-overwritten` | shared memory written while an MMA was reading it |
| `mma-swizzle-mismatch` | an operand's descriptor names a different swizzle than the TMA load that wrote the buffer |
| `mma-tf32-operand-not-converted` | a `kind::tf32` operand has bits below TF32 precision; the MMA drops them, toward zero, as a B200 does |

Two of these report the simulator's uncertainty more than a bug in the kernel:

- **`tmem-dealloc-repeated`.** The book's `hgemm_v7` to `v9` release their
  tensor memory once per warpgroup, and they run on real hardware: on a B200
  they finished with the outputs warplens computes. A probe in which a single
  warp releases the same allocation twice was ended by the same GPU
  (`cudaErrorLaunchFailure`), so the repeat is not always harmless; warplens
  ignores it and says so ([b200-rental-1.md](b200-rental-1.md)).
- **`mma-tf32-operand-not-converted`.** A B200 drops the low mantissa bits of
  a TF32 operand, toward zero, and warplens does the same
  ([numerics.md](numerics.md)); the finding is there because a kernel that
  meant rounding to nearest has to convert its operands first.

Known gaps in the checkers: synchronization is
tracked per warp, not per thread; races between CTAs are not looked for; the
generic-to-async proxy rule for MMA operands in shared memory is not checked.
[architecture.md](architecture.md) has the details, unit by unit (sections 3
to 6).

### 6.5 The bank-conflict report

Shared memory has 32 banks of 4 bytes. For each shared-memory instruction,
`stats.smem` records how many wavefronts its executions needed against the
minimum for that many distinct words:

<!-- test: run -->
```bash
python examples/01_smem_transpose/run.py
```

<!-- test: output head=11 -->
```text
== linear: output matches NumPy, 4096 bytes of shared memory per CTA
Shared-memory bank conflicts (32 banks x 4 bytes)
  line  instruction                  execs  wavefronts  ideal  conflicts   worst
    48  st.shared.b32 [%r12], %r7;      64          64     64          0    none
    54  ld.shared.b32 %r17, [%r16];     64        2048     64       1984  32-way

== padded: output matches NumPy, 4224 bytes of shared memory per CTA
Shared-memory bank conflicts (32 banks x 4 bytes)
  line  instruction                  execs  wavefronts  ideal  conflicts  worst
    90  st.shared.b32 [%r11], %r7;      64          64     64          0   none
    95  ld.shared.b32 %r15, [%r14];     64          64     64          0   none
...
```

`wavefronts` is the largest number of distinct words in one bank, summed over
executions; `ideal` is the minimum for that many words; `conflicts` is the
difference. On the real Ampere GPU the conflict counters of this example match
Nsight Compute exactly. The hardware's total wavefront counter is 4160 higher
in every layout: it also counts two wavefronts per warp and one per CTA per
`__syncthreads()`, which are not shared accesses (`validation/README.md`). `stats.smem.wavefronts` and `stats.smem.conflicts`
are the totals.

**The same as a picture.** `warplens.viz.smem.bank_heatmap_svg(stats.smem)`
returns an SVG with a row per shared-memory instruction and a column per bank.
The colour of a cell shows conflict, not load. It compares the most distinct
words one execution of the instruction put in that bank, which is the number
of wavefronts the bank alone demanded, with that execution's ideal. A cell
within the ideal has the lightest shade: every cell of a conflict-free scalar
access, and also every cell of a warp that loads 16 bytes per lane, which puts
four words in each bank and needs four wavefronts in any layout. A darker cell
exceeds the ideal and shows its number of words: the linear transpose's load
shows 32 in every bank. Beside the grid are the wavefronts the worst execution
needed and its ideal ("32 needed, ideal 1"), the conflict that makes, and the
extra wavefronts over all executions. (The words summed over all executions
would not tell a conflict from none: each warp of the linear load hits a
different bank, so every bank ends up with the same total as in the
conflict-free store.) Banks an instruction never touched are grey. Each cell
has a tooltip with the numbers. [The gallery](gallery.md) reads one of these
pictures step by step.

<!-- test: run -->
```python
import numpy as np
import warplens
from warplens.viz import smem

sim = warplens.Device("B200")
data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
stats = sim.launch("examples/01_smem_transpose/transpose.ptx", "transpose_linear", grid=2, block=(32, 32),
                   args=[sim.from_numpy(data), sim.empty(data.shape, "f32")])
with open("banks.svg", "w") as f:
    f.write(smem.bank_heatmap_svg(stats.smem))
for e in stats.smem.instructions.values():
    print(e.line, e.op, "most words in a bank in one execution:", e.bank_peak.max())
```

<!-- test: output -->
```text
48 st most words in a bank in one execution: 1
54 ld most words in a bank in one execution: 32
```

### 6.6 Traces and views

**Instruction trace.** `launch(..., trace=callback)` calls
`callback(warp, pc, instruction, active_mask)` before every instruction a warp
executes. `warp.cta.id` and `warp.warp_id` say who; `instruction.name`,
`.text` and `.line` say what. `examples/00_divergent/run.py` prints one warp's
active mask as its lanes leave a data-dependent loop:

<!-- test: run -->
```bash
python examples/00_divergent/run.py
```

<!-- test: output head=20 -->
```text
  0 ################################  ld.param.b64 %rd1, [collatz_param_0];
  1 ################################  ld.param.b64 %rd2, [collatz_param_1];
  2 ################################  ld.param.b32 %r2, [collatz_param_2];
  3 ################################  mov.u32 %r3, %ctaid.x;
  4 ################################  mov.u32 %r4, %ntid.x;
  5 ################################  mov.u32 %r5, %tid.x;
  6 ################################  mad.lo.s32 %r1, %r3, %r4, %r5;
  7 ################################  setp.ge.s32 %p1, %r1, %r2;
  8 ................................  @%p1 bra $L__BB0_5;
  9 ################################  cvta.to.global.u64 %rd3, %rd1;
 10 ################################  mul.wide.s32 %rd4, %r1, 4;
 11 ################################  add.s64 %rd5, %rd3, %rd4;
 12 ################################  ld.global.b32 %r9, [%rd5];
 13 ################################  setp.lt.u32 %p2, %r9, 2;
 14 ################################  mov.b32 %r10, 0;
 15 #.......#.......#.......#.......  @%p2 bra $L__BB0_4;
 16 .#######.#######.#######.#######  mov.b32 %r10, 0;
 17 .#######.#######.#######.#######  and.b32 %r6, %r9, 1;
 18 .#######.#######.#######.#######  setp.ne.b32 %p3, %r6, 0;
 19 .#######.#######.#######.#######  mad.lo.s32 %r7, %r9, 3, 1;
...
```

**Where an MMA result lands in tensor memory.** `python -m warplens.viz.tmem M N`
prints the layout for a `tcgen05.mma` of that shape, `--svg FILE` draws it,
`--lane-offset 16` shows the second position of an M = 64 result, and `--pair`
shows a two-CTA (`cta_group::2`) MMA, for which `--svg` draws the tensor memory
of the even-ranked and of the odd-ranked CTA side by side:

<!-- test: run -->
```bash
python -m warplens.viz.tmem 64 32
python -m warplens.viz.tmem 128 16 --pair
```

<!-- test: output -->
```text
TMEM layout of a 64 x 32 tcgen05.mma result (cta_group::1, Layout F, lane offset 0)
  result column n is TMEM column n: 32 columns of 32 bits
  lanes      warp  holds
    0-15     0     result rows 0-15
   16-31     0     not written
   32-47     1     result rows 16-31
   48-63     1     not written
   64-79     2     result rows 32-47
   80-95     2     not written
   96-111    3     result rows 48-63
  112-127    3     not written
TMEM layout of a 128 x 16 tcgen05.mma result (cta_group::2, Layout B)
  CTA    lanes      holds
  even   0-63       rows 0-63, columns 0-7 in TMEM columns 0-7
  even   64-127     rows 0-63, columns 8-15 in TMEM columns 0-7
  odd    0-63       rows 64-127, columns 0-7 in TMEM columns 0-7
  odd    64-127     rows 64-127, columns 8-15 in TMEM columns 0-7
```

`python -m warplens.viz.tmem --transfer 16x128b 1` prints which TMEM cells
each thread's registers reach in a `tcgen05.ld` or `tcgen05.st` of that shape.

**Where a tile sits in shared memory.** `python -m warplens.viz.smem ROWS
ROW_BYTES` shows a tile of that many rows of that many bytes per 4-byte word:
the bank each word falls in, and whether a warp that reads down a column, one
word per lane, conflicts. `--swizzle 32B|64B|128B` places the tile as a TMA
copy with that swizzle does, `--pitch BYTES` pads the rows of an unswizzled
tile, and `--start` is its shared-memory offset.

<!-- test: run -->
```bash
python -m warplens.viz.smem 32 128 --swizzle 128B --svg tile.svg
```

<!-- test: output head=6 -->
```text
Tile of 32 rows x 128 bytes at shared offset 0, TMA swizzle 128B
  32 rows of 32 words (4 bytes each) at shared offsets 0-4095; a warp reading down a column, one word per lane, has a 4-way bank conflict
  row  offset of its first word  bank of each word
    0                         0   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    1                       144   4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11 20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27
    2                       288   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23
...
```

The linear and the padded tile of `examples/01_smem_transpose`, for comparison:

<!-- test: run -->
```bash
python -m warplens.viz.smem 32 128 | sed -n 2p
python -m warplens.viz.smem 32 128 --pitch 132 | sed -n 2p
```

<!-- test: output -->
```text
  32 rows of 32 words (4 bytes each) at shared offsets 0-4095; a warp reading down a column, one word per lane, has a 32-way bank conflict
  32 rows of 32 words (4 bytes each) at shared offsets 0-4219; a warp reading down a column, one word per lane, has no bank conflict
```

`--svg FILE` writes the picture: on the left the
tile as the program indexes it, on the right shared memory itself in lines of
128 bytes (one word per bank), every word coloured by its bank, light for bank
0 to dark for bank 31, and the first word of each row outlined in both. A
column of the left picture that is one shade is a column whose words share a
bank. At most 32 rows are drawn. The TMA swizzle moves 16-byte chunks, so it
spreads a column of 4-byte words over 8 banks, not 32: the 4-way conflict
above. From Python, `smem.tile_words(...)` gives the offsets and
`smem.layout_svg(words)` draws any array of word offsets, which is how to look
at a layout TMA did not make (for the example's `tile[y][x ^ y]`:
`4 * (32 * y + (x ^ y))`).

The views know the layouts of TMA tiles and of tiles you describe. They do not
recover the layout a kernel computes with its own index arithmetic.

**One TF32 conversion, bit by bit:**

<!-- test: run -->
```python
from warplens.numerics import explain_tf32

print(explain_tf32(0.1))
```

<!-- test: output -->
```text
float32 -> TF32 with cvt.rna.tf32.f32
              sign exponent fraction kept  fraction discarded
  float32     0 01111011 1001100110 0110011001101   0.100000001
  TF32        0 01111011 1001100110 0000000000000   0.0999755859
  discarded bits 0110011001101 (3277 of 8192 of the last kept place): rounded down; error -2.44155526e-05 (0.000244 relative)
```

**Timeline of a timed launch.** A launch with `timing=True` can write a trace
for [ui.perfetto.dev](https://ui.perfetto.dev) and a self-contained HTML
report (next section). The trace has a process per CTA, a track per warp with
run and wait slices, tracks for the TMA engine and the tensor core, tracks
for the time this SM's share of HBM and of L2 was busy, and a counter of TMA
bytes in flight. Each mbarrier has a track of its own, named by its shared
offset, with an instant per completed phase. A wait names the instruction it
waits at and, for an mbarrier, which phase of which barrier it waits for; a
TMA load names the barrier it reports to.

`timeline_svg(stats.timing)` draws the same for the first CTA as one picture:
a row per warp, engine and mbarrier, a numbered diamond for each phase
completion, and a line from it to the warps whose wait it ended. The barriers
are lettered by shared offset, so `B2` on a wait bar is "phase 2 of mbarrier
B". [The gallery](gallery.md) reads two such pictures.

Section 6.8 puts all of these on one page.

### 6.7 The timing report, and how far to trust it

<!-- test: run -->
```python
import numpy as np
import warplens
from warplens.timing.config import GpuConfig

# examples/04_pipelined_gemm: one CTA, 16 tiles along K, a 4-stage TMA pipeline.
M, N, K, stages = 128, 64, 1024, 4
rng = np.random.default_rng(0)
a = rng.uniform(-1, 1, (M, K)).astype(np.float16)
b = rng.uniform(-1, 1, (N, K)).astype(np.float16)


def run(**timing):
    sim = warplens.Device("B200")
    map_a = sim.tensor_map_tiled(sim.from_numpy(a), (128, 64), swizzle="128B")
    map_b = sim.tensor_map_tiled(sim.from_numpy(b), (64, 64), swizzle="128B")
    d = sim.empty((M, N), "f32")
    return sim.launch("examples/04_pipelined_gemm/gemm.ptx", "pipelined_gemm", grid=1, block=128,
                      args=[map_a, map_b, d, N, K // 64, stages], **timing).timing


report = run(timing=True, active_sms=8)
print(report.summary())
report.write_trace("gemm.trace.json")  # open at ui.perfetto.dev
report.write_html("gemm.html")         # the same numbers, a timeline, and every config entry

faster = run(timing=GpuConfig.load("B200").replace(tma_bytes_per_clk=512), active_sms=8)
print(f"\nwith a TMA engine 4x faster: {faster.time_ns:,.0f} ns, {faster.bottleneck}-bound")
```

<!-- test: output -->
```text
Approximate timing on B200 (model, not a measurement)
  parameters: illustrative parameters, 8 of 13 assumed and none fitted to hardware
  1 CTA(s), each alone on one SM; 8 SMs assumed to share HBM
  kernel time 2,611 ns (3,917 clocks at 1.5 GHz, an assumed clock)
  verdict: TMA engine-bound: the TMA engine is busy 91% of the time
  resource       busy   derived from
  issue slots     15%   clock_ghz*
  SMEM             0%   smem_wavefront_clk*, clock_ghz*
  TMA engine      91%   tma_bytes_per_clk*, tma_latency_ns*, clock_ghz*
  HBM             34%   hbm_bandwidth_tb_s, active_sms (launch parameter)
  L2               0%   l2_bandwidth_per_sm_gb_s*
  Tensor Core     57%   tensor_core_tflops_f16, tensor_core_tflops_tf32, sm_count, mma_latency_clk, clock_ghz*
  roofline: 7.48 TFLOP/s pipeline rate per SM; compute ceiling 13.04, memory ceiling 22.09 at 42.7 FLOP/byte
  * assumed value in this model; see the config file for every source

with a TMA engine 4x faster: 1,819 ns, Tensor Core-bound
```

How to read it:

- `parameters` says where the config's numbers come from. The configs that
  ship with warplens are *illustrative*: plausible values, most of them
  assumed. A config written by the fit of `validation/blackwell_timing`
  (`--timing-config validation/blackwell_timing/measured/b200/calibrated.yaml`)
  is labelled an *experimental fit*, with the number of fitted entries. The
  HTML report, the page of `warplens report`, the timeline and the trace carry
  the same label. Either way the result is simulated and approximate, never
  hardware evidence: what was measured on a B200 is in
  [b200-rental-1.md](b200-rental-1.md).
- `kernel time` is the simulated duration of the slowest CTA. Each CTA is
  simulated alone on one SM, starting at zero.
- `busy` is the fraction of time a resource was working: for the TMA engine,
  HBM, L2 and the tensor core, of the interval in which asynchronous work was
  in flight; for issue slots and shared memory, of the whole kernel. It is
  averaged over the CTAs that did work, meaning they used shared memory, a
  TMA or `cp.async` copy, or the tensor core. CTAs that start and return (a
  persistent kernel on a small problem) are left out, and the summary then
  says how many; `report.utilization_all_ctas` has the average over all of
  them. If no CTA did such work, all are averaged.
- The verdict names a resource that is busy at least 85% of the time;
  otherwise the kernel is called latency-bound. The threshold is a choice.
- `active_sms` is how many SMs are assumed to share the memory bandwidth.
  The default is all 148, as when a problem fills the device; a small
  experiment is closer to 1 to 8. It changes the HBM share and with it,
  often, the verdict.
- `report.time_ns`, `.utilization`, `.bottleneck`, `.verdict`, `.roofline()`
  and `.pipeline_ns` give the numbers; `.slices` and `.instants` are the
  timeline.

**How far to trust it.** Not as a prediction. The model has six resources,
each a first-come-first-served server, and these parameters:

<!-- test: run -->
```python
from warplens.timing.config import GpuConfig

for p in GpuConfig.load("B200").params.values():
    kind = "assumed" if p.is_assumption else "published"
    print(f"{p.key:<26}{p.value:>8g} {p.unit:<10}{kind}")
```

<!-- test: output -->
```text
sm_count                       148 SMs       published
clock_ghz                      1.5 GHz       assumed
hbm_capacity_gb                192 GB        published
hbm_bandwidth_tb_s           4.141 TB/s      published
l2_bandwidth_per_sm_gb_s       400 GB/s      assumed
tma_latency_ns                 100 ns        assumed
clc_latency_ns                 100 ns        assumed
tma_bytes_per_clk              128 bytes/clk assumed
smem_wavefront_clk               1 clk       assumed
global_access_latency_clk       30 clk       assumed
tmem_access_clk                  4 clk       assumed
tensor_core_tflops_f16      1929.6 TFLOPS    published
tensor_core_tflops_tf32      964.5 TFLOPS    published
mma_latency_clk               11.3 clk       published
```

The published values are from Jarmusch and Chandrasekaran, "Microbenchmarking
NVIDIA's Blackwell Architecture" ([arXiv:2512.02189](https://arxiv.org/abs/2512.02189));
the assumed ones are placeholders, and they include the clock, which scales
most of the others.

These values have been compared with a GPU once and were not changed by it.
On the B200 run of 2026-11-25, 14 benchmarks measured the mechanisms the
parameters stand for. With the config above, the model is within 10% of the
measurement for 5 of 152 benchmark configurations and within a factor of two
for 53. A fit of the parameters to that run is recorded
(`validation/blackwell_timing/measured/b200/calibrated.yaml`, usable with
`--timing-config`); with it the counts are 45 and 104. It is not the default,
for reasons [b200-rental-1.md](b200-rental-1.md) gives ("The timing config is
a decision, not a fix"): the fit itself shows where one number per mechanism
cannot follow the GPU (the latency of cluster launch control differs by form
of request, an mbarrier arrive and wait costs 33 ns where the model charges
about 3), and the examples in this guide were written for the assumed values.
The fit is of one day's measurements on one machine; a second set, taken on
the same machine after a restart, repeats the first to within 1 % for most
configurations, and the fit predicts it as well as its own
([b200-rental-2.md](b200-rental-2.md)). Version 1 ships the
assumed values (the decision is recorded in [PLAN.md](../PLAN.md)).

What the model leaves out is in section 8. What it is good for is the question the example asks:
with these parameters, is this pipeline limited by the copies, the memory
bandwidth or the tensor core, and what happens to that when a parameter or the
pipeline depth changes? Treat an answer as a hypothesis to test on hardware.
`python examples/04_pipelined_gemm/run.py out/` runs that comparison for 1, 2
and 4 stages in three scenarios.
[architecture.md](architecture.md#7-the-timing-model) describes the model.

### 6.8 Everything on one page: `warplens report`

`warplens report program` runs a program exactly as `warplens run` does
(section 5.2), with the timing model on, and writes a report of every kernel
it launched into a directory: `index.html`, and beside it every picture as an
SVG file, plus the timing report and the Perfetto trace of each launch.

<!-- test: run needs=triton -->
```bash
warplens report --quiet -o triton_report examples/05_triton/run.py 2>&1 | grep -c "wrote .*triton_report/index.html"
ls triton_report | sed "s/^p[0-9]*_//"
```

<!-- test: output -->
```text
1
01_add_kernel.timeline.svg
01_add_kernel.timing.html
01_add_kernel.trace.json
02_matmul_kernel.banks.svg
02_matmul_kernel.timeline.svg
02_matmul_kernel.timing.html
02_matmul_kernel.tmem_1.svg
02_matmul_kernel.trace.json
index.html
```

For each launch the page has, in this order:

| Part | What it shows | Also written as |
| --- | --- | --- |
| Findings | what the checkers reported (6.4), each with its PTX line | |
| Shared-memory banks | the heatmap and the table of 6.5 | `NAME.banks.svg` |
| Shared-memory layout | one picture (6.6) per shape of tile that a TMA tensor copy loaded or stored, with the PTX lines and shared offsets of the copies | `NAME.smem_layout_N.svg` |
| Tensor memory | the TMEM layout (6.6) of each shape of `tcgen05.mma` the launch issued, both CTAs for a `cta_group::2` MMA | `NAME.tmem_N.svg` |
| Timeline and mbarrier phases | the timeline of the first CTA with a row per mbarrier, each phase completion joined to the waits it ended, and the verdict | `NAME.timeline.svg`, `NAME.timing.html` (6.7), `NAME.trace.json` |

`NAME` is the launch's number and the kernel's name, cut to a safe file name;
launches made by a Triton driver in another process also carry that process's
id (`p1234_`). A part with nothing to show says so: a launch without TMA has no
layout picture, one without an MMA has no TMEM picture.

Options: `-o DIRECTORY` (default `warplens_report`), `--no-timing` to leave the
timing model off, which drops the timeline, and the options of `warplens run`
(`--model`, `--timing-config`, `--active-sms`, `--clc-limit`, `--quiet`). The
exit status is the program's. If the program launches nothing, no page is
written.

The timeline, verdict and timing report come from the approximate timing
model and say which parameters they used; section 6.7 says how far to trust
them. The other parts do not depend
on it.

A script that launches through the Python API is not a program `warplens
report` can observe; it writes the same page itself:

<!-- test: run -->
```python
import numpy as np
import warplens
from warplens.viz import write_report

sim = warplens.Device("B200")
data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
launches = []
for layout in ("linear", "padded"):
    stats = sim.launch("examples/01_smem_transpose/transpose.ptx", f"transpose_{layout}", grid=2,
                       block=(32, 32), args=[sim.from_numpy(data), sim.empty(data.shape, "f32")], timing=True)
    launches.append((f"transpose_{layout}", stats))
index = write_report("transpose_report", launches, title="Two transposes")
print(*sorted(p.name for p in index.parent.iterdir()), sep="\n")
```

<!-- test: output -->
```text
01_transpose_linear.banks.svg
01_transpose_linear.timeline.svg
01_transpose_linear.timing.html
01_transpose_linear.trace.json
02_transpose_padded.banks.svg
02_transpose_padded.timeline.svg
02_transpose_padded.timing.html
02_transpose_padded.trace.json
index.html
```

`write_report(directory, [(name, stats), ...], model="B200", title=...)` takes
the `LaunchStats` of any launches, timed or not.

## 7. Execution, numerics and the two models

### 7.1 How a kernel is executed

The short version; [architecture.md](architecture.md) is the reference, unit
by unit, and [scheduling.md](scheduling.md) is about the order warps run in.

- **PTX is parsed, never translated.** Each function becomes a control-flow
  graph. An opcode warplens does not implement is refused when the kernel is
  launched, before anything runs; an unknown modifier when its instruction
  executes.
- **A warp is the unit of execution.** One instruction runs on all 32 lanes
  at once, under an active mask. When lanes disagree at a branch the warp
  splits, and the sides rejoin at the branch's immediate post-dominator.
  Registers hold raw bits and each instruction chooses how to read them.
- **Warps take turns.** In an untimed launch every warp that can run executes
  one instruction per round, round-robin by default; `schedule=` chooses a
  seeded or an adversarial order. A warp that waits (at a barrier, on an
  mbarrier, for tensor memory) is skipped until what it waits for happens. If
  nothing can run and nothing is pending, the launch stops with a
  `DeadlockError` that says who waits on what.
- **TMA copies and MMAs are asynchronous.** The instruction issues the work
  and returns; the work completes later (`async_delay` rounds later in an
  untimed run, when the timing model says in a timed one) and then signals
  its mbarrier. The checkers do not depend on when that is: they track which
  completions a warp has synchronized with, and report a use outside that
  set even if the data happened to be there.
- **CTAs of a cluster run together; clusters run one after another.** Without
  `cluster=`, each CTA is a cluster of one, so CTAs run in turn.
- **Memory.** Global allocations are separated by unmapped gaps, so a stray
  access faults. Shared memory is per CTA, with 32 banks of 4 bytes. Tensor
  memory is 128 lanes by 512 columns of 32 bits per CTA.
- **A timed launch** (`timing=True`) runs the same kernel with the same
  results and findings, in the order the timing model's clock gives.

Other topics have a reference of their own: atomic updates
([atomics.md](atomics.md)) and cluster launch control ([clc.md](clc.md)).

### 7.2 Numerical behaviour

[numerics.md](numerics.md) is the reference: each rule, and whether it was
checked against a GPU, against exact arithmetic, or not at all. In outline:

- **Scalar arithmetic** is IEEE round-to-nearest-even, done by NumPy. `fma`
  rounds once. Integer arithmetic wraps; division by zero gives all ones and
  a finding.
- **NaN bit patterns and `.ftz`** are what an RTX 3080 Ti returned, bit for
  bit, and the same programs gave the same lines on the B200. PTX does not
  define them.
- **TF32.** `cvt.rna.tf32.f32` matches the RTX 3080 Ti on 715 recorded
  inputs. `cvt.rn` and `cvt.rz` to TF32 exist only on Hopper and newer and
  are as measured on the B200.
- **`tcgen05.mma` is not a chain of float32 additions.** One MMA computes
  each element as one sum: the K products and the old accumulator are lined
  up under the largest term, cut off toward zero 25 bits down, added exactly,
  and cut toward zero to float32. `kind::tf32` drops the low 13 bits of its
  operands. This is what a B200 was measured to do; the simulator reproduces
  the two recordings word for word, and with it the book's nine kernels
  replayed bit for bit against that GPU. A NumPy `A @ B.T` in float32
  therefore differs from the simulator, as it does from the GPU, in the last
  bits: compare with a tolerance.
- **What is not known.** Three rules of that sum were inferred from whole
  kernels and not yet probed directly; operands that are subnormal in half
  precision, and elements with an infinity or a NaN among their terms, were
  not measured. Bit-exact agreement of this model with itself does not by
  itself establish bit-exact agreement with a GPU.
- **FP8, FP4 and block-scaled formats** exist as `cvt` conversions and, on
  the host, in `warplens.numerics`. They are not MMA kinds here.

### 7.3 Models: B200 and B300

`Device("B200")` and `Device("B300")`; `warplens run --model B300`;
`warplens.triton_driver.install("B300")`. The model decides two things:

- **What the program is told.** Under `warplens run` a B200 reports compute
  capability 10.0 and a B300 reports 10.3, so TIRx compiles for `sm_100a` or
  `sm_103a`, and the CuTe DSL is pointed at the same. A B200 reports 148 SMs
  and a B300 160, the `sm_count` of its timing configuration.
  The Triton driver compiles for `sm_100a` for both models.
- **Which timing configuration is loaded**: `warplens/configs/b200.yaml` or
  `b300.yaml`.

<!-- test: run needs=tirx,shim -->
```bash
warplens run --model B300 python examples/07_tirx/run.py
```

<!-- test: output -->
```text
the last thread is 63 of block 3
saxpy over 256 elements on compute capability 10.3: matches NumPy
warplens: 1 kernel launch(es) simulated on B300, 0 finding(s)
  saxpy_kernel: 4 CTA(s), 159 warp-instructions
```

The functional simulator is the same for both: the same instructions, the same
limits. The B300 timing configuration differs from the B200 one where NVIDIA
publishes a different figure: 160 SMs instead of 148, 288 GB of memory
instead of 192, and 1,400 W instead of 1,200. Only the SM count reaches the
timing model: one SM's share of the HBM bandwidth (with the default
`active_sms`) and of the tensor core rate is 148/160 of a B200 SM's, so
**timing results for the two models differ by that much and no more**. Every
other entry is the B200 value, marked "assumed equal to B200, uncalibrated".
Nothing in it was measured on a B300, and the file says so entry by entry.

## 8. Limits

What is not supported or not modeled, in one place. The authoritative list of
PTX is [coverage.md](coverage.md); anything not in it raises
`UnsupportedError`.

**Input**

- PTX only. Machine code cannot be simulated.
- A kernel's `call`s must be to device functions defined in the same module
  (they are inlined) or to `vprintf`. An indirect call can only reach
  functions of the module, and recursion stops with an error at 1024 active
  calls per thread, whatever stack size a GPU would have. No `.tex`.

**Instructions not implemented** (the common ones; see coverage.md)

- Integer `.sat` and carry forms, `prmt` modes other than the default,
  `brx.idx`, `bar.red`, `shfl.sync` on anything but `.b32`, rounding modes
  other than round-to-nearest for floating-point arithmetic.
- TMA: im2col mode, interleaved layouts, `cp.reduce.async.bulk`,
  `tensormap.replace`, multicast on a raw `cp.async.bulk`. L2 promotion and
  cache hints are accepted and ignored.
- mbarrier: `arrive_drop`, `.noComplete`, `pending_count`. `try_wait` blocks
  until the phase completes instead of returning false, so a wait that can
  never succeed is a deadlock report.
- `tcgen05`: operand A from tensor memory, kinds other than `f16` and `tf32`
  (no bf16, integer, FP8/FP6/FP4 or block-scaled MMAs), MN-major TF32
  operands, float16 accumulators, sparsity, the `16x32bx2` transfer shape,
  `.pack`/`.unpack`, `tcgen05.cp`, `tcgen05.shift`, `.exclusive` allocation,
  a non-zero base offset in a shared-memory descriptor,
  partial commits. The FP8/FP4 formats exist as `cvt` conversions and in
  `warplens.numerics`, not as MMA kinds.

**Semantics that are approximations**

- CTAs of different clusters run one after another, so races between them on
  global memory are not explored. CTAs within a cluster do interleave.
- An MMA element is computed as a B200 was measured to compute it (7.2).
  Three rules of that sum are inferred, not probed, and subnormal
  half-precision operands and elements with an infinity or NaN among their
  terms were not measured: for those, agreement with a GPU is to float32
  rounding at best.
- Of the six cases in which a B200 ended a kernel that the simulator let
  return, four are errors now and a tensor-memory leak fails the launch under
  `warplens run`. The sixth, a repeated `tcgen05.dealloc`, stays a finding
  (6.4). Three more tensor-memory behaviours of the GPU are not
  modeled: allocations the PTX contract rules out and the GPU makes, transfers
  past a warp's lane window, and the contents of tensor memory never written
  (section 2).
- Shuffles do not rendezvous lanes on separate divergent paths. Per-thread
  barrier semantics in divergent code are reported, not modeled.
- TMEM loads and stores complete at once; `tcgen05.wait::ld/st` checks
  participation and models no delay.
- Cluster-wide addresses use warplens's own encoding, and so do the tensor
  maps warplens makes. A tensor map in the driver's own layout is read as
  well (the layout was worked out from a B200's encoder; `numerics` aside,
  it is the one thing here taken from hardware bytes alone): section 9, CuTe
  DSL.
- Misaligned loads and stores are allowed.
- Lanes of one warp whose `atom` names the same bytes are applied in lane
  order, lowest first. A GPU's order is not specified; results that depend
  on it (which old value each thread receives, the word an `atomicExch`
  leaves, and floating sums) may differ from a GPU's. Integer sums,
  minima/maxima, bit masks and the set of values an integer counter hands out
  do not depend on this lane order.

**`warplens run`**

- Linux only. Compiled programs need the build flags of 5.2.
- The options (`--timing-config`, `--active-sms`, `--clc-limit`) are for the
  whole run: a program cannot give one launch a different value.
- The shim implements about sixty runtime and driver functions: what the four
  CUTLASS tutorials, TVM's runtime and the CuTe DSL call. A program that calls
  anything else fails to start with an unresolved symbol, or reaches the real
  runtime if one is installed. Not there: multiple devices, unified memory,
  textures, `cudaMemcpy2D`. Streams and events are tokens; everything is
  synchronous.
- A CUDA runtime linked statically into the program, or into a library it
  loads, cannot be replaced. That rules out PyTorch and cuda-python's runtime
  module.
- Triton: no torch tensors, no benchmarking interface, no kernels that need a
  global or profile scratch buffer, no host tensor-descriptor arguments. Some
  multi-CTA Triton matmuls emit `tcgen05.st.16x32bx2` and fail on it.
- CuTe DSL: the tutorial that uses TMA (`fp16_gemm_0.py`) does not run as a
  script, because its arrays are PyTorch's; its kernel does (section 9).

**Timing model**

- The parameters that ship are assumed, not calibrated; a fit to one B200
  run is recorded and not applied (6.7).
- One parameter per mechanism. The B200 measurements show two places where
  that is not enough: cluster launch control (236 ns for a failed request in
  one cluster, 423 to 493 ns for the other forms) and the cost of an mbarrier
  arrive and wait.
- CTAs do not interact in time; each starts at zero on its own SM.
- L2 has no capacity or eviction: a byte fetched once counts as cached, and
  so does a byte a TMA store wrote. A store to lines L2 does not hold is
  charged to HBM; the later write-back of cached lines is not modeled.
- A global load costs a fixed wait and a store nothing. An `atom` is charged
  as a load followed by a store of its memory (assumed). TMEM access and
  `tcgen05.wait` have a fixed cost. `cp.async` is charged at the TMA engine's
  rates, and its source bytes to L2 or HBM as a TMA load's are.
- An accumulating MMA's dependence on the one before it is not modeled beyond
  queueing.
- In a cluster, a multicast load is charged to every receiving SM, and a
  two-CTA MMA to the issuing SM only.

## 9. Troubleshooting

**"the program carries no PTX, only machine code for a real GPU"** / **"the
program's PTX is compressed"** / **"the program launched no kernels"**. The
program was not built with the flags of 5.2. The first is what a build for
`code=sm_100a` gives:

<!-- test: run needs=nvcc -->
```bash
nvcc -gencode arch=compute_100a,code=sm_100a --cudart shared saxpy.cu -o saxpy_machine_code
warplens run ./saxpy_machine_code || echo "exit status $?"
```

<!-- test: output -->
```text
warplens: the program carries no PTX, only machine code for a real GPU. Build it with nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared
warplens: the program launched no kernels
The program must carry PTX and use the shared CUDA runtime. Build it like this:
  nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog
exit status 70
```

Plain `nvcc -arch=sm_100a` embeds PTX but compresses it, which gives the
second message. A program linked without `--cudart shared` never reaches the
shim: `warplens run` reports only that it launched no kernels, and the program
itself runs against the real runtime if there is one (on the development
machine it then failed with a CUDA error, because an Ampere GPU cannot load
`sm_100a` code). The fix is the same in all three cases:

```text
nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog
```

**"the module is machine code for a real GPU, and the simulator runs PTX"**.
A runtime handed the driver a compiled module. With TIRx this is the default;
ask for PTX with `backend_config={"cuda": {"target_format": "ptx"}}` on the
`device_entry`, or in the target as `examples/08_book/run.py` does. Here is
the TIRx example with that setting removed:

<!-- test: run needs=tirx,shim -->
```bash
sed 's/, backend_config=PTX//' examples/07_tirx/run.py > tirx_machine_code.py
(warplens run python tirx_machine_code.py 2>&1 || true) | grep "^warplens:"
```

<!-- test: output -->
```text
warplens: kernel `saxpy_kernel`: the module is machine code for a real GPU, and the simulator runs PTX. Ask the compiler for PTX (TIRx: backend_config={"cuda": {"target_format": "ptx"}})
warplens: the program launched no kernels
```

The script itself then stops with `CUDA_ERROR_NOT_FOUND` from
`cuModuleGetFunction`.

**A TIRx or CuTe DSL script fails although it works with `python`.** Look
for this line in what `warplens run script.py` printed:

```text
warplens: running script.py without the shim libraries: its Triton kernels are simulated, but a TIRx or CuTe DSL script will not find a device
```

The shim libraries could not be built (the reason is printed above that line;
see "could not build the shim libraries" below). A Triton script does not
need them, so the script was run in the `warplens` process with only the
Triton driver, and a TIRx or CuTe DSL script then fails where it first needs
a device, for TIRx with `ValueError: CUDA compilation requires an explicit
backend_config arch or Target arch when no CUDA device is available`. With
the shim, the three kinds of script take the same command:

<!-- test: run needs=tirx,shim -->
```bash
warplens run examples/07_tirx/run.py 2>&1 | tail -n 2
```

<!-- test: output -->
```text
warplens: 1 kernel launch(es) simulated on B200, 0 finding(s)
  saxpy_kernel: 4 CTA(s), 159 warp-instructions
```

**TIRx fails to compile with an error about the architecture `sm_100a`.**
The NVRTC library that was loaded is older than CUDA 12.8 and does not know
Blackwell. This could not be reproduced on the development machine (its older
NVRTC has a different library name and is not picked up); the report is from
the write-up of the front ends ([history/m6.md](history/m6.md)). To see
which NVRTC a script will get:

<!-- test: run needs=tirx -->
```python
from pathlib import Path

from cuda.pathfinder import load_nvidia_dynamic_lib

print(Path(load_nvidia_dynamic_lib("nvrtc").abs_path).name)
```

<!-- test: output unchecked -->
```text
libnvrtc.so.13
```

It should be `libnvrtc.so.12` from CUDA 12.8 or later, or `libnvrtc.so.13`.
Installing `cuda-nvrtc` 13 into the environment (as `env/environment.yml`
does) fixes it.

**"could not build the shim libraries"**. `warplens run` compiles its shim on
first use and needs a C compiler and the CUDA headers. With no `nvcc` on
`PATH` and an empty cache, `warplens run ./saxpy` printed:

```text
warplens: building the shim libraries (first run)
warplens: could not build the shim libraries:
error: nvcc not found; pass the CUDA include directory

Run .../warplens/shim/build.sh <CUDA include directory> by hand.
```

Either put `nvcc` on `PATH` (activate the conda env), or run
`warplens/shim/build.sh /path/to/cuda/include`, where the directory is the one
that holds `cuda_runtime_api.h`. `CC` selects the compiler. A build made by
hand lands in `warplens/shim/lib/` and is used from then on, as long as it is
a build of the shim sources that are there (see the next entry). If the CuTe DSL's
runtime cannot be relinked, `warplens run` says so and continues; the DSL's
launches are then not simulated.

**"was built from other shim sources ...; rebuilding it"**. The libraries in
`warplens/shim/lib/` were built by hand before `cudart_shim.c`, `cuda_shim.c`,
`protocol.h` or `build.sh` changed (after a `git pull`, say). `build.sh`
records what each build is of, `warplens run` compares that once per run, and
a build of other sources is rebuilt in place, against the CUDA headers it was
built with. Nothing needs doing. When the rebuild is impossible the run stops
with "... and it could not be rebuilt", the compiler's output, and "The stale
libraries are not used": run `build.sh` by hand as above, or remove
`warplens/shim/lib/` to have the libraries built in the cache. A `.py` script
continues without the shim, as it does when the libraries cannot be built at
all.

**`UnsupportedError`**. The kernel uses PTX that is not implemented. The
message names the instruction and its line; [coverage.md](coverage.md) lists
what is there. Unknown opcodes are found when the kernel is launched, before
anything runs; unknown modifiers when the instruction executes.

<!-- test: run -->
```python
import warplens

ptx = """
.version 9.5
.target sm_100a
.address_size 64
.visible .entry count_bits()
{
    .reg .b32 %r1;
    popc.b32 %r1, %r1;
    ret;
}
"""
try:
    warplens.Device("B200").launch(warplens.parse_ptx(ptx, name="demo.ptx"), "count_bits", grid=1, block=32)
except warplens.UnsupportedError as error:
    print(error)
```

<!-- test: output -->
```text
kernel `count_bits` uses instructions warplens does not implement yet: popc
  first at demo.ptx:8: popc.b32 %r1, %r1;
```

There is no workaround short of implementing the instruction or changing the
source so that the compiler does not emit it.

**PyTorch tensors do not work.** PyTorch links NVIDIA's CUDA runtime
statically, so its allocations and copies never reach the shim, and it needs a
real GPU to create a CUDA tensor at all. (PyTorch is not installed in the
development environment; this is from how the shim works
([architecture.md](architecture.md#9-warplens-run-the-shim-and-the-front-ends)),
not from a run.) Use instead:

| Front end | Arrays that work |
| --- | --- |
| Python API | `sim.from_numpy(array)`, `sim.empty(shape, dtype)` |
| Triton | `gpu.tensor(array)`, `gpu.empty(shape, dtype)` from the warplens driver |
| TIRx | `tvm_tirx.runtime.tensor(array, device=tvm_tirx.cuda(0))` |
| CuTe DSL | driver memory through `cutlass.base_dsl.runtime.cuda` and `make_ptr`, as in `examples/06_cute_dsl/saxpy.py`; or any array that allocates through the shared CUDA runtime and speaks DLPack, such as TVM's |

The Triton driver refuses anything with a `data_ptr()` that is not one of its
own tensors, which is how a torch tensor looks to it:

<!-- test: run needs=triton -->
```python
import warnings

import numpy as np
import triton
import triton.language as tl
import warplens
import warplens.triton_driver as wt

warnings.filterwarnings("ignore", module="triton")

@triton.jit
def double(x_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(x_ptr + offsets, 2 * tl.load(x_ptr + offsets, mask=offsets < n), mask=offsets < n)


class ForeignTensor:  # stands in for a tensor of another library
    dtype, shape = np.dtype("float32"), (64,)

    def data_ptr(self):
        return 0x1000


gpu = wt.install("B200")
try:
    double[(1,)](ForeignTensor(), 64, BLOCK=64)
except warplens.LaunchError as error:
    print("LaunchError:", error)
finally:
    wt.uninstall()
```

<!-- test: output -->
```text
LaunchError: use a warplens tensor for simulated pointer arguments
```

**A CuTe DSL kernel that uses TMA.** The DSL does not ask the driver for its
tensor maps; its compiled host code writes the descriptor itself, in the
driver's undocumented layout. warplens reads that layout since the first
Blackwell run, from which it was worked out
([b200-rental-1.md](b200-rental-1.md)). The kernel of CUTLASS's
`fp16_gemm_0.py`, as it was captured on a B200 with its parameters and its
memory, replays with every byte of its output equal to the GPU's
(`validation/blackwell/e2e/replay_launch.py`). The script itself has not been
run under `warplens run`: it holds its arrays in PyTorch, which needs a GPU.

Bytes that are neither kind of tensor map are still an error:

<!-- test: run -->
```python
import numpy as np
import warplens

sim = warplens.Device("B200")
foreign = sim.from_numpy(np.arange(128, dtype=np.uint8))  # 128 bytes that warplens did not encode
try:
    sim.launch("examples/02_tma_copy/tma_copy.ptx", "tma_copy", grid=1, block=128,
               args=[foreign, foreign, 32, 16, 0])
except warplens.ExecutionError as error:
    print(error)
```

<!-- test: output -->
```text
the tensor-map operand points at neither a tensor map created by Device.tensor_map_tiled nor one in the driver's layout
  at tma_copy.ptx:73: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r12], [%rd1, {%r21, %r22}], [%r11];
  in kernel `tma_copy`, CTA (0, 0, 0), warp 0
  lane 0 (thread (0, 0, 0))
```

CUDA C++ programs and TIRx do not have this problem: they call
`cuTensorMapEncodeTiled`, which the shim serves.

**A launch never finishes.** Pass `max_steps=` to `Device.launch` to turn a
runaway loop into an `ExecutionError`. A kernel that waits for something that
cannot happen is reported as a deadlock without it.

**It is slow.** Shrink the problem to a few tiles. Persistent kernels that
launch one CTA per SM (148) pay for every CTA even when most find no work.

**`warplens run` reports a finding you do not believe.** Read section 6.4:
two kinds mark places where hardware behaviour is unknown, and the checkers
have documented blind spots. If a finding is wrong, it is a bug worth
reporting with the PTX (`WARPLENS_DUMP_PTX=directory warplens run ...` keeps
the PTX of modules a program loads itself, as TIRx does).

## 10. Where to read more

Tutorials:

- [examples/08_book/book_tour.ipynb](../examples/08_book/book_tour.ipynb): the
  book's GEMM kernels, chapter by chapter, in the simulator
  ([how to run it](../examples/08_book/README.md)).
- [gallery.md](gallery.md): three pictures read step by step: a bank
  conflict, a TMA copy that is read too early, a pipelined GEMM.

References, each the authority on its subject:

- [coverage.md](coverage.md): every PTX construct and instruction supported.
- [numerics.md](numerics.md): what is computed exactly and what is not; TF32;
  the MMA sum; the FP8/FP4 host-side formats.
- [architecture.md](architecture.md): how the simulator works, unit by unit.
- [scheduling.md](scheduling.md): the order warps run in, and the schedules
  that expose a race.
- [atomics.md](atomics.md): `atom` and `red`.
- [clc.md](clc.md): cluster launch control.

Hardware evidence:

- [b200-rental-1.md](b200-rental-1.md): the first run on a B200, 2026-11-25:
  what ran, what agreed, what did not, what was changed.
- [validation-matrix.md](validation-matrix.md): every claim about Blackwell
  behaviour, with the probe that tests it and its pass criterion.
- [validation/README.md](../validation/README.md): the comparisons with the
  RTX 3080 Ti. `validation/blackwell/README.md` and
  `validation/blackwell_timing/README.md` describe the two probe suites;
  what a B200 printed is under their `measured/b200/` directories.
- [hardware-capture.md](hardware-capture.md): how whole launches are captured
  on a GPU and replayed, and the format they are kept in.

The project:

- [PLAN.md](../PLAN.md): what is still owed for v1.0, and the roadmap after
  it.
- [history/](history/README.md): the plan as first written and the write-up
  of each milestone, M0 to M6. They are records and are not updated; where
  they disagree with this guide, the guide is newer.
