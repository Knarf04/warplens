# warplens

Understand a Blackwell GPU by watching it run, on your CPU.

warplens executes GPU kernels on the CPU. It takes PTX, the assembly that
`nvcc`, Triton, TIRx and the CuTe DSL all produce, compiled for datacenter
Blackwell (`sm_100a`), and runs it under a model of a B200 or B300: warps,
shared memory, TMA, mbarriers, the `tcgen05` tensor core, tensor memory and
clusters. No GPU is involved at any point. A run gives you

- the kernel's results, to compare with NumPy;
- findings from checkers for races and for misuse of the TMA, mbarrier and
  tensor-memory protocols, each with the PTX line responsible;
- a shared-memory bank-conflict report;
- on request, an approximate and uncalibrated timing model that says what
  limits the kernel under stated parameters.

## Quick start

```bash
git clone https://github.com/Knarf04/warplens && cd warplens
pip install -e ".[dev]"
```

<!-- test: run -->
```bash
python examples/00_vecadd/run.py          # an nvcc-compiled sm_100a kernel, checked against NumPy
python examples/02_tma_copy/run.py        # TMA and mbarriers, then five deliberate bugs and their reports
python examples/04_pipelined_gemm/run.py  # what limits a pipelined GEMM, under the timing model
```

That needs Python 3.11 or newer, NumPy and PyYAML, and nothing from CUDA: the
examples' PTX is checked in. In Python:

<!-- test: run -->
```python
import numpy as np, warplens

a, b = np.arange(1000, dtype=np.float32), np.ones(1000, dtype=np.float32)
sim = warplens.Device("B200")
A, B, C = sim.from_numpy(a), sim.from_numpy(b), sim.empty(1000, "f32")
stats = sim.launch("examples/00_vecadd/vecadd.ptx", "vecadd", grid=4, block=256, args=[A, B, C, 1000])
assert np.array_equal(C.numpy(), a + b) and not stats.findings
```

Unmodified programs run too: `warplens run ./program` for a CUDA C++ program
built to carry PTX, `warplens run script.py` for Triton, and
`warplens run python script.py` for TIRx and the CuTe DSL.

## Documentation

- **[docs/guide.md](docs/guide.md)**: the user guide. Installation, every way
  to get code in, how to read the results, limits, troubleshooting.
- [notebooks/book_tour.ipynb](notebooks/book_tour.ipynb): the nine GEMM kernels
  of the book *Modern GPU Programming for MLSys*, chapter by chapter, in the
  simulator. It runs them from your own clone of the book.
- [docs/coverage.md](docs/coverage.md): the PTX that is supported.
  [docs/numerics.md](docs/numerics.md): what is computed exactly.
- [validation/README.md](validation/README.md): the comparisons with a real GPU.
- [PLAN.md](PLAN.md): goals, architecture and what is still owed.
  [docs/m0.md](docs/m0.md) to [docs/m6.md](docs/m6.md) record the design
  decisions of each milestone.

## Status

Version 0.0.1; not on PyPI. What runs today, unmodified and matching its
reference: the first four CUTLASS Blackwell tutorials (the last two on
clusters of 16 CTAs, with TMA multicast and a two-CTA MMA), a Triton matmul,
and the nine GEMM kernels of the book, which are written in TIRx. CuTe DSL
kernels run unless they use TMA.

What that rests on:

- **Checked against a real GPU** (an RTX 3080 Ti, Ampere): everything Ampere
  shares with Blackwell that has a recorded comparison. Shuffles, barriers,
  `printf`, calls to device functions, `ldmatrix`, `bfe`, `prmt`, `cp.async`,
  `.ftz` arithmetic and the float32-to-TF32 conversion match exactly, and so
  do the shared-memory bank-conflict counters.
- **Not checked against any hardware**: everything specific to Hopper and
  Blackwell. TMA, tensor maps, mbarriers, `tcgen05`, tensor memory and
  clusters follow the PTX specification, the CUDA headers and the CUTLASS
  sources, and the evidence for them is that real kernels from three
  independent code generators produce correct output. Nothing has run on a
  Blackwell GPU.
- **Uncalibrated**: the timing model. Several of its parameters are
  assumptions, and no simulated time has been compared with a measured one.

PTX that is not implemented stops a launch with an error naming the
instruction; [docs/guide.md](docs/guide.md#8-limits) lists the limits.

## License

Apache-2.0; see [LICENSE](LICENSE). warplens is not related to NVIDIA Warp.
