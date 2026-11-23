# warplens

Understand an NVIDIA Blackwell GPU by watching it run. No GPU required.

warplens takes real GPU kernel code compiled to PTX for datacenter Blackwell
(`sm_100a`) and executes it on the CPU, with no GPU anywhere in the loop.

> **Status: early.** Milestones M0 to M2 are done: CUDA kernels using global
> and shared memory, barriers, warp shuffles, TMA and mbarriers run and match
> NumPy, with a shared-memory bank-conflict report and checkers for races and
> for misuse of the TMA/mbarrier protocol. The Blackwell tensor-core
> instructions (tcgen05) are not implemented yet. See [PLAN.md](PLAN.md) for the
> roadmap.

## Try it

```bash
pip install -e ".[dev]"
python examples/00_vecadd/run.py       # run an nvcc-compiled kernel, compare with NumPy
python examples/00_divergent/run.py    # watch a warp's active mask as its lanes diverge
python examples/01_smem_transpose/run.py  # bank conflicts of three shared-memory layouts
python examples/02_tma_copy/run.py     # TMA + mbarrier, then five deliberate bugs and their reports
pytest
```

```python
import numpy as np, warplens

sim = warplens.Device("B200")
A, B = sim.from_numpy(a), sim.from_numpy(b)
C = sim.empty(n, "f32")
sim.launch("examples/00_vecadd/vecadd.ptx", "vecadd", grid=4, block=256, args=[A, B, C, n])
assert np.array_equal(C.numpy(), a + b)
```

Running needs only Python 3.11+ and NumPy. Compiling your own CUDA C++ to PTX
needs `nvcc` from CUDA 12.8 or newer; [env/](env/README.md) sets up an isolated
conda env with it.

## Documentation

- [PLAN.md](PLAN.md): goals, architecture, milestones
- [docs/m0.md](docs/m0.md): what M0 covers and how the executor works
- [docs/m1.md](docs/m1.md): shared memory, bank model, barriers, findings
- [docs/m2.md](docs/m2.md): TMA, tensor maps, mbarriers, protocol checkers
- [docs/coverage.md](docs/coverage.md): which PTX is supported
- [validation/README.md](validation/README.md): cross-checks against a real GPU

warplens is not related to NVIDIA Warp.
