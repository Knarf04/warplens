# Hardware validation

Cross-checks of the simulator against a real GPU. Results recorded here came
from an RTX 3080 Ti Laptop GPU (Ampere, `sm_86`), driver 580.173.02, CUDA 12.4.
Only features that Ampere shares with Blackwell can be checked this way.

Build everything with `validation/ampere/build.sh`. It needs a full CUDA
toolkit for the GPU binaries and the `warplens` conda env for the PTX.

| Check | How | Status |
| --- | --- | --- |
| `shfl.sync` semantics | `bin/shfl_golden > shfl_golden.txt` records 720 cases (4 modes x 5 widths x 36 operands) from the GPU. `tests/test_hardware_golden.py` runs the same kernel's PTX in the simulator. | all 720 match |
| Barrier with exited threads | `bin/barrier_exit_hw 0` and `1`: whole warps, then half of every warp, return before `__syncthreads()`. | both complete on the GPU with correct output; the simulator does the same (`tests/test_smem.py`) |
| `printf`, by-value struct parameter, per-thread local array | `bin/printf_hw \| sort > printf_golden.txt`; `tests/test_hardware_golden.py` runs `printf_probe.ptx` in the simulator. | output identical, including `%f`, `%5.2f`, `%hd`, `%ld`, `%#x`, `%s`, `%c` |
| `cvt.rna.tf32.f32` (float32 to TF32) | `bin/tf32_golden > tf32_golden.txt` records 715 inputs: rounding boundaries at five exponents, subnormals, infinities, NaNs and pseudo-random patterns. `tests/test_tf32.py` runs `tf32_probe.ptx` in the simulator. | all 715 match bit for bit. The `.rn` and `.rz` variants need Hopper or newer and are not checked |
| `ldmatrix` and `bfe` | `bin/ldmatrix_golden > ldmatrix_golden.txt`: all six `ldmatrix.m8n8` forms (x1, x2, x4, each with and without `.trans`) with a permuted row address per lane, and `bfe.u32`/`bfe.s32` over 726 edge cases. | all match |
| `cp.async` with a source size | `bin/cpasync_golden > cpasync_golden.txt`: 32 lanes copy 16 bytes each, reading 0 to 16 of them. | all match; the GPU zero-fills the bytes it does not read |
| Transpose output | `bin/transpose_hw` | correct on the GPU and in the simulator |
| Shared-memory bank counters | `run_ncu.sh` profiles `bin/transpose_hw` with Nsight Compute and compares its counters with the simulator's bank model. Recorded in `transpose_ncu_3080ti_grid64.csv`; `tests/test_hardware_golden.py` checks the simulator against it. | conflict and instruction counters match exactly; the total wavefront counter does not (see below) |

## Running the bank-counter comparison

`ncu` refuses to read performance counters as a normal user on this machine
(`ERR_NVGPUCTRPERM`). Either run the script as root:

```bash
sudo env "PATH=$PATH" validation/ampere/run_ncu.sh 64
```

or allow non-admin profiling once, as described at
https://developer.nvidia.com/ERR_NVGPUCTRPERM, and run it normally.

The script prints each counter from `ncu` next to the simulator's prediction
and exits non-zero if any differ.

## Bank-counter results (grid 64, block 32x32, 2026-11-23)

| Counter (`.sum`) | linear: ncu / warplens | padded | swizzled |
| --- | --- | --- | --- |
| `smsp__inst_executed_op_shared_ld` | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| `smsp__inst_executed_op_shared_st` | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | 63488 / 63488 | 0 / 0 | 0 / 0 |
| `..._mem_shared_op_ld` | 63488 / 63488 | 0 / 0 | 0 / 0 |
| `..._mem_shared_op_st` | 0 / 0 | 0 / 0 | 0 / 0 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared` | 71744 / 67584 | 8256 / 4096 | 8256 / 4096 |
| `..._wavefronts_mem_shared_ideal` | n/a | n/a | n/a |

What this shows:

- The **bank-conflict counters match exactly**, in total and split by load
  and store. A 32-way conflict costs 31 extra wavefronts per warp access, as
  the model says.
- The **total wavefront counter is 4160 higher on hardware in every layout**,
  conflicts or not. That is 65 per CTA here, or equivalently one extra per
  shared load and store plus one per CTA. This run cannot tell those apart,
  and the cause is unknown. So hardware "wavefronts minus conflicts" is not
  the simulator's `ideal`. Runs with other block shapes would separate the
  per-instruction part from the per-CTA part.
- The `ideal` counter is not available on this GPU with ncu 2024.1.
