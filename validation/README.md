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
| `.ftz` arithmetic (`add`, `sub`, `mul`, `div`, `fma`, `sqrt` on `.f32`) | `bin/ftz_golden > ftz_golden.txt` records 7075 cases around the subnormal range. `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match bit for bit (NaN results compared as NaNs). A result is flushed when it is tiny after rounding: `0x00800000 * 0x3f7fffff` gives zero, `0x00800001 * 0x3f7ffffe` gives the smallest normal number |
| Calls to device functions | `bin/call_golden > call_golden.txt`: 256 threads call `__noinline__` functions with early returns, nested calls, a local array, mixed argument types, a store through a pointer, and a call only some lanes make. `tests/test_hardware_golden.py` runs `call_probe.ptx` in the simulator. | all 1280 values match |
| `cp.async` with a source size | `bin/cpasync_golden > cpasync_golden.txt`: 32 lanes copy 16 bytes each, reading 0 to 16 of them. | all match; the GPU zero-fills the bytes it does not read |
| Transpose output | `bin/transpose_hw` | correct on the GPU and in the simulator |
| Shared-memory bank counters | `run_ncu.sh` profiles `bin/transpose_hw` with Nsight Compute and compares its counters with the simulator's bank model. Recorded in `transpose_ncu_3080ti_grid64.csv`; `tests/test_hardware_golden.py` checks the simulator against it. | conflict and instruction counters match exactly; the total wavefront counter is 4160 higher, which `run_wavefront_probe.sh` explains (see below) |

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
  conflicts or not. The probe below shows what that is: two wavefronts per
  warp and one per CTA per `__syncthreads()`, here 2 x 2048 + 64. Neither has
  to do with shared-memory accesses, so hardware "wavefronts minus conflicts"
  is not the simulator's `ideal`.
- The `ideal` counter is not available on this GPU with ncu 2024.1.

## The wavefront offset (TIM-17)

```bash
sudo env "PATH=$PATH" validation/ampere/run_wavefront_probe.sh
```

`bin/wavefront_probe` launches one conflict-free kernel 59 times and changes
one thing at a time: the grid, the block shape (whole and partial warps), the
number of shared loads and stores per thread, of global loads and stores, and
of `__syncthreads()` calls. A second kernel has no shared memory at all.
`wavefront_fit.py` joins `ncu`'s counters with the launch list and fits the
offset. Recorded in `wavefront_probe_ncu_3080ti.csv` and
`wavefront_probe_configs_3080ti.csv` (2026-11-23);
`tests/test_hardware_golden.py` checks the rule against them.

In all 59 launches, exactly:

    l1tex__data_pipe_lsu_wavefronts_mem_shared
        = shared loads + shared stores      (one per warp access; plus conflicts)
        + 2 per warp launched
        + 1 per CTA per __syncthreads()

| What was varied | Effect on the offset |
| --- | --- |
| Shared loads and stores per thread, 0 to 5 | none: each access adds exactly the one wavefront the model counts |
| Global loads and stores per thread, 0 to 4 | none |
| Grid, 1 to 500 | proportional |
| Block shape, 1 to 32 warps per CTA | 2 per warp. A warp of 1, 16 or 31 threads counts like a full one |
| `__syncthreads()` per thread, 0 to 4, at 1, 2, 4, 5 and 32 warps per CTA | 1 per CTA for each, whatever the number of warps |
| A kernel with no shared memory and an empty body | still 2 per warp |

What this does not show:

- **Why.** The counter evidently includes events that are not shared loads or
  stores. That the two per warp are the warp's start and exit, and the one per
  CTA the barrier's release, is a guess.
- **Blackwell.** Measured on Ampere only. `validation/blackwell_timing/ncu_banks`
  should check the same rule on a B200 before it is relied on there.
- **Other barriers and kernels that end early.** Only `__syncthreads()` executed
  by every thread was measured: not named barriers, not a barrier some warps
  skip.

The simulator's `wavefronts` is left as it is: the count of what shared
accesses cost. `stats.smem.ncu_wavefronts` adds the two terms above, and is
what `compare_ncu.py` compares with the `ncu` counter.
