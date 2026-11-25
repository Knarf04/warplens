# Hardware validation

Cross-checks of the simulator against a real GPU. Results recorded here came
from an RTX 3080 Ti Laptop GPU (Ampere, `sm_86`), driver 580.173.02, CUDA 12.4.
Only features that Ampere shares with Blackwell can be checked this way. The
comparison with a B200 is elsewhere: `blackwell/` and `blackwell_timing/` are
the two probe suites, each with what the B200 printed on 2026-11-25 under
`measured/b200/`, and `docs/b200-rental-1.md` is the record of that run.

Build everything with `validation/ampere/build.sh`. It needs a full CUDA
toolkit for the GPU binaries and the `warplens` conda env for the PTX.

Every program below that records a `*_golden.txt` file is rerun on the rented
B200 by the Blackwell probe suite (on 2026-11-25 each one reproduced its file
line for line), which includes these sources and predicts
these files (`validation/blackwell/README.md`, "Reruns of the Ampere
programs"; `tests/test_ampere_replay.py`). A new golden program needs a run
there: write its host code with `CHECK` and `CHECK_LAUNCH` of `host_check.h`,
which the probe suite redefines to report failures its own way.

| Check | How | Status |
| --- | --- | --- |
| `shfl.sync` semantics | `bin/shfl_golden > shfl_golden.txt` records 720 cases (4 modes x 5 widths x 36 operands) from the GPU. `tests/test_hardware_golden.py` runs the same kernel's PTX in the simulator. | all 720 match |
| Barrier with exited threads | `bin/barrier_exit_hw 0` and `1`: whole warps, then half of every warp, return before `__syncthreads()`. | both complete on the GPU with correct output; the simulator does the same (`tests/test_smem.py`) |
| `printf`, by-value struct parameter, per-thread local array | `bin/printf_hw \| sort > printf_golden.txt`; `tests/test_hardware_golden.py` runs `printf_probe.ptx` in the simulator. | output identical, including `%f`, `%5.2f`, `%hd`, `%ld`, `%#x`, `%s`, `%c` |
| `cvt.rna.tf32.f32` (float32 to TF32) | `bin/tf32_golden > tf32_golden.txt` records 715 inputs: rounding boundaries at five exponents, subnormals, infinities, NaNs and pseudo-random patterns. `tests/test_tf32.py` runs `tf32_probe.ptx` in the simulator. | all 715 match bit for bit. The `.rn` and `.rz` variants need Hopper or newer and are not checked |
| `ldmatrix` and `bfe` | `bin/ldmatrix_golden > ldmatrix_golden.txt`: all six `ldmatrix.m8n8` forms (x1, x2, x4, each with and without `.trans`) with a permuted row address per lane, and `bfe.u32`/`bfe.s32` over 726 edge cases. | all match |
| `bfi` and `shf` | `bin/bitfield_golden > bitfield_golden.txt`: `bfi.b32`, `bfi.b64` and `shf.{l,r}.{wrap,clamp}.b32` over 2925 combinations of operands, positions and lengths, including positions and lengths past the register and past 255. `tests/test_hardware_golden.py` runs `bitfield_probe.ptx` in the simulator. | all match. `shf` is as PTX gives it. `bfi.b32` uses only the low 8 bits of the position and length; `bfi.b64` does not wrap them at 256 (a position of 256, 257, 0x120 or 0xFFFFFFFF inserts nothing, a length of 256, 0x108 or 0xFFFFFFFF is the whole register; the simulator takes all 32 bits), where PTX only says both are restricted to 0..255 |
| `[reg+offset]` with a 32-bit register that has wrapped | `bin/addrwrap_golden > addrwrap_golden.txt`: six kernels, one CTA of 64 threads each, read or write a shared array of 64 words in reverse. Two are C++ (`fixed[63 - tid]` as a load and as a store). Four are inline PTX on the register `fixed - 4 * tid`, which is below zero in 32 bits for every lane but lane 0 (`0xfffffffc`, ...; the array is at shared address 0): `ld.shared.b32 [%a+252]`, `st.shared.b32 [%a+252]`, the register a further 2^31 - 256 lower with `[%a+2147483644]`, and for comparison a register that has not wrapped with `[%a+-4]`. `tests/test_hardware_golden.py` runs `addrwrap_probe.ptx` (nvcc 13.5, `sm_100a`) and `addrwrap_probe_cuda12.ptx` (nvcc 12.4, `sm_86`) in the simulator. | all 384 values match for both; five more runs on the GPU gave the same file. The offset is added in the register's 32 bits: every lane reads and writes the word the wrapped sum names. nvcc 13.5 writes the wrapping form itself for the C++ (`sub.s32 %r6, %r4, %r3; ld.shared.b32 %r7, [%r6+252]`, and the same with `st.shared.b32`); nvcc 12.4, which builds the GPU binary, computes `fixed + 4 * (63 - tid)` and uses `[%r9]`, so on this GPU the form was executed only as the inline PTX, which is the same text under both. The simulator added the offset in 64 bits and reported a memory fault until `warplens/units/addressing.py:effective_address` was changed. An address that is outside shared memory after wrapping was not run on the GPU |
| `.ftz` arithmetic (`add`, `sub`, `mul`, `div`, `fma`, `sqrt` on `.f32`) | `bin/ftz_golden > ftz_golden.txt` records 7075 cases around the subnormal range. `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match bit for bit, the 642 NaN results (all `0x7fffffff`) included. A result is flushed when it is tiny after rounding: `0x00800000 * 0x3f7fffff` gives zero, `0x00800001 * 0x3f7ffffe` gives the smallest normal number |
| `prmt.b32` | `bin/prmt_golden > prmt_golden.txt` records 304 (a, b, control) triples. `tests/test_hardware_golden.py` runs the same instruction in the simulator. | all match |
| Floating `min` / `max`, and float-to-integer `cvt` | `bin/minmax_cvt_golden > minmax_cvt_golden.txt` records 15844 cases: `min` and `max` on f32 and f64 over all pairs of zeros, ones, infinities, quiet and signalling NaNs and a few ordinary values; `cvt` from f32 and f64 to s32, u32, s64 and u64 in all four rounding modes over NaNs, infinities, halves, the values at and next to every integer limit, and pseudo-random patterns. `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match bit for bit, NaN results included. Every `cvt` case matched as first recorded. The simulator returned `0x7fc00000` (f32) or `0x7ff8000000000000` (f64) when both operands of `min` / `max` are NaN; the GPU gives `0x7fffffff` for f32 and the second operand made quiet for f64, and the simulator now does the same |
| NaN results of `add`, `sub`, `mul`, `div`, `fma`, `sqrt`, `neg`, `abs`, `mov`, and `cvt` between f16, f32 and f64 | `bin/nan_golden > nan_golden.txt` records 14270 cases: every pair (for `fma`, triples of a smaller set) of zeros, ones, infinities, ordinary values and ten NaNs (quiet and signalling, either sign, with and without a payload, all ones), for `.f32`, `.rn.f32`, `.rn.ftz.f32`, `.f64` and `.rn.f64`. `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match bit for bit, after the simulator was changed. **f32**: every NaN the GPU computes is `0x7fffffff`, whatever the operands and with or without `.ftz`; `neg` and `abs` of a NaN give it too; `cvt` between f16 and f32 gives `0x7fff` / `0x7fffffff`. **f64**: an invalid operation gives `0xfff8000000000000`; with NaN operands the result is one of them with the quiet bit set, sign and payload kept: the second operand before the first for `add`, `sub` and `mul`, the first for `div`, the second, then the third, then the first for `fma`; `neg` and `abs` leave a NaN's sign alone. `cvt` to or from f64 keeps the sign and the leading payload bits and sets the quiet bit. `mov` copies bits. The simulator returned NumPy's NaNs (`0x7fc00000`, `0xffc00000`, the first operand). `cvt.ftz.f64.f32` gives `0x7fffffffe0000000` for every NaN (the f32 NaN first becomes `0x7fffffff`); `cvt.rn.ftz.f32.f64` treats a NaN as without `.ftz`. The PTX manual does not define any of this, and Blackwell may differ |
| `cvt` with `.ftz`, and `cvt` to an integer in the same floating type | `bin/cvt_ftz_golden > cvt_ftz_golden.txt` records 8120 cases, each `.ftz` form next to the form without it: `cvt[.ftz].f64.f32`, `cvt.rn[.ftz].f32.f64`, `cvt.rn[.ftz].f16.f32`, `cvt[.ftz].f32.f16`, `cvt[.ftz].f32.f32`, `cvt.{rni,rzi,rmi,rpi}[.ftz].f32.f32` and `.f64.f64` (`floorf` and the like), `cvt.{rni,rzi,rmi,rpi}.ftz.{s32,u32,s64,u64}.f32`, over zeros, subnormals, the smallest normal numbers, NaNs and the values that become subnormal or nearly so in the narrower type. `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match bit for bit, after the simulator was given `.ftz` on `cvt` and the conversions to an integer in f32 and f64, both of which it refused. A subnormal f32 source counts as a zero of its sign (`cvt.rpi.ftz.f32.f32` of the smallest subnormal is 0, without `.ftz` 1). An f32 result is flushed when it is tiny after rounding, as for `.ftz` arithmetic: `cvt.rn.ftz.f32.f64` gives zero below 2^-126 (1 - 2^-25) (`0x380fffffefffffff`), although without `.ftz` everything from 2^-126 (1 - 2^-24) on rounds to the smallest normal number. Subnormal f16 values are not flushed, as source or as result. An f32 NaN source counts as `0x7fffffff`. ptxas refuses `.ftz` when neither type is f32 |
| Integer `div` and `rem`, with division by zero | `bin/intdiv_golden > intdiv_golden.txt` records 5808 cases: s16, u16, s32, u32, s64 and u64 over all pairs of 22 values (small numbers of either sign, the limits and their neighbours). `tests/test_hardware_golden.py` runs the same instructions in the simulator. | all match, after the simulator was changed. Division by zero gives all ones for `div` and for `rem`, in every type; the minimum over -1 gives the minimum and remainder 0. The simulator returned the dividend (`div`) and 0 (`rem`) for a zero divisor. The PTX manual leaves division by zero undefined, and Blackwell may differ |
| The token of `mbarrier.arrive` | `bin/mbarrier_token_golden > mbarrier_token_golden.txt`: 1, 2, 3 and 5 arrivals per phase, eight phases each; after every arrival `test_wait` with the token of every arrival so far, with the made-up tokens `1 << b`, and `test_wait.parity` with 0 to 3. `tests/test_hardware_golden.py` runs `mbarrier_token_probe.ptx` in the simulator. | all 88 lines match, after the simulator was changed. The GPU answers from bit 63 of the token, the parity of the phase the arrival was made in: `test_wait` is true when the barrier's phase parity differs from it, so a token is true one phase later, false two phases later, true three phases later. No other bit matters (`1 << 63` behaves like a token of an odd phase, every other `1 << b` like one of an even phase). The simulator compared phase numbers, so an old token stayed true. `try_wait` needs `sm_90` and was not measured |
| Calls to device functions | `bin/call_golden > call_golden.txt`: 256 threads call `__noinline__` functions with early returns, nested calls, a local array, mixed argument types, a store through a pointer, and a call only some lanes make. `tests/test_hardware_golden.py` runs `call_probe.ptx` in the simulator. | all 1280 values match |
| Structs by value, function pointers, recursion | `bin/callforms_golden > callforms_golden.txt`: 256 threads pass and return structs of 8, 40 and 256 bytes by value, call through function pointers taken from a local table, from a global table and from an argument, and recurse (two recursive calls in one function, mutual recursion, a local array per call that the next call reads, a struct argument). `tests/test_hardware_golden.py` runs `callforms_probe.ptx` in the simulator. | all 3072 values match |
| `vote` | `bin/vote_golden > vote_golden.txt`: `__all_sync`, `__any_sync`, `__uni_sync`, `__ballot_sync` (`vote.sync.{all,any,uni}.pred`, `vote.sync.ballot.b32`) and a ballot with a negated source, 444 launches each: a full warp; partial warps of 1, 5, 31, 33, 48 and 64 threads; lanes that returned before the vote; lanes on a branch without a vote; lanes on a branch with a ballot of their own; mixtures; several groups with their own masks at one instruction; member masks that name lanes which do not exist or have returned. Afterwards every remaining lane takes a ballot. `bin/vote_legacy_golden > vote_legacy_golden.txt` is the same launches with `vote.{all,any,uni}.pred` and `vote.ballot.b32`, the forms without `.sync`: ptxas refuses them for sm_70 and newer, so that program is built for sm_60 and the driver compiles its PTX. `tests/test_hardware_golden.py` runs `vote_probe.ptx` (nvcc 13.5, `sm_100a`), `vote_probe_cuda12.ptx` (nvcc 12.4, `sm_86`) and `vote_legacy_probe.ptx` (nvcc 12.4, `sm_60`) in the simulator. | all 2220 lines of each file match, for the three PTX files; a second run on the GPU gave the same files. A vote is among the lanes that execute the instruction together and are in the caller's mask; a lane that returned or does not exist has no vote and no bit, whatever the mask says. What is not in the golden files, because the GPU gives no answer or PTX does not allow it, is under "`vote.sync` and lanes that are elsewhere" below |
| Atomic updates (`atom`) | `bin/atom_golden > atom_golden.txt`: 4 CTAs of 128 threads update the same global words, and each CTA's shared words, with `atomicAdd`, `Sub`, `Exch`, `Min`, `Max`, `And`, `Or`, `Xor`, `CAS`, `Inc` and `Dec` on 32- and 64-bit integers, `atomicAdd` and `atomicExch` on `float`, `atomicAdd` on `double`, some `_block` and `_system` variants, and through pointers that are shared for some lanes of a warp and global for others. Only what is the same for every order of the threads is recorded: sums, minima and maxima, bit masks, wrapping counters, an increment built from compare-and-swap, the sorted old values a counter handed out, and an exchanged word plus the sum of what the exchanges returned. `tests/test_hardware_golden.py` runs the probe in the simulator twice: `atom_probe.ptx` (nvcc 13.5, `sm_100a`) and `atom_probe_cuda12.ptx` (nvcc 12.4, `sm_86`: the compiler and target of the GPU binary). | all 1512 values match for both; five runs on the GPU gave the same output. CUDA 12.4 spells the instructions `atom.global.cas.b32` and turns every function whose result is unused into `red` (`red.global.add.u32`, `red.shared.min.s64`); 13.5 spells them `atom.relaxed.gpu.global.add.u32` and emits no `red`. The second run compares the simulator with the GPU on the instruction forms the GPU executed. Which lane of a warp goes first on a shared address cannot be seen in these results and is not checked |
| `cp.async` with a source size | `bin/cpasync_golden > cpasync_golden.txt`: 32 lanes copy 16 bytes each, reading 0 to 16 of them. | all match; the GPU zero-fills the bytes it does not read |
| Where diverged lanes rejoin | `python reconvergence_golden.py > reconvergence_golden.txt` (PTX loaded through the driver API with `cuda.bindings`; not part of `build.sh`): `if (tid == 0) shared = 7;` and a join that reads the word, in four layouts of the lanes' paths to the join, two of which have the branch's post-dominator past the join; with and without `bar.sync` at the join, with all or half of the lanes reaching it. 20 kernels, 100 launches each. `tests/test_hardware_golden.py` runs the same PTX in the simulator. | all match: every lane that passes the join reads what lane 0 wrote, in every layout. The simulator let lanes 1 to 31 through first in the two layouts with a late post-dominator until `core/warp.py:_regroup` |
| Transpose output | `bin/transpose_hw` | correct on the GPU and in the simulator |
| Shared-memory bank counters | `run_ncu.sh` profiles `bin/transpose_hw` with Nsight Compute and compares its counters with the simulator's bank model. Recorded in `transpose_ncu_3080ti_grid64.csv`; `tests/test_hardware_golden.py` checks the simulator against it. | conflict and instruction counters match exactly; the total wavefront counter is 4160 higher, which `run_wavefront_probe.sh` explains (see below) |

## Launch errors

`bin/launch_errors_hw > launch_errors_3080ti.txt` (runtime API) and
`bin/driver_errors_hw > driver_errors_3080ti.txt` (driver API) record what
CUDA 12.4 returns on the RTX 3080 Ti for requests it refuses before a kernel
runs. The GPU's limits are 49152 bytes of shared memory per block by default
and 101376 with opt-in; it has no clusters.

| Request | Runtime API | Driver API |
| --- | --- | --- |
| `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` with 0 to the opt-in maximum | `cudaSuccess` | `CUDA_SUCCESS` |
| the same with the maximum + 1, + 16, + 1024, 2^30, or -1 | `cudaErrorInvalidValue` | `CUDA_ERROR_INVALID_VALUE` |
| the same for a kernel with static shared memory: up to the opt-in maximum less the static bytes; one byte more | `cudaSuccess`; `cudaErrorInvalidValue` (1024 and 61440 static bytes) | `CUDA_SUCCESS`; `CUDA_ERROR_INVALID_VALUE` (40960 static bytes) |
| `PreferredSharedMemoryCarveout` with -1 to 100; with 101 or -2 | `cudaSuccess`; `cudaErrorInvalidValue` | 101: `CUDA_ERROR_INVALID_VALUE` |
| a number that names no attribute (99, runtime API); an attribute that can only be read (`MAX_THREADS_PER_BLOCK`, driver API) | `cudaErrorInvalidValue` | `CUDA_ERROR_INVALID_VALUE` |
| launch with more dynamic shared memory than the kernel's maximum (below), which static + dynamic above the opt-in maximum always is | `cudaErrorInvalidValue` | `CUDA_ERROR_INVALID_VALUE` |
| `cudaFuncSetAttribute` or `cudaFuncGetAttributes` for a function that is no kernel | `cudaErrorInvalidDeviceFunction` | not run |
| launch with a block of 1025 threads, (33, 32, 1), (1, 1, 65) or (0, 1, 1); a grid of (0, 1, 1) or (1, 65536, 1) | `cudaErrorInvalidConfiguration` | `CUDA_ERROR_INVALID_VALUE` |
| launch with a cluster dimension attribute, any shape, (1, 1, 1) included | `cudaErrorInvalidClusterSize` | `CUDA_ERROR_INVALID_CLUSTER_SIZE` |

In every case the error is returned by the call, `cudaPeekAtLastError` shows
it, `cudaDeviceSynchronize` (`cuCtxSynchronize`) returns success,
`cudaGetLastError` returns it once and then success, and the next ordinary
launch runs.

The cluster attributes of a function (required cluster shape, non-portable
sizes, scheduling policy) were not tried; the shim accepts them unchecked.

### The shared-memory opt-in

The lines after "launch, ordinary, afterwards" in `launch_errors_3080ti.txt`,
and after the second "cuLaunchKernel, ordinary" in `driver_errors_3080ti.txt`.
Every kernel has a maximum of dynamic shared memory, which
`cudaFuncGetAttributes` reports as `maxDynamicSharedSizeBytes`
(`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`), next to its static shared
memory `sharedSizeBytes`. With D = 49152 (the default, `sharedMemPerBlock`)
and O = 101376 (`sharedMemPerBlockOptin`) on this GPU:

| Static shared memory of the kernel | Maximum before any `cudaFuncSetAttribute` | Measured with |
| --- | --- | --- |
| 0 | 49152 = D | launches with 49152 (accepted), 49153 and 49168 (refused) |
| 40960 | 8192 = D - static | 8192 accepted, 8208 refused |
| 49152 (= D) | 0 | 0 accepted, 16 refused |
| 49168 (D + 16) | 52208 = O - static | 16 and 52208 accepted, 52224 refused |
| 61440 | 39936 = O - static | 0, 16 and 39936 accepted, 39952 refused |
| 101376 (= O) | 0 | 0 accepted, 16 refused |

- A launch is refused exactly when its dynamic shared memory is above the
  kernel's maximum. The launch call returns `cudaErrorInvalidValue`
  (`CUDA_ERROR_INVALID_VALUE`); `cudaPeekAtLastError` shows it,
  `cudaDeviceSynchronize` returns success, `cudaGetLastError` returns it once,
  and the next launch runs. The two APIs agree in every case both ran.
- Static shared memory counts against the default: 40960 static + 8208
  dynamic bytes need the opt-in although neither is above 48 KB.
- **Static shared memory above the default needs no opt-in.** Such a kernel
  launches as it is, and without any call it may also use all the dynamic
  shared memory up to the opt-in maximum. A B200 (driver 610.57.04) launches
  the same way but *reports* 0 as that kernel's maximum, where this GPU
  reports O - static: `validation/blackwell/measured/b200/e2e/smem_static.txt`.
  The simulator reports what the B200 does (`reported_max_dynamic` in
  `warplens/units/smem.py`).
- `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n)`
  replaces the maximum by n. It is accepted for 0 <= n <= O - static and
  refused above, leaving the maximum as it was. It also *lowers* the maximum:
  after n = 1024 a launch with 1040 bytes, or with the default 49152, is
  refused; after 60000 and then 0, every launch with dynamic shared memory is.
- The maximum belongs to one kernel: opting one in does not let another
  launch.

An earlier version of this table said the attribute "is checked against the
device, not against what a launch could use". That was an artifact: the
kernel meant to have 1024 static bytes had its array removed by the compiler
(only one thread used it), which `sharedSizeBytes` now shows. The arrays are
shared between threads now, and the row is corrected above.

Which static sizes exist at all is the compiler's decision, not the launch's.
`ptxas` 12.4 (the `nvcc` these programs are built with) compiles 61440 and
101376 static bytes for `sm_86`. The driver's own compiler (580.173.02, on
this GPU) and `ptxas` 13.5 (for `sm_86`, `sm_89`, `sm_90` and `sm_100`) refuse
more than 49152 static bytes: "uses too much shared data (0xf000 bytes, 0xc000
max)". `ptxas` 13.5 accepts 232448 (0x38c00, 227 KB) and refuses 232449 for
the architecture-specific targets `sm_100a` and `sm_103a`, and accepts 232448
for `sm_90a`. So the same
program built with PTX only (`-gencode arch=compute_86,code=compute_86`)
fails here before any of this: every call that needs the module, the
attribute calls included, returns `cudaErrorInvalidPtx`, and
`cuModuleLoadData` of such PTX returns `CUDA_ERROR_INVALID_PTX`
(`driver_errors_3080ti.txt`, 49153 static bytes; 49152 load and launch).

`warplens/shim` returns these codes at the simulated device's limits (48 KB
by default, 227 KB with opt-in; clusters of up to 16 CTAs) and follows the
opt-in rule (`default_max_dynamic` in `warplens/units/smem.py`, applied by
`warplens/host/shim_server.py`): `tests/test_driver_shim.py::test_refusals_carry_the_codes_a_gpu_gives`,
`tests/test_driver_shim.py::test_launches_need_the_opt_in_a_gpu_needs`,
`tests/test_shim.py::test_a_refused_launch_has_the_runtime_code_for_that_refusal`,
`tests/test_shim.py::test_a_launch_is_held_to_the_shared_memory_the_kernel_was_opted_in_to`.
The Python API (`Device.launch`) has no opt-in step and is not held to it.
What this does not show: that a B200 returns `cudaErrorInvalidClusterSize`
for a cluster of more than 16 CTAs or a grid that is not a multiple of the
cluster. The code is the one the CUDA headers define for "a kernel launch
error due to cluster misconfiguration" and the one Ampere gives for any
cluster, but no cluster launch could be made here (RT-12). Nor that a B200 follows the opt-in
rule as measured here: that its default is 48 KB and its maximum 227 KB
(RT-03, RT-04), that a kernel with more static shared memory than the default
launches without opt-in there too (`validation/blackwell_timing/pipelined_gemm.cu`
has 98 KiB and relies on it), and that its driver compiles such a kernel from
`sm_100a` PTX as `ptxas` 13.5 does. The simulator does not model the
compiler's limit: PTX for a target that is not architecture-specific with
more than 48 KB of static shared memory is run, where a driver would refuse
to load it.

## `vote.sync` and lanes that are elsewhere

`validation/ampere/run_vote_other.sh > vote_other_3080ti.txt` runs 100 launches of
`vote_probe` one process at a time under a timeout (5 s for the recording; a launch that
returns takes well under one). They are not golden results.

| Launch | RTX 3080 Ti | warplens |
| --- | --- | --- |
| Some lanes take a branch without a vote, or with a ballot of their own, and then the final ballot with the mask `0xffffffff`; the others call `__all_sync`, `__any_sync` or `__uni_sync` with a mask that names them too | **never returns** (24 of 24 launches; in an earlier run of 216 such launches with `__all_sync`, all 216). Each group waits for the other at its own instruction | refuses the launch: `UnsupportedError`, "the member mask names a lane that is on another path ..." |
| The same with `__ballot_sync` for the voters | returns, with the bits of the lanes on the other branch in the voters' ballot (16 of 16). In the machine code for `sm_86`, a `vote.sync` whose mask is not the lanes present calls a routine that waits (`WARPSYNC mask; VOTE`), and the ballots of the kernel share one such routine, so the two groups, at different ballots of the source, wait at the same machine instruction and vote as one (`cuobjdump -sass` on the probe's object file shows it). This is the compiler's layout, not a rule | refuses it in the same way |
| Every lane votes, and no lane is in its own mask: the mask is the next lane only, or every lane but the caller | `cudaErrorIllegalInstruction` (30 of 30) | runs, votes among the lanes of the mask, and reports `vote-membermask-mismatch` |
| Every lane votes with the mask `0x0000ffff` (half of the callers are not in it) | returns what the model gives: each lane the vote of lanes 0 to 15 | the same values, and `vote-membermask-mismatch` |
| Every lane votes with the mask 0 | returns the vote of all 32 lanes | an empty vote (`all` and `uni` true, `any` false, ballot 0), and `vote-membermask-mismatch` |

So on this GPU `vote.sync` really waits: a mask may name lanes that returned or never
existed, and must not name a lane that will not come. `tests/test_hardware_golden.py::test_vote_masks_a_gpu_does_not_answer_are_refused_or_reported`
holds the simulator to the third column. None of these launches is rerun on the B200.

`shfl.sync`, `bar.warp.sync` and `elect.sync` have the same kind of mask and were not
measured in this way; the simulator does not make them wait.

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
