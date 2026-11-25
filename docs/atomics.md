# Scalar atomics

The simulator executes scalar atomic read/modify/write operations (`atom`) and
reductions (`red`) for ordinary CUDA kernels. `atom` returns the value before the update; `red` and
an `atom` destination of `_` update memory without keeping that value.

| Operation | Supported types |
| --- | --- |
| `atom.add`, `red.add` | `u32`, `s32`, `u64`, `f32`, `f64` |
| `atom.and`, `atom.or`, `atom.xor`, and the same three for `red` | `b32`, `b64` |
| `atom.exch`, `atom.cas` (PTX has no `red.exch` or `red.cas`) | `b32`, `b64` |
| `atom.inc`, `atom.dec`, `red.inc`, `red.dec` | `u32` |
| `atom.min`, `atom.max`, `red.min`, `red.max` | `s32`, `u32`, `s64`, `u64` |

Supported spaces are `.global`, `.shared`, `.shared::cta`, and generic addresses
into global or shared memory. Generic shared pointers can address a peer CTA in
the cluster. Ordering may be omitted or `.relaxed`; scope may be omitted (PTX
defaults to `.gpu`), `.cta`, `.gpu`, or `.sys`. `add.noftz.f32` preserves subnormal
values.

Two generations of compiler spell the same functions differently, and both run:

| | nvcc 13.5, `-arch=sm_100a` | nvcc 12.4, `-arch=sm_80` / `sm_86` (and 13.5 for those targets) |
| --- | --- | --- |
| `atomicAdd(p, v)`, result used | `atom.relaxed.gpu.global.add.u32`, `atom.relaxed.gpu.shared.add.u32`; `atom.relaxed.gpu.add.u32` when the compiler cannot tell which memory `p` is in | `atom.global.add.u32`, `atom.shared.add.u32`, `atom.add.u32` |
| `_block`, `_system` | `atom.global.cta.add.u32`, `atom.shared.sys.add.u32`, `atom.cta.add.u32`, ... | the same |
| result discarded | the same `atom` | nvcc 12.4 only: `red.global.add.u32`, `red.shared.min.s64`, `red.or.b32`, `red.global.cta.xor.b32`, ... for every function but `atomicExch` and `atomicCAS`, which stay `atom` |

The short forms are not a different instruction: PTX defines an omitted ordering
as `.relaxed` and an omitted scope as `.gpu`, and `red` as the `atom` update
without a destination. `tests/execution/test_atom.py` runs every spelling in this table for
every operation and type.

Every active lane performs a complete read/modify/write before the next lane,
in increasing lane order. Other warps cannot interleave within that operation.
This prevents lost updates when lanes share an address and preserves each old
value. The chosen order is deterministic; hardware may serialize the same lanes
in another order, producing different floating results or assigning different
old values to lanes. Integer additions wrap at their operand width; `inc` and
`dec` use PTX's limit-and-wrap behavior rather than ordinary increment/decrement.

Floating addition rounds to nearest even. Without `.noftz`, global `f32`
addition flushes subnormal inputs and results to zeros of the same sign;
shared `f32` and `f64` preserve subnormals. The value returned by `atom` is the
original stored value, before any input flushing. NaN payload propagation is
not a hardware bit-exact promise. These rules follow the NVIDIA PTX
[`atom`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom)
and [`red`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
sections. PTX does not define `add.s64`; use the compiler-emitted `u64` form for
64-bit integer addition.

Shared atomics do not report races against other atomics, but ordinary
loads/stores without a barrier can still produce `smem-race`. Atomic updates
preserve prior ordinary-access history so a later atomic cannot hide such a
race. Uninitialized shared reads and missing synchronization with asynchronous
copies remain findings. Proxy fences track the latest writer separately from
that race history. As with other peer shared accesses, asynchronous-copy hazard
checking for remote shared atomics is not modeled. Atomic accesses require
natural alignment and valid active-lane addresses; destination and address
validation precede memory changes. Masked lanes do not access memory.

Acquire, release and acquire-release atomics, cluster scope, explicit
`.shared::cluster`, `red.exch` and `red.cas` (which PTX does not define), `sub`, half/bfloat and packed/vector
forms, and cache-policy modifiers are rejected explicitly. Accepted scopes
describe a functional execution with serialized accesses. The simulator does
not explore weak-memory outcomes, prove sufficient scope, or model concurrent
host/system memory accesses.

Timing retains the assumed load-plus-store model: two instruction issues,
with the ordinary global-access latency charged once. Shared operations use
twice the ordinary bank-wave estimate; the bank counters themselves record one
access. That estimate merges same-word accesses as ordinary broadcasts and
therefore does not predict atomic contention or serialization. Generic remote
shared accesses are omitted from these bank counters, as with ordinary remote
accesses. Atomic throughput has not been calibrated on hardware.

`tests/execution/test_atom.py` covers the operation/type matrix, race diagnostics, remote
shared addressing, and the assumed timing model. The checked-in RTX 3080 Ti
atomic capture supplies order-independent integer/bit-operation reference
results in `tests/validation/golden_replay/test_hardware_golden.py`, which runs the probe twice: as nvcc
13.5 compiles it for `sm_100a` (`atom_probe.ptx`) and as nvcc 12.4, the compiler of
the GPU binary, compiles it for `sm_86` (`atom_probe_cuda12.ptx`, legacy `atom`
spellings and `red`). `tests/execution/test_atomics.py` additionally
compiles CUDA 13.5 integer, float, double and shared unsigned-long-long
`atomicAdd` kernels for `sm_100a` and for `sm_86` and executes their PTX without a GPU, and checks flushing,
masking, validation and proxy-fence regressions. Those compiler and simulator
checks are not a Blackwell hardware comparison. There is one: the RTX 3080 Ti
atomic program was rerun on a B200 on 2026-11-25 and gave the same
order-independent results (the run `e2e/ampere_atom`, identical in
`validation/blackwell/measured/b200/compare-pinned-nvcc-13.5.37.txt`).
