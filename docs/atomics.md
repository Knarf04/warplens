# Atomic addition

The simulator executes scalar `atom.add` and `red.add` for ordinary CUDA
`atomicAdd` kernels. `atom` returns the value before the update; `red` and an
`atom` destination of `_` update memory without keeping that value.

| Field | Supported |
| --- | --- |
| Operation | `add` |
| Type | `u32`, `s32`, `u64`, `f32`, `f64` |
| Space | `.global`, `.shared`, `.shared::cta`, or generic addresses into global or this CTA's shared memory |
| Ordering | omitted or `.relaxed` |
| Scope | omitted (PTX defaults to `.gpu`), `.cta`, `.gpu`, `.sys` |
| Floating modifier | `.noftz` for `f32`; otherwise the memory-space rules below |

Examples include `atom.relaxed.gpu.global.add.u32`,
`atom.relaxed.gpu.shared.add.u64`, and `red.global.add.f32`. CUDA 13.5 compiled
integer, float, double and shared unsigned-long-long `atomicAdd` kernels run in
`tests/test_atomics.py` without a GPU. This is compiler and simulator validation;
it is not a Blackwell hardware comparison.

Every active lane performs a complete read/modify/write before the next lane,
in increasing lane order. Other warps cannot interleave within that operation.
This prevents lost updates when lanes share an address and preserves each old
value. The chosen order is deterministic; hardware may serialize the same lanes
in another order, producing different floating results or assigning different
old values to lanes. Integer additions wrap at their operand width.

Floating addition rounds to nearest even. Without `.noftz`, global `f32`
addition flushes subnormal inputs and results to zeros of the same sign;
shared `f32` and `f64` preserve subnormals. The value returned by `atom` is the
original stored value, before any input flushing. NaN payload propagation is
not a hardware bit-exact promise. These rules follow the NVIDIA PTX
[`atom`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom)
and [`red`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)
sections. PTX does not define `add.s64`; use the compiler-emitted `u64` form for
64-bit integer addition.

Shared atomics do not report races against other atomic additions, but ordinary
loads/stores without a barrier can still produce `smem-race`. Uninitialized
shared reads and missing synchronization with asynchronous copies remain
findings. Atomic accesses require natural alignment and valid active-lane
addresses; masked lanes do not access memory.

Acquire, release and acquire-release atomics, cluster scope and remote shared
addresses, other reduction operations, half/bfloat and packed/vector forms, and
cache-policy modifiers are rejected explicitly. Accepted scopes describe a
functional execution with serialized accesses. The simulator does not explore
weak-memory outcomes, prove sufficient scope, or model concurrent host/system
memory accesses.

Timing is approximate: global atomics use the existing global-access latency
once per warp instruction, in addition to instruction issue cost. Atomic
throughput and contention are not modeled. Shared atomic bank counters are not
reported, because the ordinary-load broadcast model would incorrectly merge
updates to one address; shared atomics currently incur instruction issue cost
only.
