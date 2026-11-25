# How the simulator works

The design of warplens, unit by unit: what each part models and the rule it
follows. The [user guide](guide.md) says how to use it and what it has been
checked against (section 2); this file is the reference behind the guide's
section 7.1. The rules are stated as they are in the code today.

These explanations were first written milestone by milestone. The write-ups
they come from are kept as they were under [history/](history/README.md),
with the acceptance results and the measurements of the time; where one of
them disagrees with this file or with the guide, it is the older one.

Related references: [coverage.md](coverage.md) (every PTX form supported),
[numerics.md](numerics.md) (what is computed exactly),
[scheduling.md](scheduling.md) (the order warps run in),
[atomics.md](atomics.md), [clc.md](clc.md) (cluster launch control),
[validation-matrix.md](validation-matrix.md) (every claim, with its probe).

Contents:

1. [From PTX to a launch](#1-from-ptx-to-a-launch)
2. [The warp executor](#2-the-warp-executor)
3. [Shared memory, barriers and the race check](#3-shared-memory-barriers-and-the-race-check)
4. [TMA and mbarriers](#4-tma-and-mbarriers)
5. [The tensor core and tensor memory](#5-the-tensor-core-and-tensor-memory)
6. [Clusters](#6-clusters)
7. [The timing model](#7-the-timing-model)
8. [Reports and views](#8-reports-and-views)
9. [`warplens run`: the shim and the front ends](#9-warplens-run-the-shim-and-the-front-ends)

## 1. From PTX to a launch

```
 CUDA C++ / CUTLASS ── nvcc ──┐
 Triton ──────────────────────┤
 TIRx ── NVRTC ───────────────┤   PTX
 CuTe DSL ────────────────────┤
 hand-written PTX ────────────┘
              │
              ▼
   host side: warplens.Device (Python), or the program's own host code
   with its CUDA calls served by the shim (`warplens run`)
              │  memory, tensor maps, launch (grid, block, cluster, shared memory)
              ▼
   frontend: parse → IR → control-flow graph → immediate post-dominators
              │
              ▼
   engine: warps (32 lanes under an active mask), the TMA engine, the tensor core
   state:  global memory | shared memory per CTA | tensor memory per CTA |
           registers per lane | mbarriers | bulk groups | tensor maps
              │
              ├── functional results (device memory, what the kernel printed)
              ├── checkers (findings, with the PTX line)
              └── timing model, on request (trace, utilization, verdict)
```

| Package | What is in it |
| --- | --- |
| `warplens/frontend/` | the PTX parser, the IR, inlining of device functions, the control-flow graph |
| `warplens/host/` | `Device`, launches, tensor maps, `run_program`, the server the shim talks to |
| `warplens/core/` | the engine, clusters, CTAs, warps, the reconvergence stack, vector clocks, the launch queue of cluster launch control |
| `warplens/units/` | what instructions do: arithmetic, memory, shared memory and its banks, barriers, mbarriers, TMA, `tcgen05`, atomics, calls, cluster launch control, the small number formats |
| `warplens/check/` | findings |
| `warplens/timing/` | the timing model, its engine, the configs' loader, placement |
| `warplens/viz/` | the trace, the HTML reports, the shared-memory and tensor-memory views |
| `warplens/numerics/` | FP8/FP4 and block-scaled formats on the host |
| `warplens/shim/` | the C sources of the `libcudart` and `libcuda` replacements |
| `warplens/configs/` | `b200.yaml`, `b300.yaml` (inside the package, so that they install with it) |

**Unsupported PTX is a hard error.** Unknown opcodes are rejected when the
kernel is launched, before anything runs. Unknown modifiers are rejected when
the instruction executes. Nothing is run as something else.

## 2. The warp executor

**Registers hold raw bits.** nvcc declares most registers as `.b32` / `.b64`
and lets each instruction choose the interpretation (`add.f32 %r8, %r6, %r7`
on `.b32` registers). A register is therefore 32 unsigned integers of its
width, and `Warp.read(operand, type)` reinterprets them per instruction.

**A warp executes one instruction on 32 lanes at once** as NumPy arrays, and
writes results only to lanes in the active mask. Lanes past the end of a
partial warp are never active.

**Divergence uses a reconvergence stack** (`core/simt_stack.py`). When lanes
disagree at a branch, the warp splits into one stack entry per side, and the
sides rejoin at the branch's immediate post-dominator (`frontend/cfg.py`).
The stack also performs a rendezvous at named CTA barriers and `tcgen05`
`.sync.aligned` instructions: pending paths run until every non-exited lane
reaches the same instruction. This is necessary when the static
post-dominator lies beyond the actual join, including joins inside cycles.
Rendezvous at different instruction addresses or across divergent recursive
activations is not supported.

Four other cases needed care:

- *Loops.* A loop branch reconverges at the loop exit on every iteration. If
  the top entry already rejoins there, it is replaced rather than nested, so
  stack depth does not grow with the trip count.
- *Early `ret`.* Returning lanes are removed from every stack entry, so they
  cannot be revived when an outer region reconverges.
- *Infinite loops.* Blocks that cannot reach the exit get a virtual edge to
  it, so a branch into such a loop reconverges nowhere.
- *Lanes that meet before the post-dominator.* In
  `if (a) {...; goto join;} if (b) return; join: ...` the first branch's
  post-dominator is the return, yet every lane that stays passes `join`. The
  recorded Ampere programs run them as one group from `join` on
  (`validation/ampere/reconvergence_golden.py`). To reproduce this, among the
  stack entries that rejoin at the same pc, an entry that can still reach the
  pc of the top one (and cannot be reached from it) runs first, and entries
  at the same pc are merged (`core/warp.py`, `_regroup`). Entries on a common
  cycle keep their order. This grouping implies no hardware guarantee for
  unsynchronized communication.

**The engine steps warps one instruction each**, round-robin by default, so
that warps interleave with the TMA engine, the tensor core and each other.
Other orders (seeded, adversarial) are in [scheduling.md](scheduling.md). A
warp has a `blocked` field; the engine skips blocked warps and asks the CTA
to release and reset completed barriers after every warp instruction, before
another warp can join the next phase. Named barriers, mbarrier waits and
tensor-memory allocation all block this way.

**Deadlock.** If no warp can run and nothing asynchronous is pending, the
launch stops with `DeadlockError`, listing each barrier, how many warps
arrived, what it needs, and the PTX line where each warp waits.

**Global memory faults on stray accesses.** Allocations are separated by
unmapped gaps. An out-of-bounds load or store raises `MemoryFault` with the
PTX line, CTA, warp, lane and thread. Misaligned loads and stores are
allowed. A device array belongs to the `Device` that allocated it: launch
arguments from another `Device` are rejected before execution, even when
their numeric addresses match a local allocation.

**Fused multiply-add rounds once.** `fma` and floating-point `mad` combine
integer significands exactly before rounding to the destination format. The
other numerical rules are in [numerics.md](numerics.md).

**Inline assembly.** nvcc passes `asm` blocks through, including `{ ... }`
blocks with their own `.reg` declarations and labels (the mbarrier wait
loop). Shared variables in those blocks are scoped too, so a nested
declaration cannot overwrite an enclosing variable. Names declared in a
nested block are stored as `name@scope`.

**General PTX around the kernels.** `.global`/`.const` data, per-thread
`.local` memory, structs passed by value (parameters have an address space of
their own, reached with `cvta.param`), `.v2`/`.v4` loads and stores, vector
`mov`, `elect.sync`, calls to device functions (inlined) and to `vprintf`.
Text printed by the kernel is returned in `stats.output`; CUDA `printf`
returns the number of parsed arguments.

## 3. Shared memory, barriers and the race check

**Shared addresses are offsets.** nvcc keeps shared addresses in 32-bit
registers (`mov.b32 %r9, tile`), so a shared address is a byte offset into
the CTA's shared memory. `cvta.shared` maps it into a fixed window of the
generic address space, and a generic `ld`/`st` routes by address.
`ld.global` cannot reach shared memory.

**Layout.** Sized `.shared` variables are placed in declaration order with
their alignment. Dynamic `.extern .shared` arrays share a base aligned to at
least 16 bytes and to the strongest alignment declared by any dynamic alias.
They get the `smem=` bytes passed to `launch`; alignment padding also counts
towards the per-CTA limit of 227 KB (the maximum per thread block for compute
capability 10.x in the CUDA C++ Programming Guide; 228 KB is the amount per
SM).

**Bank model** (`units/smem_banks.py`). 32 banks of 4 bytes; the word at byte
offset `o` is in bank `(o // 4) % 32`. For one warp instruction:

- `wavefronts` = the largest number of distinct words in any one bank;
- `ideal` = `ceil(distinct words / 32)`;
- `conflicts` = `wavefronts - ideal`.

Lanes touching the same word do not conflict. A vector access counts as one
full-width instruction. `conflicts` agrees exactly with Nsight Compute on an
RTX 3080 Ti and on a B200. The absolute `wavefronts` number does not: the
hardware counter also includes two wavefronts per warp and one per CTA per
`__syncthreads()`, which are not shared accesses
([validation/README.md](../validation/README.md), "The wavefront offset");
`stats.smem.ncu_wavefronts` adds them, to compare with `ncu`.

**Barriers count warps.** `bar.sync n` blocks the warp until the barrier
completes. With a thread count, it completes when arrived warps x 32 reach
the count. Without one, it completes when every warp of the CTA that has not
finished has arrived. An explicit sync count of zero has the same meaning as
an omitted count. Other counts must be multiples of 32, and `bar.arrive`
requires a nonzero count. Conflicting explicit counts or repeated arrivals by
a warp before phase completion raise an error. `bar.arrive` arrives without
blocking.

**Findings** (`check/report.py`) are problems that do not stop the kernel.
`launch` returns them in `stats.findings`; each names the PTX line and the
first CTA, warp and lanes, and repeats are counted per instruction. Distinct
instructions on the same source line keep separate counters. The guide's
section 6.4 lists every kind.

**What a barrier orders.** Each warp of a cluster has a vector clock
(`core/clocks.py`): for every other warp, how much of that warp's work it is
ordered after. A warp that arrives at a barrier releases what it has done so
far; a warp that waits at the barrier and passes it acquires what the
arrivals released. This holds for `bar.sync` and `bar.arrive` (which releases
and acquires nothing), for an mbarrier (an arrival releases, a successful
non-relaxed wait acquires, `.relaxed` forms do neither) and for the cluster
barrier. So only the warps that took part are ordered: a `bar.sync 1, 64`
between two warps says nothing about a third, and an mbarrier phase completed
by warp 1 and awaited by warp 2 does not order warp 0. Two exceptions are
deliberate. A barrier with no thread count, and the cluster barrier, complete
without the warps that have returned, so what those warps did counts as
before it. And a `tcgen05.commit` arrives on its mbarrier with what its warp
had done when it issued the commit.

**The race check.** Shared memory keeps, per byte, the last ordinary writer
and its clock, and the readers. One writer is enough, since two unordered
stores are themselves reported. Readers do not conflict with each other, so a
store must be checked against all that are not ordered one after another: a
byte holds one reader while each new one is ordered after the last, and a
clock per warp from the moment two unordered warps have read it, until the
next store (`units/smem.py`, `_Accesses`). Atomic updates are recorded the
same way and do not race with each other, only with another warp's ordinary
load or store.

The check does not depend on the schedule. A kernel with a missing
`__syncthreads()` gives the right answer under round-robin scheduling and is
still reported; [scheduling.md](scheduling.md) has schedules under which its
answer changes too.

Limits of the check:

- It works per warp. Two lanes of one warp are never reported against each
  other, and a barrier that only some lanes of a warp reach counts for the
  warp. Per-thread barrier semantics in divergent code are reported
  (`barrier-in-divergent-code`), not modeled.
- The per-warp reader clocks take 4 bytes per byte of shared memory per warp
  of the cluster, allocated when a CTA first has two unordered readers of a
  byte and touched only for such bytes: at most 29 MB for a CTA of 32 warps
  with all 227 KB read that way, but 16 times that per CTA in a cluster of 16
  such CTAs. In the worst case tried (32 warps each read a 4 KB table, then
  every thread writes a word of it) a launch took 10% longer than without
  them.
- An access from another CTA of the cluster is recorded, and a local access
  that races with it is reported, but no finding is raised at the remote
  access itself, and races between CTAs are not reported.
- Shuffles do not rendezvous lanes executing separate divergent paths. Such
  rendezvous is legal on `sm_70+`; the executor reports participation and
  source findings and does not guarantee those results. Exited or unlaunched
  lanes may remain in a shuffle's member mask, and disjoint groups with
  different member masks may execute the same instruction together.

## 4. TMA and mbarriers

**Tensor maps.** A real `CUtensorMap` is 128 opaque bytes written by the
driver. `Device.tensor_map_tiled` plays the role of `cuTensorMapEncodeTiled`,
applies the same documented argument checks, and writes a 128-byte record of
warplens's own that the TMA unit decodes (`host/tensor_map.py`). The kernel
receives a pointer to it in global memory, or the bytes inside a struct
passed by value, as CUTLASS does (`TensorMap.tobytes()`). `box` is given in
the array's NumPy axis order; TMA coordinates inside the kernel are innermost
first. Dimensions, box sizes and element strides must be integers; the
maximum tensor dimension is 2^32, following CUDA 13.5. Decoding validates
descriptor fields again, so a corrupted map is a source-located error.

A tensor map in the **driver's own layout** is read as well
(`host/tensor_map.py:decode_driver`). The layout was worked out from a sweep
of the B200 driver's encoder: every one of the 1087 maps it encoded reads
back as the parameters that made it, and so do the four the CuTe DSL wrote
for CUTLASS's `fp16_gemm_0.py` (`tests/test_tensor_map_driver_layout.py`).
The driver's map carries the bytes of one box as `box // element stride`,
rounded down, where `TensorMapInfo.counts` rounds up; they agree when the
strides divide the box, which is all that has run on a GPU.

**A TMA instruction issues a copy; the copy completes later.** Completion is
an event `async_delay` scheduling rounds after issue (default 4), or
immediately if no warp can run; in a timed run the timing model decides. A
load reads global memory when issued and writes shared memory when it
completes, then performs `complete_tx` on its mbarrier. A store reads shared
memory when it completes. Copy bounds are checked before allocating
transfer-sized temporary arrays. A tensor copy at a shared address that is
not a multiple of 32 bytes is an error: a B200 ends the kernel at 16
(`cudaErrorMisalignedAddress`), and 32 is the weakest alignment that
measurement allows (`TENSOR_COPY_ALIGNMENT`, an assumption a probe run asks a
GPU about).

**Swizzle is a function of the shared-memory address**, not of the offset
into the buffer: CuTe's `Swizzle<B,4,3>` with B = 1, 2, 3 for 32B, 64B, 128B,
applied to the address. The two views agree when the buffer is aligned to the
swizzle period (256, 512, 1024 bytes), which is why misalignment is reported
(`tma-swizzle-misaligned`). Identical on a B200 for aligned and misaligned
buffers.

**Out of bounds.** Elements of a box outside the tensor are filled, zero or
NaN, and `complete_tx` counts the whole box. The NaN is the 16-bit pattern
`0x7FF7` repeated across the element (`0x7FF77FF7` for f32), as measured on a
B200 (`validation/blackwell/measured/b200/tma/oob_fill.txt`); it is not
documented, and it is not the NaN arithmetic produces.

**mbarrier.** Each mbarrier tracks pending arrivals and expected bytes for
its current phase; the phase completes when both are zero. Every lane that
executes an `arrive` arrives. Transaction counts are signed: completion bytes
can precede `expect_tx`, and a phase cannot complete until the count balances
to exactly zero. More arrivals than the phase needs is an error
(`mbarrier-over-arrival`): a B200 ends the kernel. Generic-address forms
accept pointers produced by `cvta.shared`. Use before `mbarrier.init` is an
error.

`try_wait` blocks the warp until the phase completes instead of returning
false, so a wait that can never succeed becomes a deadlock report. PTX lets
it return false after a system-dependent time limit (a B200 returned false
each of 17 times for a phase that never completes); the model takes that
limit as unbounded, and a kernel with a non-blocking fallback is outside v1
(matrix row MB-11). `test_wait` never blocks.

*Tokens.* The model encodes an arrival token with its phase parity in bit 63
and the low 32 bits of the phase number below. A state wait's predicate
compares bit 63 with the barrier's current parity; the `.parity` forms
compare bit 0 of their operand. This reproduces the predicates recorded on
the RTX 3080 Ti (`validation/ampere/mbarrier_token_golden.txt`), and that
program gave the same lines on the B200. The model's low token bits are for
diagnostics; hardware token bits are opaque and differ. A state token is
valid only in its own phase or the one after
([PTX, `mbarrier.test_wait` / `try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)).
Older or malformed tokens give `mbarrier-invalid-token` and no
synchronization credit, even when the parity fallback returns true; the
predicate keeps the measured fallback only so that exploratory kernels can
continue. The model cannot tell a token of another barrier, or of a
reinitialized one with the same phase number, and parity waits cannot tell
phases an even number of completions apart.

*A TMA copy against a smaller `expect_tx`.* For the instruction, the signed
count holds on hardware: `mbarrier.complete_tx 64` against `expect_tx 32`
goes from 32 to -32 and completes nothing until an `expect_tx 32` balances
it. A TMA copy is different: on a B200 a 256-byte load against `expect_tx
128` completes the phase, and the wait returns with the rest of the data
still on its way. The copy reports its bytes in pieces and the count passes
through zero. `Mbarrier.deliver` models this: a copy larger than the bytes
still expected completes the phase, charges the rest to the next phase, and
reports `mbarrier-tx-mismatch`. The copy's data is still written whole, so a
kernel with this bug computes the right result in the simulator where a GPU
may read a half-written tile; the finding is what says so. The piece size, 64
bytes (`TX_PIECE_BYTES`), is an assumption: the largest the measurements
allow. A count that is not a whole number of pieces is passed over, as it is
by the instruction.

**Bulk groups belong to each issuing thread.** A commit includes that
thread's uncommitted stores, including an empty group if there were none. A
wait with count N orders all but its N newest groups, regardless of which
copies happen to have completed. Another lane's commit or wait cannot
complete that protocol. The older per-thread `cp.async` with commit and wait
groups is tracked the same way.

**Checks do not depend on timing.** Shared memory records, per byte, the TMA
load that last targeted it and the TMA store that last read it. Each warp
keeps the set of copies it has synchronized with: by a successful mbarrier
wait, by `wait_group`, or by meeting another warp at a barrier or mbarrier
that already had. Touching bytes of a copy outside that set is reported, even
when the data happened to arrive in time. The tests run the broken kernels
with delays of 0, 4 and 1000 rounds and get the finding each time.

Barrier arrivals publish a snapshot of the copies already synchronized with
at arrival. An mbarrier acquire wait observes its completed phase, not copies
contributed to the next unfinished phase. Explicit `.relaxed` arrivals and
waits do not transfer that knowledge. It is kept per warp, not per thread: a
successful wait makes its copy visible to the whole simulated warp, so
cross-lane misuse after only some lanes wait can be missed.

**Fences.** An ordinary store marks its bytes as not yet visible to TMA.
`fence.proxy.async` clears the mark for the warp's own writes and for writes
that a completed barrier already ordered before it (that a fence after a
barrier covers other warps' earlier writes is an assumption). A TMA store of
marked bytes is reported. Its `.global` form does not fence shared memory.
Supported ordering fences and `fence.proxy.alias` have no additional modeled
effect; unsupported proxy modes raise `UnsupportedError`.

## 5. The tensor core and tensor memory

**Tensor memory** (`units/tcgen05.py`) is 128 lanes x 512 columns of 32 bits
per CTA; an address is `(lane << 16) | column`. `tcgen05.alloc` reserves a
power-of-two number of columns (at least 32) and writes the address to shared
memory. An allocation waits if tensor memory is full; another warp's
deallocation can release it. Deallocation must name the returned base address
and the original size. These rules follow the
[PTX allocation contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc),
which a B200 does not enforce in full: it allocated 64 columns after 32, 48
columns, and 32 columns after the permit was relinquished
([b200-rental-1.md](b200-rental-1.md)). warplens refuses the last two, which
the contract rules out in so many words, and makes and reports the first
(`tmem-alloc-increasing`). Tensor memory still allocated when the kernel
returns is `tmem-leak`, and under `warplens run` the launch fails with
`cudaErrorTensorMemoryLeak`, as on a B200. A repeated `dealloc` of the same
columns is ignored and reported (`tmem-dealloc-repeated`).

**`tcgen05.mma`** decodes the instruction descriptor (M, N, formats,
negation) and the two shared-memory descriptors (start, leading and stride
byte offsets, swizzle), reads A (M x K) and B (N x K) from shared memory in
K-major layout (or MN-major, for half precision), and writes or accumulates
`A * B^T` as float32. With `enable-input-d`, the optional `scale-input-d`
scales the previous accumulator by `2^-scale` (0 to 15). K is 32 bytes of
elements: 16 for half precision, 8 for TF32. Row m of the result is lane m
for M = 128, and lane `32 * (m // 16) + m % 16 + a` for M = 64 (Layout F),
where `a`, the lane of the destination address, must be 0 or 16. An operand
is swizzled by its absolute shared address; a non-zero matrix base offset in
the descriptor is refused. The docstring of `units/tcgen05.py` states each
rule and the CuTe header it comes from. How an element is summed (one sum per
MMA, cut toward zero, as a B200 was measured to do) is in
[numerics.md](numerics.md).

**MMA is asynchronous**, like TMA. It completes `async_delay` rounds after it
is issued and reads its shared-memory operands then. Each `tcgen05.commit`
tracks all prior MMAs from its issuing thread; another lane's commit cannot
cover them. Several commits can track the same earlier operations on
different barriers. A successful acquire wait, or a successful relaxed wait
followed by `tcgen05.fence::after_thread_sync`, makes those MMA results
visible to the waiting warp. Read-only and partial commits are rejected.

**`tcgen05.ld.32x32b.xN [addr]`** gives the thread on lane l the columns
starting at `column(addr)` of one tensor-memory lane in its warp's window.
Warp w of a warpgroup reaches lanes `32 * (w % 4) .. + 31` and nothing else.
The encoded row is a local 5-bit index, `lane(addr) % 32`, so the physical
row is `32 * (w % 4) + lane(addr) % 32 + l`. NVIDIA's
[CUTLASS `tcgen05_ld` documentation](https://docs.nvidia.com/cutlass/4.8.1/media/docs/pythonDSL/primitives.html#cutlass.experimental.primitives.tcgen05_ld)
documents this row aliasing. CuTe's per-warp base rows and TIRx's shared
row-zero base therefore reach the same cells. A transfer whose local base row
plus shape spans beyond row 31 is rejected as unsupported (on a B200 such
transfers faulted with `cudaErrorMisalignedAddress`). The `16x64b`, `16x128b`
and `16x256b` shapes cover 16 lanes and give each thread 1, 2 or 4 registers
per unit, spread over lanes and columns; `transfer_cells` in
`units/tcgen05.py` has the mapping, and
`python -m warplens.viz.tmem --transfer 16x128b 1` prints it. Loads and
stores require all 32 lanes of the warp and a uniform base address;
allocation, deallocation, waits and permit relinquishment also require the
full warp.

**What is not ordered.** Tensor-memory loads and stores complete at once, so
`tcgen05.wait::ld/st` validate warp participation and model no delay. A
missing `tcgen05.wait::ld` or `fence::before_thread_sync` is not diagnosed.

Generic-to-async proxy ordering for MMA operands in shared memory is not
diagnosed either. The
[PTX shared-memory access rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-shared-memory-accesses)
requires a proxy fence, while the pinned CUTLASS tutorial 1 uses ordinary
shared stores followed by `bar.sync` and an MMA, with no `fence.proxy.async`
in its generated PTX. The check would be cheap (`SharedMemory.unfenced`
already records the bytes), and it was left out because it would call
NVIDIA's reference kernel wrong on the strength of one sentence of the
manual. That tutorial's output was bit-identical on a B200, which shows the
sequence working there and does not settle whether the fence is required.
The sequence is neither diagnosed nor vouched for.

## 6. Clusters

**A launch is made of clusters.** `sim.launch(..., cluster=(4, 4, 1))` groups
the grid's CTAs; without it each CTA is a cluster of one. The engine
schedules a whole cluster, so warps of different CTAs of a cluster
interleave. Clusters run one after another: CTAs of different clusters never
run concurrently, so races between them on global memory are not explored. A
CTA's rank in its cluster is `x + nx * (y + ny * z)`. The special registers
`%cluster_ctarank`, `%cluster_nctarank`, `%cluster_ctaid`, `%cluster_nctaid`,
`%clusterid`, `%nclusterid` and `%is_explicit_cluster` are provided. Clusters
hold at most 16 CTAs, the width of a multicast mask. A program run under the
shim gets a cluster of more than 8 only for a kernel it allowed non-portable
cluster sizes (`cudaFuncAttributeNonPortableClusterSizeAllowed`, which
CUTLASS sets before every cluster launch), as on a B200.

**Cluster barrier.** `barrier.cluster.arrive` registers the warp and
snapshots the synchronization knowledge it releases (unless `.relaxed`);
`barrier.cluster.wait` blocks until every running warp of the cluster has
arrived. A duplicate arrival before the current phase completes is an error.
An aligned barrier reached by only part of a warp produces a finding;
divergent non-aligned cluster rendezvous is rejected.

**TMA multicast.** A load with `.multicast::cluster` and a 16-bit mask writes
the same box at the same shared-memory offset in every CTA whose rank is in
the mask, and performs `complete_tx` on each one's mbarrier at the given
offset. With `.cta_group::2` the barrier pointer can select either member of
the receiving pair; multicast completions use that pointer's CTA-rank parity
to choose the corresponding member of every receiving pair.
`tcgen05.commit ... .multicast::cluster` arrives on the mbarrier at that
offset in every CTA of the mask.

**Two-CTA MMA** (`cta_group::2`). Issued by one thread from either CTA of a
pair. Each CTA holds half of A's rows and half of B's rows in its own shared
memory at the same descriptor addresses, and receives its half of the result
in its own tensor memory:

- M = 256: 128 rows on 128 lanes per CTA, N columns.
- M = 128 (PTX's Layout B): 64 rows per CTA. They fill half the lanes, so the
  columns are folded: columns below N/2 on lanes 0-63, the rest on lanes
  64-127, in N/2 tensor-memory columns.

`tcgen05.alloc`, `dealloc` and `relinquish_alloc_permit` with `cta_group::2`
act on the issuing CTA's own tensor memory; both CTAs of a pair run them and
get the same columns.

**Distributed shared memory.** `mapa.shared::cluster` turns a shared address
into the cluster-wide address of the same location in another CTA.
`ld`/`st.shared::cluster` and `mbarrier.arrive.shared::cluster` accept such
addresses. Other mbarrier operations on a remote mbarrier are errors. The
encoding of cluster-wide addresses is warplens's own: shared symbols and
`mapa` pointers carry absolute CTA rank bits above the byte offset, and bit
24 selects the peer within a pair so that CUTLASS's `Sm100MmaPeerBitMask`
address arithmetic works.

**Synchronization knowledge** (which TMA copies and MMAs a warp may rely on)
crosses CTAs: operation ids are unique within a cluster, and the cluster
barrier and remote or multicast arrivals carry knowledge as local ones do.
Outstanding asynchronous operations are diagnosed at each CTA's exit, even
while its peers continue running.

**Cluster launch control** has a write-up of its own: [clc.md](clc.md).

## 7. The timing model

The model is approximate. The configs that ship with warplens hold assumed
values for most parameters; what a B200 measured, and why the fit of that
measurement is recorded and not applied, is in the guide (sections 2 and
6.7) and in [b200-rental-1.md](b200-rental-1.md). What the model is good for
is showing why a kernel is limited by what it is limited by, under stated
parameters.

Time is in nanoseconds. Each CTA is simulated alone on one SM
(`timing/model.py`):

| Resource | Behaviour | From |
| --- | --- | --- |
| issue slots | 4 per SM; warp w uses slot w mod 4; one instruction per clock | `clock_ghz`* |
| SMEM | one wavefront per clock, so bank conflicts cost time | `smem_wavefront_clk`* |
| TMA engine | a fixed latency per copy plus bytes at a fixed rate | `tma_latency_ns`*, `tma_bytes_per_clk`* |
| HBM | this SM's share of the device bandwidth, for bytes of lines no CTA fetched or stored before | `hbm_bandwidth_tb_s` (published STREAM measurement), `sm_count` |
| L2 | a fixed rate for bytes of lines some CTA of this launch already fetched or stored | `l2_bandwidth_per_sm_gb_s`* |
| Tensor Core | this SM's share of the device FLOP rate, plus a latency per MMA | `tensor_core_tflops_*`, `mma_latency_clk` (both published) |

\* assumed. Published values are from Jarmusch and Chandrasekaran,
"Microbenchmarking NVIDIA's Blackwell Architecture"
([arXiv:2512.02189v3](https://arxiv.org/html/2512.02189v3)): Table VII's
throughput and Table V's FP16 m128n128k16 latency. Applying that latency to
other shapes and to TF32 is an assumption. The assumed clock also scales the
shared-memory and tensor-memory costs, TMA throughput and MMA latency.

Each resource is a first-come-first-served server. A warp's instruction
starts when the warp and its issue slot are free. A TMA copy or an MMA queues
on its servers and completes later; warps waiting on its mbarrier resume
then. The timed engine (`timing/engine.py`) always advances whichever warp or
completion is earliest, so cause precedes effect. Functional results and
findings are the same as in an untimed run.

TMA loads, TMA stores and `cp.async` copies share the L2 record, kept per
128-byte line of global memory. The lines a copy reads or writes are charged
to L2 if an earlier copy left them there and to HBM otherwise, and are in L2
once the copy completes. So a store to lines just loaded, a repeated store,
and a `cp.async` from a warm source are charged to L2, and a load of lines a
store wrote is warm. `cp.async` is charged at the TMA engine's rates and
start-up latency, which approximates a separate hardware path; zero-filled
bytes count toward the shared-memory transfer, not toward HBM traffic.

`active_sms` says how many SMs share HBM. The default is all of them, as in a
problem that fills the device. It must be an integer from 1 to the config's
SM count. `warplens.timing.placement.place` gives the number a launch's own
shape implies (grid, cluster size, and how many CTAs of the kernel one SM
holds), assuming CTAs are spread over the SMs; the timing benchmarks'
predictions use it per launch (`validation/blackwell_timing/README.md`,
"Active SMs"). A launch does not use it by itself. Config rates and clocks
must be finite and positive; latency costs must be finite and nonnegative.
Invalid configurations raise `LaunchError` before execution, and unknown
override names are rejected.

The roofline's achieved rate is a **pipeline rate**: mean per-CTA MMA FLOPs
divided by the mean interval containing asynchronous work. It excludes kernel
setup and epilogue time; the report shows the complete kernel time
separately. The compute ceiling follows the MMA kinds actually executed; for
mixed FP16/TF32 work it divides the total FLOPs by the sum of each kind's
ideal service time at its configured rate. The memory ceiling counts every
byte as coming from HBM.

**Verdict.** A resource busy at least 85% of the time is named the
bottleneck. If none is, the kernel is called latency-bound. For the TMA
engine, HBM, L2 and the Tensor Core, "the time" is the interval in which
asynchronous work was in flight; the threshold is a choice, not a derived
number. Busy fractions and the pipeline window are averaged over the CTAs
that did work (used shared memory, a TMA or `cp.async` copy, or the Tensor
Core), so CTAs that start and return do not dilute them; the summary says how
many were left out, and `report.utilization_all_ctas` keeps the average over
every CTA.

**Why a pipeline's bottleneck does not pass through all three.** In steady
state a pipeline with S stages takes, per K tile, the largest of: the load
time, the multiply time, and the whole round trip divided by S. The first two
are throughput limits and do not depend on S. Adding stages only shrinks the
third. So the limit moves from "latency" to whichever throughput limit is
larger, and stays there. Which one is larger depends on the kernel's FLOPs
per byte and on how much memory bandwidth this SM gets
(`examples/04_pipelined_gemm` shows each of HBM, the TMA engine and the
Tensor Core as the limit in one of three scenarios).

**`b300.yaml`** is a configuration of its own. Three entries are NVIDIA's
stated figures for Blackwell Ultra and differ from B200: `sm_count` (160
against 148), `hbm_capacity_gb` (288 against 192) and `max_power_w` (1,400
against 1,200); the last two are recorded and not used by the model. No other
B300 figure is public, so every other entry is the B200 value with the source
"assumption: assumed equal to B200, uncalibrated", starred in the report.
The whole-GPU HBM bandwidth and Tensor Core rates are among them, so with
more SMs one B300 SM's share of each is 148/160 of a B200 SM's.

What the model leaves out:

- CTAs do not interact in time. Each starts at zero on its own SM; they share
  only the record of which bytes are already in L2, in launch order. CTAs
  within a cluster share pending L2-fill times; a fetch cannot become an L2
  hit in a peer before it completes.
- No L2 capacity, associativity or eviction: bytes fetched once count as
  cached. Within a CTA, an outstanding fetch does not count as a completed
  cache fill, and in-flight requests are not merged. A TMA store to lines L2
  does not hold is charged to HBM, as if written through; the later
  write-back of lines a store found in L2 is not charged to anything.
- A warp is charged a fixed wait for a global load and nothing for a store.
  An `atom` is charged as the load and then the store it stands for
  (assumed). Ordinary loads and stores neither consult nor update the L2
  record.
- The Tensor Core does not model the dependency of an accumulating MMA on the
  one before it, beyond queueing them.
- Tensor-memory loads and stores have a fixed cost (`tmem_access_clk`).
  `tcgen05.wait` costs only the issue clock every instruction costs.
- In a cluster, a multicast load is charged to each receiving SM as if it
  were its own copy, and a two-CTA MMA to the issuing SM only.
- One parameter per mechanism. The B200 fit shows where that is not enough:
  the latency of cluster launch control differs by form of the request, and
  an mbarrier arrive and wait costs about ten times what the model charges
  ([b200-rental-1.md](b200-rental-1.md), "The timing config is a decision").

## 8. Reports and views

A timed launch writes a trace and an HTML report:

```python
stats = sim.launch(ptx, "kernel", grid=..., block=..., args=[...], timing=True, active_sms=8)
t = stats.timing
print(t.summary())                 # time, utilization, verdict, roofline
t.write_trace("run.trace.json")    # open at ui.perfetto.dev
t.write_html("run.html")           # self-contained report
```

The trace has one process per CTA, a track per warp (run and wait slices, the
wait naming its instruction and, for an mbarrier, the barrier and phase),
tracks for the TMA engine and the Tensor Core, tracks for the busy time of
this SM's share of HBM and of L2, a track per mbarrier with an instant for
each completed phase, and a counter of TMA bytes in flight from issue through
completion, including copies queued for the TMA engine. Overlapping
asynchronous lifetimes use separate display lanes in the trace and the HTML
timeline. These lanes do not represent extra engines; they avoid crossing
duration events on one track, which the
[Chrome/Perfetto format does not permit](https://perfetto.dev/docs/getting-started/other-formats).
Every row of the report's resource and number tables links to the config
entries it was derived from, and each entry shows its source.

`warplens report` (and `warplens.viz.write_report`) puts every static view of
a run on one page:

| View | Module | What it draws | From |
| --- | --- | --- | --- |
| Findings | `check/report.py` | each finding with its PTX line | `stats.findings` |
| Bank heatmap | `viz/smem.py:bank_heatmap_svg` | a row per shared-memory instruction, a column per bank; a cell is coloured by the most distinct words one execution put in that bank relative to that execution's ideal wavefronts (within the ideal = no conflict, whatever the width of the access), and shows the words where they exceed it | `stats.smem.instructions[...].bank_peak`, `.bank_ratio` |
| SMEM layout | `viz/smem.py:layout_svg` | a logical tile beside the 128-byte lines of shared memory it occupies, per 4-byte word, coloured by bank; one picture per shape of TMA tensor copy | `stats.smem.tiles` |
| TMEM map | `viz/tmem.py:accumulator_svg`, `pair_svg` | where the rows and columns of each shape of MMA result land; both CTAs for `cta_group::2` (M = 256, and M = 128, the PTX manual's Layout B) | `stats.smem.mmas` |
| mbarrier timeline | `viz/html_report.py:timeline_svg` | warps, TMA engine, Tensor Core, HBM and L2 of the first CTA over simulated time, then a row per mbarrier: a numbered mark per phase completion, joined to the waits it ended | `stats.timing` |

Every picture is inline in the page and written beside it as an SVG file; a
timed launch also gets its full timing report (`*.timing.html`) and its trace
(`*.trace.json`), linked from the page. `python -m warplens.viz.smem` and
`python -m warplens.viz.tmem` draw the two layout views without a run. The
guide's sections 6.5, 6.6 and 6.8 show all of them in use, and
[gallery.md](gallery.md) reads three of them step by step.

What the recording costs: `SmemReport` (`units/smem_banks.py`) keeps
`bank_peak` per instruction and two dictionaries, `tiles` and `mmas`, filled
by one call each in `units/tma.py` and `units/tcgen05.py`. Nothing in the
simulator reads them.

Not done:

- The SMEM layout view draws tiles placed by TMA tensor copies, or described
  by hand. A layout a kernel computes itself (the transpose example's padded
  and XOR-swizzled tiles) is not recovered from a run, and neither are the
  operands an MMA descriptor addresses or raw `cp.async.bulk` ranges.
- The pictures are static SVG with tooltips; there is no dark theme and no
  interaction.
- The timeline shows the first CTA only. The HBM and L2 rows show when this
  SM's share of each was busy with a copy, not a device-wide picture.
- A wait that an mbarrier ends is joined to that barrier's phase. A wait on a
  named barrier (`bar.sync`), a bulk group or a tensor-memory allocation
  names its kind and its instruction only.
- Layout A/B views reject nonzero lane offsets. Layouts for `.ws` MMAs and
  structured sparsity do not exist.
- Launches made through the Python API inside a script are not seen by
  `warplens report`; the script calls `write_report`.
- There is no `warplens check` command.

## 9. `warplens run`: the shim and the front ends

### Compiled programs

A program built by `nvcc` normally contains machine code for a real GPU and
links the CUDA runtime statically. Two things have to change, both at build
time and neither in the source:

```
nvcc -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared prog.cu -o prog
```

- `code=compute_100a` embeds PTX instead of machine code; `--no-compress`
  keeps it readable.
- `--cudart shared` links `libcudart.so`, which `warplens run` replaces.

`warplens run prog` starts the program with two small shim libraries
preloaded (`warplens/shim/`, C, built on first use against the installed CUDA
headers). They implement the runtime calls programs make and forward each one
over a socket to the `warplens run` process, which executes it in the
simulator:

| What the program does | What happens |
| --- | --- |
| registers its fat binary and kernels (compiler-generated) | the PTX is extracted and parsed; parameter sizes are read from it |
| `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaFree` | simulated global memory; freed addresses are never reused |
| `kernel<<<grid, block, smem>>>(args)`, `cudaLaunchKernelExC` | a launch, with the cluster shape if one is given; argument bytes are copied as they are |
| `cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled")`, then calling it | a warplens tensor map, validated like the real call |
| `printf` in a kernel | printed by the program, in order with its own output |
| device and function queries | a simulated B200: compute capability 10.0, 148 SMs (10.3 and 160 for B300) |
| loads a module itself: `cuModuleLoadData`, `cudaLibraryLoadData`, then `cuModuleGetFunction` or `cudaLibraryGetKernel` | PTX text or a fat binary with PTX is parsed. Machine code is accepted only if the compiler kept the PTX (the CuTe DSL, below) |
| `cuLaunchKernel`, `cuLaunchKernelEx`, `cudaLaunchKernelExC` with such a kernel | a launch, as above |
| streams, events, `cudaMallocHost` | accepted; everything is synchronous and an elapsed time is zero |

**Errors.** An error in a kernel (a `MemoryFault`, a deadlock, unsupported
PTX) is printed and reaches the program as `cudaErrorLaunchFailure`, from the
launch and from the synchronize calls after it. A call to `cudaGetLastError`
clears the simulator's remembered kernel fault so another experiment can run.
This is a simulator recovery policy: real CUDA launch failures may leave the
process or context unusable.

A launch that is refused before the kernel runs is printed too and has the
code the CUDA runtime gives it: `cudaErrorInvalidValue` for more shared
memory than 227 KB per CTA or more dynamic shared memory than the kernel was
opted in to, `cudaErrorInvalidConfiguration` for a block of more than 1024
threads or a zero dimension, block dimensions above 1024 x 1024 x 64, or grid
dimensions above 2147483647 x 65535 x 65535, `cudaErrorInvalidClusterSize`
for a cluster of more than 16 CTAs, one of more than 8 for a kernel that was
not allowed non-portable cluster sizes, or a grid that is not a whole number
of clusters. A kernel that returns with tensor memory still allocated gives
`cudaErrorTensorMemoryLeak`. Such an error is returned by the launch call and
once by `cudaGetLastError`; `cudaDeviceSynchronize` does not repeat it.
Through the driver API the first two are `CUDA_ERROR_INVALID_VALUE` and the
third is `CUDA_ERROR_INVALID_CLUSTER_SIZE`. The codes and where they show up
were measured on an RTX 3080 Ti at its own limits
([validation/README.md](../validation/README.md), "Launch errors"). On a B200
the two cluster-size refusals and the tensor-memory leak were measured
(`validation/blackwell/measured/b200/cluster/`, `cluster-optin/`,
`tmem/alloc_leak.stderr.txt`), and the probes of the shared-memory limit and
of the device attributes were identical to their predictions.

**The shared-memory opt-in.** Like a GPU, the shim requires it. A kernel may
use 48 KB of shared memory, static plus dynamic; a launch with more dynamic
shared memory than that leaves is refused with `cudaErrorInvalidValue` unless
the program first called
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n)`
or the driver's `cuFuncSetAttribute` with n at least the dynamic size of the
launch. The attribute belongs to one kernel, replaces the default (a smaller
n lowers it), and is what `cudaFuncGetAttributes` reports as
`maxDynamicSharedSizeBytes`. A kernel whose static shared memory alone is
more than 48 KB needs no opt-in and may use all the dynamic shared memory
that is left of 227 KB; for such a kernel `maxDynamicSharedSizeBytes` reads 0
until the program sets it, as on a B200 (`reported_max_dynamic` in
`warplens/units/smem.py`). The rule was measured on the RTX 3080 Ti at its
own limits and held on the B200, apart from that reported 0.
`Device.launch` of the Python API takes the shared memory of the launch and
has no opt-in step, and neither has the Triton driver.

**Building the shim.** The libraries are built from `warplens/shim/` by
`build.sh`, which records next to them what they were built from
(`warplens-shim.stamp`: a digest of the C sources, `protocol.h` and
`build.sh`, and one of the CUDA headers). Every `warplens run` compares that
with the sources it has, once, before the program starts. A build made by
hand in `warplens/shim/lib/` that is of other sources is rebuilt in place,
with a line that says so; if it cannot be rebuilt (no compiler, no headers)
the run stops with the reason instead of loading it. Builds in the user cache
are kept per source and toolkit and checked the same way.

### Triton

`warplens.triton_driver.install("B200")` makes Triton compile for `sm_100a`
and launch into the simulator. Triton's bundled toolchain does the compiling;
no GPU or CUDA driver is involved. `kernel[grid](...)` then works as usual,
with tensors created by `gpu.tensor(numpy_array)` and `gpu.empty(shape,
dtype)`. `warplens run script.py` installs the driver and runs the script.

`uninstall()` restores the driver state preceding `install()` without probing
for a physical GPU; installs may be nested. `active_driver()` returns the
active warplens driver, or `None`, without initializing Triton's hardware
backend. Tensors belong to the driver that allocated them. Kernel caches are
separate from real CUDA devices. Tuple arguments (including nested
constexprs), launch hooks, empty grids and `tl.device_print` are supported.
With `num_ctas > 1`, grid dimensions count Triton programs: each program
launches a cluster of `num_ctas` physical CTAs. The compiler target is
`sm_100a` for both models.

Triton's matmul compiles to a `tcgen05.mma` on this target, so it exercises
the same tensor core as the CUTLASS examples by a different code generator.

### Runtimes that launch for themselves: TIRx and the CuTe DSL

Both were traced on an RTX 3080 Ti with a debugger breaking on every export
of `libcuda`:

| | TIRx (`apache-tvm-tirx` 0.1.0) | CuTe DSL (`nvidia-cutlass-dsl` 4.8.1) |
| --- | --- | --- |
| Who launches | TVM's runtime library, linked against `libcudart` and `libcuda` | host code the DSL compiles, calling wrappers in `libcute_dsl_runtime.so` |
| CUDA runtime | the shared `libcudart.so.13` | NVIDIA's, linked into `libcute_dsl_runtime.so`; cuda-python has another copy |
| Calls for a launch | `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncGetAttribute`, `cuDeviceGetAttribute`, `cuLaunchKernelEx`; memory through `cudaMalloc` and `cudaMemcpyAsync` | `cudaLibraryLoadData`, `cudaLibraryGetKernel`, `cuKernelGetAttribute`, `cudaKernelSetAttributeForDevice`, `cudaLaunchKernelExC` |
| What the driver is handed | machine code by default; PTX text if the script asks (`target_format`) | a fat binary with one entry of machine code, never PTX |

A copy of the CUDA runtime that is linked into a library cannot be replaced
by preloading, and it will not start on the `libcuda` shim: it asks the
driver for private tables (`cuGetExportTable`) and gives up without them. The
two runtimes are reached differently instead.

**TIRx.** TVM's runtime uses the shared libraries, so the two shims serve it.
It asks the device for its compute capability, gets 10.0 (10.3 for `--model
B300`) and compiles for `sm_100a` (`sm_103a`) with NVRTC, in process. The
script has to ask for PTX,

```python
with T.device_entry(launch=..., backend_config={"cuda": {"target_format": "ptx"}}):
```

because the default is machine code; a module of machine code is refused with
that advice. The same script runs on a GPU, whose driver takes PTX too,
provided the driver is at least as new as the NVRTC that wrote the PTX.
Arrays are TVM's own (`tvm_tirx.runtime.tensor`), which use `cudaMalloc`.
TIRx compiles with fast-math, hence `.ftz` arithmetic, and `T.cuda.printf`
becomes a device function that is called.

**The CuTe DSL.** `warplens run` arranges three things, all through the DSL's
own settings, and the script is not involved:

1. *The runtime library.* The DSL also ships its wrappers as a static archive
   that leaves the CUDA runtime to the linker.
   `warplens/shim/build_cute_dsl.sh` links that archive against the
   `libcudart` shim, and `CUTE_DSL_LIBS` makes the DSL launch through the
   result. It is built on first use from the installed package; nothing is
   copied. Runtime functions the archive names and the shim lacks (30 of
   them: graphs, managed and asynchronous memory, and others) fail when
   called.
2. *The PTX.* `CUTE_DSL_KEEP=ptx` and `CUTE_DSL_DUMP_DIR` make the DSL write
   the PTX of every kernel it compiles. The module the host code loads is
   machine code; the server accepts it and finds each kernel by name in the
   newest kept file that defines it at module-load time. Sources are frozen
   then, so a later compilation with the same kernel name cannot change an
   already loaded module. `CUTE_DSL_ARCH` is set to `sm_100a` or `sm_103a`.
   Keeping artifacts turns off the DSL's compile cache.
3. *The device count.* Before a launch the DSL's Python asks cuda-python's
   runtime module how many devices there are. That module is the second
   built-in runtime. A `sitecustomize.py` put first on `PYTHONPATH` answers
   that one call; it runs any `sitecustomize` it hides.

Settings the user has already made are left alone, except that `ptx` is added
to `CUTE_DSL_KEEP`.

*Arrays.* DSL examples hold their arrays in PyTorch tensors on the GPU, and
PyTorch does not run on the shims. `examples/06_cute_dsl/saxpy.py` holds them
the way the DSL does underneath: device memory from the driver, through the
DSL's own helpers (`cutlass.base_dsl.runtime.cuda`), passed as pointers
(`make_ptr`). Those helpers use cuda-python's driver module, which finds
every driver function by name (`cuGetProcAddress`); the `libcuda` shim has
the context, memory and stream calls it then makes. An array from any library
that allocates through the shared CUDA runtime and speaks DLPack also works
with `from_dlpack`; TVM's tensors do.

*Tensor maps.* The DSL does not ask the driver for its tensor maps; its
compiled host code writes the 128-byte descriptor itself, in the driver's
layout, which the simulator reads (section 4). The kernel of CUTLASS's
`fp16_gemm_0.py`, captured on a B200 with its parameters and memory, replays
with no byte of its output different from the GPU's
(`validation/blackwell/e2e/replay_launch.py`,
[hardware-capture.md](hardware-capture.md)). The tutorial as a script has not
been run under `warplens run`: it holds its arrays in PyTorch.

### Limits of `warplens run`

The guide's section 8 has the list: the build flags, the sixty or so runtime
and driver functions the shim has, statically linked runtimes, and what each
front end cannot do. Everything is synchronous and single-device. Context
handles name one simulated context; nested pushes and pops preserve the
per-thread current-context stack. The runtime `cudaLibrary*` API used by the
CuTe DSL needs CUDA 12.8 or newer headers; the driver shim also builds with
12.4.
