# PTX coverage

What warplens executes today. Anything not listed raises `UnsupportedError`
that names the instruction and its PTX line.

## Module and function structure

| Construct | Status |
| --- | --- |
| `.version`, `.target`, `.address_size` | parsed |
| `.entry` kernels with scalar `.param`s (`.ptr`, `.align` accepted) | yes |
| `.entry` declarations without a body | no; rejected explicitly |
| `.reg` declarations, including `%r<N>` ranges and `.b128` registers | yes |
| decimal floating-point literals, including `1e-3`, `.5`, and `1.` | yes |
| labels, `@%p` / `@!%p` predication on any instruction | yes |
| performance directives before the body (`.maxntid`, ...), `.pragma` | ignored |
| `.shared` variables, in a function or at module level, and `.extern .shared name[]` / `name[0]` | yes; alignments and sizes accept PTX integer literals |
| paired destinations (`%r1\|%p1`) | yes |
| nested `{ }` blocks with their own registers, shared/local variables and labels (inline assembly) | yes |
| DWARF debug directives (`.loc`, `.file`, `.section .debug_* { }`), as Triton emits | skipped |
| a scalar written as a one-element `{ %r1 }` operand | yes |
| register names without `%`, `_` as a discarded destination, `[tensor_map, {coords}]` operands | yes |
| `.global` / `.const` variables with numeric initializers, `.local` variables (one copy per thread) | yes |
| structs passed by value (`.param .align 8 .b8 name[24]`, read with `ld.param` at an offset) | yes; pass bytes, a NumPy record or a ctypes structure |
| `.extern .func` declarations and `call` | only `vprintf` (device `printf`); its text is returned in `stats.output` |
| `.func` definitions and calls to them | yes, with scalar parameters and results (see Control below); they are in `module.device_functions`, apart from the kernels |
| `.tex` | no |

## Instructions

| Group | Instructions | Notes |
| --- | --- | --- |
| Moves | `mov` (including the address of a variable, and packing or unpacking a `{vector}`), `cvta[.to].{global,const,shared,local,param}` | `mov.b128` moves a 128-bit register, loads an integer literal, and packs or unpacks two 64-bit halves; nothing else computes with `.b128` |
| Integer | `add`, `sub`, `mul.{lo,hi,wide}`, `mad.{lo,hi,wide}`, `div`, `rem`, `min`, `max`, `neg`, `abs` | no `.sat`, no carry forms |
| Float | `add`, `sub`, `mul`, `div`, `fma`, `mad`, `min`, `max`, `neg`, `abs`, `sqrt` | round-to-nearest only; `fma` and floating `mad` round once; `.ftz` on `.f32` for `add`, `sub`, `mul`, `div`, `fma`, `sqrt` (what fast-math emits), checked on a GPU |
| Bitwise | `and`, `or`, `xor`, `not`, `shl`, `shr` | also on `.pred` (not shifts) |
| Bit fields | `bfe.{u32,s32,u64,s64}`, `bfi.{b32,b64}`, `shf.{l,r}.{wrap,clamp}.b32`, `prmt.b32` (default mode) | all checked on a GPU (`validation/ampere/bitfield_golden.txt` for `bfi` and `shf`, which nvcc 12.4 emits and 13.5 for `sm_100a` does not); no other `prmt` modes |
| Matrix load | `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` | also `.shared::cta` or a generic shared pointer; the six layouts were checked on a GPU; bank counts use one 128-byte transaction per matrix |
| Async copy | `cp.async.{ca,cg}.shared.global` with optional source size, `cp.async.commit_group`, `cp.async.wait_group` | zero-fill past the source size checked on a GPU |
| Compare | `setp` with `eq ne lt le gt ge lo ls hi hs`, unordered `equ neu ltu leu gtu geu`, `num`, `nan`; `selp` | no combined-predicate or paired-destination forms |
| Convert | `cvt` int↔int, int↔float, float↔float, float→int with `.rni/.rzi/.rmi/.rpi` (saturating), float to an integer in the same floating type with the same four modes (`floorf`, `ceil`, ...); `.ftz` on float→float and float→int when the source or the result is `.f32` (what fast-math emits: `cvt.ftz.f64.f32`, `cvt.rn.ftz.f32.f64`, `cvt.rzi.ftz.s32.f32`, `cvt.rmi.ftz.f32.f32`); `cvt.{rna,rn,rz}.tf32.f32`; `cvt.rn.f16x2.f32` (two float32 to half precision in one register, the first source in the high half) | `.ftz` and the float→float forms checked on a GPU around the subnormal range (`validation/ampere/cvt_ftz_golden.txt`); see `numerics.md`, "Flush to zero". Float→float rounds to nearest only (no `.rz`, `.rm`, `.rp`); no `.sat`; no `.ftz` with an integer source |
| Convert, small formats | what `cuda_fp8.h` and `cuda_fp4.h` emit for `sm_100a` / `sm_103a`, two values at a time: `cvt.rn.satfinite.{e4m3x2,e5m2x2,e2m1x2}.{f32,f16x2,bf16x2}`, `cvt.rn.{f16x2,bf16x2}.{e4m3x2,e5m2x2,e2m1x2}`, `cvt.{rz,rp}[.satfinite].ue8m0x2.{f32,bf16x2}`, `cvt.rn.bf16x2.ue8m0x2` | agrees with the headers' own host implementation on every tested input; not compared with a GPU, so NaN results and the E8M0 edge cases are the headers' (listed in `warplens/units/small_formats.py`). No `ue5m3x2`, `e2m3x2`, `e3m2x2`, `.rz` to `e2m1x2`, or forms without `.satfinite` for E4M3/E5M2/E2M1; no scalar `.bf16` |
| Memory | `ld.param`, `ld.const`, `st.param` (call arguments), `ld`/`st` with `.global`, `.shared`, `.local` or no space (generic), scalar or `.v2`/`.v4` | parameter pointers can be loaded through `.param`, `.global` or generic addresses; `.b128` loads and stores; `ld.weak`, `ld.relaxed.<scope>` and `ld.acquire.<scope>` load like a plain `ld`; `ld.volatile` and `st.volatile` (a C++ `volatile` object) behave like the plain forms |
| Atomic updates | `atom` with `add.{s32,u32,u64,f32,f64}`, `{and,or,xor,exch,cas}.{b32,b64}`, `{inc,dec}.u32`, `{min,max}.{s32,u32,s64,u64}`; `red` with the same operations and types except `exch` and `cas`; global, shared and generic addresses; omitted/relaxed ordering and cta/gpu/sys scope | Every spelling nvcc 13.5 (`sm_100a`) and nvcc 12.4 (`sm_80`, `sm_86`) emit for the plain, `_block` and `_system` atomic functions: `atom.relaxed.gpu.global.add.u32`, the older `atom.global.add.u32` / `atom.shared.add.u32` / `atom.add.u32` without ordering or scope (PTX: relaxed, gpu), and the `red.*` forms 12.4 uses when the result is discarded. Generic remote shared pointers. Deterministic lane serialization; naturally aligned accesses required. No acquire/release, cluster scope, half/bfloat/vector or 128-bit forms. Supported spellings, FTZ rules, diagnostics and timing limits: [`atomics.md`](atomics.md). Recorded Ampere results: `validation/ampere/atom_golden.txt`, compared with the probe as compiled by both toolkits |
| Election | `elect.sync d\|p, membermask` | the lowest participating lane is elected |
| Vote | `vote.sync.{all,any,uni}.pred d, {!}a, membermask`, `vote.sync.ballot.b32` (`__all_sync`, `__any_sync`, `__uni_sync`, `__ballot_sync`); `vote.{all,any,uni}.pred d, {!}a` and `vote.ballot.b32` without `.sync` | a vote is among the lanes that execute the instruction together and are in the caller's mask. `vote.sync` waits for the lanes of its mask: the other paths run until they arrive or return, and a mask that names a lane which can do neither is an error. Checked on a GPU, both spellings (`validation/ampere/vote_golden.txt`, `vote_legacy_golden.txt`; `validation/README.md`). ptxas refuses the forms without `.sync` for sm_70 and newer; the simulator runs them for any target. A caller outside its own mask is reported as `vote-membermask-mismatch` |
| Barriers | `bar.sync a[, count]`, `bar.arrive a, count`; also spelled `barrier` and `.cta`, including `barrier[.cta].{sync,arrive}.aligned` | no `bar.red`; sync count zero means the whole CTA |
| Warp barrier | `bar.warp.sync membermask` (`__syncwarp()`) | waits for the lanes of its mask, as `vote.sync` does: the other paths run until they arrive or return, and a mask that names a lane which can do neither is an error. Measured on a GPU (`validation/ampere/sync_other_3080ti.txt`; `validation/README.md`). A mask that leaves out its caller, or members with another mask, are reported as `warp-sync-membermask-mismatch` |
| Shuffle | `shfl.sync.{up,down,bfly,idx}.b32` with optional predicate destination | waits for the lanes of its mask in the same way (the same measurement). A source lane that has returned, or is not executing the shuffle, gives 0 and is reported; see `architecture.md`, section 3 |
| mbarrier | `mbarrier.init`, `.arrive` (with count), `.arrive.expect_tx`, `.expect_tx`, `.complete_tx`, `.test_wait`, `.try_wait` (both with `.parity`), `.inval` | no `arrive_drop`, `.noComplete`, `pending_count` |
| TMA | `cp.async.bulk.tensor.{1..5}d` load (`shared <- global`, `.mbarrier::complete_tx::bytes`) and store (`global <- shared`, `.bulk_group`), tile mode; `cp.async.bulk` without a tensor map; `.commit_group`, `.wait_group[.read]`; `prefetch.tensormap` and `cp.async.bulk.prefetch` (no-ops); `.multicast::cluster` (see Clusters) | no im2col, `cp.reduce`, `tensormap.replace`; `.L2::cache_hint` ignored |
| tcgen05 | `tcgen05.alloc`, `.dealloc`, `.relinquish_alloc_permit`; `.mma` `kind::f16` / `kind::tf32` with K-major shared-memory operands, plus MN-major for `kind::f16`, and float32 accumulators, M = 64 or 128, optional accumulator scaling, optional `.collector::a::discard` (which is what an MMA without it does); per-thread `.commit` to an mbarrier; `.ld` / `.st` shapes `32x32b`, `16x64b`, `16x128b`, `16x256b` with supported power-of-two `.xN`; `.wait::ld/st` and `.fence::*`; `cta_group::1`, and `cta_group::2` as listed under Clusters | no TMEM operand A, no collector (`.collector::a::fill`, `::use`, `::lastuse`), TF32 MN-major (requires the unimplemented 128B swizzle with 32B atomicity), other kinds, `16x32bx2`, `.pack`/`.unpack`, `.cp`, `.shift`, `.exclusive`, or partial commits; wait/fence model limitations are in `architecture.md`, section 5 |
| Fences | `fence.proxy.async[.global/.shared::cta/.shared::cluster]`; `fence.mbarrier_init.release.cluster`, `fence.{acq_rel,sc}.{cta,cluster,gpu,sys}`, `fence.proxy.alias` | only async fences affecting shared memory change checker state; other proxy modes are unsupported, including the `fence.proxy.async::generic...sync_restrict` pair of the manual's `clusterlaunchcontrol` example |
| Control | `bra`, `bra.uni`, `ret`, `exit`; `call` to a `.func` defined in the module: scalar and by-value struct arguments and results (`.param .align 16 .b8 param0[40]`), under a predicate (`@%p call`), through a register with `.callprototype` or `.calltargets`, and recursive | calls are inlined before the kernel runs, each call site getting its own copy of the callee's locals. An indirect call is a comparison with the address of each function it could mean, so it can only reach functions of the module; another address is an error. A recursive call saves the function's registers and gives it a new frame of local memory at run time; more than 1024 recursive calls active at once in a thread is a "call stack overflow" error (`warplens.units.call.MAX_CALL_DEPTH`; a GPU's limit is a stack size in bytes, which is not modeled). All but the predicated form are checked on a GPU. No `brx.idx`; no recursion through a function that contains `exit` or an endless loop |

## Special registers

`%tid`, `%ntid`, `%ctaid`, `%nctaid` (each with `.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid`; for clusters `%clusterid`, `%nclusterid`,
`%cluster_ctaid`, `%cluster_nctaid` (each with `.x/.y/.z`), `%cluster_ctarank`,
`%cluster_nctarank`, `%is_explicit_cluster`.

`%clock64`, with its halves `%clock` and `%clock_hi` (`clock64()` and `clock()`
both compile to `%clock64`). The counter is deterministic and is the simulator's
own, not a prediction of what a GPU would read: in a timed run it is the model's
time when the instruction issues, in whole clocks (nanoseconds times
`clock_ghz`); otherwise it is the number of scheduling rounds since the launch
began, where in each round every warp that can run executes one instruction.
Every lane of a warp reads the same value.

## Clusters

| Group | Instructions | Notes |
| --- | --- | --- |
| Cluster barrier | `barrier.cluster.arrive`, `barrier.cluster.wait` (with `.aligned`) | |
| Distributed shared memory | `mapa.shared::cluster`, `mapa.u64` on generic addresses, `ld`/`st.shared::cluster`, `mbarrier.arrive.shared::cluster` | other mbarrier operations cannot be remote |
| TMA | `.multicast::cluster` on tensor loads, also with `.cta_group::2`, and on raw `cp.async.bulk` loads | `.cta_group` only on tensor loads |
| tcgen05 | `cta_group::2` forms of `alloc`, `dealloc`, `relinquish_alloc_permit`, `mma` (M = 128 or 256), `commit`; `commit` with `.multicast::cluster` | |
| Cluster launch control | `clusterlaunchcontrol.try_cancel.async[.shared::cta].mbarrier::complete_tx::bytes[.multicast::cluster::all].b128`; `clusterlaunchcontrol.query_cancel` with `.is_canceled.pred.b128`, `.get_first_ctaid.v4.b32.b128` and `.get_first_ctaid::{x,y,z}.b32.b128` | clusters start one at a time, and a request cancels the next one that has not started; see `clc.md` for the model, its `clc_limit` control and what is assumed |
