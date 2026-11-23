# PTX coverage

What warplens executes today. Anything not listed raises `UnsupportedError`
that names the instruction and its PTX line.

## Module and function structure

| Construct | Status |
| --- | --- |
| `.version`, `.target`, `.address_size` | parsed |
| `.entry` kernels with scalar `.param`s (`.ptr`, `.align` accepted) | yes |
| `.entry` declarations without a body | no; rejected explicitly |
| `.reg` declarations, including `%r<N>` ranges | yes |
| decimal floating-point literals, including `1e-3`, `.5`, and `1.` | yes |
| labels, `@%p` / `@!%p` predication on any instruction | yes |
| performance directives before the body (`.maxntid`, ...), `.pragma` | ignored |
| `.shared` variables, in a function or at module level, and `.extern .shared name[]` / `name[0]` | yes; alignments and sizes accept PTX integer literals |
| paired destinations (`%r1\|%p1`) | yes |
| nested `{ }` blocks with their own registers, shared/local variables and labels (inline assembly) | yes |
| register names without `%`, `_` as a discarded destination, `[tensor_map, {coords}]` operands | yes |
| `.global` / `.const` variables with numeric initializers, `.local` variables (one copy per thread) | yes |
| structs passed by value (`.param .align 8 .b8 name[24]`, read with `ld.param` at an offset) | yes; pass bytes, a NumPy record or a ctypes structure |
| `.extern .func` declarations and `call` | only `vprintf` (device `printf`); its text is returned in `stats.output` |
| `.func` definitions, calls to other functions, `.tex` | no |

## Instructions

| Group | Instructions | Notes |
| --- | --- | --- |
| Moves | `mov` (including the address of a variable, and packing or unpacking a `{vector}`), `cvta[.to].{global,const,shared,local,param}` | |
| Integer | `add`, `sub`, `mul.{lo,hi,wide}`, `mad.{lo,hi,wide}`, `div`, `rem`, `min`, `max`, `neg`, `abs` | no `.sat`, no carry forms |
| Float | `add`, `sub`, `mul`, `div`, `fma`, `mad`, `min`, `max`, `neg`, `abs`, `sqrt` | round-to-nearest only; `fma` and floating `mad` round once |
| Bitwise | `and`, `or`, `xor`, `not`, `shl`, `shr` | also on `.pred` (not shifts) |
| Compare | `setp` with `eq ne lt le gt ge lo ls hi hs`, unordered `equ neu ltu leu gtu geu`, `num`, `nan`; `selp` | no combined-predicate or paired-destination forms |
| Convert | `cvt` int↔int, int↔float, float↔float, float→int with `.rni/.rzi/.rmi/.rpi` (saturating); `cvt.{rna,rn,rz}.tf32.f32` | no `.sat` on int→int |
| Memory | `ld.param`, `ld.const`, `st.param` (call arguments), `ld`/`st` with `.global`, `.shared`, `.local` or no space (generic), scalar or `.v2`/`.v4` | parameter pointers can be loaded through `.param`, `.global` or generic addresses |
| Election | `elect.sync d\|p, membermask` | the lowest participating lane is elected |
| Barriers | `bar.sync a[, count]`, `bar.arrive a, count`; also spelled `barrier` and `.cta`, including `barrier[.cta].{sync,arrive}.aligned` | no `bar.red`; sync count zero means the whole CTA |
| Shuffle | `shfl.sync.{up,down,bfly,idx}.b32` with optional predicate destination | no rendezvous across divergent paths; see `m1.md` |
| mbarrier | `mbarrier.init`, `.arrive` (with count), `.arrive.expect_tx`, `.expect_tx`, `.complete_tx`, `.test_wait`, `.try_wait` (both with `.parity`), `.inval` | no `arrive_drop`, `.noComplete`, `pending_count` |
| TMA | `cp.async.bulk.tensor.{1..5}d` load (`shared <- global`, `.mbarrier::complete_tx::bytes`) and store (`global <- shared`, `.bulk_group`), tile mode; `cp.async.bulk` without a tensor map; `.commit_group`, `.wait_group[.read]`; `prefetch.tensormap` and `cp.async.bulk.prefetch` (no-ops) | no multicast, im2col, `cp.reduce`, `tensormap.replace`; `.L2::cache_hint` ignored |
| tcgen05 | `tcgen05.alloc`, `.dealloc`, `.relinquish_alloc_permit` (`cta_group::1`); `.mma` `kind::f16` / `kind::tf32` with K-major shared-memory operands and float32 accumulators, M = 64 or 128, optional accumulator scaling; per-thread `.commit` to an mbarrier; `.ld` / `.st` shapes `32x32b`, `16x64b`, `16x128b`, `16x256b` with supported power-of-two `.xN`; `.wait::ld/st` and `.fence::*` | no `cta_group::2`, TMEM operand A, MN-major, other kinds, `16x32bx2`, `.pack`/`.unpack`, `.cp`, `.shift`, `.exclusive`, or partial commits; wait/fence model limitations are in `m3.md` |
| Fences | `fence.proxy.async[.global/.shared::cta/.shared::cluster]`; `fence.mbarrier_init.release.cluster`, `fence.{acq_rel,sc}.{cta,cluster,gpu,sys}`, `fence.proxy.alias` | only async fences affecting shared memory change checker state; other proxy modes are unsupported |
| Control | `bra`, `bra.uni`, `ret`, `exit` | no `brx.idx`, no `call` |

## Special registers

`%tid`, `%ntid`, `%ctaid`, `%nctaid` (each with `.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid`; for clusters `%clusterid`, `%nclusterid`,
`%cluster_ctaid`, `%cluster_nctaid` (each with `.x/.y/.z`), `%cluster_ctarank`,
`%cluster_nctarank`, `%is_explicit_cluster`.

## Clusters

| Group | Instructions | Notes |
| --- | --- | --- |
| Cluster barrier | `barrier.cluster.arrive`, `barrier.cluster.wait` (with `.aligned`) | |
| Distributed shared memory | `mapa.shared::cluster`, `ld`/`st.shared::cluster`, `mbarrier.arrive.shared::cluster` | other mbarrier operations cannot be remote |
| TMA | `.multicast::cluster` on tensor loads, also with `.cta_group::2` | not on raw `cp.async.bulk` |
| tcgen05 | `cta_group::2` forms of `alloc`, `dealloc`, `relinquish_alloc_permit`, `mma` (M = 128 or 256), `commit`; `commit` with `.multicast::cluster` | no `clusterlaunchcontrol` |
