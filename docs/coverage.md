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
| `.shared` variables, in a function or at module level, and `.extern .shared name[]` | yes |
| paired destinations (`%r1\|%p1`) | yes |
| nested `{ }` blocks with their own registers and labels (inline assembly) | yes |
| register names without `%`, `_` as a discarded destination, `[tensor_map, {coords}]` operands | yes |
| `.func` and `call`, `.global` / `.const` / `.local` variables | no |
| array parameters | no |

## Instructions

| Group | Instructions | Notes |
| --- | --- | --- |
| Moves | `mov` (including the address of a `.shared` variable), `cvta[.to].{global,shared}` | |
| Integer | `add`, `sub`, `mul.{lo,hi,wide}`, `mad.{lo,hi,wide}`, `div`, `rem`, `min`, `max`, `neg`, `abs` | no `.sat`, no carry forms |
| Float | `add`, `sub`, `mul`, `div`, `fma`, `mad`, `min`, `max`, `neg`, `abs`, `sqrt` | round-to-nearest only; `fma` and floating `mad` round once |
| Bitwise | `and`, `or`, `xor`, `not`, `shl`, `shr` | also on `.pred` (not shifts) |
| Compare | `setp` with `eq ne lt le gt ge lo ls hi hs`, unordered `equ neu ltu leu gtu geu`, `num`, `nan`; `selp` | no combined-predicate or paired-destination forms |
| Convert | `cvt` int↔int, int↔float, float↔float, float→int with `.rni/.rzi/.rmi/.rpi` (saturating) | no `.sat` on int→int |
| Memory | `ld.param`, `ld`/`st` with `.global`, `.shared` or no space (generic) | no vector `.v2/.v4`, no `.local` |
| Barriers | `bar.sync a[, count]`, `bar.arrive a, count`; also spelled `barrier` and `.cta` | no `bar.red` |
| Shuffle | `shfl.sync.{up,down,bfly,idx}.b32` with optional predicate destination | |
| mbarrier | `mbarrier.init`, `.arrive` (with count), `.arrive.expect_tx`, `.expect_tx`, `.complete_tx`, `.test_wait`, `.try_wait` (both with `.parity`), `.inval` | no `arrive_drop`, `.noComplete`, `pending_count` |
| TMA | `cp.async.bulk.tensor.{1..5}d` load (`shared <- global`, `.mbarrier::complete_tx::bytes`) and store (`global <- shared`, `.bulk_group`), tile mode; `cp.async.bulk` without a tensor map; `.commit_group`, `.wait_group[.read]`; `prefetch.tensormap` and `cp.async.bulk.prefetch` (no-ops) | no multicast, im2col, `cp.reduce`, `tensormap.replace`; `.L2::cache_hint` ignored |
| Fences | `fence.proxy.async`; `fence.mbarrier_init`, `fence.acq_rel`, `fence.sc`, `fence.proxy.*` accepted with no effect | |
| Control | `bra`, `bra.uni`, `ret`, `exit` | no `brx.idx`, no `call` |

## Special registers

`%tid`, `%ntid`, `%ctaid`, `%nctaid` (each with `.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid`.
