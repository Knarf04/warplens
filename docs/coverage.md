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
| `.func` and `call`, `.global` / `.const` / `.local` variables | no |
| nested `{ }` blocks (inline asm scopes), array parameters | no |

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
| Barriers | `bar.sync a[, count]`, `bar.arrive a, count`; also spelled `barrier` and `.cta`, including `barrier[.cta].{sync,arrive}.aligned` | no `bar.red`; sync count zero means the whole CTA |
| Shuffle | `shfl.sync.{up,down,bfly,idx}.b32` with optional predicate destination | no rendezvous across divergent paths; see `m1.md` |
| Control | `bra`, `bra.uni`, `ret`, `exit` | no `brx.idx`, no `call` |

## Special registers

`%tid`, `%ntid`, `%ctaid`, `%nctaid` (each with `.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid`.
