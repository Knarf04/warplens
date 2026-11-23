# PTX coverage

What warplens executes today. Anything not listed raises `UnsupportedError`
that names the instruction and its PTX line.

## Module and function structure

| Construct | Status |
| --- | --- |
| `.version`, `.target`, `.address_size` | parsed |
| `.entry` kernels with scalar `.param`s (`.ptr`, `.align` accepted) | yes |
| `.reg` declarations, including `%r<N>` ranges | yes |
| labels, `@%p` / `@!%p` predication on any instruction | yes |
| performance directives before the body (`.maxntid`, ...), `.pragma` | ignored |
| `.func` and `call`, `.global` / `.const` / `.shared` / `.local` variables | no |
| nested `{ }` blocks (inline asm scopes), array parameters | no |

## Instructions

| Group | Instructions | Notes |
| --- | --- | --- |
| Moves | `mov`, `cvta[.to].global` | no symbol addresses |
| Integer | `add`, `sub`, `mul.{lo,hi,wide}`, `mad.{lo,hi,wide}`, `div`, `rem`, `min`, `max`, `neg`, `abs` | no `.sat`, no carry forms |
| Float | `add`, `sub`, `mul`, `div`, `fma`, `min`, `max`, `neg`, `abs`, `sqrt` | round-to-nearest only |
| Bitwise | `and`, `or`, `xor`, `not`, `shl`, `shr` | also on `.pred` (not shifts) |
| Compare | `setp` with `eq ne lt le gt ge lo ls hi hs`, unordered `equ neu ltu leu gtu geu`, `num`, `nan`; `selp` | no combined-predicate or paired-destination forms |
| Convert | `cvt` int↔int, int↔float, float↔float, float→int with `.rni/.rzi/.rmi/.rpi` (saturating) | no `.sat` on int→int |
| Memory | `ld.param`, `ld.global`, `st.global` (no space = global) | no vector `.v2/.v4`, no `.shared` / `.local` |
| Control | `bra`, `bra.uni`, `ret`, `exit` | no `brx.idx`, no `call` |

## Special registers

`%tid`, `%ntid`, `%ctaid`, `%nctaid` (each with `.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid`.
