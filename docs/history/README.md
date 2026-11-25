# Development history

Documents that record how warplens was built. They are kept as they were and
are not updated; where one disagrees with the [user guide](../guide.md) or
with [architecture.md](../architecture.md), it is the older one. In
particular, most of them were first written before anything had run on
Blackwell hardware, and parts of them still say so in the present tense.

| File | What it records |
| --- | --- |
| [plan.md](plan.md) | the project plan as it stood on 2026-11-25: goals, survey of related work, the architecture as first sketched, the frozen v1 instruction scope and its amendment (section 3.5), the milestones, and the path to the first B200 run (section 11). What is left of it is [PLAN.md](../../PLAN.md) |
| [m0.md](m0.md) | M0: the PTX parser, the warp executor, global memory, the Python launch API |
| [m1.md](m1.md) | M1: shared memory, the bank model and its comparison with `ncu` on Ampere, barriers, the race check and what it costs |
| [m2.md](m2.md) | M2: TMA, tensor maps, mbarriers, fences; the assumptions that were to be confirmed on hardware |
| [m3.md](m3.md) | M3: `tcgen05.mma`, tensor memory, the first two CUTLASS tutorials |
| [m4.md](m4.md) | M4: the timing model, the configs, the trace, `warplens report` |
| [m5.md](m5.md) | M5: clusters, multicast, the two-CTA MMA |
| [m6.md](m6.md) | M6: `warplens run`, the shim, Triton, TIRx, the CuTe DSL |

The validation matrix, the probe sources under `validation/blackwell/` and a
few comments elsewhere name these write-ups, with a section, as the place
where a claim was first stated. That is why they keep their file names.

Not here, because it is evidence and not history:
[b200-rental-1.md](../b200-rental-1.md), the record of the first run on a
B200.
