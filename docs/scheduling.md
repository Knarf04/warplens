# Scheduling: the order in which warps run

In an untimed launch the engine (`core/engine.py`) works in rounds. By default
every warp that can run executes one instruction per round, in warp order
within a CTA and in rank order across the CTAs of a cluster: round-robin. An
asynchronous operation (a TMA copy, an MMA, a `tcgen05.commit`) completes
`async_delay` rounds after it is issued, or at once when no warp can run.

That is one order among the many a GPU may produce. Nothing orders two warps
that no barrier synchronizes, so a correct kernel computes the same whichever
of them is stepped first. Round-robin is a kind order: warps that execute the
same code stay in step, so a kernel that writes in one phase and reads in the
next finds everything written even when the `__syncthreads()` between the two
is missing. The race check reports such a kernel whatever the order
([architecture.md](architecture.md#3-shared-memory-barriers-and-the-race-check)); another schedule shows what the missing barrier does to the
result.

## Choosing a schedule

| Spelling | Schedule |
| --- | --- |
| `None`, `"round-robin"` | the default, unchanged |
| `7`, `"7"`, `"seed:7"` | seeded: each round, the warps that can run are stepped in a pseudo-random order drawn from the seed. One instruction per warp per round, as in round-robin |
| `"adversarial"` | the warps are taken last to first, and a warp keeps running until it blocks, exits, polls an mbarrier with `test_wait`, or has executed 64 instructions (`ADVERSARIAL_QUANTUM`). The other warps stand still meanwhile |
| `"adversarial:7"` | adversarial, with the order of the warps and the length of each turn (1 to 64 instructions) drawn from the seed |

The same spelling works in three places:

```python
stats = sim.launch(ptx, "kernel", grid=1, block=256, args=[...], schedule="adversarial")
```

```
$ warplens run --schedule adversarial:3 ./program
$ WARPLENS_SCHEDULE=3 python script.py
```

`Device.launch(schedule=...)` is for one launch. The environment variable
`WARPLENS_SCHEDULE` is the schedule of every untimed launch that names none,
in any process: under the shim, in a Triton driver, in the tests
(`WARPLENS_SCHEDULE=2 pytest tests/test_hardware_golden.py`). `warplens run
--schedule` sets that variable for the run and for the program it starts.

The same schedule and the same inputs give the same run: the seed feeds one
generator per launch (`random.Random(seed).random()`, whose sequence is fixed
across Python versions), and nothing else in the engine is random.

A timed launch is ordered by the timing model. It refuses a `schedule`
argument, ignores `WARPLENS_SCHEDULE`, and `warplens run` refuses `--schedule`
together with `--timing`.

## What the adversarial schedule does to asynchronous work

Rounds still count `async_delay`, but an adversarial round is up to 64
instructions per warp instead of one. A copy that round-robin delivers four
instructions after its issue arrives up to 64 times later in instructions, so
a warp that uses the data without waiting for it finds the bytes that were
there before. This is the plan's "delay async completions to expose races".
When every warp is blocked the oldest operation in flight completes at once,
as under any schedule, so a kernel that waits properly is not slowed down.
A warp that polls with `mbarrier.test_wait` ends its turn at the poll, which
keeps spin loops from burning whole turns.

`%clock64` counts rounds; within an adversarial turn it also advances by one
per instruction, so a warp that spins on the clock makes progress.

## What to expect

- A correct kernel: the same output and no findings under every schedule.
  `tests/test_scheduling.py` checks this for the examples it runs, and the
  results recorded from real hardware (`tests/test_hardware_golden.py`) hold
  under seeded and adversarial schedules.
- A kernel with an unsynchronized read-modify-write: a different result for
  different seeds, and `smem-race` under all of them.
- A kernel with a missing barrier between a write phase and a read phase: the
  right result under round-robin, stale or unwritten data under
  `"adversarial"`, `smem-race` under both.

What does depend on the schedule, besides results: which of two racing
instructions a `smem-race` finding names (the one that ran second), instruction
counts of kernels that poll, and findings that depend on values, such as
`smem-uninitialized-read` for a byte read before its unsynchronized writer got
to it. Errors that depend on the order do too: a warp that runs ahead may
reach a second `bar.arrive` on a barrier whose waiters have not run yet, which
the simulator refuses ("arrived ... more than once").

## Limits

- The schedule permutes warps within a cluster. Clusters still start one at a
  time, in grid order ([clc.md](clc.md)), so CTAs of different clusters never
  interleave and races between them through global memory are not explored.
- Lanes of a warp execute an instruction together under every schedule.
- A schedule is one sample. Passing under some seeds shows nothing about the
  orders that were not tried; the race check, which does not depend on the
  order, is the stronger tool for shared memory.
- The adversarial quantum is a constant. A warp that needs more than 64
  instructions between its write and a neighbour's read is not run far enough
  ahead in one turn to expose a missing barrier between them.
