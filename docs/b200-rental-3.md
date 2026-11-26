# The third B200 rental (2026-11-26): the open questions

A short run on the machine of the first two rentals, for the questions that
`PLAN.md` listed as open and that only a B200 could answer. The record of
what was asked, what the GPU said, and what changed.

**Measured** is in the archives and files named below; **reading** is an
interpretation that no further experiment has checked.

## The machine and the runs

The same instance, started again: one NVIDIA B200 on a VM, driver 610.57.04
(CUDA 13.3). The two installs of the earlier rentals were used as they were.

| Run | Compiler | Archive | SHA-256 |
| --- | --- | --- | --- |
| By hand, first | nvcc 13.3.73, the new probe cases and four small experiments, from a copy of the working tree | `validation/blackwell/measured/b200/rental-3-by-hand.raw.txt` | in the repository |
| Main | nvcc 13.3.73 (the fallback), commit `ad47b978f6` | `warplens-b200-20261125T012055Z-ad47b978f6.tar.gz` (66 MB) | `cf03e8c7c98b…` |
| Pinned | nvcc 13.5.37, machine code, `--no-e2e`, the same commit | `warplens-b200-20261125T022247Z-ad47b978f6.tar.gz` | `2d776c3c6026…` |

The by-hand runs came first because the machine was up before the probes
were committed: each new case was built there from a copy of the tree and
run three times. What they showed changed the simulator and two of the
probes while the full runs were under way, so the two archives hold the
probes as they were at `ad47b978f6`, and the probes as committed were run by
hand afterwards (the last section of the raw file).

## What was asked, and the answers

### A second `tcgen05.dealloc` of the same columns (TM-11)

**Open since the first rental:** one warp that released an allocation twice
ended the kernel, and the book's `hgemm_v7` to `v9`, in which two warps each
relinquish the allocation permit and release the same allocation, ran.

**Measured** (`tmem/dealloc_who`, five cases, three runs each, all alike):

| Case | B200 |
| --- | --- |
| two warps release once each, nothing between them | runs to the end |
| two warps each relinquish the permit and release (the book) | runs to the end |
| two warps, the second after a barrier that follows the first | `cudaErrorLaunchFailure` |
| one warp relinquishes the permit, then releases twice | `cudaErrorLaunchFailure` |
| one warp releases once | runs to the end |

What separates them is order, and nothing else: a release of columns that an
earlier release has already freed ends the kernel; two releases that nothing
orders both go through. Relinquishing the permit changes nothing.

**Changed.** `warplens/units/tcgen05.py`: a repeated release that is ordered
after the first (the same warp, or another warp after a barrier; the
happens-before clocks of `core/clocks.py` decide) is an error with the GPU's
code. One that nothing orders is still the finding `tmem-dealloc-repeated`,
which now says that a GPU runs it. The simulator gives the GPU's result in
all five cases and in the first rental's probe, and the book's kernels run as
before.

**Reading.** The two releases of the book's kernels race, and the kernels
work because of how that race falls on this GPU. The finding stays.

### `elect.sync` with a mask that names lanes which are elsewhere (RT-05)

**Measured** (`e2e/elect other-branch`, `other-callers`; three runs each):
one half of the warp elects with the mask `0xffffffff` while the other half
is on a branch of its own, where each lane sets a flag. The electing lanes
read their partner's flag as 0 before the instruction and 0 after it, in
both halves, exactly as in the control whose mask names only the callers.
`elect.sync` did not wait.

The first form of this probe could not say so: on the B200 the other branch
ran first whatever the mask (the flag read 1 in the control too). The probe
was rewritten so that the electing half runs first and reads the flag on
both sides of the instruction.

**Not changed.** The simulator did not make `elect.sync` wait, and does not.

### `bar.warp.sync`, `shfl.sync` and `vote.sync`, on the B200

The day before, on the RTX 3080 Ti, a warp barrier was found to wait for the
lanes of its mask, and the simulator was made to wait for a shuffle's too,
which that GPU's machine code suggested (`validation/README.md`). With the
B200 at hand the same experiment was run there (`sync_order`, in the raw
file; nvcc 13.3 through the driver, and nvcc 13.5 as machine code, alike):

| Instruction, mask `0xffffffff`, the other half on a branch | flag before | flag after | |
| --- | --- | --- | --- |
| `bar.warp.sync` | 0 | 1 | waited (0 and 0 in the control) |
| `shfl.sync` | 0 | 0 | did not wait; read 0 from the absent lane |
| `vote.sync.ballot` | 0 | 0 | did not wait; the ballot has the callers' bits |
| `elect.sync` | 0 | 0 | did not wait |

The 16 launches of `validation/ampere/sync_other_probe.cu` gave the same
lines on the B200 as on the 3080 Ti but one, which no rule decides. The 100
launches of `run_vote_other.sh` were rerun too: the same 24 never return
(two groups of `vote.sync.all/any/uni` waiting for each other), and 18 that
returned on the 3080 Ti end with an error on the B200.

**Changed.** `shfl.sync` is taken out of the waiting again
(`warplens/core/warp.py`): `bar.warp.sync` and `vote.sync` wait, `shfl.sync`
and `elect.sync` do not.

**Open.** The one ballot that did not wait. `vote.sync` as a whole is left
waiting, which is what the 24 launches that never return say on both GPUs;
the two observations have not been reconciled.

### Launch errors at the B200's own limits (RT-04, RT-12)

**Measured** (`e2e/launch_errors`, the program of
`validation/ampere/launch_errors_hw.cu`, which asks the device for its
limits): every shared-memory limit, opt-in and attribute case as the
simulator has it. Two codes differ from the RTX 3080 Ti's recording, which
was made with the runtime of CUDA 12.4:

| Request | CUDA 12.4, RTX 3080 Ti | CUDA 13.3, B200 |
| --- | --- | --- |
| a block or grid the launch refuses (six cases) | `cudaErrorInvalidConfiguration` | `cudaErrorInvalidValue` |
| an attribute of a function that is not a kernel | `cudaErrorInvalidDeviceFunction` | `cudaErrorInvalidResourceHandle` |

**Changed.** The runtime shim returns the B200's codes
(`warplens/shim/cudart_shim.c`). **Reading:** the difference is the
runtime's release and not the GPU; nothing here separates the two.

### Half-precision subnormals in an MMA (MMA-04)

**Measured** (`mma/subnormals`): identical to the prediction, every word.
Subnormal operands are multiplied like any other number (2^-24 times 2^-24
is 2^-48), and a subnormal next to a normal term is lined up and cut as the
model has it.

### What an undersized `expect_tx` does, once more

`mbarrier/expect_tx_small` and `.wait` (128 bytes expected of a 256-byte
load) had completed the phase early in both earlier rentals. In the main run
of this one the phase did not complete and the wait loop hung, and in ten
more runs by hand it did not complete either. That is the run-to-run
behaviour the second rental counted, now seen on the probes that had always
agreed. The script is its own probe run without a prediction since
(`mbarrier/expect_tx_small.too_small`), and `.wait` has none either.

## What agreed in the full runs

Both runs repeated the whole suite, 141 probe runs now, and the main run the
end-to-end programs.

| | Main (nvcc 13.3), against the predictions it ran with | Pinned (nvcc 13.5), against the predictions as committed now |
| --- | --- | --- |
| identical | 107 | 112 |
| end-to-end, bit-identical | 8 | not run |
| informational | 17 | 21 |
| the kernel failed where the probe allows a fault | 3 | 0 |
| the GPU faulted where the simulator has no model | 5 | 3 |
| timed out, as predicted | 2 | 2 |
| timed out, predicted to finish | 1 | 0 |
| differs | 5 | 2 |

- The main run's predictions were the simulator's of `ad47b978f6`, before
  the answers above. Its three failed kernels are the repeated releases; its
  five differences are the launch-error codes, the three `elect` cases in
  their first form, and the undersized `expect_tx`, whose wait loop is the
  one that timed out.
- The pinned run's two differences are of the probes, not of the GPU: the
  archive has `e2e/launch_errors` printing the device's name and
  `mbarrier/expect_tx_small` still with the undersized script in it. Both
  programs as committed were run by hand at the end and are identical to
  their predictions (`rental-3-by-hand.raw.txt`, under `final/`).
- Everything the second rental changed is identical in both: a tensor copy at
  an address that is not a multiple of 128 bytes ends the kernel, which was a
  prediction this time. The eight whole programs are bit-identical again.
- The timing suite ran in both and its numbers were not read. `ncu` had the
  counters of all 11 benchmarks in both runs: in the pinned run `transpose`
  was profiled for the first time, built as machine code by the `run_ncu.sh`
  that the second rental's write-up describes.

## Still open after this run

- The ballot that did not wait, above.
- `kind::i8` in an MMA was not asked: the simulator refuses that kind, so
  there is nothing of version 1 to hold against a GPU. It is with the other
  MMA kinds in `PLAN.md`, after v1.
- UE4M3 (NUM-05): no toolkit has a conversion for it.
- The changes of this run (the order rule for a repeated release, the two
  runtime codes, the warp barrier's wait) reproduce what was recorded on the
  day they were made. Only the probes run by hand at the end ran against
  them as predictions.
