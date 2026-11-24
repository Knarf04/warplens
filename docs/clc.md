# Cluster launch control

A kernel with cluster launch control (CLC) launches one cluster per piece of work, as
usual, but a cluster that finishes does not exit. It asks the launch to cancel a cluster
that has not started, receives that cluster's coordinate, and does its work too. The
book's chapter *Advanced Scheduling: Cluster Launch Control* explains why.

```
$ python examples/09_clc/run.py
cancellations granted: every request while clusters are left
  2 of 12 CTAs started, 10 were canceled; output correct, 0 finding(s)
  tile -> CTA that computed it: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
cancellations granted: 2 per cluster
  4 of 12 CTAs started, 8 were canceled; output correct, 0 finding(s)
  tile -> CTA that computed it: [0, 1, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7]
cancellations granted: 0 per cluster
  12 of 12 CTAs started, 0 were canceled; output correct, 0 finding(s)
  tile -> CTA that computed it: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

**Nothing here has run on Blackwell hardware.** The instruction forms are those the
toolchain accepts; what they do is taken from the PTX manual. What the manual leaves open
is listed under [Assumptions](#assumptions-to-check-on-a-b200).

## Instructions

```
clusterlaunchcontrol.try_cancel.async{.shared::cta}.mbarrier::complete_tx::bytes
                                     {.multicast::cluster::all}.b128  [addr], [mbar];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128               p, response;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128         {x, y, z, _}, response;
clusterlaunchcontrol.query_cancel.get_first_ctaid::{x,y,z}.b32.b128   d, response;
```

A response is 16 bytes and is queried from a `.b128` register, so these came with it:
`.reg .b128`, `ld` and `st` of `.b128`, `mov.b128` between registers, from an integer
literal, and to or from two 64-bit halves (`mov.b128 %rq1, {%rd3, %rd4}`, low half first,
which is how nvcc builds one). No arithmetic accepts `.b128`. `ld` also accepts `.weak`,
`.relaxed.<scope>` and `.acquire.<scope>`, because TIRx loads the response with
`ld.acquire.cta.shared.b128`; they do not change what a load does here, since loads
happen one at a time in program order.

## Where each rule comes from

Sources: the PTX ISA 9.5 manual, sections 9.7.15.18 (`try_cancel`) and 9.7.15.19
(`query_cancel`); CUTLASS `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp`; the CLC
helpers of TIRx (`tvm_tirx/backend/cuda/lang/clc.py`, `tile_scheduler.py`) and of the CuTe
DSL (`cutlass/cute/arch/clc.py`); the book's chapter; and nvcc and ptxas 13.5 for which
spellings exist.

| Rule | Source |
| --- | --- |
| The spellings above, and that the bare `get_first_ctaid` exists only as `.v4` | manual syntax; ptxas assembles the kernels of `tests/test_clc.py` and `examples/09_clc`, and rejects `get_first_ctaid.b32.b128 d, response` |
| A request cancels a cluster "that has not started running yet", never one that runs | manual |
| No two successful responses of a grid name the same cluster | manual |
| The response is written asynchronously, and completion is a `complete_tx` of the bytes stored on `mbar` | manual. 16 bytes: the manual's example expects 16, as do CUTLASS and TIRx |
| The response is written through the async proxy | manual ("weak async-proxy writes", said of the multicast form); the book says it of every form |
| `.multicast::cluster::all` writes the response at the same address in every CTA of the requesting cluster and completes the mbarrier in each CTA's own shared memory | manual |
| Without `.shared::cta` both addresses are generic and must point into this CTA's shared memory | manual ("the result is undefined if any of address operands do not fall within the address window of `.shared::cta`"); here it is an error |
| `addr` is 16-byte aligned | manual ("naturally aligned"); here a misaligned address is an error |
| Each thread that executes the instruction makes its own request | book |
| On success the response holds the id of the first CTA of the canceled cluster; `get_first_ctaid` returns it, the fourth element of `.v4` is unspecified | manual |
| `get_first_ctaid` of a failed response is undefined | manual; here it reads as zero and is reported |
| A `try_cancel` after the CTA has observed a failed one is undefined | manual; here it fails again and is reported |
| Multicast is undefined once a CTA of the cluster has exited | manual; here the exited CTA's memory is still written, and it is reported |

## The launch model

A launch already ran its clusters one at a time, in order (x fastest, then y, then z), each
to completion. CLC keeps that. The clusters that have not started are a queue
(`warplens/core/launch_queue.py`); a `try_cancel` takes the next one out of the queue, and
that cluster is never created. So "not launched yet" means: later in launch order than the
cluster now running, and not canceled already.

- The launch answers a request when it is issued, so requests are served in program
  order. The response is readable later: it lands after `async_delay` scheduling rounds,
  like a TMA load, and then completes the mbarrier. In a timed launch it lands
  `clc_latency_ns` after it was issued. How long a GPU takes to answer is not known: the
  100 ns in the config files is an uncalibrated placeholder, set equal to `tma_latency_ns`.
- With every request granted, the first cluster takes over the whole grid: the outcome on
  a GPU that has room for one cluster. A GPU with room for all of them starts every
  cluster before any request is served, and every request fails. `launch(...,
  clc_limit=n)` grants each cluster that starts at most `n` cancellations: `0` is the
  second extreme, `None` (the default) the first. A correct kernel computes the same
  result for any value, so running a few is a cheap test of the take-over path and of the
  normal start.
- A canceled cluster's CTAs do not exist: nothing of theirs is executed, counted or
  traced. `stats.ctas` stays the size of the grid; `stats.canceled_ctas` says how many of
  them never started.
- The response bytes are warplens's own encoding (see `warplens/units/clc.py`). On a GPU
  the response is opaque; a kernel that reads it other than with `query_cancel` is not
  portable, and here it would see the encoding.

## Findings

| Kind | Raised when |
| --- | --- |
| `clc-canceled-cluster-lost` | a cluster finishes, having canceled another whose coordinate no thread read with `get_first_ctaid`: that cluster's work was done by nobody |
| `clc-not-waited` | a CTA returns while its request is in flight |
| `clc-response-unsynchronized` | the response is read, or overwritten by another request, by a warp that has not waited on the request's mbarrier |
| `clc-query-not-a-response` | `query_cancel` is given a value that no `try_cancel` wrote; it is treated as a failed request |
| `clc-ctaid-of-failed-cancel` | `get_first_ctaid` of a failed response |
| `clc-try-cancel-after-failure` | `try_cancel` after a thread of the CTA saw `is_canceled` return false |
| `clc-multicast-to-exited-cta` | a multicast request after a CTA of the cluster returned |

A request that is not announced with `expect_tx`, or announced with the wrong byte count,
shows up as the usual `mbarrier-tx-mismatch` finding or as a deadlock on the mbarrier.

The CuTe DSL's `cute.arch.clc_response` reads all three coordinates without a predicate,
so a CuTe DSL kernel gets one `clc-ctaid-of-failed-cancel` when its last request fails.
That is what the manual calls undefined, so it is reported.

## Checked against

- `tests/test_clc.py`: every form, the cases above, and each error.
- `examples/09_clc`: CUDA C++ through nvcc, two-CTA clusters with multicast.
- A TIRx kernel that uses TIRx's own `query_cancel_first_ctaid_x`, and a CuTe DSL kernel
  that uses `cute.arch.issue_clc_query` and `clc_response`, both under `warplens run`
  (not in the test suite). The book's chapter has no kernel to run: it describes the
  mechanism in prose and pseudocode.

## Limits

- Clusters never run concurrently, so a kernel cannot be shown racing a cluster that is
  about to start, and the mix of started and canceled clusters comes from `clc_limit`,
  not from timing.
- `clc_limit` is the only control, and only through the Python API. `warplens run` always
  grants every request.
- The manual's example ends an iteration with
  `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster` and its
  `.acquire` counterpart. Those two fences are not implemented (they are rejected, not
  ignored), so the example does not run as printed. CUTLASS, TIRx and the CuTe DSL use
  `fence.proxy.async.shared::cta` there, which is.
- Whether a kernel executed that fence between reading a response and the next request
  into the same bytes is not checked.
- `.multicast::cluster::all` is accepted for any `.target`; on a GPU it needs `sm_100a` or
  one of the other architectures the manual lists.

## Assumptions to check on a B200

1. **Which cluster a request cancels.** Here: the next in launch order. The manual says
   only that it has not started.
2. **When requests fail.** Here: only when no cluster is left, or by `clc_limit`. The book
   says a request can also fail while clusters are pending. How many requests of a real
   launch succeed is unknown.
3. **Every executing thread makes a separate request** (book; the manual does not say).
4. **A failed response decodes as coordinate 0.** Undefined in the manual; the CuTe DSL
   relies on it being harmless.
5. **A request after an observed failure fails again**, and "observed" means a thread
   executed `is_canceled` on the failed response. Undefined in the manual.
6. **The non-multicast form writes only the requesting CTA's response** and completes only
   its mbarrier. Implied by the manual, not stated.
7. **`complete_tx` is 16 bytes per receiving CTA.**
8. **A canceled cluster's coordinate is that of its first CTA in grid coordinates**
   (`clusterid * cluster shape`), also when the launch gave no cluster shape and each CTA
   is its own cluster.
9. **Latency of a request**, for the timing model: `clc_latency_ns`, 100 ns, a placeholder
   with nothing behind it.
