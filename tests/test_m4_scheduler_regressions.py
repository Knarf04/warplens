"""Regressions for timed issue ordering and asynchronous-operation retirement."""

import numpy as np
import pytest
import warplens

from conftest import kernel
from warplens.timing.config import GpuConfig


@pytest.mark.parametrize("timing", [0, 1, "B200", {}, []])
def test_invalid_timing_argument_is_rejected(sim, timing):
    with pytest.raises(warplens.LaunchError, match="timing must be"):
        sim.launch(kernel("ret;"), "k", grid=1, block=1, timing=timing)


def bulk_load_kernel(*, wait):
    return kernel("""
        .reg .b64 %src;
        .reg .b32 %buf, %bar;
        .reg .pred %ready;
        .shared .align 16 .b8 buf[64];
        .shared .align 8 .b64 bar;
        ld.param.u64 %src, [src];
        mov.b32 %buf, buf;
        mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 64;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
            [%buf], [%src], 64, [%bar];
    """ + ("mbarrier.try_wait.parity.shared::cta.b64 %ready, [%bar], 0;\n" if wait else "")
        + "ret;", ".param .u64 src")


def test_timed_exit_reports_copy_still_in_flight(sim):
    src = sim.from_numpy(np.arange(16, dtype=np.uint32))
    mod = bulk_load_kernel(wait=False)
    plain = sim.launch(mod, "k", grid=1, block=1, args=[src], async_delay=100)
    timed = sim.launch(mod, "k", grid=1, block=1, args=[src], timing=True)
    assert "tma-load-not-waited" in {finding.kind for finding in plain.findings}
    assert {finding.kind for finding in timed.findings} == {finding.kind for finding in plain.findings}
    copy, = [s for s in timed.timing.slices if s.track == "TMA engine"]
    phase, = timed.timing.instants
    assert phase[1] == pytest.approx(copy.end)


def test_warps_issue_only_when_their_shared_slot_is_free(sim):
    mod = kernel(".reg .b32 %r; mov.u32 %r, %tid.x; add.u32 %r, %r, 1; ret;")
    issued = []
    config = GpuConfig.load("B200").replace(clock_ghz=1)
    sim.launch(mod, "k", grid=1, block=256, timing=config,
               trace=lambda w, pc, ins, mask: issued.append((w.time, w.warp_id % 4)))
    assert [t for t, _ in issued] == sorted(t for t, _ in issued)
    for slot in range(4):
        times = [t for t, s in issued if s == slot]
        assert all(b - a >= 1 for a, b in zip(times, times[1:]))


def test_repeated_bulk_loads_use_the_same_l2_bookkeeping_as_tensor_loads(sim):
    src = sim.from_numpy(np.arange(16, dtype=np.uint32))
    stats = sim.launch(bulk_load_kernel(wait=True), "k", grid=2, block=1,
                       args=[src], timing=True)
    assert not stats.findings
    copies = [s for s in stats.timing.slices if s.track == "TMA engine"]
    assert [s.args["from_hbm"] for s in copies] == [64, 0]


@pytest.mark.parametrize("coordinate, fetched_bytes", [(0, 64), (8, 32), (16, 0)])
def test_tensor_oob_fill_does_not_consume_global_bandwidth(sim, coordinate, fetched_bytes):
    source = sim.from_numpy(np.arange(16, dtype=np.uint32))
    tensor_map = sim.tensor_map_tiled(source, 16)
    mod = kernel(f"""
        .reg .b64 %map;
        .reg .b32 %buf, %bar;
        .reg .pred %ready;
        .shared .align 16 .b8 buf[64];
        .shared .align 8 .b64 bar;
        ld.param.u64 %map, [map];
        mov.b32 %buf, buf;
        mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 64;
        cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes
            [%buf], [%map, {{{coordinate}}}], [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %ready, [%bar], 0;
        ret;
    """, ".param .u64 map")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[tensor_map], timing=True)
    assert not stats.findings
    copy, = [s for s in stats.timing.slices if s.track == "TMA engine"]
    assert copy.args["bytes"] == 64  # padding is still written to shared memory
    assert copy.args["from_hbm"] == fetched_bytes
    if fetched_bytes == 0:
        assert stats.timing.utilization["HBM"] == 0
        assert stats.timing.utilization["L2"] == 0
