"""Cluster barrier phase tracking and per-CTA asynchronous retirement."""

import numpy as np
import pytest

import warplens

from tests.support import kernel


@pytest.mark.parametrize("timing", [False, True])
def test_cluster_wait_without_arrival_cannot_complete(sim, timing):
    with pytest.raises(warplens.DeadlockError, match="cluster barrier"):
        sim.launch(kernel("barrier.cluster.wait.aligned; ret;"), "k",
                   grid=2, block=32, cluster=2, timing=timing)


@pytest.mark.parametrize("timing", [False, True])
def test_cluster_rejects_two_arrivals_before_phase_completion(sim, timing):
    delay = "mov.u32 %rank, 1;\n" * 20
    mod = kernel(f"""
        .reg .b32 %rank; .reg .pred %first;
        mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %first, %rank, 0;
        @!%first bra LATE;
        barrier.cluster.arrive.aligned;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ret;
      LATE:
        {delay}
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="cluster barrier twice"):
        sim.launch(mod, "k", grid=2, block=32, cluster=2, timing=timing)


@pytest.mark.parametrize("timing", [False, True])
def test_arrival_after_a_completed_phase_does_not_require_an_intermediate_wait(sim, timing):
    # The whole cluster is one warp, so each arrival completes its phase.
    # PTX forbids two arrivals before completion, rather than every consecutive
    # pair of arrivals. The final wait belongs to the most recent arrival.
    mod = kernel("barrier.cluster.arrive.aligned; barrier.cluster.arrive.aligned; "
                 "barrier.cluster.wait.aligned; ret;")
    stats = sim.launch(mod, "k", grid=1, block=32, cluster=1, timing=timing)
    assert not stats.findings


@pytest.mark.parametrize("timing", [False, True])
def test_fast_cta_can_arrive_next_phase_before_slow_cta_waits_previous_phase(sim, timing):
    delay = "mov.u32 %rank, 1;\n" * 20
    mod = kernel(f"""
        .reg .b32 %rank; .reg .pred %first;
        mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %first, %rank, 0;
        barrier.cluster.arrive.aligned;
        @%first bra WAIT;
        {delay}
      WAIT:
        barrier.cluster.wait.aligned;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=2, block=32, cluster=2, timing=timing)
    assert not stats.findings


@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("pattern", ["before_copy", "relaxed", "release"])
def test_cluster_arrival_only_publishes_prior_release_knowledge(sim, timing, pattern):
    early = "barrier.cluster.arrive.aligned;" if pattern == "before_copy" else ""
    late = "" if early else f"barrier.cluster.arrive.{pattern}.aligned;"
    delay = "mov.u32 %v, 0;\n" * 800
    mod = kernel(f"""
        .reg .b64 %src; .reg .b32 %v, %tid; .reg .pred %ready, %one, %producer;
        .shared .align 16 .b8 data[16]; .shared .align 8 .b64 loaded;
        mov.u32 %tid, %tid.x;
        setp.eq.u32 %one, %tid, 0;
        setp.lt.u32 %producer, %tid, 32;
        @%one mbarrier.init.shared::cta.b64 [loaded], 1;
        bar.sync 0;
        @!%producer bra CONSUMER;
        {early}
        ld.param.u64 %src, [src];
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 16;
        @%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [loaded];
        @%one mbarrier.try_wait.parity.shared::cta.b64 %ready, [loaded], 0;
        {late}
        barrier.cluster.wait.aligned;
        ret;
      CONSUMER:
        {delay}
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ld.shared.u32 %v, [data];
        ret;
    """, ".param .u64 src")
    source = sim.from_numpy(np.arange(16, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=64, cluster=1, args=[source],
                       timing=timing, async_delay=0)
    assert [f.kind for f in stats.findings] == (
        [] if pattern == "release" else ["tma-load-unsynchronized"])


@pytest.mark.parametrize("timing", [False, True])
def test_cta_retirement_reports_pending_copies_while_its_peer_keeps_running(sim, timing):
    delay = "mov.u32 %rank, 1;\n" * 1200
    mod = kernel(f"""
        .reg .b64 %src; .reg .b32 %rank; .reg .pred %first;
        .shared .align 16 .b8 data[16]; .shared .align 8 .b64 loaded;
        mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %first, %rank, 0;
        @!%first bra PEER;
        ld.param.u64 %src, [src];
        mbarrier.init.shared::cta.b64 [loaded], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 16;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [loaded];
        ret;
      PEER:
        {delay}
        ret;
    """, ".param .u64 src")
    source = sim.from_numpy(np.arange(16, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=2, block=1, cluster=2, args=[source],
                       timing=timing, async_delay=8)
    assert [f.kind for f in stats.findings] == ["tma-load-not-waited"]


@pytest.mark.parametrize("grid", [2, 4])
def test_concurrent_cluster_fetches_do_not_hit_l2_before_data_arrives(sim, grid):
    mod = kernel("""
        .reg .b64 %src; .reg .pred %ready;
        .shared .align 16 .b8 data[64]; .shared .align 8 .b64 loaded;
        ld.param.u64 %src, [src];
        mbarrier.init.shared::cta.b64 [loaded], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 64;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 64, [loaded];
        mbarrier.try_wait.parity.shared::cta.b64 %ready, [loaded], 0;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 64;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 64, [loaded];
        mbarrier.try_wait.parity.shared::cta.b64 %ready, [loaded], 1;
        ret;
    """, ".param .u64 src")
    source = sim.from_numpy(np.arange(16, dtype=np.uint32))
    stats = sim.launch(mod, "k", grid=grid, block=1, cluster=2, args=[source], timing=True)
    assert not stats.findings
    copies = [s for s in stats.timing.slices if s.track == "TMA engine"]
    for cta in range(grid):
        own = [s for s in copies if s.cta == (cta, 0, 0)]
        # The first two CTAs start together: neither may hit an unfinished fetch.
        # Subsequent clusters run after the first finishes, so they start warm.
        assert [s.args["from_hbm"] for s in own] == ([64, 0] if cta < 2 else [0, 0])
        assert own[1].start >= own[0].end


@pytest.mark.parametrize("timing", [False, True])
def test_multicast_reports_the_receiving_cta_exiting_before_delivery(sim, timing):
    delay = "mov.u32 %rank, 0;\n" * 1200
    mod = kernel(f"""
        .reg .b64 %map; .reg .b32 %rank; .reg .b16 %mask; .reg .pred %first;
        .shared .align 16 .b8 data[64]; .shared .align 8 .b64 loaded;
        ld.param.u64 %map, [map];
        mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %first, %rank, 0;
        mov.b16 %mask, 2;
        mbarrier.init.shared::cta.b64 [loaded], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        @!%first mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 64;
        @%first cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
            [data], [%map, {{0}}], [loaded], %mask;
        @!%first ret;
        {delay}
        ret;
    """, ".param .u64 map")
    source = sim.from_numpy(np.arange(16, dtype=np.uint32))
    stats = sim.launch(mod, "k", grid=2, block=1, cluster=2,
                       args=[sim.tensor_map_tiled(source, 16)], timing=timing, async_delay=8)
    finding, = stats.findings
    assert finding.kind == "tma-load-not-waited"
    assert "CTA (1, 0, 0) returned" in finding.message
