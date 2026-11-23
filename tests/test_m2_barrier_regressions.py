"""Mbarrier address, transaction-credit and synchronization regressions."""

import numpy as np
import pytest

import warplens

from conftest import kernel


def _barrier_kernel(body):
    return kernel("""
        .reg .b32 %v; .reg .b64 %bar, %out, %token; .reg .pred %p;
        .shared .align 8 .b64 bar;
        mov.u64 %bar, bar;
        ld.param.u64 %out, [out];
    """ + body + "\nret;", ".param .u64 out")


def _record(index):
    return f"selp.u32 %v, 1, 0, %p; st.global.u32 [%out+{index * 4}], %v;"


def test_generic_mbarrier_addresses_support_the_whole_lifecycle(sim):
    mod = _barrier_kernel(f"""
        cvta.shared.u64 %bar, %bar;
        mbarrier.init.b64 [%bar], 1;
        mbarrier.expect_tx.b64 [%bar], 8;
        mbarrier.arrive.expect_tx.b64 %token, [%bar], 8;
        mbarrier.complete_tx.b64 [%bar], 16;
        mbarrier.test_wait.b64 %p, [%bar], %token;
        {_record(0)}
        mbarrier.inval.b64 [%bar];
        mbarrier.init.b64 [%bar], 1;
        mbarrier.arrive.b64 %token, [%bar];
        mbarrier.try_wait.parity.b64 %p, [%bar], 0;
        {_record(1)}
    """)
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [1, 1]
    assert not stats.findings


def test_generic_mbarrier_rejects_a_nonshared_address(sim):
    mod = _barrier_kernel("mbarrier.init.b64 [%out], 1;")
    with pytest.raises(warplens.ExecutionError, match="does not point to shared memory"):
        sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(1, "u32")])


@pytest.mark.parametrize("expect_before_arrive", [False, True])
def test_completion_credits_can_precede_the_expected_bytes(sim, expect_before_arrive):
    expect = "mbarrier.expect_tx.shared::cta.b64 [%bar], 16;"
    arrive = "mbarrier.arrive.shared::cta.b64 %token, [%bar];"
    body = "\n".join((expect, arrive) if expect_before_arrive else (arrive, expect))
    mod = _barrier_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.complete_tx.shared::cta.b64 [%bar], 16;
        {body}
        mbarrier.test_wait.shared::cta.b64 %p, [%bar], %token;
        {_record(0)}
    """)
    out = sim.empty(1, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [1]
    assert not stats.findings


def test_negative_transaction_count_does_not_complete_a_phase(sim):
    mod = _barrier_kernel(f"""
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.complete_tx.shared::cta.b64 [%bar], 16;
        mbarrier.arrive.shared::cta.b64 %token, [%bar];
        mbarrier.test_wait.shared::cta.b64 %p, [%bar], %token;
        {_record(0)}
        mbarrier.expect_tx.shared::cta.b64 [%bar], 8;
        mbarrier.test_wait.shared::cta.b64 %p, [%bar], %token;
        {_record(1)}
        mbarrier.expect_tx.shared::cta.b64 [%bar], 8;
        mbarrier.test_wait.shared::cta.b64 %p, [%bar], %token;
        {_record(2)}
    """)
    out = sim.empty(3, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [0, 0, 1]
    assert not stats.findings


def test_unbalanced_completion_credits_are_reported_at_kernel_exit(sim):
    mod = _barrier_kernel("""
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.complete_tx.shared::cta.b64 [%bar], 16;
        mbarrier.expect_tx.shared::cta.b64 [%bar], 8;
    """)
    stats = sim.launch(mod, "k", grid=1, block=1, args=[sim.empty(1, "u32")])
    (finding,) = stats.findings
    assert finding.kind == "mbarrier-tx-mismatch"
    assert "8 completion byte(s)" in finding.message
    assert finding.text.startswith("mbarrier.complete_tx")


@pytest.mark.parametrize("delay", [0, 4, 1000])
@pytest.mark.parametrize("wait_semantics", ["acquire", "relaxed"])
def test_only_an_acquire_wait_makes_tma_data_visible(sim, delay, wait_semantics):
    mod = kernel(f"""
        .reg .b64 %src; .reg .b32 %v; .reg .pred %p;
        .shared .align 16 .b8 data[16]; .shared .align 8 .b64 bar;
        ld.param.u64 %src, [src];
        mbarrier.init.shared::cta.b64 [bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [bar];
        mbarrier.try_wait.parity.{wait_semantics}.cta.shared::cta.b64 %p, [bar], 0;
        ld.shared.u32 %v, [data];
        ret;
    """, ".param .u64 src")
    src = sim.from_numpy(np.arange(16, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src], async_delay=delay)
    assert [f.kind for f in stats.findings] == (
        ["tma-load-unsynchronized"] if wait_semantics == "relaxed" else [])


def test_wait_on_previous_phase_does_not_acquire_current_phase_copies(sim):
    mod = kernel("""
        .reg .b64 %src; .reg .b32 %v; .reg .pred %p;
        .shared .align 16 .b8 data[32]; .shared .align 8 .b64 bar;
        ld.param.u64 %src, [src];
        mbarrier.init.shared::cta.b64 [bar], 1;
        mbarrier.arrive.shared::cta.b64 _, [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 32;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [bar];
        // Phase 0 is complete, but phase 1 is waiting for the second copy.
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        ld.shared.u32 %v, [data];
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data+16], [%src+16], 16, [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 1;
        ld.shared.u32 %v, [data];
        ret;
    """, ".param .u64 src")
    src = sim.from_numpy(np.arange(32, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src], async_delay=0)
    (finding,) = stats.findings
    assert finding.kind == "tma-load-unsynchronized"
    assert finding.count == 1


@pytest.mark.parametrize("arrive_semantics", ["release", "relaxed"])
def test_only_a_release_arrival_publishes_completed_copy_knowledge(sim, arrive_semantics):
    mod = kernel(f"""
        .reg .b64 %src; .reg .b32 %v, %tid; .reg .pred %p, %one, %consumer;
        .shared .align 16 .b8 data[16];
        .shared .align 8 .b64 loaded; .shared .align 8 .b64 handoff;
        mov.u32 %tid, %tid.x;
        setp.eq.u32 %one, %tid, 0;
        setp.ge.u32 %consumer, %tid, 32;
        @%one mbarrier.init.shared::cta.b64 [loaded], 1;
        @%one mbarrier.init.shared::cta.b64 [handoff], 1;
        bar.sync 0;
        ld.param.u64 %src, [src];
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 16;
        @%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [loaded];
        @%one mbarrier.try_wait.parity.shared::cta.b64 %p, [loaded], 0;
        @%one mbarrier.arrive.{arrive_semantics}.cta.shared::cta.b64 _, [handoff];
        @%consumer mbarrier.try_wait.parity.shared::cta.b64 %p, [handoff], 0;
        @%consumer ld.shared.u32 %v, [data];
        ret;
    """, ".param .u64 src")
    src = sim.from_numpy(np.arange(16, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=64, args=[src])
    assert [f.kind for f in stats.findings] == (
        ["tma-load-unsynchronized"] if arrive_semantics == "relaxed" else [])


@pytest.mark.parametrize("pattern", ["arrive_before_copy", "consumer_arrive", "synchronized"])
def test_named_barrier_only_publishes_prior_knowledge_to_waiters(sim, pattern):
    producer_before = "bar.arrive 1, 64;" if pattern == "arrive_before_copy" else ""
    producer_after = "" if pattern == "arrive_before_copy" else "bar.sync 1, 64;"
    consumer_barrier = "bar.arrive 1, 64;" if pattern == "consumer_arrive" else "bar.sync 1, 64;"
    delay = "\n".join(["mov.u32 %v, 0;"] * 20)
    mod = kernel(f"""
        .reg .b64 %src; .reg .b32 %v, %tid; .reg .pred %p, %one, %producer;
        .shared .align 16 .b8 data[16]; .shared .align 8 .b64 loaded;
        mov.u32 %tid, %tid.x;
        setp.eq.u32 %one, %tid, 0;
        setp.lt.u32 %producer, %tid, 32;
        @%one mbarrier.init.shared::cta.b64 [loaded], 1;
        bar.sync 0;
        @!%producer bra CONSUMER;
        {producer_before}
        ld.param.u64 %src, [src];
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [loaded], 16;
        @%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [loaded];
        @%one mbarrier.try_wait.parity.shared::cta.b64 %p, [loaded], 0;
        {producer_after}
        ret;
    CONSUMER:
        {delay}
        {consumer_barrier}
        ld.shared.u32 %v, [data];
        ret;
    """, ".param .u64 src")
    src = sim.from_numpy(np.arange(16, dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=64, args=[src], async_delay=0)
    assert [f.kind for f in stats.findings] == (
        [] if pattern == "synchronized" else ["tma-load-unsynchronized"])


@pytest.mark.parametrize("space", ["", ".shared::cta", ".shared::cluster", ".global"])
def test_proxy_fence_only_publishes_writes_in_its_state_space(sim, space):
    mod = kernel(f"""
        .reg .b64 %dst;
        .shared .align 16 .b8 data[16];
        ld.param.u64 %dst, [dst];
        st.shared.u32 [data], 1;
        st.shared.u32 [data+4], 2;
        st.shared.u32 [data+8], 3;
        st.shared.u32 [data+12], 4;
        fence.proxy.async{space};
        cp.async.bulk.global.shared::cta.bulk_group [%dst], [data], 16;
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
        ret;
    """, ".param .u64 dst")
    dst = sim.empty(4, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[dst])
    assert dst.numpy().tolist() == [1, 2, 3, 4]
    assert [f.kind for f in stats.findings] == (
        ["tma-store-missing-fence"] if space == ".global" else [])


@pytest.mark.parametrize("instruction", [
    "fence.proxy.tensormap::generic.release.gpu;",
    "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
    "fence.proxy.async.invalid;",
])
def test_unsupported_proxy_fences_are_not_silently_ignored(sim, instruction):
    with pytest.raises(warplens.UnsupportedError, match="fence.proxy"):
        sim.launch(kernel(instruction + "\nret;"), "k", grid=1, block=1)
