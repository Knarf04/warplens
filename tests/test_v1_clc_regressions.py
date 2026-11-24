"""CLC responses must not silently reuse a pending asynchronous reader's bytes."""

import pytest

from conftest import kernel
from test_m2_tma_regressions import launch


@pytest.mark.parametrize("delay", [0, 1000])
@pytest.mark.parametrize("wait", [False, True])
def test_clc_response_reuse_of_tma_store_source_requires_wait(sim, delay, wait):
    _, stats = launch(sim, f"""
        mbarrier.init.shared::cta.b64 [bar], 1;
        cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;
        cp.async.bulk.commit_group;
        {'cp.async.bulk.wait_group.read 0;' if wait else ''}
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [buf], [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        cp.async.bulk.wait_group.read 0;
    """, delay=delay)
    assert ("tma-store-source-overwritten" in {f.kind for f in stats.findings}) is not wait
    if not wait:
        assert "clusterlaunchcontrol.try_cancel" in stats.findings[0].text


@pytest.mark.parametrize("wait", [False, True])
def test_multicast_checks_async_readers_in_the_receiving_cta(sim, wait):
    mod = kernel(f"""
        .reg .u32 %rank;
        .reg .u64 %out;
        .reg .pred %leader, %p;
        .shared .align 16 .b8 response[16];
        .shared .align 8 .u64 bar;
        ld.param.u64 %out, [out];
        mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %leader, %rank, 0;
        st.shared.v4.u32 [response], {{1, 2, 3, 4}};
        fence.proxy.async.shared::cta;
        mbarrier.init.shared::cta.b64 [bar], 1;
        @!%leader cp.async.bulk.global.shared::cta.bulk_group [%out], [response], 16;
        @!%leader cp.async.bulk.commit_group;
        {'@!%leader cp.async.bulk.wait_group.read 0;' if wait else ''}
        barrier.cluster.arrive;
        barrier.cluster.wait;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
        @%leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [response], [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        @!%leader cp.async.bulk.wait_group.read 0;
        ret;
    """, ".param .u64 out")
    stats = sim.launch(mod, "k", grid=2, block=1, cluster=2,
                       args=[sim.empty(4, "u32")], async_delay=1000)
    findings = [f for f in stats.findings if f.kind == "tma-store-source-overwritten"]
    assert bool(findings) is not wait
    if not wait:
        assert "CTA (1, 0, 0)" in findings[0].message


@pytest.mark.parametrize("wait", [False, True])
def test_clc_response_reuse_of_mma_operand_requires_commit_wait(sim, monkeypatch, wait):
    import test_tcgen05 as helpers

    def with_clc_response(body, params):
        body = body.replace(".shared .align 8 .u64 bar;",
                            ".shared .align 8 .u64 bar; .shared .align 8 .u64 clc_bar;")
        body = body.replace("shl.b32 %x, %w, 21;", """
            @%one mbarrier.init.shared::cta.b64 [clc_bar], 1;
            bar.sync 1;
            @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [clc_bar], 16;
            @%one clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%buf], [clc_bar];
            mbarrier.try_wait.parity.shared::cta.b64 %p, [clc_bar], 0;
            shl.b32 %x, %w, 21;
        """)
        return kernel(body, params)

    monkeypatch.setattr(helpers, "kernel", with_clc_response)
    _, _, stats = helpers.run(sim, n=8, wait=int(wait))
    findings = [f for f in stats.findings if f.kind == "mma-operand-overwritten"]
    assert bool(findings) is not wait
    if not wait:
        assert "tcgen05.commit" in findings[0].message
