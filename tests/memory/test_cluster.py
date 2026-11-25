"""Clusters: cluster launch, the cluster barrier, multicast, and the two-CTA MMA."""

import numpy as np
import pytest

import warplens
from warplens.viz import tmem as tmem_view

from tests.support import kernel
from tests.support.cluster import pair_kernel
from tests.support.tcgen05 import place

OUT = ".param .u64 out"


# -- launch and special registers ---------------------------------------------------------

def test_cluster_special_registers(sim):
    mod = kernel("""
        .reg .b32 %v<8>, %i; .reg .b64 %o, %off;
        mov.u32 %v0, %cluster_ctarank; mov.u32 %v1, %cluster_nctarank;
        mov.u32 %v2, %cluster_ctaid.x; mov.u32 %v3, %cluster_ctaid.y;
        mov.u32 %v4, %clusterid.x; mov.u32 %v5, %clusterid.y;
        mov.u32 %v6, %nclusterid.x; mov.u32 %v7, %is_explicit_cluster;
        mov.u32 %i, %ctaid.y; mov.u32 %off, 0;
        mov.u32 %i, %ctaid.y;
        mul.lo.u32 %i, %i, 4;
        mov.u32 %v1, %v1;
        add.u32 %i, %i, 0;
        .reg .b32 %cx;
        mov.u32 %cx, %ctaid.x;
        add.u32 %i, %i, %cx;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %i, 32;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %v0; st.global.u32 [%o+4], %v1; st.global.u32 [%o+8], %v2;
        st.global.u32 [%o+12], %v3; st.global.u32 [%o+16], %v4; st.global.u32 [%o+20], %v5;
        st.global.u32 [%o+24], %v6; st.global.u32 [%o+28], %v7;
        ret;
    """, OUT)
    out = sim.empty((2, 4, 8), "u32")  # indexed [ctaid.y][ctaid.x]
    stats = sim.launch(mod, "k", grid=(4, 2), block=1, args=[out], cluster=(2, 2))
    got = out.numpy()
    for y in range(2):
        for x in range(4):
            assert got[y, x].tolist() == [x % 2 + 2 * (y % 2), 4, x % 2, y % 2, x // 2, y // 2, 2, 1]
    assert stats.ctas == 8
    plain = sim.empty((2, 4, 8), "u32")
    sim.launch(mod, "k", grid=(4, 2), block=1, args=[plain])
    assert plain.numpy()[1, 3].tolist() == [0, 1, 0, 0, 3, 1, 4, 0]  # every CTA is its own cluster


def test_cluster_shape_errors(sim):
    mod = kernel("ret;")
    with pytest.raises(warplens.LaunchError, match="not a whole number of clusters"):
        sim.launch(mod, "k", grid=(3, 2), block=1, cluster=(2, 2))
    with pytest.raises(warplens.LaunchError, match="more than 16 CTAs"):
        sim.launch(mod, "k", grid=(8, 4), block=1, cluster=(8, 4))


# -- the cluster barrier --------------------------------------------------------------------

BARRIER = """
    .reg .b32 %r, %v, %c; .reg .b64 %o; .reg .pred %p;
    .shared .align 4 .u32 cell;
    mov.u32 %r, %cluster_ctarank;
    setp.eq.u32 %p, %r, 0;
    ld.param.u64 %o, [out];
    @%p ld.global.u32 %v, [%o];
    @%p ld.global.u32 %v, [%o];
    @%p ld.global.u32 %v, [%o];
    @%p st.global.u32 [%o], 7;
    {sync}
    @!%p ld.global.u32 %v, [%o];
    @!%p st.global.u32 [%o+4], %v;
    ret;
"""


def test_cluster_barrier_orders_one_cta_after_another(sim):
    # CTA 0 writes late; CTA 1 reads after the cluster barrier and must see it.
    synced = kernel(BARRIER.format(sync="barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;"), OUT)
    out = sim.empty(2, "u32")
    sim.launch(synced, "k", grid=2, block=32, args=[out], cluster=2)
    assert out.numpy().tolist() == [7, 7]


def test_cluster_barrier_that_cannot_complete_is_a_deadlock(sim):
    mod = kernel("""
        .reg .b32 %r; .reg .pred %p;
        mov.u32 %r, %cluster_ctarank;
        setp.eq.u32 %p, %r, 0;
        @%p barrier.cluster.arrive.aligned;
        @%p barrier.cluster.wait.aligned;
        @!%p bar.sync 1, 64;
        ret;
    """)
    with pytest.raises(warplens.DeadlockError) as info:
        sim.launch(mod, "k", grid=2, block=32, cluster=2)
    msg = str(info.value)
    assert "deadlock in cluster (0, 0, 0)" in msg and "in CTA (1, 0, 0) (rank 1 in the cluster)" in msg
    assert "cluster barrier: 1 warp(s) arrived, needs all 2 running warp(s)" in msg


# -- TMA multicast -------------------------------------------------------------------------

def test_tma_multicast_delivers_to_every_cta_in_the_mask(sim):
    # CTA 0 loads once with mask 0b101: CTAs 0 and 2 receive it, CTA 1 never does.
    data = np.arange(64, dtype=np.uint32) + 1
    mod = kernel("""
        .reg .b64 %s, %o, %off; .reg .b32 %r, %buf, %bar, %v; .reg .b16 %m; .reg .pred %p, %first, %got;
        .shared .align 1024 .b8 buf[64];
        .shared .align 8 .u64 bar;
        ld.param.u64 %s, [src]; ld.param.u64 %o, [out];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        mov.u32 %r, %cluster_ctarank;
        setp.eq.u32 %first, %r, 0;
        setp.ne.u32 %got, %r, 1;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        @%got mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 64;
        mov.b16 %m, 5;
        @%first cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%buf], [%s, {16}], [%bar], %m;
        @%got mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        mov.u32 %v, 0;
        @%got ld.shared.u32 %v, [%buf+8];
        mul.wide.u32 %off, %r, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %v;
        ret;
    """, ".param .u64 src, .param .u64 out")
    out = sim.empty(3, "u32")
    stats = sim.launch(mod, "k", grid=3, block=1, args=[sim.tensor_map_tiled(sim.from_numpy(data), 16), out],
                       cluster=3)
    assert out.numpy().tolist() == [19, 0, 19] and not stats.findings  # data[16 + 2]
    with pytest.raises(warplens.ExecutionError, match="CTA rank 2 does not exist in a cluster of 1"):
        sim.launch(mod, "k", grid=3, block=1, args=[sim.tensor_map_tiled(sim.from_numpy(data), 16), out])


# -- the two-CTA MMA -----------------------------------------------------------------------

def run_pair(sim, m, n, columns):
    """A is m x 16 and B is n x 16, half in each CTA. Returns (TMEM dumps [cta][lane][column], A B^T)."""
    rng = np.random.default_rng(m + n)
    a = rng.uniform(-2, 2, (m, 16)).astype(np.float16)
    b = rng.uniform(-2, 2, (n, 16)).astype(np.float16)
    image = np.zeros((2, 32768), dtype=np.uint8)
    for cta in range(2):
        desc_a = place(a[cta * m // 2:(cta + 1) * m // 2], "128B", 0, image[cta])
        desc_b = place(b[cta * n // 2:(cta + 1) * n // 2], "128B", 16384, image[cta])
    idesc = (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
    out = sim.from_numpy(np.full((2, 128, columns), np.nan, dtype=np.float32))
    stats = sim.launch(pair_kernel(), "k", grid=2, block=128, cluster=2,
                       args=[sim.from_numpy(image), out, desc_a, desc_b, idesc, columns])
    return out.numpy(), a.astype(np.float32) @ b.astype(np.float32).T, stats


def test_two_cta_mma_with_256_rows(sim):
    got, expect, stats = run_pair(sim, 256, 16, 16)
    np.testing.assert_allclose(got[0], expect[:128], rtol=1e-5, atol=1e-5)  # even CTA: rows 0-127
    np.testing.assert_allclose(got[1], expect[128:], rtol=1e-5, atol=1e-5)  # odd CTA: rows 128-255
    assert not stats.findings and stats.ctas == 2


def test_layout_b_and_the_course_notes_worked_example(sim):
    """cta_group::2 with M = 128 (Layout B). The course notes: 64 rows only fill half the lanes,
    so N is folded; with N = 16, C[10,3] is on the even CTA's lane 10 and C[10,11] on lane 74."""
    got, expect, stats = run_pair(sim, 128, 16, 8)
    assert not stats.findings
    for m in range(128):
        for n in range(16):
            cta, lane, column = tmem_view.locate(m, n, 128, 16, pair=True)
            assert got[cta, lane, column] == pytest.approx(expect[m, n], rel=1e-5, abs=1e-5), (m, n)
    assert tmem_view.locate(10, 3, 128, 16, pair=True) == (0, 10, 3)
    assert tmem_view.locate(10, 11, 128, 16, pair=True) == (0, 74, 3)
    assert got[0, 74, 3] == pytest.approx(expect[10, 11], rel=1e-5, abs=1e-5)
    assert tmem_view.locate(70, 11, 128, 16, pair=True) == (1, 70, 3)  # rows 64-127 are on the odd CTA


def test_pair_layout_view(capsys):
    text = tmem_view.pair_text(128, 16)
    assert "Layout B" in text and "even   64-127     rows 0-63, columns 8-15 in TMEM columns 0-7" in text
    assert "odd    0-127      rows 128-255" in tmem_view.pair_text(256, 64)
    assert tmem_view.main(["128", "16", "--pair"]) == 0 and "cta_group::2" in capsys.readouterr().out
    assert tmem_view.locate(5, 9, 64, 32) == (0, 5, 9) and tmem_view.locate(20, 1, 64, 32) == (0, 36, 1)
    with pytest.raises(ValueError, match="128 or 256"):
        tmem_view.pair_text(64, 16)


def test_two_cta_mma_requires_a_complete_pair(sim):
    mod = kernel("""
        .reg .b32 %a; .reg .b64 %d; .reg .pred %p;
        setp.ne.u32 %p, %a, %a;
        tcgen05.mma.cta_group::2.kind::f16 [%a], %d, %d, 0x08200010, {0, 0, 0, 0, 0, 0, 0, 0}, %p;
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="requires both CTAs of a pair; this is rank 0 in a cluster of 1"):
        sim.launch(mod, "k", grid=1, block=1)


# -- distributed shared memory and remote mbarrier arrival ---------------------------------------

def test_store_into_another_ctas_shared_memory(sim):
    # CTA 0 writes into CTA 1's `cell` through a cluster address from mapa, and reads it back.
    mod = kernel("""
        .reg .b32 %r, %a, %v, %w; .reg .b64 %o, %off; .reg .pred %p;
        .shared .align 4 .u32 pad[3];
        .shared .align 4 .u32 cell;
        mov.u32 %r, %cluster_ctarank;
        setp.eq.u32 %p, %r, 0;
        mapa.shared::cluster.u32 %a, cell, 1;
        @%p st.shared::cluster.u32 [%a], 4242;
        @%p ld.shared::cluster.u32 %w, [%a];
        @!%p st.shared.u32 [pad], 5;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mov.u32 %v, 0;
        @!%p ld.shared.u32 %v, [cell];
        @%p mov.u32 %v, %w;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %r, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %v;
        ret;
    """, OUT)
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=2, block=1, args=[out], cluster=2)
    assert out.numpy().tolist() == [4242, 4242] and not stats.findings
    with pytest.raises(warplens.ExecutionError, match="mapa to CTA rank 1, but the cluster has 1"):
        sim.launch(mod, "k", grid=2, block=1, args=[out])


def test_arrive_on_another_ctas_mbarrier(sim):
    # CTA 1 waits on its own mbarrier; CTA 0 completes it remotely.
    mod = kernel("""
        .reg .b32 %r, %a, %bar; .reg .b64 %o, %off; .reg .pred %p, %q;
        .shared .align 8 .u64 bar;
        mov.u32 %r, %cluster_ctarank;
        setp.eq.u32 %p, %r, 0;
        mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mapa.shared::cluster.u32 %a, bar, 1;
        @%p mbarrier.arrive.shared::cluster.b64 _, [%a];
        @!%p mbarrier.try_wait.parity.shared::cta.b64 %q, [%bar], 0;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %r, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], 1;
        ret;
    """, OUT)
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=2, block=1, args=[out], cluster=2)
    assert out.numpy().tolist() == [1, 1] and not stats.findings
    remote_wait = kernel("""
        .reg .b32 %a; .reg .pred %q;
        .shared .align 8 .u64 bar;
        mov.b32 %a, bar;
        mbarrier.init.shared::cta.b64 [%a], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mapa.shared::cluster.u32 %a, bar, 1;
        mbarrier.test_wait.parity.shared::cluster.b64 %q, [%a], 0;
        ret;
    """)
    with pytest.raises(warplens.ExecutionError, match="only\\s+arrive may be remote"):
        sim.launch(remote_wait, "k", grid=2, block=1, cluster=2)


def test_timed_run_of_a_cluster(sim):
    got, expect, stats = run_pair_timed(sim)
    np.testing.assert_allclose(got[0], expect[:128], rtol=1e-5, atol=1e-5)
    t = stats.timing
    assert t.ctas == 2 and t.time_ns > 0 and not stats.findings
    assert {s.cta for s in t.slices} == {(0, 0, 0), (1, 0, 0)}
    assert sum(s.track == "Tensor Core" for s in t.slices) == 1  # one MMA, charged to the issuing CTA


def run_pair_timed(sim):
    rng = np.random.default_rng(7)
    a = rng.uniform(-2, 2, (256, 16)).astype(np.float16)
    b = rng.uniform(-2, 2, (16, 16)).astype(np.float16)
    image = np.zeros((2, 32768), dtype=np.uint8)
    for cta in range(2):
        desc_a = place(a[cta * 128:(cta + 1) * 128], "128B", 0, image[cta])
        desc_b = place(b[cta * 8:(cta + 1) * 8], "128B", 16384, image[cta])
    out = sim.from_numpy(np.full((2, 128, 16), np.nan, dtype=np.float32))
    stats = sim.launch(pair_kernel(), "k", grid=2, block=128, cluster=2, timing=True,
                       args=[sim.from_numpy(image), out, desc_a, desc_b,
                             (1 << 4) | (2 << 17) | (16 << 24), 16])
    return out.numpy(), a.astype(np.float32) @ b.astype(np.float32).T, stats
