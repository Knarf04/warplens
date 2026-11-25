"""Cluster TMA addressing and completion routing, independent of CUTLASS."""

import numpy as np
import pytest

import warplens

from tests.support import kernel


def copy_kernel(*, multicast=False, group=2, destination=0, barrier=1, raw=False,
                issuer=0, pointer_mode="mapped"):
    """One CTA issues; an explicitly addressed CTA owns the completion barrier."""
    form = "" if raw else ".tensor.1d"
    grouping = f".cta_group::{group}" if group is not None else ""
    multi = ".multicast::cluster" if multicast else ""
    src = "[%src], 16" if raw else "[%src, {0}]"
    cta_mask = ", 3" if multicast else ""
    expected = 32 if multicast and group == 2 else 16
    receivers = ("mov.pred %receiver, 1;" if multicast and group != 2 else
                 f"setp.eq.u32 %receiver, %rank, {barrier};")
    pointers = f"""
        mapa.shared::cluster.u32 %dst, buffer, {destination};
        mapa.shared::cluster.u32 %bar, barrier, {barrier};
    """
    if pointer_mode != "mapped":
        pointers = "mov.u32 %dst, buffer; mov.u32 %bar, barrier;"
        if pointer_mode == "clear_peer":
            pointers += "and.b32 %bar, %bar, 0xfeffffff;"
    return kernel(f"""
        .reg .u32 %rank, %tid, %dst, %bar, %value;
        .reg .u64 %src, %out, %offset;
        .reg .pred %leader, %issue, %receiver, %p;
        .shared .align 16 .b8 buffer[16];
        .shared .align 8 .b64 barrier;
        mov.u32 %rank, %cluster_ctarank;
        mov.u32 %tid, %tid.x;
        setp.eq.u32 %leader, %tid, 0;
        @%leader mbarrier.init.shared::cta.b64 [barrier], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        {receivers}
        and.pred %receiver, %receiver, %leader;
        @%receiver mbarrier.arrive.expect_tx.shared::cta.b64 _, [barrier], {expected};
        {pointers}
        ld.param.u64 %src, [src];
        setp.eq.u32 %issue, %rank, {issuer};
        and.pred %issue, %issue, %leader;
        @%issue cp.async.bulk{form}{grouping}.shared::cluster.global.mbarrier::complete_tx::bytes{multi} [%dst], {src}, [%bar]{cta_mask};
        @%receiver mbarrier.try_wait.parity.shared::cta.b64 %p, [barrier], 0;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mov.u32 %value, 0;
        setp.eq.u32 %p, %rank, {destination};
        {"mov.pred %p, 1;" if multicast else ""}
        and.pred %p, %p, %leader;
        @%p ld.shared.u32 %value, [buffer];
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %rank, 4;
        add.u64 %out, %out, %offset;
        @%leader st.global.u32 [%out], %value;
        ret;
    """, ".param .u64 src, .param .u64 out")


@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("timing", [False, True])
def test_group2_completion_uses_explicit_odd_barrier(sim, multicast, timing):
    data = sim.from_numpy(np.array([17, 2, 3, 4], dtype=np.uint32))
    tensor_map = sim.tensor_map_tiled(data, 4)
    out = sim.empty(2, "u32")
    stats = sim.launch(copy_kernel(multicast=multicast), "k", grid=2, block=32, cluster=2,
                       args=[tensor_map, out], timing=timing)
    assert out.numpy().tolist() == ([17, 17] if multicast else [17, 0])
    assert not stats.findings


@pytest.mark.parametrize("raw", [False, True])
def test_cluster_copy_can_target_another_ctas_shared_memory(sim, raw):
    data = sim.from_numpy(np.array([29, 2, 3, 4], dtype=np.uint32))
    source = data if raw else sim.tensor_map_tiled(data, 4)
    out = sim.empty(2, "u32")
    stats = sim.launch(copy_kernel(group=None if raw else 1, destination=1, barrier=1, raw=raw),
                       "k", grid=2, block=32, cluster=2, args=[source, out])
    assert out.numpy().tolist() == [0, 29]
    assert not stats.findings


def test_group1_does_not_allow_barrier_in_the_destination_peer(sim):
    data = sim.from_numpy(np.arange(4, dtype=np.uint32))
    with pytest.raises(warplens.ExecutionError, match="same CTA"):
        sim.launch(copy_kernel(group=1), "k", grid=2, block=32, cluster=2,
                   args=[sim.tensor_map_tiled(data, 4), sim.empty(2, "u32")])


@pytest.mark.parametrize("pointer_mode,barrier", [("symbol", 1), ("clear_peer", 0)])
def test_group2_symbol_pointer_preserves_cutlass_peer_bit(sim, pointer_mode, barrier):
    data = sim.from_numpy(np.array([43, 2, 3, 4], dtype=np.uint32))
    out = sim.empty(2, "u32")
    stats = sim.launch(copy_kernel(destination=1, barrier=barrier, issuer=1,
                                   pointer_mode=pointer_mode),
                       "k", grid=2, block=32, cluster=2,
                       args=[sim.tensor_map_tiled(data, 4), out])
    assert out.numpy().tolist() == [0, 43]
    assert not stats.findings


@pytest.mark.parametrize("raw", [False, True])
def test_multicast_reports_to_each_destination_barrier(sim, raw):
    data = sim.from_numpy(np.array([53, 2, 3, 4], dtype=np.uint32))
    out = sim.empty(2, "u32")
    source = data if raw else sim.tensor_map_tiled(data, 4)
    stats = sim.launch(copy_kernel(multicast=True, raw=raw, group=None if raw else 1,
                                   destination=0, barrier=0),
                       "k", grid=2, block=32, cluster=2, args=[source, out])
    assert out.numpy().tolist() == [53, 53]
    assert not stats.findings


@pytest.mark.parametrize("raw", [False, True])
def test_bulk_store_from_odd_cta_uses_local_shared_offset(sim, raw):
    out = sim.empty(8, "u32")
    destination = out if raw else sim.tensor_map_tiled(out, 4)
    store = ("cp.async.bulk.global.shared::cta.bulk_group [%dst], [buffer], 16;" if raw else
             "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%out, {%column}], [buffer];")
    ptx = kernel(f"""
        .reg .u32 %rank, %tid, %column, %value;
        .reg .u64 %out, %dst, %offset;
        .shared .align 16 .b8 buffer[16];
        .reg .pred %leader;
        mov.u32 %tid, %tid.x;
        setp.eq.u32 %leader, %tid, 0;
        @!%leader ret;
        mov.u32 %rank, %cluster_ctarank;
        add.u32 %value, %rank, 61;
        st.shared.v4.u32 [buffer], {{%value, %value, %value, %value}};
        fence.proxy.async.shared::cta;
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %rank, 16;
        add.u64 %dst, %out, %offset;
        mul.lo.u32 %column, %rank, 4;
        {store}
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
        ret;
    """, ".param .u64 out")
    stats = sim.launch(ptx, "k", grid=2, block=32, cluster=2, args=[destination])
    assert out.numpy().tolist() == [61] * 4 + [62] * 4
    assert not stats.findings


@pytest.mark.parametrize("instruction,message", [
    ("cp.async.bulk.cta_group::1.shared::cluster.global.mbarrier::complete_tx::bytes [0], [0], 16, [0];",
     "only valid on tensor loads"),
    ("cp.async.bulk.tensor.1d.cta_group::2.global.shared::cta.bulk_group [0, {0}], [0];",
     "only valid on tensor loads"),
    ("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.multicast::cluster [0], [0], 16, [0], 1;",
     "requires .shared::cluster"),
])
def test_tma_rejects_invalid_group_and_multicast_forms(sim, instruction, message):
    with pytest.raises(warplens.UnsupportedError, match=message):
        sim.launch(kernel(instruction + "ret;"), "k", grid=1, block=1)
