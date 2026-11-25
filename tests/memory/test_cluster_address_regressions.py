"""Cluster pointers must retain their target through ordinary PTX arithmetic."""

import pytest

import warplens

from tests.support import kernel


@pytest.mark.parametrize("generic", [False, True])
def test_mapped_pointer_can_be_remapped_and_accessed_generically(sim, generic):
    convert = "cvt.u64.u32 %generic, %addr; cvta.shared::cluster.u64 %generic, %generic;" if generic else ""
    store = "st.u32 [%generic], 42;" if generic else "st.shared::cluster.u32 [%addr], 42;"
    load = "ld.u32 %value, [%generic];" if generic else "ld.shared::cluster.u32 %value, [%addr];"
    mod = kernel("""
        .reg .b32 %rank, %addr, %value;
        .reg .b64 %generic, %out, %offset;
        .shared .align 4 .u32 cell;
        mov.u32 %rank, %cluster_ctarank;
        mapa.shared::cluster.u32 %addr, cell, 1;
        mapa.shared::cluster.u32 %addr, %addr, %rank;
    """ + convert + store + load + """
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %rank, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %value;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=2, block=1, cluster=2, args=[out])
    assert out.numpy().tolist() == [42, 42]
    assert not stats.findings


@pytest.mark.parametrize("generic", [False, True])
def test_cutlass_peer_bit_mask_selects_even_cta_from_each_pair(sim, generic):
    # CUTLASS copy_sm100_tma.hpp clears bit 24 to select the even peer.
    mod = kernel("""
        .reg .b32 %rank, %addr, %value;
        .reg .b64 %out, %offset, %generic;
        .shared .align 4 .u32 cell;
        mov.u32 %rank, %cluster_ctarank;
        st.shared.u32 [cell], %rank;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mov.u32 %addr, cell;
        and.b32 %addr, %addr, 0xfeffffff;
    """ + ("cvt.u64.u32 %generic, %addr; cvta.shared::cluster.u64 %generic, %generic; "
           "ld.u32 %value, [%generic];" if generic else
           "ld.shared::cluster.u32 %value, [%addr];") + """
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %rank, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %value;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ret;
    """, ".param .u64 out")
    out = sim.empty(4, "u32")
    stats = sim.launch(mod, "k", grid=4, block=1, cluster=4, args=[out])
    assert out.numpy().tolist() == [0, 0, 2, 2]
    assert not stats.findings


def test_aligned_cluster_barrier_reports_partial_warp_participation(sim):
    mod = kernel("""
        .reg .b32 %lane; .reg .pred %one;
        mov.u32 %lane, %laneid;
        setp.eq.u32 %one, %lane, 0;
        @%one barrier.cluster.arrive.aligned;
        @%one barrier.cluster.wait.aligned;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=2, block=32, cluster=2)
    assert "barrier-in-divergent-code" in {f.kind for f in stats.findings}


@pytest.mark.parametrize("mods", ["arrive.acquire", "wait.release", "wait.relaxed",
                                 "arrive.aligned.aligned"])
def test_invalid_cluster_barrier_qualifiers_are_not_silently_ignored(sim, mods):
    with pytest.raises(warplens.UnsupportedError):
        sim.launch(kernel(f"barrier.cluster.{mods}; ret;"), "k", grid=1, block=1)


def test_mapa_without_a_space_maps_generic_addresses(sim):
    """`mapa.u64 d, a, rank`, as TIRx emits it: `a` and the result are generic addresses."""
    mod = kernel("""
        .reg .b32 %rank, %other, %value;
        .reg .b64 %here, %there, %out, %offset;
        .shared .align 4 .u32 cell;
        mov.u32 %rank, %cluster_ctarank;
        xor.b32 %other, %rank, 1;
        mov.u64 %here, cell;
        cvta.shared.u64 %here, %here;
        add.u32 %value, %rank, 10;
        st.u32 [%here], %value;
        barrier.cluster.arrive.release.aligned;
        barrier.cluster.wait.acquire.aligned;
        mapa.u64 %there, %here, %other;
        ld.u32 %value, [%there];
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %rank, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %value;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=2, block=32, cluster=2, args=[out])
    assert out.numpy().tolist() == [11, 10]  # each CTA read the other's cell
    assert not stats.findings


def test_mapa_without_a_space_rejects_an_address_outside_shared_memory(sim):
    mod = kernel("""
        .reg .b64 %a;
        ld.param.u64 %a, [out];
        mapa.u64 %a, %a, 0;
        ret;
    """, ".param .u64 out")
    with pytest.raises(warplens.ExecutionError, match="not a generic address of shared memory"):
        sim.launch(mod, "k", grid=2, block=1, cluster=2, args=[sim.empty(1, "u32")])
