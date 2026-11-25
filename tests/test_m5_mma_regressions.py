"""CTA-pair MMA and cluster-address commit contracts from the PTX specification."""

import numpy as np
import pytest

from tests.support import kernel
from tests.support.cluster import pair_kernel
from tests.support.tcgen05 import place
from warplens.frontend.ir import Imm, Reg
from warplens.viz import tmem as tmem_view


@pytest.mark.parametrize("m", [128, 256])
@pytest.mark.parametrize("kind", ["f16", "tf32"])
@pytest.mark.parametrize("timing", [False, True])
def test_odd_cta_may_issue_the_pair_mma(sim, m, kind, timing):
    n = 16
    columns = n // 2 if m == 128 else n
    rng = np.random.default_rng(m)
    k = 16 if kind == "f16" else 8
    dtype = np.float16 if kind == "f16" else np.float32
    # Half values are exactly representable as TF32 too.
    a = rng.uniform(-2, 2, (m, k)).astype(np.float16).astype(dtype)
    b = rng.uniform(-2, 2, (n, k)).astype(np.float16).astype(dtype)
    image = np.zeros((2, 32768), dtype=np.uint8)
    for rank in range(2):
        da = place(a[rank * m // 2:(rank + 1) * m // 2], "128B", 0, image[rank])
        db = place(b[rank * n // 2:(rank + 1) * n // 2], "128B", 16384, image[rank])
    mod = pair_kernel()
    for ins in mod.functions["k"].instrs:
        if ins.opcode == "setp" and ins.operands[:2] == (Reg("%lead"), Reg("%rank")):
            ins.operands = (*ins.operands[:2], Imm("int", 1))
        if ins.opcode == "tcgen05" and "mma" in ins.mods:
            ins.mods = tuple(f"kind::{kind}" if mod.startswith("kind::") else mod for mod in ins.mods)
    out = sim.from_numpy(np.full((2, 128, columns), np.nan, dtype=np.float32))
    idesc = (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
    if kind == "tf32":
        idesc |= (2 << 7) | (2 << 10)
    stats = sim.launch(mod, "k", grid=2, block=128, cluster=2, timing=timing,
                       args=[sim.from_numpy(image), out, da, db, idesc, columns])
    expected = a.astype(np.float32) @ b.astype(np.float32).T
    got = out.numpy()
    for row in range(m):
        for col in range(n):
            rank, lane, column = tmem_view.locate(row, col, m, n, pair=True)
            assert got[rank, lane, column] == pytest.approx(expected[row, col], rel=1e-5, abs=1e-5)
    assert not stats.findings


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("timing", [False, True])
def test_commit_resolves_cluster_address_before_selecting_barriers(sim, generic, multicast, timing):
    # Each CTA commits to its peer, or CTA 0 multicasts to both using a mapped
    # barrier pointer. The rank bits of that pointer are not part of the offset.
    space = "" if generic else ".shared::cluster"
    pointer = "%generic" if generic else "%addr"
    commit = f"tcgen05.commit.cta_group::1.mbarrier::arrive::one{space}"
    commit += ".multicast::cluster.b64" if multicast else ".b64"
    predicate = "@%lead " if multicast else ""
    mask = ", %both" if multicast else ""
    mod = kernel(f"""
        .reg .b32 %rank, %other, %addr; .reg .b64 %generic;
        .reg .b16 %both; .reg .pred %p, %lead;
        .shared .align 8 .u64 bar;
        mov.u32 %rank, %cluster_ctarank;
        xor.b32 %other, %rank, 1;
        setp.eq.u32 %lead, %rank, 0;
        mbarrier.init.shared::cta.b64 [bar], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mapa.shared::cluster.u32 %addr, bar, %other;
        cvt.u64.u32 %generic, %addr;
        cvta.shared.u64 %generic, %generic;
        mov.b16 %both, 3;
        {predicate}{commit} [{pointer}]{mask};
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=2, cluster=2, block=1, timing=timing)
    assert not stats.findings


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("timing", [False, True])
def test_alloc_writes_the_local_shared_destination_in_each_cta(sim, generic, timing):
    space = "" if generic else ".shared::cta"
    pointer = "%generic" if generic else "slot"
    mod = kernel(f"""
        .reg .b32 %rank, %lane, %column; .reg .b64 %out, %dst, %generic;
        .reg .pred %lead;
        .shared .align 4 .u32 slot;
        ld.param.u64 %out, [out];
        mov.u32 %rank, %cluster_ctarank;
        mov.u32 %lane, %laneid;
        setp.eq.u32 %lead, %lane, 0;
        cvta.shared.u64 %generic, slot;
        tcgen05.alloc.cta_group::1.sync.aligned{space}.b32 [{pointer}], 32;
        ld.shared.u32 %column, [slot];
        mad.wide.u32 %dst, %rank, 4, %out;
        @%lead st.global.u32 [%dst], %column;
        tcgen05.dealloc.cta_group::1.sync.aligned.b32 %column, 32;
        ret;
    """, ".param .u64 out")
    out = sim.from_numpy(np.full(2, 0xFFFFFFFF, dtype=np.uint32))
    stats = sim.launch(mod, "k", grid=2, cluster=2, block=32, timing=timing, args=[out])
    np.testing.assert_array_equal(out.numpy(), [0, 0])
    assert not stats.findings


@pytest.mark.parametrize("n", [1, 8, 17, 272])
def test_pair_view_rejects_unsupported_column_counts(n):
    with pytest.raises(ValueError, match="multiple of 16"):
        tmem_view.pair_text(128, n)


def test_pair_view_rejects_ignored_lane_offset_and_draws_the_svg(capsys, tmp_path):
    with pytest.raises(ValueError, match="lane offset"):
        tmem_view.locate(0, 0, 128, 16, pair=True, lane_offset=16)
    assert tmem_view.main(["128", "16", "--pair", "--lane-offset", "16"]) == 2
    assert "lane offset" in capsys.readouterr().err
    # The option used to be refused as not implemented; the picture exists now.
    assert tmem_view.main(["128", "16", "--pair", "--svg", str(tmp_path / "pair.svg")]) == 0
    assert (tmp_path / "pair.svg").read_text().startswith("<svg")
    assert tmem_view.main(["128", "16", "--pair", "--lane-offset", "16", "--svg", str(tmp_path / "no.svg")]) == 2
    assert not (tmp_path / "no.svg").exists()
