"""TMA group ordering and buffer reuse regression cases."""

import numpy as np
import pytest

import warplens
from conftest import kernel


SETUP = """
    .reg .b64 %s, %d;
    .reg .b32 %t;
    .reg .pred %p, %one;
    .shared .align 16 .b8 buf[16];
    .shared .align 8 .u64 bar;
    ld.param.u64 %s, [src]; ld.param.u64 %d, [dst];
    mov.u32 %t, %tid.x;
    setp.eq.u32 %one, %t, 0;
    @%one st.shared.u32 [buf], 7;
    @%one st.shared.u32 [buf+4], 7;
    @%one st.shared.u32 [buf+8], 7;
    @%one st.shared.u32 [buf+12], 7;
    fence.proxy.async.shared::cta;
"""


def launch(sim, body, *, block=1, delay=1000):
    src = sim.from_numpy(np.arange(8, dtype=np.uint32))
    dst = sim.empty(4, "u32")
    stats = sim.launch(kernel(SETUP + body + "\nret;",
                              ".param .u64 src, .param .u64 dst"),
                       "k", grid=1, block=block, args=[src, dst], async_delay=delay)
    return dst.numpy(), stats


@pytest.mark.parametrize("wait", ["wait_group", "wait_group.read"])
def test_empty_group_is_counted_among_the_newest_groups(sim, wait):
    dst, stats = launch(sim, f"""
        cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;
        cp.async.bulk.commit_group;
        cp.async.bulk.commit_group;
        cp.async.bulk.{wait} 1;
        st.shared.u32 [buf], 99;
    """)
    np.testing.assert_array_equal(dst, [7, 7, 7, 7])
    assert not stats.findings


@pytest.mark.parametrize("foreign_action", ["commit_group", "wait_group.read 0"])
def test_another_lane_cannot_commit_or_wait_for_the_issuers_store(sim, foreign_action):
    commit = "@%one cp.async.bulk.commit_group;" if foreign_action.startswith("wait") else ""
    _, stats = launch(sim, f"""
        @%one cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;
        {commit}
        @!%one cp.async.bulk.{foreign_action};
        @%one cp.async.bulk.wait_group.read 0;
        @%one st.shared.u32 [buf], 99;
    """, block=2)
    # A foreign wait is a no-op, but the issuer's subsequent wait is valid;
    # a foreign commit must leave the issuer's store uncommitted.
    kinds = {f.kind for f in stats.findings}
    if foreign_action == "commit_group":
        assert "tma-store-source-overwritten" in kinds
    else:
        assert not kinds


def test_another_lanes_wait_does_not_synchronize_a_store(sim):
    _, stats = launch(sim, """
        @%one cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;
        @%one cp.async.bulk.commit_group;
        @!%one cp.async.bulk.wait_group.read 0;
        @%one st.shared.u32 [buf], 99;
    """, block=2)
    assert "tma-store-source-overwritten" in {f.kind for f in stats.findings}


@pytest.mark.parametrize("delay", [0, 4, 1000])
@pytest.mark.parametrize("wait", [False, True])
def test_tma_load_reuse_of_a_store_source_requires_wait(sim, delay, wait):
    _, stats = launch(sim, f"""
        mbarrier.init.shared::cta.b64 [bar], 1;
        cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;
        cp.async.bulk.commit_group;
        {'cp.async.bulk.wait_group.read 0;' if wait else ''}
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [%s], 16, [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        cp.async.bulk.wait_group.read 0;
    """, delay=delay)
    kinds = {f.kind for f in stats.findings}
    assert ("tma-store-source-overwritten" in kinds) is not wait


@pytest.mark.parametrize("delay", [0, 1000])
def test_overlapping_tma_loads_need_ordering(sim, delay):
    _, stats = launch(sim, """
        mbarrier.init.shared::cta.b64 [bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 32;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [%s], 16, [bar];
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [%s+16], 16, [bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
    """, delay=delay)
    assert "tma-load-unsynchronized" in {f.kind for f in stats.findings}


@pytest.mark.parametrize("count", ["-1", "%t"])
def test_bulk_wait_count_must_be_a_nonnegative_integer_constant(sim, count):
    with pytest.raises(warplens.ExecutionError, match="nonnegative integer constant"):
        launch(sim, f"cp.async.bulk.wait_group {count};")


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("coord", [-1, 1])
def test_tensor_coordinate_must_preserve_bounding_box_alignment(sim, direction, coord):
    data = sim.from_numpy(np.arange(8, dtype=np.uint32))
    descriptor = sim.tensor_map_tiled(data, 4)
    copy = (f"cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes "
            f"[buf], [%s, {{{coord}}}], [bar];") if direction == "load" else (
            f"cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%s, {{{coord}}}], [buf];")
    mod = kernel(SETUP + "mbarrier.init.shared::cta.b64 [bar], 1;\n" + copy + "\nret;",
                 ".param .u64 src, .param .u64 dst")
    with pytest.raises(warplens.ExecutionError, match="16-byte aligned bounding box"):
        sim.launch(mod, "k", grid=1, block=1, args=[descriptor, data])


def test_explicit_tile_mode_matches_the_default(sim):
    from test_tma import box_kernel

    data = np.arange(8, dtype=np.uint32)
    src, dst = sim.from_numpy(data), sim.empty(4, "u32")
    mod = box_kernel(1)
    # Exercise both load and store spelling without duplicating the round trip kernel.
    for ins in mod.functions["k"].instrs:
        if ins.opcode == "cp" and "tensor" in ins.mods:
            mods = list(ins.mods)
            mods.insert(6, "tile")
            ins.mods = tuple(mods)
    stats = sim.launch(mod, "k", grid=1, block=1,
                       args=[sim.tensor_map_tiled(src, 4), sim.tensor_map_tiled(dst, 4), 16, 4])
    np.testing.assert_array_equal(dst.numpy(), data[4:])
    assert not stats.findings


@pytest.mark.parametrize("tensor", [False, True])
def test_oversized_copy_is_rejected_before_materializing_byte_indices(sim, monkeypatch, tensor):
    # If the bounds check regresses, fail without attempting a multi-GB allocation.
    arange = np.arange

    def bounded_arange(stop, *args, **kwargs):
        assert stop <= 1024, "oversized copy allocated indices before checking shared bounds"
        return arange(stop, *args, **kwargs)

    if tensor:
        data = sim.empty((1, 1, 1, 1, 2), "u64")
        src = sim.tensor_map_tiled(data, (256,) * 5)
        copy = ("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[buf], [%s, {0, 0, 0, 0, 0}], [bar];")
    else:
        src = sim.empty(16, "u8")
        copy = ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[buf], [%s], 4294967280, [bar];")
    mod = kernel(SETUP + "mbarrier.init.shared::cta.b64 [bar], 1;\n" + copy + "\nret;",
                 ".param .u64 src, .param .u64 dst")
    monkeypatch.setattr(np, "arange", bounded_arange)
    with pytest.raises(warplens.ExecutionError, match="runs past"):
        sim.launch(mod, "k", grid=1, block=1, args=[src, src])


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("initialize", ["none", "partial", "ordinary", "tma"])
def test_tma_store_reports_never_written_source_bytes(sim, tensor, initialize):
    setup = "\n".join(line for line in SETUP.splitlines() if "st.shared" not in line)
    if initialize == "tma":
        setup += """
            mbarrier.init.shared::cta.b64 [bar], 1;
            mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
            cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [%s], 16, [bar];
            mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
        """
    elif initialize != "none":
        words = 1 if initialize == "partial" else 4
        setup += "\n" + "\n".join(f"st.shared.u32 [buf+{4 * i}], 7;" for i in range(words))
        setup += "\nfence.proxy.async.shared::cta;\n"
    copy = ("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%d, {0}], [buf];"
            if tensor else "cp.async.bulk.global.shared::cta.bulk_group [%d], [buf], 16;")
    mod = kernel(setup + "\n" + copy + "\ncp.async.bulk.commit_group;\n"
                 "cp.async.bulk.wait_group 0;\nret;", ".param .u64 src, .param .u64 dst")
    src = sim.from_numpy(np.arange(4, dtype=np.uint32))
    dst = sim.empty(4, "u32")
    target = sim.tensor_map_tiled(dst, 4) if tensor else dst
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src, target], async_delay=1000)
    if initialize in ("none", "partial"):
        (finding,) = stats.findings
        assert finding.kind == "smem-uninitialized-read" and finding.text == copy
        assert finding.line == next(ins.line for ins in mod.functions["k"].instrs if ins.text == copy)
    else:
        assert not stats.findings
        np.testing.assert_array_equal(dst.numpy(), np.arange(4) if initialize == "tma" else [7] * 4)
