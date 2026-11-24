"""Structure of the timing model that calibration cannot fix: which memory level a TMA
store or a cp.async copy is charged to, B300 as a configuration of its own, and which
CTAs the reported utilization is averaged over."""

import numpy as np
import pytest

import warplens
from warplens.timing.config import ASSUMED_EQUAL_TO_B200, GpuConfig
from warplens.timing.model import RESOURCE_SOURCES

from conftest import kernel

LOAD = """
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 256;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [{src}], 256, [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], {phase};
"""
STORE = """
        cp.async.bulk.global.shared::cta.bulk_group [{dst}], [%buf], 256;
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
"""


def copies_kernel(body: str):
    """256 bytes move between global arrays `src`/`dst` and one shared buffer."""
    return kernel(f"""
        .reg .b64 %s, %d; .reg .b32 %buf, %bar; .reg .pred %p;
        .shared .align 16 .b8 buf[256];
        .shared .align 8 .u64 bar;
        ld.param.u64 %s, [src]; ld.param.u64 %d, [dst];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        {body}
        ret;
    """, ".param .u64 src, .param .u64 dst")


def tma(stats, name):
    return [s for s in stats.timing.slices if s.track == "TMA engine" and s.name == name]


# -- 1. TMA stores and cp.async use the L2 record ---------------------------------------------

def test_tma_store_is_charged_to_hbm_only_for_lines_l2_does_not_hold(sim):
    src, dst = sim.from_numpy(np.arange(64, dtype=np.uint32)), sim.empty(64, "u32")
    mod = copies_kernel(LOAD.format(src="%s", phase=0) + STORE.format(dst="%d"))
    stats = sim.launch(mod, "k", grid=2, block=1, args=[src, dst], timing=True)
    assert not stats.findings and np.array_equal(dst.numpy(), src.numpy())
    cold, warm = tma(stats, "TMA store")
    # The first CTA writes lines nobody touched; the second finds them in L2.
    assert (cold.args["to_hbm"], warm.args["to_hbm"]) == (256, 0)
    assert warm.end - warm.args["issued_ns"] < cold.end - cold.args["issued_ns"]


def test_tma_store_to_lines_this_cta_loaded_goes_to_l2(sim):
    src = sim.from_numpy(np.arange(64, dtype=np.uint32))
    mod = copies_kernel(LOAD.format(src="%s", phase=0) + STORE.format(dst="%s"))
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src, src], timing=True)
    assert not stats.findings
    store, = tma(stats, "TMA store")
    assert store.args["to_hbm"] == 0
    assert stats.timing.utilization["L2"] > 0


def test_load_of_lines_a_store_wrote_finds_them_in_l2(sim):
    src, dst = sim.from_numpy(np.arange(64, dtype=np.uint32)), sim.empty(64, "u32")
    mod = copies_kernel(LOAD.format(src="%s", phase=0) + STORE.format(dst="%d")
                        + "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 256;"
                        + LOAD.format(src="%d", phase=1).split("\n", 3)[3])
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src, dst], timing=True)
    assert not stats.findings
    first, again = tma(stats, "TMA load")
    assert (first.args["from_hbm"], again.args["from_hbm"]) == (256, 0)


def test_tensor_store_charges_only_the_elements_inside_the_tensor(sim):
    src, dst = sim.from_numpy(np.arange(64, dtype=np.uint32)), sim.empty(16, "u32")
    tensor_map = sim.tensor_map_tiled(dst, 16)
    mod = copies_kernel(LOAD.format(src="%s", phase=0) + """
        cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%d, {8}], [%buf];
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
    """)
    stats = sim.launch(mod, "k", grid=1, block=1, args=[src, tensor_map], timing=True)
    store, = tma(stats, "TMA store")
    assert store.args["bytes"] == 64 and store.args["to_hbm"] == 32  # half the box is outside


def test_cp_async_tells_a_warm_source_from_a_cold_one(sim):
    mod = kernel("""
        .reg .u64 %src, %offset;
        .reg .u32 %lane, %dst;
        .shared .align 16 .b8 buffer[128];
        ld.param.u64 %src, [src];
        mov.u32 %lane, %tid.x;
        mul.wide.u32 %offset, %lane, 4;
        add.u64 %src, %src, %offset;
        mov.u32 %dst, buffer;
        mad.lo.u32 %dst, %lane, 4, %dst;
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.wait_group 0;
        ret;
    """, ".param .u64 src")
    src = sim.from_numpy(np.arange(32, dtype=np.uint32))
    stats = sim.launch(mod, "k", grid=2, block=32, args=[src], timing=True)
    assert not stats.findings
    cold, warm = tma(stats, "cp.async copy")
    assert (cold.args["from_hbm"], warm.args["from_hbm"]) == (128, 0)
    assert warm.end - warm.args["issued_ns"] < cold.end - cold.args["issued_ns"]


# -- 3. utilization is averaged over the CTAs that did work ------------------------------------

def one_worker_kernel():
    """A persistent-style kernel: CTA 0 loads 256 bytes, every other CTA returns at once."""
    return kernel("""
        .reg .b64 %s; .reg .b32 %buf, %bar, %cta; .reg .pred %p, %idle;
        .shared .align 16 .b8 buf[256];
        .shared .align 8 .u64 bar;
        mov.u32 %cta, %ctaid.x;
        setp.ne.u32 %idle, %cta, 0;
        @%idle bra done;
        ld.param.u64 %s, [src];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
    """ + LOAD.format(src="%s", phase=0) + """
    done:
        ret;
    """, ".param .u64 src")


def test_idle_ctas_do_not_dilute_utilization(sim):
    mod = one_worker_kernel()
    launch = lambda grid: sim.launch(mod, "k", grid=grid, block=1, timing=True, args=[
        sim.from_numpy(np.arange(64, dtype=np.uint32))]).timing  # a fresh, cold source each time
    alone, among_idle = launch(1), launch(8)
    assert (alone.ctas, alone.working_ctas) == (1, 1)
    assert (among_idle.ctas, among_idle.working_ctas) == (8, 1)
    assert alone.utilization["HBM"] > 0 and alone.utilization["TMA engine"] > 0
    for name in ("SMEM", "TMA engine", "HBM", "L2", "Tensor Core"):
        assert among_idle.utilization[name] == pytest.approx(alone.utilization[name]), name
        # The figure over every CTA stays available, and is the diluted one.
        assert among_idle.utilization_all_ctas[name] == pytest.approx(alone.utilization[name] / 8)
    assert among_idle.pipeline_ns == pytest.approx(alone.pipeline_ns)
    assert among_idle.verdict == alone.verdict
    assert "averaged over the 1 CTA(s) that did work; 7 did none" in among_idle.summary()
    assert "did work" not in alone.summary()


def test_a_launch_in_which_no_cta_uses_a_modeled_resource_averages_over_all(sim):
    mod = kernel(".reg .b32 %r; mov.u32 %r, %tid.x; add.u32 %r, %r, 1; ret;")
    report = sim.launch(mod, "k", grid=4, block=32, timing=True).timing
    assert (report.ctas, report.working_ctas) == (4, 4)
    assert report.utilization == report.utilization_all_ctas
    assert report.utilization["issue slots"] > 0 and "did work" not in report.summary()


def test_roofline_rate_does_not_depend_on_idle_ctas(sim):
    mod = one_worker_kernel()
    reports = [sim.launch(mod, "k", grid=grid, block=1, timing=True, args=[
        sim.from_numpy(np.arange(64, dtype=np.uint32))]).timing for grid in (1, 8)]
    assert reports[0].roofline() == pytest.approx(reports[1].roofline())


# -- 2. B300 is a timing configuration of its own ----------------------------------------------

PUBLIC_B300_DIFFERENCES = {"sm_count": (148, 160), "hbm_capacity_gb": (192, 288),
                           "max_power_w": (1200, 1400)}


def test_b300_entries_are_public_differences_or_marked_as_assumed_equal():
    b200, b300 = GpuConfig.load("B200"), GpuConfig.load("B300")
    assert set(b200.params) == set(b300.params)
    for key, theirs in b300.params.items():
        ours = b200.params[key]
        if key in PUBLIC_B300_DIFFERENCES:
            # A figure NVIDIA publishes for this GPU: cited, linked, and not starred.
            assert (ours.value, theirs.value) == PUBLIC_B300_DIFFERENCES[key], key
            assert theirs.url.startswith("https://") and not theirs.is_assumption, key
            assert ours.url.startswith("https://") and not ours.is_assumption, key
        else:
            # No public figure: the B200 value, and it says so in the agreed words.
            assert theirs.value == ours.value, key
            assert theirs.source.startswith(ASSUMED_EQUAL_TO_B200), key
            assert theirs.is_assumption, key
    # At least one public difference reaches the model.
    assert "sm_count" in {key for keys in RESOURCE_SOURCES.values() for key in keys}


def test_b300_timing_is_not_b200_timing():
    mod = copies_kernel(LOAD.format(src="%s", phase=0))
    reports, hbm_ns = {}, {}
    for model in ("B200", "B300"):
        device = warplens.Device(model)
        src = device.from_numpy(np.arange(64, dtype=np.uint32))
        report = device.launch(mod, "k", grid=1, block=1, args=[src, src], timing=True).timing
        load, = [s for s in report.slices if s.name == "TMA load"]
        assert load.args["from_hbm"] == 256
        reports[model] = report
        hbm_ns[model] = load.end - load.args["issued_ns"] - report.config["tma_latency_ns"]
    b200, b300 = reports["B200"], reports["B300"]
    assert (b200.config.name, b300.config.name) == ("B200", "B300")
    assert (b200.active_sms, b300.active_sms) == (148, 160)
    # A cold load is limited by one SM's share of HBM, which is smaller with more SMs.
    assert hbm_ns["B300"] == pytest.approx(hbm_ns["B200"] * 160 / 148)
    assert b300.time_ns > b200.time_ns
    # The whole-GPU Tensor Core rate is assumed equal, so one SM's share shrinks too.
    assert b300.roofline()["compute_ceiling_flops_per_s"] == \
        pytest.approx(b200.roofline()["compute_ceiling_flops_per_s"] * 148 / 160)
