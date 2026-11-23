"""The approximate timing model: configs, the time-ordered engine, reports."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.timing.config import CONFIG_DIR, GpuConfig
from warplens.timing.model import RESOURCE_SOURCES, Server
from warplens.viz.html_report import render_html
from warplens.viz.trace_perfetto import trace_events

from conftest import EXAMPLES, kernel

spec = importlib.util.spec_from_file_location("gemm_run", EXAMPLES / "04_pipelined_gemm" / "run.py")
gemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemm)


# -- configuration -------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["B200", "B300"])
def test_every_config_number_has_a_source(name):
    config = GpuConfig.load(name)
    assert config.name == name and len(config.params) >= 12
    for p in config.params.values():
        assert p.source and p.unit, p.key
        assert p.url.startswith("https://") or p.is_assumption or "not measured on B300" in p.source, p.key
    used = {key for keys in RESOURCE_SOURCES.values() for key in keys}
    assert used <= set(config.params)


def test_config_errors_and_overrides(tmp_path):
    with pytest.raises(warplens.LaunchError, match="no timing config"):
        GpuConfig.load("Z900")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: X\nclock_ghz:\n  value: 2\n")
    with pytest.raises(warplens.LaunchError, match="needs a `value` and a `source`"):
        GpuConfig.load(bad)
    config = GpuConfig.load("B200")
    faster = config.replace(clock_ghz=3)
    assert faster["clock_ghz"] == 3 and config["clock_ghz"] == 1.5
    assert "override" in faster.params["clock_ghz"].source and "1.5" in faster.params["clock_ghz"].note
    assert sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) == ["b200.yaml", "b300.yaml"]


def test_server_is_first_come_first_served():
    s = Server("x")
    assert s.reserve(10, 5, 1) == (10, 15)
    assert s.reserve(12, 5, 1) == (15, 20)  # queues behind the first job
    assert s.reserve(40, 5, 1) == (40, 45)  # idle gap
    assert (s.busy, s.work, s.first, s.last) == (15, 3, 10, 45)


# -- the engine ----------------------------------------------------------------------------

def test_timed_runs_compute_the_same_results(sim):
    data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", "transpose_linear",
                       grid=2, block=(32, 32), args=[sim.from_numpy(data), out], timing=True)
    np.testing.assert_array_equal(out.numpy(), data.transpose(0, 2, 1))
    assert not stats.findings and stats.timing.ctas == 2 and stats.timing.time_ns > 0
    plain = sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", "transpose_linear",
                       grid=2, block=(32, 32), args=[sim.from_numpy(data), sim.empty(data.shape, "f32")])
    assert plain.timing is None and plain.instructions == stats.instructions


def test_bank_conflicts_cost_time(sim):
    def time_of(layout):
        data = np.zeros((1, 32, 32), dtype=np.float32)
        stats = sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", f"transpose_{layout}",
                           grid=1, block=(32, 32), args=[sim.from_numpy(data), sim.empty(data.shape, "f32")],
                           timing=True)
        return stats.timing
    linear, swizzled = time_of("linear"), time_of("swizzled")
    # 32 warps each pay 31 extra wavefronts on the conflicting read.
    assert linear.time_ns > swizzled.time_ns
    assert linear.utilization["SMEM"] > 3 * swizzled.utilization["SMEM"]


def test_a_waiting_warp_resumes_when_the_copy_completes(sim):
    data = np.arange(64, dtype=np.uint32)
    mod = kernel("""
        .reg .b64 %s; .reg .b32 %buf, %bar, %v; .reg .pred %p;
        .shared .align 1024 .b8 buf[64];
        .shared .align 8 .u64 bar;
        ld.param.u64 %s, [src];
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 64;
        cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%s, {0}], [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        ld.shared.u32 %v, [%buf];
        ret;
    """, ".param .u64 src")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[sim.tensor_map_tiled(sim.from_numpy(data), 16)],
                       timing=True)
    t = stats.timing
    (copy,) = [s for s in t.slices if s.track == "TMA engine"]
    (wait,) = [s for s in t.slices if s.name.startswith("wait")]
    latency = t.config["tma_latency_ns"]
    assert copy.end - copy.start >= latency and copy.args["bytes"] == 64
    assert wait.end == pytest.approx(copy.end) and wait.start < copy.end
    assert t.time_ns > copy.end and not stats.findings
    assert [name for name, _, _ in t.instants] == ["mbarrier phase completes"]


def test_deadlock_is_still_reported_in_a_timed_run(sim):
    mod = kernel(".reg .b32 %bar; .reg .pred %p;\n.shared .align 8 .u64 bar;\nmov.b32 %bar, bar;\n"
                 "mbarrier.init.shared::cta.b64 [%bar], 2;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;\nret;")
    with pytest.raises(warplens.DeadlockError, match="still needs 2 arrival"):
        sim.launch(mod, "k", grid=1, block=1, timing=True)


# -- M4 acceptance: the pipelined GEMM over 1, 2 and 4 stages -----------------------------------

@pytest.fixture(scope="module")
def sweeps():
    return {title: [gemm.run(stages, **timing).timing for stages in (1, 2, 4)]
            for title, timing in gemm.SCENARIOS}


def test_more_stages_never_make_it_slower(sweeps):
    for reports in sweeps.values():
        times = [r.time_ns for r in reports]
        assert times[0] > times[1] * 1.1  # one stage serializes loading and multiplying
        assert times[1] >= times[2] * 0.999
        for r in reports:
            assert all(0 <= u <= 1.001 for u in r.utilization.values())


def test_what_limits_the_pipeline_in_each_scenario(sweeps):
    whole, few, fast_tma = (sweeps[title] for title, _ in gemm.SCENARIOS)
    assert [r.bottleneck for r in whole] == ["latency", "HBM", "HBM"]
    assert [r.bottleneck for r in few] == ["latency", "latency", "TMA engine"]
    assert [r.bottleneck for r in fast_tma] == ["latency", "latency", "Tensor Core"]
    # The HBM-bound time is the bytes moved over this SM's share of the bandwidth.
    r = whole[2]
    share = r.config["hbm_bandwidth_tb_s"] * 1e3 / r.config["sm_count"]  # bytes per ns
    assert r.pipeline_ns == pytest.approx(r.bytes_loaded / share, rel=0.03)
    # The Tensor-Core-bound rate approaches this SM's share of the measured throughput.
    roof = fast_tma[2].roofline()
    assert 0.8 < roof["achieved_flops_per_s"] / roof["compute_ceiling_flops_per_s"] <= 1.0


def test_summary_marks_assumptions_and_names_the_verdict(sweeps):
    text = sweeps[gemm.SCENARIOS[0][0]][2].summary()
    assert "HBM-bound" in text and "model, not a measurement" in text
    assert "clock_ghz*" in text and "hbm_bandwidth_tb_s," in text and "* assumed value" in text


# -- outputs -------------------------------------------------------------------------------

def test_perfetto_trace(sweeps, tmp_path):
    report = sweeps[gemm.SCENARIOS[1][0]][2]
    events = trace_events(report)
    tracks = {e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "thread_name"}
    assert {"warp 0", "warp 1", "warp 2", "warp 3", "TMA engine", "Tensor Core"} <= tracks
    slices = [e for e in events if e["ph"] == "X"]
    assert sum(e["name"].startswith("MMA 128x64x16") for e in slices) == 64  # 16 tiles x 4
    assert sum(e["name"] == "TMA load" for e in slices) == 32  # A and B per tile
    assert all(e["dur"] > 0 and e["ts"] >= 0 for e in slices)
    assert sum(e["ph"] == "i" for e in events) == 33  # 16 full + 16 empty + done
    counter = [e["args"]["bytes"] for e in events if e["ph"] == "C"]
    assert max(counter) >= 24576 and counter[-1] == 0 and min(counter) >= 0
    path = tmp_path / "t.json"
    report.write_trace(path)
    assert json.loads(path.read_text())["displayTimeUnit"] == "ns"


def test_html_report_links_every_number_to_a_sourced_config_entry(sweeps, tmp_path):
    report = sweeps[gemm.SCENARIOS[0][0]][2]
    html = render_html(report, "GEMM")
    anchors = set(re.findall(r'id="cfg-([a-z0-9_]+)"', html))
    links = re.findall(r'href="#cfg-([a-z0-9_]+)"', html)
    assert anchors == set(report.config.params) and links and set(links) <= anchors
    # Every row of the resource and number tables carries at least one such link.
    rows = [r for r in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S) if 'class="num"' in r]
    assert len(rows) >= 12 and all('href="#cfg-' in r for r in rows)
    assert "arXiv:2512.02189" in html and "https://arxiv.org/abs/2512.02189" in html
    assert 'class="assumed"' in html and "not a measurement" in html and "<svg" in html
    report.write_html(tmp_path / "r.html")
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")


def test_timing_is_deterministic():
    a, b = (gemm.run(2, timing=True).timing for _ in range(2))
    assert a.time_ns == b.time_ns and a.utilization == b.utilization
