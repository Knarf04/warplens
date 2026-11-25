"""The exported timing report (HTML page and Perfetto trace) must preserve the model's meaning."""

import re
from types import SimpleNamespace

from warplens.timing.config import GpuConfig
from warplens.timing.model import RESOURCE_SOURCES, Slice
from warplens.viz.html_report import render_html
from warplens.viz.trace_perfetto import trace_events


def _report():
    return SimpleNamespace(
        config=GpuConfig.load("B200"), active_sms=8, ctas=1, time_ns=20,
        pipeline_ns=10, clocks=30, slices=[], instants=[], verdict="latency-bound",
        utilization={key: 0.1 for key in RESOURCE_SOURCES},
        compute_source_keys=("tensor_core_tflops_tf32",),
        roofline=lambda: {"achieved_flops_per_s": 1e12, "compute_ceiling_flops_per_s": 2e12,
                          "memory_ceiling_flops_per_s": 3e12, "flops_per_byte": 4},
    )


def _number_row(html, label):
    return re.search(rf"<tr><td>{re.escape(label)}</td>.*?</tr>", html).group()


def test_html_distinguishes_pipeline_rate_and_launch_sharing_from_config():
    html = render_html(_report())
    assert "Achieved rate per SM" not in html
    assert "Pipeline rate per SM" in html and "excluding kernel setup and epilogue" in html
    assert 'id="launch-active-sms">8</span>' in html
    row = _number_row(html, "Memory ceiling per SM")
    assert '#launch-active-sms' in row and '#cfg-sm_count' not in row
    row = _number_row(html, "Compute ceiling per SM")
    assert '#cfg-tensor_core_tflops_tf32' in row and '#cfg-tensor_core_tflops_f16' not in row
    for label in ("Kernel time", "Kernel time in clocks", "Pipeline rate per SM"):
        row = _number_row(html, label)
        assert '#cfg-tmem_access_clk' in row and '#cfg-smem_wavefront_clk' in row
        assert '#cfg-tensor_core_tflops_tf32' in row and '#launch-active-sms' in row


def test_b300_borrowed_measurements_are_visibly_assumptions():
    report = _report()
    report.config = GpuConfig.load("B300")
    html = render_html(report)
    assert 'id="cfg-tensor_core_tflops_f16" class="assumed"' in html
    assert 'id="cfg-hbm_bandwidth_tb_s" class="assumed"' in html


def test_tma_counter_includes_queued_copies_and_coalesces_simultaneous_changes():
    report = _report()
    report.slices = [
        Slice("TMA engine", "TMA load", 0, 10, args={"bytes": 16, "issued_ns": 0}),
        Slice("TMA engine", "TMA load", 5, 20, args={"bytes": 32, "issued_ns": 1}),
        Slice("TMA engine", "TMA load", 20, 25, args={"bytes": 64, "issued_ns": 20}),
    ]
    counters = [event for event in trace_events(report) if event["ph"] == "C"]
    assert [(e["ts"], e["args"]["bytes"]) for e in counters] == [
        (0, 16), (.001, 48), (.010, 32), (.020, 64), (.025, 0),
    ]


def test_overlapping_async_lifetimes_have_separate_display_lanes():
    report = _report()
    report.slices = [
        Slice("Tensor Core", "MMA 1", 0, 10),
        Slice("Tensor Core", "MMA 2", 5, 15),
        Slice("Tensor Core", "MMA 3", 10, 20),
    ]
    events = trace_events(report)
    intervals = [e for e in events if e["ph"] == "X"]
    assert [(e["name"], e["ts"], e["dur"]) for e in intervals] == [
        ("MMA 1", 0, .010), ("MMA 2", .005, .010), ("MMA 3", .010, .010),
    ]
    assert intervals[0]["tid"] == intervals[2]["tid"] != intervals[1]["tid"]
    html = render_html(report)
    assert "Tensor Core (overlap 2)" in html
    assert "not additional hardware engines" in html
