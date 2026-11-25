"""Timing inputs and exported reports must preserve the model's meaning."""

import importlib.util
import re
from types import SimpleNamespace

import pytest
import yaml

from warplens.errors import LaunchError
from warplens.timing.config import GpuConfig, Param
from warplens.timing.model import RESOURCE_SOURCES, Slice
from warplens.viz.html_report import render_html
from warplens.viz.trace_perfetto import trace_events

from tests.support import EXAMPLES


POSITIVE_PARAMS = (
    "sm_count", "clock_ghz", "hbm_bandwidth_tb_s", "l2_bandwidth_per_sm_gb_s",
    "tma_bytes_per_clk", "smem_wavefront_clk", "tensor_core_tflops_f16", "tensor_core_tflops_tf32",
)
NONNEGATIVE_PARAMS = (
    "tma_latency_ns", "global_access_latency_clk", "tmem_access_clk", "mma_latency_clk",
    "clc_latency_ns",
)


@pytest.mark.parametrize("key", sorted(POSITIVE_PARAMS))
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_rates_fail_before_scheduling(key, value):
    with pytest.raises(LaunchError, match=key):
        GpuConfig.load("B200").replace(**{key: value})


@pytest.mark.parametrize("key", sorted(NONNEGATIVE_PARAMS))
def test_latencies_allow_zero_but_not_negative_or_nonfinite(key):
    cfg = GpuConfig.load("B200")
    assert cfg.replace(**{key: 0})[key] == 0
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(LaunchError, match=key):
            cfg.replace(**{key: value})


def test_nonintegral_sm_count_and_misspelled_override_are_rejected():
    cfg = GpuConfig.load("B200")
    with pytest.raises(LaunchError, match="sm_count.*integer"):
        cfg.replace(sm_count=1.5)
    with pytest.raises(LaunchError, match="tma_bytes_per_clock"):
        cfg.replace(tma_bytes_per_clock=512)


def test_direct_constructor_normalizes_validated_values_and_copies_parameters():
    params = dict(GpuConfig.load("B200").params)
    params["clock_ghz"] = Param("clock_ghz", "1.5", source="assumption")
    config = GpuConfig("Custom", params)
    assert 1.0 / config["clock_ghz"] == pytest.approx(2 / 3)
    params.clear()
    assert config["clock_ghz"] == 1.5


@pytest.mark.parametrize("contents", ["", "[]", "scalar", "clock_ghz: ["])
def test_malformed_config_is_a_launch_error(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(LaunchError):
        GpuConfig.load(path)


def test_missing_and_invalid_file_parameters_are_rejected(tmp_path):
    cfg = GpuConfig.load("B200")
    raw = {key: {"value": p.value, "source": p.source} for key, p in cfg.params.items()}
    path = tmp_path / "custom.yaml"
    raw.pop("clock_ghz")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(LaunchError, match="missing entries: clock_ghz"):
        GpuConfig.load(path)
    raw["clock_ghz"] = {"value": "not a number", "source": "assumption"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(LaunchError, match="clock_ghz.*finite number"):
        GpuConfig.load(path)


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


@pytest.mark.parametrize("stages", [0, -1, 5, 1.5, True])
def test_example_rejects_stage_counts_outside_its_kernel_storage(stages):
    path = EXAMPLES / "04_pipelined_gemm/run.py"
    spec = importlib.util.spec_from_file_location("gemm_run_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="stages must be an integer from 1 to 4"):
        module.run(stages)
