"""Timing inputs are checked before anything is scheduled: rates, latencies, overrides and the
configuration files they are read from."""

import pytest
import yaml

from warplens.errors import LaunchError
from warplens.timing.config import GpuConfig, Param

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
