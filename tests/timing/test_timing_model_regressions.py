"""Timing contracts independent of a particular GEMM's expected totals."""

from types import SimpleNamespace

import numpy as np
import pytest

import warplens
from warplens.timing.config import GpuConfig
from warplens.timing.model import Timing


def operation(op_kind, time=0, lane=0, **work):
    return SimpleNamespace(kind=op_kind, warp=SimpleNamespace(time=time, warp_id=0),
                           lane=lane, ins=SimpleNamespace(line=1), work=work)


@pytest.mark.parametrize("active_sms", [0, -1, 1.5, True, float("nan"), 149])
def test_active_sms_rejects_invalid_counts(active_sms):
    with pytest.raises(warplens.LaunchError, match="active_sms"):
        Timing(GpuConfig.load("B200"), active_sms)


def test_active_sms_accepts_numpy_integer():
    assert Timing(GpuConfig.load("B200"), np.int64(8)).active_sms == 8


def test_tf32_roofline_uses_tf32_compute_ceiling():
    cfg = GpuConfig.load("B200")
    timing = Timing(cfg)
    timing.timeline((0, 0, 0)).schedule(operation("mma", flops=8192, kind="tf32"))
    report = timing.report()
    assert report.roofline()["compute_ceiling_flops_per_s"] == pytest.approx(
        cfg["tensor_core_tflops_tf32"] * 1e12 / cfg["sm_count"])
    assert report.compute_source_keys == ("tensor_core_tflops_tf32",)


def test_mixed_mma_roofline_weights_time_at_each_dtype_peak():
    cfg = GpuConfig.load("B200")
    timing = Timing(cfg)
    line = timing.timeline((0, 0, 0))
    line.schedule(operation("mma", flops=8192, kind="f16"))
    line.schedule(operation("mma", time=1, flops=4096, kind="tf32"))
    expected = 12288 / (8192 / cfg["tensor_core_tflops_f16"] +
                        4096 / cfg["tensor_core_tflops_tf32"]) * 1e12 / cfg["sm_count"]
    assert timing.report().roofline()["compute_ceiling_flops_per_s"] == pytest.approx(expected)


def test_inflight_fetch_does_not_warm_l2_before_completion():
    cfg = GpuConfig.load("B200").replace(hbm_bandwidth_tb_s=0.001)
    timing = Timing(cfg)
    line = timing.timeline((0, 0, 0))
    first = line.schedule(operation("load", bytes=128, lines=np.array([7])))
    second = line.schedule(operation("load", time=1, bytes=128, lines=np.array([7])))
    assert second >= first
    assert line.hbm.work == 256 and line.l2.work == 0
    line.schedule(operation("load", time=first + 1, bytes=128, lines=np.array([7])))
    assert line.hbm.work == 256 and line.l2.work == 128


def test_commit_tracks_the_issuing_thread():
    line = Timing(GpuConfig.load("B200")).timeline((0, 0, 0))
    first = line.schedule(operation("mma", lane=0, flops=8192, kind="f16"))
    other = line.schedule(operation("mma", time=1, lane=1, flops=819200, kind="f16"))
    assert other > first
    assert line.schedule(operation("commit", time=2, lane=0)) == first
