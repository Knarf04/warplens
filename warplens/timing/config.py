"""GPU timing configurations: every number carries its unit and its source."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from ..errors import LaunchError

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Rates divide work into durations; latencies may be zero when exploring an
# idealized pipeline. Reject bad inputs before they create backwards/NaN time.
POSITIVE_PARAMS = frozenset({
    "sm_count", "clock_ghz", "hbm_bandwidth_tb_s", "l2_bandwidth_per_sm_gb_s",
    "tma_bytes_per_clk", "smem_wavefront_clk", "tensor_core_tflops_f16",
    "tensor_core_tflops_tf32",
})
NONNEGATIVE_PARAMS = frozenset({
    "tma_latency_ns", "global_access_latency_clk", "tmem_access_clk", "mma_latency_clk",
    "clc_latency_ns",
})
TIMING_PARAMS = POSITIVE_PARAMS | NONNEGATIVE_PARAMS
# Public facts about the device that the model does not use. Optional in a config.
INFORMATIONAL_PARAMS = frozenset({"hbm_capacity_gb", "max_power_w"})
# How a config marks a value it borrows from B200 because none is public for its GPU.
ASSUMED_EQUAL_TO_B200 = "assumption: assumed equal to B200, uncalibrated"
OVERRIDE = "override passed by the caller"


def _number(key, value) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise LaunchError(f"timing config entry `{key}` must be a finite number") from None
    if not math.isfinite(result):
        raise LaunchError(f"timing config entry `{key}` must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class Param:
    key: str
    value: float
    unit: str = ""
    source: str = ""
    url: str = ""
    note: str = ""

    @property
    def is_assumption(self) -> bool:
        return self.source.strip().lower().startswith("assumption")

    @property
    def is_fitted(self) -> bool:
        return self.source.strip().lower().startswith("measured:")


class GpuConfig:
    def __init__(self, name: str, params: dict[str, Param]):
        self.name = name
        missing = TIMING_PARAMS - params.keys()
        if missing:
            raise LaunchError(f"timing config `{name}` is missing entries: {', '.join(sorted(missing))}")
        checked = {}
        for key, param in params.items():
            value = _number(key, param.value)
            if key in POSITIVE_PARAMS | INFORMATIONAL_PARAMS and value <= 0:
                raise LaunchError(f"timing config entry `{key}` must be positive")
            if key in NONNEGATIVE_PARAMS and value < 0:
                raise LaunchError(f"timing config entry `{key}` must be nonnegative")
            if key == "sm_count" and not value.is_integer():
                raise LaunchError("timing config entry `sm_count` must be an integer")
            checked[key] = replace(param, value=value)
        self.params = checked

    @property
    def fitted(self) -> list[str]:
        """The timing entries fitted to measurements of a GPU (`validation/blackwell_timing/fit.py`
        writes their source as "measured: ...")."""
        return sorted(k for k in TIMING_PARAMS if self.params[k].is_fitted)

    @property
    def basis(self) -> str:
        """Where the timing entries come from, in a few words, for the label of every report:
        the default configs are illustrative, a config written by the fit is experimental. In
        both cases the result is simulated and approximate."""
        total = len(TIMING_PARAMS)
        assumed = sum(self.params[k].is_assumption for k in TIMING_PARAMS)
        overridden = sum(self.params[k].source == OVERRIDE for k in TIMING_PARAMS)
        if self.fitted:
            text = (f"experimental fit, {len(self.fitted)} of {total} timing parameters fitted to "
                    f"microbenchmarks of one GPU, {assumed} assumed")
        else:
            text = (f"illustrative parameters, {assumed} of {total} assumed and none fitted to "
                    "hardware")
        return text + (f", {overridden} overridden by the caller" if overridden else "")

    @property
    def caveat(self) -> str:
        """One sentence on how far a result under this config can be trusted."""
        if self.fitted:
            return ("The fit moves the model toward the microbenchmarks of one GPU; it is the same "
                    "simplified model, the fit is experimental, and the result is not a measurement.")
        return ("The parameters are plausible values, not calibrated against hardware: an "
                "illustration of where time goes, not a prediction.")

    def __getitem__(self, key: str) -> float:
        try:
            return self.params[key].value
        except KeyError:
            raise LaunchError(f"timing config `{self.name}` has no entry `{key}`") from None

    def replace(self, **values) -> "GpuConfig":
        """A copy with some values overridden; the overrides are marked as such."""
        params = dict(self.params)
        for key, value in values.items():
            if key not in params:
                raise LaunchError(f"timing config `{self.name}` has no entry `{key}`")
            old = self.params[key]
            params[key] = Param(key, _number(key, value), old.unit, OVERRIDE,
                                note=f"config file value: {old.value:g}")
        return GpuConfig(self.name, params)

    @classmethod
    def load(cls, name_or_path) -> "GpuConfig":
        import yaml

        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIG_DIR / f"{str(name_or_path).lower()}.yaml"
        if not path.exists():
            raise LaunchError(f"no timing config at {path}")
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise LaunchError(f"cannot read timing config {path}: {exc}") from None
        if not isinstance(raw, dict):
            raise LaunchError(f"{path.name}: timing config must be a mapping")
        params = {}
        for key, entry in raw.items():
            if key == "name":
                continue
            if not isinstance(entry, dict) or "value" not in entry or not entry.get("source"):
                raise LaunchError(f"{path.name}: entry `{key}` needs a `value` and a `source`")
            params[key] = Param(key, _number(key, entry["value"]), str(entry.get("unit", "")),
                                str(entry["source"]), str(entry.get("url", "")),
                                str(entry.get("note", "")))
        return cls(str(raw.get("name", path.stem)), params)
