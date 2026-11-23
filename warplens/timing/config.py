"""GPU timing configurations: every number carries its unit and its source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import LaunchError

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


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


class GpuConfig:
    def __init__(self, name: str, params: dict[str, Param]):
        self.name = name
        self.params = params

    def __getitem__(self, key: str) -> float:
        try:
            return self.params[key].value
        except KeyError:
            raise LaunchError(f"timing config `{self.name}` has no entry `{key}`") from None

    def replace(self, **values) -> "GpuConfig":
        """A copy with some values overridden; the overrides are marked as such."""
        params = dict(self.params)
        for key, value in values.items():
            old = self.params.get(key, Param(key, 0.0))
            params[key] = Param(key, float(value), old.unit, "override passed by the caller",
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
        raw = yaml.safe_load(path.read_text())
        params = {}
        for key, entry in raw.items():
            if key == "name":
                continue
            if not isinstance(entry, dict) or "value" not in entry or not entry.get("source"):
                raise LaunchError(f"{path.name}: entry `{key}` needs a `value` and a `source`")
            params[key] = Param(key, float(entry["value"]), str(entry.get("unit", "")),
                                str(entry["source"]), str(entry.get("url", "")),
                                str(entry.get("note", "")))
        return cls(str(raw.get("name", path.stem)), params)
