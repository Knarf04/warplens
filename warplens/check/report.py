"""Findings: things a kernel did that are wrong or suspicious but did not stop it.

Each finding points at the PTX line responsible. Repeats of the same finding
at the same instruction are counted, not listed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Finding:
    kind: str  # short stable identifier, e.g. "smem-uninitialized-read"
    message: str
    line: int
    text: str
    kernel: str
    cta: tuple[int, int, int]  # first occurrence
    warp: int
    lanes: list[int]
    count: int = 1  # how many times this instruction triggered it

    def __str__(self) -> str:
        times = f" ({self.count} times)" if self.count > 1 else ""
        return (f"[{self.kind}] {self.message}{times}\n"
                f"  line {self.line}: {' '.join(self.text.split())}\n"
                f"  first in CTA {self.cta}, warp {self.warp}, lanes {_ranges(self.lanes)}")


def _ranges(lanes: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2, 5'"""
    out, i = [], 0
    while i < len(lanes):
        j = i
        while j + 1 < len(lanes) and lanes[j + 1] == lanes[j] + 1:
            j += 1
        out.append(str(lanes[i]) if i == j else f"{lanes[i]}-{lanes[j]}")
        i = j + 1
    return ", ".join(out) or "none"


@dataclass(slots=True)
class Findings:
    items: list[Finding] = field(default_factory=list)
    _index: dict = field(default_factory=dict)

    def add(self, kind: str, message: str, warp, ins, lanes) -> None:
        # Source lines need not uniquely identify instructions: PTX accepts
        # multiple statements, even identical ones, on the same line.
        key = (kind, warp.program.fn.name, warp.stack.top.pc)
        if key in self._index:
            self._index[key].count += 1
            return
        f = Finding(kind, message, ins.line, ins.text, warp.program.fn.name,
                    warp.cta.id, warp.warp_id, [int(x) for x in lanes])
        self._index[key] = f
        self.items.append(f)
