"""Shared-memory bank model.

SMEM has 32 banks, each 4 bytes wide: the word at byte offset `o` lives in
bank `(o // 4) % 32`. One warp instruction is served in *wavefronts*; a
wavefront can touch each bank once. Lanes that touch the same word share a
wavefront, lanes that touch different words in the same bank cannot. So

    wavefronts = the largest number of distinct words in any one bank
    ideal      = ceil(distinct words / 32), what a conflict-free layout needs

and an access is an N-way conflict when wavefronts = N * ideal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BANKS = 32
BANK_BYTES = 4


def analyze(offsets: np.ndarray, size: int) -> tuple[int, int, np.ndarray]:
    """(wavefronts, ideal, distinct words per bank) for one warp access of
    `size` bytes at each of `offsets`."""
    first = offsets // BANK_BYTES
    span = int(((offsets % BANK_BYTES).max() + size - 1) // BANK_BYTES) + 1 if len(offsets) else 0
    if span <= 1:
        words = np.unique(first)
    else:
        last = (offsets + (size - 1)) // BANK_BYTES
        cover = first[:, None] + np.arange(span, dtype=first.dtype)
        words = np.unique(cover[cover <= last[:, None]])
    per_bank = np.bincount((words % BANKS).astype(np.intp), minlength=BANKS)
    return int(per_bank.max(initial=0)), -(-len(words) // BANKS), per_bank


@dataclass(slots=True)
class InstrBanks:
    """Bank statistics of one shared-memory instruction, over all its executions."""
    line: int
    text: str
    kernel: str
    op: str = ""  # "ld" or "st"
    executions: int = 0  # warp-level executions with at least one active lane
    wavefronts: int = 0
    ideal: int = 0
    worst: int = 0  # most wavefronts in a single execution
    worst_ideal: int = 1
    bank_words: np.ndarray = field(default_factory=lambda: np.zeros(BANKS, dtype=np.int64))

    @property
    def conflicts(self) -> int:
        """Wavefronts beyond what a conflict-free layout would need."""
        return self.wavefronts - self.ideal

    @property
    def worst_way(self) -> int:
        """N for the worst single execution being an N-way conflict (1 = none)."""
        return -(-self.worst // self.worst_ideal)


@dataclass(slots=True)
class SmemReport:
    instructions: dict[int, InstrBanks] = field(default_factory=dict)  # by PTX line

    def record(self, ins, kernel: str, offsets: np.ndarray, size: int) -> None:
        if not len(offsets):
            return
        entry = self.instructions.get(ins.line)
        if entry is None:
            entry = self.instructions[ins.line] = InstrBanks(ins.line, ins.text, kernel, ins.opcode)
        waves, ideal, per_bank = analyze(offsets, size)
        entry.executions += 1
        entry.wavefronts += waves
        entry.ideal += ideal
        entry.bank_words += per_bank
        if waves * entry.worst_ideal > entry.worst * ideal:
            entry.worst, entry.worst_ideal = waves, ideal

    @property
    def wavefronts(self) -> int:
        return sum(e.wavefronts for e in self.instructions.values())

    @property
    def conflicts(self) -> int:
        return sum(e.conflicts for e in self.instructions.values())

    def format(self) -> str:
        if not self.instructions:
            return "No shared-memory accesses."
        rows = [("line", "instruction", "execs", "wavefronts", "ideal", "conflicts", "worst")]
        for e in sorted(self.instructions.values(), key=lambda e: e.line):
            way = "none" if e.worst_way <= 1 else f"{e.worst_way}-way"
            rows.append((str(e.line), " ".join(e.text.split()), str(e.executions),
                         str(e.wavefronts), str(e.ideal), str(e.conflicts), way))
        widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        lines = ["Shared-memory bank conflicts (32 banks x 4 bytes)"]
        for r in rows:
            cells = [r[c].ljust(widths[c]) if c == 1 else r[c].rjust(widths[c]) for c in range(len(r))]
            lines.append("  " + "  ".join(cells).rstrip())
        return "\n".join(lines)
