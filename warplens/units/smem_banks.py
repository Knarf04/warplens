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
    # Per bank, the most distinct words a single execution put there: the wavefronts that
    # bank alone demanded. The sum above cannot show a conflict whose bank moves from one
    # execution to the next. Read by warplens.viz.smem.
    bank_peak: np.ndarray = field(default_factory=lambda: np.zeros(BANKS, dtype=np.int64))
    # Per bank, the most a single execution asked of it relative to the ideal of that
    # execution: words in the bank / wavefronts a conflict-free layout needs. 1 or less is
    # no conflict, whatever the width of the access; N is the bank's share of an N-way one.
    bank_ratio: np.ndarray = field(default_factory=lambda: np.zeros(BANKS, dtype=np.float64))

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
    instructions: dict[tuple[str, int], InstrBanks] = field(default_factory=dict)  # by kernel and PC
    warps: int = 0  # warps launched
    barrier_completions: int = 0  # completed `bar.sync` / `bar.arrive` phases, over all CTAs
    # What the layout views of warplens.viz draw, each with how often it happened. Nothing
    # in the simulator reads these.
    #   tiles: (kernel, line, "load" | "store", shared offset, rows, bytes per row,
    #           element size, swizzle bits) of each TMA tensor copy
    #   mmas:  (kernel, line, M, N, cta_group::2?, lane of the destination) of each tcgen05.mma
    tiles: dict[tuple, int] = field(default_factory=dict)
    mmas: dict[tuple, int] = field(default_factory=dict)

    def record(self, ins, kernel: str, offsets: np.ndarray, size: int, *, pc: int) -> int:
        """Returns the wavefronts this access took."""
        return self.record_groups(ins, kernel, [offsets], size, pc=pc)

    def record_groups(self, ins, kernel: str, groups: list[np.ndarray], size: int, *, pc: int) -> int:
        """One instruction with independently serviced groups, e.g. ldmatrix's matrices."""
        groups = [offsets for offsets in groups if len(offsets)]
        if not groups:
            return 0
        # PTX permits several instructions on one source line. Their costs
        # must remain separate, including distinct load/store counters.
        key = (kernel, pc)
        entry = self.instructions.get(key)
        if entry is None:
            entry = self.instructions[key] = InstrBanks(ins.line, ins.text, kernel, ins.opcode)
        waves = ideal = 0
        per_bank = np.zeros(BANKS, dtype=np.int64)
        for offsets in groups:
            group_waves, group_ideal, group_banks = analyze(offsets, size)
            waves += group_waves
            ideal += group_ideal
            per_bank += group_banks
        entry.executions += 1
        entry.wavefronts += waves
        entry.ideal += ideal
        entry.bank_words += per_bank
        np.maximum(entry.bank_peak, per_bank, out=entry.bank_peak)
        np.maximum(entry.bank_ratio, per_bank / ideal, out=entry.bank_ratio)
        if waves * entry.worst_ideal > entry.worst * ideal:
            entry.worst, entry.worst_ideal = waves, ideal
        return waves

    @property
    def wavefronts(self) -> int:
        return sum(e.wavefronts for e in self.instructions.values())

    @property
    def conflicts(self) -> int:
        return sum(e.conflicts for e in self.instructions.values())

    @property
    def ncu_wavefronts(self) -> int:
        """`wavefronts` as Nsight Compute counts it (l1tex__data_pipe_lsu_wavefronts_mem_shared).

        The hardware counter also includes two wavefronts per warp and one per
        CTA per completed barrier, which are not shared-memory accesses. This is
        measured on Ampere only, with `__syncthreads()` that every thread
        executes (validation/README.md, "The wavefront offset"); for named or
        partial barriers and for other architectures it is an extrapolation.
        """
        return self.wavefronts + 2 * self.warps + self.barrier_completions

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
        lines.append(f"  total wavefronts {self.wavefronts}; as ncu counts them {self.ncu_wavefronts} "
                     f"(+2 per warp, +1 per CTA per barrier; measured on Ampere)")
        return "\n".join(lines)
