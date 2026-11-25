"""Helpers for `book_tour.ipynb`: run a program in the simulator from a notebook, and keep
what the simulator knows about each kernel launch.

`warplens run program` starts the program with its CUDA runtime replaced and executes the
runtime calls it makes in the simulator. It prints a summary and exits, so the launch
statistics, findings and timing reports are gone when it returns. `warplens.run_program` does
the same job from inside the notebook process and returns those objects, and is what
`run_book_kernel()` here uses. The program still runs in a child process: the notebook's own
process never loads TVM or a CUDA runtime, and a crash in a kernel's host code cannot take
the notebook down.

Only warplens's public API is used: `warplens.run_program` and what it returns
(`ProgramRun`, `ProgramLaunch`, `LaunchStats`).

Nothing in this file comes from the book. `run_book_kernel()` executes the kernel definitions
found in the reader's clone, through `run.py` next to this file.
"""

from __future__ import annotations

import collections
import sys
import tempfile
from pathlib import Path

from warplens import ProgramLaunch as Launch, ProgramRun as Run, run_program

RUNNER = Path(__file__).resolve().parent / "run.py"  # the example this notebook sits beside

# The child process: take the kernel builders from the clone with the runner's own function,
# and run one of them at a size the notebook chooses.
CHILD = """\
import runpy, sys
from pathlib import Path
import tvm_tirx
runner = runpy.run_path(sys.argv[1])  # examples/08_book/run.py, without running its main()
book, version = Path(sys.argv[2]), sys.argv[3]
size = [int(v) for v in sys.argv[4:7]] or runner["SIZES"].get(version, (128, 128, 64))
runner["run"](version, runner["kernels"](book)[version], tvm_tirx.cuda(0), *size)
"""


def run_book_kernel(book, version: str, size=None, **options) -> Run:
    """Compile `hgemm_<version>` from the clone at `book` and run it in the simulator, on an
    M x N x K problem (`size`; the runner's default when omitted). Raises if the kernel fails
    or its result does not match NumPy, which the runner checks."""
    with tempfile.TemporaryDirectory(prefix="warplens_notebook_") as scratch:
        script = Path(scratch, "child.py")
        script.write_text(CHILD)
        done = run_program([sys.executable, script, RUNNER, book, version, *(size or ())], **options)
    if done.returncode or "matches NumPy" not in done.stdout:
        raise RuntimeError(f"hgemm_{version} did not run:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
    return done


# -- reading a launch ---------------------------------------------------------------------

# Instruction families worth watching, by the start of the PTX instruction name.
FAMILIES = [
    ("TMA load", ("cp.async.bulk.tensor",)),  # refined by direction in family()
    ("TMA store", ()),
    ("mbarrier arrive", ("mbarrier.arrive", "mbarrier.expect_tx")),
    ("mbarrier wait", ("mbarrier.try_wait", "mbarrier.test_wait")),
    ("MMA", ("tcgen05.mma",)),
    ("MMA commit", ("tcgen05.commit",)),
    ("TMEM load", ("tcgen05.ld",)),
    ("TMEM alloc", ("tcgen05.alloc",)),
    ("TMEM dealloc", ("tcgen05.dealloc",)),
    ("SMEM store", ("st.shared",)),
    ("SMEM load", ("ld.shared",)),
    ("barrier", ("bar.sync", "bar.arrive", "barrier.")),
    ("store wait", ("cp.async.bulk.wait_group",)),
    ("global load", ("ld.global",)),
    ("global store", ("st.global",)),
]


def family(ins) -> str | None:
    """The family of a PTX instruction, or None for ordinary arithmetic and control flow."""
    name = ins.name
    if name.startswith("cp.async.bulk.tensor"):
        mods = list(ins.mods)
        space = next(i for i, m in enumerate(mods) if m.startswith("shared"))
        return "TMA load" if space < mods.index("global") else "TMA store"
    for label, prefixes in FAMILIES:
        if prefixes and name.startswith(prefixes):
            return label
    return None


def key_instructions(launch: Launch) -> str:
    """The Blackwell-specific and synchronization instructions in a kernel's PTX: how many
    times each form occurs in the text, and the line of the first."""
    seen: dict[str, list] = {}
    for ins in launch.instructions:
        if ins.name.startswith(("tcgen05.", "cp.async", "mbarrier.", "bar.", "barrier.", "fence.",
                                "mapa", "elect.", "st.shared", "ld.shared")):
            seen.setdefault(ins.name, [0, ins.line])[0] += 1
    function = launch.module.functions[launch.kernel]
    params = ", ".join(f"{p.size}-byte struct" if p.by_value else f".{p.type}" for p in function.params)
    lines = [f"{len(launch.instructions)} PTX instructions for {launch.module.target}; parameters: {params}",
             "  count  first line  instruction"]
    lines += [f"  {count:>5}  {line:>10}  {name}" for name, (count, line) in sorted(seen.items())]
    return "\n".join(lines)


def launch_line(launch: Launch) -> str:
    s = launch.stats
    cluster = f", clusters of {launch.cluster}" if launch.cluster else ""
    return (f"grid {launch.grid}, block {launch.block}{cluster}: {s.ctas} CTA(s), {s.warps} warp(s), "
            f"{s.smem_bytes:,} bytes of shared memory per CTA\n"
            f"{s.instructions:,} warp-instructions executed ({s.lane_instructions:,} counted per thread), "
            f"{s.divergent_branches} divergent branch(es), {len(s.findings)} finding(s)")


class Watch:
    """A `trace` callback that records which warp executed which family of instruction.

    The simulator calls it before every instruction a warp executes. `counts` is per
    (CTA, warp); `events` is the order in which the watched CTAs' instructions ran.
    """

    def __init__(self, ctas=((0, 0, 0),), keep=("TMA load", "TMA store", "mbarrier wait", "MMA", "MMA commit",
                                                 "TMEM load")):
        self.counts: dict = collections.defaultdict(collections.Counter)
        self.events: list[tuple] = []  # (CTA, warp, family, PTX line)
        self.mma_shapes: set = set()  # (M, N, two-CTA?) of the MMAs issued
        self.ctas, self.keep = set(ctas), set(keep)

    def __call__(self, warp, pc, ins, active) -> None:
        what = family(ins)
        if what is None or not active.any():
            return
        cta = warp.cta.id
        self.counts[cta, warp.warp_id][what] += 1
        if cta in self.ctas and what in self.keep:
            self.events.append((cta, warp.warp_id, what, ins.line))
        if what == "MMA":  # the instruction descriptor holds the shape (PTX ISA, tcgen05.mma)
            lane = int(active.argmax())
            descriptor = int(warp.read(ins.operands[3], "u32")[lane])
            self.mma_shapes.add((((descriptor >> 24) & 31) << 4, ((descriptor >> 17) & 63) << 3,
                                 "cta_group::2" in ins.mods))

    def working_ctas(self, what: str = "MMA") -> list:
        return sorted({cta for (cta, _), counter in self.counts.items() if counter[what]})

    def roles(self, cta=(0, 0, 0)) -> str:
        """Per warp of one CTA, how many instructions of each family it executed."""
        warps = sorted(w for c, w in self.counts if c == cta)
        columns = [label for label, _ in FAMILIES if any(self.counts[cta, w][label] for w in warps)]
        width = [max(len(c), 5) for c in columns]
        lines = [f"CTA {cta}: instructions executed, by warp",
                 "  warp  " + "  ".join(c.rjust(n) for c, n in zip(columns, width))]
        for w in warps:
            cells = [str(self.counts[cta, w][c] or ".").rjust(n) for c, n in zip(columns, width)]
            lines.append(f"  {w:>4}  " + "  ".join(cells))
        return "\n".join(lines)

    def order(self, limit: int = 24, warps=None, cta=(0, 0, 0)) -> str:
        """The first `limit` watched steps of one CTA (of `warps`, if given), in the order the
        simulator ran them. A warp that polls an mbarrier repeats its wait; repeats are folded."""
        steps: list[list] = []
        for where, warp, what, line in self.events:
            if where != cta or (warps is not None and warp not in warps):
                continue
            if steps and steps[-1][:3] == [warp, what, line]:
                steps[-1][3] += 1
            else:
                steps.append([warp, what, line, 1])
        lines = [f"CTA {cta}: first {min(limit, len(steps))} of {len(steps)} asynchronous steps, in the "
                 "order this run's scheduler took them"]
        for warp, what, line, times in steps[:limit]:
            lines.append(f"  warp {warp:>2}  {what:<14} PTX line {line}" + (f", {times} times" if times > 1 else ""))
        return "\n".join(lines)


def bank_summary(launch: Launch, rows: int = 3) -> str:
    """Shared-memory bank conflicts of a launch: the totals, and the instructions that cost
    the most. `launch.stats.smem.format()` has the full table."""
    report = launch.stats.smem
    entries = sorted(report.instructions.values(), key=lambda e: -e.conflicts)
    lines = [f"{len(entries)} shared-memory instruction(s): {report.wavefronts:,} wavefronts, of which "
             f"{report.conflicts:,} are bank conflicts"]
    for e in entries[:rows]:
        if e.conflicts:
            lines.append(f"  line {e.line}: {' '.join(e.text.split())}  {e.executions} execution(s), "
                         f"{e.conflicts} conflict(s), worst {e.worst_way}-way")
    more = sum(1 for e in entries if e.conflicts) - rows
    if more > 0:
        lines.append(f"  ... and {more} more instruction(s) with conflicts")
    return "\n".join(lines)


def changes(before: Launch, after: Launch) -> str:
    """Instruction forms that one kernel's PTX has and the other's has not."""
    def forms(launch):
        return collections.Counter(i.name for i in launch.instructions
                                   if i.name.startswith(("tcgen05.", "cp.async", "mbarrier.", "barrier.",
                                                         "bar.", "fence.", "mapa", "ld.global", "st.global")))
    old, new = forms(before), forms(after)
    lines = ["PTX instruction forms that appear:"]
    lines += [f"  + {name} ({new[name]})" for name in sorted(new) if name not in old] or ["  (none)"]
    lines += ["and that are gone:"]
    lines += [f"  - {name} ({old[name]})" for name in sorted(old) if name not in new] or ["  (none)"]
    return "\n".join(lines)


def show_svg(svg: str, fallback: str = "") -> None:
    """Display an SVG in a notebook; outside one, print `fallback`."""
    try:
        from IPython.display import SVG, display
    except ImportError:
        print(fallback or f"(an SVG picture of {len(svg):,} bytes; run in Jupyter to see it)")
        return
    display(SVG(svg))
