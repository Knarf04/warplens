"""The report of a run: one HTML page with every view of each launch, and the views as files.

    from warplens.viz import write_report
    stats = sim.launch(ptx, "kernel", ..., timing=True)
    write_report("out", [("kernel", stats)])        # out/index.html and the files it links

`warplens report program` (warplens/cli.py) does the same for every launch of a program.

For each launch the page has the findings of the checkers, the bank-conflict table and its
heatmap (`viz.smem`), a layout picture of each shape of tile a TMA copy placed in shared
memory (`viz.smem`), the TMEM layout of each shape of MMA (`viz.tmem`) and, for a timed
launch, the timeline with a row per mbarrier (`viz.html_report`), with links to
the full timing report and the Perfetto trace. The pictures are inline in the page and also
written beside it as SVG files.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from ..host.tensor_map import SWIZZLE_MODES
from . import smem, tmem
from .html_report import _STYLE, TIMELINE_NOTE, timeline_svg

MAX_PICTURES = 6  # of one kind per launch; the rest are listed, not drawn
_PAGE_STYLE = _STYLE + """
body{max-width:1200px}pre{background:#f6f6f6;padding:.7em 1em;overflow-x:auto;font-size:.88em}
.view{overflow-x:auto;margin:.4em 0 1em}section{border-top:1px solid #ccc;margin-top:2em}
.finding{border-left:4px solid #d6604d}.ok{color:#2a7a2a}
"""


def _stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60].strip("_.") or "kernel"


def _some(values, most: int = 8) -> str:
    values = list(values)
    return ", ".join(map(str, values[:most])) + (", ..." if len(values) > most else "")


def tile_groups(report) -> list[dict]:
    """The TMA tensor copies of a launch (`LaunchStats.smem.tiles`) grouped by what their
    picture would be: kernel, direction, shape, swizzle and position within the swizzle's
    period. Dicts with kernel, verb, rows, row_bytes, elem_size, swizzle, the sorted PTX
    `lines` and shared offsets (`starts`) of the copies, and their number (`count`)."""
    groups: dict[tuple, dict] = {}
    for (kernel, line, verb, start, rows, row_bytes, elem, swizzle), count in report.tiles.items():
        phase = start % (128 << swizzle) if swizzle else start % 128
        g = groups.setdefault((kernel, verb, rows, row_bytes, elem, swizzle, phase), dict(
            kernel=kernel, verb=verb, rows=rows, row_bytes=row_bytes, elem_size=elem,
            swizzle=swizzle, lines=set(), starts=set(), count=0))
        g["lines"].add(line)
        g["starts"].add(start)
        g["count"] += count
    out = sorted(groups.values(), key=lambda g: (g["kernel"], min(g["lines"]), g["verb"], g["rows"]))
    for g in out:
        g["lines"], g["starts"] = sorted(g["lines"]), sorted(g["starts"])
    return out


def mma_groups(report) -> list[dict]:
    """The shapes of tcgen05.mma result of a launch (`LaunchStats.smem.mmas`): dicts with m,
    n, pair, lane, the PTX lines that issued that shape and the number of MMAs."""
    groups: dict[tuple, dict] = {}
    for (_, line, m, n, pair, lane), count in report.mmas.items():
        g = groups.setdefault((pair, m, n, lane), dict(m=m, n=n, pair=pair, lane=lane, lines=set(), count=0))
        g["lines"].add(line)
        g["count"] += count
    out = [groups[k] for k in sorted(groups)]
    for g in out:
        g["lines"] = sorted(g["lines"])
    return out


def _tile_picture(g: dict) -> tuple[str, str]:
    """(caption, SVG) of one group of `tile_groups`. The SVG is empty if it cannot be drawn."""
    name = SWIZZLE_MODES[g["swizzle"]][0]
    starts = g["starts"]
    caption = (f"TMA {g['verb']} of {g['rows']} rows x {g['row_bytes']} bytes "
               f"({g['row_bytes'] // g['elem_size']} elements of {g['elem_size']} byte(s) per row), swizzle "
               f"{name}: {g['count']} copies, issued at line(s) {_some(g['lines'])}, at shared offset(s) "
               f"{_some(starts)}." + (f" Drawn at offset {starts[0]}; the others look the same."
                                      if len(starts) > 1 else ""))
    try:
        words = smem.tile_words(g["rows"], g["row_bytes"], swizzle=g["swizzle"], start=starts[0])
    except ValueError:
        return caption + " Not drawn: its rows are not whole 4-byte words.", ""
    title = smem.tile_title(g["rows"], g["row_bytes"], g["swizzle"], starts[0])
    return caption, smem.layout_svg(words, title)


def _mma_picture(g: dict) -> tuple[str, str]:
    caption = (f"{g['count']} tcgen05.mma of this shape, issued at line(s) {_some(g['lines'])}."
               + (" Each CTA of the pair receives its half in its own tensor memory." if g["pair"] else ""))
    if g["pair"]:
        return caption, tmem.pair_svg(g["m"], g["n"])
    return caption, tmem.accumulator_svg(g["m"], g["n"], g["lane"])


def launch_section(name: str, stats, directory=None, stem: str | None = None) -> str:
    """The part of the page about one launch (`stats` is its LaunchStats), as an HTML
    fragment. With `directory`, every picture is also written there as `<stem>.<view>.svg`,
    and a timed launch gets `<stem>.timing.html` and `<stem>.trace.json`, which the fragment
    links."""
    stem = stem or _stem(name)
    directory = None if directory is None else Path(directory)
    files: list[str] = []

    def view(kind: str, svg: str) -> str:
        if directory is not None:
            (directory / f"{stem}.{kind}.svg").write_text(svg)
            files.append(f"{stem}.{kind}.svg")
        return f'<div class="view" data-view="{kind}">{svg}</div>'

    shown = name if len(name) <= 100 else name[:97] + "..."
    out = [f'<section><h2 title="{escape(name)}">{escape(shown)}</h2>']
    line = (f"{stats.ctas} CTA(s), {stats.warps} warp(s), {stats.instructions:,} warp-instructions, "
            f"{stats.smem_bytes:,} bytes of shared memory per CTA")
    if stats.canceled_ctas:
        line += f"; {stats.canceled_ctas} CTA(s) were canceled before starting"
    out.append(f"<p>{line}.</p>")

    out.append("<h3>Findings</h3>")
    if stats.findings:
        out.append(f"<p>{len(stats.findings)} finding(s), each with the PTX line that caused it.</p>")
        out += [f'<pre class="finding">{escape(str(f))}</pre>' for f in stats.findings]
    else:
        out.append('<p class="ok">The checkers reported nothing.</p>')

    out.append("<h3>Shared-memory banks</h3>")
    heatmap = smem.bank_heatmap_svg(stats.smem, "Banks loaded by each shared-memory instruction")
    if heatmap:
        out.append(view("banks", heatmap))
        out.append('<p class="note">Colour shows conflict, not load. A wavefront serves each bank once, so an '
                   'execution needs as many wavefronts as its fullest bank holds distinct 4-byte words; its '
                   'ideal is what a conflict-free layout of the same words needs (4 for a warp loading 16 '
                   'bytes per lane, which is not a conflict). A cell in the lightest shade stays within the '
                   'ideal. A darker cell exceeds it and shows its number of words.</p>\n')
        out.append(f"<pre>{escape(stats.smem.format())}</pre>")
    else:
        out.append('<p class="note">No instruction of this launch loaded or stored shared memory '
                   '(TMA copies and MMA operands do not go through the banks).</p>')

    out.append("<h3>Shared-memory layout</h3>")
    tiles = tile_groups(stats.smem)
    if tiles:
        for k, g in enumerate(tiles):
            caption, svg = _tile_picture(g)
            if k >= MAX_PICTURES or not svg:
                out.append(f'<p class="note">{escape(caption)}'
                           f'{" Not drawn: this page draws the first %d." % MAX_PICTURES if svg else ""}</p>')
                continue
            out.append(f"<p>{escape(caption)}</p>")
            out.append(view(f"smem_layout_{k + 1}", svg))
    else:
        out.append('<p class="note">This launch made no TMA tensor copy, the one way a tile gets a layout '
                   'the simulator knows. <code>python -m warplens.viz.smem</code> draws a tile from its '
                   'shape.</p>')

    out.append("<h3>Tensor memory</h3>")
    mmas = mma_groups(stats.smem)
    if mmas:
        for k, g in enumerate(mmas):
            caption, svg = _mma_picture(g)
            if k >= MAX_PICTURES:
                out.append(f'<p class="note">{g["m"]} x {g["n"]}: {escape(caption)} Not drawn: this page '
                           f'draws the first {MAX_PICTURES}.</p>')
                continue
            out.append(view(f"tmem_{k + 1}", svg))
            out.append(f'<p class="note">{escape(caption)}</p>')
    else:
        out.append('<p class="note">This launch issued no tcgen05.mma.</p>')

    out.append("<h3>Timeline and mbarrier phases</h3>")
    timing = stats.timing
    if timing is None:
        out.append('<p class="note">The launch was not timed, so there is no timeline. '
                   '<code>warplens report</code> times launches unless given <code>--no-timing</code>; '
                   'from Python, launch with <code>timing=True</code>.</p>')
    else:
        out.append(f'<p><b>{escape(timing.verdict)}.</b> Simulated time {timing.time_ns:,.0f} ns on '
                   f'{escape(timing.config.name)} by an approximate model ({escape(timing.config.basis)}). '
                   f'{escape(timing.config.caveat)}</p>')
        timeline = timeline_svg(timing)
        if timeline:
            out.append(view("timeline", timeline))
            out.append(f'<p class="note">{TIMELINE_NOTE}</p>')
        else:
            out.append('<p class="note">Nothing in this launch took simulated time.</p>')
        if directory is not None:
            timing.write_html(directory / f"{stem}.timing.html", f"{name}: timing report")
            timing.write_trace(directory / f"{stem}.trace.json")
            files += [f"{stem}.timing.html", f"{stem}.trace.json"]
            out.append(f'<p><a href="{escape(stem)}.timing.html">Full timing report</a>: utilization of each '
                       'resource, the roofline numbers, and the config entry and source behind every number. '
                       f'<a href="{escape(stem)}.trace.json">Trace</a> for '
                       '<a href="https://ui.perfetto.dev">ui.perfetto.dev</a>.</p>')
    if files:
        links = ", ".join(f'<a href="{escape(f)}">{escape(f)}</a>' for f in files)
        out.append(f'<p class="note">Files of this launch: {links}</p>')
    out.append("</section>")
    return "\n".join(out)


def render_page(sections, model: str = "B200", title: str = "warplens report", findings: int | None = None) -> str:
    """The whole page from the fragments of `launch_section`."""
    sections = list(sections)
    count = "" if findings is None else f", {findings} finding(s)"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title><style>{_PAGE_STYLE}</style></head><body>
<h1>{escape(title)}</h1>
<p>{len(sections)} kernel launch(es) simulated on {escape(model)}{count}. Executed on the CPU by warplens;
nothing here was measured on a GPU.</p>
{''.join(sections)}
</body></html>
"""


def write_report(directory, launches, model: str = "B200", title: str = "warplens report") -> Path:
    """Write the report of `launches`, a list of (kernel name, LaunchStats), into `directory`
    (created if missing). Returns the path of its `index.html`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    launches = list(launches)
    sections = [launch_section(name, stats, directory, f"{i + 1:02d}_{_stem(name)}")
                for i, (name, stats) in enumerate(launches)]
    index = directory / "index.html"
    index.write_text(render_page(sections, model, title, sum(len(s.findings) for _, s in launches)))
    return index
