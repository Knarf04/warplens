"""A self-contained HTML report of a timed launch.

Every number is shown with the config entries it was derived from, and every
config entry is listed with its source. Assumed values are marked.
"""

from __future__ import annotations

from html import escape

from ..timing.config import TIMING_PARAMS
from ..timing.model import RESOURCE_SOURCES, SATURATED
from .trace_perfetto import slice_lanes

# One colour per kind of bar; the row says which warp or engine. Checked with the palette
# validator of the chart guidelines this project follows (separable under colour blindness).
_COLORS = {"run": "#2a78d6", "wait": "#eb6834", "TMA": "#1baf7a", "cp.async": "#1baf7a",
           "MMA": "#6250d6", "HBM": "#eda100", "L2": "#e87ba4"}
_LEGEND = [("run", "warp runs"), ("wait", "warp waits"), ("TMA", "TMA or cp.async copy, start to landing"),
           ("MMA", "MMA"), ("HBM", "HBM busy"), ("L2", "L2 busy")]
_INK, _MUTED, _GRID, _ROW = "#222222", "#666666", "#e1e0d9", "#f4f4f2"
_ENGINES = ("TMA engine", "Tensor Core", "HBM", "L2")
_STYLE = """
body{font-family:system-ui,sans-serif;max-width:1040px;margin:2em auto;padding:0 1em;color:#222}
table{border-collapse:collapse;margin:.6em 0}td,th{border:1px solid #ccc;padding:4px 9px;text-align:left;vertical-align:top}
th{background:#f3f3f3}.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{background:#e9e9e9;width:220px;height:13px}.bar div{height:13px;background:#4c78a8}.bar div.hot{background:#d6604d}
.assumed{background:#fff4ce}.fitted{background:#e3f1e6}.note{color:#555;font-size:.9em}.warn{background:#fff4ce;border:1px solid #e0c060;padding:.6em 1em}
a.cfg{font-family:ui-monospace,monospace;font-size:.9em}
"""


def _links(config, keys) -> str:
    return ", ".join(f'<a class="cfg" href="#cfg-{k}">{k}</a>' + ("*" if config.params[k].is_assumption else "")
                     for k in keys)


TIMELINE_NOTE = (
    "Each mbarrier has a row; a diamond is one of its phases completing, and the line from it rises to "
    "the warps whose wait that ended. A wait bar names what it waits for (B2: phase 2 of mbarrier B) and a "
    "TMA load the barrier it reports to; hover over any mark for the same in words, with the PTX line. "
    "A TMA or MMA bar runs from when its engine takes it up to its completion, latency included, and overlapping ones take extra "
    "display lanes, which are not additional hardware engines. The HBM and L2 rows show only the time this SM's "
    "share of each was busy.")


def barrier_names(instants, slices=()) -> dict[int, str]:
    """A letter for each mbarrier of one CTA, by shared offset: A for the lowest. The timeline
    writes `A3` for phase 3 of barrier A."""
    offsets = {i.args["mbarrier"] for i in instants if "mbarrier" in i.args}
    for s in slices:
        offsets.update(o for o, _ in s.args.get("mbarriers", ()))
        if "mbarrier" in s.args:
            offsets.add(s.args["mbarrier"])
    names = {}
    for k, offset in enumerate(sorted(offsets)):
        name = ""
        k += 1
        while k:
            k, r = divmod(k - 1, 26)
            name = chr(65 + r) + name
        names[offset] = name
    return names


def merged_waits(slices, slack: float):
    """The slices of one CTA with each interrupted mbarrier wait drawn as one bar.

    A warp in `mbarrier.try_wait` is woken by every phase completion of its CTA and waits
    again if it was not the one it asked for. That records several waits for the same phase
    with one instruction of `run` between them; here they are one wait, with the number of
    times it was woken for another barrier in `args["rewoken"]`."""
    out, last = [], {}
    for s in sorted(slices, key=lambda s: (s.start, s.end)):
        if not s.track.startswith("warp"):
            out.append(s)
            continue
        wait = last.get(s.track)
        if (wait is not None and "_retry" not in wait.args and s.name == "run"
                and s.end - s.start <= slack and s.start <= wait.end + slack):
            wait.args["_retry"] = s
            continue  # the re-executed try_wait: kept only if the wait is not resumed
        retry = None if wait is None else wait.args.pop("_retry", None)
        if (wait is not None and retry is not None and s.name == wait.name
                and s.args.get("mbarriers") and s.args.get("mbarriers") == wait.args.get("mbarriers")):
            wait.end = s.end
            wait.args["rewoken"] = wait.args.get("rewoken", 0) + 1
            continue
        if retry is not None:
            out.append(retry)
        if s.args.get("mbarriers"):
            s = type(s)(s.track, s.name, s.start, s.end, s.cta, dict(s.args))
            last[s.track] = s
        else:
            last.pop(s.track, None)
        out.append(s)
    for wait in last.values():
        retry = wait.args.pop("_retry", None)
        if retry is not None:
            out.append(retry)
    return out


def _ticks(span: float) -> list[float]:
    step = 10 ** len(str(int(span / 6))) / 10 if span >= 6 else span
    for k in (1, 2, 5, 10):
        if span / (step * k) <= 8:
            step *= k
            break
    return [step * i for i in range(int(span / step) + 1)]


def timeline_svg(report) -> str:
    """The timeline of a timed launch as a standalone SVG. `report` is `LaunchStats.timing`.

    One row per warp, per engine (TMA, Tensor Core, HBM, L2) and per mbarrier of the first
    CTA. A diamond in a barrier's row is one of its phases completing, numbered; a line rises
    from it to the warps whose wait it ended. A wait bar carries the barrier's letter and the
    phase it waits for (`B2`), a TMA load the letter of the barrier it reports to. Every mark
    has a tooltip with the same in words. Returns an empty string when the launch recorded
    nothing with a duration. `render_html` embeds the same picture."""
    slices = [s for s in report.slices if s.end > s.start]
    if not slices:
        return ""
    cta = min(s.cta for s in slices)
    clock = 1.0 / report.config["clock_ghz"]
    slices = merged_waits([s for s in slices if s.cta == cta], 2.5 * clock)
    instants = [i for i in report.instants if i.cta == cta and "mbarrier" in i.args]
    names = barrier_names(instants, slices)
    lanes = list(slice_lanes(slices))

    def order(track: str):
        if track.startswith("warp"):
            return 0, int(track.split()[1]), track
        base = track.split(" (")[0]
        return 1, _ENGINES.index(base) if base in _ENGINES else len(_ENGINES), track

    tracks = sorted({track for _, track in lanes}, key=order)
    span = max([s.end for s in slices] + [i.when for i in instants]) or 1.0
    left, width, row, top = 190, 810, 22, 8
    bar_h = row - 5
    rows = {track: top + k * row for k, track in enumerate(tracks)}
    barrier_top = top + len(tracks) * row + (18 if names else 0)
    barrier_y = {offset: barrier_top + k * row for k, offset in enumerate(sorted(names))}
    axis = barrier_top + len(names) * row + 2
    height = axis + 62 + (34 if names else 0)

    def x(t: float) -> float:
        return left + t / span * width

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 16}" height="{height}" '
           f'font-family="sans-serif" font-size="11" fill="{_INK}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>']
    for tick in _ticks(span):  # behind everything
        out.append(f'<line x1="{x(tick):.2f}" y1="{top}" x2="{x(tick):.2f}" y2="{axis}" stroke="{_GRID}"/>')
        out.append(f'<text x="{x(tick):.2f}" y="{axis + 13}" text-anchor="middle" fill="{_MUTED}">'
                   f'{tick:,.0f}{" ns" if tick == 0 else ""}</text>')
    for track, y in rows.items():
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end">{escape(track)}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{bar_h}" fill="{_ROW}"/>')
    released: dict[tuple, list[str]] = {}  # (offset, phase) -> the warp rows it released
    for s, track in lanes:
        y = rows[track]
        x0, w = x(s.start), max((s.end - s.start) / span * width, 0.6)
        kind = s.name.split(":")[0].split()[0]
        color = _COLORS.get(kind, "#9a9a95")
        args = {k: v for k, v in s.args.items() if k not in ("mbarriers", "on", "mbarrier", "rewoken")}
        tip = f"{track}, {s.name}"
        labels = []  # the first that fits is written in the bar
        if s.name.startswith("wait: "):
            labels = [s.name[6:]]  # bar.sync, a bulk group, a tensor-memory allocation, ...
        elif track.startswith(("TMA engine", "HBM", "L2")):
            labels = [s.name.split()[-1]]  # load, store, copy; fetch, write
        waits_on = [(o, p) for o, p in s.args.get("mbarriers", ())]
        if waits_on:
            tip += " " + " and ".join(f"{names[o]}{p} (phase {p} of the mbarrier at shared offset {o})"
                                      for o, p in waits_on)
            labels = ["+".join(f"{names[o]}{p}" for o, p in waits_on)]
            for key in waits_on:
                released.setdefault(key, []).append((track, s.end))
        elif "mbarrier" in s.args:
            tip += f", reports to mbarrier {names[s.args['mbarrier']]} (shared offset {s.args['mbarrier']})"
            arrow = f"→{names[s.args['mbarrier']]}"
            labels = [f"{labels[0]} {arrow}", arrow]
        tip += f": {s.start:,.0f} to {s.end:,.0f} ns" + "".join(
            f", {k}={v:,.0f}" if isinstance(v, float) else f", {k}={v}" for k, v in args.items())
        if s.args.get("rewoken"):
            tip += (f"; woken {s.args['rewoken']} time(s) by the completion of another mbarrier and "
                    "waited again")
        out.append(f'<rect x="{x0:.2f}" y="{y}" width="{w:.2f}" height="{bar_h}" fill="{color}" '
                   f'stroke="#ffffff" stroke-width="0.5"><title>{escape(tip)}</title></rect>')
        label = next((text for text in labels if w >= 5.6 * len(text) + 6), "")
        if label:
            ink = "#0b0b0b" if kind == "HBM" else "#ffffff"
            out.append(f'<text x="{x0 + w / 2:.2f}" y="{y + 12}" text-anchor="middle" font-size="10" '
                       f'fill="{ink}" pointer-events="none">{escape(label)}</text>')
    if names:
        out.append(f'<text x="{left - 8}" y="{barrier_top - 4}" text-anchor="end" fill="{_MUTED}" '
                   'font-size="10">mbarrier, by shared offset</text>')
    for offset, y in barrier_y.items():
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end"><tspan font-weight="bold">'
                   f'{names[offset]}</tspan> at {offset}</text>')
        out.append(f'<line x1="{left}" y1="{y + bar_h / 2}" x2="{left + width}" y2="{y + bar_h / 2}" '
                   f'stroke="{_GRID}"/>')
    label_end = dict.fromkeys(barrier_y, 0.0)  # right edge of the last phase number in each row
    for i in sorted(instants, key=lambda i: i.when):
        offset, phase = i.args["mbarrier"], i.args["phase"]
        xi, y = x(i.when), barrier_y[offset] + bar_h / 2
        woke = sorted({track for track, end in released.get((offset, phase), ())
                       if abs(end - i.when) <= 1e-6 * max(span, 1.0) and track in rows},
                      key=lambda t: rows[t])
        tip = (f"mbarrier {names[offset]} (shared offset {offset}): phase {phase} completes at "
               f"{i.when:,.0f} ns; " + (f"ends the wait of {', '.join(woke)}" if woke else
                                        "no warp was waiting for it"))
        if woke:
            out.append(f'<line x1="{xi:.2f}" y1="{rows[woke[0]] + bar_h}" x2="{xi:.2f}" y2="{y - 5}" '
                       f'stroke="{_INK}" stroke-width="0.9" pointer-events="none"/>')
            for track in woke:
                out.append(f'<circle cx="{xi:.2f}" cy="{rows[track] + bar_h}" r="2.6" fill="{_INK}" '
                           f'stroke="#ffffff" stroke-width="0.8" pointer-events="none"/>')
        out.append(f'<path d="M{xi:.2f} {y - 5.5}l5.5 5.5l-5.5 5.5l-5.5 -5.5z" fill="{_INK}" stroke="#ffffff" '
                   f'stroke-width="0.8"><title>{escape(tip)}</title></path>')
        text = str(phase)
        if xi + 8 >= label_end[offset] and xi + 8 + 6 * len(text) <= left + width + 14:
            out.append(f'<text x="{xi + 8:.2f}" y="{y + 4}" font-size="10" fill="{_MUTED}" '
                       f'pointer-events="none">{text}</text>')
            label_end[offset] = xi + 8 + 6 * len(text) + 3
    y = axis + 32
    xk = left
    used = {s.name.split(":")[0].split()[0] for s, _ in lanes}
    for kind, text in _LEGEND:
        if kind not in used and not (kind == "TMA" and "cp.async" in used):
            continue
        out.append(f'<rect x="{xk}" y="{y - 10}" width="12" height="12" fill="{_COLORS[kind]}"/>')
        out.append(f'<text x="{xk + 17}" y="{y}">{text}</text>')
        xk += 17 + 6.1 * len(text) + 18
    if names:
        y += 18
        out.append(f'<path d="M{left + 6} {y - 10}l5.5 5.5l-5.5 5.5l-5.5 -5.5z" fill="{_INK}"/>')
        out.append(f'<text x="{left + 17}" y="{y}">an mbarrier completes the numbered phase; the line rises to '
                   'the warps whose wait it ends</text>')
        y += 16
        out.append(f'<text x="{left}" y="{y}">On a wait, B2: the warp waits for phase 2 of mbarrier B. '
                   'On a TMA load, →B: it reports its bytes to B.</text>')
    out.append(f'<text x="{left}" y="{y + 20}" fill="{_MUTED}">CTA {cta}. Simulated time from warplens\'s approximate '
               f'model, not a measurement. {escape(report.config.name)}: {escape(report.config.basis)}.</text>')
    out.append("</svg>")
    return "\n".join(out)


def render_html(report, title: str = "warplens timing report") -> str:
    """The whole report of a timed launch (`LaunchStats.timing`) as one HTML page."""
    cfg = report.config
    r = report.roofline()
    timing_keys = tuple(sorted(TIMING_PARAMS))
    time_sources = _links(cfg, timing_keys) + ', <a href="#launch-active-sms">active_sms</a>'
    compute_keys = (*report.compute_source_keys, "sm_count")
    rows = []
    for name, value in report.utilization.items():
        hot = ' class="hot"' if value >= SATURATED else ""
        rows.append(f'<tr><td>{name}</td><td class="num">{value:.0%}</td>'
                    f'<td><div class="bar"><div{hot} style="width:{min(value, 1) * 220:.0f}px"></div></div></td>'
                    f'<td>{_links(cfg, RESOURCE_SOURCES[name])}'
                    + (', <a href="#launch-active-sms">active_sms</a>' if name == "HBM" else "")
                    + '</td></tr>')
    numbers = [
        ("Kernel time", f"{report.time_ns:,.0f} ns", time_sources),
        ("Kernel time in clocks", f"{report.clocks:,.0f}", time_sources),
        ("Mean asynchronous pipeline window", f"{report.pipeline_ns:,.0f} ns", time_sources),
        ("SMs assumed to share HBM", f'<span id="launch-active-sms">{report.active_sms}</span>',
         f'Launch parameter <code>active_sms</code>; defaults to {_links(cfg, ("sm_count",))}'),
        ("Pipeline rate per SM", f"{r['achieved_flops_per_s'] / 1e12:.2f} TFLOP/s", time_sources),
        ("Compute ceiling per SM", f"{r['compute_ceiling_flops_per_s'] / 1e12:.2f} TFLOP/s",
         _links(cfg, compute_keys)),
        ("Memory ceiling per SM", f"{r['memory_ceiling_flops_per_s'] / 1e12:.2f} TFLOP/s at "
                                  f"{r['flops_per_byte']:.1f} FLOP/byte",
         _links(cfg, ("hbm_bandwidth_tb_s",)) + ', <a href="#launch-active-sms">active_sms</a>'),
    ]
    number_rows = "".join(f'<tr><td>{n}</td><td class="num">{v}</td><td>{k}</td></tr>'
                          for n, v, k in numbers)
    config_rows = []
    for p in cfg.params.values():
        source = f'<a href="{escape(p.url)}">{escape(p.source)}</a>' if p.url else escape(p.source)
        cls = ' class="assumed"' if p.is_assumption else ' class="fitted"' if p.is_fitted else ""
        config_rows.append(f'<tr id="cfg-{p.key}"{cls}><td><code>{p.key}</code></td><td class="num">{p.value:g}</td>'
                           f'<td>{escape(p.unit)}</td><td>{source}</td><td class="note">{escape(p.note)}</td></tr>')
    working = getattr(report, "working_ctas", report.ctas)
    idle = "" if working >= report.ctas else (
        f' Busy fractions and the pipeline window are averaged over the <span id="working-ctas">'
        f'{working}</span> CTA(s) that did work; {report.ctas - working} did none and are left out.')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title><style>{_STYLE}</style></head><body>
<h1>{escape(title)}</h1>
<p class="warn">Simulated by warplens's approximate timing model for {escape(cfg.name)}; it is not a measurement.
<b id="basis">Parameters: {escape(cfg.basis)}.</b> {escape(cfg.caveat)}
Values marked * are model assumptions, including measurements borrowed from another GPU.</p>
<h2>Verdict</h2><p><b>{escape(report.verdict)}</b></p>
<p>{report.ctas} CTA(s) simulated, each alone on one SM.{idle}</p>
<h2>Resources</h2>
<table><tr><th>Resource</th><th>Busy</th><th></th><th>Derived from</th></tr>{''.join(rows)}</table>
<p class="note">Busy is the fraction of time the resource was working: for the TMA engine, HBM, L2 and
Tensor Core, of the interval in which asynchronous work was in flight; otherwise of the whole kernel.
A resource busy at least {SATURATED:.0%} of the time is called the bottleneck. Resource links identify
the service-cost parameters; the denominator also depends on the timing parameters below.</p>
<h2>Numbers</h2>
<table><tr><th>Quantity</th><th>Value</th><th>Derived from</th></tr>{number_rows}</table>
<p class="note">Pipeline rate divides the mean MMA FLOPs per CTA by the mean asynchronous pipeline
window, excluding kernel setup and epilogue time. FLOPs and TMA bytes come from the executed kernel;
the memory ceiling treats all those bytes as HBM traffic. For mixed MMA kinds, the compute ceiling
uses their FLOP-weighted service times. Listed timing parameters can affect the scheduled duration;
parameters for unused operations do not contribute.</p>
<h2>Timeline</h2>{timeline_svg(report)}
<p class="note">{TIMELINE_NOTE}</p>
<h2>Configuration and sources</h2>
<p class="note"><span class="assumed">Shaded like this</span>: an assumption. <span class="fitted">Like
this</span>: fitted to microbenchmarks by <code>validation/blackwell_timing/fit.py</code>. Unshaded: a
published figure or the caller's override.</p>
<table><tr><th>Entry</th><th>Value</th><th>Unit</th><th>Source</th><th>Note</th></tr>{''.join(config_rows)}</table>
</body></html>
"""


def write_html(report, path, title: str = "warplens timing report") -> None:
    """Write `render_html(report, title)` to `path`."""
    with open(path, "w") as f:
        f.write(render_html(report, title))
