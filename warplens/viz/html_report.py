"""A self-contained HTML report of a timed launch.

Every number is shown with the config entries it was derived from, and every
config entry is listed with its source. Assumed values are marked.
"""

from __future__ import annotations

from html import escape

from ..timing.config import TIMING_PARAMS
from ..timing.model import RESOURCE_SOURCES, SATURATED
from .trace_perfetto import slice_lanes

_COLORS = {"run": "#9ecae1", "wait": "#fdd0a2", "TMA": "#74c476", "MMA": "#9e9ac8"}
_STYLE = """
body{font-family:system-ui,sans-serif;max-width:1040px;margin:2em auto;padding:0 1em;color:#222}
table{border-collapse:collapse;margin:.6em 0}td,th{border:1px solid #ccc;padding:4px 9px;text-align:left;vertical-align:top}
th{background:#f3f3f3}.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{background:#e9e9e9;width:220px;height:13px}.bar div{height:13px;background:#4c78a8}.bar div.hot{background:#d6604d}
.assumed{background:#fff4ce}.note{color:#555;font-size:.9em}.warn{background:#fff4ce;border:1px solid #e0c060;padding:.6em 1em}
a.cfg{font-family:ui-monospace,monospace;font-size:.9em}
"""


def _links(config, keys) -> str:
    return ", ".join(f'<a class="cfg" href="#cfg-{k}">{k}</a>' + ("*" if config.params[k].is_assumption else "")
                     for k in keys)


def timeline_svg(report) -> str:
    """The timeline of a timed launch as a standalone SVG: one row per warp and engine of the
    first CTA, with what each was doing when. `report` is `LaunchStats.timing`. Returns an
    empty string when the launch recorded nothing with a duration. `render_html` embeds the
    same picture."""
    slices = [s for s in report.slices if s.end > s.start]
    if not slices:
        return ""
    cta = min(s.cta for s in slices)
    slices = [s for s in slices if s.cta == cta]
    lanes = list(slice_lanes(slices))
    tracks = sorted({track for _, track in lanes}, key=lambda t: (not t.startswith("warp"), t))
    span = max(s.end for s in slices) or 1.0
    left, width, row = 180, 810, 22
    height = len(tracks) * row + 34
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 10}" height="{height}" '
           'font-family="sans-serif" font-size="11">']
    for i, track in enumerate(tracks):
        y = 4 + i * row
        out.append(f'<text x="{left - 6}" y="{y + 13}" text-anchor="end">{escape(track)}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{row - 5}" fill="#f7f7f7"/>')
        for s, display_track in lanes:
            if display_track != track:
                continue
            x = left + s.start / span * width
            w = max((s.end - s.start) / span * width, 0.6)
            color = _COLORS.get(s.name.split(":")[0].split()[0], "#bbbbbb")
            tip = f"{s.name}: {s.start:,.0f} to {s.end:,.0f} ns" + \
                "".join(f", {k}={v}" for k, v in s.args.items())
            out.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row - 5}" fill="{color}" '
                       f'stroke="#ffffff" stroke-width="0.4"><title>{escape(tip)}</title></rect>')
    y = 4 + len(tracks) * row
    for name, when, where in report.instants:
        if where == cta:
            x = left + when / span * width
            out.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{y - 4}" stroke="#d6604d" '
                       f'stroke-width="0.7"><title>{escape(name)} at {when:,.0f} ns</title></line>')
    out.append(f'<text x="{left}" y="{y + 12}">0 ns</text>')
    out.append(f'<text x="{left + width}" y="{y + 12}" text-anchor="end">{span:,.0f} ns</text>')
    out.append(f'<text x="{left + width / 2}" y="{y + 26}" text-anchor="middle" fill="#555">CTA {cta}: '
               'blue run, orange wait, green TMA copy, purple MMA; red lines are mbarrier phase completions</text>')
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
        cls = ' class="assumed"' if p.is_assumption else ""
        config_rows.append(f'<tr id="cfg-{p.key}"{cls}><td><code>{p.key}</code></td><td class="num">{p.value:g}</td>'
                           f'<td>{escape(p.unit)}</td><td>{source}</td><td class="note">{escape(p.note)}</td></tr>')
    working = getattr(report, "working_ctas", report.ctas)
    idle = "" if working >= report.ctas else (
        f' Busy fractions and the pipeline window are averaged over the <span id="working-ctas">'
        f'{working}</span> CTA(s) that did work; {report.ctas - working} did none and are left out.')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title><style>{_STYLE}</style></head><body>
<h1>{escape(title)}</h1>
<p class="warn">Simulated by warplens's approximate timing model for {escape(cfg.name)}. It shows where time
goes under the stated parameters; it is not a measurement and has not been calibrated against hardware.
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
<p class="note">Overlapping asynchronous operation lifetimes use extra display lanes; these are
not additional hardware engines. Their bars include completion latency, not only resource busy time.</p>
<h2>Configuration and sources</h2>
<table><tr><th>Entry</th><th>Value</th><th>Unit</th><th>Source</th><th>Note</th></tr>{''.join(config_rows)}</table>
</body></html>
"""


def write_html(report, path, title: str = "warplens timing report") -> None:
    """Write `render_html(report, title)` to `path`."""
    with open(path, "w") as f:
        f.write(render_html(report, title))
