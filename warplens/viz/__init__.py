"""Pictures and reports of what a launch did.

    from warplens.viz import write_report                             # every view of a run, as a page
    from warplens.viz import render_html, write_html, timeline_svg   # of a timed launch
    from warplens.viz import smem                                     # bank heatmap, tile layout
    from warplens.viz import tmem                                     # where MMA results land in TMEM

`warplens.viz.smem`, `warplens.viz.tmem` and `warplens.viz.trace_perfetto` are modules of
their own; the first two are also commands (`python -m warplens.viz.smem`).
"""

from .html_report import render_html, timeline_svg, write_html


def write_report(directory, launches, model: str = "B200", title: str = "warplens report"):
    """Write the report of `launches`, a list of (kernel name, LaunchStats), into `directory`:
    `index.html` and the files it links. Returns the path of `index.html`. See `viz.report`."""
    # Imported here: `python -m warplens.viz.smem` must not find its module already loaded.
    from .report import write_report as write
    return write(directory, launches, model, title)


__all__ = ["render_html", "timeline_svg", "write_html", "write_report"]
