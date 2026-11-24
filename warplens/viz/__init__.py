"""Pictures and reports of what a launch did.

    from warplens.viz import render_html, write_html, timeline_svg   # of a timed launch
    from warplens.viz import tmem                                     # where MMA results land in TMEM

`warplens.viz.tmem` and `warplens.viz.trace_perfetto` are modules of their own; the first is
also a command (`python -m warplens.viz.tmem`).
"""

from .html_report import render_html, timeline_svg, write_html

__all__ = ["render_html", "timeline_svg", "write_html"]
