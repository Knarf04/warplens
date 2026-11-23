"""Write a timed launch as a trace for Perfetto (ui.perfetto.dev) or chrome://tracing.

One process per CTA; one track per warp and per asynchronous unit. Slices are
runs, waits, TMA copies and MMAs; instants mark mbarrier phase completions; a
counter shows the bytes TMA has in flight. Times are simulated nanoseconds.
"""

from __future__ import annotations

import json
from collections import defaultdict


def slice_lanes(slices):
    """Give overlapping operations separate display lanes, without changing time.

    Chrome/Perfetto complete events on one track must nest. Async operation
    lifetimes can cross instead, even when their server's busy periods do not.
    """
    ends = defaultdict(list)
    for s in sorted(slices, key=lambda s: (s.cta, s.track, s.start, s.end)):
        if s.end <= s.start:
            continue
        lanes = ends[s.cta, s.track]
        lane = next((i for i, end in enumerate(lanes) if end <= s.start), len(lanes))
        if lane == len(lanes):
            lanes.append(s.end)
        else:
            lanes[lane] = s.end
        track = s.track if lane == 0 else f"{s.track} (overlap {lane + 1})"
        yield s, track


def trace_events(report) -> list[dict]:
    events: list[dict] = []
    ctas = sorted({s.cta for s in report.slices} | {c for _, _, c in report.instants})
    pid = {cta: i + 1 for i, cta in enumerate(ctas)}
    tids: dict[tuple, int] = {}

    def tid(cta, track: str) -> int:
        key = (cta, track)
        if key not in tids:
            tids[key] = len([k for k in tids if k[0] == cta]) + 1
            events.append({"ph": "M", "name": "thread_name", "pid": pid[cta], "tid": tids[key],
                           "args": {"name": track}})
        return tids[key]

    for cta in ctas:
        events.append({"ph": "M", "name": "process_name", "pid": pid[cta],
                       "args": {"name": f"CTA {cta} on its SM"}})
    for s, track in slice_lanes(report.slices):
        events.append({"ph": "X", "name": s.name, "cat": s.track.split()[0], "pid": pid[s.cta],
                       "tid": tid(s.cta, track), "ts": s.start / 1e3, "dur": (s.end - s.start) / 1e3,
                       "args": s.args})
    for name, when, cta in report.instants:
        events.append({"ph": "i", "name": name, "s": "p", "pid": pid[cta], "tid": 0, "ts": when / 1e3})
    for cta in ctas:  # bytes in flight in TMA
        changes = defaultdict(int)
        for s in report.slices:
            if s.cta == cta and s.track == "TMA engine":
                # Copies are in flight from issue, including time queued for
                # the TMA server. Coalesce simultaneous completions/issues.
                changes[s.args.get("issued_ns", s.start)] += s.args.get("bytes", 0)
                changes[s.end] -= s.args.get("bytes", 0)
        level = 0
        for when, delta in sorted(changes.items()):
            level += delta
            events.append({"ph": "C", "name": "TMA bytes in flight", "pid": pid[cta],
                           "ts": when / 1e3, "args": {"bytes": level}})
    return events


def write_trace(report, path) -> None:
    with open(path, "w") as f:
        json.dump({"traceEvents": trace_events(report), "displayTimeUnit": "ns",
                   "otherData": {"model": f"warplens approximate timing, {report.config.name}",
                                 "warning": "simulated time from an uncalibrated model"}}, f)
