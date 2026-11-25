"""The timeline: which barrier and phase each wait is for, the HBM row, and what the timing
outputs say about where their parameters come from."""

import importlib.util
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from warplens.timing.config import GpuConfig
from warplens.timing.model import Slice
from warplens.viz import render_html, smem, timeline_svg
from warplens.viz.html_report import barrier_names, merged_waits
from warplens.viz.trace_perfetto import trace_events

from conftest import EXAMPLES, kernel

SVG = "{http://www.w3.org/2000/svg}"
FIT = EXAMPLES.parent / "validation" / "blackwell_timing" / "measured" / "b200" / "calibrated.yaml"


@pytest.fixture(scope="module")
def gemm():
    spec = importlib.util.spec_from_file_location("gemm_run_timeline", EXAMPLES / "04_pipelined_gemm" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def tips(svg, tag):
    return [e.find(SVG + "title").text for e in ET.fromstring(svg).iter(SVG + tag)
            if e.find(SVG + "title") is not None]


def test_every_phase_completion_names_its_barrier_and_phase(gemm):
    t = gemm(2, timing=True, active_sms=8).timing
    offsets = sorted({i.args["mbarrier"] for i in t.instants})
    assert len(t.instants) == 33 and len(offsets) == 5  # full[2], empty[2], done
    for offset in offsets:  # the phases of one barrier complete in order, each once
        phases = [i.args["phase"] for i in t.instants if i.args["mbarrier"] == offset]
        assert phases == list(range(len(phases)))
        assert all(i.name == f"mbarrier at shared offset {offset}: phase {i.args['phase']} completes"
                   for i in t.instants if i.args["mbarrier"] == offset)
    # Every wait on an mbarrier says which phase of which barrier, and the last wait for a
    # phase ends when that phase completes.
    done = {(i.args["mbarrier"], i.args["phase"]): i.when for i in t.instants}
    waits = [s for s in t.slices if s.name == "wait: mbarrier"]
    assert waits and all(len(s.args["mbarriers"]) == 1 for s in waits)
    last = {}
    for s in waits:
        key = (s.track, *s.args["mbarriers"][0])
        last[key] = max(last.get(key, 0.0), s.end)
        assert s.args["on"] == "mbarrier at shared offset {}, phase {}".format(*s.args["mbarriers"][0])
    assert all(end == pytest.approx(done[key[1:]]) for key, end in last.items())
    # The loads of a stage report to its `full` barrier, the two lowest offsets.
    loads = [s for s in t.slices if s.name == "TMA load"]
    assert {s.args["mbarrier"] for s in loads} == set(offsets[:2])


def test_timeline_joins_each_wait_to_the_phase_that_ends_it(gemm):
    t = gemm(2, timing=True, active_sms=8).timing
    svg = timeline_svg(t)
    root = ET.fromstring(svg)
    texts = [e.text for e in root.iter(SVG + "text") if e.text]
    marks = tips(svg, "path")
    assert len(marks) == 33 and all(re.match(r"mbarrier [A-E] \(shared offset \d+\): phase \d+ completes", m)
                                    for m in marks)
    # full[s] ends the consumer's wait, empty[s] the producer's, done everybody's.
    assert sum("ends the wait of warp 1" in m for m in marks if m.startswith(("mbarrier A", "mbarrier B"))) == 16
    assert all("ends the wait of warp 0" in m or "no warp was waiting" in m
               for m in marks if m.startswith(("mbarrier C", "mbarrier D")))
    (last,) = [m for m in marks if m.startswith("mbarrier E")]
    assert "ends the wait of warp 0, warp 1, warp 2, warp 3" in last
    # A wait is one bar although every other completion woke the warp in between.
    bars = tips(svg, "rect")
    a0 = [b for b in bars if b.startswith("warp 1, wait: mbarrier A0 (phase 0 of the mbarrier at shared offset")]
    assert len(a0) == 1 and "A0" in texts and "E0" in texts
    assert any("woken 1 time(s) by the completion of another mbarrier" in b for b in bars)
    assert any(b.startswith("TMA engine, TMA load, reports to mbarrier A") for b in bars)
    # The engines the plan promises, and a statement of what the times are.
    assert {"warp 0", "TMA engine", "Tensor Core", "HBM"} <= set(texts)
    assert any(b.startswith("HBM, HBM fetch: ") and "bytes=" in b for b in bars)
    assert any("not a measurement" in x and "illustrative parameters, 8 of 13 assumed" in x for x in texts)


def test_interrupted_waits_merge_only_for_the_same_phase():
    on = lambda offset, phase: {"mbarriers": [[offset, phase]]}  # noqa: E731
    raw = [Slice("warp 0", "run", 0, 10), Slice("warp 0", "wait: mbarrier", 10, 50, args=on(8, 0)),
           Slice("warp 0", "run", 50, 51), Slice("warp 0", "wait: mbarrier", 51, 90, args=on(8, 0)),
           Slice("warp 0", "run", 90, 91), Slice("warp 0", "wait: mbarrier", 91, 120, args=on(16, 0)),
           Slice("warp 0", "run", 120, 121), Slice("warp 0", "run", 121, 200),
           Slice("TMA engine", "TMA load", 5, 90)]
    out = sorted(merged_waits(raw, 2.0), key=lambda s: (s.track, s.start))
    assert [(s.name, s.start, s.end) for s in out if s.track == "warp 0"] == [
        ("run", 0, 10), ("wait: mbarrier", 10, 90), ("run", 90, 91), ("wait: mbarrier", 91, 120),
        ("run", 120, 121), ("run", 121, 200)]
    assert out[2].args["rewoken"] == 1 and "rewoken" not in raw[1].args  # the report is not changed
    assert barrier_names([], raw) == {8: "A", 16: "B"}


def test_trace_has_a_track_per_barrier_and_one_for_hbm(gemm):
    t = gemm(2, timing=True, active_sms=8).timing
    events = trace_events(t)
    tracks = {e["tid"]: e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "thread_name"}
    barriers = {name for name in tracks.values() if name.startswith("mbarrier at shared offset ")}
    assert len(barriers) == 5 and "HBM" in tracks.values()
    instants = [e for e in events if e["ph"] == "i"]
    assert all(tracks[e["tid"]] == f"mbarrier at shared offset {e['args']['mbarrier']}"
               and e["name"] == f"phase {e['args']['phase']} completes" for e in instants)
    wait = next(e for e in events if e["ph"] == "X" and e["name"] == "wait: mbarrier")
    assert wait["args"]["on"].startswith("mbarrier at shared offset ")
    hbm = [e for e in events if e["ph"] == "X" and e["name"] == "HBM fetch"]
    assert sum(e["args"]["bytes"] for e in hbm) == t.bytes_loaded  # nothing was in L2 before


def test_reports_say_whether_parameters_are_illustrative_or_fitted(gemm):
    default, fit = GpuConfig.load("B200"), GpuConfig.load(FIT)
    assert default.basis == "illustrative parameters, 8 of 13 assumed and none fitted to hardware"
    assert fit.basis.startswith("experimental fit, ") and "fitted to microbenchmarks of one GPU" in fit.basis
    assert not default.fitted and "clock_ghz" in fit.fitted
    assert default.replace(clock_ghz=1).basis.endswith(", 1 overridden by the caller")
    for config, label in ((default, "illustrative parameters"), (fit, "experimental fit")):
        t = gemm(1, timing=config, active_sms=8).timing
        html = render_html(t)
        assert f'<b id="basis">Parameters: {label}' in html and "it is not a measurement" in html
        assert ('id="cfg-clock_ghz" class="fitted"' in html) == (config is fit)
        assert f"  parameters: {label}" in t.summary() and "model, not a measurement" in t.summary()
        assert label in timeline_svg(t) and "not a measurement" in timeline_svg(t)
    assert "a fitted issue rate, not a clock" in t.summary()


def test_a_wide_load_without_conflict_is_not_drawn_as_one(sim):
    mod = kernel("""
        .shared .align 16 .b32 buf[256];
        .reg .b32 %t, %o, %a, %b, %c, %d; .reg .b64 %p, %base;
        mov.u64 %base, buf;
        mov.u32 %t, %tid.x; shl.b32 %o, %t, 4; cvt.u64.u32 %p, %o; add.u64 %p, %p, %base;
        ld.shared.v4.u32 {%a, %b, %c, %d}, [%p];
        and.b32 %o, %t, 1; mul.lo.u32 %o, %o, 128; cvt.u64.u32 %p, %o; add.u64 %p, %p, %base;
        ld.shared.u32 %a, [%p];
        ret;""")
    stats = sim.launch(mod, "k", grid=1, block=32)
    wide, pair = sorted(stats.smem.instructions.values(), key=lambda e: e.line)
    # 32 lanes x 16 bytes: four words in every bank, four wavefronts, and that is the ideal.
    assert wide.bank_peak.tolist() == [4] * 32 and wide.bank_ratio.tolist() == [1.0] * 32
    assert (wide.worst, wide.worst_ideal, wide.conflicts) == (4, 4, 0)
    assert pair.bank_peak[0] == 2 and pair.bank_ratio[0] == 2.0 and pair.worst_way == 2
    svg = smem.bank_heatmap_svg(stats.smem)
    root = ET.fromstring(svg)
    cells = [(r.find(SVG + "title").text, r.get("fill")) for r in root.iter(SVG + "rect")
             if r.find(SVG + "title") is not None]
    fills = {fill for tip, fill in cells if "line 11" in tip}
    assert len(fills) == 1 and all("no more than a conflict-free layout needs" in tip
                                   for tip, _ in cells if "line 11" in tip)
    (hot,) = [fill for tip, fill in cells if "up to 2 times what a conflict-free layout needs" in tip]
    assert hot not in fills
    texts = [e.text for e in root.iter(SVG + "text")]
    assert "4 needed, ideal 4" in texts and "2 needed, ideal 1" in texts
    in_cells = [e.text for e in root.iter(SVG + "text") if e.get("pointer-events") == "none"]
    assert in_cells == ["2"]  # only the conflicting bank shows its words


def test_layout_can_follow_one_element_into_memory(tmp_path, capsys):
    words = smem.tile_words(32, 128, pitch=132)
    svg = smem.layout_svg(words, mark=(5, 7))
    assert "Ringed: row 5, bytes 28-31 of the tile is at shared offset 688: the line at 640, bank 12." in svg
    assert svg.count('stroke="#eb6834"') == 2  # in the logical tile and in shared memory
    assert "Ringed" not in smem.layout_svg(words)
    with pytest.raises(ValueError, match="outside a tile"):
        smem.layout_svg(words, mark=(32, 0))
    out = tmp_path / "t.svg"
    assert smem.main(["16", "128", "--swizzle", "128B", "--mark", "5", "7", "--svg", str(out)]) == 0
    assert "row 5, word 7 is at shared offset 716, bank 19" in capsys.readouterr().out
    assert "shared offset 716" in out.read_text()
    assert smem.main(["16", "128", "--mark", "16", "0"]) == 2
