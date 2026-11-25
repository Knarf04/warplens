"""`warplens report` and `warplens.viz.write_report`: what the page holds and which files it writes."""

import importlib.util
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import warplens
from warplens import cli
from warplens.viz import report as viz_report
from warplens.viz import write_report

from conftest import EXAMPLES

SVG = "{http://www.w3.org/2000/svg}"


def gemm(stages=2, **timing):
    spec = importlib.util.spec_from_file_location("gemm_run_report", EXAMPLES / "04_pipelined_gemm" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(stages, **timing)


def transpose(layout="linear"):
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
    return sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", f"transpose_{layout}",
                      grid=2, block=(32, 32), args=[sim.from_numpy(data), sim.empty(data.shape, "f32")])


def broken_copy():
    """examples/02_tma_copy with its fence.proxy.async left out (bug 4)."""
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal((48, 64)).astype(np.float32)
    maps = [sim.tensor_map_tiled(sim.from_numpy(data), (16, 32), swizzle="128B") for _ in range(2)]
    return sim.launch(EXAMPLES / "02_tma_copy" / "tma_copy.ptx", "tma_copy", grid=(2, 3), block=128,
                      args=[*maps, 32, 16, 4])


def views(html):
    """The kinds of the pictures in a page, in order."""
    return re.findall(r'<div class="view" data-view="([a-z_0-9]+)">', html)


def test_report_of_a_timed_gemm_has_every_view_inline_and_as_a_file(tmp_path):
    stats = gemm(2, timing=True, active_sms=8)
    index = write_report(tmp_path / "out", [("pipelined_gemm", stats)], title="T")
    html = index.read_text()
    assert index == tmp_path / "out" / "index.html" and "<title>T</title>" in html
    assert views(html) == ["banks", "smem_layout_1", "smem_layout_2", "tmem_1", "timeline"]
    names = sorted(p.name for p in index.parent.iterdir())
    assert names == ["01_pipelined_gemm." + suffix for suffix in (
        "banks.svg", "smem_layout_1.svg", "smem_layout_2.svg", "timeline.svg", "timing.html",
        "tmem_1.svg", "trace.json")] + ["index.html"]
    for name in names:  # every file is linked from the page and is what its name says
        if name == "index.html":
            continue
        assert f'href="{name}"' in html
        text = (index.parent / name).read_text()
        if name.endswith(".svg"):
            ET.fromstring(text)
            assert text in html
    assert "traceEvents" in json.loads((index.parent / "01_pipelined_gemm.trace.json").read_text())
    assert "Configuration and sources" in (index.parent / "01_pipelined_gemm.timing.html").read_text()
    # The two tiles of the kernel, each drawn once though four stage buffers hold them.
    assert "TMA load of 128 rows x 128 bytes (64 elements of 2 byte(s) per row), swizzle 128B: 16 copies" in html
    assert "TMA load of 64 rows x 128 bytes" in html and "TMEM layout of a 128 x 64 tcgen05.mma result" in html
    assert "64 tcgen05.mma of this shape" in html and "The checkers reported nothing." in html
    assert stats.timing.verdict in html and "not a prediction" in html
    assert "Each mbarrier has a row" in html and "illustrative parameters, 8 of 13 assumed" in html


def test_report_shows_findings_with_their_ptx_line_and_says_what_is_absent(tmp_path):
    stats = broken_copy()
    assert stats.findings and stats.timing is None
    html = write_report(tmp_path, [("tma_copy", stats), ("transpose_linear", transpose())]).read_text()
    assert f"{len(stats.findings)} finding(s)" in html.split("<section>")[0]  # the page's count
    copy, linear = html.split("<section>")[1:]
    for finding in stats.findings:
        assert f"[{finding.kind}]" in copy and f"line {finding.line}:" in copy
    assert copy.count('<pre class="finding">') == len(stats.findings)
    assert "The launch was not timed" in copy and "timing.html" not in copy
    assert "This launch issued no tcgen05.mma" in copy
    assert views(copy)[-2:] == ["smem_layout_1", "smem_layout_2"]  # its load and its store
    assert "TMA store of 16 rows x 128 bytes" in copy
    assert views(linear) == ["banks"] and "32-way" in linear
    assert "This launch made no TMA tensor copy" in linear
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "01_tma_copy.banks.svg", "01_tma_copy.smem_layout_1.svg", "01_tma_copy.smem_layout_2.svg",
        "02_transpose_linear.banks.svg", "index.html"]


def test_tiles_are_grouped_by_the_picture_they_would_make():
    groups = viz_report.tile_groups(gemm(4).smem)
    assert [(g["rows"], g["row_bytes"], g["swizzle"], len(g["starts"]), g["count"]) for g in groups] == [
        (128, 128, 3, 4, 16), (64, 128, 3, 4, 16)]
    assert all(len(g["lines"]) == 4 for g in groups)  # the kernel has one copy instruction per stage
    (mma,) = viz_report.mma_groups(gemm(1).smem)
    assert (mma["m"], mma["n"], mma["pair"], mma["lane"], mma["count"]) == (128, 64, False, 0, 64)


def test_a_pair_mma_is_drawn_with_both_ctas():
    from types import SimpleNamespace
    caption, svg = viz_report._mma_picture(
        dict(m=256, n=256, pair=True, lane=0, lines=[7, 9], count=3))
    assert "issued at line(s) 7, 9" in caption and "odd-ranked CTA of the pair" in svg
    fake = SimpleNamespace(mmas={("k", 7, 256, 256, True, 0): 2, ("k", 9, 256, 256, True, 0): 1,
                                 ("k", 9, 64, 32, False, 16): 5})
    assert [(g["m"], g["lane"], g["count"], g["lines"]) for g in viz_report.mma_groups(fake)] == [
        (64, 16, 5, [9]), (256, 0, 3, [7, 9])]


def test_long_and_unsafe_kernel_names_make_safe_file_names(tmp_path):
    name = "_Z4gemmI" + "N7cutlass6half_tE" * 20 + "<>/..\\ x"
    write_report(tmp_path, [(name, transpose("padded"))])
    (svg,) = [p.name for p in tmp_path.iterdir() if p.suffix == ".svg"]
    assert re.fullmatch(r"01_[A-Za-z0-9_.-]{1,60}\.banks\.svg", svg)
    html = (tmp_path / "index.html").read_text()
    assert "&lt;&gt;/..\\ x" in html and "<>/" not in html.split("<svg")[0].split("<h2")[1]


# -- the command ----------------------------------------------------------------------------

def test_report_command_writes_the_page_for_what_the_run_launched(tmp_path, monkeypatch, capsys):
    stats = transpose()
    seen = {}

    def fake_run(args, collect=None):
        seen.update(timing=args.timing, report_dir=args.report_dir, model=args.model, program=args.program)
        collect([("transpose_linear", stats)],
                [{"lines": ["  triton_kernel: 1 CTA(s)"], "findings": 2, "section": "<section>FROM-TRITON</section>"},
                 {"lines": ["  an older child"], "findings": 0}])
        return 3

    monkeypatch.setattr(cli, "run", fake_run)
    out = tmp_path / "deep" / "report"
    assert cli.main(["report", "-o", str(out), "--model", "B300", "./prog", "--its-flag"]) == 3
    assert seen == dict(timing=True, report_dir=str(out), model="B300", program="./prog")
    html = (out / "index.html").read_text()
    assert "warplens report: prog" in html and "2 kernel launch(es) simulated on B300, 2 finding(s)" in html
    assert html.index("transpose_linear") < html.index("FROM-TRITON")
    assert (out / "01_transpose_linear.banks.svg").is_file()
    assert f"warplens: wrote {out / 'index.html'}" in capsys.readouterr().err


def test_report_command_without_launches_writes_no_page(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda args, collect=None: collect([], []) or 0)
    assert cli.main(["report", "-o", str(tmp_path / "r"), "--no-timing", "prog"]) == 0
    assert not (tmp_path / "r" / "index.html").exists()
    assert "no report written: no kernel was launched" in capsys.readouterr().err


def test_report_command_options(monkeypatch, capsys, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "run", lambda args, collect=None: seen.append(args) or 0)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["report", "--no-timing", "prog"]) == 0
    assert seen[-1].timing is False and seen[-1].output == "warplens_report"
    assert cli.main(["report", "--active-sms", "8", "prog"]) == 0
    assert seen[-1].timing is True and seen[-1].active_sms == 8
    assert cli.main(["report", "--timing", "prog"]) == 0  # not an abbreviation of --timing-config
    assert seen[-1].timing is True and seen[-1].program == "prog" and seen[-1].timing_config is None
    for arguments, message in [(["--no-timing", "--active-sms", "8"], "not allowed with"),
                               (["--active-sms", "100000"], "must be from 1 to 148"),
                               (["--timing", "--no-timing"], "not allowed with")]:
        with pytest.raises(SystemExit) as stop:
            cli.main(["report", *arguments, "prog"])
        assert stop.value.code == 2 and message in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["report", "--help"])
    text = capsys.readouterr().out
    assert "--output DIRECTORY" in text and "--no-timing" in text and "--clc-limit" in text


def test_a_triton_child_renders_its_launches_into_the_report_directory(tmp_path, monkeypatch):
    """What `install_triton_driver` does at exit under `warplens report`, without Triton."""
    from types import SimpleNamespace
    import atexit
    import sys

    stats = transpose()
    driver = SimpleNamespace(launches=[("add_kernel", stats)])
    monkeypatch.setitem(sys.modules, "warplens.triton_driver",
                        SimpleNamespace(install=lambda model, **options: driver))
    monkeypatch.setattr(warplens, "triton_driver", sys.modules["warplens.triton_driver"], raising=False)
    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    told = tmp_path / "told"
    monkeypatch.setenv(cli.RUN_OPTIONS, json.dumps({
        "model": "B200", "timing": False, "timing_config": None, "active_sms": None, "clc_limit": None,
        "report": str(told), "report_dir": str(tmp_path), "package": "x"}))
    cli.install_triton_driver()
    registered[0]()
    (launch,) = json.loads(told.read_text())
    assert launch["findings"] == 0 and launch["section"].startswith("<section>")
    (svg,) = [p.name for p in tmp_path.glob("*.svg")]
    assert re.fullmatch(r"p\d+_01_add_kernel\.banks\.svg", svg) and f'href="{svg}"' in launch["section"]
