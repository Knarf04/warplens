"""The shared-memory views: the structure of what they draw, checked against the examples."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from warplens.host.tensor_map import swizzle_offsets
from warplens.viz import smem

from conftest import EXAMPLES

SVG = "{http://www.w3.org/2000/svg}"


def transpose(sim, layout):
    data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
    out = sim.empty(data.shape, "f32")
    return sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", f"transpose_{layout}",
                      grid=2, block=(32, 32), args=[sim.from_numpy(data), out])


def cells(svg):
    """(tooltip, fill) of every cell that has a tooltip."""
    root = ET.fromstring(svg)
    return [(rect.find(SVG + "title").text, rect.get("fill"))
            for rect in root.iter(SVG + "rect") if rect.find(SVG + "title") is not None]


def test_peak_words_per_bank_tell_a_moving_conflict_from_none(sim):
    store, load = sorted(transpose(sim, "linear").smem.instructions.values(), key=lambda e: e.line)
    # Summed over executions the two look alike; the peak of one execution does not.
    assert store.bank_words.tolist() == load.bank_words.tolist() == [64] * 32
    assert store.bank_peak.tolist() == [1] * 32 and load.bank_peak.tolist() == [32] * 32
    assert all(e.bank_peak.max() == 1 for e in transpose(sim, "padded").smem.instructions.values())


def test_heatmap_has_a_row_per_instruction_and_a_cell_per_bank(sim):
    stats = transpose(sim, "linear")
    svg = smem.bank_heatmap_svg(stats.smem)
    root = ET.fromstring(svg)
    tips = cells(svg)
    assert len(tips) == 2 * 32
    assert sum("at most 32 distinct word(s) in one execution" in tip for tip, _ in tips) == 32
    assert sum("at most 1 distinct word(s) in one execution" in tip for tip, _ in tips) == 32
    assert len({fill for _, fill in tips}) == 2  # conflict-free and 32-way are different shades
    texts = [e.text for e in root.iter(SVG + "text")]
    assert texts.count("32") == 32 and "32-way" in texts and "none" in texts
    assert any(t and t.startswith("line 54  ld.shared.b32") for t in texts)
    assert "1,984 in 64 execution(s)" in texts  # the table's conflicts column


def test_heatmap_of_a_launch_without_shared_accesses_is_empty(sim):
    stats = sim.launch(EXAMPLES / "00_vecadd" / "vecadd.ptx", "vecadd", grid=1, block=32,
                       args=[sim.from_numpy(np.zeros(32, np.float32))] * 3 + [32])
    assert smem.bank_heatmap_svg(stats.smem) == ""


def test_untouched_banks_are_marked(sim):
    from conftest import kernel
    mod = kernel(".shared .b32 cell[4]; .reg .b32 %v; st.shared.u32 [cell+8], 7; ret;")
    stats = sim.launch(mod, "k", grid=1, block=32)
    tips = [tip for tip, _ in cells(smem.bank_heatmap_svg(stats.smem))]
    assert sum("not touched" in tip for tip in tips) == 31
    assert any(tip.startswith("line") and "bank 2: at most 1" in tip for tip in tips)


@pytest.mark.parametrize("swizzle, bits", [(None, 0), ("32B", 1), ("64B", 2), ("128B", 3)])
def test_tile_words_follow_the_tma_swizzle(swizzle, bits):
    rows, row_bytes, start = 16, 128, 2048
    words = smem.tile_words(rows, row_bytes, swizzle=swizzle, start=start)
    linear = start + np.arange(rows * row_bytes, dtype=np.uint64)
    expected = swizzle_offsets(linear, bits).reshape(rows, row_bytes)[:, ::4]
    assert np.array_equal(words, expected.astype(np.int64))
    assert len(np.unique(words)) == words.size  # a permutation: no two words share bytes


def test_column_conflicts_of_the_three_transpose_layouts():
    assert smem.column_conflict(smem.tile_words(32, 128)) == 32  # tile[32][32]
    assert smem.column_conflict(smem.tile_words(32, 128, pitch=132)) == 1  # tile[32][33]
    assert smem.column_conflict(smem.tile_words(32, 128, swizzle="128B")) == 4  # 16-byte chunks move
    # The example's own swizzle, tile[y][x ^ y], as an array of word offsets.
    y, x = np.mgrid[0:32, 0:32]
    assert smem.column_conflict(4 * (32 * y + (x ^ y))) == 1


def test_layout_svg_draws_each_word_twice_and_colours_it_by_bank():
    words = smem.tile_words(32, 128, swizzle="128B")
    svg = smem.layout_svg(words, "t")
    tips = cells(svg)
    assert len(tips) == 2 * words.size  # once in the logical tile, once in shared memory
    by_bank = {}
    for tip, fill in tips:
        by_bank.setdefault(int(tip.rsplit("bank ", 1)[1]), set()).add(fill)
    assert sorted(by_bank) == list(range(32))
    assert all(len(fills) == 1 for fills in by_bank.values())  # one colour per bank
    assert len({next(iter(f)) for f in by_bank.values()}) == 32  # and a different one for each
    assert ("row 1, bytes 0-3: shared offset 144, bank 4", next(iter(by_bank[4]))) in tips
    root = ET.fromstring(svg)
    outlines = [r for r in root.iter(SVG + "rect") if r.get("fill") == "none"]
    assert len(outlines) == 2 * 32  # the first word of each row, in both pictures
    assert "has a 4-way bank conflict" in svg


def test_layout_svg_draws_a_bounded_number_of_rows_and_says_so():
    svg = smem.layout_svg(smem.tile_words(128, 128, swizzle="128B"))
    assert len(cells(svg)) == 2 * 32 * 32 and "The first 32 of 128 rows are drawn." in svg
    ET.fromstring(smem.layout_svg(smem.tile_words(4, 256)))  # rows wider than a line


@pytest.mark.parametrize("arguments", [
    dict(rows=0, row_bytes=128), dict(rows=4, row_bytes=6), dict(rows=4, row_bytes=128, start=2),
    dict(rows=4, row_bytes=128, swizzle="128B", pitch=132), dict(rows=4, row_bytes=128, pitch=64),
    dict(rows=4, row_bytes=128, swizzle="256B"), dict(rows=4, row_bytes=24, swizzle="32B"),
])
def test_impossible_tiles_are_refused(arguments):
    with pytest.raises(ValueError):
        smem.tile_words(**arguments)


def test_text_view_and_command(capsys, tmp_path):
    text = smem.layout_text(smem.tile_words(32, 128, swizzle="128B"), "T")
    assert text.splitlines()[0] == "T" and "... 16 more row(s)" in text
    assert "    1                       144   4  5  6  7  0  1  2  3" in text
    assert smem.main(["32", "128", "--pitch", "132", "--svg", str(tmp_path / "t.svg")]) == 0
    out = capsys.readouterr().out
    assert "rows 132 bytes apart" in out and "has no bank conflict" in out
    assert (tmp_path / "t.svg").read_text().startswith("<svg")
    assert smem.main(["32", "128", "--swizzle", "bogus"]) == 2
    assert "unknown swizzle" in capsys.readouterr().err


def test_a_launch_records_the_tiles_tma_placed_and_the_mmas(sim):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gemm_run_viz", EXAMPLES / "04_pipelined_gemm" / "run.py")
    gemm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm)
    report = gemm.run(2).smem
    shapes = {(verb, rows, row_bytes, elem, swizzle) for _, _, verb, _, rows, row_bytes, elem, swizzle
              in report.tiles}
    assert shapes == {("load", 128, 128, 2, 3), ("load", 64, 128, 2, 3)}  # A and B tiles, 128B swizzle
    assert sum(report.tiles.values()) == 2 * 16  # 16 tiles along K of each
    assert {key[2:] for key in report.mmas} == {(128, 64, False, 0)}
    assert sum(report.mmas.values()) == 64  # K = 1024 in MMAs of K = 16
