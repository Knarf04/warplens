import xml.etree.ElementTree as ET

import numpy as np
import pytest

from warplens.viz import tmem

NS = "{http://www.w3.org/2000/svg}"


def blocks(svg):
    """Of a pair picture: its text labels, and {even/odd: [(lanes, rows, columns) per block]}."""
    labels = [e.text for e in ET.fromstring(svg).iter(NS + "text")]
    out, current = {}, None
    for i, label in enumerate(labels):
        if label.endswith("CTA of the pair"):
            current = out.setdefault(label.split("-")[0], [])
        elif label.startswith("rows ") and current is not None:
            current.append((labels[i + 2], label, labels[i + 1]))
    return labels, out


def test_pair_svg_of_layout_b_agrees_with_locate():
    labels, held = blocks(tmem.pair_svg(128, 16))
    assert labels[0] == "TMEM layout of a 128 x 16 tcgen05.mma result (cta_group::2, Layout B)"
    assert held == {
        "even": [("lanes 0-31", "rows 0-31", "columns 0-7"), ("lanes 32-63", "rows 32-63", "columns 0-7"),
                 ("lanes 64-95", "rows 0-31", "columns 8-15"), ("lanes 96-127", "rows 32-63", "columns 8-15")],
        "odd": [("lanes 0-31", "rows 64-95", "columns 0-7"), ("lanes 32-63", "rows 96-127", "columns 0-7"),
                ("lanes 64-95", "rows 64-95", "columns 8-15"), ("lanes 96-127", "rows 96-127", "columns 8-15")]}
    # The plan's worked example, C[10, 11] of the 128 x 16 result: lane 74 of the even CTA,
    # which is the block the picture labels rows 0-31, columns 8-15 on lanes 64-95.
    assert tmem.locate(10, 11, 128, 16, pair=True) == (0, 74, 3)
    assert labels.count("TMEM columns 0-7: result column n is TMEM column n mod 8") == 2
    assert [t for t in labels if t.startswith("reachable")] == [f"reachable by warp {w} (mod 4)" for w in range(4)]


def test_pair_svg_of_256_rows_and_its_likeness_to_the_single_cta_picture():
    labels, held = blocks(tmem.pair_svg(256, 64))
    assert [rows for _, rows, _ in held["even"]] == ["rows 0-31", "rows 32-63", "rows 64-95", "rows 96-127"]
    assert [rows for _, rows, _ in held["odd"]] == ["rows 128-159", "rows 160-191", "rows 192-223", "rows 224-255"]
    assert {columns for _, _, columns in held["even"] + held["odd"]} == {"columns 0-63"}
    for cta in (0, 1):  # the blocks against the placement the simulator's tests pin down
        for lane in (0, 31, 32, 127):
            assert tmem.locate(128 * cta + lane, 5, 256, 64, pair=True) == (cta, lane, 5)

    def geometry(svg):
        return {(r.get("height"), r.get("fill")) for r in ET.fromstring(svg).iter(NS + "rect")
                if r.get("fill") != "none"}

    # The same lane scale and the same colour per warp as the cta_group::1 picture of 128 rows.
    assert geometry(tmem.pair_svg(256, 64)) == geometry(tmem.accumulator_svg(128, 64))
    for m, n in [(64, 32), (128, 24)]:
        with pytest.raises(ValueError):
            tmem.pair_svg(m, n)


def test_pair_svg_from_the_command(capsys, tmp_path):
    assert tmem.main(["128", "16", "--pair", "--svg", str(tmp_path / "p.svg")]) == 0
    assert "Layout B" in capsys.readouterr().out
    assert (tmp_path / "p.svg").read_text() == tmem.pair_svg(128, 16)
    assert tmem.main(["64", "16", "--pair", "--svg", str(tmp_path / "q.svg")]) == 2
    assert not (tmp_path / "q.svg").exists()


def test_accumulator_rows():
    assert tmem.accumulator_rows(128).tolist() == list(range(128))
    rows = tmem.accumulator_rows(64)
    for warp in range(4):
        assert rows[32 * warp:32 * warp + 16].tolist() == list(range(16 * warp, 16 * warp + 16))
        assert (rows[32 * warp + 16:32 * warp + 32] == -1).all()


def test_text_view_of_the_64_row_layout():
    text = tmem.accumulator_text(64, 32)
    assert "Layout F" in text and "  0-15     0     result rows 0-15" in text
    assert " 16-31     0     result rows 0-15" in tmem.accumulator_text(64, 32, lane_offset=16)
    assert " 16-31     0     not written" in text
    assert " 96-111    3     result rows 48-63" in text
    assert tmem.accumulator_text(128, 8).count("result rows") == 4


def test_svg_is_well_formed_and_labels_every_block():
    root = ET.fromstring(tmem.accumulator_svg(64, 32))
    labels = [e.text for e in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "rows 16-31" in labels and "lanes 32-47" in labels and labels.count("not written") == 4


def test_transfer_view_and_cli(capsys, tmp_path):
    text = tmem.transfer_text("16x128b", 1)
    assert "2 register(s) per thread, 16 lanes x 4 columns" in text
    assert tmem.main(["64", "32", "--svg", str(tmp_path / "t.svg")]) == 0
    assert (tmp_path / "t.svg").read_text().startswith("<svg")
    assert tmem.main(["--transfer", "16x64b", "2"]) == 0
    assert "tcgen05.ld/st.16x64b.x2" in capsys.readouterr().out
    assert tmem.main(["96", "8"]) == 2
    assert np.array_equal(tmem.accumulator_rows(64) >= 0, np.arange(128) % 32 < 16)
