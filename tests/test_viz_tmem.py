import xml.etree.ElementTree as ET

import numpy as np

from warplens.viz import tmem


def test_accumulator_rows():
    assert tmem.accumulator_rows(128).tolist() == list(range(128))
    rows = tmem.accumulator_rows(64)
    for warp in range(4):
        assert rows[32 * warp:32 * warp + 16].tolist() == list(range(16 * warp, 16 * warp + 16))
        assert (rows[32 * warp + 16:32 * warp + 32] == -1).all()


def test_text_view_of_the_64_row_layout():
    text = tmem.accumulator_text(64, 32)
    assert "  0-15     0     result rows 0-15" in text
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
