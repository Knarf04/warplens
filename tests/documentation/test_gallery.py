"""docs/gallery.md shows what the code draws today: see tools/make_gallery.py."""

import importlib.util
import re

from tests.support import REPO as ROOT

spec = importlib.util.spec_from_file_location("make_gallery", ROOT / "tools" / "make_gallery.py")
make_gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_gallery)


def test_the_gallery_is_what_the_examples_draw_and_the_page_shows_all_of_it(capsys):
    assert make_gallery.main(["--check"]) == 0, capsys.readouterr().err
    page = (ROOT / "docs" / "gallery.md").read_text()
    prose = " ".join(page.split())
    files = {p.name for p in make_gallery.GALLERY.iterdir()}
    shown = set(re.findall(r"\]\(gallery/([^)]+)\)", page))
    assert shown == {name for name in files if name.endswith(".svg")}
    for line in (make_gallery.GALLERY / "tma_copy_no_wait.findings.txt").read_text().splitlines():
        assert line in page
    # What the text says of the pictures, where a number could drift.
    linear = (make_gallery.GALLERY / "transpose_linear.layout.svg").read_text()
    padded = (make_gallery.GALLERY / "transpose_padded.layout.svg").read_text()
    swizzled = (make_gallery.GALLERY / "tma_copy.layout.svg").read_text()
    assert "shared offset 668: the line at 640, bank 7." in linear and "668" in page
    assert "shared offset 688: the line at 640, bank 12." in padded and "688" in page
    assert "shared offset 716: the line at 640, bank 19." in swizzled and "716" in page
    caption = "illustrative parameters, 8 of 13 assumed and none fitted to hardware"
    assert caption in (make_gallery.GALLERY / "gemm_4_stages.timeline.svg").read_text() and caption in prose
