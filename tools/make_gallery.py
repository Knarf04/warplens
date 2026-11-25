"""The pictures of docs/gallery.md, made from the examples.

    python tools/make_gallery.py            # rewrite docs/gallery/
    python tools/make_gallery.py --check    # exit 1 if a file there is not what this would write

Every picture is the output of a public function of `warplens.viz` on a launch of a kernel
in `examples/`, so a reader can make the same one. tests/documentation/test_gallery.py runs the check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warplens  # noqa: E402
from warplens.viz import report, smem, timeline_svg  # noqa: E402

EXAMPLES = ROOT / "examples"
GALLERY = ROOT / "docs" / "gallery"
MARK = (5, 7)  # the element followed into memory in every layout picture: row 5, word 7


def transpose(layout: str):
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
    return sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", f"transpose_{layout}", grid=2,
                      block=(32, 32), args=[sim.from_numpy(data), sim.empty(data.shape, "f32")])


def tma_copy(bug: int):
    """examples/02_tma_copy, timed. Bug 3 leaves out the wait on the load's mbarrier."""
    sim = warplens.Device("B200")
    data = np.random.default_rng(0).standard_normal((48, 64)).astype(np.float32)
    dst = sim.from_numpy(np.full(data.shape, np.nan, dtype=np.float32))
    maps = [sim.tensor_map_tiled(t, (16, 32), swizzle="128B") for t in (sim.from_numpy(data), dst)]
    stats = sim.launch(EXAMPLES / "02_tma_copy" / "tma_copy.ptx", "tma_copy", grid=(2, 3), block=128,
                       args=[*maps, 32, 16, bug], timing=True)
    return stats, bool(np.array_equal(dst.numpy(), 2 * data))


def gemm(stages: int):
    spec = importlib.util.spec_from_file_location("gallery_gemm", EXAMPLES / "04_pipelined_gemm" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(stages, timing=True, active_sms=8)


def pictures() -> dict[str, str]:
    """File name -> contents of everything in docs/gallery/."""
    out = {}
    for layout, pitch in (("linear", 128), ("padded", 132)):
        stats = transpose(layout)
        out[f"transpose_{layout}.banks.svg"] = smem.bank_heatmap_svg(
            stats.smem, f"transpose_{layout}: banks loaded by each shared-memory instruction")
        out[f"transpose_{layout}.layout.svg"] = smem.layout_svg(
            smem.tile_words(32, 128, pitch=pitch), smem.tile_title(32, 128, pitch=pitch), mark=MARK)
    good, correct = tma_copy(0)
    assert correct and not good.findings
    bad, correct = tma_copy(3)
    assert not correct and bad.findings
    out["tma_copy.timeline.svg"] = timeline_svg(good.timing)
    out["tma_copy_no_wait.timeline.svg"] = timeline_svg(bad.timing)
    out["tma_copy_no_wait.findings.txt"] = "\n".join(str(f) for f in bad.findings) + "\n"
    load = next(g for g in report.tile_groups(good.smem) if g["verb"] == "load")
    out["tma_copy.layout.svg"] = smem.layout_svg(
        smem.tile_words(load["rows"], load["row_bytes"], swizzle=load["swizzle"], start=load["starts"][0]),
        smem.tile_title(load["rows"], load["row_bytes"], load["swizzle"], load["starts"][0]), mark=MARK)
    out["gemm_4_stages.timeline.svg"] = timeline_svg(gemm(4).timing)
    return out


def main(argv=None) -> int:
    check = "--check" in (sys.argv[1:] if argv is None else argv)
    stale = []
    for name, text in pictures().items():
        path = GALLERY / name
        if check:
            if not path.is_file() or path.read_text() != text:
                stale.append(name)
            continue
        GALLERY.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print("docs/gallery is out of date (python tools/make_gallery.py): " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
