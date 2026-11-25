"""The CuTe DSL kernel with TMA that was captured on a B200 (matrix row RT-11), replayed.

The capture is CUTLASS's fp16_gemm_0.py at --mnk 256,512,512: its PTX, its launch
parameters (two tensor maps the DSL wrote itself among them) and the device memory before
the kernel and after it. It is 42 MB and is in the rental's archive, not in the repository:
set WARPLENS_RT11_CAPTURE to `rows/e2e/cute_dsl_tma/capture` of an unpacked archive
(docs/hardware-capture.md). Without it the replay is left out of the run (deselected, as the
tests that need a GPU are: tests/conftest.py); the two instruction forms the
kernel needed are tested without it below."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import warplens

from conftest import kernel

CAPTURE = os.environ.get("WARPLENS_RT11_CAPTURE", "")
TOOL = Path(__file__).parents[1] / "validation/blackwell/e2e/replay_launch.py"


@pytest.mark.rental_archive("WARPLENS_RT11_CAPTURE")
@pytest.mark.skipif(not (Path(CAPTURE) / "launches.json").is_file(),
                    reason="WARPLENS_RT11_CAPTURE is not a capture of matrix row RT-11")
def test_the_captured_kernel_replays_byte_for_byte():
    spec = importlib.util.spec_from_file_location("replay_launch", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats, results = module.replay(Path(CAPTURE))
    assert results and all(different == 0 for _, _, different, _ in results)
    assert sum(written for _, _, _, written in results) > 0   # the GPU did write something
    assert stats.ctas == 4


def test_cvt_rn_f16x2_f32_packs_its_first_source_into_the_high_half(sim):
    """The order is a B200's: the capture above replays only this way."""
    mod = kernel("""
        .reg .f32 %a, %b; .reg .b32 %d; .reg .b64 %o;
        ld.param.u64 %o, [out];
        mov.f32 %a, 0f3FC00000;   // 1.5   -> 0x3E00
        mov.f32 %b, 0fC0000000;   // -2.0  -> 0xC000
        cvt.rn.f16x2.f32 %d, %a, %b;
        st.global.u32 [%o], %d;
        mov.f32 %a, 0f3F800400;   // 1 + 2^-13: a tie between 1 and the next half, to even
        mov.f32 %b, 0f7F7FFFFF;   // the largest float32: an infinity in half precision
        cvt.rn.f16x2.f32 %d, %a, %b;
        st.global.u32 [%o+4], %d;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    assert not sim.launch(mod, "k", grid=1, block=1, args=[out]).findings
    assert [hex(v) for v in out.numpy().tolist()] == ["0x3e00c000", "0x3c007c00"]
    for form in ("cvt.rz.f16x2.f32 %d, %a, %b;", "cvt.f16x2.f32 %d, %a, %b;", "cvt.rn.ftz.f16x2.f32 %d, %a, %b;"):
        bad = kernel(f".reg .f32 %a, %b; .reg .b32 %d;\nmov.f32 %a, 0f00000000;\nmov.f32 %b, 0f00000000;\n{form}\nret;")
        with pytest.raises(warplens.UnsupportedError, match="needs .rn and nothing else"):
            sim.launch(bad, "k", grid=1, block=1)


def test_an_mma_with_collector_a_discard_is_the_mma_without_it(sim):
    from test_tcgen05 import run
    plain = run(sim)
    discard = run(warplens.Device(), mods=".collector::a::discard")
    np.testing.assert_array_equal(plain[0], discard[0])
    assert not discard[2].findings
    for other in ("fill", "use", "lastuse"):
        with pytest.raises(warplens.UnsupportedError, match=f"MMA modifier .collector::a::{other}"):
            run(warplens.Device(), mods=f".collector::a::{other}")
