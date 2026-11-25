"""The frozen v1 instruction scope (docs/history/plan.md section 3.5): what is in is pinned, what is out
is refused by name, never run as something else."""

from types import SimpleNamespace

import numpy as np
import pytest

import warplens
from warplens.frontend.ir import Instr
from warplens.host.tensor_map import swizzle_offsets
from warplens.units.tcgen05 import _operand_bytes

from conftest import kernel

MMA = Instr("tcgen05", ("mma", "cta_group::1", "kind::f16"), (), None, False, 7,
            "tcgen05.mma.cta_group::1.kind::f16 [%d], %a, %b, %i;")
WARP = SimpleNamespace(cta=SimpleNamespace(smem=SimpleNamespace(size=65536)))
# layout name: (descriptor layout type, swizzled bits, bytes per operand row)
LAYOUTS = {"32B": (6, 1, 32), "64B": (4, 2, 64), "128B": (2, 3, 128)}


def descriptor(start, row_bytes, layout, base_offset=0):
    """K-major, as validation/blackwell/common.h `mma_smem_desc`, plus the base-offset field."""
    return ((start >> 4) | (1 << 16) | ((8 * row_bytes >> 4) << 32) | (1 << 46) | (base_offset << 49)
            | (layout << 61))


@pytest.mark.parametrize("name, start", [("128B", 128), ("128B", 384), ("128B", 896), ("128B", 64),
                                         ("64B", 128), ("64B", 384), ("32B", 128)])
def test_an_operand_off_the_swizzle_period_is_swizzled_by_address(name, start):
    """What validation/blackwell/mma/base_offset predicts (matrix row MMA-12)."""
    layout, bits, row_bytes = LAYOUTS[name]
    idx = _operand_bytes(WARP, MMA, descriptor(start, row_bytes, layout), 128, np.dtype(np.float16), "A")
    r = np.arange(128, dtype=np.uint64)[:, None, None]
    k = np.arange(16, dtype=np.uint64)[None, :, None]
    b = np.arange(2, dtype=np.uint64)[None, None, :]
    offsets = (r % 8) * row_bytes + (r // 8) * 8 * row_bytes + (k % 8) * 2 + (k // 8) * 16 + b
    by_address = swizzle_offsets(np.uint64(start) + offsets, bits)
    relative = np.uint64(start) + swizzle_offsets(offsets, bits)
    assert np.array_equal(idx, by_address)
    assert not np.array_equal(idx, relative)  # the two rules differ in every case here


@pytest.mark.parametrize("base_offset", range(1, 8))
def test_a_nonzero_base_offset_is_refused(base_offset):
    with pytest.raises(warplens.UnsupportedError, match="non-zero base offset.*outside the v1 scope"):
        _operand_bytes(WARP, MMA, descriptor(128, 128, 2, base_offset), 128, np.dtype(np.float16), "A")


def test_a_single_try_wait_on_an_open_phase_blocks_and_is_a_deadlock():
    """No non-blocking fallback in v1: on hardware the kernel below is expected to store 7
    after try_wait returned false; the simulator reports the wait instead (matrix row MB-11)."""
    mod = kernel("""
        .reg .b32 %bar, %v; .reg .b64 %out; .reg .pred %p;
        .shared .align 8 .u64 bar;
        ld.param.u64 %out, [out];
        mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        selp.u32 %v, 1, 7, %p;
        st.global.u32 [%out], %v;
        ret;""", params=".param .u64 out")
    sim = warplens.Device()
    out = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    with pytest.raises(warplens.DeadlockError, match="mbarrier.try_wait"):
        sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy()[0] == 0


def test_a_try_wait_on_a_completed_phase_returns_true_without_blocking():
    mod = kernel("""
        .reg .b32 %bar, %v; .reg .b64 %out, %token; .reg .pred %p;
        .shared .align 8 .u64 bar;
        ld.param.u64 %out, [out];
        mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.shared::cta.b64 %token, [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        selp.u32 %v, 1, 7, %p;
        st.global.u32 [%out], %v;
        mbarrier.try_wait.shared::cta.b64 %p, [%bar], %token;
        selp.u32 %v, 1, 7, %p;
        st.global.u32 [%out+4], %v;
        ret;""", params=".param .u64 out")
    sim = warplens.Device()
    out = sim.from_numpy(np.zeros(2, dtype=np.uint32))
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [1, 1]


def test_tcgen05_cp_is_refused_by_name():
    mod = kernel(".reg .b32 %a; .reg .b64 %d;\nmov.b32 %a, 0;\nmov.b64 %d, 0;\n"
                 "tcgen05.cp.cta_group::1.128x256b [%a], %d;\nret;")
    with pytest.raises(warplens.UnsupportedError, match="tcgen05.cp"):
        warplens.Device().launch(mod, "k", grid=1, block=32)


def test_elect_by_a_lane_outside_its_member_mask_ends_the_kernel():
    """Measured on a B200 (validation/blackwell/measured/b200/e2e/elect.all.txt): the mask
    0xFFFF0000 executed by all 32 lanes is cudaErrorIllegalInstruction. A mask naming lanes
    that do not execute it elects the lowest lane that does, as predicted and as measured."""
    body = (".reg .b32 %l, %t; .reg .pred %p, %q; .reg .b64 %o, %i;\n"
            "ld.param.u64 %o, [out];\nmov.u32 %t, %tid.x;\n{guard}"
            "elect.sync %l|%p, {mask};\n"
            "cvt.u64.u32 %i, %t;\nshl.b64 %i, %i, 2;\nadd.u64 %o, %o, %i;\nst.global.u32 [%o], %l;\n"
            "done:\nret;")
    sim = warplens.Device()
    with pytest.raises(warplens.ExecutionError, match=r"lane\(s\) 0, 1, .*0xffff0000 does not name.*IllegalInstruction"):
        sim.launch(kernel(body.format(guard="", mask="0xffff0000"), ".param .u64 out"), "k", grid=1, block=32,
                   args=[sim.empty(32, "u32")])
    out = sim.from_numpy(np.full(32, 99, dtype=np.uint32))
    odd = "and.b32 %l, %t, 1;\nsetp.eq.u32 %q, %l, 0;\n@%q bra done;\n"
    sim.launch(kernel(body.format(guard=odd, mask="0xffffffff"), ".param .u64 out"), "k", grid=1, block=32, args=[out])
    assert out.numpy().tolist() == [99, 1] * 16
