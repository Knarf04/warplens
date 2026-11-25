"""`elect.sync` when its member mask and the lanes that execute it disagree, as measured on a B200.

Part of the frozen v1 instruction scope (docs/history/plan.md section 3.5): what is in is pinned, what
is out is refused by name, never run as something else."""

import numpy as np
import pytest

import warplens

from tests.support import kernel


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
