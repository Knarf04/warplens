"""PTX shuffle participation beyond the full-warp hardware golden cases."""

import numpy as np
import pytest

from tests.support import kernel


def _shuffle(sim, *, block=32, setup="", member="-1", source=0, mode="idx"):
    mod = kernel(f"""
        .reg .b32 %lane, %value, %member;
        .reg .pred %p;
        .reg .b64 %out, %offset;
        mov.u32 %lane, %laneid;
        add.u32 %value, %lane, 100;
        {setup}
        shfl.sync.{mode}.b32 %value, %value, {source}, 31, {member};
        ld.param.u64 %out, [out];
        mul.wide.u32 %offset, %lane, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %value;
        ret;
    """, ".param .u64 out")
    out = sim.empty(block, "u32")
    stats = sim.launch(mod, "k", grid=1, block=block, args=[out])
    return out.numpy(), stats


@pytest.mark.parametrize("block", [1, 16, 31])
def test_shuffle_full_mask_allows_unlaunched_lanes(sim, block):
    out, stats = _shuffle(sim, block=block)
    np.testing.assert_array_equal(out, np.full(block, 100, dtype=np.uint32))
    assert not stats.findings


def test_shuffle_full_mask_allows_exited_lanes(sim):
    out, stats = _shuffle(sim, setup="setp.ge.u32 %p, %lane, 16; @%p ret;")
    np.testing.assert_array_equal(out, [100] * 16 + [0] * 16)
    assert not stats.findings


_HALF_MASKS = """
    setp.ge.u32 %p, %lane, 16;
    selp.u32 %member, 0xffff0000, 0xffff, %p;
"""


def test_shuffle_allows_disjoint_groups_at_one_instruction(sim):
    out, stats = _shuffle(sim, setup=_HALF_MASKS, member="%member", mode="bfly", source=1)
    np.testing.assert_array_equal(out, [100 + (lane ^ 1) for lane in range(32)])
    assert not stats.findings


def test_shuffle_reports_source_outside_callers_group(sim):
    _, stats = _shuffle(sim, setup=_HALF_MASKS, member="%member", mode="bfly", source=16)
    (finding,) = stats.findings
    assert finding.kind == "shfl-inactive-source"
    assert finding.lanes == list(range(32))
    assert "outside the caller's member mask" in finding.message


def test_shuffle_reports_live_members_using_different_masks(sim):
    setup = "setp.ge.u32 %p, %lane, 16; selp.u32 %member, -1, 0xffff, %p;"
    _, stats = _shuffle(sim, setup=setup, member="%member")
    (finding,) = stats.findings
    assert finding.kind == "shfl-membermask-mismatch"
    assert finding.lanes == list(range(16, 32))


def test_shuffle_does_not_treat_predicated_off_members_as_exited(sim):
    mod = kernel("""
        .reg .b32 %lane, %value; .reg .pred %p;
        mov.u32 %lane, %laneid;
        setp.lt.u32 %p, %lane, 16;
        @%p shfl.sync.idx.b32 %value, %lane, 0, 31, -1;
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=32)
    (finding,) = stats.findings
    assert finding.kind == "shfl-membermask-mismatch"
    assert finding.lanes == list(range(16))


def test_shuffle_source_that_exited_is_still_undefined(sim):
    _, stats = _shuffle(sim, setup="setp.ge.u32 %p, %lane, 16; @%p ret;", source=16)
    (finding,) = stats.findings
    assert finding.kind == "shfl-inactive-source"
    assert finding.lanes == list(range(16))
