"""`vote`: the forms the simulator takes and refuses. What the instruction returns is held to
an RTX 3080 Ti in tests/test_hardware_golden.py (test_vote_matches_real_gpu)."""

import numpy as np
import pytest

import warplens

from conftest import kernel

PARAMS = ".param .u64 o, .param .u32 m"


def _run(sim, vote: str, block: int = 32, mask: int = 0xFFFFFFFF, before: str = ""):
    """Lane t has the predicate `t is odd`; out[t] is the result of `vote` (a predicate as 0/1)."""
    module = kernel(f"""
        .reg .b64 %po, %off; .reg .b32 %t, %bit, %m, %r; .reg .pred %p, %q, %early;
        ld.param.u64 %po, [o]; ld.param.u32 %m, [m];
        mov.u32 %t, %tid.x;
        and.b32 %bit, %t, 1;
        setp.ne.u32 %p, %bit, 0;
        {before}
        {vote}
        mul.wide.u32 %off, %t, 4;
        add.u64 %po, %po, %off;
        st.global.u32 [%po], %r;
    done:
        ret;
    """, PARAMS)
    out = sim.from_numpy(np.full(block, 0xEEEEEEEE, dtype=np.uint32))
    stats = sim.launch(module, "k", grid=1, block=block, args=[out, np.uint32(mask)])
    return out.numpy().tolist(), stats


@pytest.mark.parametrize("form, expect", [("all", 0), ("any", 1), ("uni", 0)])
def test_vote_predicates_with_and_without_sync(sim, form, expect):
    for vote in (f"vote.sync.{form}.pred %q, %p, %m;", f"vote.{form}.pred %q, %p;"):
        out, stats = _run(sim, vote + " selp.u32 %r, 1, 0, %q;")
        assert out == [expect] * 32 and not stats.findings


def test_ballot_and_a_negated_source(sim):
    assert _run(sim, "vote.sync.ballot.b32 %r, %p, %m;")[0] == [0xAAAAAAAA] * 32
    assert _run(sim, "vote.sync.ballot.b32 %r, !%p, %m;")[0] == [0x55555555] * 32
    assert _run(sim, "vote.ballot.b32 %r, !%p;", block=5)[0] == [0b10101] * 5
    # A lane that does not exist has no bit, and the mask may name it.
    assert _run(sim, "vote.sync.ballot.b32 %r, %p, %m;", block=5)[0] == [0b01010] * 5
    # Only the lanes of the mask vote.
    assert _run(sim, "vote.sync.ballot.b32 %r, %p, %m;", mask=0x0000FFFF)[0][:16] == [0xAAAA] * 16


def test_vote_sync_waits_for_lanes_that_return(sim):
    """Lanes 0 to 7 branch to the `ret`. The mask names them, so the voters wait until they are
    gone; none of them has a vote."""
    out, stats = _run(sim, "vote.sync.ballot.b32 %r, %p, %m;",
                      before="setp.lt.u32 %early, %t, 8; @%early bra done;")
    assert out == [0xEEEEEEEE] * 8 + [0xAAAAAA00] * 24 and not stats.findings


def test_a_mask_without_its_caller_is_reported(sim):
    out, stats = _run(sim, "vote.sync.ballot.b32 %r, %p, %m;", mask=0x0000FFFF)
    assert out == [0xAAAA] * 32
    assert [f.kind for f in stats.findings] == ["vote-membermask-mismatch"]


def test_vote_sync_that_can_never_be_complete_is_refused(sim):
    """Lanes 0 to 7 take another `vote.sync` whose mask names everyone, while the others wait
    for them at theirs: on an RTX 3080 Ti such a launch does not return."""
    with pytest.raises(warplens.UnsupportedError, match="names a lane that is on another path"):
        _run(sim, "vote.sync.all.pred %q, %p, %m; selp.u32 %r, 1, 0, %q;",
             before="setp.lt.u32 %early, %t, 8; @%early vote.sync.any.pred %q, %p, %m; @%early bra done;")


@pytest.mark.parametrize("vote, why", [
    ("vote.sync.all.b32 %r, %p, %m;", "expected vote"),
    ("vote.sync.ballot.pred %q, %p, %m;", "expected vote"),
    ("vote.sync.eq.pred %q, %p, %m;", "expected vote"),
    ("vote.all.pred %q, %p, %m;", "operand"),
    ("vote.sync.all.pred %q, %p;", "operand"),
])
def test_vote_forms_that_are_refused(sim, vote, why):
    with pytest.raises((warplens.UnsupportedError, warplens.ExecutionError), match=why):
        _run(sim, vote)


def test_negation_needs_a_predicate_register():
    with pytest.raises(warplens.PtxSyntaxError, match="`!` needs a .pred register"):
        kernel(".reg .b32 %r; .reg .pred %q; vote.ballot.b32 %r, !%r; ret;")
