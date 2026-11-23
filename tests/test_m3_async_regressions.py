"""Regression tests for the thread scope and lifetime of tcgen05 completion."""

from types import SimpleNamespace

import numpy as np
import pytest

import warplens
from warplens.check.report import Findings
from warplens.core.cta import Cta
from warplens.frontend.ir import Imm, Instr, Mem, Reg
from warplens.units.smem import GENERIC_WINDOW
from warplens.units.tcgen05 import _tcgen05


def commit_context():
    cta = Cta((0, 0, 0), 32, None, Findings(), async_delay=10)
    w = SimpleNamespace(cta=cta, warp_id=0, seen=set(), tcgen_issued=[],
                        program=SimpleNamespace(fn=SimpleNamespace(name="k")))
    w.read_address = lambda operand: np.full(32, operand.value, dtype=np.uint64)
    for offset in (0, 8):
        cta.mbarrier_init(offset, 1, w, None)
    for lane in (0, 1):
        ins = Instr("tcgen05", ("mma",), (), None, False, 2, "tcgen05.mma")
        operation = cta.new_async("mma", w, ins, lane)
        w.tcgen_issued.append(operation)
        cta.schedule(operation, lambda: None)
    return cta, w


def issue_commit(w, lane, offset=0, *, generic=False, extra=()):
    mods = ("commit", "cta_group::1", "mbarrier::arrive::one", "b64", *extra)
    if not generic:
        mods += ("shared::cluster",)
    ins = Instr("tcgen05", mods, (Mem(Imm("int", offset)),), None, False, 1, "tcgen05.commit")
    mask = np.arange(32) == lane
    _tcgen05(w, ins, mask)


def drain(cta):
    while cta.complete_next_async():
        pass


def test_commit_only_tracks_the_executing_threads_mmas():
    cta, w = commit_context()
    issue_commit(w, 1)
    drain(cta)
    assert cta.mbarriers[0].completed_knowledge == {w.tcgen_issued[1].id}


def test_two_commits_can_track_the_same_prior_mma_on_different_barriers():
    cta, w = commit_context()
    issue_commit(w, 0)
    issue_commit(w, 0, 8)
    drain(cta)
    assert cta.mbarriers[0].completed_knowledge == {w.tcgen_issued[0].id}
    assert cta.mbarriers[8].completed_knowledge == {w.tcgen_issued[0].id}


def test_commit_knowledge_is_captured_at_issue_not_completion():
    cta, w = commit_context()
    w.seen.add(100)
    issue_commit(w, 0)
    w.seen.add(101)
    drain(cta)
    assert cta.mbarriers[0].completed_knowledge == {w.tcgen_issued[0].id, 100}


def test_commit_accepts_generic_shared_address():
    cta, w = commit_context()
    issue_commit(w, 0, GENERIC_WINDOW + 8, generic=True)
    drain(cta)
    assert cta.mbarriers[8].completed_knowledge == {w.tcgen_issued[0].id}


def test_commit_rejects_generic_address_outside_shared_memory():
    cta, w = commit_context()
    with pytest.raises(warplens.ExecutionError, match="does not point to shared"):
        issue_commit(w, 0, 8, generic=True)


def test_read_only_commit_is_not_silently_treated_as_full_mma_completion():
    cta, w = commit_context()
    with pytest.raises(warplens.UnsupportedError):
        issue_commit(w, 0, extra=("sync_restrict::shared::read::mma::a",))


def test_another_lane_cannot_make_mma_result_synchronized(sim, monkeypatch):
    import test_tcgen05 as helpers

    original = helpers.mma_kernel

    def wrong_committer(kind):
        module = original(kind)
        for ins in module.functions["k"].instrs:
            if ins.opcode == "setp" and ins.operands[0] == Reg("%yes"):
                ins.operands = (ins.operands[0], ins.operands[1], Imm("int", 1))
            if ins.opcode == "tcgen05" and ins.mods[0] == "commit":
                ins.pred = Reg("%yes")
        return module

    monkeypatch.setattr(helpers, "mma_kernel", wrong_committer)
    _, _, stats = helpers.run(sim, n=8)
    assert "tmem-read-before-commit" in {f.kind for f in stats.findings}


def test_store_cannot_erase_unwaited_mma_dependency(sim, monkeypatch):
    import test_tcgen05 as helpers

    original = helpers.mma_kernel

    def store_instead_of_load(kind):
        module = original(kind)
        for ins in module.functions["k"].instrs:
            if ins.opcode == "tcgen05" and ins.mods[0] == "ld":
                ins.mods = ("st", *ins.mods[1:])
                ins.operands = tuple(reversed(ins.operands))
        return module

    monkeypatch.setattr(helpers, "mma_kernel", store_instead_of_load)
    _, _, stats = helpers.run(sim, n=8, wait=0)
    assert "tmem-write-before-commit" in {f.kind for f in stats.findings}


def test_freeing_unwaited_mma_destination_is_reported(sim):
    from test_tcgen05 import run

    _, _, stats = run(sim, n=8, wait=0)
    assert "tmem-dealloc-before-commit" in {f.kind for f in stats.findings}


def test_tma_overwriting_mma_operand_names_the_actual_reader():
    from warplens.units.tma import _load

    cta, w = commit_context()
    cta.smem.async_reader[:16] = w.tcgen_issued[0].id
    ins = Instr("cp", ("async", "bulk"), (), None, False, 3, "cp.async.bulk")
    _load(w, ins, 0, 0, np.arange(16), np.zeros(16, np.uint8), cta.mbarriers[0])
    (finding,) = cta.findings.items
    assert finding.kind == "mma-operand-overwritten"
    assert "tcgen05.mma" in finding.message and "tcgen05.commit" in finding.message


@pytest.mark.parametrize("fence", [False, True])
@pytest.mark.parametrize("phase", [0, 1])
def test_relaxed_wait_fence_acquires_only_tcgen_completion(sim, monkeypatch, fence, phase):
    import test_tcgen05 as helpers

    original = helpers.mma_kernel

    def relaxed_wait(kind):
        module = original(kind)
        fn = module.functions["k"]
        for pc, ins in enumerate(fn.instrs):
            if ins.name.startswith("mbarrier.try_wait") and ins.operands[-1] == Imm("int", phase):
                ins.mods += ("relaxed", "cta")
                if fence:
                    fn.instrs.insert(pc + 1, Instr("tcgen05", ("fence::after_thread_sync",), (),
                                                  None, False, ins.line, "tcgen05.fence::after_thread_sync;"))
                    fn.labels = {name: target + (target > pc) for name, target in fn.labels.items()}
                break
        return module

    monkeypatch.setattr(helpers, "mma_kernel", relaxed_wait)
    _, _, stats = helpers.run(sim, n=8)
    kinds = {f.kind for f in stats.findings}
    if phase == 0:
        # A tcgen05 fence must not make a relaxed TMA wait into an acquire.
        assert "tma-load-unsynchronized" in kinds
    elif fence:
        assert not kinds
    else:
        assert "tmem-read-before-commit" in kinds
