"""Regressions for M1 shared-memory reporting."""

from conftest import kernel


def test_same_line_load_and_store_have_separate_bank_counters(sim):
    mod = kernel("""
        .reg .b32 %v;
        .shared .b32 cell;
        st.shared.b32 [cell], 1; ld.shared.b32 %v, [cell];
        ret;
    """)
    stats = sim.launch(mod, "k", grid=3, block=32)

    store, load = stats.smem.instructions.values()
    assert store.line == load.line
    assert (store.op, load.op) == ("st", "ld")
    assert store.executions == load.executions == 3
    assert store.wavefronts == load.wavefronts == 3
    assert stats.smem.conflicts == 0
    assert not stats.findings


def test_identical_same_line_loads_have_separate_findings(sim):
    mod = kernel("""
        .reg .b32 %v;
        .shared .b32 cell;
        ld.shared.b32 %v, [cell]; ld.shared.b32 %v, [cell];
        ret;
    """)
    stats = sim.launch(mod, "k", grid=3, block=32)

    first, second = stats.findings
    assert first.line == second.line
    assert first.kind == second.kind == "smem-uninitialized-read"
    assert first.count == second.count == 3
    assert first.lanes == second.lanes == list(range(32))
    assert len(stats.smem.instructions) == 2
