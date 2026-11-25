"""Seeded and adversarial scheduling: the order in which runnable warps are stepped.

A kernel that is correct computes the same under every schedule. One that relies on an
order nothing guarantees gives results that change with the seed.
"""

import numpy as np
import pytest

import warplens
from warplens import cli
from warplens.core.engine import ADVERSARIAL_QUANTUM, SCHEDULE_VARIABLE, Engine, Schedule

from tests.support import EXAMPLES, kernel

SCHEDULES = [None, 0, 1, 2, 12345, "adversarial", "adversarial:1", "adversarial:2"]

# Every warp adds its number to a shared cell: unsynchronized (load, add, store), or atomically.
COUNTER = """
    .reg .u32 %t, %v, %w; .reg .u64 %o; .reg .pred %first, %lane0; .shared .u32 cell;
    mov.u32 %t, %tid.x; setp.eq.u32 %first, %t, 0;
    mov.u32 %w, %laneid; setp.eq.u32 %lane0, %w, 0;
    mov.u32 %w, %warpid; add.u32 %w, %w, 1;
    @%first st.shared.u32 [cell], 0;
    bar.sync 0;
    {update}
    bar.sync 0;
    ld.shared.u32 %v, [cell];
    ld.param.u64 %o, [out];
    @%first st.global.u32 [%o], %v;
    ret;
"""
RACY = "ld.shared.u32 %v, [cell]; add.u32 %v, %v, %w; st.shared.u32 [cell], %v;"
ATOMIC = "@%lane0 atom.shared.add.u32 %v, [cell], %w;"

# Every thread writes its slot and reads the slot of the thread 32 further on: another warp's.
NEIGHBOUR = """
    .reg .u32 %t, %n, %v; .reg .u64 %o, %a, %base; .shared .align 4 .b8 buf[256];
    mov.u32 %t, %tid.x;
    mov.u64 %base, buf;
    mul.wide.u32 %a, %t, 4; add.u64 %a, %a, %base;
    add.u32 %v, %t, 100;
    st.shared.u32 [%a], %v;
    {barrier}
    add.u32 %n, %t, 32; rem.u32 %n, %n, 64;
    mul.wide.u32 %a, %n, 4; add.u64 %a, %a, %base;
    ld.shared.u32 %v, [%a];
    ld.param.u64 %o, [out];
    mul.wide.u32 %a, %t, 4; add.u64 %o, %o, %a;
    st.global.u32 [%o], %v;
    ret;
"""


def count(sim, update: str, schedule, warps: int = 8):
    out = sim.empty(1, "u32")
    stats = sim.launch(kernel(COUNTER.format(update=update), ".param .u64 out"), "k", grid=1,
                       block=32 * warps, args=[out], schedule=schedule)
    return int(out.numpy()[0]), sorted({f.kind for f in stats.findings})


def neighbours(sim, barrier: str, schedule):
    out = sim.empty(64, "u32")
    stats = sim.launch(kernel(NEIGHBOUR.format(barrier=barrier), ".param .u64 out"), "k", grid=1,
                       block=64, args=[out], schedule=schedule)
    return out.numpy().tolist(), sorted({f.kind for f in stats.findings})


# -- what a schedule is ------------------------------------------------------------------------

@pytest.mark.parametrize("spec, expect", [
    (None, Schedule()), ("round-robin", Schedule()), ("", Schedule()),
    (7, Schedule(seed=7)), ("7", Schedule(seed=7)), ("seed:7", Schedule(seed=7)), (0, Schedule(seed=0)),
    ("adversarial", Schedule(adversarial=True)), ("Adversarial:3", Schedule(seed=3, adversarial=True)),
    (Schedule(seed=2), Schedule(seed=2)),
])
def test_schedule_spellings(spec, expect):
    assert Schedule.parse(spec) == expect
    assert Schedule.parse(str(expect)) == expect  # what --schedule puts in the environment


@pytest.mark.parametrize("spec", [-1, True, 1.5, "random", "seed", "seed:x", "adversarial:", "7:7", "-3"])
def test_bad_schedule_is_refused(sim, spec):
    with pytest.raises(warplens.LaunchError, match="schedule"):
        count(sim, ATOMIC, spec)


def test_seeded_order_is_a_permutation_and_repeats_for_the_seed():
    warps = list(range(13))
    first = [Engine(schedule=Schedule(seed=5))._order(warps) for _ in range(2)]
    assert first[0] == first[1] != warps and sorted(first[0]) == warps
    engine = Engine(schedule=Schedule(seed=5))
    assert engine._order(warps) != engine._order(warps)  # drawn anew every round
    assert Engine(schedule=Schedule(seed=6))._order(warps) != first[0]
    assert Engine()._order(warps) is warps
    assert Engine(schedule=Schedule(adversarial=True))._order(warps) == warps[::-1]


def test_seeded_order_does_not_depend_on_the_python_version():
    """Only Random.random() is used, whose sequence for a seed is fixed."""
    assert Engine(schedule=Schedule(seed=1))._order(list(range(8))) == [0, 2, 3, 6, 7, 4, 5, 1]


# -- a racy kernel shows, a correct one does not ---------------------------------------------

def test_racy_kernel_gives_different_results_under_different_seeds(sim):
    results = {seed: count(sim, RACY, seed) for seed in range(8)}
    assert len({value for value, _ in results.values()}) > 1
    assert all(findings == ["smem-race"] for _, findings in results.values())
    for seed, result in results.items():  # the same seed is the same run
        assert count(sim, RACY, seed) == result
        assert count(sim, RACY, f"seed:{seed}") == result


def test_default_schedule_is_round_robin_as_before(sim):
    # All eight warps load 0 in one round and store in a later one; warp 7 is stepped last.
    assert count(sim, RACY, None) == (8, ["smem-race"])
    assert count(sim, RACY, "round-robin") == (8, ["smem-race"])
    # Adversarial: last warp first, each running to the barrier, so the updates do not overlap.
    assert count(sim, RACY, "adversarial") == (36, ["smem-race"])


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_correct_kernel_is_schedule_invariant(sim, schedule):
    assert count(sim, ATOMIC, schedule) == (36, [])


def test_adversarial_schedule_exposes_a_missing_barrier(sim):
    """Round-robin steps the two warps together, so each finds the other's slots written and
    the kernel looks right. Run one warp ahead and it reads slots nobody has written."""
    right = [100 + (t + 32) % 64 for t in range(64)]
    values, findings = neighbours(sim, "", None)
    assert values == right and findings == ["smem-race"]
    values, findings = neighbours(sim, "", "adversarial")
    assert values[32:] == [0] * 32 and values[:32] == right[:32]  # warp 1 ran first
    assert "smem-race" in findings
    assert any(neighbours(sim, "", f"adversarial:{seed}")[0] != right for seed in range(4))
    for schedule in SCHEDULES:  # with the barrier, every schedule agrees
        assert neighbours(sim, "bar.sync 0;", schedule) == (right, [])


@pytest.mark.parametrize("schedule", SCHEDULES[1:])
def test_examples_compute_the_same_under_every_schedule(sim, schedule):
    data = np.random.default_rng(0).standard_normal((2, 32, 32)).astype(np.float32)
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", "transpose_padded", grid=2,
                       block=(32, 32), args=[sim.from_numpy(data), out], schedule=schedule)
    np.testing.assert_array_equal(out.numpy(), data.transpose(0, 2, 1))
    assert not stats.findings

    x = np.random.default_rng(1).integers(-1000, 1000, (3, 256)).astype(np.int32)
    out = sim.empty(3, "s32")
    stats = sim.launch(EXAMPLES / "01_warp_reduce" / "reduce.ptx", "block_sum", grid=3, block=256,
                       smem=8 * 4, args=[sim.from_numpy(x), out], schedule=schedule)
    np.testing.assert_array_equal(out.numpy(), x.sum(axis=1))
    assert not stats.findings


# -- the engine under an adversarial schedule --------------------------------------------------

def test_adversarial_turn_ends_at_the_quantum_and_counts_every_instruction(sim):
    spin = kernel(""".reg .u32 %i; .reg .pred %p; mov.u32 %i, 0;
        loop: add.u32 %i, %i, 1; setp.lt.u32 %p, %i, 1000; @%p bra loop; ret;""")
    plain = sim.launch(spin, "k", grid=1, block=64)
    order = []
    stats = sim.launch(spin, "k", grid=1, block=64, schedule="adversarial",
                       trace=lambda w, pc, ins, mask: order.append(w.warp_id))
    assert stats.instructions == plain.instructions == len(order)
    turns = [order[i:i + ADVERSARIAL_QUANTUM] for i in range(0, len(order), ADVERSARIAL_QUANTUM)]
    assert turns[0] == [1] * ADVERSARIAL_QUANTUM and turns[1] == [0] * ADVERSARIAL_QUANTUM
    with pytest.raises(warplens.ExecutionError, match="max_steps=500"):
        sim.launch(spin, "k", grid=1, block=64, schedule="adversarial", max_steps=500)


def test_clock_advances_within_an_adversarial_turn(sim):
    out = sim.empty(2, "u64")
    mod = kernel(""".reg .u64 %a, %b, %o; .reg .pred %first; .reg .u32 %t;
        mov.u32 %t, %tid.x; setp.eq.u32 %first, %t, 0;
        mov.u64 %a, %clock64; mov.u64 %b, %clock64;
        ld.param.u64 %o, [out];
        @%first st.global.u64 [%o], %a; @%first st.global.u64 [%o+8], %b; ret;""", ".param .u64 out")
    sim.launch(mod, "k", grid=1, block=64, args=[out], schedule="adversarial")
    first, second = out.numpy().tolist()
    assert second == first + 1


def test_deadlock_is_still_found(sim):
    mod = kernel(".reg .pred %w0; .reg .u32 %w; mov.u32 %w, %warpid; setp.eq.u32 %w0, %w, 0;"
                 " @%w0 bar.sync 1, 64; ret;")
    for schedule in SCHEDULES:
        with pytest.raises(warplens.DeadlockError, match="barrier 1"):
            sim.launch(mod, "k", grid=1, block=64, schedule=schedule)


@pytest.mark.parametrize("schedule", SCHEDULES[1:])
def test_tma_round_trip_is_schedule_invariant(schedule):
    """Asynchronous copies complete later under an adversarial schedule, which a kernel that
    waits for them does not notice."""
    from tests.support.tma import NO_WAIT, tma_copy
    ok, stats = tma_copy("128B", (16, 32), schedule=schedule)
    assert ok and not stats.findings
    ok, stats = tma_copy("128B", (16, 32), NO_WAIT, schedule=schedule)  # and one that does not
    assert "tma-load-unsynchronized" in {f.kind for f in stats.findings}


# -- where a schedule comes from ---------------------------------------------------------------

def test_environment_sets_the_schedule_of_launches_that_name_none(sim, monkeypatch):
    plain = count(sim, RACY, None)
    seeded = next(seed for seed in range(8) if count(sim, RACY, seed) != plain)
    monkeypatch.setenv(SCHEDULE_VARIABLE, str(seeded))
    assert count(sim, RACY, None) == count(sim, RACY, seeded) != plain
    assert count(sim, RACY, "round-robin") == plain  # the argument wins
    monkeypatch.setenv(SCHEDULE_VARIABLE, "sideways")
    with pytest.raises(warplens.LaunchError, match="WARPLENS_SCHEDULE.*'sideways'"):
        count(sim, RACY, None)


def test_timed_launch_refuses_a_schedule_and_ignores_the_environment(sim, monkeypatch):
    vecadd = EXAMPLES / "00_vecadd" / "vecadd.ptx"
    a = sim.from_numpy(np.ones(64, dtype=np.float32))
    args = dict(grid=1, block=64, args=[a, a, sim.empty(64, "f32"), 64], timing=True)
    with pytest.raises(warplens.LaunchError, match="untimed"):
        sim.launch(vecadd, "vecadd", schedule=3, **args)
    monkeypatch.setenv(SCHEDULE_VARIABLE, "adversarial")
    assert sim.launch(vecadd, "vecadd", **args).timing is not None


def test_cli_schedule_reaches_the_run_through_the_environment(monkeypatch, capsys):
    import os
    seen = []
    monkeypatch.setattr(cli, "run", lambda args: seen.append(os.environ.get(SCHEDULE_VARIABLE)) or 0)
    monkeypatch.delenv(SCHEDULE_VARIABLE, raising=False)
    assert cli.main(["run", "--schedule", "Adversarial:4", "program"]) == 0
    assert cli.main(["run", "--schedule", "9", "program"]) == 0
    assert cli.main(["run", "program"]) == 0
    assert seen == ["adversarial:4", "seed:9", None]
    assert SCHEDULE_VARIABLE not in os.environ  # restored after the run
    for arguments, message in ((["--schedule", "fastest"], "schedule must be"),
                               (["--schedule", "3", "--timing"], "not allowed with the timing model")):
        with pytest.raises(SystemExit):
            cli.main(["run", *arguments, "program"])
        assert message in capsys.readouterr().err
