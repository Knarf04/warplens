"""The active-SM assumption of the timing predictions.

`warplens/timing/placement.py` turns a launch shape into the number of SMs the launch keeps
busy; `validation/blackwell_timing/predict.py` times every launch of a benchmark with that
number and writes it next to the prediction. Nothing here needs a GPU or nvcc: the launches
are of a PTX kernel written below, and the committed predictions are read as they are.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import warplens
from warplens.host import device as host_device
from warplens.timing.placement import LIMITS, place

from conftest import kernel

SUITE = Path(__file__).resolve().parent.parent / "validation" / "blackwell_timing"
PREDICTED = SUITE / "predicted"


def load(name: str):
    if str(SUITE) not in sys.path:
        sys.path.insert(0, str(SUITE))
    spec = importlib.util.spec_from_file_location(f"blackwell_timing_{name}", SUITE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load("fit")
predict = load("predict")

# What each benchmark's launch shape implies on a B200 (148 SMs), configuration by
# configuration: {prefix of the id: (active SMs, CTAs per SM)}. The longest prefix wins.
# This is the table of validation/blackwell_timing/README.md ("Active SMs").
EXPECTED = {
    "launch/empty/ctas=1": (1, 32), "launch/empty/ctas=allsm": (148, 32),
    "alu_chain/warps=32": (1, 2), "alu_chain/warps=16": (1, 4), "alu_chain/warps=8": (1, 8),
    "alu_chain/warps=5": (1, 12), "alu_chain/warps=4": (1, 16), "alu_chain/warps=2": (1, 32),
    "alu_chain/warps=1": (1, 32),
    "smem_wavefronts/": (1, None), "global_load/": (1, 32), "cp_async/": (1, 32),
    "mbar/": (1, 32), "tmem_access/": (1, 32),
    "gstream/triad": (148, 8),
    "tma/": (1, 7), "tma/load/cold/bytes=16384/ctas=8": (8, 7), "tma/load/cold/bytes=16384/ctas=37": (37, 7),
    "tma/load/cold/bytes=16384/ctas=allsm": (148, 7),
    "mma_rate/": (1, 7), "mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm": (148, 7),
    "mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm": (148, 7),
    "mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm": (148, 7),
    "mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm": (148, 7),
    "mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm": (148, 7),
    "mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm": (148, 7),
    "mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm": (148, 7),
    "mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm": (148, 7),
    "clc_latency/fail/plain/ctas=1": (1, 32), "clc_latency/fail/plain/ctas=2": (2, 32),
    "clc_latency/fail/multicast/ctas=2": (2, 32),
    "clc_latency/succeed/": (148, 2),
    "cluster_costs/load/multicast/ctas=1": (1, 14), "cluster_costs/load/multicast/ctas=2": (2, 14),
    "cluster_costs/load/multicast/ctas=4": (4, 14), "cluster_costs/load/separate/ctas=1": (1, 14),
    "cluster_costs/load/separate/ctas=2": (2, 14), "cluster_costs/load/separate/ctas=4": (4, 14),
    "cluster_costs/mma/": (2, 7),
    "pipelined_gemm/stages=1/ctas=8": (8, 2), "pipelined_gemm/stages=2/ctas=8": (8, 2),
    "pipelined_gemm/stages=4/ctas=8": (8, 2), "pipelined_gemm/stages=1/ctas=allsm": (148, 2),
    "pipelined_gemm/stages=2/ctas=allsm": (148, 2), "pipelined_gemm/stages=4/ctas=allsm": (148, 2),
    "ncu_banks/": (64, None),
}


def expected(ident: str):
    return EXPECTED[max((prefix for prefix in EXPECTED if ident.startswith(prefix)), key=len)]


# -- the rule ------------------------------------------------------------------------------

@pytest.mark.parametrize("shape, active, per_sm, limit, resident, packed", [
    # one CTA of one warp: one SM, whatever an SM could hold
    (dict(grid_ctas=1), 1, 32, "CTA count", 1, 1),
    # fewer CTAs than SMs: one SM each
    (dict(grid_ctas=37, smem_bytes=32 * 1024), 37, 7, "shared memory", 37, 6),
    # the pipelined GEMM on 8 CTAs: two would fit on an SM, but they are spread
    (dict(grid_ctas=8, block_threads=128, smem_bytes=98380), 8, 2, "shared memory", 8, 4),
    # one CTA per SM
    (dict(grid_ctas=148, block_threads=128, smem_bytes=98380), 148, 2, "shared memory", 148, 74),
    # more CTAs than fit: every SM full, the rest wait
    (dict(grid_ctas=1 << 19, block_threads=256), 148, 8, "threads", 148 * 8, 148),
    (dict(grid_ctas=64, block_threads=1024), 64, 2, "threads", 64, 32),
    # clc_latency/succeed: 100 KB per CTA, so two CTAs per SM and 296 of the 20002 at once
    (dict(grid_ctas=20002, cluster_ctas=2, smem_bytes=102432), 148, 2, "shared memory", 296, 148),
    # only whole clusters start: 9 clusters of 16 on 148 SMs that hold one CTA each
    (dict(grid_ctas=320, cluster_ctas=16, smem_bytes=227 * 1024), 144, 1, "shared memory", 144, 144),
    # a cluster's CTAs are on different SMs even when one SM could hold them all
    (dict(grid_ctas=4, cluster_ctas=4), 4, 32, "CTA count", 4, 4),
])
def test_placement_of_a_launch_shape(shape, active, per_sm, limit, resident, packed):
    placed = place(sm_count=148, **shape)
    assert (placed.active_sms, placed.ctas_per_sm, placed.limited_by, placed.resident_ctas, placed.packed_sms) \
        == (active, per_sm, limit, resident, packed)
    assert f"on {active} of 148 SMs" in placed.derivation
    assert placed.as_dict()["active_sms"] == active


def test_placement_uses_the_device_it_is_given():
    assert place(200, sm_count=160).active_sms == 160      # a B300
    assert place(200, sm_count=58).active_sms == 58
    # Limits a GPU reports replace the built-in ones.
    assert place(1000, sm_count=4, block_threads=256).ctas_per_sm == 8
    assert place(1000, sm_count=4, block_threads=256, limits={"max_threads_per_sm": 1536}).ctas_per_sm == 6
    assert LIMITS == {"smem_per_sm_bytes": 228 * 1024, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32}


@pytest.mark.parametrize("shape, message", [
    (dict(grid_ctas=0), "at least one CTA"),
    (dict(grid_ctas=6, cluster_ctas=4), "whole number of clusters"),
    (dict(grid_ctas=320, cluster_ctas=160), "does not fit on 148 SMs"),
    (dict(grid_ctas=1, smem_bytes=229 * 1024), "does not fit on an SM"),
])
def test_placement_refuses_shapes_that_cannot_run(shape, message):
    with pytest.raises(ValueError, match=message):
        place(sm_count=148, **shape)


# -- predict.py applies it per launch --------------------------------------------------------

@pytest.fixture
def placed_launches(monkeypatch):
    """predict.install_placement for the duration of a test: (launch, placed)."""
    def install(mode="launch"):
        monkeypatch.setattr(host_device, "run_kernel", host_device.run_kernel)  # restored afterwards
        placed = predict.install_placement(mode)
        sim = warplens.Device("B200")
        return (lambda **shape: sim.launch(kernel("ret;"), "k", **shape)), placed
    return install


def test_each_launch_is_timed_with_the_active_sms_of_its_own_shape(placed_launches):
    launch, placed = placed_launches()
    timed = []
    for shape, active, resident in [
        (dict(grid=1, block=32), 1, 1),
        (dict(grid=8, block=128, smem=96 * 1024), 8, 8),
        (dict(grid=(37, 1, 1), block=32), 37, 37),
        (dict(grid=148, block=32), 148, 148),
        (dict(grid=600, block=(32, 32), smem=64), 148, 296),        # 1024 threads: two CTAs per SM
        (dict(grid=6, block=32, cluster=2), 6, 6),
    ]:
        stats = launch(timing=True, **shape)
        timed.append(stats)  # kept alive: `placed` is keyed by id(), which a freed object's successor may reuse
        assert stats.timing.active_sms == active, shape
        assert "SMs assumed to share HBM" in stats.timing.summary() and f"; {active} SMs" in stats.timing.summary()
        record = placed[id(stats)]
        assert (record["active_sms"], record["resident_ctas"], record["sim_grid_ctas"]) == \
            (active, resident, stats.ctas), shape
    # A launch without the timing model is left alone and not recorded.
    untimed = launch(grid=4, block=32)
    assert untimed.timing is None and id(untimed) not in placed


def test_the_old_assumption_is_still_available(placed_launches):
    launch, placed = placed_launches("all-sms")
    stats = launch(grid=8, block=32, timing=True)
    assert stats.timing.active_sms == 148          # every SM, as before
    assert placed[id(stats)]["active_sms"] == 8    # what its shape implies is still recorded
    with pytest.raises(ValueError, match="placement must be one of"):
        predict.install_placement("some")
    with pytest.raises(ValueError, match="placement must be one of"):
        predict.predict(Path("unused"), placement="some")


def test_the_hardware_grid_is_used_where_the_simulator_build_launches_a_smaller_one(placed_launches):
    launch, placed = placed_launches()
    sim = warplens.Device("B200")
    for name, grid, cluster, hardware in [("_Z12triad_kernelPmPdS0_S0_d", 8, None, 1 << 19),
                                          ("_Z18clc_succeed_kernelPmi", 34, 2, 20002)]:
        stats = sim.launch(kernel("ret;", name=name), name, grid=grid, block=32, cluster=cluster, timing=True)
        record = placed[id(stats)]
        assert stats.timing.active_sms == 148 and stats.ctas == grid
        assert (record["grid_ctas"], record["sim_grid_ctas"]) == (hardware, grid)


def test_a_launch_the_simulator_refuses_is_refused_as_before(placed_launches):
    launch, placed = placed_launches()
    with pytest.raises(warplens.LaunchError, match="whole number of clusters"):
        launch(grid=5, block=32, cluster=2, timing=True)
    assert not placed


def test_prediction_lines_and_metadata_carry_the_assumption():
    sim = {"time_ns": 1.0, "instructions": 2, "bound": "latency", "findings": 0, "active_sms": 8,
           "placement": {**place(8, sm_count=148, block_threads=128, smem_bytes=98380).as_dict(),
                         "sim_grid_ctas": 8, "derivation": "text"}}
    assert predict.placement_fields(sim) == " sim_active_sms=8 sim_ctas_per_sm=2 sim_resident_ctas=8"
    entry = predict.assumption(sim)
    assert (entry["active_sms"], entry["spread_sms"], entry["packed_sms"], entry["limited_by"]) == \
        (8, 8, 4, "shared memory")
    assert "sm_count" not in entry
    # A launch record that does not say (written by another runner): no fields, no failure.
    bare = {"time_ns": 1.0, "instructions": 2, "bound": "latency", "findings": 0}
    assert predict.placement_fields(bare) == "" and predict.assumption(bare) == {"active_sms": None}
    # fit.py reads the fields as numbers, and its tables show them.
    row = fit.parse_line("BENCH id=x reps=1" + predict.placement_fields(sim))
    assert {k: row[k] for k in ("sim_active_sms", "sim_ctas_per_sm", "sim_resident_ctas")} == \
        {"sim_active_sms": "8", "sim_ctas_per_sm": "2", "sim_resident_ctas": "8"}
    assert {"sim_active_sms", "sim_ctas_per_sm", "sim_resident_ctas"} <= set(fit.NUMERIC)


# -- the committed predictions ---------------------------------------------------------------

def test_committed_predictions_state_the_active_sms_of_every_configuration():
    meta = json.loads((PREDICTED / "meta.json").read_text())
    assert meta["placement"]["mode"] == "launch" and meta["placement"]["sm_count"] == 148
    assert meta["placement"]["limits"] == LIMITS
    assert "active_sms" not in meta, "one number for every benchmark is what this replaced"
    rows = fit.read_dir(PREDICTED)[1]
    seen = set()
    for row in rows:
        active, per_sm = expected(row["id"])
        assert row["sim_active_sms"] == active, row["id"]
        if per_sm is not None:
            assert row["sim_ctas_per_sm"] == per_sm, row["id"]
        entry = meta["benchmarks"][row["file"]]["active_sms"][row["id"]]
        assert entry["active_sms"] == active == entry["spread_sms"], row["id"]
        assert "active_sms_of_other_launches" not in entry, row["id"]
        assert entry["resident_ctas"] == row["sim_resident_ctas"] and entry["derivation"]
        seen.add(row["id"])
    assert len(seen) > 100
    assert {b for b in meta["benchmarks"]} == {row["file"] for row in rows}
    text = (PREDICTED / "predictions.md").read_text()
    assert "SMs active (assumed)" in text and "148 SMs assumed to share HBM" not in text
    assert "| tma/load/cold/bytes=16384/ctas=37 | " in text
    line = next(l for l in text.splitlines() if l.startswith("| pipelined_gemm/stages=4/ctas=8 |"))
    assert line.split("|")[5].strip() == "8"


def test_active_sms_change_the_predictions_that_draw_on_hbm():
    """One SM of 148 gets 28 GB/s of HBM; one of 8 gets 518, more than its TMA engine moves."""
    table = fit.summarize(fit.read_dir(PREDICTED)[1], "sim_ns")
    cold = {n: table[f"tma/load/cold/bytes=16384/ctas={n}"]["ns_per_rep"] for n in ("1", "8", "37", "allsm")}
    warm = table["tma/load/warm/bytes=16384/ctas=1"]["ns_per_rep"]
    assert cold["1"] == pytest.approx(warm) and cold["8"] == pytest.approx(warm, abs=1)
    assert cold["8"] < cold["37"] < cold["allsm"]
    # 16384 bytes at 4141 / 37 and 4141 / 148 GB/s, plus the 100 ns of tma_latency_ns.
    assert cold["37"] - 100 == pytest.approx(16384 / (4141 / 37), rel=0.05)
    assert cold["allsm"] - 100 == pytest.approx(16384 / (4141 / 148), rel=0.05)
    gemm = {n: [table[f"pipelined_gemm/stages={s}/ctas={n}"]["ns_per_rep"] for s in (1, 2, 4)] for n in ("8", "allsm")}
    assert all(few < all_sms for few, all_sms in zip(gemm["8"], gemm["allsm"]))
    for times in gemm.values():  # TIM-18: more stages never make it slower
        assert times[0] >= times[1] >= times[2]
