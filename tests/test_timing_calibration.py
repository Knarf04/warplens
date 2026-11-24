"""validation/blackwell_timing: the fitting code, and the committed predictions.

Nothing here needs a GPU. The fit tests read a small committed SYNTHETIC results
directory (fixture_synthetic/, written by make_synthetic.py: the simulator run with
known, different parameters) and check that fit.py recovers those parameters. The
prediction test runs the fast benchmarks in the simulator and compares with predicted/;
it needs nvcc for sm_100a to build them and is skipped without it.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from warplens.timing.config import GpuConfig

SUITE = Path(__file__).resolve().parent.parent / "validation" / "blackwell_timing"
FIXTURE = SUITE / "fixture_synthetic"
PREDICTED = SUITE / "predicted"
# The benchmarks cheap enough to re-simulate in a test (mma_rate takes minutes).
FAST = ["launch", "alu_chain", "smem_wavefronts", "global_load", "cp_async", "mbar", "tma", "tmem_access"]


def load(name: str):
    if str(SUITE) not in sys.path:
        sys.path.insert(0, str(SUITE))
    spec = importlib.util.spec_from_file_location(f"blackwell_timing_{name}", SUITE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load("fit")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    return fit.run(FIXTURE, out=tmp_path_factory.mktemp("fit"))


def test_fit_recovers_the_parameters_of_the_synthetic_hardware(fitted):
    truth = json.loads((FIXTURE / "truth.json").read_text())
    params = fitted["params"]
    for key, expected in truth.items():
        if key == "l2_bandwidth_per_sm_gb_s":
            continue  # not identifiable when the TMA engine is the slower one; checked below
        # The Tensor Core rates come from a straight line through times that are not exactly
        # linear in the batch size, so they are a little off; everything else is exact.
        tolerance = 2e-3 if key.startswith("tensor_core") else 1e-6
        assert params[key]["value"] == pytest.approx(expected, rel=tolerance), key
    assert params["sm_count"]["value"] == 148


def test_l2_rate_is_reported_as_a_lower_bound_when_the_engine_limits_warm_loads(fitted):
    truth = json.loads((FIXTURE / "truth.json").read_text())
    l2 = fitted["params"]["l2_bandwidth_per_sm_gb_s"]
    engine = truth["tma_bytes_per_clk"] * truth["clock_ghz"]  # bytes per ns
    assert l2["value"] == pytest.approx(engine, rel=1e-6)
    assert l2["value"] < truth["l2_bandwidth_per_sm_gb_s"]
    assert "LOWER BOUND" in l2["note"]


def test_synthetic_input_is_labelled_in_everything_written(fitted):
    out = fitted["out"]
    for name in ("derived.yaml", "calibrated.yaml", "report.md"):
        assert "SYNTHETIC" in (out / name).read_text(), name
    assert (FIXTURE / "SYNTHETIC.txt").exists()
    for file in FIXTURE.glob("*.out"):
        assert "synthetic=1" in file.read_text().splitlines()[0], file.name


def test_calibrated_config_loads_and_carries_the_derived_values(fitted):
    config = GpuConfig.load(fitted["out"] / "calibrated.yaml")
    for key, entry in fitted["params"].items():
        assert config[key] == pytest.approx(entry["value"], rel=1e-4), key
        assert config.params[key].is_assumption  # synthetic numbers must never read as measured
        assert config.params[key].unit == fit.UNITS[key]
    # The fragment alone has the same schema: every entry has a value, a unit and a source.
    import yaml
    fragment = yaml.safe_load((fitted["out"] / "derived.yaml").read_text())
    assert set(fragment) == set(fitted["params"])
    assert all({"value", "unit", "source", "note"} <= set(entry) for entry in fragment.values())


def test_every_measured_configuration_is_compared_with_its_prediction(fitted):
    report = fitted["report"]
    assert "no prediction" not in report
    with (fitted["out"] / "comparison.csv").open() as handle:
        rows = handle.read().splitlines()
    assert len(rows) == len(fitted["measured"]) + 1
    # alu_chain: the synthetic clock is 1.9 GHz against the config's 1.5.
    row = next(r for r in rows if r.startswith("alu_chain/warps=1,")).split(",")
    assert float(row[3]) == pytest.approx(1.5 / 1.9, rel=1e-6)


def write_results(path: Path, lines: list[str], cc: str = "10.0") -> Path:
    path.mkdir()
    device = f"DEVICE bench=alu_chain name=Test_GPU cc={cc} sm_count=148 total_mem_bytes=196000000000 synthetic=0"
    (path / "alu_chain.out").write_text("\n".join([device, *lines]) + "\n")
    return path


def test_slope_statistics_and_spread(tmp_path):
    lines = []
    # 2 ns per repetition plus a launch overhead that differs per trial; trial 2 is 10% slow.
    for trial, (overhead, rate) in enumerate([(5000, 2.0), (9000, 2.0), (5000, 2.2)]):
        for reps in (1000, 2000):
            lines.append(f"BENCH id=alu_chain/warps=1 reps={reps} trial={trial} "
                         f"event_ns={overhead + rate * reps} wall_ns=0 clk={3 * reps} check=0")
    lines.append("BENCH id=alu_chain/warps=1 reps=1000 trial=-1 event_ns=1 wall_ns=0 clk=0 check=0")  # warm-up
    _, rows, problems = fit.read_dir(write_results(tmp_path / "r", lines))
    info = fit.summarize(rows)["alu_chain/warps=1"]
    assert not problems
    assert info["ns_per_rep"] == pytest.approx(2.0)          # fastest launch at each count
    assert info["ns_per_rep_median"] == pytest.approx(2.0)
    assert info["spread_pct"] == pytest.approx(10.0)
    assert info["trials"] == 3
    assert info["clk_per_rep"] == pytest.approx(3.0)


def test_partial_results_from_another_gpu_are_labelled_and_fit_what_they_can(tmp_path):
    predicted = fit.summarize(fit.read_dir(PREDICTED)[1], "sim_ns")["alu_chain/warps=1"]["ns_per_rep"]
    lines = [f"BENCH id=alu_chain/warps=1 reps={reps} trial=0 event_ns={2 * predicted * reps} wall_ns=0 clk=0 check=0"
             for reps in (1000, 2000)]
    lines.append('BENCH_ERROR where="cudaGetLastError()" error=cudaErrorLaunchFailure')
    result = fit.run(write_results(tmp_path / "r", lines, cc="8.6"), out=tmp_path / "fit")
    assert set(result["params"]) == {"sm_count", "hbm_capacity_gb", "clock_ghz"}
    assert result["params"]["clock_ghz"]["value"] == pytest.approx(0.75)  # twice as slow as predicted at 1.5
    assert "NOT a Blackwell" in result["report"]
    assert "cudaErrorLaunchFailure" in result["report"]
    assert "`tma_latency_ns`" in result["report"]  # listed as not derived
    assert GpuConfig.load(result["out"] / "calibrated.yaml").params["clock_ghz"].is_assumption


def test_committed_predictions_cover_every_benchmark():
    meta = json.loads((PREDICTED / "meta.json").read_text())
    predict = load("predict")
    assert set(meta["benchmarks"]) == set(predict.BENCHES)
    assert all(entry["status"] == "ok" for entry in meta["benchmarks"].values()), meta["benchmarks"]
    assert meta["config"] == fit.load_config_values(fit.DEFAULT_CONFIG), \
        "warplens/configs/b200.yaml changed: regenerate validation/blackwell_timing/predicted with predict.py"
    table, _, _ = fit.load_predicted(PREDICTED)
    assert len(table) > 100
    assert table["mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm"]["ns_per_rep"] > 0


def test_committed_predictions_match_the_simulator(tmp_path):
    """Re-run the fast benchmarks under `warplens run --timing` and compare every launch."""
    nvcc = shutil.which("nvcc")
    sources = [*SUITE.glob("*.cu"), *SUITE.glob("*.h")]
    newest = max(p.stat().st_mtime for p in sources)
    stale = [b for b in FAST if not (SUITE / "bin/sim" / b).exists()
             or (SUITE / "bin/sim" / b).stat().st_mtime < newest]
    if stale:
        if not nvcc:
            pytest.skip("the simulator builds of the timing benchmarks are missing or stale and nvcc is not on PATH")
        built = subprocess.run([str(SUITE / "build.sh"), "sim"], capture_output=True, text=True,
                               env=dict(os.environ, NVCC=nvcc, BENCHES=" ".join(stale)))
        if built.returncode:
            pytest.skip(f"could not build the timing benchmarks for sm_100a: {built.stdout[-300:]}{built.stderr[-300:]}")
    predict = load("predict")
    meta = predict.predict(tmp_path / "now", benches=FAST, quiet=True)
    assert all(entry["status"] == "ok" for entry in meta["benchmarks"].values()), meta["benchmarks"]
    keys = ("id", "reps", "sim_ns", "sim_instr", "sim_bound", "check")
    for bench in FAST:
        now = [tuple(r.get(k) for k in keys) for r in fit.read_dir(tmp_path / "now")[1] if r["file"] == bench]
        committed = [tuple(r.get(k) for k in keys) for r in fit.read_dir(PREDICTED)[1] if r["file"] == bench]
        assert len(now) == len(committed) and now, bench
        for a, b in zip(now, committed):
            assert a[:2] == b[:2] and a[3:] == b[3:], (bench, a, b)
            assert a[2] == pytest.approx(b[2], rel=1e-9), (bench, a, b)
