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

from tests.support import BLACKWELL_TIMING

SUITE = BLACKWELL_TIMING
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
    keys = ("id", "reps", "sim_ns", "sim_instr", "sim_bound", "check", "sim_active_sms", "sim_ctas_per_sm",
            "sim_resident_ctas")
    for bench in FAST:
        now = [tuple(r.get(k) for k in keys) for r in fit.read_dir(tmp_path / "now")[1] if r["file"] == bench]
        committed = [tuple(r.get(k) for k in keys) for r in fit.read_dir(PREDICTED)[1] if r["file"] == bench]
        assert len(now) == len(committed) and now, bench
        for a, b in zip(now, committed):
            assert a[:2] == b[:2] and a[3:] == b[3:], (bench, a, b)
            assert a[2] == pytest.approx(b[2], rel=1e-9), (bench, a, b)
            assert a[6] is not None, (bench, a)  # every launch says how many SMs it was timed with
        assert meta["benchmarks"][bench]["active_sms"] == \
            json.loads((PREDICTED / "meta.json").read_text())["benchmarks"][bench]["active_sms"], bench


# -- cluster launch control latency ----------------------------------------------------------

CLC_PLAIN, CLC_PLAIN_2, CLC_MULTICAST = fit.CLC_FAIL
CLC_TAKEN, CLC_TAKEN_MULTICAST = fit.CLC_SUCCEED


def predicted_slopes() -> dict:
    return {ident: info["ns_per_rep"] for ident, info in fit.summarize(fit.read_dir(PREDICTED)[1], "sim_ns").items()
            if "ns_per_rep" in info}


def slope_lines(ident: str, ns_per_rep: float, extra: str = "nctas=1", ticks_per_ns: float = 0.0) -> list[str]:
    """Two trials of a configuration whose launch takes 7000 ns plus `ns_per_rep` per repetition."""
    return [f"BENCH id={ident} {extra} reps={reps} trial={trial} event_ns={7000 + ns_per_rep * reps} wall_ns=0 "
            f"clk={ticks_per_ns * ns_per_rep * reps:.0f} check={reps}"
            for trial in (0, 1) for reps in (10000, 20000)]


def clc_results(tmp_path, latency: dict, slowdown: float = 1.0, ticks_per_ns: float = 0.0, alu: bool = True):
    """Results of a pretend B200 whose instructions take `slowdown` times the model's and whose
    try_cancel responses land `latency[id]` ns after the request."""
    predicted = predicted_slopes()
    lines = slope_lines("alu_chain/warps=1", slowdown * predicted["alu_chain/warps=1"],
                        ticks_per_ns=ticks_per_ns) if alu else []
    for ident, value in latency.items():
        round_trip = value + slowdown * (predicted[ident] - 100.0)  # 100: clc_latency_ns of b200.yaml
        if ident in fit.CLC_SUCCEED:
            # A grid of clusters: the launch time says nothing; the first cluster took 40 others
            # and its 41 requests took `clk` ticks.
            lines += [f"BENCH id={ident} nctas=1 reps={reps} trial={trial} event_ns={50000 + 3 * reps} wall_ns=0 "
                      f"clk={ticks_per_ns * round_trip * 41:.0f} check=40"
                      for trial in (0, 1) for reps in (10000, 20000)]
        else:
            lines += slope_lines(ident, round_trip, "nctas=" + ident[-1], ticks_per_ns)
    return fit.run(write_results(tmp_path / "r", lines), out=tmp_path / "fit")


def test_clc_latency_is_the_round_trip_of_a_failing_request_minus_its_instructions(tmp_path):
    result = clc_results(tmp_path, {CLC_PLAIN: 260.0})
    entry = result["params"]["clc_latency_ns"]
    assert entry["value"] == pytest.approx(260.0)
    assert CLC_PLAIN in entry["formula"] and "NOT ONE NUMBER" not in entry["note"]
    assert "| `clc_latency_ns` | 260 ns | 100 | 2.6 |" in result["report"]
    assert not result["problems"]
    config = GpuConfig.load(result["out"] / "calibrated.yaml")
    assert config["clc_latency_ns"] == pytest.approx(260.0)
    assert config.params["clc_latency_ns"].unit == "ns"
    assert config.params["clc_latency_ns"].source.startswith("measured: validation/blackwell_timing")


def test_clc_latency_is_separated_from_instruction_time_at_the_fitted_clock(tmp_path):
    """Instructions twice as slow as the model's: the clock is fitted at 0.75 GHz, the loop's
    instructions take twice the predicted time, and the latency left over is still the latency."""
    result = clc_results(tmp_path, {CLC_PLAIN: 40.0, CLC_PLAIN_2: 40.0, CLC_MULTICAST: 40.0}, slowdown=2.0)
    assert result["params"]["clock_ghz"]["value"] == pytest.approx(0.75)
    entry = result["params"]["clc_latency_ns"]
    assert entry["value"] == pytest.approx(40.0)
    assert "fitted clock_ghz = 0.75" in entry["note"] and "NOT ONE NUMBER" not in entry["note"]
    implied = {name: note for name, _, _, note in result["diagnostics"] if name.startswith("CLC request")}
    assert set(implied) == {"CLC request, fail/plain/ctas=1", "CLC request, fail/plain/ctas=2",
                            "CLC request, fail/multicast/ctas=2"}
    assert all("= 40 ns of latency" in note for note in implied.values())


def test_without_the_clc_benchmark_the_latency_stays_the_configs_assumption(tmp_path):
    result = clc_results(tmp_path, {})
    assert "clc_latency_ns" not in result["params"] and "clock_ghz" in result["params"]
    missing = next(line for line in result["report"].splitlines() if line.startswith("Not derived"))
    assert "`clc_latency_ns`" in missing
    assert not any(name.startswith("CLC") or name == "clc_latency_ns" for name, *_ in result["diagnostics"])
    config = GpuConfig.load(result["out"] / "calibrated.yaml")
    assert config["clc_latency_ns"] == 100 and config.params["clc_latency_ns"].is_assumption
    assert "clc_latency_ns" not in (result["out"] / "derived.yaml").read_text()


def test_clc_latency_falls_back_to_a_two_cta_cluster_and_says_so(tmp_path):
    result = clc_results(tmp_path, {CLC_PLAIN_2: 75.0, CLC_MULTICAST: 75.0})
    entry = result["params"]["clc_latency_ns"]
    assert entry["value"] == pytest.approx(75.0)
    assert CLC_PLAIN_2 in entry["formula"]
    assert f"{CLC_PLAIN} was not usable" in entry["note"] and "cluster barrier" in entry["note"]


def test_clc_forms_that_disagree_are_named_and_the_failing_request_is_still_the_parameter(tmp_path):
    latency = {CLC_PLAIN: 200.0, CLC_PLAIN_2: 210.0, CLC_MULTICAST: 500.0, CLC_TAKEN: 900.0, CLC_TAKEN_MULTICAST: 205.0}
    result = clc_results(tmp_path, latency, ticks_per_ns=1.5)
    entry = result["params"]["clc_latency_ns"]
    assert entry["value"] == pytest.approx(200.0)
    assert "NOT ONE NUMBER" in entry["note"]
    assert f"{CLC_MULTICAST} implies 500 ns" in entry["note"] and f"{CLC_TAKEN} implies 900 ns" in entry["note"]
    assert CLC_PLAIN_2 not in entry["note"] and CLC_TAKEN_MULTICAST not in entry["note"]  # within 20%
    # The requests that succeed come from the first cluster's own ticks, not from the launch time.
    diag = {name: (value, note) for name, value, _, note in result["diagnostics"]}
    predicted = predicted_slopes()
    value, note = diag["CLC request, succeed/plain/ctas=1"]
    assert value == pytest.approx(900.0 + predicted[CLC_TAKEN] - 100.0, rel=1e-4)
    assert "clk / (check + 1)" in note and "1.5 ticks per ns" in note and "= 900 ns of latency" in note
    assert "these two slopes are not round trips" in result["report"]


def test_successful_requests_need_the_clock_tick_rate_and_never_set_the_parameter(tmp_path):
    # Ticks in the lines, but no alu_chain to say how long a tick is.
    result = clc_results(tmp_path, {CLC_TAKEN: 300.0}, ticks_per_ns=1.5, alu=False)
    assert "clc_latency_ns" not in result["params"]
    diag = {name: (value, note) for name, value, _, note in result["diagnostics"]}
    assert diag["CLC request, succeed/plain/ctas=1"][0] == "not derived"
    assert "no ticks per ns" in diag["CLC request, succeed/plain/ctas=1"][1]
    assert diag["clc_latency_ns"][0] == "not derived" and "reported, not fitted" in diag["clc_latency_ns"][1]


def test_successful_requests_without_clock_ticks_are_not_derived(tmp_path):
    # A build that did not read %clock64 (clk=0): the slope of a grid of clusters must not be used.
    result = clc_results(tmp_path, {CLC_PLAIN: 150.0, CLC_TAKEN: 300.0}, ticks_per_ns=0.0)
    diag = {name: (value, note) for name, value, _, note in result["diagnostics"]}
    assert diag["CLC request, succeed/plain/ctas=1"][0] == "not derived"
    assert "clk = 0" in diag["CLC request, succeed/plain/ctas=1"][1]
    assert result["params"]["clc_latency_ns"]["value"] == pytest.approx(150.0)


def test_unusable_clc_measurements_are_reported_not_fitted(tmp_path):
    predicted = predicted_slopes()
    # A slope that is not positive is no measurement; the next configuration is used.
    lines = slope_lines("alu_chain/warps=1", predicted["alu_chain/warps=1"])
    lines += slope_lines(CLC_PLAIN, -3.0) + slope_lines(CLC_PLAIN_2, predicted[CLC_PLAIN_2] + 20.0, "nctas=2")
    lines.append('BENCH_ERROR where="clc_latency/succeed/plain/ctas=1" error=cudaErrorLaunchFailure class=failed')
    result = fit.run(write_results(tmp_path / "a", lines), out=tmp_path / "fit_a")
    diag = {name: (value, note) for name, value, _, note in result["diagnostics"]}
    assert diag["CLC request, fail/plain/ctas=1"][0] == "not derived"
    assert result["params"]["clc_latency_ns"]["value"] == pytest.approx(120.0)
    assert "cudaErrorLaunchFailure" in result["report"]
    # A round trip shorter than the instructions the simulator counts: clamped to the model's lower limit.
    lines = slope_lines("alu_chain/warps=1", predicted["alu_chain/warps=1"]) + slope_lines(CLC_PLAIN, 2.0)
    result = fit.run(write_results(tmp_path / "b", lines), out=tmp_path / "fit_b")
    entry = result["params"]["clc_latency_ns"]
    assert entry["value"] == 0.0 and "clamped to 0" in entry["note"]
    assert GpuConfig.load(result["out"] / "calibrated.yaml")["clc_latency_ns"] == 0.0


def test_clc_latency_needs_a_prediction_of_the_same_loop(tmp_path):
    """Predictions without clc_latency (a simulator that could not run it): the round trip is
    reported, and no latency is made up from it."""
    predicted = tmp_path / "predicted"
    predicted.mkdir()
    for name in ("alu_chain.out", "meta.json"):
        shutil.copy(PREDICTED / name, predicted / name)
    lines = slope_lines("alu_chain/warps=1", 2.5) + slope_lines(CLC_PLAIN, 300.0)
    result = fit.run(write_results(tmp_path / "r", lines), predicted=predicted, out=tmp_path / "fit")
    assert "clc_latency_ns" not in result["params"]
    diag = {name: (value, note) for name, value, _, note in result["diagnostics"]}
    assert diag["CLC request, fail/plain/ctas=1"][0] == pytest.approx(300.0)
    assert "no prediction of this configuration" in diag["CLC request, fail/plain/ctas=1"][1]
    assert diag["clc_latency_ns"][0] == "not derived"
    # Predictions made with a config that has no clc_latency_ns at all (before the model had one).
    meta = json.loads((predicted / "meta.json").read_text())
    del meta["config"]["clc_latency_ns"]
    shutil.copy(PREDICTED / "clc_latency.out", predicted / "clc_latency.out")
    params, diag = fit.derive(result["measured"], fit.load_predicted(predicted)[0], meta["config"], result["device"])
    assert "clc_latency_ns" not in {q["key"] for q in params}
    assert any(name == "clc_latency_ns" and "has no clc_latency_ns entry" in note for name, _, _, note in diag)


# -- the active-SM assumption of the predictions, against the run ------------------------------

def test_report_shows_the_assumed_active_sms_and_what_the_run_contradicts(tmp_path):
    predicted = predicted_slopes()
    cold = "tma/load/cold/bytes=16384/ctas={}"
    lines = slope_lines("alu_chain/warps=1", predicted["alu_chain/warps=1"])
    lines += slope_lines(cold.format(8), 190.0, "bytes=16384 nctas=8")
    lines += slope_lines(cold.format("allsm"), 700.0, "bytes=16384 nctas=148")
    result = fit.run(write_results(tmp_path / "same", lines), out=tmp_path / "fit_same")
    assert not result["problems"]
    table = [line for line in result["report"].splitlines() if line.startswith(f"| {cold.format(8)} |")]
    assert table[-1].split("|")[5].strip() == "8"
    with (result["out"] / "comparison.csv").open() as handle:
        rows = {row.split(",")[0]: row.split(",") for row in handle.read().splitlines()}
    assert rows["id"][4] == "predicted_active_sms"
    assert (rows[cold.format(8)][4], rows[cold.format("allsm")][4], rows["alu_chain/warps=1"][4]) == ("8", "148", "1")

    # Another GPU: 58 SMs, other per-SM limits, and "all SMs" is a grid of 58.
    device = ("DEVICE bench=tma name=Other_GPU cc=10.0 sm_count=58 total_mem_bytes=1 synthetic=0 "
              "smem_per_sm_bytes=102400 max_threads_per_sm=1536 max_blocks_per_sm=32")
    other = tmp_path / "other"
    other.mkdir()
    (other / "tma.out").write_text("\n".join([device, *slope_lines(cold.format(8), 190.0, "bytes=16384 nctas=8"),
                                              *slope_lines(cold.format("allsm"), 700.0, "bytes=16384 nctas=58")]) + "\n")
    problems = "\n".join(fit.run(other, out=tmp_path / "fit_other")["problems"])
    assert "the device has 58 SMs and the predictions place launches on 148" in problems
    assert "smem_per_sm_bytes = 102400" in problems and "max_threads_per_sm = 1536" in problems
    assert "max_blocks_per_sm" not in problems
    assert "1 configuration(s) were launched with a grid" in problems
    assert f"`{cold.format('allsm')}` (58 CTAs launched, 58 SMs; predicted for 148)" in problems


def test_predictions_without_an_active_sm_assumption_are_flagged(tmp_path):
    predicted = tmp_path / "predicted"
    predicted.mkdir()
    old = [" ".join(t for t in line.split() if not t.startswith(("sim_active_sms", "sim_ctas_per_sm", "sim_resident")))
           for line in (PREDICTED / "alu_chain.out").read_text().splitlines()]
    (predicted / "alu_chain.out").write_text("\n".join(old) + "\n")
    lines = slope_lines("alu_chain/warps=1", 2.5)
    result = fit.run(write_results(tmp_path / "r", lines), predicted=predicted, out=tmp_path / "fit")
    assert any("record no active-SM assumption" in problem for problem in result["problems"])
    assert result["params"]["clock_ghz"]["value"] == pytest.approx(1.5)
