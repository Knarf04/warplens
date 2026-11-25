"""The Blackwell probe suite (validation/blackwell): its committed predictions are what the
simulator produces now.

The predictions are what a run on a rented B200 will be compared against, so they must not
go stale when the simulator changes: every probe program is run under `warplens run` and its
output compared with `validation/blackwell/predicted/`. After an intended change, regenerate
them with `validation/blackwell/predict.sh` and review the diff.

The probes that run in the simulator need the programs, which need nvcc (with `sm_100a`) to
build; without it those tests are skipped. The checks of the manifest and of the small-format
predictions (which come from `warplens.numerics`) always run.
"""

import concurrent.futures
import os
import shutil
import subprocess
import sys

import pytest

from tests.support import REPO as ROOT

SUITE = ROOT / "validation" / "blackwell"
PREDICTED = SUITE / "predicted"
sys.path.insert(0, str(SUITE))
sys.path.insert(0, str(SUITE / "formats"))
import manifest  # noqa: E402
import predict_small_formats  # noqa: E402

SIMULATED = [p for p in manifest.PROBES if p.predict == "sim"]
SMALL_FORMATS = [p for p in manifest.PROBES if p.program == "formats/small_formats"]


def test_manifest_and_predictions_agree():
    for probe in manifest.PROBES:
        assert (SUITE / (probe.program + ".cu")).is_file(), probe.name
        assert probe.claim and probe.risk in ("", "hang", "fault") and probe.compare in ("exact", "ulp")
        has_prediction = (PREDICTED / (probe.name + ".txt")).is_file()
        assert has_prediction == (probe.predict != "none"), probe.name
        assert (PREDICTED / (probe.name + ".sim.txt")).is_file() == has_prediction, probe.name
    known = {p.name for p in manifest.PROBES} | set(manifest.E2E)
    stray = [str(f.relative_to(PREDICTED)) for f in PREDICTED.rglob("*.txt")
             if str(f.relative_to(PREDICTED)).removesuffix(".sim.txt").removesuffix(".txt") not in known]
    assert not stray, f"predictions with no manifest entry: {stray}"
    sources = {str(f.relative_to(SUITE)).removesuffix(".cu") for f in SUITE.glob("*/*.cu")}
    assert sources == {p.program for p in manifest.PROBES}, "a probe source that the manifest never runs"


@pytest.mark.parametrize("probe", SMALL_FORMATS, ids=lambda p: p.name)
def test_small_format_predictions_are_what_numerics_says(probe):
    """The prediction is the probe's output under `warplens run` (the `cvt` instructions);
    predict_small_formats.py computes the same lines from the warplens.numerics tables."""
    assert len(SMALL_FORMATS) == 4
    compared = lambda text: [line for line in text.splitlines() if not line.startswith("info:")]  # as compare.py does
    assert compared((PREDICTED / (probe.name + ".txt")).read_text()) == compared(predict_small_formats.predict(*probe.args))


def test_predicted_rna_column_is_what_the_rtx_3080_ti_gave():
    """formats/tf32_modes uses the inputs of validation/ampere/tf32_golden.cu, so its predicted
    .rna column must be the output recorded from the GPU."""
    lines = (PREDICTED / "formats" / "tf32_modes.txt").read_text().splitlines()[1:]
    golden = (ROOT / "validation" / "ampere" / "tf32_golden.txt").read_text().splitlines()
    assert [" ".join(line.split()[:2]) for line in lines] == golden


def _stale(probe) -> bool:
    binary = SUITE / "bin" / probe.program
    if not binary.is_file():
        return True
    sources = [SUITE / (probe.program + ".cu"), *SUITE.glob("*.h"), *(SUITE / probe.program).parent.glob("*.h")]
    return any(source.stat().st_mtime > binary.stat().st_mtime for source in sources)


@pytest.fixture(scope="session")
def simulated():
    """Build the probe programs if needed, then run each one under `warplens run`, a few at a
    time. Returns {name: (exit status text, stdout)}."""
    if any(_stale(p) for p in SIMULATED):
        if not shutil.which("nvcc"):
            pytest.skip("the probe programs are not built and nvcc is not on PATH")
        done = subprocess.run([str(SUITE / "build.sh")], capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    shims = ROOT / "warplens" / "shim" / "lib"
    if not (shims / "libcuda.so.1").is_file() and not shutil.which("nvcc"):
        pytest.skip("the shim libraries are not built and nvcc is not on PATH")
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
           "CUDA_VISIBLE_DEVICES": ""}

    def run(probe):
        done = subprocess.run([sys.executable, "-m", "warplens.cli", "run", str(SUITE / "bin" / probe.program),
                               *probe.args], capture_output=True, text=True, timeout=900, env=env, cwd=ROOT)
        code = done.returncode
        return probe.name, ("ok" if code == 0 else f"exit {code}" if code > 0 else f"signal {-code}",
                            done.stdout, done.stderr)

    run(SIMULATED[0])  # builds the shim libraries, if this is their first use, before the pool starts
    with concurrent.futures.ThreadPoolExecutor(min(6, os.cpu_count() or 1)) as pool:
        return dict(pool.map(run, SIMULATED))


@pytest.mark.parametrize("probe", SIMULATED, ids=lambda p: p.name)
def test_prediction_is_what_the_simulator_prints_now(simulated, probe):
    status, stdout, stderr = simulated[probe.name]
    recorded = (PREDICTED / (probe.name + ".sim.txt")).read_text().splitlines()[0]
    assert f"status: {status}" == recorded, stderr[-2000:]
    expected = (PREDICTED / (probe.name + ".txt")).read_text()
    if stdout != expected:
        got, want = stdout.splitlines(), expected.splitlines()
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        pytest.fail(f"{probe.name}: the simulator's output differs from predicted/{probe.name}.txt at line "
                    f"{at + 1}:\n  predicted: {want[at] if at < len(want) else '<end>'}\n"
                    f"  now:       {got[at] if at < len(got) else '<end>'}\n"
                    "If the change is intended, run validation/blackwell/predict.sh and review the diff.")


def test_probes_the_simulator_refuses_say_so(simulated):
    """A probe whose kernel the simulator will not run predicts 'KERNEL FAILED', never a
    made-up result; and the ones expected to run did run."""
    for probe in SIMULATED:
        status, stdout, _ = simulated[probe.name]
        assert (status == "ok") != ("KERNEL FAILED" in stdout), probe.name
        if not probe.risk:
            assert status == "ok", f"{probe.name} is not marked as risky but the simulator refuses it"


def test_readme_tables_are_the_manifest():
    """README.md lists every run with what the simulator does with it. The tables are written
    by `python manifest.py --tables`, so they cannot drift from the manifest or from the
    committed predictions."""
    readme = (SUITE / "README.md").read_text()
    start = readme.index("<!-- tables:")
    start = readme.index("\n\n", start) + 2
    end = readme.index("<!-- end of tables -->")
    assert readme[start:end] == manifest.tables(), \
        "run `python validation/blackwell/manifest.py --update-readme`"
