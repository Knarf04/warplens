"""The whole predict path, as one command, on the CPU.

    validation/blackwell/run_all.py --mode predict [--book CLONE]

is what `validation/blackwell/predict.sh` runs to produce everything a rented B200's results
are compared with: it builds every probe, runs each under `warplens run`, writes the
small-format predictions from `warplens.numerics`, then runs the end-to-end programs (the
four CUTLASS tutorials, the Triton kernels, the book's nine GEMMs) and writes their
predictions too. tests/test_blackwell_probes.py checks the probes one by one with its own
runner; this runs the script itself, into a temporary directory (`--predicted`), and holds
the result against the committed files, so the path that will be used to regenerate the
predictions after the simulator changes is known to work from end to end.

Needs nvcc for the probes. The end-to-end parts need what their own tests need and are
skipped without it: the CUTLASS programs (examples/03_cutlass_mma/build.sh --programs),
Triton, and for the book TIRx and a clone in WARPLENS_BOOK. CI's `full` job has all of them.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "validation" / "blackwell"
COMMITTED = SUITE / "predicted"
CUTLASS_BIN = ROOT / "examples" / "03_cutlass_mma" / "bin"
# run_e2e.py builds its capturing copies of the tutorials from this clone (build.sh makes it).
CUTLASS_CLONE = ROOT / "examples" / "03_cutlass_mma" / ".cutlass"
BOOK = os.environ.get("WARPLENS_BOOK", "")
HAVE_BOOK = bool(BOOK) and (Path(BOOK) / "chapter_gemm_basics" / "index.md").is_file()
sys.path.insert(0, str(SUITE))
import compare  # noqa: E402
import manifest  # noqa: E402

TUTORIALS = [f"e2e/cutlass_tutorials.tutorial_0{n}" for n in range(1, 5)]


def have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


key_lines = compare.key_lines  # the lines compare.py shows for an end-to-end program, without the times


@pytest.mark.parametrize("error", ["0", "0.0156", "1.56e-2", "1.56E-02"])
def test_book_comparison_normalizes_only_finite_success_distances(error):
    first = "hgemm_v6 128x128x256: matches NumPy (max error 0.0078, 1.23 s)"
    another_cpu = f"hgemm_v6 128x128x256: matches NumPy (max error {error}, 2.34 s)"
    assert compare.claims(key_lines(first)) == compare.claims(key_lines(another_cpu))
    assert compare.claims(key_lines(first)) != compare.claims(key_lines(
        another_cpu.replace("128x128x256", "128x128x64")))
    assert compare.claims(key_lines(first)) != compare.claims(key_lines(
        another_cpu.replace("matches NumPy", "FAILED")))


@pytest.mark.parametrize("error", ["nan", "NaN", "inf", "-inf", "1e999", "-0.0156", "1.2.3"])
def test_book_comparison_retains_invalid_or_nonfinite_distances(error):
    line = f"hgemm_v6 128x128x256: matches NumPy (max error {error})"
    assert compare.claims([line]) == [line]
    assert compare.claims([line]) != ["hgemm_v6 128x128x256: matches NumPy"]


def test_book_comparison_does_not_rewrite_other_program_results():
    lines = ["tutorial_01: matches NumPy (max error 0.0156)",
             "hgemm_v6 128x128x256: FAILED (max error 0.0156)",
             "hgemm_v6 128x128x256: matches NumPy (max error 0.0156) extra diagnostic"]
    assert compare.claims(lines) == lines


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One run of the whole path. Returns (process, predicted directory, results directory,
    {name: status})."""
    if not shutil.which("nvcc"):
        pytest.skip("nvcc is not on PATH: the probe programs cannot be built")
    base = tmp_path_factory.mktemp("predict")
    predicted, out = base / "predicted", base / "out"
    command = [sys.executable, str(SUITE / "run_all.py"), "--mode", "predict", "--predicted", str(predicted),
               "--out", str(out), "--jobs", str(min(6, os.cpu_count() or 1))]
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
           "CUDA_VISIBLE_DEVICES": ""}
    if HAVE_BOOK and have("tvm_tirx"):
        command += ["--book", BOOK]
    else:
        env["WARPLENS_BOOK"] = ""
    if not all((CUTLASS_BIN / Path(name).name.split(".")[1]).exists() for name in TUTORIALS):
        # Not built here: do not let the script fetch and build CUTLASS inside a test.
        env["CUTLASS_BIN"] = str(base / "no-cutlass-programs")
    done = subprocess.run(command, capture_output=True, text=True, timeout=3600, env=env, cwd=ROOT)
    statuses = {}
    for line in (out / "status.tsv").read_text().splitlines()[1:]:
        name, status = line.split("\t")[:2]
        statuses[name] = status
    return done, predicted, out, statuses


def test_the_script_runs_to_the_end_and_records_every_run(run):
    done, predicted, out, statuses = run
    # The collection is complete only with every end-to-end program: a step that cannot run
    # here (no clone of the book, no CUTLASS clone) is recorded as skipped and fails the run.
    absent = set()
    if not CUTLASS_CLONE.is_dir():
        absent |= set(TUTORIALS)
    if not (HAVE_BOOK and have("tvm_tirx")):
        absent.add("e2e/book_hgemm")
    not_ok = {name for name in manifest.E2E if statuses.get(name) != "ok"}
    if all(have(module) for module in ("triton", "tvm_tirx", "cutlass", "cuda.bindings")):
        assert not_ok == absent, {name: statuses.get(name) for name in not_ok ^ absent}
    assert all(statuses[name].startswith("skipped (") for name in absent)
    assert done.returncode == (1 if not_ok else 0), done.stdout[-3000:] + done.stderr[-3000:]
    assert ("end-to-end runner: ok" in done.stdout) == (not not_ok)
    assert "build: ok" in done.stdout
    assert set(statuses) == {probe.name for probe in manifest.PROBES} | set(manifest.E2E)
    assert not [name for name, status in statuses.items() if status.startswith(("not built", "timeout", "signal"))]
    assert Path(str(out) + ".tar.gz").is_file()
    # The compiler whose PTX was simulated is written next to the predictions.
    built = next(line.split("=", 1)[1] for line in (SUITE / "bin" / "toolkit.txt").read_text().splitlines()
                 if line.startswith("nvcc_version="))
    assert json.loads((predicted / "provenance.json").read_text())["nvcc_version"] == built


def test_the_compiler_is_the_one_the_committed_predictions_name(run):
    """With another nvcc on PATH (a rented machine whose driver forced an older toolkit) this
    test fails and says so, while the tests below still answer the question that matters
    there: does the simulator print the committed predictions for THIS compiler's code?"""
    _, predicted, _, _ = run
    now = json.loads((predicted / "provenance.json").read_text())["nvcc_version"]
    committed = json.loads((COMMITTED / "provenance.json").read_text())["nvcc_version"]
    assert now == committed, (f"the probes were built by nvcc {now}; the committed predictions were made with nvcc "
                              f"{committed}. The per-probe tests of this file say whether that changes any prediction")


@pytest.mark.parametrize("probe", [p for p in manifest.PROBES if p.predict != "none"], ids=lambda p: p.name)
def test_every_probe_prediction_is_the_committed_one(run, probe):
    _, predicted, _, statuses = run
    committed = (COMMITTED / (probe.name + ".txt")).read_text()
    assert (predicted / (probe.name + ".txt")).read_text() == committed, \
        f"{probe.name}: run validation/blackwell/predict.sh and review the diff"
    note = (predicted / (probe.name + ".sim.txt")).read_text().splitlines()[0]
    assert note == (COMMITTED / (probe.name + ".sim.txt")).read_text().splitlines()[0]
    if probe.predict == "sim":
        assert note == f"status: {statuses[probe.name]}"


def test_probes_without_a_prediction_write_none(run):
    _, predicted, _, statuses = run
    for probe in manifest.PROBES:
        if probe.predict == "none":
            assert statuses[probe.name].startswith("skipped"), probe.name
            assert not (predicted / (probe.name + ".txt")).exists()


def test_compare_accepts_the_simulators_own_run(run):
    """compare.py on the results of this run: every probe the simulator predicts is
    `identical`, and nothing claims the programs were built by another toolkit."""
    _, predicted, out, _ = run
    done = subprocess.run([sys.executable, str(SUITE / "compare.py"), str(out), "--predicted", str(predicted)],
                          capture_output=True, text=True)
    assert "NOT COMPARABLE" not in done.stdout and "TOOLKIT NOT RECORDED" not in done.stdout
    verdicts = {}
    for line in done.stdout.split("\n\n")[0].splitlines()[1:]:
        fields = line.split(None, 1)
        verdicts[fields[0]] = fields[1].strip() if len(fields) > 1 else ""
    for probe in manifest.PROBES:
        if probe.predict == "sim":
            assert verdicts[probe.name].startswith("identical"), (probe.name, verdicts[probe.name])
    assert not [name for name, verdict in verdicts.items() if verdict.startswith(("FAILED TO LAUNCH", "HOST ERROR"))]


def test_cutlass_tutorials_are_predicted(run):
    if not CUTLASS_CLONE.is_dir():
        pytest.skip("no CUTLASS clone (examples/03_cutlass_mma/build.sh --programs fetches and builds it)")
    _, predicted, _, statuses = run
    for name in TUTORIALS:
        assert statuses[name] == "ok", name
        now = key_lines((predicted / (name + ".txt")).read_text())
        assert now and now == key_lines((COMMITTED / (name + ".txt")).read_text()), name
        assert (predicted / (name + ".sim.txt")).read_text().startswith("status: ok\n")


def test_triton_is_predicted(run):
    pytest.importorskip("triton")
    _, predicted, _, statuses = run
    assert statuses["e2e/triton"] == "ok"
    # The matmul's max error is not compared: it is measured against this machine's NumPy
    # (see compare.TRITON_ERROR). The vector add's is, and so are the simulator's counts.
    now = key_lines((predicted / "e2e" / "triton.txt").read_text())
    assert len(now) == 2 and all(": max abs error vs NumPy " in line for line in now), now
    assert compare.claims(now) == compare.claims(key_lines((COMMITTED / "e2e" / "triton.txt").read_text()))


def test_the_triton_matmuls_max_error_is_not_part_of_what_is_compared():
    """CI's runner printed 5.72e-06 where the committed prediction says 5.25e-06, for the same
    kernel output: on one machine the number moves between the two with OMP_NUM_THREADS."""
    committed = key_lines((COMMITTED / "e2e" / "triton.txt").read_text())
    assert "matmul: max abs error vs NumPy 5.25e-06; " in committed[1]
    elsewhere = [line.replace("5.25e-06", "5.72e-06") for line in committed]
    assert elsewhere != committed and compare.claims(elsewhere) == compare.claims(committed)
    assert compare.claims(committed)[1].startswith("matmul: max abs error vs NumPy; 4 CTA(s), ")
    # The GPU's line has the number and nothing after it.
    assert compare.claims(["matmul: max abs error vs NumPy 4.77e-06"]) == ["matmul: max abs error vs NumPy"]
    # What is a claim still counts: the simulator's counts, the vector add's exact result, a
    # distance that is not a number.
    assert compare.claims([committed[1].replace("8480", "8481")]) != compare.claims([committed[1]])
    assert compare.claims([committed[0].replace("NumPy 0;", "NumPy 1e-07;")]) != compare.claims([committed[0]])
    for bad in ("nan", "inf", "-1e-06"):
        line = f"matmul: max abs error vs NumPy {bad}"
        assert compare.claims([line]) == [line]


@pytest.mark.parametrize("name,module", [("tirx", "tvm_tirx"), ("cute_dsl", "cutlass")])
def test_scripts_that_run_unchanged_are_predicted(run, name, module):
    """Matrix row RT-09: examples/07_tirx/run.py and examples/06_cute_dsl/saxpy.py under
    `warplens run`. On the rented machine the same files run with plain `python`."""
    pytest.importorskip(module)
    pytest.importorskip("cuda.bindings")
    _, predicted, _, statuses = run
    step = f"e2e/unchanged_scripts.{name}"
    assert statuses[step] == "ok"
    now = key_lines((predicted / (step + ".txt")).read_text())
    assert len(now) == 1 and "matches NumPy" in now[0]
    assert now == key_lines((COMMITTED / (step + ".txt")).read_text())


def test_the_gpu_steps_that_use_tvm_get_the_cuda_runtime_on_their_library_path(tmp_path, monkeypatch):
    """Plain `python` does not find libcudart.so.13 for TVM's CUDA runtime, and TIRx then says
    there is no CUDA device (measured on the RTX 3080 Ti). run_e2e.py puts the toolkit's
    library directory on LD_LIBRARY_PATH for those steps, and only for those: PyTorch brings
    its own CUDA runtime, which must not be shadowed."""
    sys.path.insert(0, str(SUITE / "e2e"))
    try:
        import run_e2e
    finally:
        sys.path.remove(str(SUITE / "e2e"))
    text = (SUITE / "e2e" / "run_e2e.py").read_text()
    assert 'uses_tvm = ("e2e/unchanged_scripts.tirx", "e2e/book_hgemm")' in text
    assert "step_env = dict(tvm_env if name in uses_tvm else env)" in text
    assert "env=step_env" in text
    prefix = tmp_path / "env"
    (prefix / "lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setenv("NVCC", "no-such-nvcc")
    assert run_e2e.cudart_directory() == ""
    (prefix / "lib" / "libcudart.so.13").touch()
    assert run_e2e.cudart_directory() == str(prefix / "lib")


def test_the_book_prediction_is_committed_and_names_all_nine_kernels():
    """Always runs: the committed prediction is there and says what tests/test_book.py
    expects of the simulator."""
    lines = (COMMITTED / "e2e" / "book_hgemm.txt").read_text().splitlines()
    sizes = {**{f"v{n}": "128x128x64" for n in range(1, 6)}, "v6": "128x128x256", "v7": "128x128x256",
             "v8": "256x256x256", "v9": "512x256x256"}
    assert len(lines) == 9
    for line, (version, size) in zip(lines, sizes.items()):
        assert line.startswith(f"hgemm_{version} {size}: matches NumPy (max error "), line
    note = (COMMITTED / "e2e" / "book_hgemm.sim.txt").read_text()
    assert note.startswith("status: ok\n") and "9 kernel launch(es) simulated on B200" in note
    assert "e2e/book_hgemm" in manifest.E2E


@pytest.mark.skipif(not HAVE_BOOK, reason="WARPLENS_BOOK is not a clone of the book")
def test_the_book_is_predicted(run):
    pytest.importorskip("tvm_tirx")
    pytest.importorskip("cuda.bindings")
    _, predicted, _, statuses = run
    assert statuses["e2e/book_hgemm"] == "ok"
    # Names, sizes and "matches NumPy" are compared; the max error is not, because it is
    # measured against this machine's NumPy (see compare.MAX_ERROR).
    now = key_lines((predicted / "e2e" / "book_hgemm.txt").read_text())
    assert len(now) == 9 and all(": matches NumPy (max error " in line for line in now), now
    assert compare.claims(now) == compare.claims(key_lines((COMMITTED / "e2e" / "book_hgemm.txt").read_text()))


def test_a_max_error_is_not_part_of_what_is_compared(tmp_path):
    """CI's runner printed another max error than the committed prediction says (0.0156
    where it said 0.0078 at the time), for the same kernel output: the reference is NumPy's
    float32 product, and its BLAS sums in an order that depends on the CPU and the thread
    count."""
    committed = key_lines((COMMITTED / "e2e" / "book_hgemm.txt").read_text())
    elsewhere = [line.replace("max error 0.0156", "max error 0.0078") for line in committed]
    assert elsewhere != committed and compare.claims(elsewhere) == compare.claims(committed)
    assert compare.claims(committed)[5] == "hgemm_v6 128x128x256: matches NumPy"
    # What is a claim still counts: another size, a failure, a missing kernel.
    assert compare.claims([committed[5].replace("128x128x256", "128x128x64")]) != compare.claims([committed[5]])
    assert compare.claims(["hgemm_v6: FAILED: AssertionError: Not equal to tolerance"]) != compare.claims([committed[5]])
    assert compare.claims(committed[:8]) != compare.claims(committed)
    # Other programs' result lines are compared as they are.
    tutorial = key_lines((COMMITTED / "e2e" / "cutlass_tutorials.tutorial_01.txt").read_text())
    assert tutorial and compare.claims(tutorial) == tutorial

    # compare.py on a rental's results: both sets of lines are shown, and the note says why
    # they are not a difference.
    results = tmp_path / "results"
    (results / "e2e").mkdir(parents=True)
    text = (COMMITTED / "e2e" / "book_hgemm.txt").read_text().replace("max error 0.0156", "max error 0.0078")
    (results / "e2e" / "book_hgemm.stdout").write_text(text)
    (results / "status.tsv").write_text("name\tstatus\ne2e/book_hgemm\tok\n")
    shown = subprocess.run([sys.executable, str(SUITE / "compare.py"), str(results)],
                           capture_output=True, text=True, timeout=120).stdout
    assert "    GPU: hgemm_v6 128x128x256: matches NumPy (max error 0.0078)" in shown
    assert "    sim: hgemm_v6 128x128x256: matches NumPy (max error 0.0156)" in shown
    assert "the same claims: a max error is the distance to NumPy's result" in shown


def test_without_a_clone_the_book_step_says_skipped(tmp_path):
    """The end-to-end driver alone, with nothing to run: every step is recorded as skipped
    with its reason, never left out."""
    done = subprocess.run([sys.executable, str(SUITE / "e2e" / "run_e2e.py"), "--mode", "sim", "--out", str(tmp_path),
                           "--cutlass-bin", str(tmp_path / "none"), "--cutlass-source", str(tmp_path / "none"),
                           "--book", str(tmp_path / "no-book")],
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "WARPLENS_BOOK": ""})
    assert done.returncode == 1, done.stderr  # the tutorials are not built: an incomplete collection, said so
    statuses = dict(line.split("\t")[:2] for line in (tmp_path / "status.tsv").read_text().splitlines()[1:])
    assert statuses["e2e/book_hgemm"] == "skipped (no clone of the book given (--book or WARPLENS_BOOK))"
    assert all(statuses[name].startswith("skipped (") and "is not built" in statuses[name] for name in TUTORIALS)
