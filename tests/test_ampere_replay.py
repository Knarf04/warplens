"""The reruns of validation/ampere in the Blackwell probe suite predict what Ampere gave.

Every program of validation/ampere that recorded a golden file from the RTX 3080 Ti has a
run in validation/blackwell that executes the same source on the B200 and prints the same
lines. The simulator reproduces the golden files (tests/test_hardware_golden.py), so the
committed prediction of such a run, which is what the program prints under `warplens run`,
must be the golden file. This file checks that from the committed files alone: if it holds,
a line that differs on the B200 is a difference between Blackwell and Ampere.

It also checks that no golden file is without a run, and that the PTX the reconvergence
probe carries is what validation/ampere/reconvergence_golden.py generates.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AMPERE = ROOT / "validation" / "ampere"
SUITE = ROOT / "validation" / "blackwell"
PREDICTED = SUITE / "predicted"
sys.path.insert(0, str(SUITE))
import manifest  # noqa: E402

def _whole(lines):
    return lines


def _sorted(lines):
    """Device printf and the host's line reach standard output through different buffers."""
    return sorted(lines)


def _ftz_nan_as_text(lines):
    """formats/ftz prints a NaN result as the word `nan` (its bits go on an info: line)."""
    return [" ".join(words[:4] + ["nan"]) if (int(words[4], 16) & 0x7FFFFFFF) > 0x7F800000 else line
            for line, words in ((line, line.split()) for line in lines)]


# golden file -> (run of the manifest, the source the run's program includes, what of the
# golden file the run prints, what of the prediction is compared)
REPLAYED = {
    "shfl_golden.txt": ("e2e/ampere_replay.shfl", "shfl_golden.cu", _whole, _whole),
    "printf_golden.txt": ("e2e/ampere_replay.printf", "printf_hw.cu", _whole, _sorted),
    "ldmatrix_golden.txt": ("e2e/ampere_replay.ldmatrix", "ldmatrix_golden.cu", _whole, _whole),
    "prmt_golden.txt": ("e2e/ampere_replay.prmt", "prmt_golden.cu", _whole, _whole),
    "cpasync_golden.txt": ("e2e/ampere_replay.cpasync", "cpasync_golden.cu", _whole, _whole),
    "call_golden.txt": ("e2e/ampere_replay.call", "call_golden.cu", _whole, _whole),
    "callforms_golden.txt": ("e2e/ampere_replay.callforms", "callforms_golden.cu", _whole, _whole),
    "vote_golden.txt": ("e2e/ampere_replay.vote", "vote_golden.cu", _whole, _whole),
    "atom_golden.txt": ("e2e/ampere_atom", "atom_golden.cu", _whole, _whole),
    "reconvergence_golden.txt": ("e2e/ampere_reconvergence", "reconvergence_kernels.h", _whole, _whole),
    "mbarrier_token_golden.txt": ("mbarrier/ampere_token", "mbarrier_token_golden.cu", _whole, _whole),
    "minmax_cvt_golden.txt": ("formats/ampere_replay.minmax_cvt", "minmax_cvt_golden.cu", _whole, _whole),
    "intdiv_golden.txt": ("formats/ampere_replay.intdiv", "intdiv_golden.cu", _whole, _whole),
    "bitfield_golden.txt": ("formats/ampere_replay.bitfield", "bitfield_golden.cu", _whole, _whole),
    "nan_golden.txt": ("formats/ampere_replay.nan", "nan_golden.cu", _whole, _whole),
    "cvt_ftz_golden.txt": ("formats/ampere_replay.cvt_ftz", "cvt_ftz_golden.cu", _whole, _whole),
    "ftz_golden.txt": ("formats/ftz", "ftz_golden.cu", _ftz_nan_as_text, _whole),
}
# Golden files a run covers in another way, and where that is checked.
OTHERWISE = {
    "vote_legacy_golden.txt": "vote without .sync: ptxas refuses it for sm_70 and newer, so nothing built for "
                              "sm_100a can hold it; tests/test_hardware_golden.py::test_vote_matches_real_gpu",
    "tf32_golden.txt": "formats/tf32_modes prints the same inputs with the .rn and .rz columns beside .rna; "
                       "tests/test_blackwell_probes.py::test_predicted_rna_column_is_what_the_rtx_3080_ti_gave",
}


def _compared(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.startswith("info:")]  # as compare.py does


def test_every_golden_file_has_a_run_on_the_b200():
    """A golden file added to validation/ampere must be given a run here (or a reason)."""
    recorded = {path.name for path in AMPERE.glob("*_golden.txt")}
    assert recorded == set(REPLAYED) | set(OTHERWISE)
    for run, *_ in REPLAYED.values():
        probe = manifest.BY_NAME[run]
        assert probe.predict == "sim" and probe.compare == "exact" and probe.ids, run
        assert not probe.risk, f"{run}: nothing an Ampere GPU ran may be expected to fault or hang"


@pytest.mark.parametrize("golden", sorted(REPLAYED))
def test_the_b200_prediction_is_the_ampere_golden_file(golden):
    run, _, of_golden, of_prediction = REPLAYED[golden]
    predicted = (PREDICTED / (run + ".txt")).read_text()
    want = of_golden((AMPERE / golden).read_text().splitlines())
    got = of_prediction(_compared(predicted))
    differing = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
    assert len(got) == len(want) and not differing, (
        f"predicted/{run}.txt is not validation/ampere/{golden}: {len(got)} lines against {len(want)}"
        + (f", first difference at line {differing[0] + 1}:\n  predicted: {got[differing[0]]}\n"
           f"  Ampere:    {want[differing[0]]}" if differing else ""))
    if of_golden is _whole and of_prediction is _whole:
        assert predicted == (AMPERE / golden).read_text()  # byte for byte: no info: lines, same order
    assert (PREDICTED / (run + ".sim.txt")).read_text().splitlines()[0] == "status: ok"


@pytest.mark.parametrize("golden", sorted(REPLAYED))
def test_the_run_includes_the_source_that_recorded_the_golden_file(golden):
    """Included, not copied: the replay and the Ampere program cannot drift apart."""
    run, source, *_ = REPLAYED[golden]
    assert (AMPERE / source).is_file()
    program = (SUITE / (manifest.BY_NAME[run].program + ".cu")).read_text()
    assert f'#include "../../ampere/{source}"' in program, f"{run} does not include validation/ampere/{source}"


def test_the_nan_run_predicts_the_conversions_with_ftz_too():
    """`cvt` with `.ftz` was once a run of its own without a prediction, because the simulator
    did not implement it. Its 44 lines are now among the predicted ones."""
    predicted = (PREDICTED / "formats" / "ampere_replay.nan.txt").read_text().splitlines()
    assert len(predicted) == 14270
    assert sum(line.startswith(("cvt.ftz.f64.f32 ", "cvt.rn.ftz.f32.f64 ")) for line in predicted) == 44
    assert "formats/ampere_replay.nan_cvt_ftz" not in manifest.BY_NAME


def test_mbarrier_token_replay_reports_the_invalid_uses():
    """The token run repeats Ampere's answers for stale and made-up tokens, which are invalid
    PTX use: the simulator must say so next to the prediction, and nothing else."""
    note = (PREDICTED / "mbarrier" / "ampere_token.sim.txt").read_text()
    kinds = {line.split("]")[0].strip().lstrip("[") for line in note.splitlines() if line.strip().startswith("[")}
    assert kinds == {"mbarrier-invalid-token"}


def test_reconvergence_header_is_what_the_golden_script_writes():
    """validation/ampere/reconvergence_kernels.h carries the PTX of reconvergence_golden.py
    to the probe. It is committed so that building needs no Python; it must not go stale."""
    spec = importlib.util.spec_from_file_location("reconvergence_golden", AMPERE / "reconvergence_golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert (AMPERE / "reconvergence_kernels.h").read_text() == golden.header(), \
        "run `python validation/ampere/reconvergence_golden.py --header > validation/ampere/reconvergence_kernels.h`"
    # The literals spell the modules the recording loaded, in the order of the golden file.
    names = [line.split()[0] for line in (AMPERE / "reconvergence_golden.txt").read_text().splitlines()]
    assert names == list(golden.kernels())
    text = golden.header()
    for name, body in golden.kernels().items():
        literal = "".join(f'        "{line}\\n"\n' for line in (golden.HEADER + body).splitlines())
        assert f'{{"{name}",\n{literal.rstrip()}}}' in text, name


def test_the_ampere_dry_run_builds_every_replay():
    """build.sh --ampere must build every rerun, so that each can be held against its golden
    file on the development machine's GPU."""
    script = (SUITE / "build.sh").read_text()
    listed = script[script.index("AMPERE_PROBES=("):].split(")", 1)[0].split()
    for run, *_ in REPLAYED.values():
        assert manifest.BY_NAME[run].program in listed, run
