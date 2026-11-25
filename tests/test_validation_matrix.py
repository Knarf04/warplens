"""docs/validation-matrix.md must stay well formed: see tools/check_validation_matrix.py."""

import importlib.util
import re
import sys

from tests.support import REPO as ROOT

spec = importlib.util.spec_from_file_location("check_validation_matrix",
                                              ROOT / "tools" / "check_validation_matrix.py")
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker  # dataclasses looks the module up while the class is built
spec.loader.exec_module(checker)

TEXT = checker.MATRIX.read_text()
ROWS, MALFORMED = checker.parse(TEXT)


def test_the_matrix_is_well_formed():
    assert checker.check(TEXT) == []


def test_every_area_has_rows_and_ids_are_unique():
    assert not MALFORMED
    assert {row.area for row in ROWS} == set(checker.AREAS)
    assert len({row.id for row in ROWS}) == len(ROWS)


def test_claims_the_plan_names_are_present():
    """Plan step 6 names four claims taken from review; later work added the others."""
    claims = " ".join(row.cells["claim"] for row in ROWS)
    for phrase in ("Undersized `expect_tx`", "larger than the previous one", "UE8M0",
                   "Either CTA of a pair may issue", "tmem-dealloc-repeated", "`.ftz`",
                   "Calls to device functions", "cuTensorMapEncodeTiled"):
        assert phrase in claims, phrase


def test_every_entry_of_the_timing_config_has_a_row():
    config = (ROOT / "warplens" / "configs" / "b200.yaml").read_text()
    keys = re.findall(r"^(\w+):\n  value:", config, flags=re.M)
    assert len(keys) >= 12
    where = " ".join(row.cells["where"] for row in ROWS if row.area == "TIM")
    for key in keys:
        assert f"b200.yaml:{key}`" in where, key


def test_the_checker_catches_what_it_claims_to():
    assert checker.check_citation("warplens/units/tma.py:_load") is None
    assert checker.check_citation("tests/test_tma.py::test_phases_flip_parity") is None
    assert "no such path" in checker.check_citation("warplens/units/nothing.py")
    assert "does not occur" in checker.check_citation("warplens/units/tma.py:_no_such_function")
    row = next(line for line in TEXT.splitlines() if line.startswith("| TMA-01 |"))
    assert any("also the id" in e for e in checker.check(TEXT + "\n" + row))
    assert any("columns" in e for e in checker.check(TEXT.replace(row, row + " extra |")))
    broken = row.replace("**bit-exact**", "exact")
    assert broken != row
    assert any("criterion must start" in e for e in checker.check(TEXT.replace(row, broken)))
