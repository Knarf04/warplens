"""Check that docs/validation-matrix.md is well formed, and print its row counts.

    python tools/check_validation_matrix.py            # errors, then the counts
    python tools/check_validation_matrix.py --summary  # the two summary tables, ready to paste

What is checked:

* every row has the seven columns, none empty, and an id that is unique and belongs to
  the section it is in;
* the basis starts with a known tag, and an `ampere` basis cites a file in validation/ampere/;
* the probe names a planned location in one of the agreed probe directories;
* the pass criterion starts with **bit-exact**, **tolerance** or **informational**, and a
  tolerance states a number;
* every repository path cited anywhere in the document exists, and for `path:symbol` or
  `path::test` the symbol occurs in that file;
* the summary tables in the document agree with the rows.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "docs" / "validation-matrix.md"

COLUMNS = ("id", "claim", "where", "basis", "probe", "predicted", "criterion")
AREAS = {"TMA": "TMA", "MB": "mbarrier", "TM": "TMEM", "MMA": "MMA (tcgen05)", "CL": "clusters",
         "NUM": "numerics and small formats", "RT": "launch and runtime", "TIM": "timing"}
BASES = ("ampere", "b200", "cutlass", "ptx-manual", "kernel-output", "review", "assumption", "published")
CRITERIA = ("bit-exact", "tolerance", "informational")
PROBE = re.compile(r"`(validation/blackwell/(?:tma|mbarrier|tmem|mma|cluster|formats|e2e)/[a-z0-9_]+"
                   r"|validation/blackwell_timing/[a-z0-9_]+)`")
ROW = re.compile(r"^\|\s*([A-Z]+)-(\d\d)\s*\|")
CITED = re.compile(r"`((?:warplens|tests|docs|validation|examples|tools)/[^`\s]*|PLAN\.md[^`\s]*)`")


@dataclass
class Row:
    line: int
    section: str
    cells: dict[str, str]

    @property
    def id(self) -> str:
        return self.cells["id"]

    @property
    def area(self) -> str:
        return self.id.split("-")[0]

    @property
    def basis(self) -> str:
        return next((b for b in BASES if self.cells["basis"].startswith(f"`{b}`")), "")

    @property
    def criterion(self) -> str:
        return next((c for c in CRITERIA if self.cells["criterion"].startswith(f"**{c}**")), "")


def parse(text: str) -> tuple[list[Row], list[str]]:
    """The rows of the matrix, and the lines that look like rows but are malformed."""
    rows, errors, section = [], [], ""
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            section = line[3:].strip()
        if not ROW.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            errors.append(f"line {number}: {cells[0]} has {len(cells)} columns, expected {len(COLUMNS)}")
            continue
        rows.append(Row(number, section, dict(zip(COLUMNS, cells))))
    return rows, errors


def check_citation(token: str) -> str | None:
    """None if the cited path (and symbol, if any) exists; otherwise what is wrong."""
    if any(ch in token for ch in "*{<"):
        return None  # a pattern, not one path
    if token.startswith(("validation/blackwell/", "validation/blackwell_timing/")):
        return None  # planned probe locations; their form is checked per row
    path, _, symbol = token.partition(":")
    symbol = symbol.lstrip(":")
    target = ROOT / path
    if not target.exists():
        return f"`{token}`: no such path {path}"
    if symbol:
        if not target.is_file():
            return f"`{token}`: {path} is not a file"
        text = target.read_text(errors="replace")
        for part in symbol.split("."):
            if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(part)}(?![A-Za-z0-9_])", text):
                return f"`{token}`: `{part}` does not occur in {path}"
    return None


def counts(rows: list[Row]) -> tuple[Counter, Counter]:
    return Counter(r.area for r in rows), Counter(r.criterion for r in rows)


def summary(rows: list[Row]) -> str:
    by_area, by_criterion = counts(rows)
    lines = ["| Area | Ids | Rows |", "| --- | --- | --- |"]
    lines += [f"| {name} | `{prefix}-nn` | {by_area[prefix]} |" for prefix, name in AREAS.items()]
    lines += [f"| **total** | | {len(rows)} |", "",
              "| Pass criterion | Rows |", "| --- | --- |"]
    lines += [f"| {c} | {by_criterion[c]} |" for c in CRITERIA]
    return "\n".join(lines)


def check(text: str) -> list[str]:
    rows, errors = parse(text)
    seen: dict[str, int] = {}
    for row in rows:
        where = f"line {row.line}: {row.id}"
        if row.id in seen:
            errors.append(f"{where} is also the id of line {seen[row.id]}")
        seen[row.id] = row.line
        if row.area not in AREAS:
            errors.append(f"{where}: unknown area prefix")
        elif f"`{row.area}-nn`" not in row.section:
            errors.append(f"{where} is under the section \"{row.section}\"")
        for name, cell in row.cells.items():
            if not cell:
                errors.append(f"{where}: the {name} column is empty")
        if not row.basis:
            errors.append(f"{where}: the basis must start with one of {', '.join(BASES)} in backticks")
        if row.basis == "ampere" and "`validation/ampere/" not in row.cells["basis"]:
            errors.append(f"{where}: an ampere basis must cite its golden file")
        if row.basis == "b200" and "`validation/blackwell/measured/b200/" not in row.cells["basis"] \
                and "`validation/blackwell_timing/measured/b200/" not in row.cells["basis"]:
            errors.append(f"{where}: a b200 basis must cite its recorded output under measured/b200/")
        if not re.search(r"`warplens/", row.cells["where"]):
            errors.append(f"{where}: no code citation")
        if not PROBE.search(row.cells["probe"]):
            errors.append(f"{where}: the probe names no planned location in an agreed directory")
        if not row.criterion:
            errors.append(f"{where}: the criterion must start with one of {', '.join(CRITERIA)} in bold")
        if row.criterion == "tolerance" and not re.search(r"\d", row.cells["criterion"]):
            errors.append(f"{where}: a tolerance must state a number")
    if not rows:
        errors.append("no rows found")
    for number, line in enumerate(text.splitlines(), 1):
        for token in CITED.findall(line):
            problem = check_citation(token.rstrip(".,;"))
            if problem:
                errors.append(f"line {number}: {problem}")
    if summary(rows) not in text:
        errors.append("the summary tables do not match the rows; paste the output of --summary")
    return errors


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = MATRIX.read_text()
    rows, _ = parse(text)
    if "--summary" in argv:
        print(summary(rows))
        return 0
    errors = check(text)
    for error in errors:
        print(error)
    by_area, by_criterion = counts(rows)
    print(f"{len(rows)} rows: " + ", ".join(f"{p} {by_area[p]}" for p in AREAS)
          + "; " + ", ".join(f"{c} {by_criterion[c]}" for c in CRITERIA))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
