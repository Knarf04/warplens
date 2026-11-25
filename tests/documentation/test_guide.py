"""Keep `docs/guide.md`, `README.md` and `examples/08_book/book_tour.ipynb` true.

The code in the guide and the README is executed. A fenced block is picked up when the line before it is an HTML
comment (invisible in the rendered page) of one of these forms:

    <!-- test: file NAME -->             the block is written to NAME in the working directory
    <!-- test: run [needs=a,b] [exit=N] -->
                                         a `bash` or `python` block that is executed and must
                                         exit with N (default 0)
    <!-- test: output [unchecked] [head=N] -->
                                         what the preceding `run` block printed. Every line of
                                         it must appear in the block's real output (stdout and
                                         stderr together); a line that is only `...` stands
                                         for lines left out. `unchecked` output is shown but
                                         not compared (numbers that depend on the machine).

Blocks without such a comment are illustrations and are not executed. All blocks run in one
scratch directory, in document order, with `examples/`, `docs/`, `validation/` and `env/`
linked into it, so the paths in the guide are the paths a reader types at the
top of a checkout, and files a block writes do not land in the repository. `warplens` and
`python` resolve to this interpreter.

`needs=` names what a block requires beyond NumPy; the block is skipped without it:

    nvcc       the CUDA compiler on PATH (and so the CUDA headers the shim is built against)
    shim       `warplens run` can start a program: Linux, and either a shim built by hand in
               warplens/shim/lib or what it takes to build one (nvcc's headers, a C compiler)
    cutlass    examples/03_cutlass_mma/build.sh --programs has been run
    triton, tirx, cute_dsl     those packages
    book       WARPLENS_BOOK is a clone of the book (see tests/runtime/test_book.py)

To refresh the outputs shown in the guide after a change in behaviour, run from the checkout

    python -m tests.documentation.test_guide --update

which executes every `run` block and rewrites the `output` block that follows it.

The notebook is executed end to end when WARPLENS_BOOK is set: with Jupyter's machinery when
`nbclient` and `ipykernel` are installed, and otherwise as a plain Python script made of its
code cells, which the notebook is written to allow.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tests.support import REPO as ROOT

GUIDE = ROOT / "docs" / "guide.md"
README = ROOT / "README.md"
REPOSITORY_FILE = "https://github.com/Knarf04/warplens/blob/main/"  # how the README links to a file
NOTEBOOK = ROOT / "examples" / "08_book" / "book_tour.ipynb"
BOOK = os.environ.get("WARPLENS_BOOK", "")
MARK = re.compile(r"<!--\s*test:\s*(file|run|output)\b(.*?)-->\s*$")
LINKED = ("examples", "docs", "validation", "env", "PLAN.md", "README.md")


@dataclass
class Block:
    kind: str  # file, run or output
    document: Path
    options: dict
    language: str
    text: str
    line: int  # of the opening fence, 1-based
    first: int = 0  # index of the first content line in the file's lines
    last: int = 0  # index one past the last content line
    output: "Block | None" = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return f"{self.document.stem}-line{self.line}-{self.language or 'text'}"


def parse(document: Path) -> list[Block]:
    """The marked blocks of a Markdown document, each `run` block with its `output`."""
    lines, blocks, i = document.read_text().splitlines(), [], 0
    while i < len(lines):
        mark = MARK.match(lines[i].strip())
        if not mark or i + 1 >= len(lines) or not lines[i + 1].startswith("```"):
            i += 1
            continue
        words = mark.group(2).split()
        options = dict(w.split("=", 1) if "=" in w else (w, "") for w in words)
        end = next(j for j in range(i + 2, len(lines)) if lines[j].startswith("```"))
        block = Block(mark.group(1), document, options, lines[i + 1][3:].strip(), "\n".join(lines[i + 2:end]) + "\n",
                      i + 2, i + 2, end)
        if block.kind == "output":
            assert blocks and blocks[-1].kind == "run" and blocks[-1].output is None, \
                f"line {block.line}: an output block must follow a run block"
            blocks[-1].output = block
        else:
            blocks.append(block)
        i = end + 1
    return blocks


def available(need: str) -> bool:
    if need == "nvcc":
        return shutil.which("nvcc") is not None
    if need == "shim":
        built = (ROOT / "warplens" / "shim" / "lib" / "libcuda.so.1").is_file()
        can_build = shutil.which("nvcc") and shutil.which(os.environ.get("CC", "cc"))
        return sys.platform.startswith("linux") and bool(built or can_build)
    if need == "cutlass":
        built = ROOT / "examples" / "03_cutlass_mma"
        return (built / "tutorial_04.ptx").is_file() and (built / "bin" / "tutorial_04").is_file()
    if need == "book":
        return bool(BOOK) and (Path(BOOK) / "chapter_gemm_basics" / "index.md").is_file()
    modules = {"triton": ["triton"], "tirx": ["tvm_tirx", "cuda.bindings"], "cute_dsl": ["cutlass"]}[need]
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except (ImportError, ValueError):
        return False


def missing(block: Block) -> list[str]:
    return [n for n in filter(None, block.options.get("needs", "").split(",")) if not available(n)]


def prepare(work: Path, blocks: list[Block]) -> dict:
    """Lay out the scratch directory and return the environment blocks run in."""
    for name in LINKED:
        if not (work / name).exists():
            (work / name).symlink_to(ROOT / name)
    tools = work / ".tools"
    tools.mkdir(exist_ok=True)
    for name, command in (("warplens", f'exec "{sys.executable}" -m warplens.cli "$@"'),
                          ("python", f'exec "{sys.executable}" "$@"')):
        (tools / name).write_text(f"#!/bin/sh\n{command}\n")
        (tools / name).chmod(0o755)
    for block in blocks:
        if block.kind == "file":
            (work / next(iter(block.options))).write_text(block.text)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
           "PATH": f"{tools}{os.pathsep}{os.environ.get('PATH', '')}",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    return env


def execute(block: Block, work: Path, env: dict) -> tuple[int, str]:
    """Run one block; returns its exit status and what it printed."""
    if block.language == "python":
        script = work / ".block.py"
        script.write_text(block.text)
        command = [sys.executable, str(script)]
    elif block.language == "bash":
        command = ["bash", "-euo", "pipefail", "-c", block.text]
    else:
        raise AssertionError(f"line {block.line}: cannot run a `{block.language}` block")
    done = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=1800, stdin=subprocess.DEVNULL)
    return done.returncode, done.stdout


def shown(block: Block, printed: str) -> str:
    """The part of a run's output that its output block shows."""
    lines = [l.rstrip() for l in printed.rstrip().splitlines()]
    head = int(block.options.get("head", 0))
    if head and len(lines) > head:
        lines = lines[:head] + ["..."]
    return "\n".join(lines) + "\n"


def unmatched(expected: str, printed: str) -> list[str]:
    have = {l.strip() for l in printed.splitlines()}
    return [l for l in expected.splitlines() if l.strip() and l.strip() != "..." and l.strip() not in have]


# -- tests ----------------------------------------------------------------------------------

try:
    import pytest
except ImportError:  # `python -m tests.documentation.test_guide --update` needs no pytest
    pytest = None

if pytest is not None:
    BLOCKS = parse(GUIDE) + parse(README)
    RUNS = [b for b in BLOCKS if b.kind == "run"]

    @pytest.fixture(scope="module")
    def scratch(tmp_path_factory):
        work = tmp_path_factory.mktemp("guide")
        return work, prepare(work, BLOCKS)

    def test_the_guide_has_runnable_blocks():
        assert len(RUNS) >= 15 and sum(b.output is not None for b in RUNS) >= 10
        assert any(b.document == README for b in RUNS)

    @pytest.mark.parametrize("block", RUNS, ids=[b.id for b in RUNS])
    def test_guide_block_runs_and_prints_what_the_guide_shows(block, scratch):
        if lacking := missing(block):
            pytest.skip(f"needs {', '.join(lacking)}")
        work, env = scratch
        status, printed = execute(block, work, env)
        where = f"{block.document.name} line"
        assert status == int(block.options.get("exit", 0)), f"{where} {block.line}:\n{printed[-3000:]}"
        if block.output is not None and "unchecked" not in block.output.options:
            absent = unmatched(block.output.text, printed)
            assert not absent, (f"{where} {block.output.line} shows output the block no longer "
                                f"prints:\n" + "\n".join(absent) + f"\n--- it printed:\n{printed[-3000:]}")

    def slug(heading: str) -> str:
        """GitHub's anchor for a heading."""
        heading = re.sub(r"[`*_]", "", heading.strip().lower())
        return re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")

    def anchors(path: Path) -> set[str]:
        text = re.sub(r"^```.*?^```", "", path.read_text(), flags=re.S | re.M)
        return {slug(h) for h in re.findall(r"^#+\s+(.*)$", text, flags=re.M)}

    @pytest.mark.parametrize("document", [README, GUIDE], ids=["README.md", "docs/guide.md"])
    def test_relative_links_resolve(document):
        text = re.sub(r"^```.*?^```", "", document.read_text(), flags=re.S | re.M)
        broken = []
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            # The README is also the project page on PyPI, which does not resolve a link
            # relative to the repository: its links to the repository's files are full URLs,
            # and are checked against the checkout like the relative ones of the guide.
            in_repository = target.startswith(REPOSITORY_FILE)
            if re.match(r"[a-z][a-z0-9+.-]*:", target) and not in_repository:  # http:, https:, mailto:
                continue
            path, _, anchor = target.removeprefix(REPOSITORY_FILE).partition("#")
            file = ((ROOT if in_repository else document.parent) / path).resolve() if path else document
            if not file.exists():
                broken.append(target)
            elif anchor and file.suffix == ".md" and anchor not in anchors(file):
                broken.append(target)
        assert not broken, f"{document.name} links to things that do not exist: {broken}"

    def test_the_readme_has_no_link_that_pypi_cannot_resolve():
        text = re.sub(r"^```.*?^```", "", README.read_text(), flags=re.S | re.M)
        targets = re.findall(r"\]\(([^)\s]+)\)", text)
        assert sum(t.startswith(REPOSITORY_FILE) for t in targets) > 10
        assert not [t for t in targets if not re.match(r"[a-z][a-z0-9+.-]*:", t)]

    def test_every_finding_kind_is_explained_in_the_guide():
        kinds = set()
        for source in (ROOT / "warplens").rglob("*.py"):
            kinds |= set(re.findall(r"""add\(\s*["']([a-z0-9]+(?:-[a-z0-9]+)+)["']""", source.read_text()))
            kinds |= set(re.findall(r"""["']((?:smem|tmem|tma|mma|mbarrier|shfl|barrier|cp-async|tcgen05)"""
                                    r"""-[a-z0-9-]+)["']""", source.read_text()))
        assert len(kinds) > 20
        guide = GUIDE.read_text()
        assert not sorted(k for k in kinds if f"`{k}`" not in guide)

    def test_the_guide_names_both_models_and_every_cli_option():
        guide = GUIDE.read_text()
        from warplens.host.device import MODELS
        for word in (*MODELS, "--model", "--timing", "--quiet"):
            assert word in guide

    # -- the notebook ---------------------------------------------------------------------

    def notebook() -> dict:
        return json.loads(NOTEBOOK.read_text())

    def test_notebook_is_valid_and_modest():
        cells = notebook()["cells"]
        assert notebook()["nbformat"] == 4 and len(cells) > 20
        assert NOTEBOOK.stat().st_size < 600_000, "keep the outputs kept in the notebook modest"
        code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        # Plain Python, so that the test below can run it without Jupyter.
        assert not [line for cell in code for line in cell.splitlines() if line.lstrip().startswith(("%", "!"))]
        # The book has no license: its kernels are read from the reader's clone, never kept here.
        everything = NOTEBOOK.read_text()
        assert "T.device_entry" not in everything and "def hgemm_v" not in everything
        assert "WARPLENS_BOOK" in everything
        # Kept outputs must not carry the paths of whoever ran it.
        assert str(Path.home()) not in everything and "/home/" not in everything
        prose = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
        links = [t for t in re.findall(r"\]\(([^)\s]+)\)", prose) if not re.match(r"[a-z][a-z0-9+.-]*:", t)]
        assert links and not [t for t in links if not (NOTEBOOK.parent / t.partition("#")[0]).exists()]

    def test_notebook_runs_end_to_end(tmp_path, monkeypatch):
        for need in ("book", "tirx", "shim"):
            if not available(need):
                pytest.skip(f"needs {need}")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WARPLENS_BOOK": str(Path(BOOK).resolve()),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
        try:
            jupyter = all(importlib.util.find_spec(m) for m in ("nbclient", "nbformat", "ipykernel"))
        except (ImportError, ValueError):
            jupyter = False
        if jupyter:
            import nbclient
            import nbformat
            # A kernel that is certainly this interpreter, whatever kernels the machine has.
            spec = tmp_path / "kernels" / "warplens-test"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "warplens test", "language": "python"}))
            monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            book = nbformat.read(NOTEBOOK, as_version=4)
            nbclient.NotebookClient(book, kernel_name="warplens-test", timeout=1800,
                                    resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
            printed = "\n".join(o.get("text", "") for c in book.cells if c.cell_type == "code"
                                for o in c.get("outputs", []) if o.get("output_type") == "stream")
        else:
            script = tmp_path / "book_tour.py"
            script.write_text("\n\n".join("".join(c["source"]) for c in notebook()["cells"]
                                          if c["cell_type"] == "code"))
            done = subprocess.run([sys.executable, str(script)], cwd=NOTEBOOK.parent, env=env,
                                  capture_output=True, text=True, timeout=3600)
            assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
            printed = done.stdout
        for version in range(1, 10):
            assert f"hgemm_v{version} " in printed and "matches NumPy" in printed
        assert "tmem-dealloc-repeated" in printed
        assert "Approximate timing on B200 (model, not a measurement)" in printed


def update() -> int:
    """Rewrite every output block of the guide from a real run of the block before it."""
    import tempfile
    lines = GUIDE.read_text().splitlines()
    blocks = parse(GUIDE)
    failed = 0
    with tempfile.TemporaryDirectory(prefix="warplens_guide_") as scratch:
        work = Path(scratch)
        env = prepare(work, blocks)
        replacements = []
        for block in (b for b in blocks if b.kind == "run"):
            if lacking := missing(block):
                print(f"line {block.line}: skipped, needs {', '.join(lacking)}")
                continue
            status, printed = execute(block, work, env)
            ok = status == int(block.options.get("exit", 0))
            failed += not ok
            print(f"line {block.line}: {'ok' if ok else f'EXIT {status}'}")
            if not ok or block.output is None:
                print(printed[-2000:] if not ok else "", end="")
            if block.output is not None:
                replacements.append((block.output, shown(block.output, printed).splitlines()))
        for output, new in sorted(replacements, key=lambda r: -r[0].first):
            lines[output.first:output.last] = new
    GUIDE.write_text("\n".join(lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit(__doc__)
    sys.exit(update())
