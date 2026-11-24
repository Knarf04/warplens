"""What a built wheel and sdist carry. `pip install -e .` runs the checkout, so it cannot show
a file that is missing from a wheel; these tests can.

Three kinds of test, named for what they need:
  - none: the files the imported package reads through paths next to its modules are there.
    CI's `dist` job runs this with the wheel, and with the sdist, installed in place of the
    checkout;
  - `build` (the `dev` extra; skipped without it): the wheel and the sdist are built from this
    tree, and their file lists and metadata are read;
  - `build` and git: the sdist is the checked-in tree, and holds nothing that git ignores.

Building takes setuptools: the one already installed when it is new enough for
pyproject.toml, and otherwise one that `build` downloads into an environment of its own.
"""

import email
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import warplens
from warplens import cli
from warplens.timing import config

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(warplens.__file__).resolve().parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

# Every file that is not a module of an ordinary package and that the package opens or runs at
# run time. The wheel is also compared with the whole source tree below; this list is what
# must be there even if that comparison were to go wrong.
RUNTIME_FILES = [
    "configs/b200.yaml",                # warplens/timing/config.py
    "configs/b300.yaml",
    "shim/build.sh",                    # warplens/cli.py, shim_libraries: run on first use
    "shim/cudart_shim.c",
    "shim/cuda_shim.c",
    "shim/protocol.h",
    "shim/build_cute_dsl.sh",           # warplens/cli.py, cute_dsl_runtime
    "shim/python/sitecustomize.py",     # put on the PYTHONPATH of the programs `warplens run` starts
]
SCRIPTS = [name for name in RUNTIME_FILES if name.endswith(".sh")]


def source_version() -> str:
    """`__version__` as the checkout has it (the imported package may be an installed one)."""
    (version,) = re.findall(r'^__version__ = "([^"]+)"$', (ROOT / "warplens" / "__init__.py").read_text(), re.M)
    return version


def checkout_files() -> tuple[set[str], set[str]] | None:
    """(tracked, new): the files git has, and the files it neither has nor ignores, as paths
    from ROOT. A tracked file must be shipped; a new one may be (a test not yet added is, the
    pytest.log that CI writes into the checkout is not). None where ROOT is not the top of a
    git checkout (an unpacked sdist) or there is no git."""
    if not shutil.which("git") or not (ROOT / ".git").exists():
        return None
    listed = []
    for which in (["--cached"], ["--others", "--exclude-standard"]):
        done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *which], capture_output=True, text=True)
        if done.returncode:
            return None
        listed.append({name for name in done.stdout.split("\0") if name and (ROOT / name).is_file()})
    return listed[0], listed[1]


def package_source_files() -> tuple[set[str], set[str]]:
    """(what the wheel has to hold, what it may also hold): every file under warplens/ that
    is not a build output; without git, every file there but those known to be built."""
    def of_the_package(names):
        return {name for name in names if name.startswith("warplens/") and Path(name).name != ".gitignore"}

    listed = checkout_files()
    if listed is None:
        found = {path.relative_to(ROOT).as_posix() for path in (ROOT / "warplens").rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and (ROOT / "warplens" / "shim" / "lib") not in path.parents}
        return of_the_package(found), set()
    return of_the_package(listed[0]), of_the_package(listed[1])


# -- the imported package ------------------------------------------------------------------------

def test_the_package_has_the_files_it_reads_at_run_time():
    missing = [name for name in RUNTIME_FILES if not (PACKAGE / name).is_file()]
    assert not missing, f"missing from {PACKAGE}: {missing}"
    assert cli.SHIM_DIR == PACKAGE / "shim" and config.CONFIG_DIR == PACKAGE / "configs"
    assert config.GpuConfig.load("B200")["sm_count"] and config.GpuConfig.load("B300")["sm_count"]


def test_the_build_scripts_of_the_package_can_be_run():
    """`warplens run` executes them directly; a wheel that dropped the mode would fail there."""
    for name in SCRIPTS:
        assert (PACKAGE / name).stat().st_mode & 0o111, f"{PACKAGE / name} is not executable"


# -- pyproject.toml ------------------------------------------------------------------------------

def test_the_version_is_written_in_one_place():
    project = PYPROJECT["project"]
    assert "version" not in project and "version" in project["dynamic"]
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "warplens.__version__"}
    assert str(Version(source_version())) == source_version()  # in the form PyPI will show it


def test_the_changelog_has_a_section_for_the_version():
    headings = re.findall(r"^## (.+)$", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert any(heading.split()[0] == source_version() for heading in headings), headings


def test_the_supported_pythons_are_the_ones_ci_runs():
    project = PYPROJECT["project"]
    listed = sorted(c.rsplit(" ", 1)[1] for c in project["classifiers"]
                    if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c))
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    (matrix,) = re.findall(r'^ +python: \[("3\.\d+"(?:, "3\.\d+")*)\]$', workflow, re.M)
    assert listed == sorted(re.findall(r"3\.\d+", matrix))
    assert project["requires-python"] == f">={listed[0]}"


def test_the_extra_for_everything_names_every_other_extra():
    extras = PYPROJECT["project"]["optional-dependencies"]
    (everything,) = (Requirement(r) for r in extras["all"])
    assert everything.name == "warplens" and everything.extras == set(extras) - {"all"}


# -- the wheel and the sdist ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """(wheel, sdist), built as for a release: the sdist from this tree, the wheel from the sdist.

    The build runs in a directory that stands for the checkout: its directories linked, its
    files copied. setuptools writes warplens.egg-info next to pyproject.toml and, when one is
    already there, takes the files its SOURCES.txt lists into the sdist again: an egg-info left
    in the checkout by an earlier build would hide a file that MANIFEST.in no longer names."""
    pytest.importorskip("build", reason="needs `build` (pip install build, or the dev extra)")
    out = tmp_path_factory.mktemp("dist")
    tree = tmp_path_factory.mktemp("tree")
    for entry in ROOT.iterdir():
        if entry.name in (".git", "build", "dist") or entry.name.endswith(".egg-info"):
            continue
        if entry.is_dir():
            (tree / entry.name).symlink_to(entry, target_is_directory=True)
        else:
            shutil.copy2(entry, tree / entry.name)
    command = [sys.executable, "-m", "build", "--outdir", str(out), str(tree)]
    try:
        import setuptools
        wanted = [Requirement(r) for r in PYPROJECT["build-system"]["requires"]]
        if all(r.name == "setuptools" and r.specifier.contains(setuptools.__version__) for r in wanted):
            command.insert(3, "--no-isolation")  # no download: what is installed will do
    except ImportError:
        pass
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800, cwd=tree)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    (wheel,) = out.glob("*.whl")
    (sdist,) = out.glob("*.tar.gz")
    return wheel, sdist


def test_the_wheel_holds_every_file_of_the_package(built):
    with zipfile.ZipFile(built[0]) as wheel:
        names = set(wheel.namelist())
    assert not [name for name in RUNTIME_FILES if f"warplens/{name}" not in names]
    source, new = package_source_files()
    assert len(source) > len(RUNTIME_FILES)
    missing = sorted(source - names)
    assert not missing, f"not in the wheel (add them to package-data in pyproject.toml): {missing}"
    # And nothing else: no tests or examples in site-packages, no build output of this machine.
    info = f"warplens-{source_version()}.dist-info/"
    extra = sorted(name for name in names - source - new if not name.startswith(info))
    assert not extra, f"in the wheel but not in the source tree: {extra}"


def test_the_wheel_keeps_the_build_scripts_executable(built):
    with zipfile.ZipFile(built[0]) as wheel:
        for name in SCRIPTS:
            mode = wheel.getinfo(f"warplens/{name}").external_attr >> 16
            assert mode & 0o111, f"{name} has mode {mode:o} in the wheel"


def test_the_wheel_says_what_it_is(built):
    info = f"warplens-{source_version()}.dist-info/"
    with zipfile.ZipFile(built[0]) as wheel:
        metadata = email.message_from_bytes(wheel.read(info + "METADATA"))
        entry_points = wheel.read(info + "entry_points.txt").decode()
        assert info + "licenses/LICENSE" in wheel.namelist()
    project = PYPROJECT["project"]
    assert built[0].name == f"warplens-{source_version()}-py3-none-any.whl"  # no compiled code in it
    assert (metadata["Name"], metadata["Version"]) == ("warplens", source_version())
    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata["Requires-Python"] == project["requires-python"]
    assert metadata["Summary"] == project["description"]
    assert set(metadata.get_all("Provides-Extra")) == set(project["optional-dependencies"])
    required = [r for r in metadata.get_all("Requires-Dist") if "extra ==" not in r]
    assert sorted(Requirement(r).name for r in required) == ["numpy", "pyyaml"]
    assert metadata.get_payload().strip() == (ROOT / "README.md").read_text().strip()
    assert re.search(r"^\[console_scripts\]\nwarplens = warplens\.cli:main$", entry_points, re.M)


def test_the_sdist_is_the_checked_in_tree(built):
    """The suite runs from an unpacked sdist only if all of it is there (MANIFEST.in), and a
    build output or a download of this machine must not be shipped in it."""
    listed = checkout_files()
    if listed is None:
        pytest.skip("not a git checkout: nothing to compare the sdist with")
    tracked, new = listed
    top = f"warplens-{source_version()}/"
    with tarfile.open(built[1]) as sdist:
        members = [member.name for member in sdist.getmembers() if member.isfile()]
    assert all(name.startswith(top) for name in members)
    names = {name[len(top):] for name in members}
    generated = {name for name in names
                 if name in ("PKG-INFO", "setup.cfg") or name.startswith("warplens.egg-info/")}
    missing = sorted(tracked - names)
    assert not missing, f"not in the sdist (see MANIFEST.in): {missing}"
    extra = sorted(names - tracked - new - generated)
    assert not extra, f"in the sdist but ignored by git (see MANIFEST.in): {extra}"
