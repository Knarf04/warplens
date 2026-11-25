"""First-use and manual builds of the shim libraries, with a stand-in compiler and no CUDA
dependency: where they are built, when a build is trusted, and what a failed one leaves."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from warplens import cli
from warplens.host import runtime, shim_server


@pytest.fixture
def build_tree(tmp_path, monkeypatch):
    """Run the real build script with a deterministic compiler standing in for cc."""
    source = tmp_path / "installed" / "shim"
    source.mkdir(parents=True)
    for name in ("build.sh", "cudart_shim.c", "cuda_shim.c", "protocol.h"):
        shutil.copy2(cli.SHIM_DIR / name, source / name)
    include = tmp_path / "toolkit" / "include"
    include.mkdir(parents=True)
    (include / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 13000\n")
    nvcc = include.parent / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("#!/bin/sh\nexit 0\n")
    nvcc.chmod(0o755)
    compiler = tmp_path / "cc"
    compiler.write_text("#!/bin/sh\n"
                        'while [ "$#" -gt 0 ]; do\n'
                        '  if [ "$1" = "-o" ]; then shift; out="$1"; fi\n'
                        '  shift\n'
                        'done\n'
                        'case "$out" in *libcuda.so.1) [ "${FAIL_DRIVER:-}" != 1 ] || exit 1;; esac\n'
                        'printf compiled > "$out"\n')
    compiler.chmod(0o755)
    monkeypatch.setenv("CC", str(compiler))
    monkeypatch.setenv("PATH", str(nvcc.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(runtime, "SHIM_DIR", source)
    return source, include


def test_first_use_builds_in_user_cache_and_reuses_it(build_tree):
    source, _ = build_tree
    library = cli.shim_libraries()
    assert library != source / "lib", "installed package directories can be read-only"
    assert (library / "libcuda.so.1").is_file()
    first_build = (library / "libcuda.so.1").stat().st_mtime_ns
    assert cli.shim_libraries() == library
    assert (library / "libcuda.so.1").stat().st_mtime_ns == first_build
    # A changed protocol must not reuse a binary from an older source tree.
    with (source / "protocol.h").open("a") as stream:
        stream.write("\n/* new source */\n")
    assert cli.shim_libraries() != library


def test_incomplete_local_build_is_not_accepted(build_tree):
    source, _ = build_tree
    (source / "lib").mkdir()
    (source / "lib" / "libcudart.so.13").write_text("incomplete")
    library = cli.shim_libraries()
    assert (library / "libcuda.so.1").is_file()


def test_cached_build_changes_with_toolkit_headers(build_tree):
    _, include = build_tree
    first = cli.shim_libraries()
    (include / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 14000\n")
    second = cli.shim_libraries()
    assert second != first
    assert (second / "libcudart.so.14").is_file()


def test_manual_build_failure_preserves_existing_libraries(build_tree, monkeypatch):
    source, include = build_tree
    library = source / "lib"
    library.mkdir()
    for name in ("libcuda.so.1", "libcudart.so.13"):
        (library / name).write_text("previous good binary")
    monkeypatch.setenv("FAIL_DRIVER", "1")
    result = subprocess.run([str(source / "build.sh"), str(include)], capture_output=True, text=True)
    assert result.returncode != 0
    assert (library / "libcudart.so.13").read_text() == "previous good binary"
    assert (library / "libcuda.so.1").read_text() == "previous good binary"


def test_manual_build_failure_does_not_leave_a_partial_pair(build_tree, monkeypatch):
    source, include = build_tree
    monkeypatch.setenv("FAIL_DRIVER", "1")
    result = subprocess.run([str(source / "build.sh"), str(include)], capture_output=True, text=True)
    assert result.returncode != 0
    assert not list((source / "lib").glob("*.so.*"))


def test_successful_manual_rebuild_removes_stale_runtime_abis(build_tree):
    source, include = build_tree
    library = source / "lib"
    library.mkdir()
    (library / "libcudart.so.12").write_text("previous runtime ABI")
    (library / "libcuda.so.1").write_text("previous driver")
    result = subprocess.run([str(source / "build.sh"), str(include)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in library.glob("*.so.*")) == ["libcuda.so.1", "libcudart.so.13"]
    assert (library / "libcuda.so.1").read_text() == "compiled"


def manual_build(source, include):
    """A build made by hand, into the package's own lib/. Returns the time of its libraries."""
    result = subprocess.run([str(source / "build.sh"), str(include)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return built_at(source / "lib")


def built_at(library):
    return {path.name: path.stat().st_mtime_ns for path in library.glob("*.so.*")}


def test_current_prebuilt_libraries_do_not_need_a_toolchain(build_tree, monkeypatch):
    source, include = build_tree
    first = manual_build(source, include)
    monkeypatch.setenv("PATH", "")
    assert cli.shim_libraries() == source / "lib"
    assert built_at(source / "lib") == first, "a current build must not be rebuilt"
    # Nor do they need the headers they were built against to be installed still.
    shutil.rmtree(include)
    assert cli.shim_libraries() == source / "lib"
    assert built_at(source / "lib") == first


def test_the_build_script_and_the_command_agree_on_the_stamp(build_tree):
    source, include = build_tree
    manual_build(source, include)
    recorded = dict(line.split(" ", 1) for line in (source / "lib" / cli.SHIM_STAMP).read_text().splitlines())
    assert recorded == {**cli.shim_stamp(include), "include": str(include.resolve())}
    assert cli.shim_staleness(source / "lib") is None


@pytest.mark.parametrize("changed", ["cudart_shim.c", "cuda_shim.c", "protocol.h", "build.sh"])
def test_a_manual_build_of_older_sources_is_rebuilt(build_tree, capsys, changed):
    """What made four tests fail after a merge: lib/ held libraries of the sources before it."""
    source, include = build_tree
    first = manual_build(source, include)
    with (source / changed).open("a") as stream:
        stream.write("\n# a later source\n" if changed == "build.sh" else "\n/* a later source */\n")
    assert "was built from other shim sources" in cli.shim_staleness(source / "lib")
    assert cli.shim_libraries() == source / "lib"
    message = capsys.readouterr().err
    assert "was built from other shim sources" in message and "rebuilding it" in message
    rebuilt = built_at(source / "lib")
    assert all(rebuilt[name] > first[name] for name in first)
    assert cli.shim_staleness(source / "lib") is None
    assert cli.shim_libraries() == source / "lib"  # and now it is current: no second build
    assert built_at(source / "lib") == rebuilt and capsys.readouterr().err == ""


def test_a_stale_manual_build_is_refused_when_it_cannot_be_rebuilt(build_tree, monkeypatch):
    source, include = build_tree
    first = manual_build(source, include)
    with (source / "protocol.h").open("a") as stream:
        stream.write("\n/* a later source */\n")
    monkeypatch.setenv("FAIL_DRIVER", "1")  # no working compiler
    with pytest.raises(SystemExit) as stop:
        cli.shim_libraries()
    message = str(stop.value)
    assert "was built from other shim sources" in message and "could not be rebuilt" in message
    assert "The stale libraries are not used" in message and "build.sh" in message
    assert built_at(source / "lib") == first


def test_libraries_without_a_stamp_are_not_trusted(build_tree, monkeypatch, capsys):
    """Libraries from a build.sh that recorded nothing: nothing says they are current."""
    source, _ = build_tree
    library = source / "lib"
    library.mkdir()
    for name in ("libcuda.so.1", "libcudart.so.13"):
        (library / name).write_text("prebuilt")
    assert cli.shim_libraries() == library  # rebuilt, with the toolkit of the nvcc on PATH
    assert f"has no {cli.SHIM_STAMP}" in capsys.readouterr().err
    assert (library / "libcuda.so.1").read_text() == "compiled" and cli.shim_staleness(library) is None
    (library / cli.SHIM_STAMP).unlink()
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SystemExit, match="could not be rebuilt"):
        cli.shim_libraries()


def test_a_manual_build_is_rebuilt_when_its_cuda_headers_changed(build_tree, capsys):
    source, include = build_tree
    manual_build(source, include)
    (include / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 14000\n")
    assert cli.shim_libraries() == source / "lib"
    assert "which have changed since" in capsys.readouterr().err
    assert sorted(path.name for path in (source / "lib").glob("*.so.*")) == ["libcuda.so.1", "libcudart.so.14"]


def test_a_cached_build_without_its_stamp_is_built_again(build_tree):
    """An interrupted replacement leaves libraries and no stamp."""
    library = cli.shim_libraries()
    first = built_at(library)
    assert cli.shim_libraries() == library and built_at(library) == first
    (library / cli.SHIM_STAMP).unlink()
    assert cli.shim_libraries() == library
    assert all(built_at(library)[name] > first[name] for name in first)


def test_a_run_checks_the_libraries_once_and_a_launch_never(build_tree, monkeypatch):
    """Start-up cost: the sources are hashed when the libraries are looked up, which a run
    does once; the server that handles the launches does not look at them."""
    source, include = build_tree
    manual_build(source, include)
    calls = []
    stamp = cli.shim_stamp
    monkeypatch.setattr(runtime, "shim_stamp", lambda *a: calls.append(a) or stamp(*a))
    assert cli.shim_libraries() == source / "lib"
    assert len(calls) == 2  # the sources, and the headers the build recorded
    text = Path(shim_server.__file__).read_text()
    assert "shim_libraries" not in text and "shim_stamp" not in text and "shim_staleness" not in text
