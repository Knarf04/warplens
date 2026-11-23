"""CLI child lifetime and first-use shim build regressions, without a CUDA dependency."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from warplens import cli
from warplens.host import shim_server


def args(program, *arguments):
    return SimpleNamespace(program=str(program), arguments=list(arguments), model="B200",
                           timing=False, quiet=True)


@pytest.fixture
def empty_shims(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(cli, "shim_libraries", lambda: lib)
    return lib


def test_cli_runs_an_existing_bare_local_executable(tmp_path, monkeypatch, empty_shims):
    program = tmp_path / "local-program"
    program.write_text("#!/bin/sh\nexit 23\n")
    program.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    assert cli.run(args(program.name)) == 23


def test_cli_maps_child_signal_to_shell_status(empty_shims):
    assert cli.run(args(sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)")) == 143


@pytest.mark.parametrize("failure", [RuntimeError("server failed"), KeyboardInterrupt()])
def test_cli_reaps_child_and_closes_both_sockets_on_server_failure(monkeypatch, empty_shims, failure):
    children, sockets = [], []
    real_popen, real_pair = subprocess.Popen, cli.socket.socketpair

    def popen(*arguments, **options):
        child = real_popen(*arguments, **options)
        children.append(child)
        return child

    def pair():
        result = real_pair()
        sockets.extend(result)
        return result

    def fail(*_):
        raise failure

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.socket, "socketpair", pair)
    monkeypatch.setattr(shim_server.ShimServer, "serve", fail)
    try:
        with pytest.raises(type(failure)):
            cli.run(args(sys.executable, "-c", "import time; time.sleep(60)"))
        assert children[0].poll() is not None, "the failed CLI left its child running"
        assert all(sock.fileno() == -1 for sock in sockets)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        for sock in sockets:
            sock.close()


def test_cli_reports_exec_error_without_traceback(tmp_path, empty_shims):
    program = tmp_path / "not-executable"
    program.write_text("hello")
    with pytest.raises(SystemExit, match="warplens: cannot execute.*not-executable"):
        cli.run(args(program))


def test_cli_closes_sockets_when_exec_fails(tmp_path, monkeypatch, empty_shims):
    sockets = []
    real_pair = cli.socket.socketpair

    def pair():
        result = real_pair()
        sockets.extend(result)
        return result

    monkeypatch.setattr(cli.socket, "socketpair", pair)
    try:
        with pytest.raises((OSError, SystemExit)):
            cli.run(args(tmp_path))  # directories cannot be executed
        assert all(sock.fileno() == -1 for sock in sockets)
    finally:
        for sock in sockets:
            sock.close()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("close_channel", [False, True])
def test_signalling_cli_terminates_and_reaps_its_child(tmp_path, signum, close_channel):
    """Exercise interruption both while serving RPCs and while waiting for exit."""
    library = tmp_path / "lib"
    library.mkdir()
    pidfile = tmp_path / "child.pid"
    child_source = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        + ('os.close(int(os.environ["WARPLENS_SHIM_FD"])); ' if close_channel else "")
        + f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    source = ("import pathlib, sys; from warplens import cli; "
              f"cli.shim_libraries = lambda: pathlib.Path({str(library)!r}); "
              f"sys.exit(cli.main(['run', '--quiet', {sys.executable!r}, '-c', {child_source!r}]))")
    wrapper = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and wrapper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists(), "the CLI's child did not become ready"
        child_pid = int(pidfile.read_text())
        wrapper.send_signal(signum)
        _, stderr = wrapper.communicate(timeout=5)
        assert wrapper.returncode == 128 + signum
        assert b"Traceback" not in stderr
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)  # no running child or zombie left behind
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
        wrapper.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
    monkeypatch.setattr(cli, "SHIM_DIR", source)
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


def test_complete_prebuilt_libraries_do_not_need_a_toolchain(build_tree, monkeypatch):
    source, _ = build_tree
    library = source / "lib"
    library.mkdir()
    for name in ("libcuda.so.1", "libcudart.so.13"):
        (library / name).write_text("prebuilt")
    monkeypatch.setenv("PATH", "")
    assert cli.shim_libraries() == library
