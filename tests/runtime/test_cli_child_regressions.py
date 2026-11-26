"""CLI child lifetime regressions (exit status, signals, reaping, sockets), without a CUDA dependency."""

import os
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

    monkeypatch.setattr(subprocess, "Popen", popen)
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
        # The child creates the file and then writes its pid: a file that exists may still be
        # empty (CI read it so once, and int("") failed the test).
        def written() -> str:
            return pidfile.read_text() if pidfile.exists() else ""
        while not written() and wrapper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert written(), "the CLI's child did not become ready"
        child_pid = int(written())
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
