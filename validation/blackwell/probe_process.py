"""Bounded Linux probe processes, including cleanup when their runner is interrupted."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

_active = set()
_lock = threading.RLock()
_interrupted = 0
_main_spawning = False


def _descendant_groups(children):
    """Snapshot detached descendants too: nested runners create new process groups."""
    groups = {child.pid for child in children}
    pending = list(groups)
    visited = set()
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            groups.add(os.getpgid(pid))
            # A Popen in a worker belongs to that task's children list, not the leader's.
            for listing in Path(f"/proc/{pid}/task").glob("*/children"):
                try:
                    pending.extend(map(int, listing.read_text().split()))
                except (OSError, ValueError):
                    pass
        except ProcessLookupError:
            pass
    # Never signal our own group, even if a child deliberately joined it.
    groups.discard(os.getpgrp())
    return groups


def _signal_groups(groups, signum):
    for group in groups:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass


def _terminate(children):
    groups = _descendant_groups(children)
    _signal_groups(groups, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A parent and its nested runner can reach their grace deadlines together. Kill
    # detached descendants first, allowing the nested runner to reap and exit before
    # killing its group; otherwise a TERM-ignoring grandchild can outlive both runners.
    _signal_groups(groups - {child.pid for child in children}, signal.SIGKILL)
    deadline = time.monotonic() + 1
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    _signal_groups(groups, signal.SIGKILL)


@contextmanager
def forward_interrupts():
    """CLI-only context: TERM/INT stop detached probes before this runner exits."""
    global _interrupted
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(signum, _frame):
        global _interrupted
        if _interrupted:
            return  # Do not interrupt cleanup on a repeated TERM/INT.
        _interrupted = signum
        with _lock:
            children = tuple(_active)
        _terminate(children)
        # Popen may already have spawned a process but not returned it. Let that
        # critical section finish so the process is registered and can be reaped.
        if not _main_spawning:
            raise SystemExit(128 + signum)

    _interrupted = 0
    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        interrupted, _interrupted = _interrupted, 0
        if interrupted:
            raise SystemExit(128 + interrupted)


def run_process(command, timeout, env=None, cwd=None):
    """Return (status, elapsed seconds, stdout bytes, stderr bytes)."""
    global _main_spawning
    start = time.monotonic()
    child = None
    main = threading.current_thread() is threading.main_thread()
    try:
        if main:
            _main_spawning = True
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, cwd=cwd, start_new_session=True)
            with _lock:
                _active.add(child)
        except OSError as error:
            return f"not run ({error.strerror})", 0.0, b"", str(error).encode()
        finally:
            if main:
                _main_spawning = False
        if _interrupted:
            _terminate([child])
            if main:
                raise SystemExit(128 + _interrupted)
        try:
            out, err = child.communicate(timeout=timeout)
            code = child.returncode
            status = "ok" if code == 0 else f"exit {code}" if code > 0 else f"signal {-code}"
        except subprocess.TimeoutExpired:
            _terminate([child])
            out, err = child.communicate()
            status = "timeout"
        return status, time.monotonic() - start, out, err
    finally:
        if child is not None:
            with _lock:
                _active.discard(child)
            if child.poll() is None:
                _terminate([child])
            child.wait()
