"""The paths of env/remote/run.sh that a rental takes only when something goes wrong, or only
on a machine that is not the development one: Ctrl-C under `| tee`, processes a stage leaves
behind, a CUTLASS clone that is not there yet, a checkout that belongs to another user, a
setup that only warned, the timing suite's own word about its counters, the kernel's log.

Everything here runs stand-ins in throwaway checkouts: no installs, no network, no GPU. The
stand-ins are real processes, pipes and signals; what they stand in for (the suites, ncu,
dmesg, github.com) is named in each test.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.remote import (REMOTE, REPO, archive_of, bash, checkout, clean, fake_book, fake_tool, git, inherited,
                                  machine, manifest_of, members, read_pins, ready_suites, rental,
                                  setups_record, stand_in_cutlass, suite)


def wait_for(path: Path, seconds: float = 20) -> None:
    for _ in range(int(seconds / 0.05)):
        if path.exists() and path.read_text().strip():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} did not appear")


def gone(pid: int, seconds: float = 0) -> bool:
    """No such process, or one that has ended and only waits to be reaped."""
    for _ in range(int(seconds / 0.05) + 1):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------------------------
# Ctrl-C under `run.sh 2>&1 | tee run.log`, the runbook's own command
# ---------------------------------------------------------------------------------------------

def test_ctrl_c_under_tee_still_writes_the_archive(checkout, tmp_path):
    """A terminal's Ctrl-C goes to the whole foreground group: run.sh, which traps it, and tee,
    which dies of it. run.sh's next line then goes into a pipe nobody reads. That used to be
    its last (SIGPIPE), before anything was packed."""
    root, env, args = checkout
    suite(root, "blackwell", f'echo $$ > "{tmp_path}/started"\nsleep 300 & echo $! > "{tmp_path}/pid"\nwait\n',
          "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", f'touch "{tmp_path}/timing-ran"\n')
    read_end, write_end = os.pipe()
    run = subprocess.Popen(["bash", str(root / "env/remote/run.sh"), *args], env={**inherited(), **env},
                           stdout=write_end, stderr=write_end, process_group=0)
    tee = subprocess.Popen(["tee", str(tmp_path / "run.log")], stdin=read_end, stdout=subprocess.DEVNULL,
                           process_group=run.pid)
    os.close(read_end)
    os.close(write_end)
    try:
        wait_for(tmp_path / "pid")
        os.killpg(run.pid, signal.SIGINT)
        assert run.wait(timeout=90) == 130
        assert tee.wait(timeout=10) == -signal.SIGINT           # the reader did die of it
    finally:
        for process in (run, tee):
            if process.poll() is None:
                process.kill()
    seen = (tmp_path / "run.log").read_text()
    assert "build-blackwell: started" in seen and "archive:" not in seen     # tee was gone by then
    files = members(archive_of(tmp_path / "home"))
    assert "build-blackwell\tinterrupted\t130" in files["stages.tsv"].decode()
    assert "stage_build_blackwell=interrupted" in files["manifest.txt"].decode()
    assert not (tmp_path / "timing-ran").exists()
    assert gone(int((tmp_path / "pid").read_text()))            # stopped before the archive, not after
    # What the terminal no longer showed is in the run directory, and the lines before in the archive.
    console = (tmp_path / "home/run/logs/run.console.log").read_text()
    assert "the terminal or pipe this run was writing to is gone" in console
    assert re.search(r"^run: archive:  .*\.tar\.gz$", console, re.M) and "run: sha256:" in console
    assert "run: stages: " in console and "build-blackwell=interrupted" in console
    assert b"build-blackwell: started" in files["logs/run.console.log"]


def test_the_runbook_pipes_through_a_tee_that_ignores_ctrl_c():
    runbook = (REMOTE / "RUNBOOK.md").read_text()
    commands = re.findall(r"^env/remote/run\.sh.*\|\s*tee\b.*$", runbook, re.M)      # the lines to type
    assert commands and all(re.search(r"\|\s*tee -i ", line) for line in commands), commands


# ---------------------------------------------------------------------------------------------
# What a stage started is gone when the stage is over
# ---------------------------------------------------------------------------------------------

# The timing suite runs each benchmark under `timeout`, which puts its child into a process
# group of its own: a signal to the stage's group does not reach it.
INNER_TIMEOUT = ('timeout 300 bash -c \'echo $$ > "$1"; exec sleep 300\' _ "{inner}" &\n'
                 'echo $$ > "{started}"\nwait\n')


def test_a_stage_at_its_limit_leaves_no_process_behind(checkout, tmp_path):
    root, env, args = checkout
    inner = tmp_path / "inner-pid"
    suite(root, "blackwell", INNER_TIMEOUT.format(inner=inner, started=tmp_path / "started"), "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", f'[[ ! -e /proc/$(cat "{inner}") ]] && echo clear > "{tmp_path}/clear"\n')
    done = bash(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BUILD_MINUTES": "0.03"})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "build-blackwell: timeout" in done.stdout
    assert re.search(r"build-blackwell: \d+ process\(es\) it started were still running after it ended; stopping them",
                     done.stdout)
    assert gone(int(inner.read_text()))
    # Gone before the next stage started, which is what a benchmark on the same GPU needs.
    assert "run-timing: ok" in done.stdout and (tmp_path / "clear").read_text() == "clear\n"


def test_an_interrupt_stops_what_the_stage_started_before_packing(checkout, tmp_path):
    root, env, args = checkout
    inner = tmp_path / "inner-pid"
    suite(root, "blackwell", INNER_TIMEOUT.format(inner=inner, started=tmp_path / "started"), "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    process = subprocess.Popen(["bash", str(root / "env/remote/run.sh"), *args], env={**inherited(), **env},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_for(inner)
    wait_for(tmp_path / "started")
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=90)
    assert process.returncode == 130, output
    assert gone(int(inner.read_text())), output
    assert "build-blackwell: interrupted" in output and "archive:" in output


def test_a_file_that_changes_while_it_is_packed_does_not_fail_the_archive(checkout, tmp_path):
    """GNU tar exits 1 for "file changed as we read it"; the archive is whole. Stood in for by
    a tar that packs for real and then says so."""
    root, env, args = checkout
    ready_suites(root, tmp_path)
    real = shutil.which("tar")
    env["PATH"] = fake_tool(tmp_path, "tar", f'echo call >> "{tmp_path}/tar-calls"\n"{real}" "$@" || exit 2\n'
                            'echo "tar: run/logs/run-timing.log: file changed as we read it" >&2\nexit 1\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "archive:" in done.stdout and "archive or checksum failed" not in done.stdout
    assert "changed while it was packed" in done.stdout and "file changed as we read it" in done.stdout
    assert (tmp_path / "tar-calls").read_text() == "call\ncall\n"        # tried once more first
    assert "logs/run-timing.log" in members(archive_of(tmp_path / "home"))


# ---------------------------------------------------------------------------------------------
# CUTLASS: fetched by setup.sh, built before the preflight, never fetched while the GPU stages run
# ---------------------------------------------------------------------------------------------

CUTLASS_FILES = ("examples/cute/tutorial/blackwell/01_mma_sm100.cu", "include/cutlass/cutlass.h",
                 "examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py")


@pytest.fixture
def cutlass_upstream(tmp_path):
    """A stand-in for CUTLASS's repository: one commit without the tutorials, two with them.
    Served over file://, by commit, as github.com serves it."""
    source = tmp_path / "cutlass-upstream"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "uploadpack.allowAnySHA1InWant", "true")
    commits = []
    for step, paths in enumerate((("README.md",), CUTLASS_FILES, ("CHANGELOG.md",))):
        for path in paths:
            (source / path).parent.mkdir(parents=True, exist_ok=True)
            (source / path).write_text(f"stand-in {step}\n")
        git(source, "add", ".")
        git(source, "commit", "-q", "-m", f"step {step}")
        commits.append(git(source, "rev-parse", "HEAD"))
    return source, commits


def remote_copy(tmp_path: Path) -> Path:
    """env/remote in a directory of its own, so that lib.sh's checkout is a throwaway one."""
    root = tmp_path / "copy"
    shutil.copytree(REMOTE, root / "env" / "remote")
    return root


def lib_in(root: Path, code: str, **env: str):
    done = clean("-c", 'source "$1"; eval "$2"', "_", str(root / "env/remote/lib.sh"), code,
                 env={"WARPLENS_REMOTE_HOME": str(root.parent / "home"), **env})
    return done.returncode, done.stdout.strip(), done.stderr


def test_cutlass_is_fetched_at_exactly_the_pinned_commit(tmp_path, cutlass_upstream):
    source, (bare, first, pinned) = cutlass_upstream
    root = remote_copy(tmp_path)
    clone = root / "examples/03_cutlass_mma/.cutlass"
    url = {"WARPLENS_CUTLASS_URL": f"file://{source}"}
    nowhere = {"WARPLENS_CUTLASS_URL": f"file://{tmp_path}/no-such-repository"}

    # Absent and no network: said, with the address and the commit, and nothing half left.
    status, out, _ = lib_in(root, f"ensure_cutlass {pinned}", **nowhere)
    assert status == 1 and out.startswith(f"failed|could not fetch commit {pinned}") and "no-such-repository" in out
    assert not clone.exists() and not Path(str(clone) + ".part").exists()

    # Absent: that commit alone, without history, into the checkout where the suites look.
    status, out, err = lib_in(root, f"ensure_cutlass {pinned}", **url)
    assert status == 0 and out.startswith("fetched|"), out + err
    assert git(clone, "rev-parse", "HEAD") == pinned and git(clone, "rev-list", "--count", "HEAD") == "1"
    assert all((clone / path).is_file() for path in CUTLASS_FILES) and not Path(str(clone) + ".part").exists()
    assert lib_in(root, 'is_cutlass_clone "$CUTLASS_DIR" && git_head "$CUTLASS_DIR"')[1] == pinned

    # Present: nothing is fetched (there is nowhere to fetch from).
    status, out, _ = lib_in(root, f"ensure_cutlass {pinned}", **nowhere)
    assert status == 0 and out.startswith("present|")

    # A commit that is not CUTLASS with its tutorials is not put in the place of a good clone.
    status, out, _ = lib_in(root, f"ensure_cutlass {bare}", **url)
    assert status == 1 and out.startswith("failed|") and "lacks the Blackwell tutorials" in out
    assert git(clone, "rev-parse", "HEAD") == pinned

    # At another commit and the pinned one cannot be had: left as it was.
    status, out, _ = lib_in(root, f"ensure_cutlass {first}", **nowhere)
    assert status == 1 and f"left as it was, at commit {pinned}" in out and git(clone, "rev-parse", "HEAD") == pinned
    # With the network back it is replaced.
    status, out, _ = lib_in(root, f"ensure_cutlass {first}", **url)
    assert status == 0 and out.startswith("fetched|") and git(clone, "rev-parse", "HEAD") == first


def test_a_developers_clone_with_history_is_checked_out_not_replaced(tmp_path, cutlass_upstream):
    source, (_, first, pinned) = cutlass_upstream
    root = remote_copy(tmp_path)
    clone = root / "examples/03_cutlass_mma/.cutlass"
    clone.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(source), str(clone)], check=True)
    git(clone, "checkout", "-q", "--detach", first)
    (clone / "notes.txt").write_text("mine\n")
    status, out, _ = lib_in(root, f"ensure_cutlass {pinned}", WARPLENS_CUTLASS_URL=f"file://{tmp_path}/nowhere")
    assert status == 0 and out.startswith("present|")
    assert git(clone, "rev-parse", "HEAD") == pinned and git(clone, "rev-list", "--count", "HEAD") == "3"
    assert (clone / "notes.txt").read_text() == "mine\n"


def test_a_broken_clone_is_not_taken_for_the_checkouts_commit(checkout):
    """A .git that is not a whole repository (a fetch cut short): git, asked there, answers for
    the repository the directory lies in. That must not read as the clone's commit."""
    root, _, _ = checkout
    clone = root / "examples/03_cutlass_mma/.cutlass"
    (clone / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    outer = git(root, "rev-parse", "HEAD")
    assert lib_in(root, 'git_head "$CUTLASS_DIR"')[1] not in (outer, read_pins()["CUTLASS_COMMIT"])
    assert lib_in(root, 'git_head "$CUTLASS_DIR"')[1] == "unknown"
    assert lib_in(root, 'git_head "$REPO_DIR"')[1] == outer


FAKE_GIT = '''echo "$*" >> "{log}"
case "$*" in
  *"rev-parse HEAD") echo "{commit}" ;;
  *) echo "fake git: not expected: $*" >&2; exit 97 ;;
esac
'''
FAKE_NVCC = '''out=""; source=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift ;; *.cu) source="$1" ;; esac
  shift
done
case "$source" in *"${FAKE_NVCC_FAILS:-no-such-source}"*) echo "nvcc: error in $source" >&2; exit 1 ;; esac
echo "compiled from $source" > "$out"
'''


def example_copy(tmp_path: Path, clone: bool) -> Path:
    example = tmp_path / "example"
    example.mkdir()
    shutil.copy(REPO / "examples" / "03_cutlass_mma" / "build.sh", example / "build.sh")
    if clone:
        (example / ".cutlass").mkdir()
    return example


def test_build_sh_never_fetches_a_clone_that_is_there(tmp_path):
    """The real build.sh with a git that only knows the clone's commit and an nvcc that writes
    its output file: nothing is cloned, fetched or checked out, with one compile at a time
    and with four."""
    pin = read_pins()["CUTLASS_COMMIT"]
    example = example_copy(tmp_path, clone=True)
    log = tmp_path / "git-calls"
    fake_tool(tmp_path, "git", FAKE_GIT.format(log=log, commit=pin))
    path = fake_tool(tmp_path, "nvcc", FAKE_NVCC)
    for jobs in ("1", "4"):
        shutil.rmtree(example / "bin", ignore_errors=True)
        done = bash(example / "build.sh", "--programs", env={"PATH": path, "WARPLENS_CUTLASS_JOBS": jobs})
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stdout.splitlines() == [line for n in range(1, 5) for line in
                                            (f"built tutorial_0{n}.ptx from CUTLASS {pin[:7]}", f"built bin/tutorial_0{n}")]
        assert sorted(p.name for p in (example / "bin").iterdir()) == [f"tutorial_0{n}" for n in range(1, 5)]
        assert "02_mma_tma_sm100.cu" in (example / "bin/tutorial_02").read_text()
    assert set(log.read_text().splitlines()) == {"-C .cutlass rev-parse HEAD"}
    # Without --programs: PTX only, as before.
    shutil.rmtree(example / "bin")
    done = bash(example / "build.sh", env={"PATH": path})
    assert done.returncode == 0 and not (example / "bin").exists() and (example / "tutorial_04.ptx").is_file()

    # A tutorial that does not compile fails the build, and leaves nothing that looks built.
    for jobs in ("1", "4"):
        for built in list(example.glob("*.ptx")) + list(example.glob("bin/*")):
            built.unlink()
        done = bash(example / "build.sh", "--programs",
                    env={"PATH": path, "WARPLENS_CUTLASS_JOBS": jobs, "FAKE_NVCC_FAILS": "03_mma_tma_multicast"})
        assert done.returncode == 1 and "nvcc: error" in done.stdout + done.stderr
        assert not (example / "tutorial_03.ptx").exists() and not (example / "bin/tutorial_03").exists()


def test_build_sh_told_not_to_fetch_fails_at_once_without_a_clone(tmp_path):
    example = example_copy(tmp_path, clone=False)
    log = tmp_path / "git-calls"
    path = fake_tool(tmp_path, "git", FAKE_GIT.format(log=log, commit="0" * 40))
    started = time.monotonic()
    done = bash(example / "build.sh", "--programs", env={"PATH": path, "WARPLENS_CUTLASS_NO_FETCH": "1"})
    assert done.returncode == 1 and time.monotonic() - started < 10
    assert "there is no CUTLASS clone" in done.stderr and "env/remote/setup.sh fetches it" in done.stderr
    assert not log.exists() and sorted(p.name for p in example.iterdir()) == ["build.sh"]
    # A clone that is there at another commit, which it does not have: the same, and not fetched.
    (example / ".cutlass").mkdir()
    done = bash(example / "build.sh", env={"PATH": path, "WARPLENS_CUTLASS_NO_FETCH": "1"})
    assert done.returncode == 1 and "does not have commit" in done.stderr
    assert not [call for call in log.read_text().splitlines() if "fetch" in call or "clone" in call]


def test_run_builds_the_cutlass_programs_before_the_gpu_stages(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", 'echo "no_fetch=${WARPLENS_CUTLASS_NO_FETCH:-unset}" > "$WARPLENS_RESULTS_DIR/e.txt"\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.index("build-cutlass: ok") < done.stdout.index("preflight: skipped") \
        < done.stdout.index("run-blackwell: started")
    files = members(archive_of(tmp_path / "home"))
    # With --programs, in the env, and told not to use the network; so is the GPU stage.
    assert files["logs/build-cutlass.log"] == b"cutlass build: --programs nvcc=nvcc no_fetch=1\n"
    assert files["out/blackwell/e.txt"] == b"no_fetch=1\n"
    assert manifest_of(files)["stage_build_cutlass"] == "ok"
    assert manifest_of(files)["cutlass_commit"] == read_pins()["CUTLASS_COMMIT"]
    example = root / "examples/03_cutlass_mma"
    assert (example / "build-calls").read_text() == "--programs\n"

    # Done is done; but programs that have gone since are built again, and the suite runs again.
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert "build-cutlass: finished earlier, skipping" in again.stdout
    shutil.rmtree(example / "bin")
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert again.returncode == 0 and "build-cutlass: ok" in again.stdout and "run-blackwell: ok" in again.stdout
    assert (example / "build-calls").read_text() == "--programs\n--programs\n"

    # Without the end-to-end programs nothing of CUTLASS is needed.
    for selection in ("--no-e2e", "--only tma/*"):
        probes = bash(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BLACKWELL_RUN_ARGS": selection})
        assert probes.returncode == 0 and "build-cutlass: skipped: not needed" in probes.stdout
    assert (example / "build-calls").read_text() == "--programs\n--programs\n"


def test_cutlass_programs_that_do_not_build_stop_the_run_at_preflight(checkout, tmp_path, machine):
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    fake_book(tmp_path / "home" / "book", read_pins()["BOOK_COMMIT"])
    example = root / "examples/03_cutlass_mma"
    (example / "fail-the-build").touch()
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 3, done.stdout + done.stderr
    assert "build-cutlass: failed" in done.stdout and "the tutorial does not compile" in done.stdout
    assert re.search(r"PASS\s+cutlass\s", done.stdout) and re.search(r"FAIL\s+cutlass-built\s.*tutorial_01", done.stdout)
    assert not (tmp_path / "blackwell-runs").exists()
    # Not something setup.sh provides: the next run builds again and does not repeat setup.
    (example / "fail-the-build").unlink()
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "setup: finished earlier, skipping" in done.stdout and "build-cutlass: ok" in done.stdout
    assert re.search(r"PASS\s+cutlass-built", done.stdout) and (tmp_path / "blackwell-runs").exists()


# ---------------------------------------------------------------------------------------------
# A setup that only warned: the run after the failed preflight repeats it by itself
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("missing,check", [("book", "native-e2e"), ("cutlass", "cutlass")])
def test_setup_runs_again_after_a_preflight_that_failed_for_what_it_provides(checkout, tmp_path, machine,
                                                                             missing, check):
    """setup.sh ends well when a fetch only warned, and its stage is done. The stand-in setup
    fails that fetch the first time and makes it the second, as a network that came back."""
    root, _, _ = checkout
    home = tmp_path / "home"
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok"))
    setups_record(home, "2026.4.0.10")
    clone = root / "examples/03_cutlass_mma/.cutlass"
    kept = {"book": fake_book(tmp_path / "kept-book", read_pins()["BOOK_COMMIT"]), "cutlass": tmp_path / "kept-cutlass"}
    target = {"book": home / "book", "cutlass": clone}
    shutil.move(clone, kept["cutlass"])
    with (root / "env/remote/setup.sh").open("a") as setup:
        for name in ("book", "cutlass"):
            always = "true" if name != missing else f'(( $(wc -l < "{tmp_path}/setup-args") >= 2 ))'
            setup.write(f'if {always} && [[ ! -e "{target[name]}" ]]; then cp -r "{kept[name]}" "{target[name]}"; fi\n')

    first = clean(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 3, first.stdout + first.stderr
    assert "setup: ok" in first.stdout and re.search(rf"FAIL\s+{check}\s", first.stdout)
    assert f"({check}): env/remote/run.sh again runs setup again by itself" in first.stdout
    assert not (tmp_path / "blackwell-runs").exists()

    second = clean(root / "env/remote/run.sh", *args, env=env)        # the same command, nothing added
    assert second.returncode == 0, second.stdout + second.stderr
    assert f"setup: running again: the last preflight failed for want of something setup.sh provides ({check})" in second.stdout
    assert (tmp_path / "setup-args").read_text() == "\n\n"
    assert re.search(rf"PASS\s+{check}\s", second.stdout) and (tmp_path / "blackwell-runs").exists()

    third = clean(root / "env/remote/run.sh", *args, env=env)         # and once is enough
    assert third.returncode == 0 and "setup: finished earlier, skipping" in third.stdout
    assert (tmp_path / "setup-args").read_text() == "\n\n"


def test_a_refused_counter_does_not_repeat_setup(checkout, tmp_path, machine):
    """ncu is there and the driver refuses it: nothing setup.sh could install helps."""
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="permission", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    for _ in range(2):
        done = clean(root / "env/remote/run.sh", *args, env=env)
        assert done.returncode == 3 and re.search(r"FAIL\s+ncu", done.stdout)
        assert "runs setup again" not in done.stdout and "setup: running again" not in done.stdout
    assert (tmp_path / "setup-args").read_text() == "\n"
    # No ncu at all is setup's to provide.
    env = {**env, "PATH": machine(False)["PATH"]}
    done = clean(root / "env/remote/run.sh", *args, "--redo", "preflight", env=env)
    assert done.returncode == 3 and "for want of something setup.sh provides (ncu)" in done.stdout
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert "setup: running again" in done.stdout and (tmp_path / "setup-args").read_text() == "\n\n"


# ---------------------------------------------------------------------------------------------
# Counters: what the timing suite says of its own profiles is what the archive says
# ---------------------------------------------------------------------------------------------

def timing_stand_in(root: Path, tmp_path: Path, status_line: str | None) -> None:
    """The timing suite's part: ncu/STATUS.txt in its results, first line as agreed, and what
    it was handed as NCU_BIN."""
    body = f'#!/usr/bin/env bash\nset -eu\necho "${{NCU_BIN:-unset}}" > "{tmp_path}/ncu-bin"\n'
    if status_line is not None:
        body += (f'mkdir -p "$WARPLENS_RESULTS_DIR/ncu"\n'
                 f'printf "%s\\n" "{status_line}" "second line: detail" > "$WARPLENS_RESULTS_DIR/ncu/STATUS.txt"\n')
    (root / "validation/blackwell_timing/run_all.sh").write_text(body)


@pytest.mark.parametrize("line,verdict,flagged", [
    ("ncu ok: 7 of 7 profiles read counters", "read", False),
    ("ncu partial: 5 of 7 profiles read counters", "partial", False),
    ("ncu failed: 0 of 7 profiles read counters (ERR_NVGPUCTRPERM)", "not-read", True),
    ("ncu not run: no ncu binary", "not-read", True),
    ("something else entirely", "not-read", True),
    (None, "not-read", True),
])
def test_the_archive_says_what_the_timing_suite_read_not_what_preflight_did(checkout, tmp_path, machine,
                                                                            line, verdict, flagged):
    """Preflight read its one counter in every case here. The suite's profiles are what counts."""
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    timing_stand_in(root, tmp_path, line)
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    files = members(archive_of(tmp_path / "home"))
    manifest = manifest_of(files)
    counters = manifest["ncu_counters"]
    assert manifest["ncu"] == "ok" and counters.startswith(verdict + " ("), counters
    assert files["resolved/versions.txt"].decode().splitlines()[-1] == "ncu_counters=" + counters
    if line is None:
        assert "left no ncu/STATUS.txt" in counters
    else:
        assert f'"{line}"' in counters and "second line" not in counters
    assert ("NCU-COUNTERS-NOT-READ.txt" in files) == flagged
    assert ("performance counters: " + verdict in done.stdout) == (verdict != "read")
    if flagged:
        assert "this run REQUIRED them" in counters and counters.encode() in files["NCU-COUNTERS-NOT-READ.txt"]
    # The ncu preflight found is the one the suite is handed.
    ncu = Path(env["PATH"].split(os.pathsep)[0]) / "ncu"
    assert (tmp_path / "ncu-bin").read_text() == f"{ncu}\n" and manifest["ncu_bin"] == str(ncu)
    # Packing again later says the same: the suite's file is still what is read.
    again = clean(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 0 and manifest_of(members(archive_of(tmp_path / "home")))["ncu_counters"] == counters


def test_an_ncu_outside_path_reaches_the_timing_suite(checkout, tmp_path, machine):
    """preflight.sh also looks in /usr/local/cuda/bin and /opt/nvidia/nsight-compute; `ncu` on
    PATH would not find that one again."""
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(False, FAKE_NCU="ok", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    timing_stand_in(root, tmp_path, "ncu ok: 7 of 7 profiles read counters")
    elsewhere = tmp_path / "ncu-tools" / "ncu"                    # the stand-in ncu; not on this PATH
    done = clean(root / "env/remote/run.sh", *args, env={**env, "PREFLIGHT_NCU_CANDIDATES": str(elsewhere)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "ncu-bin").read_text() == f"{elsewhere}\n"
    # One the caller names is left alone.
    mine = tmp_path / "my-ncu"
    done = clean(root / "env/remote/run.sh", *args, "--redo", "run-timing",
                 env={**env, "PREFLIGHT_NCU_CANDIDATES": str(elsewhere), "NCU_BIN": str(mine)})
    assert done.returncode == 0 and (tmp_path / "ncu-bin").read_text() == f"{mine}\n"


# ---------------------------------------------------------------------------------------------
# The self-test: after the GPU stages with the predictions' own compiler, before them otherwise
# ---------------------------------------------------------------------------------------------

def with_selftest(tmp_path: Path, env: dict, args: list) -> tuple[dict, list]:
    """A `python` that takes `-m pytest` for a suite that passes, and is the real one otherwise."""
    path = fake_tool(tmp_path, "python", f'if [[ " $* " == *" -m pytest "* ]]; then echo "cuda=[${{CUDA_VISIBLE_DEVICES-unset}}]" '
                     f'>> "{tmp_path}/selftest-ran"; echo "3046 passed"; exit 0; fi\nexec "{sys.executable}" "$@"\n')
    kept = [part for i in range(0, len(args), 2) if args[i + 1] != "selftest" for part in args[i:i + 2]]
    return {**env, "PATH": path}, kept


def versions(home: Path, state: str) -> None:
    (home / "resolved").mkdir(parents=True, exist_ok=True)
    (home / "resolved" / "versions.txt").write_text(
        f"cuda_nvcc=13.5.37\npredictions_comparable={state}\npredictions_comparable_why=stand-in: {state}\n")


def test_the_selftest_runs_after_the_gpu_stages_with_the_pinned_toolkit(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    env, args = with_selftest(tmp_path, env, args)
    versions(tmp_path / "home", "comparable")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert out.index("selftest: after the GPU stages") < out.index("build-blackwell: started")
    assert out.index("run-timing: ok") < out.index("Packing what they left before the self-test") \
        < out.index("archive:") < out.index("selftest: started") < out.index("selftest: ok") < out.rindex("archive:")
    assert out.count("run: archive:") == 2                       # what was measured is packed before it starts
    assert (tmp_path / "selftest-ran").read_text() == "cuda=[]\n"
    files = members(archive_of(tmp_path / "home"))
    assert manifest_of(files)["stage_selftest"] == "ok" and b"3046 passed" in files["logs/selftest.log"]
    stages = [line.split("\t")[0] for line in files["stages.tsv"].decode().splitlines()[1:]]
    assert stages.index("run-timing") < stages.index("selftest")
    # Finished is finished: the next run packs once.
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert "selftest: finished earlier, skipping" in again.stdout and again.stdout.count("run: archive:") == 1
    # --skip selftest still leaves it out.
    skipped = bash(root / "env/remote/run.sh", *args, "--skip", "selftest", "--redo", "all", env=env)
    assert "selftest: skipped: asked to" in skipped.stdout and skipped.stdout.count("run: archive:") == 1
    assert (tmp_path / "selftest-ran").read_text() == "cuda=[]\n"


@pytest.mark.parametrize("state", ["not-comparable", "comparable-native", None])
def test_the_selftest_stays_before_the_gpu_stages_otherwise(checkout, tmp_path, state):
    """A fallback toolkit: the self-test's failures name the predictions that do not hold for
    the installed compiler, which is worth knowing before the GPU is used. The same when the
    driver refuses the toolkit's PTX, and when no toolkit is recorded."""
    root, env, args = checkout
    ready_suites(root, tmp_path)
    env, args = with_selftest(tmp_path, env, args)
    if state:
        versions(tmp_path / "home", state)
    done = bash(root / "env/remote/run.sh", *args, "--skip", "predict-toolkit", env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.index("selftest: ok") < done.stdout.index("build-blackwell: started")
    assert done.stdout.count("run: archive:") == 1 and "selftest: after the GPU stages" not in done.stdout


# ---------------------------------------------------------------------------------------------
# The kernel's log
# ---------------------------------------------------------------------------------------------

def test_the_kernel_log_is_archived_with_what_is_new_since_before_the_gpu(checkout, tmp_path):
    """dmesg is stood in for by a program that prints a file; the GPU stage adds a fault to it."""
    root, env, args = checkout
    log = tmp_path / "kernel-log"
    old = "[   12.000001] NVRM: loading NVIDIA UNIX x86_64 Kernel Module  590.44.01"
    new = "[ 4711.000002] NVRM: Xid (PCI:0000:01:00): 13, pid=4242, name=tma_probe, Graphics Exception"
    log.write_text(f"[    0.000000] Linux version 6.8.0-test\n{old}\n")
    suite(root, "blackwell", "echo built\n", f'echo "{new}" >> "{log}"\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    env["PATH"] = fake_tool(tmp_path, "dmesg", f'cat "{log}"\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    files = members(archive_of(tmp_path / "home"))
    assert files["kernel/dmesg.txt"].decode() == log.read_text()
    assert files["kernel/dmesg-before-gpu.txt"].decode().splitlines()[-1] == old
    assert files["kernel/dmesg-xid-nvrm.txt"].decode() == f"{old}\n{new}\n"
    assert files["kernel/dmesg-xid-nvrm-new.txt"].decode() == f"{new}\n"
    verdict = "readable: 2 line(s) with Xid or NVRM, 1 of them not in the baseline from before the GPU was used"
    assert manifest_of(files)["kernel_log"] == verdict and f"kernel log: {verdict}" in done.stdout
    assert files["kernel/summary.txt"].decode().startswith(f"kernel_log={verdict}\n")
    # The baseline is the first attempt's: a later run does not take the fault for old.
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert members(archive_of(tmp_path / "home"))["kernel/dmesg-xid-nvrm-new.txt"].decode() == f"{new}\n"
    assert again.returncode == 0


def test_a_kernel_log_that_cannot_be_read_is_said_not_failed(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    fake_tool(tmp_path, "journalctl", "exit 1\n")              # the journal's copy is refused as well
    env["PATH"] = fake_tool(tmp_path, "dmesg", 'echo "dmesg: read kernel buffer failed: Operation not permitted" >&2\nexit 1\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    files = members(archive_of(tmp_path / "home"))
    for name in ("kernel/dmesg.txt", "kernel/dmesg-before-gpu.txt"):
        assert files[name].decode().startswith("not readable: dmesg failed as uid ")
        assert "Operation not permitted" in files[name].decode()
    assert manifest_of(files)["kernel_log"].startswith("not readable: ")
    assert "kernel/dmesg-xid-nvrm.txt" not in files and "kernel log:" not in done.stdout
    assert "nvidia-smi-q.txt" in files

    # Where dmesg is refused and the journal is not, the journal's copy of the kernel log is taken.
    line = "[   12.000001] host kernel: NVRM: loading NVIDIA UNIX x86_64 Kernel Module"
    fake_tool(tmp_path, "journalctl", f'[[ "$*" == "-k "* ]] && echo "{line}"\n')
    done = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    files = members(archive_of(tmp_path / "home"))
    assert files["kernel/dmesg.txt"].decode() == line + "\n" == files["kernel/dmesg-xid-nvrm.txt"].decode()
    assert manifest_of(files)["kernel_log"].startswith("readable: 1 line(s) with Xid or NVRM")


# ---------------------------------------------------------------------------------------------
# A checkout that belongs to another user; a machine whose python3 is not the env's
# ---------------------------------------------------------------------------------------------

def test_a_checkout_of_another_owner_is_still_read_and_frozen(checkout, tmp_path):
    """Uploaded as one user, run as root: git refuses with "dubious ownership". Under fakeroot
    the process believes it is root and the checkout another user's, which is all git looks at."""
    fakeroot = shutil.which("fakeroot")
    if not fakeroot:
        pytest.skip("fakeroot is not installed")
    root, _, _ = checkout
    script = (f'chown -R 4242 "{root}"\n'
              f'git -C "{root}" rev-parse HEAD > "{tmp_path}/plain" 2>&1; echo "plain=$?"\n'
              f'bash -c \'source "$1/env/remote/lib.sh"; repo_git "$1" rev-parse HEAD; echo "branch=$(repo_git "$1" rev-parse --abbrev-ref HEAD)";'
              f' echo "changed=$(repo_git "$1" status --porcelain | wc -l)"; echo "head=$(git_head "$1")"\' _ "{root}"\n'
              f'python3 "{root}/env/remote/freeze_source.py" --repo "{root}" --out "{tmp_path}/frozen"; echo "freeze=$?"\n')
    # The machine's own git configuration is left out: GitHub's runner images trust every
    # directory in the system one (safe.directory = *), and plain git then refuses nothing.
    (tmp_path / "gitconfig").write_text("")
    own = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    done = subprocess.run([fakeroot, "sh", "-c", script], capture_output=True, text=True, timeout=120, env=own)
    lines = done.stdout.splitlines()
    if "plain=0" in lines:
        pytest.skip("git under fakeroot did not see another owner here")
    commit = git(root, "rev-parse", "HEAD")
    assert "dubious ownership" in (tmp_path / "plain").read_text()
    assert lines[1] == commit and f"head={commit}" in lines and "freeze=0" in lines, done.stdout + done.stderr
    assert any(line.startswith("branch=") and line != "branch=" for line in lines)
    assert "changed=1" in lines                                 # the stand-in CUTLASS example, untracked
    assert (tmp_path / "frozen" / "source.tar.gz").is_file()
    assert f'"commit": "{commit}"' in (tmp_path / "frozen" / "manifest.json").read_text()


def test_the_snapshot_is_made_by_the_envs_python_and_else_the_machines(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    # The env's python is tried first; here it is one that cannot run the script at all.
    runner = tmp_path / "env-run"
    runner.write_text(f'#!/usr/bin/env bash\nif [[ "$1" == python && "$2" == *freeze_source.py ]]; then\n'
                      f'  echo "env python called" >> "{tmp_path}/calls"; echo "no python in this env" >&2; exit 127\nfi\nexec "$@"\n')
    runner.chmod(0o755)
    done = bash(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_ENV_RUN": str(runner)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "calls").read_text() == "env python called\n"
    files = members(archive_of(tmp_path / "home"))
    assert "source/source.tar.gz" in files and "SOURCE-SNAPSHOT-FAILED.txt" not in files
    log = files["source-snapshot.log"].decode()
    assert "# with the env's python" in log and "no python in this env" in log and "# with the machine's python3" in log


# ---------------------------------------------------------------------------------------------
# A GPU the probe suite found unhealthy; a probe stage that was cut short
# ---------------------------------------------------------------------------------------------

# The probe suite's part (validation/blackwell/run_all.py): status.tsv as it goes, and
# GPU-UNHEALTHY.txt when its canary no longer runs; --resume takes the file away again.
PROBES_STAND_IN = ('echo "$*" >> "{calls}"\n'
                   'printf "name\\tstatus\\n" > "$WARPLENS_RESULTS_DIR/status.tsv"\n'
                   'if [[ " $* " == *" --resume "* ]]; then rm -f "$WARPLENS_RESULTS_DIR/GPU-UNHEALTHY.txt"; exit 0; fi\n'
                   '{body}')
UNHEALTHY_TEXT = "THE GPU WAS UNHEALTHY AFTER tmem/fault_probe"


def test_an_unhealthy_gpu_stops_the_timing_suite_and_the_next_run_resumes(checkout, tmp_path):
    root, env, args = checkout
    calls = tmp_path / "probe-calls"
    suite(root, "blackwell", "echo built\n", PROBES_STAND_IN.format(
        calls=calls, body=f'echo "{UNHEALTHY_TEXT}" > "$WARPLENS_RESULTS_DIR/GPU-UNHEALTHY.txt"\nexit 1\n'))
    suite(root, "blackwell_timing", "echo built\n", f'echo ran >> "{tmp_path}/timing-runs"\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "run-blackwell: failed" in done.stdout
    assert "run-timing: skipped: the probe suite found the GPU unhealthy" in done.stdout
    assert not (tmp_path / "timing-runs").exists()
    lines = done.stdout.splitlines()
    at = lines.index(UNHEALTHY_TEXT)
    assert lines[at - 1] == "!" * 100 and "the timing suite was NOT started" in lines[at + 1]
    assert "env/remote/run.sh --timing-on-unhealthy-gpu --skip run-blackwell" in done.stdout
    files = members(archive_of(tmp_path / "home"))
    assert manifest_of(files)["gpu_unhealthy"] == UNHEALTHY_TEXT and manifest_of(files)["stage_run_timing"] == "skipped"
    assert files["out/blackwell/GPU-UNHEALTHY.txt"].decode() == UNHEALTHY_TEXT + "\n"
    # Packing again says it again and is not a success.
    packed = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert packed.returncode == 1 and UNHEALTHY_TEXT in packed.stdout

    # The GPU is back: the same command takes the probes up, and the timing suite runs.
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "run-blackwell: taking up the earlier attempt where it stopped" in again.stdout
    assert calls.read_text().splitlines() == ["", "--no-build --resume"]
    assert "run-timing: ok" in again.stdout and UNHEALTHY_TEXT not in again.stdout
    assert manifest_of(members(archive_of(tmp_path / "home")))["gpu_unhealthy"] == "no"


def test_the_timing_suite_runs_on_an_unhealthy_gpu_only_when_told(checkout, tmp_path):
    root, env, args = checkout
    calls = tmp_path / "probe-calls"
    suite(root, "blackwell", "echo built\n", PROBES_STAND_IN.format(
        calls=calls, body=f'echo "{UNHEALTHY_TEXT}" > "$WARPLENS_RESULTS_DIR/GPU-UNHEALTHY.txt"\nexit 1\n'))
    suite(root, "blackwell_timing", "echo built\n", f'echo ran >> "{tmp_path}/timing-runs"\n')
    assert bash(root / "env/remote/run.sh", *args, env=env).returncode == 1
    done = bash(root / "env/remote/run.sh", *args, "--timing-on-unhealthy-gpu", "--skip", "run-blackwell", env=env)
    assert "run-timing: ok" in done.stdout and (tmp_path / "timing-runs").read_text() == "ran\n"
    assert done.returncode == 1 and UNHEALTHY_TEXT in done.stdout       # the probes are still owed
    assert calls.read_text() == "\n"
    # --redo starts the probes over instead of resuming them.
    done = bash(root / "env/remote/run.sh", *args, "--redo", "run-blackwell", env=env)
    assert calls.read_text() == "\n\n" and "taking up the earlier attempt" not in done.stdout


def test_a_probe_stage_cut_short_by_its_limit_is_resumed(checkout, tmp_path):
    root, env, args = checkout
    calls = tmp_path / "probe-calls"
    suite(root, "blackwell", "echo built\n", PROBES_STAND_IN.format(calls=calls, body="sleep 300\n"))
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    first = bash(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_RUN_MINUTES": "0.03"})
    assert first.returncode == 1 and "run-blackwell: timeout" in first.stdout
    assert "run-timing: ok" in first.stdout                 # a limit is not an unhealthy GPU
    again = bash(root / "env/remote/run.sh", *args, env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert calls.read_text().splitlines() == ["", "--no-build --resume"]
    # The caller's own selection is not resumed over.
    suite_dir = root / "validation/blackwell"
    (suite_dir / "run_all.sh").write_text("#!/usr/bin/env bash\n" + PROBES_STAND_IN.format(calls=calls, body="exit 0\n"))
    only = bash(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BLACKWELL_RUN_ARGS": "--only tma/*"})
    assert only.returncode == 0 and calls.read_text().splitlines()[-1] == "--only tma/*"


def test_counters_the_timing_suite_required_and_did_not_read_are_said(checkout, tmp_path, machine):
    """The suite's run_all.sh exits non-zero for this alone when preflight read a counter."""
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    line = "ncu failed: 0 of 9 profiles read counters"
    (root / "validation/blackwell_timing/run_all.sh").write_text(
        '#!/usr/bin/env bash\nmkdir -p "$WARPLENS_RESULTS_DIR/ncu"\n'
        f'echo "{line}" > "$WARPLENS_RESULTS_DIR/ncu/STATUS.txt"\n'
        'echo "run_all: COUNTERS REQUIRED AND NOT READ: WARPLENS_NCU=ok" > "$WARPLENS_RESULTS_DIR/status.txt"\nexit 1\n')
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1 and "run-timing: failed" in done.stdout
    assert "the suite's ncu profiles read none" in done.stdout and line in done.stdout
    files = members(archive_of(tmp_path / "home"))
    assert manifest_of(files)["ncu_counters"].startswith(f'not-read (the timing suite: "{line}"')
    assert "NCU-COUNTERS-NOT-READ.txt" in files
