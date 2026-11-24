"""Remote orchestration checks using throwaway checkouts; no installs or GPU access."""
import os
import re
import subprocess
from pathlib import Path

import pytest

from test_remote_env import REMOTE, archive_of, bash, checkout, fake_smi, members, read_pins, suite


def ready_suites(root, tmp_path):
    for name in ("blackwell", "blackwell_timing"):
        suite(root, name, f'echo build >> "{tmp_path}/{name}-builds"\n',
              f'echo run >> "{tmp_path}/{name}-runs"\n')


def fake_tool(tmp_path, name, body):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    tool = tools / name
    tool.write_text("#!/usr/bin/env bash\n" + body)
    tool.chmod(0o755)
    return str(tools) + os.pathsep + os.environ["PATH"]


def test_archive_failure_is_not_a_success(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    env["PATH"] = fake_tool(tmp_path, "tar", "echo archive-failed >&2\nexit 75\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "archive:" not in done.stdout
    assert "collect: writing" in done.stdout


def test_rebuilding_a_suite_invalidates_its_successful_run(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    first = bash(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    second = bash(root / "env/remote/run.sh", *args, "--redo", "build-blackwell", env=env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "blackwell-runs").read_text() == "run\nrun\n"
    assert (tmp_path / "blackwell_timing-runs").read_text() == "run\n"


def test_setup_options_are_part_of_the_resume_identity(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    # Exercise setup dispatch without any installer or network access.
    setup = root / "env/remote/setup.sh"
    setup.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/setup-args"\n')
    args = [part for i in range(0, len(args), 2) if args[i + 1] != "setup" for part in args[i:i + 2]]
    for toolkit in ("pinned", "auto"):
        done = bash(root / "env/remote/run.sh", *args, "--toolkit", toolkit, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "setup-args").read_text().splitlines() == ["--toolkit pinned", "--toolkit auto"]
    assert (tmp_path / "blackwell-runs").read_text() == "run\nrun\n"


def test_collect_only_preserves_recorded_gpu_capabilities(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    run_dir = tmp_path / "home/run"
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.status").write_text("ptx-jit PASS\nncu PASS\n")
    first = bash(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    again = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    manifest = members(archive_of(tmp_path / "home"))["manifest.txt"].decode()
    assert "ptx_jit=ok\n" in manifest
    assert "ncu=ok\n" in manifest


def check_module():
    import importlib.util
    from test_remote_env import REMOTE
    spec = importlib.util.spec_from_file_location("remote_check_env", REMOTE / "check_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_e2e_checks_reject_missing_and_cpu_only_torch(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    check = check_module()
    book = tmp_path / "book/chapter_gemm_basics"
    book.mkdir(parents=True)
    (book / "index.md").touch()
    monkeypatch.setenv("WARPLENS_BOOK", str(book.parent))

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name}")

    monkeypatch.setattr(check.importlib, "import_module", missing)
    assert check.e2e() == 1
    assert "native Triton" in capsys.readouterr().out
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None))
    monkeypatch.setattr(check.importlib, "import_module", lambda name: torch)
    assert check.e2e() == 1
    assert "CPU-only" in capsys.readouterr().out
    torch.version.cuda = "13.0"
    torch.__version__ = "test"
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    assert check.e2e() == 0
    monkeypatch.delenv("WARPLENS_BOOK")
    assert check.e2e() == 1
    assert "WARPLENS_BOOK" in capsys.readouterr().out
    assert check.e2e(str(book.parent)) == 0


def test_remote_dependency_gate_tracks_selected_end_to_end_programs(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    (root / "env/remote/preflight.sh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/preflight-args"\n')
    args = [part for i in range(0, len(args), 2) if args[i + 1] != "preflight" for part in args[i:i + 2]]
    for selected, expected in (("", True), ("--no-e2e", False), ("--only tmem/*", False),
                               ("--book /custom/book", True)):
        env["WARPLENS_BLACKWELL_RUN_ARGS"] = selected
        done = bash(root / "env/remote/run.sh", *args, "--redo", "preflight", env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        line = (tmp_path / "preflight-args").read_text().splitlines()[-1]
        assert ("--e2e" in line) == expected
        if selected.startswith("--book"):
            assert "--book /custom/book" in line


def test_archive_checksum_failure_is_not_a_success(checkout, tmp_path):
    import shutil
    root, env, args = checkout
    ready_suites(root, tmp_path)
    real = shutil.which("sha256sum")
    env["PATH"] = fake_tool(tmp_path, "sha256sum",
        f'if [[ "${{1:-}}" == *.tar.gz ]]; then exit 75; fi\nexec "{real}" "$@"\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "archive:" not in done.stdout
    assert "archive or checksum failed" in done.stdout


def test_refused_ptx_blocks_full_e2e_but_allows_native_probes(tmp_path):
    from test_remote_env import REMOTE, fake_smi
    runner = tmp_path / "fake-env"
    runner.write_text('''#!/usr/bin/env python3
import pathlib
import sys
args = sys.argv[1:]
if args[0] == "timeout":
    args = args[1:]
    if args[0].startswith("--kill-after="):
        args = args[1:]
    args = args[1:]
if args[0] == "nvcc":
    if "--version" in args:
        print("Cuda compilation tools, release 13.5, V13.5.37")
    else:
        pathlib.Path(args[args.index("-o") + 1]).write_text(".version 9.5\\n")
elif args[0] == "python" and args[1].endswith("check_env.py"):
    print("nvrtc=13.5\\nnvrtc_ptx_version=9.5")
elif args[0] == "python" and args[1].endswith("gpu_probe.py"):
    if args[2] == "info":
        print("compute_cap=10.0\\nname=NVIDIA B200\\ndriver_api=13.0\\ntotal_mem_mib=192000")
    elif args[2] == "vecadd" and args[3].endswith(".ptx"):
        print("error_name=CUDA_ERROR_UNSUPPORTED_PTX_VERSION\\nerror=PTX version refused")
        sys.exit(1)
    else:
        print("elements=1000\\ntmem_address=0\\nok=1")
else:
    sys.exit(1)
''')
    runner.chmod(0o755)
    env = {"WARPLENS_REMOTE_HOME": str(tmp_path / "home"), "WARPLENS_ENV_RUN": str(runner),
           "NVIDIA_SMI": str(fake_smi(tmp_path, name="NVIDIA B200", cap="10.0"))}
    for e2e in (False, True):
        status = tmp_path / "status"
        # --no-ncu: this stand-in machine reads no counter, which fails a preflight of its own
        # accord since counters became required; the PTX refusal is what is under test here.
        done = bash(REMOTE / "preflight.sh", *( ["--e2e"] if e2e else []), "--no-ncu", "--status", str(status), env=env)
        checks = dict(line.split() for line in status.read_text().splitlines())
        assert checks["ptx-jit"] == "WARN"
        assert checks.get("e2e-ptx") == ("FAIL" if e2e else None)
        assert done.returncode == int(e2e), done.stdout + done.stderr


def test_failed_source_snapshot_still_preserves_raw_results(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    (root / "env/remote/freeze_source.py").write_text('raise RuntimeError("snapshot test failure")\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1
    files = members(archive_of(tmp_path / "home"))
    assert "SOURCE-SNAPSHOT-FAILED.txt" in files
    assert b"snapshot test failure" in files["source-snapshot.log"]
    assert "logs/run-blackwell.log" in files


# ---------------------------------------------------------------------------------------------
# The book: fetched by setup.sh at the pinned commit, the default WARPLENS_BOOK, never archived
# ---------------------------------------------------------------------------------------------

BOOK_MARKER = "TEXT-OF-THE-BOOK-THAT-MUST-NOT-BE-COPIED"
BOOK_VARIABLES = ("WARPLENS_BOOK", "WARPLENS_BOOK_URL", "WARPLENS_BLACKWELL_RUN_ARGS")


def clean(script, *args, env=None, cwd=None):
    """bash(), but without the book the machine running the tests may have exported (CI does)
    and without the ncu it may have installed outside PATH."""
    full = {k: v for k, v in os.environ.items() if k not in BOOK_VARIABLES}
    full.update({"PREFLIGHT_OFFLINE": "1", "PREFLIGHT_NCU_CANDIDATES": "/no/such/ncu", **(env or {})})
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=full, cwd=cwd,
                          timeout=120)


def lib_call(home: Path, code: str, **env: str):
    """Run shell code with lib.sh sourced; returns (exit status, stdout, stderr)."""
    done = clean("-c", 'source "$1"; eval "$2"', "_", str(REMOTE / "lib.sh"), code,
                 env={"WARPLENS_REMOTE_HOME": str(home), **env})
    return done.returncode, done.stdout.strip(), done.stderr


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(directory), "-c", "user.name=fixture", "-c",
                           "user.email=fixture@example.invalid", *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    """A stand-in for the book's repository with three commits: one before the book has its
    GEMM chapter, and two with it. Served over file://, by commit, as github.com serves it."""
    source = tmp_path / "upstream"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "uploadpack.allowAnySHA1InWant", "true")
    commits = []
    for text, path in (("no chapter yet", "README.md"), ("first " + BOOK_MARKER, "chapter_gemm_basics/index.md"),
                       ("second " + BOOK_MARKER, "chapter_gemm_basics/index.md")):
        (source / path).parent.mkdir(exist_ok=True)
        (source / path).write_text(text + "\n")
        git(source, "add", ".")
        git(source, "commit", "-q", "-m", text)
        commits.append(git(source, "rev-parse", "HEAD"))
    return {"WARPLENS_BOOK_URL": f"file://{source}"}, commits


def test_the_book_is_fetched_at_exactly_the_pinned_commit(tmp_path, upstream):
    url, (bare, old, pinned) = upstream
    home = tmp_path / "home"
    book = home / "book"
    nowhere = {"WARPLENS_BOOK_URL": f"file://{tmp_path}/no-such-repository"}

    # Absent: fetched, one commit and no history, and nothing left beside it.
    status, out, err = lib_call(home, f"provide_book 1 {pinned}", **url)
    assert status == 0 and out.startswith("fetched|"), out + err
    assert git(book, "rev-parse", "HEAD") == pinned and git(book, "rev-list", "--count", "HEAD") == "1"
    assert (book / "chapter_gemm_basics/index.md").read_text() == f"second {BOOK_MARKER}\n"
    assert not (home / "book.part").exists()

    # Present at the right commit: nothing is fetched (there is nowhere to fetch from).
    status, out, _ = lib_call(home, f"provide_book 1 {pinned}", **nowhere)
    assert status == 0 and out.startswith("present|") and pinned in out

    # Present at another commit and the fetch fails: said, and what was there is left alone.
    status, out, _ = lib_call(home, f"provide_book 1 {old}", **nowhere)
    assert status == 1 and out.startswith("failed|could not fetch commit " + old)
    assert "no-such-repository" in out and f"left as it was, at commit {pinned}" in out
    assert git(book, "rev-parse", "HEAD") == pinned and not (home / "book.part").exists()

    # Present at another commit: replaced by the pinned one.
    status, out, err = lib_call(home, f"provide_book 1 {old}", **url)
    assert status == 0 and out.startswith("fetched|"), out + err
    assert git(book, "rev-parse", "HEAD") == old
    assert (book / "chapter_gemm_basics/index.md").read_text() == f"first {BOOK_MARKER}\n"

    # A commit that is not the book (no GEMM chapter) is not installed over a good clone.
    status, out, _ = lib_call(home, f"provide_book 1 {bare}", **url)
    assert status == 1 and out.startswith("failed|") and "has no chapter_gemm_basics/index.md" in out
    assert git(book, "rev-parse", "HEAD") == old


def test_a_failed_fetch_of_the_book_leaves_nothing_behind(tmp_path, upstream):
    url, (_, _, pinned) = upstream
    home = tmp_path / "home"
    # Absent and no network: a failure with the address and the commit, and no half clone.
    status, out, _ = lib_call(home, f"provide_book 1 {pinned}",
                              WARPLENS_BOOK_URL=f"file://{tmp_path}/no-such-repository")
    assert status == 1 and out.startswith("failed|could not fetch commit " + pinned)
    assert "is there network access" in out
    assert not (home / "book").exists() and not (home / "book.part").exists()
    assert lib_call(home, "book_path")[1] == ""
    # A commit the repository does not have is the same failure.
    status, out, _ = lib_call(home, "provide_book 1 " + "0" * 40, **url)
    assert status == 1 and out.startswith("failed|") and not (home / "book").exists()
    # The second run of setup.sh, with the network back, gets it.
    status, out, _ = lib_call(home, f"provide_book 1 {pinned}", **url)
    assert status == 0 and out.startswith("fetched|")


def test_the_book_is_not_fetched_when_told_not_to_or_given(tmp_path, upstream):
    url, (_, _, pinned) = upstream
    home = tmp_path / "home"
    status, out, _ = lib_call(home, f"provide_book 0 {pinned}", **url)
    assert status == 0 and out.startswith("opted-out|") and "--no-book" in out
    status, out, _ = lib_call(home, f"provide_book 1 {pinned}", WARPLENS_BOOK="/somewhere/else", **url)
    assert status == 0 and out.startswith("given|") and "/somewhere/else" in out
    assert not (home / "book").exists()
    # setup.sh hands the decision to this function with its own flag and the pin, warns of a
    # failure without stopping, and records the commit that is there.
    setup = (REMOTE / "setup.sh").read_text()
    assert 'provide_book "$with_book" "$PIN_BOOK_COMMIT"' in setup and "--no-book) with_book=0" in setup
    assert re.search(r'\*\) warn "the book: ', setup) and 'echo "book_commit=' in setup
    help_text = clean(REMOTE / "setup.sh", "--help").stdout
    assert "--no-book" in help_text and "WARPLENS_BOOK_URL" in help_text


def fake_book(directory: Path, commit: str | None) -> Path:
    """A directory the scripts take for a clone of the book; `commit` is what its .git says."""
    (directory / "chapter_gemm_basics").mkdir(parents=True)
    (directory / "chapter_gemm_basics" / "index.md").write_text(BOOK_MARKER + "\n")
    if commit:
        (directory / ".git").mkdir()
        (directory / ".git" / "HEAD").write_text(commit + "\n")
    return directory


def test_the_default_book_is_setups_clone_and_a_named_one_wins(tmp_path):
    home = tmp_path / "home"
    assert lib_call(home, "book_path")[1] == ""                              # nothing fetched, nothing named
    assert lib_call(home, "book_path", WARPLENS_BOOK="/my/clone")[1] == "/my/clone"
    fake_book(home / "book", "1" * 40)
    assert lib_call(home, "book_path")[1] == str(home / "book")
    assert lib_call(home, "book_path", WARPLENS_BOOK="/my/clone")[1] == "/my/clone"
    assert lib_call(home, "book_path", WARPLENS_BOOK="")[1] == ""            # set to nothing: no book
    assert lib_call(home, f'book_commit "{home}/book"')[1] == "1" * 40
    # Files of the book with no .git of their own are at no commit, wherever they lie.
    copy = fake_book(tmp_path / "copy", None)
    assert lib_call(home, f'book_commit "{copy}"')[1] == "unknown"


# ---------------------------------------------------------------------------------------------
# Counters: a full run must read one for real
# ---------------------------------------------------------------------------------------------

FAKE_ENV = r'''#!/usr/bin/env python3
"""Stands in for the env runner: a B200 on which everything works, with the outcome of the
trivial kernel taken from FAKE_TRIVIAL. Anything it does not know is run as it is."""
import os
import pathlib
import sys
args = sys.argv[1:]
real = list(args)
if args[0] == "timeout":
    args = args[1:]
    if args[0].startswith("--kill-after="):
        args = args[1:]
    args = args[1:]
if args[0] == "nvcc":
    if "--version" in args:
        print("Cuda compilation tools, release 13.5, V13.5.37")
    else:
        pathlib.Path(args[args.index("-o") + 1]).write_text(".version 9.5\n")
elif args[0] == "python" and args[1].endswith("check_env.py"):
    if args[2] == "e2e":
        if not (pathlib.Path(args[3] or "/nonexistent") / "chapter_gemm_basics" / "index.md").is_file():
            print("error=book GEMMs: set WARPLENS_BOOK to a clone of modern-gpu-programming-for-mlsys")
            sys.exit(1)
    else:
        print("nvrtc=13.5\nnvrtc_ptx_version=9.5")
elif args[0] == "python" and args[1].endswith("gpu_probe.py"):
    if args[2] == "info":
        print("compute_cap=10.0\nname=NVIDIA B200\ndriver_api=13.5\ntotal_mem_mib=192000")
    elif args[2] == "vecadd" and args[3].endswith(".cubin") and os.environ.get("FAKE_TRIVIAL") == "fails":
        print("error=the launch failed")
        sys.exit(1)
    else:
        print("elements=1000\ntmem_address=0\nok=1")
else:
    os.execvp(real[0], real)
'''

FAKE_NCU = r'''#!/usr/bin/env bash
# Stands in for ncu: FAKE_NCU is ok, permission or silent.
if [[ "$1" == --version ]]; then echo "NVIDIA (R) Nsight Compute Command Line Profiler"; echo "Version 2026.4.0.10 (build 1)"; exit 0; fi
echo "==PROF== Connected to process 4242"
case "${FAKE_NCU:-ok}" in
  ok) echo '"0","4242","python3.12","127.0.0.1","pf_vecadd","1","7","(256, 1, 1)","(4, 1, 1)","0","10.0","Command line profiler metrics","smsp__inst_executed.sum","inst","1,234"' ;;
  permission) echo "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM" ;;
  silent) echo "==WARNING== No kernels were profiled." ;;
esac
'''


@pytest.fixture
def machine(tmp_path):
    """The environment of a B200 on which preflight.sh passes every check that is not under
    test: stand-ins for nvidia-smi, the env runner and ncu. `machine(False)` has no ncu."""
    runner = tmp_path / "fake-env"
    runner.write_text(FAKE_ENV)
    runner.chmod(0o755)
    tools = tmp_path / "ncu-tools"
    tools.mkdir()
    (tools / "ncu").write_text(FAKE_NCU)
    (tools / "ncu").chmod(0o755)
    # A PATH on which the only ncu is the stand-in, whatever the machine has installed.
    rest = [d for d in os.environ["PATH"].split(os.pathsep) if d and not (Path(d) / "ncu").exists()]
    env = {"WARPLENS_REMOTE_HOME": str(tmp_path / "home"), "WARPLENS_ENV_RUN": str(runner),
           "NVIDIA_SMI": str(fake_smi(tmp_path, name="NVIDIA B200", cap="10.0"))}

    def with_ncu(present: bool, **more: str) -> dict:
        return {**env, **more, "PATH": os.pathsep.join(([str(tools)] if present else []) + rest)}
    return with_ncu


def preflight(tmp_path, env, *args):
    status = tmp_path / "status"
    done = clean(REMOTE / "preflight.sh", *args, "--status", str(status), env=env)
    checks = dict(line.split() for line in status.read_text().splitlines())
    # Only the disk check may depend on the machine running the tests.
    failed = {check for check, result in checks.items() if result == "FAIL"} - {"disk"}
    return done, checks, failed


@pytest.mark.parametrize("outcome,words", [
    ("absent", "no ncu in the env"),
    ("permission", "may not read performance counters"),
    ("silent", "did not report the counter"),
])
def test_a_counter_that_was_not_read_fails_preflight_unless_told_otherwise(tmp_path, machine, outcome, words):
    env = machine(outcome != "absent", FAKE_NCU=outcome)
    done, checks, failed = preflight(tmp_path, env)
    assert checks["ncu"] == "FAIL" and failed == {"ncu"}, done.stdout
    assert done.returncode == 1 and "preflight: FAILED" in done.stdout
    assert re.search(rf"^FAIL\s+ncu\s.*{words}", done.stdout, re.M)
    assert "env/remote/run.sh --no-ncu" in done.stdout              # the way on, said where it fails
    if outcome == "permission":
        assert "ERR_NVGPUCTRPERM" in done.stdout and "--cap-add=SYS_ADMIN" in done.stdout   # the remedies stay

    done, checks, failed = preflight(tmp_path, env, "--no-ncu")
    assert checks["ncu"] == "WARN" and not failed, done.stdout
    assert re.search(rf"^WARN\s+ncu\s+counters are not needed \(--no-ncu\) and were not read: .*{words}",
                     done.stdout, re.M)
    assert "env/remote/run.sh --no-ncu" not in done.stdout
    if checks["disk"] != "FAIL":
        assert done.returncode == 0


def test_a_counter_that_was_read_passes_either_way(tmp_path, machine):
    for args in ((), ("--no-ncu",)):
        done, checks, failed = preflight(tmp_path, machine(True, FAKE_NCU="ok"), *args)
        assert checks["ncu"] == "PASS" and not failed, done.stdout
        assert "read a performance counter: smsp__inst_executed.sum = 1,234" in done.stdout


def test_ncu_that_had_nothing_to_profile_does_not_count_as_a_counter(tmp_path, machine):
    """The trivial kernel did not run, so ncu was never started: that is not a counter read."""
    env = machine(True, FAKE_NCU="ok", FAKE_TRIVIAL="fails")
    done, checks, failed = preflight(tmp_path, env)
    assert checks["run-trivial"] == "FAIL" and checks["ncu"] == "FAIL", done.stdout
    assert "no counter was read" in done.stdout
    done, checks, _ = preflight(tmp_path, env, "--no-ncu")
    assert checks["run-trivial"] == "FAIL" and checks["ncu"] == "SKIP", done.stdout
    assert done.returncode == 1                                     # the kernel itself still fails it


def rental(root, tmp_path, machine_env):
    """run.sh on the stand-in machine: the real preflight.sh, a setup.sh that records its
    arguments, suites that record what they were given."""
    (root / "env/remote/setup.sh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/setup-args"\n')
    for name in ("blackwell", "blackwell_timing"):
        suite(root, name, "echo built\n",
              f'echo "ncu=$WARPLENS_NCU book=${{WARPLENS_BOOK-unset}}" >> "{tmp_path}/{name}-runs"\n')
    return {**machine_env, "WARPLENS_RUN_MINUTES": "1"}, ["--skip", "selftest"]


def manifest_of(files) -> dict:
    return dict(line.split("=", 1) for line in files["manifest.txt"].decode().splitlines()
                if "=" in line and not line.startswith("#"))


def setups_record(home: Path, ncu: str) -> Path:
    versions = home / "resolved" / "versions.txt"
    versions.parent.mkdir(parents=True)
    versions.write_text(f"nsight_compute={ncu}\npredictions_comparable=comparable\n"
                        "predictions_comparable_why=nvcc 13.5.37 as for the predictions\n")
    return versions


def test_a_full_run_stops_before_the_gpu_stages_without_counters(checkout, tmp_path, machine):
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="permission", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    versions = setups_record(tmp_path / "home", "2026.4.0.10")

    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 3, done.stdout + done.stderr
    assert re.search(r"FAIL\s+ncu", done.stdout) and "ERR_NVGPUCTRPERM" in done.stdout
    assert "stopping: the full preflight failed" in done.stdout
    assert "do without counters: env/remote/run.sh --no-ncu" in done.stdout
    assert not (tmp_path / "blackwell-runs").exists() and not (tmp_path / "blackwell_timing-runs").exists()
    assert (tmp_path / "setup-args").read_text() == "\n"            # setup was not told to leave ncu out
    files = members(archive_of(tmp_path / "home"))
    manifest = manifest_of(files)
    assert manifest["ncu"] == "unavailable" and manifest["ncu_required"] == "1"
    assert manifest["stage_preflight"] == "failed" and manifest["stage_run_timing"] == "not-reached"
    assert b"this run REQUIRED them" in files["NCU-COUNTERS-NOT-READ.txt"]
    last = files["resolved/versions.txt"].decode().splitlines()[-1]
    assert last.startswith("ncu_counters=not-read (preflight's ncu check: FAIL")
    assert "ncu_counters" not in versions.read_text()               # setup.sh's own file is not written to

    # --keep-going is what it is for any failed preflight: the suites run, and the run is not a success.
    done = clean(root / "env/remote/run.sh", *args, "--keep-going", env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert (tmp_path / "blackwell_timing-runs").read_text().startswith("ncu=unavailable ")
    files = members(archive_of(tmp_path / "home"))
    assert manifest_of(files)["ncu"] == "unavailable" and manifest_of(files)["stage_run_timing"] == "ok"
    assert "NCU-COUNTERS-NOT-READ.txt" in files
    assert "performance counters: not-read" in done.stdout


def test_a_run_told_to_do_without_counters_goes_on_and_says_so(checkout, tmp_path, machine):
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="permission", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "MISSING")
    done = clean(root / "env/remote/run.sh", *args, "--no-ncu", env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(r"WARN\s+ncu", done.stdout) and "stopping" not in done.stdout
    assert (tmp_path / "setup-args").read_text() == "--no-ncu\n"
    assert (tmp_path / "blackwell_timing-runs").read_text().startswith("ncu=unavailable ")
    assert done.stdout.rstrip().splitlines()[-1].startswith("run: performance counters: not-read")
    files = members(archive_of(tmp_path / "home"))
    manifest = manifest_of(files)
    assert manifest["ncu"] == "unavailable" and manifest["ncu_required"] == "0"
    assert manifest["ncu_counters"].startswith("not-read") and "--no-ncu" in manifest["ncu_counters"]
    assert b"No GPU performance counter was read in this run" in files["NCU-COUNTERS-NOT-READ.txt"]
    assert b"--no-ncu" in files["NCU-COUNTERS-NOT-READ.txt"]
    assert "ncu_counters=not-read" in files["resolved/versions.txt"].decode()
    assert re.search(rb"^WARN\s+ncu", files["preflight.txt"], re.M)
    # Packing again later, without the option, does not turn that into "required".
    again = clean(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert manifest_of(members(archive_of(tmp_path / "home")))["ncu_required"] == "0"

    # Asking for counters after all is another run: preflight is done again, and fails.
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 3 and re.search(r"FAIL\s+ncu", done.stdout)


def test_counters_that_were_read_leave_no_flag(checkout, tmp_path, machine):
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok", WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "blackwell_timing-runs").read_text().startswith("ncu=ok ")
    files = members(archive_of(tmp_path / "home"))
    assert manifest_of(files)["ncu"] == "ok" and manifest_of(files)["ncu_counters"].startswith("read")
    assert files["resolved/versions.txt"].decode().splitlines()[-1].startswith("ncu_counters=read")
    assert "NCU-COUNTERS-NOT-READ.txt" not in files and "performance counters:" not in done.stdout


def test_a_run_without_the_timing_suite_needs_no_counters(checkout, tmp_path, machine):
    root, _, _ = checkout
    env, args = rental(root, tmp_path, machine(False, WARPLENS_BLACKWELL_RUN_ARGS="--no-e2e"))
    setups_record(tmp_path / "home", "2026.4.0.10")
    done = clean(root / "env/remote/run.sh", *args, "--skip", "build-timing", "--skip", "run-timing", env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(r"WARN\s+ncu", done.stdout)
    assert (tmp_path / "setup-args").read_text() == "\n"            # only --no-ncu leaves ncu out of the env
    files = members(archive_of(tmp_path / "home"))
    assert "the timing suite was left out" in manifest_of(files)["ncu_counters"]
    # With the timing suite back in, the earlier preflight does not stand for this run.
    done = clean(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 3 and re.search(r"FAIL\s+ncu", done.stdout)
    plan = clean(root / "env/remote/run.sh", "--dry-run", "--no-ncu", env=env).stdout
    assert re.search(r"preflight\s+\S+\s+\S+\s+env/remote/preflight\.sh --no-ncu", plan)
    plan = clean(root / "env/remote/run.sh", "--dry-run", env=env).stdout
    assert "preflight.sh --no-ncu" not in plan


def test_run_uses_the_book_setup_fetched_and_never_packs_it(checkout, tmp_path, machine):
    root, _, _ = checkout
    pin = read_pins()["BOOK_COMMIT"]
    home = tmp_path / "home"
    env, args = rental(root, tmp_path, machine(True, FAKE_NCU="ok"))
    setups_record(home, "2026.4.0.10")
    # setup.sh's part, stood in for: the clone appears during the setup stage, as on a first run.
    fetched = fake_book(tmp_path / "fetched", pin)
    with (root / "env/remote/setup.sh").open("a") as setup:
        setup.write(f'cp -r "{fetched}" "{home}/book"\n')
    first = clean(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    assert re.search(r"PASS\s+native-e2e", first.stdout) and re.search(r"PASS\s+book\s", first.stdout)
    assert (tmp_path / "blackwell-runs").read_text() == f"ncu=ok book={home}/book\n"
    files = members(archive_of(home))
    assert manifest_of(files)["book"] == f"{home}/book" and manifest_of(files)["book_commit"] == pin
    # The book is used where it was cloned. No file of it, by name or by content, is in the archive.
    assert not [name for name in files if "chapter_gemm_basics" in name or name.startswith("book")]
    assert not [name for name, data in files.items() if BOOK_MARKER.encode() in data]

    # The clone that setup made does not make the second run another run: everything is kept.
    second = clean(root / "env/remote/run.sh", *args, env=env)
    assert second.returncode == 0, second.stdout + second.stderr
    for stage in ("setup", "preflight", "build-blackwell", "run-blackwell", "run-timing"):
        assert f"{stage}: finished earlier, skipping" in second.stdout
    assert (tmp_path / "blackwell-runs").read_text().count("\n") == 1

    # A book the caller names wins, in the environment or among the suite's arguments.
    mine = fake_book(tmp_path / "mine", None)
    named = clean(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BOOK": str(mine)})
    assert named.returncode == 0, named.stdout + named.stderr
    assert (tmp_path / "blackwell-runs").read_text().splitlines()[-1] == f"ncu=ok book={mine}"
    assert re.search(r"WARN\s+book\s.*is at commit unknown, not the pinned", named.stdout)
    assert manifest_of(members(archive_of(home)))["book"] == str(mine)
    argued = clean(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BLACKWELL_RUN_ARGS": f"--book {mine}"})
    assert argued.returncode == 0, argued.stdout + argued.stderr
    assert manifest_of(members(archive_of(home)))["book"] == str(mine)
    # Named as nothing: no book, and the run stops where it would have without the default.
    none = clean(root / "env/remote/run.sh", *args, env={**env, "WARPLENS_BOOK": ""})
    assert none.returncode == 3 and re.search(r"FAIL\s+native-e2e", none.stdout)
    assert "env/remote/setup.sh installs the one and fetches the other" in none.stdout


def test_setups_clone_at_another_commit_fails_preflight(tmp_path, machine):
    home = tmp_path / "home"
    fake_book(home / "book", "2" * 40)
    done, checks, failed = preflight(tmp_path, machine(True, FAKE_NCU="ok"), "--e2e")
    assert checks["native-e2e"] == "PASS" and checks["book"] == "FAIL" and failed == {"book"}, done.stdout
    assert read_pins()["BOOK_COMMIT"] in done.stdout and "setup.sh fetches the pinned commit again" in done.stdout
    # --book overrides the default too.
    other = fake_book(tmp_path / "other", read_pins()["BOOK_COMMIT"])
    done, checks, failed = preflight(tmp_path, machine(True, FAKE_NCU="ok"), "--e2e", "--book", str(other))
    assert checks["book"] == "PASS" and not failed, done.stdout
