"""Stand-ins for the rented machine that env/remote's scripts run on: a throwaway checkout with the
scripts and stand-in suites, tools that answer as nvidia-smi, the env runner and ncu do, a stand-in
for the book's repository, and readers of the archive a run leaves."""

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.support.paths import REPO

REMOTE = REPO / "env" / "remote"


# What env/remote/run.sh exports to its stages. The suite is one of them (`selftest`), so on a
# rented machine these are in the environment the tests start with, and a script under test
# would take the machine's own ncu, results directory or preflight verdicts for the test's.
# The first B200 rental's self-test failed seven tests of the stand-in machine that way
# (NCU_BIN named the env's real ncu). A test that wants one of them passes it in `env`.
EXPORTED_BY_RUN = ("NCU_BIN", "WARPLENS_NCU", "WARPLENS_PTX_JIT", "WARPLENS_PREDICTIONS", "WARPLENS_TORCH_PYTHON",
                   "WARPLENS_GDB", "WARPLENS_RESULTS_DIR")


def inherited(*also_without: str) -> dict:
    """The environment of the tests without what a run.sh above them exported."""
    return {k: v for k, v in os.environ.items() if k not in EXPORTED_BY_RUN and k not in also_without}


def bash(script: str, *args: str, env: dict | None = None, cwd: Path | None = None):
    """Run one of the scripts; returns the completed process with text output."""
    full_env = {**inherited(), "PREFLIGHT_OFFLINE": "1", **(env or {})}
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env=full_env, cwd=cwd, timeout=120)


def read_pins() -> dict[str, str]:
    pins = {}
    for line in (REMOTE / "pins.env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            pins[key] = value
    return pins


def fake_smi(tmp_path: Path, name: str, cap: str, cuda: str = "13.5", used: int = 3, apps: str = "") -> Path:
    """A stand-in for nvidia-smi that answers the three ways preflight.sh calls it."""
    script = tmp_path / "fake-nvidia-smi"
    script.write_text(f"""#!/usr/bin/env bash
case "$*" in
  *--query-gpu=*compute_cap*) echo "0, {name}, {cap}, 183359, {used}, 0" ;;
  *--query-gpu=*) echo "0, {name}, 183359, {used}, 0" ;;
  *--query-compute-apps=*) printf '%b' "{apps}" ;;
  *) echo "| NVIDIA-SMI 590.44.01             Driver Version: 590.44.01     CUDA Version: {cuda}     |" ;;
esac
""")
    script.chmod(0o755)
    return script


def suite(root: Path, name: str, build: str, run: str):
    directory = root / "validation" / name
    directory.mkdir(parents=True)
    (directory / "build.sh").write_text("#!/usr/bin/env bash\nset -eu\n" + build)
    (directory / "run_all.sh").write_text("#!/usr/bin/env bash\nset -eu\n" + run)


def stand_in_cutlass(root: Path, commit: str | None = None, built: bool = False) -> Path:
    """What the scripts take for the CUTLASS example of a checkout: a clone at `commit` (the
    pinned one unless given; "" for no clone) and a build.sh that makes the four programs
    and says what it was given. `built`: the programs are there already."""
    example = root / "examples" / "03_cutlass_mma"
    clone = example / ".cutlass"
    if commit != "":
        for path in ("examples/cute/tutorial/blackwell/01_mma_sm100.cu", "include/cutlass/cutlass.h",
                     "examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py"):
            (clone / path).parent.mkdir(parents=True, exist_ok=True)
            (clone / path).write_text("stand-in\n")
        (clone / ".git").mkdir(exist_ok=True)
        (clone / ".git" / "HEAD").write_text((commit or read_pins()["CUTLASS_COMMIT"]) + "\n")
    example.mkdir(parents=True, exist_ok=True)
    (example / "build.sh").write_text(
        '#!/usr/bin/env bash\nset -eu\ncd "$(dirname "$0")"\n'
        'echo "cutlass build: $* nvcc=$NVCC no_fetch=${WARPLENS_CUTLASS_NO_FETCH:-unset}"\n'
        'echo "$*" >> build-calls\n'
        '[[ ! -f fail-the-build ]] || { echo "nvcc: the tutorial does not compile" >&2; exit 1; }\n'
        'mkdir -p bin\nfor n in 1 2 3 4; do echo program > "bin/tutorial_0$n"; done\n')
    if built:
        (example / "bin").mkdir(exist_ok=True)
        for n in range(1, 5):
            (example / "bin" / f"tutorial_0{n}").write_text("program\n")
    return example


@pytest.fixture
def checkout(tmp_path):
    """A copy of env/remote in a directory of its own, with stand-ins for the two probe suites
    and for the CUTLASS example (a clone at the pinned commit, as setup.sh leaves one).
    `env` in place of the env runner: the stages run with whatever is on PATH."""
    root = tmp_path / "repo"
    shutil.copytree(REMOTE, root / "env" / "remote")
    # Collection now freezes the actual tracked source used by the rental.
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    stand_in_cutlass(root)      # after the commit: the clone and its programs are never tracked
    env = {"WARPLENS_REMOTE_HOME": str(tmp_path / "home"), "WARPLENS_ENV_RUN": "env",
           "WARPLENS_RUN_MINUTES": "1"}
    # Skip what needs a GPU or installs things; the stage runner is what is under test.
    args = ["--skip", "preflight-quick", "--skip", "setup", "--skip", "preflight", "--skip", "selftest"]
    return root, env, args


def archive_of(home: Path) -> Path:
    (archive,) = (home / "results").glob("*.tar.gz")
    recorded = Path(str(archive) + ".sha256").read_text().split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == recorded
    return archive


def members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive) as tar:
        return {m.name.split("/", 1)[1]: tar.extractfile(m).read()
                for m in tar.getmembers() if m.isfile()}


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


BOOK_MARKER = "TEXT-OF-THE-BOOK-THAT-MUST-NOT-BE-COPIED"
BOOK_VARIABLES = ("WARPLENS_BOOK", "WARPLENS_BOOK_URL", "WARPLENS_BLACKWELL_RUN_ARGS")


def clean(script, *args, env=None, cwd=None):
    """bash(), but without the book the machine running the tests may have exported (CI does)
    and without the ncu it may have installed outside PATH."""
    full = inherited(*BOOK_VARIABLES)
    full.update({"PREFLIGHT_OFFLINE": "1", "PREFLIGHT_NCU_CANDIDATES": "/no/such/ncu", **(env or {})})
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=full, cwd=cwd,
                          timeout=120)


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(directory), "-c", "user.name=fixture", "-c",
                           "user.email=fixture@example.invalid", *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def fake_book(directory: Path, commit: str | None) -> Path:
    """A directory the scripts take for a clone of the book; `commit` is what its .git says."""
    (directory / "chapter_gemm_basics").mkdir(parents=True)
    (directory / "chapter_gemm_basics" / "index.md").write_text(BOOK_MARKER + "\n")
    if commit:
        (directory / ".git").mkdir()
        (directory / ".git" / "HEAD").write_text(commit + "\n")
    return directory


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


def rental(root, tmp_path, machine_env):
    """run.sh on the stand-in machine: the real preflight.sh, a setup.sh that records its
    arguments, suites that record what they were given."""
    (root / "env/remote/setup.sh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/setup-args"\n')
    for name in ("blackwell", "blackwell_timing"):
        suite(root, name, "echo built\n",
              f'echo "ncu=$WARPLENS_NCU book=${{WARPLENS_BOOK-unset}}" >> "{tmp_path}/{name}-runs"\n'
              # The timing suite says itself what came of its ncu profiles (its ncu/STATUS.txt);
              # this stand-in profiles when preflight read a counter, as the suite does.
              + ('mkdir -p "$WARPLENS_RESULTS_DIR/ncu"\n'
                 'if [[ "$WARPLENS_NCU" == ok ]]; then echo "ncu ok: 7 of 7 profiles read counters"\n'
                 'else echo "ncu not run: WARPLENS_NCU=$WARPLENS_NCU"; fi > "$WARPLENS_RESULTS_DIR/ncu/STATUS.txt"\n'
                 if name == "blackwell_timing" else ""))
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
