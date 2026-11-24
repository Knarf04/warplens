"""The scripts for the rented B200 machine (env/remote/), as far as they can be checked
without one: syntax, the pins, the decisions the scripts make from text, the plan `run.sh
--dry-run` prints, the stage runner with stand-in suites, and the two preflight kernels in the
simulator.

Sample inputs marked "recorded" were captured on the development machine (RTX 3080 Ti Laptop,
driver 580.173.02). The Blackwell ones are built from NVIDIA's published names and compute
capabilities (https://developer.nvidia.com/cuda/gpus); nobody has captured them from hardware.
"""

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "env" / "remote"
SCRIPTS = ["lib.sh", "setup.sh", "preflight.sh", "run.sh"]

# Recorded: the line of `nvidia-smi` that names the driver, and what ncu printed as a normal user.
SMI_HEADER_3080TI = ("| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     "
                     "CUDA Version: 13.0     |")
NCU_PERMISSION = ("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU "
                  "Performance Counters on the target device 0. For instructions on enabling permissions "
                  "and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM")


def bash(script: str, *args: str, env: dict | None = None, cwd: Path | None = None):
    """Run one of the scripts; returns the completed process with text output."""
    full_env = {**os.environ, "PREFLIGHT_OFFLINE": "1", **(env or {})}
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env=full_env, cwd=cwd, timeout=120)


def lib(function: str, *args: str) -> str:
    """Call one function of lib.sh and return what it printed."""
    done = subprocess.run(["bash", "-c", 'source "$1"; shift; "$@"', "_", str(REMOTE / "lib.sh"),
                           function, *args], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def read_pins() -> dict[str, str]:
    pins = {}
    for line in (REMOTE / "pins.env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            pins[key] = value
    return pins


# ---------------------------------------------------------------------------------------------
# Syntax
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_shell_scripts_parse(name):
    done = subprocess.run(["bash", "-n", str(REMOTE / name)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_shellcheck_is_clean_where_it_is_installed():
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:      # not on the development machine or in CI's image by default
        return
    done = subprocess.run([shellcheck, "-x", *(f"env/remote/{name}" for name in SCRIPTS)],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize("name", ["gpu_probe.py", "check_env.py"])
def test_python_helpers_compile_and_reject_bad_usage(name):
    source = (REMOTE / name).read_text()
    compile(source, name, "exec")
    done = subprocess.run(["python3", str(REMOTE / name)], capture_output=True, text=True)
    assert done.returncode == 2 and name in done.stderr       # prints its usage


def test_every_script_has_help():
    for name in ("setup.sh", "preflight.sh", "run.sh"):
        done = bash(REMOTE / name, "--help")
        assert done.returncode == 0 and f"env/remote/{name}" in done.stdout
        assert "set -" not in done.stdout      # the help is the header comment and nothing after it
        assert bash(REMOTE / name, "--no-such-option").returncode == 2


# ---------------------------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------------------------

def test_every_pin_is_an_exact_version():
    pins = read_pins()
    needed = {"MICROMAMBA_VERSION", "MICROMAMBA_SHA256_LINUX_64", "PYTHON", "NUMPY", "PYTEST",
              "CUDA_VERSION", "CUDA_NVCC", "CUDA_CUDART_DEV", "CUDA_NVRTC", "NSIGHT_COMPUTE",
              "TRITON", "NVIDIA_CUTLASS_DSL", "APACHE_TVM_TIRX", "CUDA_BINDINGS", "CUDA_VERSION_MIN"}
    assert needed <= set(pins)
    for key, value in pins.items():
        if "SHA256" in key:
            assert re.fullmatch(r"[0-9a-f]{64}", value), key
        else:
            # digits and dots, optionally a build number: no ranges, wildcards or "latest"
            assert re.fullmatch(r"\d+(\.\d+)+(-\d+)?", value), f"{key}={value} is not an exact version"
    # The toolkit packages belong to the pinned release.
    for key in ("CUDA_NVCC", "CUDA_CUDART_DEV", "CUDA_NVRTC"):
        assert pins[key].startswith(pins["CUDA_VERSION"] + ".")
    assert pins["CUDA_BINDINGS"].startswith(pins["CUDA_VERSION"] + ".")


def test_the_shell_reads_the_same_pins():
    pins = read_pins()
    printed = lib("eval", 'load_pins && for v in ${!PIN_*}; do echo "${v#PIN_}=${!v}"; done')
    assert dict(line.split("=", 1) for line in printed.splitlines()) == pins


def test_setup_uses_no_version_that_is_not_a_pin():
    """setup.sh may name packages, but a version number next to one must come from pins.env."""
    text = (REMOTE / "setup.sh").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not re.findall(r"[a-z-]+==\d", code), "a literal version in setup.sh"
    for key in ("PYTHON", "NUMPY", "PYTEST", "CUDA_NVCC", "CUDA_CUDART_DEV", "CUDA_NVRTC",
                "NSIGHT_COMPUTE", "TRITON", "NVIDIA_CUTLASS_DSL", "APACHE_TVM_TIRX", "CUDA_BINDINGS",
                "MICROMAMBA_VERSION", "MICROMAMBA_SHA256_LINUX_64"):
        assert f"$PIN_{key}" in code, f"setup.sh does not use the pin {key}"


# ---------------------------------------------------------------------------------------------
# Which GPU is this (plan step E20)
# ---------------------------------------------------------------------------------------------

GPUS = [
    # name as nvidia-smi prints it, compute capability, verdict
    ("NVIDIA B200", "10.0", "b200"),
    ("NVIDIA GB200", "10.0", "b200"),
    ("NVIDIA B300", "10.3", "b300"),
    ("NVIDIA GeForce RTX 5090", "12.0", "consumer-blackwell"),
    ("NVIDIA GeForce RTX 5080", "12.0", "consumer-blackwell"),
    # A datacenter card with "Blackwell" in its name that is not sm_100a.
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "12.0", "consumer-blackwell"),
    ("NVIDIA GB10", "12.1", "consumer-blackwell"),
    ("NVIDIA H100 80GB HBM3", "9.0", "hopper"),
    ("NVIDIA GeForce RTX 3080 Ti Laptop GPU", "8.6", "pre-hopper"),     # recorded
    ("NVIDIA A100-SXM4-80GB", "8.0", "pre-hopper"),
    ("NVIDIA B200", "[N/A]", "unknown"),        # the name alone never decides
    ("NVIDIA B200", "unknown", "unknown"),
    ("Some Future Card", "11.0", "unknown"),
]


@pytest.mark.parametrize("name,cap,verdict", GPUS)
def test_gpu_classification(name, cap, verdict):
    done = bash(REMOTE / "preflight.sh", "--classify", name, cap)
    assert done.returncode == 0, done.stderr
    got, _, why = done.stdout.strip().partition("|")
    assert got == verdict
    assert why      # every verdict explains itself


def test_arch_names():
    assert lib("arch_for_cap", "10.0") == "sm_100a"
    assert lib("arch_for_cap", "10.3") == "sm_103a"
    assert lib("arch_for_cap", "8.6") == "sm_86"


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


def quick_preflight(tmp_path, **gpu):
    env = {"NVIDIA_SMI": str(fake_smi(tmp_path, **gpu)), "WARPLENS_REMOTE_HOME": str(tmp_path / "home")}
    done = bash(REMOTE / "preflight.sh", "--quick", "--status", str(tmp_path / "status"), env=env)
    status = dict(line.split() for line in (tmp_path / "status").read_text().splitlines())
    return done, status


def test_quick_preflight_accepts_a_b200(tmp_path):
    done, status = quick_preflight(tmp_path, name="NVIDIA B200", cap="10.0")
    assert status["gpu-identity"] == "PASS" and status["driver"] == "PASS"
    assert status["gpu-memory"] == "PASS" and status["gpu-idle"] == "PASS"
    # Only the disk check can fail here: it depends on the machine running the tests.
    failed = {check for check, result in status.items() if result == "FAIL"}
    assert failed <= {"disk"}
    assert done.returncode == (1 if failed else 0)
    assert not (tmp_path / "home").exists()     # a check, not an installer


@pytest.mark.parametrize("name,cap", [("NVIDIA GeForce RTX 5090", "12.0"),
                                      ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "12.0"),
                                      ("NVIDIA GeForce RTX 3080 Ti Laptop GPU", "8.6"),
                                      ("NVIDIA H100 80GB HBM3", "9.0"),
                                      ("NVIDIA B300", "10.3")])
def test_quick_preflight_rejects_everything_else(tmp_path, name, cap):
    done, status = quick_preflight(tmp_path, name=name, cap=cap)
    assert status["gpu-identity"] == "FAIL"
    assert done.returncode == 1
    assert "not a B200" in done.stdout and "FAILED" in done.stdout


def test_quick_preflight_reports_the_driver_and_a_busy_gpu(tmp_path):
    _, status = quick_preflight(tmp_path, name="NVIDIA B200", cap="10.0", cuda="13.0")
    assert status["driver"] == "WARN"           # older than the pinned toolkit: setup will match it
    _, status = quick_preflight(tmp_path, name="NVIDIA B200", cap="10.0", cuda="12.4")
    assert status["driver"] == "FAIL"           # too old for sm_100a
    done, status = quick_preflight(tmp_path, name="NVIDIA B200", cap="10.0", used=40000,
                                   apps="4242, python, 39000 MiB\\n")
    assert status["gpu-memory"] == "WARN" and status["gpu-idle"] == "WARN"
    assert "4242, python" in done.stdout


def test_preflight_without_a_driver_fails(tmp_path):
    env = {"NVIDIA_SMI": str(tmp_path / "no-such-program"), "WARPLENS_REMOTE_HOME": str(tmp_path / "home")}
    done = bash(REMOTE / "preflight.sh", "--quick", env=env)
    assert done.returncode == 1 and re.search(r"^FAIL\s+nvidia-smi", done.stdout, re.M)


# ---------------------------------------------------------------------------------------------
# Driver and toolkit
# ---------------------------------------------------------------------------------------------

def test_driver_versions_from_nvidia_smi_text():
    assert lib("parse_smi_cuda_version", SMI_HEADER_3080TI) == "13.0"
    assert lib("parse_smi_driver_version", SMI_HEADER_3080TI) == "580.173.02"
    assert lib("parse_smi_cuda_version", "NVIDIA-SMI has failed") == ""


@pytest.mark.parametrize("driver,mode,toolkit,status", [
    ("13.5", "auto", "13.5", "ok"),
    ("14.1", "auto", "13.5", "ok"),            # a newer driver runs older toolkits
    ("13.0", "auto", "13.0", "fallback"),      # the development machine: match the driver
    ("12.9", "auto", "12.9", "fallback"),
    ("12.8", "auto", "12.8", "fallback"),
    ("12.4", "auto", "13.5", "unsupported"),   # no toolkit that old knows sm_100a
    ("", "auto", "13.5", "ok"),                # no driver at all (CI): as pinned
    ("13.0", "pinned", "13.5", "mismatch"),
    ("13.5", "pinned", "13.5", "ok"),
    ("13.0", "13.2", "13.2", "fallback"),      # an explicit release wins
    ("13.5", "12.4", "12.4", "unsupported"),
    ("13.5", "newest", "13.5", "unsupported"),
])
def test_toolkit_choice(driver, mode, toolkit, status):
    done = bash(REMOTE / "preflight.sh", "--choose-toolkit", "13.5", driver, "12.8", mode)
    got_toolkit, got_status, why = done.stdout.strip().split("|", 2)
    assert (got_toolkit, got_status) == (toolkit, status)
    assert why


def test_version_comparison_is_numeric():
    for newer, older in [("13.10", "13.9"), ("13.0", "12.9"), ("13.5", "13.5")]:
        assert lib("eval", f"version_ge {newer} {older} && echo yes") == "yes"
    assert lib("eval", "version_ge 13.9 13.10 || echo no") == "no"


# ---------------------------------------------------------------------------------------------
# ncu output (plan step E21)
# ---------------------------------------------------------------------------------------------

def test_ncu_output_is_read_strictly():
    metric = "smsp__inst_executed.sum"
    row = ('"0","4242","python3.12","127.0.0.1","pf_vecadd","1","7","(256, 1, 1)","(4, 1, 1)","0","10.0",'
           f'"Command line profiler metrics","{metric}","inst","1,234"')
    assert lib("parse_ncu_output", metric, f"==PROF== Connected\n{row}\n") == "ok|1,234"
    kind, _, line = lib("parse_ncu_output", metric, f"==PROF== Connected\n{NCU_PERMISSION}\nok=1").partition("|")
    assert kind == "permission" and line == NCU_PERMISSION
    # The metric's name in an error message is not a counter value. This exact line once passed.
    error = (f"ERROR conda.cli.main_run:execute(148): `conda run timeout 300 ncu --metrics {metric} "
             "python gpu_probe.py vecadd /tmp/tmp.1/vecadd.cubin` failed. (See above for error)")
    assert lib("parse_ncu_output", metric, error) == "none|"
    assert lib("parse_ncu_output", metric, row.replace('"1,234"', '"n/a"')) == "none|"


def test_the_commit_is_found_without_a_git_program(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(REMOTE, root / "env" / "remote")
    commit = "0123456789abcdef0123456789abcdef01234567"
    (root / ".git/refs/heads").mkdir(parents=True)
    (root / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git/refs/heads/main").write_text(commit + "\n")
    tools = tmp_path / "tools"          # a PATH with no git on it
    tools.mkdir()
    for tool in ("bash", "cat", "awk", "dirname"):
        (tools / tool).symlink_to(shutil.which(tool))

    def found():
        done = subprocess.run([shutil.which("bash"), "-c", 'source "$1"; repo_commit', "_",
                               str(root / "env/remote/lib.sh")], capture_output=True, text=True,
                              env={"PATH": str(tools), "HOME": str(tmp_path)})
        return done.stdout.strip()

    assert found() == commit
    (root / ".git/refs/heads/main").unlink()
    (root / ".git/packed-refs").write_text(f"# pack-refs with: peeled\n{commit} refs/heads/main\n")
    assert found() == commit
    shutil.rmtree(root / ".git")
    assert found() == "unknown"


def test_capability_bits():
    assert lib("eval", "has_capability 000001ffffffffff 21 && echo yes") == "yes"     # root
    assert lib("eval", "has_capability 0000000000000000 21 || echo no") == "no"
    assert lib("eval", "has_capability 0000004000000000 38 && echo yes") == "yes"     # CAP_PERFMON only
    assert lib("eval", "has_capability 0000004000000000 21 || echo no") == "no"


# ---------------------------------------------------------------------------------------------
# run.sh
# ---------------------------------------------------------------------------------------------

STAGES = ["preflight-quick", "setup", "preflight", "selftest", "build-blackwell", "build-timing",
          "run-blackwell", "run-timing", "collect"]


def test_dry_run_prints_the_plan_without_a_gpu(tmp_path):
    home = tmp_path / "home"
    env = {"WARPLENS_REMOTE_HOME": str(home), "NVIDIA_SMI": str(tmp_path / "no-such-program"),
           "CUDA_VISIBLE_DEVICES": ""}
    done = bash(REMOTE / "run.sh", "--dry-run", "--price", "6.5", env=env)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    planned = [line.split()[0] for line in lines if line.split() and line.split()[0] in STAGES]
    assert planned == STAGES                    # every stage, in order
    assert sum("basis:" in line for line in lines) == len(STAGES)
    total = re.search(r"estimated wall-clock time: (\d+) to (\d+) minutes", done.stdout)
    assert total and 0 < int(total[1]) < int(total[2])
    cost = re.search(r"at \$6\.50 per hour: \$(\d+\.\d\d) to \$(\d+\.\d\d)", done.stdout)
    assert cost and float(cost[2]) == pytest.approx(int(total[2]) / 60 * 6.5, abs=0.01)
    assert not home.exists()                    # a dry run touches nothing
    skipped = bash(REMOTE / "run.sh", "--dry-run", "--skip", "selftest", env=env)
    assert re.search(r"selftest\s+0-0\s+skip", skipped.stdout)


def test_run_rejects_bad_arguments():
    assert bash(REMOTE / "run.sh", "--redo", "no-such-stage").returncode == 2
    assert bash(REMOTE / "run.sh", "--price", "cheap", "--dry-run").returncode == 2


def suite(root: Path, name: str, build: str, run: str):
    directory = root / "validation" / name
    directory.mkdir(parents=True)
    (directory / "build.sh").write_text("#!/usr/bin/env bash\nset -eu\n" + build)
    (directory / "run_all.sh").write_text("#!/usr/bin/env bash\nset -eu\n" + run)


@pytest.fixture
def checkout(tmp_path):
    """A copy of env/remote in a directory of its own, with stand-ins for the two probe suites.
    `env` in place of the env runner: the stages run with whatever is on PATH."""
    root = tmp_path / "repo"
    shutil.copytree(REMOTE, root / "env" / "remote")
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


def test_run_builds_runs_collects_and_resumes(checkout, tmp_path):
    root, env, args = checkout
    flag = tmp_path / "second-attempt"
    suite(root, "blackwell", 'echo "built with $NVCC"\n',
          'echo "probe output, jit=$WARPLENS_PTX_JIT ncu=$WARPLENS_NCU" > "$WARPLENS_RESULTS_DIR/probes.txt"\n'
          f'echo ran >> "{tmp_path}/blackwell-runs"\n')
    # The timing suite fails the first time it runs and leaves output in its own directory.
    suite(root, "blackwell_timing", "echo built\n",
          'cd "$(dirname "$0")"\nmkdir -p results && echo 42 > results/timing.csv\n'
          f'[[ -f "{flag}" ]] || {{ echo "probe crashed" >&2; exit 7; }}\n')

    first = bash(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 1, first.stdout + first.stderr
    assert "run-timing: failed" in first.stdout and "probe crashed" in first.stdout
    assert "run-blackwell: ok" in first.stdout          # one suite failing does not stop the other
    archive = archive_of(tmp_path / "home")
    files = members(archive)
    assert files["out/blackwell/probes.txt"] == b"probe output, jit=unknown ncu=unavailable\n"
    assert files["suite-dirs/validation/blackwell_timing/results/timing.csv"] == b"42\n"
    assert b"built with nvcc" in files["logs/build-blackwell.log"]
    assert b"probe crashed" in files["logs/run-timing.log"]
    manifest = dict(line.split("=", 1) for line in files["manifest.txt"].decode().splitlines()
                    if "=" in line and not line.startswith("#"))
    assert manifest["stage_run_blackwell"] == "ok" and manifest["stage_run_timing"] == "failed"
    assert manifest["stage_setup"] == "skipped"
    assert {"warplens_commit", "first_start", "this_start", "end", "host"} <= set(manifest)
    assert files["pins.env"] == (REMOTE / "pins.env").read_bytes()
    # Every file in the archive is listed with its checksum.
    sums = dict(reversed(line.split(None, 1)) for line in files["SHA256SUMS"].decode().splitlines())
    for name, data in files.items():
        if name != "SHA256SUMS":
            assert sums["./" + name] == hashlib.sha256(data).hexdigest()

    # Again, with the cause fixed: finished stages are skipped, the failed one runs.
    flag.touch()
    second = bash(root / "env/remote/run.sh", *args, env=env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "run-blackwell: finished earlier, skipping" in second.stdout
    assert "build-timing: finished earlier, skipping" in second.stdout
    assert "run-timing: ok" in second.stdout
    assert (tmp_path / "blackwell-runs").read_text() == "ran\n"      # ran once only
    assert archive_of(tmp_path / "home") == archive                  # same archive, rebuilt
    history = members(archive)["stages.tsv"].decode()
    assert history.count("run-timing\tfailed\t7") == 1 and history.count("run-timing\tok\t0") == 1

    # --redo runs a finished stage again.
    third = bash(root / "env/remote/run.sh", *args, "--redo", "run-blackwell", env=env)
    assert third.returncode == 0 and (tmp_path / "blackwell-runs").read_text() == "ran\nran\n"


def test_run_says_which_suite_is_missing_and_goes_on(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", 'echo fine > "$WARPLENS_RESULTS_DIR/probes.txt"\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)       # no validation/blackwell_timing
    assert done.returncode == 1
    assert "validation/blackwell_timing/build.sh not found" in done.stdout
    assert "run-timing: skipped: build-timing did not succeed" in done.stdout
    files = members(archive_of(tmp_path / "home"))
    assert files["out/blackwell/probes.txt"] == b"fine\n"
    assert b"is not in this checkout" in files["logs/build-timing.log"]
    plan = bash(root / "env/remote/run.sh", "--dry-run", env=env)
    assert re.search(r"build-timing\s+\S+\s+MISSING", plan.stdout)
    assert re.search(r"run-blackwell\s+\S+\s+done", plan.stdout)


def test_run_ends_a_stage_at_its_time_limit(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "sleep 300\n", "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    env = {**env, "WARPLENS_BUILD_MINUTES": "0.02"}               # about a second
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1
    assert "build-blackwell: timeout" in done.stdout
    assert "run-blackwell: skipped" in done.stdout and "run-timing: ok" in done.stdout


def test_run_stops_at_a_failed_quick_preflight_but_still_writes_an_archive(checkout, tmp_path):
    root, env, _ = checkout
    env = {**env, "NVIDIA_SMI": str(fake_smi(tmp_path, name="NVIDIA GeForce RTX 5090", cap="12.0"))}
    suite(root, "blackwell", "echo built\n", f'touch "{tmp_path}/ran"\n')
    done = bash(root / "env/remote/run.sh", env=env)
    assert done.returncode == 3
    assert "not a B200" in done.stdout and "stopping" in done.stdout
    assert not (tmp_path / "ran").exists()
    files = members(archive_of(tmp_path / "home"))
    assert b"consumer or workstation Blackwell" in files["preflight-quick.txt"]


# ---------------------------------------------------------------------------------------------
# The preflight kernels, in the simulator
# ---------------------------------------------------------------------------------------------

def probe_module():
    spec = importlib.util.spec_from_file_location("gpu_probe", REMOTE / "gpu_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_kernels_do_what_the_probe_expects(sim):
    """kernels/pf_blackwell.ptx is pf_blackwell.cu compiled by the pinned nvcc. If these kernels
    were wrong, preflight would fail a good B200."""
    ptx = REMOTE / "kernels" / "pf_blackwell.ptx"
    out = sim.from_numpy(np.zeros(2, dtype=np.uint32))
    stats = sim.launch(ptx, "pf_tmem", grid=1, block=32, args=[out])
    assert out.numpy()[0] == 0xB200B200 and not stats.findings

    probe = probe_module()
    data = np.arange(probe.TENSOR_W * probe.TENSOR_H, dtype=np.float32).reshape(probe.TENSOR_H, probe.TENSOR_W)
    tensor_map = sim.tensor_map_tiled(sim.from_numpy(data), (probe.BOX_H, probe.BOX_W))
    box = sim.from_numpy(np.zeros(probe.BOX_W * probe.BOX_H, dtype=np.float32))
    status = sim.from_numpy(np.zeros(1, dtype=np.uint32))
    c0, c1 = 2 * probe.BOX_W, probe.BOX_H       # the coordinates gpu_probe.py uses
    stats = sim.launch(ptx, "pf_tma", grid=1, block=64, args=[tensor_map.tobytes(), box, status, c0, c1])
    assert status.numpy()[0] == 1 and not stats.findings
    want = data[c1:c1 + probe.BOX_H, c0:c0 + probe.BOX_W]
    assert np.array_equal(box.numpy().reshape(want.shape), want)


def test_preflight_kernels_compile_where_nvcc_is_installed(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:            # the basic CI job; the `full` job and the dev env have it
        return
    known = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True).stdout
    if "compute_100" not in known:  # a toolkit older than 12.8 that happens to be first on PATH
        return
    for source in ("pf_vecadd.cu", "pf_blackwell.cu"):
        done = subprocess.run([nvcc, "-arch=sm_100a", "-cubin", str(REMOTE / "kernels" / source),
                               "-o", str(tmp_path / (source + ".cubin"))], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
