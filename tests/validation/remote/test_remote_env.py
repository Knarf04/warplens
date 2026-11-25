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
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from tests.support import REPO
from tests.support.remote import REMOTE, archive_of, bash, checkout, fake_smi, members, read_pins, suite
SCRIPTS = ["lib.sh", "setup.sh", "preflight.sh", "run.sh", "predict_toolkit.sh"]

# Recorded: the line of `nvidia-smi` that names the driver, and what ncu printed as a normal user.
SMI_HEADER_3080TI = ("| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     "
                     "CUDA Version: 13.0     |")
SMI_HEADER_B200_R610 = ("| NVIDIA-SMI 610.57.04              KMD Version: 610.57.04     "
                       "CUDA UMD Version: 13.3     |")
NCU_PERMISSION = ("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU "
                  "Performance Counters on the target device 0. For instructions on enabling permissions "
                  "and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM")


def lib(function: str, *args: str) -> str:
    """Call one function of lib.sh and return what it printed."""
    done = subprocess.run(["bash", "-c", 'source "$1"; shift; "$@"', "_", str(REMOTE / "lib.sh"),
                           function, *args], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


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
    # Warnings and errors; the notes below that level differ from one shellcheck release to the next.
    done = subprocess.run([shellcheck, "-x", "--severity=warning", *(f"env/remote/{name}" for name in SCRIPTS)],
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
        elif key in ("BOOK_COMMIT", "CUTLASS_COMMIT"):      # the pins that are commits: whole, not abbreviated
            assert re.fullmatch(r"[0-9a-f]{40}", value), key
        else:
            # digits and dots, optionally a build number: no ranges, wildcards or "latest"
            assert re.fullmatch(r"\d+(\.\d+)+(-\d+)?", value), f"{key}={value} is not an exact version"
    # The toolkit packages belong to the pinned release.
    for key in ("CUDA_NVCC", "CUDA_CUDART_DEV", "CUDA_NVRTC"):
        assert pins[key].startswith(pins["CUDA_VERSION"] + ".")
    assert pins["CUDA_BINDINGS"].startswith(pins["CUDA_VERSION"] + ".")


def test_the_book_is_pinned_to_one_commit_everywhere():
    """pins.env is where the remote scripts get the book's commit; the tests and CI name the
    same one, and no script or document carries another."""
    pin = read_pins()["BOOK_COMMIT"]
    in_tests = re.search(r'^COMMIT = "([0-9a-f]{40})"', (REPO / "tests" / "runtime" / "test_book.py").read_text(), re.M)
    assert in_tests and in_tests[1] == pin
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert re.findall(r"modern-gpu-programming-for-mlsys ([0-9a-f]{7,40})", workflow) == [pin]
    for name in ("lib.sh", "setup.sh", "preflight.sh", "run.sh"):       # the pin, never a literal
        assert not re.search(r"\b[0-9a-f]{40}\b", (REMOTE / name).read_text()), name
    assert "$PIN_BOOK_COMMIT" in (REMOTE / "setup.sh").read_text()
    assert "$PIN_BOOK_COMMIT" in (REMOTE / "preflight.sh").read_text()
    # The runbook's fetch by hand is for the same commit.
    by_hand = re.findall(r"\b[0-9a-f]{40}\b", (REMOTE / "RUNBOOK.md").read_text())
    assert by_hand and set(by_hand) == {pin}


def test_cutlass_is_pinned_to_one_commit_everywhere():
    """pins.env is where setup.sh and preflight.sh get the CUTLASS commit; build.sh, which a
    developer runs without them, names the same one in full (a server gives a single commit
    only by its whole name), and the capture tool requires it by its first characters."""
    pin = read_pins()["CUTLASS_COMMIT"]
    build = (REPO / "examples" / "03_cutlass_mma" / "build.sh").read_text()
    assert re.findall(r"^COMMIT=(\S+)$", build, re.M) == [pin]
    capture = (REPO / "validation" / "blackwell" / "e2e" / "capture_cutlass.py").read_text()
    required = re.search(r'^PIN = "([0-9a-f]{7,40})"', capture, re.M)
    assert required and pin.startswith(required[1])
    assert 'ensure_cutlass "$PIN_CUTLASS_COMMIT"' in (REMOTE / "setup.sh").read_text()
    assert "$PIN_CUTLASS_COMMIT" in (REMOTE / "preflight.sh").read_text()
    # Nothing clones the whole history any more, and nothing fetches without a time limit.
    assert "git clone" not in build and "--depth 1" in build and "timeout" in build


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

@pytest.mark.parametrize("header,cuda,driver", [
    (SMI_HEADER_3080TI, "13.0", "580.173.02"),
    (SMI_HEADER_B200_R610, "13.3", "610.57.04"),
])
def test_driver_versions_from_nvidia_smi_text(header, cuda, driver):
    assert lib("parse_smi_cuda_version", header) == cuda
    assert lib("parse_smi_driver_version", header) == driver


def test_driver_versions_from_failed_nvidia_smi():
    assert lib("parse_smi_cuda_version", "NVIDIA-SMI has failed") == ""
    assert lib("parse_smi_driver_version", "NVIDIA-SMI has failed") == ""


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

# With a fallback toolkit, or before setup.sh has recorded one: the self-test before the GPU stages.
STAGES = ["preflight-quick", "setup", "build-cutlass", "preflight", "selftest", "build-blackwell", "build-timing",
          "predict-toolkit", "run-blackwell", "run-timing", "collect"]
# With the predictions' own compiler (predictions_comparable=comparable): after them.
STAGES_PINNED = [stage for stage in STAGES if stage not in ("selftest", "collect")] + ["selftest", "collect"]


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
    assert "The self-test is before the GPU stages" in done.stdout
    # The numbers of the last whole runs of the suite, not those of half the tests ago.
    assert "13.6 min" in done.stdout and "3046 tests" in done.stdout and "1513" not in done.stdout
    # Without the end-to-end programs there is nothing of CUTLASS to build.
    probes_only = bash(REMOTE / "run.sh", "--dry-run", env={**env, "WARPLENS_BLACKWELL_RUN_ARGS": "--no-e2e"})
    assert re.search(r"build-cutlass\s+0-0\s+skip", probes_only.stdout)

    # Once setup.sh has recorded the predictions' own compiler, the self-test is planned last.
    pinned = tmp_path / "pinned-home"
    (pinned / "resolved").mkdir(parents=True)
    (pinned / "resolved" / "versions.txt").write_text("predictions_comparable=comparable\n")
    done = bash(REMOTE / "run.sh", "--dry-run", env={**env, "WARPLENS_REMOTE_HOME": str(pinned)})
    planned = [line.split()[0] for line in done.stdout.splitlines() if line.split() and line.split()[0] in STAGES]
    assert planned == STAGES_PINNED and "The self-test is after the GPU stages" in done.stdout


def test_run_rejects_bad_arguments():
    assert bash(REMOTE / "run.sh", "--redo", "no-such-stage").returncode == 2
    assert bash(REMOTE / "run.sh", "--price", "cheap", "--dry-run").returncode == 2


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


def test_a_run_without_the_rt11_capture_fails_and_ends_by_saying_so(checkout, tmp_path):
    """validation/blackwell/run_rows.py leaves RT-11-NOT-CAPTURED.txt in the suite's results
    when the capture of matrix row RT-11 was required and is missing, and the suite exits 1.
    run.sh: the stage failed, the file's text is the last thing before the exit, and the file
    is in the archive. Once the capture has been made by hand (the file is gone), packing
    again is clean."""
    root, env, args = checkout
    text = "MATRIX ROW RT-11 WAS NOT CAPTURED: e2e/cute_dsl_tma ended as: skipped (gdb not found)"
    suite(root, "blackwell", "echo built\n",
          f'echo "{text}" > "$WARPLENS_RESULTS_DIR/RT-11-NOT-CAPTURED.txt"\necho "probes ran"\nexit 1\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "run-blackwell: failed" in done.stdout and "run-timing: ok" in done.stdout
    # After the line of stage statuses, which is where a reader looks; only the toolkit's own
    # flag (this stand-in checkout has no record of one) comes later.
    lines = done.stdout.splitlines()
    at = next(i for i, line in enumerate(lines) if "stages: " in line and "run-blackwell=failed" in line)
    assert lines[at + 1:at + 4] == ["!" * 100, text, "!" * 100]
    files = members(archive_of(tmp_path / "home"))
    assert files["out/blackwell/RT-11-NOT-CAPTURED.txt"].decode() == text + "\n"

    # Still there when the archive is packed again without running anything: still said, still a failure.
    again = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 1 and text in again.stdout
    # Captured by hand since (run_rows.py removes the file when the row ends as `ok`).
    (tmp_path / "home" / "run" / "out" / "blackwell" / "RT-11-NOT-CAPTURED.txt").unlink()
    packed = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert packed.returncode == 0 and "NOT CAPTURED" not in packed.stdout
    assert "out/blackwell/RT-11-NOT-CAPTURED.txt" not in members(archive_of(tmp_path / "home"))


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


def fallback_checkout(root: Path, tmp_path: Path, predict_body: str) -> None:
    """Stand-ins for what predict_toolkit.sh calls: programs built by nvcc 13.0.88, committed
    predictions made with 13.5.37, a predict mode that writes two probe predictions (one as
    committed, one not) and a timing predictor."""
    suite(root, "blackwell", 'mkdir -p "$(dirname "$0")/bin"\necho nvcc_version=13.0.88 > "$(dirname "$0")/bin/toolkit.txt"\n',
          # The GPU stage records whether the predictions were there before it ran.
          '[[ -f "$WARPLENS_RESULTS_DIR/../predicted-toolkit/toolkit.txt" ]] && echo yes > "$WARPLENS_RESULTS_DIR/predicted-first.txt"\n'
          'echo ran > "$WARPLENS_RESULTS_DIR/probes.txt"\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    probes, timing = root / "validation/blackwell", root / "validation/blackwell_timing"
    (probes / "predicted/tma").mkdir(parents=True)
    (probes / "predicted/provenance.json").write_text('{\n "nvcc_version": "13.5.37"\n}\n')
    (probes / "predicted/tma/same.txt").write_text("value 1\n")
    (probes / "predicted/tma/other.txt").write_text("value 2\n")
    (probes / "predicted/tma/not_made.txt").write_text("value 3\n")
    (probes / "run_all.py").write_text(
        "import sys, pathlib\n"
        "assert sys.argv[1:5] == ['--mode', 'predict', '--no-build', '--no-e2e'], sys.argv\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--predicted') + 1]) / 'tma'\n"
        "out.mkdir(parents=True)\n"
        "(out / 'same.txt').write_text('value 1\\n')\n"
        "(out / 'other.txt').write_text('value 2, as nvcc 13.0 compiles it\\n')\n"
        "(out.parent / 'provenance.json').write_text('{\\n \"nvcc_version\": \"13.0.88\"\\n}\\n')\n"
        + predict_body)
    (timing / "predicted").mkdir()
    (timing / "predicted/meta.json").write_text('{"nvcc_version": "13.5.37"}\n')
    (timing / "predicted/tma.out").write_text("BENCH tma instructions=10\n")
    (timing / "predict.py").write_text(
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.mkdir(parents=True)\n"
        "(out / 'tma.out').write_text('BENCH tma instructions=11\\n')\n")
    resolved = tmp_path / "home" / "resolved"
    resolved.mkdir(parents=True)
    (resolved / "versions.txt").write_text(
        "cuda_nvcc=13.0.88\npredictions_comparable=not-comparable\n"
        "predictions_comparable_why=BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: stand-in\n")


def test_a_fallback_toolkit_gets_its_own_predictions_before_the_gpu_stages(checkout, tmp_path):
    """The toolkit policy (RUNBOOK.md, "Driver older than the toolkit"): when setup.sh fell
    back to another nvcc, the predictions are made again with it, before the GPU stages,
    into the archive, with the names that differ from the committed ones."""
    root, env, args = checkout
    fallback_checkout(root, tmp_path, "")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "predict-toolkit: ok" in done.stdout
    assert done.stdout.index("predict-toolkit: ok") < done.stdout.index("run-blackwell: started")
    files = members(archive_of(tmp_path / "home"))
    assert files["out/blackwell/predicted-first.txt"] == b"yes\n"
    assert files["out/predicted-toolkit/blackwell/tma/other.txt"] == b"value 2, as nvcc 13.0 compiles it\n"
    assert b'"13.0.88"' in files["out/predicted-toolkit/blackwell/provenance.json"]
    differences = files["out/predicted-toolkit/DIFFERENCES.txt"].decode().splitlines()
    assert "differs tma/other.txt" in differences and "missing tma/not_made.txt" in differences
    assert "differs tma/same.txt" not in differences
    assert "differs tma.out" in differences[differences.index("[timing]"):]
    facts = dict(line.split("=", 1) for line in files["out/predicted-toolkit/toolkit.txt"].decode().splitlines())
    assert facts["nvcc_version"] == "13.0.88" and facts["committed_predictions_nvcc_version"] == "13.5.37"
    assert facts["probe_predictions"] == "2" and facts["probe_predictions_that_differ_from_committed"] == "1"
    assert facts["probe_prediction_files_missing_here"] == "1"
    assert facts["timing_prediction_files_that_differ_or_are_missing"] == "1"
    note = files["PREDICTIONS-NOT-COMPARABLE.txt"].decode()
    assert "stage predict-toolkit: ok" in note and "--predicted out/predicted-toolkit/blackwell" in note
    assert "stage_predict_toolkit=ok" in files["manifest.txt"].decode()
    # The committed predictions are untouched.
    assert (root / "validation/blackwell/predicted/tma/other.txt").read_text() == "value 2\n"

    # Building again invalidates the predictions made from the old programs.
    again = bash(root / "env/remote/run.sh", *args, "--redo", "build-blackwell", env=env)
    assert again.returncode == 0 and "predict-toolkit: ok" in again.stdout
    assert "predict-toolkit: finished earlier" not in again.stdout


def test_failed_fallback_predictions_do_not_stop_the_gpu_stages(checkout, tmp_path):
    root, env, args = checkout
    fallback_checkout(root, tmp_path, "sys.exit(1)\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1
    assert "predict-toolkit: failed" in done.stdout and "A STEP FAILED" in done.stdout
    assert "run-blackwell: ok" in done.stdout and "run-timing: ok" in done.stdout
    files = members(archive_of(tmp_path / "home"))
    assert "stage_predict_toolkit=failed" in files["manifest.txt"].decode()
    assert "stage predict-toolkit: failed" in files["PREDICTIONS-NOT-COMPARABLE.txt"].decode()


def test_the_pinned_toolkit_needs_no_predictions_of_its_own(checkout, tmp_path):
    root, env, args = checkout
    fallback_checkout(root, tmp_path, "")
    (tmp_path / "home/resolved/versions.txt").write_text(
        "predictions_comparable=comparable\npredictions_comparable_why=nvcc 13.5.37 as for the predictions\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "predict-toolkit: skipped: not needed: comparable" in done.stdout
    files = members(archive_of(tmp_path / "home"))
    assert not any(name.startswith("out/predicted-toolkit/") for name in files)
    assert "PREDICTIONS-NOT-COMPARABLE.txt" not in files


def test_run_ends_a_stage_at_its_time_limit(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "sleep 300\n", "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    env = {**env, "WARPLENS_BUILD_MINUTES": "0.02"}               # about a second
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1
    assert "build-blackwell: timeout" in done.stdout
    assert "run-blackwell: skipped" in done.stdout and "run-timing: ok" in done.stdout


def test_an_interrupted_run_stops_its_stage_and_still_packs(checkout, tmp_path):
    root, env, args = checkout
    started = tmp_path / "started"
    suite(root, "blackwell", f'touch "{started}"\nsleep 300 & echo $! > "{tmp_path}/pid"\nwait\n', "echo never\n")
    suite(root, "blackwell_timing", "echo built\n", f'touch "{tmp_path}/timing-ran"\n')
    process = subprocess.Popen(["bash", str(root / "env/remote/run.sh"), *args], env={**os.environ, **env},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(200):
        if started.exists() and (tmp_path / "pid").exists():
            break
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=60)
    assert process.returncode == 130, output
    assert "build-blackwell: interrupted" in output and "archive:" in output
    assert not (tmp_path / "timing-ran").exists()           # nothing after the interrupted stage ran
    sleeper = int((tmp_path / "pid").read_text())
    for _ in range(100):                                    # the stage's own children are gone
        if not Path(f"/proc/{sleeper}").exists():
            break
        time.sleep(0.05)
    assert not Path(f"/proc/{sleeper}").exists()
    files = members(archive_of(tmp_path / "home"))
    assert "build-blackwell\tinterrupted\t130" in files["stages.tsv"].decode()

    # Packing again later changes nothing but the archive; the manifest keeps the last statuses.
    again = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 0 and "archive:" in again.stdout
    manifest = members(archive_of(tmp_path / "home"))["manifest.txt"].decode()
    assert "stage_build_blackwell=interrupted" in manifest


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


# ---------------------------------------------------------------------------------------------
# Driver, toolkit, and the toolkit the predictions were made with
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("installed,predicted,driver,status,words", [
    ("13.5.37", "13.5.37", "13.5", "comparable", "takes its PTX"),
    ("13.5.37", "13.5.37", "14.0", "comparable", ""),
    ("13.5.37", "13.5.37", "", "comparable", ""),                        # no driver (CI)
    ("13.5.37", "13.5.37", "13.0", "comparable-native", "refuses this toolkit's PTX"),   # --toolkit pinned
    ("13.0.48", "13.5.37", "13.0", "not-comparable", "BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE"),
    ("12.8.93", "13.5.37", "12.8", "not-comparable", "nvcc 12.8.93"),
    ("13.5.38", "13.5.37", "13.5", "not-comparable", "nvcc 13.5.38"),    # a patch release is another compiler
    ("", "13.5.37", "13.5", "unknown", "TOOLKIT NOT RECORDED"),
    ("MISSING", "13.5.37", "13.5", "unknown", "TOOLKIT NOT RECORDED"),
    ("13.5.37", "", "13.5", "unknown", "TOOLKIT NOT RECORDED"),
])
def test_toolkit_verdict(installed, predicted, driver, status, words):
    got, _, why = lib("toolkit_verdict", installed, predicted, driver).partition("|")
    assert got == status and words in why


def test_the_predictions_toolkit_is_read_from_the_repository():
    pins = read_pins()
    assert lib("predictions_nvcc", "blackwell") == pins["CUDA_NVCC"]
    assert lib("predictions_nvcc", "timing") == pins["CUDA_NVCC"]


def test_setup_and_preflight_compare_the_three_versions():
    setup = (REMOTE / "setup.sh").read_text()
    for key in ("driver=", "driver_cuda=", "toolkit_installed=", "predictions_nvcc=", "predictions_comparable="):
        assert f'echo "{key}' in setup, key
    assert "toolkit_verdict" in setup and "toolkit_verdict" in (REMOTE / "preflight.sh").read_text()
    assert " predictions " in (REMOTE / "preflight.sh").read_text()      # a named check, so run.sh can read it


def versions_file(home: Path, state: str, why: str) -> None:
    (home / "resolved").mkdir(parents=True)
    (home / "resolved" / "versions.txt").write_text(
        f"driver=570.86.15\ndriver_cuda=12.8\ncuda_nvcc=12.8.93\npredictions_nvcc=13.5.37\n"
        f"predictions_comparable={state}\npredictions_comparable_why={why}\n")


def test_run_carries_a_fallback_toolkit_into_the_archive(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", 'echo "predictions=$WARPLENS_PREDICTIONS" > "$WARPLENS_RESULTS_DIR/p.txt"\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    why = "BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: the probes will be built by nvcc 12.8.93"
    versions_file(tmp_path / "home", "not-comparable", why)
    # The flagging is under test here; the predictions of the fallback toolkit have tests of
    # their own (this stand-in checkout has nothing to predict from).
    done = bash(root / "env/remote/run.sh", *args, "--skip", "predict-toolkit", env=env)
    assert done.returncode == 0, done.stdout + done.stderr          # flagged, not failed
    assert done.stdout.count(why) == 2                               # after setup, and as the last thing printed
    assert done.stdout.rstrip().splitlines()[-1] == "!" * 100
    files = members(archive_of(tmp_path / "home"))
    assert files["PREDICTIONS-NOT-COMPARABLE.txt"].decode().startswith(why)
    manifest = dict(line.split("=", 1) for line in files["manifest.txt"].decode().splitlines()
                    if "=" in line and not line.startswith("#"))
    assert manifest["predictions_comparable"] == "not-comparable" and manifest["predictions_comparable_why"] == why
    assert manifest["driver_cuda"] == "12.8" and manifest["toolkit_installed"] == "12.8.93"
    assert files["out/blackwell/p.txt"] == b"predictions=not-comparable\n"


def test_strict_toolkit_stops_before_the_suites(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", f'echo ran >> "{tmp_path}/ran"\n')
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    versions_file(tmp_path / "home", "not-comparable", "BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: x")
    done = bash(root / "env/remote/run.sh", *args, "--strict-toolkit", env=env)
    assert done.returncode == 3 and "stopping (--strict-toolkit)" in done.stdout
    assert not (tmp_path / "ran").exists()
    assert "PREDICTIONS-NOT-COMPARABLE.txt" in members(archive_of(tmp_path / "home"))   # the archive is still written


def test_run_with_the_predictions_toolkit_has_no_flag(checkout, tmp_path):
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", "echo ran\n")
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    for state in ("comparable", "comparable-native"):
        shutil.rmtree(tmp_path / "home", ignore_errors=True)
        versions_file(tmp_path / "home", state, "nvcc 13.5.37 as for the predictions")
        done = bash(root / "env/remote/run.sh", *args, "--strict-toolkit", "--redo", "all", env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "NOT COMPARABLE" not in done.stdout and "!!!!" not in done.stdout
        files = members(archive_of(tmp_path / "home"))
        assert "PREDICTIONS-NOT-COMPARABLE.txt" not in files
        assert f"predictions_comparable={state}\n" in files["manifest.txt"].decode()


def test_run_without_a_record_of_the_toolkit_says_so(checkout, tmp_path):
    """setup was skipped and left no versions.txt: unknown is flagged like a mismatch."""
    root, env, args = checkout
    suite(root, "blackwell", "echo built\n", "echo ran\n")
    suite(root, "blackwell_timing", "echo built\n", "echo ran\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert "TOOLKIT NOT RECORDED" in done.stdout
    assert "PREDICTIONS-NOT-COMPARABLE.txt" in members(archive_of(tmp_path / "home"))


# ---------------------------------------------------------------------------------------------
# PyTorch (for the Triton-on-GPU step)
# ---------------------------------------------------------------------------------------------

def test_torch_is_pinned_with_everything_it_brings():
    pins = read_pins()
    assert {"TORCH", "TORCH_CUDA_VERSION"} <= set(pins)
    brought = {key for key in pins if key.startswith("TORCH_PIP_")}
    assert {"TORCH_PIP_CUDA_TOOLKIT", "TORCH_PIP_NVIDIA_CUDA_RUNTIME", "TORCH_PIP_NVIDIA_CUDA_NVRTC",
            "TORCH_PIP_NVIDIA_CUDNN_CU13", "TORCH_PIP_SYMPY", "TORCH_PIP_NETWORKX"} <= brought
    # The CUDA wheels torch requires are of the release TORCH_CUDA_VERSION names, and of the
    # same major release as the toolkit, or the pinned cuda-bindings would not load with them.
    assert pins["TORCH_PIP_CUDA_TOOLKIT"].startswith(pins["TORCH_CUDA_VERSION"] + ".")
    assert pins["TORCH_PIP_NVIDIA_CUDA_RUNTIME"].startswith(pins["TORCH_CUDA_VERSION"] + ".")
    assert pins["TORCH_CUDA_VERSION"].split(".")[0] == pins["CUDA_VERSION"].split(".")[0]
    setup = (REMOTE / "setup.sh").read_text() + (REMOTE / "lib.sh").read_text()
    for use in ('"torch==$PIN_TORCH"', '"triton==$PIN_TRITON"', "${!PIN_TORCH_PIP_@}", "$PIN_TORCH_CUDA_VERSION",
                "--system-site-packages"):
        assert use in setup, use


def test_torch_is_not_installed_into_the_main_env():
    """With the CUDA 13.2 wheels in the env, cuda-bindings loads NVRTC 13.2 instead of the
    toolkit's (measured). So torch goes into its own venv, and setup.sh checks NVRTC."""
    setup = (REMOTE / "setup.sh").read_text()
    main_install = setup[setup.index('"triton==$PIN_TRITON" "nvidia-cutlass-dsl=='):setup.index("5b. PyTorch")]
    assert "torch==" not in main_install
    assert "install_torch_env" in setup and 'nvrtc_loaded" == "$toolkit"' in setup
    lib_text = (REMOTE / "lib.sh").read_text()
    assert 'TORCH_ENV="$REMOTE_HOME/torch-env"' in lib_text
    assert 'env_run python -m venv --system-site-packages "$TORCH_ENV"' in lib_text
    # pip unpacks gigabytes under TMPDIR; /tmp was too small on the development machine.
    assert 'TMPDIR="$REMOTE_HOME/tmp"' in lib_text and 'export TMPDIR="$REMOTE_HOME/tmp"' in setup


@pytest.mark.parametrize("driver,action", [
    ("13.5", "install"), ("13.2", "install"), ("14.0", "install"),
    ("13.0", "install"),      # measured: the CUDA 13.2 wheel on the development machine's driver
    ("", "install"),
    ("12.9", "skip"), ("12.8", "skip"),
])
def test_torch_follows_the_driver(driver, action):
    got, _, why = lib("choose_torch", driver, "13.2").partition("|")
    assert got == action and why
    if action == "skip":
        assert "SKIPPED" in why and "12.x driver cannot load" in why


def test_check_env_reports_a_missing_torch():
    """Run by an interpreter without torch, `check_env.py torch` says MISSING and fails;
    preflight turns that into a warning."""
    probe = subprocess.run(["python3", "-c", "import torch"], capture_output=True)
    if probe.returncode == 0:
        pytest.skip("python3 has torch")
    done = subprocess.run(["python3", str(REMOTE / "check_env.py"), "torch"], capture_output=True, text=True)
    assert done.returncode == 1 and done.stdout.startswith("torch=MISSING (")
