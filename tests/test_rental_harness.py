"""What must not fail silently on the rented machine: a probe that never launched is not a
result, and a run built by another toolkit than the predictions' says so.

The driver of a rented machine can be older than the toolkit the probes were written with.
It then refuses the toolkit's PTX (measured on the development machine: toolkit 13.5, driver
for CUDA 13.0, `cudaErrorUnsupportedPtxVersion`). These tests pin down how that shows:

* a probe prints `LAUNCH FAILED` and exits with status 4, never `KERNEL FAILED` (which for
  many probes is the predicted outcome);
* `run_all.py` records it as `launch-failed`, falls back to the machine-code programs when
  they run, and exits with status 1 when anything did not launch;
* `compare.py` gives `FAILED TO LAUNCH`, also for a probe that is allowed to fault and whose
  prediction is a refused kernel;
* the nvcc that built the programs is compared with the one the predictions were made with.

No GPU and no nvcc are needed: the programs here are shell scripts that print what a probe
prints. The real thing was run once by hand on the RTX 3080 Ti (validation/blackwell/README.md,
"Dry run").
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "validation" / "blackwell"
TIMING = ROOT / "validation" / "blackwell_timing"
REMOTE = ROOT / "env" / "remote"

LAUNCH_FAILED = ("LAUNCH FAILED: one arrival per phase, six phases\n"
                 "info: error cudaErrorUnsupportedPtxVersion at launch: the provided PTX was compiled with an "
                 "unsupported toolchain.\n")


def pins() -> dict[str, str]:
    lines = (REMOTE / "pins.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line.strip() and not line.lstrip().startswith("#"))


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def suite(tmp_path):
    """A copy of the harness (not of the probes) whose bin/ holds a program the 'driver'
    refuses and whose bin-native/ holds one that prints the committed prediction."""
    root = tmp_path / "validation" / "blackwell"
    root.mkdir(parents=True)
    for name in ("manifest.py", "run_all.py", "compare.py", "probe_process.py"):
        shutil.copy(SUITE / name, root / name)
    shutil.copytree(SUITE / "predicted", root / "predicted")
    script(root / "bin" / "mbarrier" / "script", f"printf '%s' '{LAUNCH_FAILED}'\nexit 4\n")
    script(root / "bin-native" / "mbarrier" / "script",
           'case "$1" in basic) cat "$(dirname "$0")/../../predicted/mbarrier/phase_parity.txt" ;; '
           '*) echo "KERNEL FAILED: x"; exit 3 ;; esac\n')
    nvcc = json.loads((SUITE / "predicted" / "provenance.json").read_text())["nvcc_version"]
    for directory in ("bin", "bin-native"):
        (root / directory / "toolkit.txt").write_text(f"nvcc_version={nvcc}\n")
    return root


def run_all(suite: Path, *args: str):
    return subprocess.run([sys.executable, str(suite / "run_all.py"), "--no-build", "--no-e2e", "--only",
                           "mbarrier/phase_parity", "--out", str(suite / "results" / "run"), *args],
                          capture_output=True, text=True, env={**os.environ, "PATH": os.environ["PATH"]})


def compare(suite: Path, *args: str):
    return subprocess.run([sys.executable, str(suite / "compare.py"), str(suite / "results" / "run"), *args],
                          capture_output=True, text=True)


def verdict(output: str, name: str) -> str:
    line = next(l for l in output.splitlines() if l.startswith(name + " "))
    return re.split(r"\s{2,}", line)[1]


# ---------------------------------------------------------------------------------------------
# The probes' own error handling
# ---------------------------------------------------------------------------------------------

def load_errors(header: Path) -> set[str]:
    text = header.read_text()
    body = text[text.index("_is_load_error(cudaError_t error)"):]
    body = body[:body.index("default:")]
    return set(re.findall(r"case (cudaError\w+):", body))


def test_both_suites_call_the_same_errors_a_failed_launch():
    probes, benches = load_errors(SUITE / "common.h"), load_errors(TIMING / "common.h")
    assert probes == benches
    # What the development machine's driver answers to PTX from a newer toolkit, and what a
    # program built for another GPU gets.
    assert {"cudaErrorUnsupportedPtxVersion", "cudaErrorNoKernelImageForDevice", "cudaErrorInvalidPtx",
            "cudaErrorJitCompilerNotFound"} <= probes
    # A fault inside a kernel is a result, not a failed launch.
    assert not {"cudaErrorLaunchFailure", "cudaErrorIllegalInstruction", "cudaErrorIllegalAddress",
                "cudaErrorInvalidValue"} & probes


def test_probe_sync_checks_the_launch_and_the_synchronize():
    text = (SUITE / "common.h").read_text()
    body = text[text.index("static inline void probe_sync("):]
    body = body[:body.index("\n}\n")]
    assert body.count("cudaGetLastError()") == 2 and "cudaDeviceSynchronize()" in body
    assert body.index("cudaGetLastError()") < body.index("cudaDeviceSynchronize()") < body.rindex("cudaGetLastError()")
    report = text[text.index("static inline void probe_report_failure("):text.index("static inline void probe_sync(")]
    assert 'printf("LAUNCH FAILED: %s\\n"' in report and "exit(4)" in report
    assert 'printf("KERNEL FAILED: %s\\n"' in report and "exit(3)" in report


def test_every_probe_source_checks_each_launch():
    """A launch is followed by probe_sync, in the source or in the shared header that launches
    for it; a probe that reads a launch status itself must hand load errors to
    probe_report_failure or test them with probe_is_load_error."""
    launch = re.compile(r"<<<|launch_cluster\(|cudaLaunchKernelEx\(|cudaLaunchKernel\(")
    for source in sorted(SUITE.glob("*/*.cu")) + sorted(SUITE.glob("*.h")) + sorted(SUITE.glob("*/*.h")):
        text = "\n".join(line for line in source.read_text().splitlines() if not line.lstrip().startswith("//"))
        launches = len(launch.findall(text))
        if source.name == "common.h":
            launches -= 1  # launch_cluster's own cudaLaunchKernelEx; its callers are counted
        checks = len(re.findall(r"probe_sync\(|probe_report_failure\(|probe_is_load_error\(", text))
        assert checks >= launches, f"{source.relative_to(ROOT)}: {launches} launch(es), {checks} check(s)"


def test_every_benchmark_launch_goes_through_the_checked_harness():
    text = (TIMING / "common.h").read_text()
    timed = text[text.index("void timed_launch("):text.index("// A configuration:")]
    # The launch's own error first, then the synchronize, then what the kernel left behind.
    assert re.search(r"launch\(\);\s*CHECK\(cudaGetLastError\(\)\);.*CHECK\(cudaDeviceSynchronize\(\)\);\s*"
                     r"CHECK\(cudaGetLastError\(\)\);", timed, flags=re.S)
    assert re.search(r"bench_spin_kernel<<<[^;]*;\s*CHECK\(cudaGetLastError\(\)\);", text)
    assert "class=%s" in text and "exit(load_ ? 4 : 1)" in text
    for source in sorted(TIMING.glob("*.cu")):
        code = "\n".join(line for line in source.read_text().splitlines() if not line.lstrip().startswith("//"))
        # Every benchmark launches through measure() or timed_launch(), which check; a launch
        # made any other way needs its own CHECK(cudaGetLastError()).
        launches = len(re.findall(r"<<<|cudaLaunchKernelEx\(|cudaLaunchKernel\(", code))
        guarded = len(re.findall(r"\bmeasure\(|\btimed_launch\(|CHECK\(cudaGetLastError\(\)\)", code))
        assert launches == 0 or guarded >= 1, source.name


# ---------------------------------------------------------------------------------------------
# run_all.py
# ---------------------------------------------------------------------------------------------

def harness(suite: Path):
    sys.path.insert(0, str(suite))
    for name in ("run_all", "manifest"):
        sys.modules.pop(name, None)
    try:
        import run_all
        return run_all
    finally:
        sys.path.remove(str(suite))


def test_statuses_that_are_not_results_get_their_own_name(suite):
    module = harness(suite)
    assert module.classify("exit 4", LAUNCH_FAILED.encode()) == "launch-failed"
    assert module.classify("exit 2", b"header\nHOST ERROR: cudaMalloc(&p, n) returned cudaErrorMemoryAllocation\n") == "host-error"
    assert module.classify("exit 3", b"KERNEL FAILED: alloc\ninfo: error cudaErrorLaunchFailure\n") == "exit 3"
    assert module.classify("ok", b"LAUNCH FAILED is only text here\n") == "ok"
    assert module.classify("timeout", b"") == "timeout"


def test_the_driver_refuses_the_ptx_and_the_machine_code_runs(suite):
    done = run_all(suite)                                   # --binaries auto
    assert done.returncode == 0, done.stdout + done.stderr
    note = (suite / "results" / "run" / "env" / "binaries.txt").read_text()
    assert "canary bin/mbarrier/script basic: exit 4" in note and "cudaErrorUnsupportedPtxVersion" in note
    assert "canary bin-native/mbarrier/script basic: ok" in note and note.endswith("using bin-native/\n")
    assert "mbarrier/phase_parity\tok\t" in (suite / "results" / "run" / "status.tsv").read_text()
    toolkit = dict(line.split("=", 1) for line in
                   (suite / "results" / "run" / "env" / "toolkit.txt").read_text().splitlines())
    assert toolkit["binaries"] == "bin-native" and toolkit["predictions_comparable"] == "yes"
    assert not (suite / "results" / "run" / "TOOLKIT-MISMATCH.txt").exists()
    compared = compare(suite)
    assert verdict(compared.stdout, "mbarrier/phase_parity") == "identical"
    assert "NOT COMPARABLE" not in compared.stdout


def test_a_probe_that_does_not_launch_is_never_a_result(suite):
    done = run_all(suite, "--binaries", "ptx")
    assert done.returncode == 1                              # the stage fails in env/remote/run.sh
    assert "1 PROBE(S) FAILED TO LAUNCH" in done.stdout
    assert "mbarrier/phase_parity\tlaunch-failed\t" in (suite / "results" / "run" / "status.tsv").read_text()
    compared = compare(suite)
    assert compared.returncode == 1
    assert verdict(compared.stdout, "mbarrier/phase_parity") == "FAILED TO LAUNCH"
    assert "cudaErrorUnsupportedPtxVersion" in compared.stdout and "--binaries native" in compared.stdout


def test_neither_build_runs(suite):
    script(suite / "bin-native" / "mbarrier" / "script", f"printf '%s' '{LAUNCH_FAILED}'\nexit 4\n")
    done = run_all(suite)
    assert done.returncode == 1
    assert "NEITHER BUILD RUNS" in (suite / "results" / "run" / "env" / "binaries.txt").read_text()
    assert verdict(compare(suite).stdout, "mbarrier/phase_parity") == "FAILED TO LAUNCH"


@pytest.mark.parametrize("name", ["tmem/alloc_increasing", "tma/misaligned_requests", "mbarrier/expect_tx_large"])
def test_by_design_never_covers_a_failed_launch(suite, name):
    """These probes may fault or hang by design, and the simulator's prediction for the first
    two is a refused kernel. A program that could not be loaded must still not pass."""
    out = suite / "results" / "run"
    target = out / (name + ".stdout")
    target.parent.mkdir(parents=True)
    target.write_text(LAUNCH_FAILED)
    for status in ("launch-failed", "exit 4"):             # as run_all.py writes it, and unclassified
        (out / "status.tsv").write_text(f"name\tstatus\tseconds\tcommand\n{name}\t{status}\t0.1\tx\n")
        compared = compare(suite)
        assert compared.returncode == 1
        assert verdict(compared.stdout, name) == "FAILED TO LAUNCH", compared.stdout
    # The predicted refusal itself still counts as agreement.
    predicted = (suite / "predicted" / (name + ".txt")).read_text()
    if "KERNEL FAILED" in predicted:
        target.write_text(predicted)
        (out / "status.tsv").write_text(f"name\tstatus\tseconds\tcommand\n{name}\texit 3\t0.1\tx\n")
        assert verdict(compare(suite).stdout, name) == "identical"
    # And a host error is neither.
    target.write_text("HOST ERROR: cudaMalloc(&p, 4) returned cudaErrorMemoryAllocation\n")
    (out / "status.tsv").write_text(f"name\tstatus\tseconds\tcommand\n{name}\thost-error\t0.1\tx\n")
    assert verdict(compare(suite).stdout, name) == "HOST ERROR"


# ---------------------------------------------------------------------------------------------
# Which toolkit built the programs
# ---------------------------------------------------------------------------------------------

def test_the_predictions_name_the_pinned_compiler():
    """The committed predictions are the simulator's run of PTX from one nvcc; it must be the
    one env/remote/pins.env installs, or every rental starts as 'not comparable'."""
    pinned = pins()["CUDA_NVCC"]
    assert json.loads((SUITE / "predicted" / "provenance.json").read_text())["nvcc_version"] == pinned
    assert json.loads((TIMING / "predicted" / "meta.json").read_text())["nvcc_version"] == pinned


def test_ci_and_remote_setup_select_the_same_compiler_patch():
    """A cuda-version=major.minor pin alone floats to a new patch and makes CI
    reject its own committed predictions before running any tests."""
    import yaml

    environment = yaml.safe_load((ROOT / "env" / "environment.yml").read_text())
    dependencies = dict(item.split("=", 1) for item in environment["dependencies"] if "=" in item)
    remote = pins()
    for package, key in (("cuda-version", "CUDA_VERSION"), ("cuda-nvcc", "CUDA_NVCC"),
                         ("cuda-cudart-dev", "CUDA_CUDART_DEV"), ("cuda-nvrtc", "CUDA_NVRTC")):
        assert dependencies.get(package) == remote[key], f"{package} must pin the prediction compiler"


def test_a_fallback_toolkit_is_flagged_everywhere(suite):
    for directory in ("bin", "bin-native"):
        (suite / directory / "toolkit.txt").write_text("nvcc_version=13.0.48\n")
    done = run_all(suite)
    assert done.returncode == 0                              # the probes ran; the run is flagged, not failed
    assert done.stdout.count("BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE") >= 2
    out = suite / "results" / "run"
    assert "nvcc 13.0.48" in (out / "TOOLKIT-MISMATCH.txt").read_text()
    assert "predictions_comparable=no" in (out / "env" / "toolkit.txt").read_text()
    compared = compare(suite)
    assert compared.stdout.count("BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE") == 2  # before and after
    assert verdict(compared.stdout, "mbarrier/phase_parity") == "identical"
    strict = compare(suite, "--require-same-toolkit")
    assert strict.returncode == 3 and "identical" not in strict.stdout  # 2 is "unverified differences only"
    # Predictions made again with that very nvcc (any machine can do it) are comparable:
    # compare.py judges by the predictions it is given, not by what the run recorded.
    again = suite / "predicted-13.0"
    shutil.copytree(suite / "predicted", again)
    (again / "provenance.json").write_text('{"nvcc_version": "13.0.48"}\n')
    regenerated = compare(suite, "--predicted", str(again), "--require-same-toolkit")
    assert "NOT COMPARABLE" not in regenerated.stdout
    assert verdict(regenerated.stdout, "mbarrier/phase_parity") == "identical"


def test_toolkit_report(suite):
    module = harness(suite)
    module.HERE, module.PREDICTED = suite, suite / "predicted"
    smi = "| NVIDIA-SMI 580.173.02   Driver Version: 580.173.02   CUDA Version: 13.0  |"
    same = module.toolkit_report(suite / "bin", smi, "580.173.02")
    assert same["predictions_comparable"] == "yes" and same["driver_cuda"] == "13.0"
    assert same["built_nvcc_version"] == same["predictions_nvcc_version"]
    assert module.toolkit_report(suite / "bin-native")["predictions_comparable"] == "yes"
    (suite / "bin" / "toolkit.txt").unlink()
    assert module.toolkit_report(suite / "bin")["predictions_comparable"] == "unknown"
    (suite / "bin-ampere").mkdir()
    (suite / "bin-ampere" / "toolkit.txt").write_text("nvcc_version=12.4.131\n")
    assert module.toolkit_report(suite / "bin-ampere")["predictions_comparable"] == "no"


def test_a_partial_regeneration_with_another_compiler_is_refused(suite):
    module = harness(suite)
    module.HERE, module.PREDICTED = suite, suite / "predicted"
    recorded = (suite / "predicted" / "provenance.json").read_text()
    module.write_provenance(partial=True)                    # the same compiler: nothing changes
    assert (suite / "predicted" / "provenance.json").read_text() == recorded
    (suite / "bin" / "toolkit.txt").write_text("nvcc_version=12.8.93\n")
    with pytest.raises(SystemExit, match="Regenerate all of them"):
        module.write_provenance(partial=True)
    module.write_provenance(partial=False)                   # a full regeneration may change it
    assert json.loads((suite / "predicted" / "provenance.json").read_text())["nvcc_version"] == "12.8.93"


def test_build_scripts_record_their_compiler(tmp_path):
    """build.sh writes <bin>/toolkit.txt from `nvcc --version`; with a stand-in nvcc."""
    fake = script(tmp_path / "tools" / "nvcc",
                  'case "$1" in --version) echo "nvcc: NVIDIA (R) Cuda compiler driver"; '
                  'echo "Cuda compilation tools, release 13.0, V13.0.48"; exit 0 ;; esac\n'
                  'out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done; : > "$out"\n')
    work = tmp_path / "suite"
    work.mkdir()
    shutil.copy(SUITE / "build.sh", work / "build.sh")
    (work / "e2e").mkdir()
    (work / "e2e" / "device_attributes.cu").write_text("int main() { return 0; }\n")
    done = subprocess.run(["bash", str(work / "build.sh"), "--native", "e2e/device_attributes"], capture_output=True,
                          text=True, env={**os.environ, "NVCC": str(fake)})
    assert done.returncode == 0, done.stdout + done.stderr
    for directory in ("bin", "bin-native"):
        assert "nvcc_version=13.0.48\n" in (work / directory / "toolkit.txt").read_text()
    timing = tmp_path / "timing"
    timing.mkdir()
    shutil.copy(TIMING / "build.sh", timing / "build.sh")
    (timing / "launch.cu").write_text("int main() { return 0; }\n")
    done = subprocess.run(["bash", str(timing / "build.sh"), "hw", "sim"], capture_output=True, text=True,
                          env={**os.environ, "NVCC": str(fake), "BENCHES": "launch"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "nvcc_version=13.0.48\n" in (timing / "bin" / "hw" / "toolkit.txt").read_text()
    assert "nvcc_version=13.0.48\n" in (timing / "bin" / "sim" / "toolkit.txt").read_text()


# ---------------------------------------------------------------------------------------------
# The timing suite's runner
# ---------------------------------------------------------------------------------------------

def timing_run(tmp_path: Path, launch_body: str, built: str | None):
    work = tmp_path / "timing"
    (work / "predicted").mkdir(parents=True)
    shutil.copy(TIMING / "run_all.sh", work / "run_all.sh")
    shutil.copy(TIMING / "predicted" / "meta.json", work / "predicted" / "meta.json")
    script(work / "bin" / "hw" / "launch", launch_body)
    if built:
        (work / "bin" / "hw" / "toolkit.txt").write_text(f"nvcc_version={built}\n")
    done = subprocess.run(["bash", str(work / "run_all.sh"), "--no-ncu", "--no-fit", "--out", str(work / "out")],
                          capture_output=True, text=True)
    return done, (work / "out" / "status.txt").read_text(), (work / "out" / "TOOLKIT.txt").read_text()


def test_timing_runner_tells_a_failed_launch_from_a_failed_benchmark(tmp_path):
    predicted = json.loads((TIMING / "predicted" / "meta.json").read_text())["nvcc_version"]
    refused = ('echo "DEVICE bench=launch"\n'
               'echo "BENCH_ERROR where=\\"cudaGetLastError()\\" error=cudaErrorUnsupportedPtxVersion class=launch-failed"\n'
               "exit 4\n")
    done, status, toolkit = timing_run(tmp_path / "a", refused, predicted)
    assert done.returncode > 0
    assert re.search(r"^launch: exit=4 .* LAUNCH FAILED \(the driver could not load the kernels: "
                     r"cudaErrorUnsupportedPtxVersion\)$", status, flags=re.M)
    assert "predictions_comparable=yes" in toolkit and "NOT COMPARABLE" not in status
    faulted = ('echo "BENCH_ERROR where=\\"cudaDeviceSynchronize()\\" error=cudaErrorLaunchFailure class=failed"\n'
               "exit 1\n")
    _, status, _ = timing_run(tmp_path / "b", faulted, predicted)
    assert re.search(r"^launch: exit=1 .* FAILED$", status, flags=re.M)
    _, status, toolkit = timing_run(tmp_path / "c", 'echo "BENCH id=launch"\n', "13.0.48")
    assert re.search(r"^launch: exit=0 .* ok$", status, flags=re.M)
    assert "predictions_comparable=no" in toolkit
    assert "toolkit: BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE" in status
    _, status, toolkit = timing_run(tmp_path / "d", 'echo "BENCH id=launch"\n', None)
    assert "predictions_comparable=unknown" in toolkit and "TOOLKIT NOT RECORDED" in status


# ---------------------------------------------------------------------------------------------
# Predictions must not depend on which compiler built the probes
# ---------------------------------------------------------------------------------------------

def test_no_thread_test_runs_into_a_warp_wide_allocation():
    """`if (tid == 0) {...}` directly followed by `if (warp == 0) tmem_alloc(...)` lets a
    compiler merge the two tests, so that lane 0 reaches the allocation by a path of its own
    (nvcc 13.0 does; nvcc 13.5 does not). This once made 15 predictions depend on
    the compiler; the rendezvous regression now covers that shape directly.
    Keep the probe's explicit synchronization between initialization and allocation
    so its memory-ordering intent remains clear."""
    alloc = re.compile(r"tmem_alloc\(|tcgen05\.alloc")
    thread_test = re.compile(r"if \((tid|threadIdx\.x) == 0\)")
    barrier = re.compile(r"__syncthreads\(\)|cluster_sync\(\)")
    sources = sorted(SUITE.glob("*/*.cu")) + [SUITE / "tcgen05_common.h"]
    checked = 0
    for source in sources:
        lines = [line for line in source.read_text().splitlines() if not line.lstrip().startswith("//")]
        for index, line in enumerate(lines):
            if not alloc.search(line) or "__forceinline__" in line or "printf" in line or line.lstrip().startswith("asm"):
                continue
            # Back to the start of the kernel: the last thread test must be followed by a barrier.
            start = next((i for i in range(index, -1, -1) if "__global__" in lines[i]), 0)
            before = lines[start:index]
            tests = [i for i, text in enumerate(before) if thread_test.search(text)]
            if tests:
                checked += 1
                assert any(barrier.search(text) for text in before[tests[-1]:]), \
                    f"{source.relative_to(ROOT)}: no __syncthreads() between a `tid == 0` test and the allocation: {line.strip()}"
    assert checked >= 5


def reconvergence():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reconvergence", SUITE / "findings" / "reconvergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", ["A", "B", "C", "D"])
def test_lanes_reconverge_in_the_layouts_nvcc_13_5_and_13_0_emit(shape):
    """A and B are what nvcc 13.5 emits, C and D what nvcc 13.0 does. In C and D lanes 1 to 31
    passed the join, and a `bar.sync` there, before lane 0 arrived, until the warp learned to
    bring lanes together that meet before their branch's post-dominator (core/warp.py)."""
    assert reconvergence().lanes_reading_seven(shape, barrier=True) == 32
    assert reconvergence().lanes_reading_seven(shape, barrier=False) == 32
