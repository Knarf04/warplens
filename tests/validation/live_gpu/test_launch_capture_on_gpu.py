"""The capture of matrix row RT-11 on whatever CUDA GPU this machine has: the tracer's whole path
for real (tests/validation/capture/test_launch_capture.py has what can be shown without a GPU).

Every test here is marked `gpu`: it is deselected where no GPU is in reach (tests/conftest.py),
and with one it needs gdb with Python support and skips with the reason for whatever else it
lacks.
  - the CuTe DSL's SAXPY through e2e/cute_dsl_tma.py --script, held against the frontend
    capture of the same script;
  - a CUDA C++ kernel that takes 128-byte, 64-byte-aligned structs by value with device
    addresses inside, as a kernel with tensor maps does (data/launch_capture_descriptors.cu);
  - with a CUDA PyTorch named by WARPLENS_TORCH_PYTHON as well: a CuTe DSL kernel over PyTorch
    tensors, traced with the settings the tutorial is traced with (data/cute_dsl_torch_saxpy.py).
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from tests.support import REPO, TESTS
from tests.support.probes import E2E, load_e2e

DATA = TESTS / "validation" / "live_gpu" / "data"

launch_capture = load_e2e("launch_capture_under_test", E2E / "launch_capture.py")


# ---------------------------------------------------------------------------------------------
# The tracer itself, on whatever CUDA GPU this machine has
# ---------------------------------------------------------------------------------------------

def gpu_tracer():
    """The tracer module, or a skip with the reason: no GPU in reach, no gdb."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CUDA_VISIBLE_DEVICES is empty: this run is to stay off the GPU")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("needs a CUDA GPU (no nvidia-smi)")
    if smi.returncode != 0 or not smi.stdout.strip():
        pytest.skip("needs a CUDA GPU (nvidia-smi found none)")
    tracer = load_e2e("driver_call_trace_for_capture", E2E / "driver_call_trace.py")
    if not tracer.find_gdb()[0]:
        pytest.skip("needs gdb with Python support")
    return tracer


def skip_unless_traceable(text: str) -> None:
    if "ptrace is not permitted" in text or "SKIPPED:" in text:
        pytest.skip("this machine does not let gdb trace a process")


@pytest.mark.gpu
def test_the_whole_capture_of_a_cute_dsl_launch_on_this_gpu(tmp_path):
    """e2e/cute_dsl_tma.py --script with the CuTe DSL's SAXPY, for real: module, PTX,
    allocations, the memory before and after. Held against what the frontend capture of the
    same script saves from inside the program (its PTX, its inputs, its output)."""
    gpu_tracer()
    pytest.importorskip("cutlass")
    numpy = pytest.importorskip("numpy")
    script = REPO / "examples" / "06_cute_dsl" / "saxpy.py"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = tmp_path / "rt11"
    done = subprocess.run([sys.executable, str(E2E / "cute_dsl_tma.py"), "--out", str(out), "--script", str(script)],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
    skip_unless_traceable(done.stdout)
    if "plain run of saxpy.py: ok" not in done.stdout:
        pytest.skip("the CuTe DSL's SAXPY does not run on this GPU: " + done.stdout[-300:])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "capture: complete" in done.stdout
    assert (out / "summary.txt").read_text().startswith("RT-11 capture: COMPLETE\n")
    assert launch_capture.main(["verify", str(out)]) == 0           # again, as it would be offline

    cap = out / "capture"
    capture = json.loads((cap / "capture.json").read_text())
    (launch,) = json.loads((cap / "launches.json").read_text())
    assert capture["problems"] == []
    # The code: the DSL hands the driver a fat binary; its own cubin dump is inside it.
    (module,) = capture["modules"]
    assert module["call"] == "cuLibraryLoadData" and module["format"] == "fatbin" and module["result"] == 0
    (cubin,) = (cap / "dsl_dump").glob("*.cubin")
    assert cubin.read_bytes() in (cap / module["file"]).read_bytes()
    (lookup,) = capture["lookups"]
    assert lookup["name"] == launch["kernel"] and lookup["module_handle"] == module["handle"]
    # Two allocations of 1024 bytes, followed from cuMemAlloc and confirmed by the driver.
    memory = launch["memory"]
    assert [p["parameter"] for p in memory["pointers"]] == [1, 2] and all(p["offset"] == 0 for p in memory["pointers"])
    assert all(a["size"] == a["driver_size"] == 1024 and a["known_from"] == "followed" for a in memory["allocations"])
    assert memory["launch_result"] == 0 and memory["synchronize_result"] == 0

    # The same script's frontend capture, made inside the program with no debugger.
    raw = tmp_path / "frontend"
    again = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=REPO,
                           env={**env, "WARPLENS_CAPTURE_DIR": str(raw), "WARPLENS_CAPTURE_MODE": "gpu"}, timeout=900)
    case = raw / "raw" / "cute_dsl_saxpy"
    assert (case / "y.output.npy").is_file(), again.stdout + again.stderr
    (ptx,) = (cap / "dsl_dump").glob("*.ptx")
    assert ptx.read_text() == (case / "kernel.ptx").read_text()
    by_parameter = {p["parameter"]: next(a for a in memory["allocations"] if a["base"] == p["allocation"])
                    for p in memory["pointers"]}
    contents = lambda record: numpy.fromfile(cap / record["file"], dtype=numpy.float32)[:256]
    x, y, result = (numpy.load(case / name) for name in ("x.input.npy", "y.input.npy", "y.output.npy"))
    assert (contents(by_parameter[1]["before"]) == x).all() and (contents(by_parameter[1]["after"]) == x).all()
    assert (contents(by_parameter[2]["before"]) == y).all()
    assert (contents(by_parameter[2]["after"]) == result).all()     # the expected result, bit for bit
    assert not (result == y).all()


@pytest.mark.gpu
def test_addresses_inside_128_byte_parameters_are_followed_on_this_gpu(tmp_path):
    """tests/validation/live_gpu/data/launch_capture_descriptors.cu: three 128-byte, 64-byte-aligned structs by
    value. The address at offset 0 and the one at offset 3 are found; the packed one is not,
    and its memory is in the capture all the same, because every live allocation is read."""
    tracer = gpu_tracer()
    numpy = pytest.importorskip("numpy")
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        pytest.skip("needs nvcc")
    program = tmp_path / "descriptors"
    built = subprocess.run([nvcc, "-arch=native", "-o", str(program), str(DATA / "launch_capture_descriptors.cu")],
                           capture_output=True, text=True, timeout=600)
    if built.returncode != 0:
        pytest.skip("nvcc cannot build for this GPU: " + built.stderr[-300:])
    plain = subprocess.run([str(program)], capture_output=True, text=True, timeout=300)
    if plain.stdout.strip() != "PASS":
        pytest.skip("the program does not run on this GPU: " + (plain.stdout + plain.stderr)[-300:])

    out = tmp_path / "rt11"
    (out / "capture").mkdir(parents=True)
    (out / "tutorial.stdout").write_text(plain.stdout)
    (out / "capture" / "expect.json").write_text(json.dumps({"descriptor": True, "ptx": False, "last_line": "PASS"}))
    status, facts = tracer.trace([str(program)], out / "capture", 600, capture=("module", "memory"))
    if status.startswith("skipped"):
        pytest.skip(status)
    assert status == "ok"
    (launch,) = facts["launches"]
    assert launch["kernel"] == "add_three" and [p["size"] for p in launch["parameters"]] == [4, 128, 128, 128, 8]

    problems, warnings, _ = launch_capture.verify(out)
    assert problems == []
    assert len(warnings) == 1 and "no pointer was found in parameter 3 (128 bytes)" in warnings[0]

    memory = launch["memory"]
    assert [(p["parameter"], p["offset"]) for p in memory["pointers"]] == [(1, 0), (2, 3), (4, 0)]
    assert memory["live_allocations"] == 4 and memory["all_live_allocations_read"] is True
    assert [a["referenced"] for a in memory["allocations"]] == [True, True, True, False]
    assert all(a["known_from"] == "followed" and a["size"] == a["driver_size"] == 4000 for a in memory["allocations"])
    contents = lambda record: numpy.fromfile(out / "capture" / record["file"], dtype=numpy.float32)
    by_parameter = {p["parameter"]: next(a for a in memory["allocations"] if a["base"] == p["allocation"])
                    for p in memory["pointers"]}
    index = numpy.arange(1000, dtype=numpy.float32)
    assert (contents(by_parameter[1]["before"]) == 1000 + index).all()
    assert (contents(by_parameter[2]["before"]) == 2000 + index).all()
    packed = memory["allocations"][3]                   # reached only by reading every live allocation
    assert (contents(packed["before"]) == 3000 + index).all() and packed["after"] == packed["before"]
    assert (contents(by_parameter[4]["before"]) == 0).all()
    assert (contents(by_parameter[4]["after"]) == 6000 + 3 * index).all()
    # The code: what nvcc links in is a wrapper around a fat binary; the fat binary is what is saved.
    capture = json.loads((out / "capture" / "capture.json").read_text())
    (module,) = capture["modules"]
    assert module["format"] == "wrapped fatbin"
    assert (out / "capture" / module["file"]).read_bytes()[:4] == bytes.fromhex("50ed55ba")
    assert capture["problems"] == []


@pytest.mark.gpu
def test_a_cute_dsl_launch_over_pytorch_tensors_on_this_gpu(tmp_path):
    """tests/validation/live_gpu/data/cute_dsl_torch_saxpy.py has the tutorial's host side (tensors from PyTorch's
    caching allocator, from_dlpack, a PyTorch reference computed on the GPU afterwards) around a
    kernel any GPU runs. Traced as cute_dsl_tma.py traces the tutorial: PyTorch's own kernels
    keep their parameters to themselves, their modules are listed and not kept."""
    tracer = gpu_tracer()
    numpy = pytest.importorskip("numpy")
    python = os.environ.get("WARPLENS_TORCH_PYTHON")
    if not python:
        pytest.skip("needs WARPLENS_TORCH_PYTHON: an interpreter with a CUDA PyTorch and the CuTe DSL")
    script = DATA / "cute_dsl_torch_saxpy.py"
    plain = subprocess.run([python, str(script)], capture_output=True, text=True, timeout=900)
    if plain.stdout.strip().splitlines()[-1:] != ["PASS"]:
        pytest.skip("the script does not run with that interpreter on this GPU: " + plain.stderr[-300:])

    out = tmp_path / "rt11"
    cap = out / "capture"
    (cap / "dsl_dump").mkdir(parents=True)
    (out / "tutorial.stdout").write_text(plain.stdout)
    (cap / "expect.json").write_text(json.dumps({"descriptor": False, "ptx": True, "last_line": "PASS"}))
    env = {**os.environ, "CUTE_DSL_KEEP": "cubin,ptx", "CUTE_DSL_DUMP_DIR": str(cap / "dsl_dump"),
           "WARPLENS_TRACE_MEMORY_SKIP": "^_Z", "WARPLENS_TRACE_MODULE_SKIP_WRAPPED": "1"}
    status, facts = tracer.trace([python, str(script)], cap, 900, env=env, capture=("module", "memory"))
    if status.startswith("skipped"):
        pytest.skip(status)
    assert status == "ok"
    problems, warnings, _ = launch_capture.verify(out)
    assert problems == [] and warnings == []

    capture = json.loads((cap / "capture.json").read_text())
    ours = [l for l in facts["launches"] if not l["kernel"].startswith("_Z")]
    theirs = [l for l in facts["launches"] if l["kernel"].startswith("_Z")]
    assert len(ours) == 1 and theirs, [l.get("kernel") for l in facts["launches"]]
    assert all("skipped" in l["memory"] for l in theirs)
    kept = [m for m in capture["modules"] if "file" in m]
    assert len(kept) == 1 and kept[0]["format"] == "fatbin"
    assert all(m["format"] == "wrapped fatbin" and m["size"] > 0 for m in capture["modules"] if "not_kept" in m)
    # Both tensors are inside one allocation of PyTorch's allocator, which the tracer followed
    # from cuMemAlloc (through cudaMalloc) and the driver confirms.
    memory = ours[0]["memory"]
    (allocation,) = memory["allocations"]
    assert allocation["known_from"] == "followed" and allocation["driver_size"] == allocation["size"] >= 2 * 4096 * 4
    x_at, y_at = (next(p["offset_in_allocation"] for p in memory["pointers"] if p["parameter"] == i) for i in (1, 2))
    assert x_at != y_at
    tensor = lambda record, at: numpy.frombuffer((cap / record["file"]).read_bytes(), dtype=numpy.float32,
                                                 count=4096, offset=at)
    x, y = tensor(allocation["before"], x_at), tensor(allocation["before"], y_at)
    assert set(numpy.unique(x)) <= {-2.0, -1.0, 0.0, 1.0} and len(numpy.unique(x)) > 1      # random_(-2, 2)
    assert (tensor(allocation["after"], x_at) == x).all()
    assert (tensor(allocation["after"], y_at) == numpy.float32(2.5) * x + y).all()           # exact in float32
