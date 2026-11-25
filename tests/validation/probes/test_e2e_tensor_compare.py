"""compare.py holds the GPU's end-to-end tensors against the simulator's (RT-06 to RT-09).

A missing artifact, a NaN and a deliberately wrong element must each fail the comparison,
whichever side they are on, and nothing may pass by not being compared.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from validation.blackwell import capture
from validation.blackwell.e2e_checks import EXPECTED_CASES

from tests.support import REPO as ROOT

SUITE = ROOT / "validation" / "blackwell"

HEADER = ".version 9.5\n.target sm_100a\n.address_size 64\n"
# y[0] += x[0]: what a replay of the two-pointer cases really computes.
ADD = HEADER + """.visible .entry k(.param .u64 p0, .param .u64 p1) {
.reg .b64 %a, %b;
.reg .f32 %u, %v;
ld.param.u64 %a, [p0];
ld.param.u64 %b, [p1];
ld.global.f32 %u, [%a];
ld.global.f32 %v, [%b];
add.f32 %v, %v, %u;
st.global.f32 [%b], %v;
ret;
}
"""


def nothing(params: int) -> str:
    """A kernel of `params` pointers that stores nothing."""
    return HEADER + ".visible .entry k(" + ", ".join(f".param .u64 p{i}" for i in range(params)) + ") {\nret;\n}\n"


def load(name, path=SUITE / "compare.py"):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors(case: str):
    """(inputs, output name, output) of a small stand-in for the named case."""
    rng = np.random.default_rng(len(case))
    if case.startswith("cutlass") or case == "triton_matmul":
        a = rng.integers(-2, 3, (4, 8)).astype("<f2")
        b = rng.integers(-2, 3, (8, 6) if case == "triton_matmul" else (6, 8)).astype("<f2")
        product = a.astype("<f4") @ (b if case == "triton_matmul" else b.T).astype("<f4") + np.float32(0.5)
        if case == "triton_matmul":
            return {"a": a, "b": b, "c": np.zeros_like(product)}, "c", product
        return {"a": a, "b": b, "c": np.zeros_like(product), "d": np.zeros_like(product)}, "d", product
    if case == "triton_vector_add":
        a, b = rng.standard_normal(16).astype("<f4"), rng.standard_normal(16).astype("<f4")
        return {"a": a, "b": b, "out": np.zeros(16, "<f4")}, "out", a + b
    if case.startswith("book_hgemm"):
        a, b = rng.standard_normal((4, 8)).astype("<f2"), rng.standard_normal((6, 8)).astype("<f2")
        d = (a.astype("<f4") @ b.astype("<f4").T).astype("<f2")
        return {"a": a, "b": b, "d": np.zeros_like(d)}, "d", d
    x, y = rng.random(1, dtype=np.float32) + 1, rng.random(1, dtype=np.float32) + 1
    return {"x": x, "y": y}, "y", x + y  # the saxpy cases, as ADD computes them


def save(root: Path, case: str, mode: str, *, output=None, ptx=None) -> Path:
    inputs, name, result = tensors(case)
    return capture.save_case(
        root, case, inputs=inputs, outputs={name: result if output is None else output},
        ptx=ptx or (ADD if len(inputs) == 2 else nothing(len(inputs))),
        launch={"entry": "k", "grid": [1, 1, 1], "block": [1, 1, 1], "cluster": [1, 1, 1], "smem": 0,
                "args": [{"buffer": key} for key in inputs]},
        comparison={"kind": "exact"}, mode=mode)


STDOUT = {
    "cutlass": "Relative error: 0\nExecution is successful.\n",
    "e2e/triton": "vector add: max abs error vs NumPy 0\nmatmul: max abs error vs NumPy 4.7e-06\n",
    "unchanged": "saxpy over 256 elements: matches NumPy\n",
    "e2e/book_hgemm": "".join(f"hgemm_v{n} 128x128x64: matches NumPy (max error 0.0156, 1.0 s)\n" for n in range(1, 10)),
}


def stdout_of(name: str) -> str:
    return STDOUT.get(name) or STDOUT["cutlass" if "cutlass" in name else "unchanged"]


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A GPU results directory and a simulator output directory holding the same 17 cases."""
    monkeypatch.setattr(capture, "provenance", lambda: {"test": True})
    gpu, sim = tmp_path / "gpu", tmp_path / "sim"
    rows = ["name\tstatus\tseconds\tcommand"]
    for name, cases in EXPECTED_CASES.items():
        rows.append(f"{name}\tok\t1.0\t-")
        (gpu / name).parent.mkdir(parents=True, exist_ok=True)
        (gpu / (name + ".stdout")).write_text(stdout_of(name))
        for case in cases:
            save(gpu / "captures" / name.removeprefix("e2e/"), case, "gpu")
            save(sim / "captures" / name.removeprefix("e2e/"), case, "sim")
    (gpu / "status.tsv").write_text("\n".join(rows) + "\n")
    return gpu, sim


def case_dir(root: Path, case: str) -> Path:
    name = next(name for name, cases in EXPECTED_CASES.items() if case in cases)
    return root / "captures" / name.removeprefix("e2e/") / case


def rewrite(root: Path, case: str, change, *, which="expected", keep_hash=False):
    """Change the saved tensor of `case` in place. The manifest's checksum follows the new
    bytes unless keep_hash, so the capture stays a well-formed record of a different result."""
    directory = case_dir(root, case)
    document = json.loads((directory / "capture.json").read_text())
    name = next(key for key, item in document["buffers"].items() if "expected" in item)
    item = document["buffers"][name]
    path = directory / item[which]["file"]
    values = np.frombuffer(path.read_bytes(), dtype=item["dtype"]).reshape(item["shape"]).copy()
    change(values)
    path.write_bytes(values.tobytes())
    if not keep_hash:
        item[which]["sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        (directory / "capture.json").write_text(json.dumps(document))


def run(monkeypatch, capsys, gpu, *options, only=None):
    compare = load("compare_e2e_tensors")
    monkeypatch.setattr(compare, "PROBES", [])
    monkeypatch.setattr(compare, "validate_probe_archive", lambda *args: [])
    if only:
        monkeypatch.setattr(compare, "E2E", [only])
    monkeypatch.setattr(sys, "argv", ["compare.py", str(gpu), *map(str, options)])
    status = compare.main()
    return status, capsys.readouterr().out


def row(out: str, name: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(name + " "))


def test_matching_captures_pass_and_every_case_is_compared(tree, monkeypatch, capsys):
    gpu, sim = tree
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 0, out
    assert "informational" not in out and out.count("bit-identical [") == 17
    for name in EXPECTED_CASES:
        assert " bit-identical " in row(out, name)


def bump(steps):
    def change(values):
        flat = values.reshape(-1)
        for _ in range(steps):
            flat[-1] = np.nextafter(flat[-1], np.inf, dtype=values.dtype)
    return change


def scale(factor):
    def change(values):
        values.reshape(-1)[-1] += np.abs(values).max() * factor
    return change


@pytest.mark.parametrize("case,change,passes", [
    ("cutlass_tutorial_03", scale(4e-6), True), ("cutlass_tutorial_03", scale(3e-5), False),
    ("triton_matmul", scale(4e-6), True), ("triton_matmul", scale(3e-5), False),
    ("triton_vector_add", bump(1), False),                       # RT-07: identical
    ("book_hgemm_v7", bump(1), True), ("book_hgemm_v7", bump(2), False),   # RT-08: one unit in the last place
    ("cute_dsl_saxpy", bump(1), True), ("cute_dsl_saxpy", scale(1e-5), False),  # RT-09: 1e-6 relative
    ("tirx_saxpy", scale(1e-5), False),
])
@pytest.mark.parametrize("side", ["gpu", "sim"])
def test_one_wrong_element_fails_at_the_rows_tolerance(tree, monkeypatch, capsys, case, change, passes, side):
    gpu, sim = tree
    rewrite(gpu if side == "gpu" else sim, case, change)
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    name = next(name for name, cases in EXPECTED_CASES.items() if case in cases)
    if passes:
        assert status == 0 and " within tolerance " in row(out, name), out
    else:
        assert status == 1 and " DIFFERS " in row(out, name) and "outside RT-0" in out, out


@pytest.mark.parametrize("case", ["cutlass_tutorial_01", "triton_vector_add", "triton_matmul", "book_hgemm_v1",
                                  "tirx_saxpy", "cute_dsl_saxpy"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("sides", [("gpu",), ("sim",), ("gpu", "sim")])
def test_a_nan_or_infinity_fails_even_when_both_sides_have_it(tree, monkeypatch, capsys, case, value, sides):
    gpu, sim = tree
    def change(values):
        values.reshape(-1)[0] = value
    for side in sides:
        rewrite(gpu if side == "gpu" else sim, case, change)
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and "non-finite" in out, out


def test_a_corrupted_tensor_file_fails_its_checksum(tree, monkeypatch, capsys):
    gpu, sim = tree
    rewrite(gpu, "book_hgemm_v4", bump(1), keep_hash=True)  # one element, inside the tolerance, but not what was recorded
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and "integrity mismatch" in out and " verification failed " in row(out, "e2e/book_hgemm"), out


def test_a_corrupted_simulator_tensor_is_an_error_not_a_reason_to_replay(tree, monkeypatch, capsys):
    gpu, sim = tree
    rewrite(sim, "tirx_saxpy", bump(1), keep_hash=True)
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim)
    assert status == 1 and "integrity mismatch" in out, out


@pytest.mark.parametrize("victim", ["d.output.bin", "d.input.bin", "a.input.bin", "kernel.ptx", "capture.json", ""])
@pytest.mark.parametrize("side", ["gpu", "sim"])
def test_a_missing_artifact_fails(tree, monkeypatch, capsys, victim, side):
    gpu, sim = tree
    directory = case_dir(gpu if side == "gpu" else sim, "book_hgemm_v9")
    shutil.rmtree(directory) if not victim else (directory / victim).unlink()
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1, out
    assert any(word in row(out, "e2e/book_hgemm") for word in (" verification failed ", " NOT COMPARED ", " DIFFERS "))


def test_without_simulator_tensors_and_without_replay_nothing_passes(tree, monkeypatch, capsys):
    gpu, _ = tree
    status, out = run(monkeypatch, capsys, gpu, "--no-replay")
    assert status == 1 and out.count(" NOT COMPARED ") >= len(EXPECTED_CASES), out


def test_the_results_cannot_be_their_own_simulator_reference(tree, monkeypatch, capsys):
    gpu, _ = tree
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", gpu, "--no-replay")
    assert status == 1 and "very capture under test" in out, out


def test_a_gpu_capture_is_not_accepted_as_the_simulators(tree, tmp_path, monkeypatch, capsys):
    gpu, sim = tree
    directory = case_dir(sim, "triton_matmul")
    shutil.rmtree(directory)
    save(directory.parent, "triton_matmul", "gpu")
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and "not a simulator capture" in out, out


def test_require_hardware_rejects_simulator_results(tree, monkeypatch, capsys):
    gpu, sim = tree
    assert run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay", "--require-hardware")[0] == 0
    status, out = run(monkeypatch, capsys, sim.parent / "gpu", "--sim-captures", sim, "--require-hardware")
    assert status == 0
    shutil.copy(gpu / "status.tsv", sim / "status.tsv")
    for name in EXPECTED_CASES:
        (sim / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(gpu / (name + ".stdout"), sim / (name + ".stdout"))
    status, out = run(monkeypatch, capsys, sim, "--require-hardware")
    assert status == 1 and "not hardware" in out and " NOT HARDWARE " in row(out, "e2e/triton"), out


@pytest.mark.parametrize("status_word,verdict", [("exit 1", "crashed"), ("signal 11", "crashed"),
                                                 ("timeout", "timed out"), ("skipped (no PyTorch)", "not run"),
                                                 ("capture-failed", "crashed"), ("verification-failed", "crashed"),
                                                 (None, "not run")])
def test_a_run_that_did_not_finish_is_counted_whatever_its_tensors_say(tree, monkeypatch, capsys, status_word, verdict):
    gpu, sim = tree
    lines = (gpu / "status.tsv").read_text().splitlines()
    lines = [l for l in lines if not l.startswith("e2e/triton\t")]
    if status_word:
        lines.append(f"e2e/triton\t{status_word}\t1.0\t-")
    (gpu / "status.tsv").write_text("\n".join(lines) + "\n")
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and f" {verdict} " in row(out, "e2e/triton"), out


def test_a_program_that_failed_its_own_check_fails_with_good_tensors(tree, monkeypatch, capsys):
    gpu, sim = tree
    (gpu / "e2e/cutlass_tutorials.tutorial_02.stdout").write_text("Relative error: 0.5\nExecution is failed.\n")
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and " verification failed " in row(out, "e2e/cutlass_tutorials.tutorial_02"), out


def test_book_output_far_from_numpy_fails_even_when_both_sides_agree(tree, monkeypatch, capsys):
    gpu, sim = tree
    for side in (gpu, sim):
        rewrite(side, "book_hgemm_v2", scale(1.0))
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and "against NumPy" in out and " DIFFERS " in row(out, "e2e/book_hgemm"), out


# The replay path: the simulator runs the GPU's own PTX on the GPU's own buffers.

TIRX = "e2e/unchanged_scripts.tirx"


def test_without_simulator_captures_the_gpu_capture_is_replayed(tree, tmp_path, monkeypatch, capsys):
    gpu, _ = tree
    status, out = run(monkeypatch, capsys, gpu, "--replay-out", tmp_path / "replayed", only=TIRX)
    assert status == 0 and "simulator replay of the GPU's PTX" in out, out
    _, name, result = tensors("tirx_saxpy")
    replayed = tmp_path / "replayed" / "unchanged_scripts.tirx" / "tirx_saxpy" / "y.output.bin"
    assert replayed.read_bytes() == result.tobytes()  # the simulator really ran the kernel


def test_replay_catches_a_gpu_result_the_kernel_does_not_produce(tree, monkeypatch, capsys):
    gpu, _ = tree
    rewrite(gpu, "tirx_saxpy", scale(1e-3))
    status, out = run(monkeypatch, capsys, gpu, only=TIRX)
    assert status == 1 and " DIFFERS " in row(out, TIRX), out


@pytest.mark.parametrize("what", ["ptx", "inputs", "launch"])
def test_a_simulator_capture_of_another_experiment_is_not_used(tree, monkeypatch, capsys, what):
    """Different PTX, inputs or launch: the simulator's capture says nothing about the GPU's
    run, so the GPU's capture is replayed (RT-07), or the case is not compared at all."""
    gpu, sim = tree
    directory = case_dir(sim, "tirx_saxpy")
    document = json.loads((directory / "capture.json").read_text())
    if what == "ptx":
        text = (directory / "kernel.ptx").read_text() + "\n// another compiler\n"
        (directory / "kernel.ptx").write_text(text)
        document["ptx"].update(bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest())
        (directory / "capture.json").write_text(json.dumps(document))
    elif what == "inputs":
        rewrite(sim, "tirx_saxpy", bump(1), which="initial")
    else:
        document["launch"]["block"] = [2, 1, 1]
        (directory / "capture.json").write_text(json.dumps(document))
    # The simulator's stale output is far off; it must not be what the GPU is judged against.
    rewrite(sim, "tirx_saxpy", scale(1.0))
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, only=TIRX)
    assert status == 0 and "simulator replay of the GPU's PTX" in out, out
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay", only=TIRX)
    assert status == 1 and " NOT COMPARED " in row(out, TIRX), out


def test_a_replay_the_simulator_refuses_is_not_a_pass(tree, monkeypatch, capsys):
    gpu, _ = tree
    directory = case_dir(gpu, "tirx_saxpy")
    document = json.loads((directory / "capture.json").read_text())
    text = (directory / "kernel.ptx").read_text().replace("ld.global.f32 %u, [%a];", "ld.global.f32 %u, [%a+1048576];")
    (directory / "kernel.ptx").write_text(text)
    document["ptx"].update(bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest())
    (directory / "capture.json").write_text(json.dumps(document))
    status, out = run(monkeypatch, capsys, gpu, only=TIRX)
    assert status == 1 and " NOT COMPARED " in row(out, TIRX), out


def test_the_tolerances_are_the_matrix_rows_not_the_captures_own_policy(tree, monkeypatch, capsys):
    """A capture cannot loosen its own judgement by declaring a wide comparison policy."""
    gpu, sim = tree
    directory = case_dir(gpu, "triton_vector_add")
    document = json.loads((directory / "capture.json").read_text())
    document["buffers"]["out"]["comparison"] = {"kind": "allclose", "rtol": 1.0, "atol": 1.0}
    (directory / "capture.json").write_text(json.dumps(document))
    rewrite(gpu, "triton_vector_add", bump(1))
    status, out = run(monkeypatch, capsys, gpu, "--sim-captures", sim, "--no-replay")
    assert status == 1 and " DIFFERS " in row(out, "e2e/triton"), out


def test_every_expected_case_has_a_criterion_from_its_row():
    compare = load("compare_e2e_criteria")
    from e2e_compare import CRITERIA
    assert set(CRITERIA) == {case for cases in EXPECTED_CASES.values() for case in cases}
    for name, cases in EXPECTED_CASES.items():
        assert {CRITERIA[case].row for case in cases} == set(compare.E2E[name])
    assert (CRITERIA["cutlass_tutorial_04"].kind, CRITERIA["cutlass_tutorial_04"].bound) == ("scaled", 1e-5)
    assert CRITERIA["triton_vector_add"].kind == "identical"
    assert (CRITERIA["triton_matmul"].kind, CRITERIA["triton_matmul"].bound) == ("scaled", 1e-5)
    assert (CRITERIA["book_hgemm_v9"].kind, CRITERIA["book_hgemm_v9"].bound, CRITERIA["book_hgemm_v9"].dtype,
            CRITERIA["book_hgemm_v9"].numpy_check) == ("ulp", 1, "<f2", True)
    assert (CRITERIA["tirx_saxpy"].kind, CRITERIA["tirx_saxpy"].bound) == ("relative", 1e-6)


def test_ulp_distance_crosses_zero_and_signs():
    load("compare_e2e_ulp")
    from e2e_compare import CRITERIA, compare_tensors
    criterion = CRITERIA["book_hgemm_v1"]
    a = np.array([-1, -0.0, 0.0, 1, 65504], "<f2")
    up = np.nextafter(a[:4], np.float16(np.inf))
    assert compare_tensors(np.append(up, a[4]).astype("<f2"), a, criterion)[0] is None
    assert compare_tensors(np.array([-0.0], "<f2"), np.array([0.0], "<f2"), criterion)[0] is None
    two = np.nextafter(up, np.float16(np.inf))
    assert compare_tensors(np.append(two, a[4]).astype("<f2"), a, criterion)[0]
    assert compare_tensors(a[:2], a[:3], criterion)[0]  # another shape
    assert compare_tensors(a[:0], a[:0], criterion)[0]  # nothing to compare is not a pass


# e2e/triton_gpu.py asserts RT-07's criterion itself; PyTorch and the GPU are stood in for.

class FakeTensor:
    def __init__(self, array):
        self.array = array

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeKernel:
    def __init__(self, compute):
        self.compute = compute

    def __getitem__(self, grid):
        return lambda *args, **constants: self.compute(*args)


def triton_gpu_with(monkeypatch, add_error, matmul_error):
    """Load e2e/triton_gpu.py with kernels that are exact except for one element."""
    from types import ModuleType, SimpleNamespace
    torch = ModuleType("torch")
    torch.__version__, torch.float32 = "fake", "float32"
    torch.from_numpy = lambda array: FakeTensor(array.copy())
    torch.empty = lambda shape, dtype, device: FakeTensor(np.zeros(shape, np.float32))
    torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "fake", synchronize=lambda: None,
                                 get_device_capability=lambda i: (10, 0), get_arch_list=lambda: ["sm_100"])
    triton = ModuleType("triton")
    triton.__version__, triton.cdiv = "fake", lambda a, b: -(-a // b)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.delenv("WARPLENS_CAPTURE_DIR", raising=False)

    def add(a, b, out, n):
        out.array[:] = a.array + b.array
        out.array[5] = add_error(out.array[5])

    def matmul(a, b, c, *sizes):
        c.array[:] = a.array.astype(np.float32) @ b.array.astype(np.float32)
        c.array[9, 9] += np.float32(matmul_error * np.abs(c.array).max())

    example = SimpleNamespace(add_kernel=FakeKernel(add), matmul_kernel=FakeKernel(matmul))
    module = load("triton_gpu_under_test", SUITE / "e2e" / "triton_gpu.py")
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location",
                        lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda spec: example)
    return module


@pytest.mark.parametrize("add_error,matmul_error,passes", [
    (lambda x: x, 0.0, True),
    (lambda x: x, 4e-6, True),                                  # inside 1e-5 * max|C|
    (lambda x: x, 3e-5, False),                                 # outside it
    (lambda x: x, float("nan"), False),
    (lambda x: np.nextafter(x, np.float32(np.inf)), 0.0, False),  # the vector add must be identical
    (lambda x: np.float32(np.nan), 0.0, False),
])
def test_triton_gpu_asserts_the_rt07_tolerance(monkeypatch, capsys, add_error, matmul_error, passes):
    module = triton_gpu_with(monkeypatch, add_error, matmul_error)
    if passes:
        assert module.main() == 0
        assert "matmul: max abs error vs NumPy" in capsys.readouterr().out
    else:
        with pytest.raises(AssertionError):
            module.main()
        assert "matmul: max abs error" not in capsys.readouterr().out
