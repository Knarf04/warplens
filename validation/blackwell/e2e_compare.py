"""Simulator tensors against GPU tensors: validation matrix rows RT-06 to RT-09.

compare.py calls `compare_run` for each end-to-end run. The GPU side is a capture that
run_e2e.py made (docs/hardware-capture.md). The simulator side is, per case, either

  * the capture of the same case from `run_e2e.py --mode sim` (--sim-captures), used only
    when it ran the same PTX with the same launch shape on bit-identical initial buffers, or
  * a replay of the GPU capture in the simulator: the GPU's PTX, buffers and launch. This is
    what the matrix asks for when the two environments compiled different PTX (RT-07), and
    it is what happens when no simulator capture is given.

The tolerances are the matrix's and are written here, per case. The `comparison` a capture
carries is not consulted: a fixture does not get to state how strictly it is judged. Every
element is compared; a NaN or infinity on either side fails; a case that cannot be compared
(no tensor, no simulator result, a refused replay) is an error, never a pass.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture import load_case, replay_case, _read  # noqa: E402  (compare.py puts this directory on sys.path)


@dataclass(frozen=True)
class Criterion:
    row: str            # the matrix row
    output: str         # the one buffer the kernel writes
    dtype: str          # its element type, as captured
    kind: str           # "identical" | "scaled" | "relative" | "ulp"
    bound: float = 0.0  # scaled: bound * max|sim|; relative: bound * |sim| per element; ulp: units
    numpy_check: bool = False  # RT-08: also inside the book's rtol=2e-2, atol=1e-2 against NumPy

    def describe(self) -> str:
        return {"identical": "bit-identical",
                "scaled": f"every element within {self.bound:g} * max|sim|",
                "relative": f"every element within {self.bound:g} relative",
                "ulp": f"every element within {self.bound:g} unit(s) in the last place"}[self.kind] + \
            (", and within rtol=2e-2, atol=1e-2 of NumPy" if self.numpy_check else "")


BOOK_RTOL, BOOK_ATOL = 2e-2, 1e-2
CRITERIA: dict[str, Criterion] = {
    # RT-06: every element within 1e-5 * max(abs(C)) of the simulator's.
    **{f"cutlass_tutorial_0{n}": Criterion("RT-06", "d", "<f4", "scaled", 1e-5) for n in range(1, 5)},
    # RT-07: vector add identical; matmul within 1e-5 * max(abs(C)).
    "triton_vector_add": Criterion("RT-07", "out", "<f4", "identical"),
    "triton_matmul": Criterion("RT-07", "c", "<f4", "scaled", 1e-5),
    # RT-08: half precision within 1 unit in the last place, and inside the book's own tolerance.
    **{f"book_hgemm_v{n}": Criterion("RT-08", "d", "<f2", "ulp", 1, numpy_check=True) for n in range(1, 10)},
    # RT-09: floating-point outputs within 1e-6 relative.
    "tirx_saxpy": Criterion("RT-09", "y", "<f4", "relative", 1e-6),
    "cute_dsl_saxpy": Criterion("RT-09", "y", "<f4", "relative", 1e-6),
}


def ordered(values: np.ndarray) -> np.ndarray:
    """IEEE bit patterns onto a line of integers on which neighbouring floats differ by one
    (the two zeros coincide)."""
    bits = values.dtype.itemsize * 8
    words = values.view(f"<u{values.dtype.itemsize}").astype(np.int64)
    magnitude = words & ((1 << (bits - 1)) - 1)
    return np.where(words >> (bits - 1), -magnitude, magnitude)


def compare_tensors(gpu: np.ndarray, sim: np.ndarray, criterion: Criterion) -> tuple[str | None, str]:
    """(error or None, one-line summary). `gpu` and `sim` have the same dtype and shape."""
    if gpu.dtype != sim.dtype or gpu.shape != sim.shape:
        return f"shape or type differs: GPU {gpu.dtype}{list(gpu.shape)}, simulator {sim.dtype}{list(sim.shape)}", ""
    if gpu.size == 0:
        return "empty output tensor", ""
    for side, values in (("GPU", gpu), ("simulator", sim)):
        finite = np.isfinite(values)
        if not finite.all():
            index = tuple(int(i) for i in np.argwhere(~finite)[0])
            return (f"{int((~finite).sum())} non-finite value(s) in the {side} output, first at {index} "
                    f"({values[index]})"), ""
    a, b = gpu.astype(np.float64), sim.astype(np.float64)
    difference = np.abs(a - b)
    if criterion.kind == "identical":
        wrong = gpu.view(f"<u{gpu.dtype.itemsize}") != sim.view(f"<u{sim.dtype.itemsize}")
        limit_text = "must be bit-identical"
    elif criterion.kind == "scaled":
        limit = criterion.bound * float(np.abs(b).max())
        wrong = difference > limit
        limit_text = f"limit {limit:.3g} = {criterion.bound:g} * max|sim|"
    elif criterion.kind == "relative":
        wrong = difference > criterion.bound * np.abs(b)
        limit_text = f"limit {criterion.bound:g} * |sim|"
    elif criterion.kind == "ulp":
        distance = np.abs(ordered(gpu) - ordered(sim))
        wrong = distance > criterion.bound
        limit_text = f"at most {int(distance.max())} ulp apart, limit {criterion.bound:g}"
    else:
        raise ValueError(f"unknown criterion {criterion.kind}")
    identical = gpu.tobytes() == sim.tobytes()
    summary = "bit-identical" if identical else f"max |GPU - sim| {float(difference.max()):.3g}, {limit_text}"
    if wrong.any():
        index = tuple(int(i) for i in np.argwhere(wrong)[0])
        return (f"{int(wrong.sum())} of {gpu.size} element(s) outside {criterion.row}'s tolerance "
                f"({criterion.describe()}); first at {index}: GPU {gpu[index]!r}, simulator {sim[index]!r}; "
                f"{summary}"), summary
    return None, summary


def book_reference_error(buffers: dict[str, np.ndarray], output: np.ndarray, side: str) -> str | None:
    """RT-08's second half, from the captured tensors: the book's own tolerance against NumPy."""
    if not {"a", "b"} <= buffers.keys():
        return "the capture has no input matrices a and b to form the NumPy reference from"
    a, b = buffers["a"], buffers["b"]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or output.shape != (a.shape[0], b.shape[0]):
        return f"input shapes {a.shape}, {b.shape} do not form the output {output.shape}"
    reference = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float16).astype(np.float64)
    wrong = ~(np.abs(output.astype(np.float64) - reference) <= BOOK_ATOL + BOOK_RTOL * np.abs(reference))
    if wrong.any():
        index = tuple(int(i) for i in np.argwhere(wrong)[0])
        return (f"{int(wrong.sum())} element(s) of the {side} output outside the book's rtol={BOOK_RTOL:g}, "
                f"atol={BOOK_ATOL:g} against NumPy; first at {index}: {output[index]!r}, NumPy {reference[index]!r}")
    return None


def arrays(directory: Path, case: dict, which: str) -> dict[str, np.ndarray]:
    return {name: np.frombuffer(_read(directory, item[which]), dtype=item["dtype"]).reshape(item["shape"])
            for name, item in case["buffers"].items() if which in item}


def launch_shape(case: dict) -> dict:
    launch = case["launch"]
    return {key: launch[key] for key in ("entry", "grid", "block", "cluster", "smem")}


def simulator_capture_mismatch(gpu_dir: Path, gpu: dict, sim_dir: Path) -> tuple[dict | None, str]:
    """(the simulator's capture of this case, "") if it is the same experiment as the GPU's,
    else (None, why it is not)."""
    if not (sim_dir / "capture.json").is_file():
        return None, f"no simulator capture at {sim_dir}"
    if sim_dir.resolve() == gpu_dir.resolve():
        return None, "the simulator capture given is the very capture under test"
    sim = load_case(sim_dir)  # raises on a missing or altered artifact: the caller reports it
    if sim["mode"] != "sim":
        raise ValueError(f"{sim_dir} is labelled '{sim['mode']}', not a simulator capture")
    if sim["case"] != gpu["case"]:
        raise ValueError(f"{sim_dir} holds case {sim['case']}, not {gpu['case']}")
    if sim["ptx"]["sha256"] != gpu["ptx"]["sha256"]:
        return None, "the PTX differs between the two environments"
    if launch_shape(sim) != launch_shape(gpu):
        return None, f"the launch differs: GPU {launch_shape(gpu)}, simulator {launch_shape(sim)}"
    a = {name: (item["dtype"], item["shape"], item["initial"]["sha256"]) for name, item in gpu["buffers"].items()}
    b = {name: (item["dtype"], item["shape"], item["initial"]["sha256"]) for name, item in sim["buffers"].items()}
    if a != b:
        names = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
        return None, f"the initial buffers differ ({', '.join(names)})"
    return sim, ""


def compare_case(gpu_dir: Path, sim_dir: Path | None, *, replay: bool, replay_out: Path | None,
                 require_hardware: bool = False) -> tuple[list[str], str]:
    """Compare one captured case. ([errors], summary); an empty list is a pass."""
    gpu_dir = Path(gpu_dir)
    gpu = load_case(gpu_dir)
    name = gpu["case"]
    criterion = CRITERIA.get(name)
    if criterion is None:
        return [f"{name}: no comparison is defined for this case"], ""
    if require_hardware and gpu["mode"] != "gpu":
        return [f"{name}: the results are a '{gpu['mode']}' capture, not hardware"], ""
    outputs = arrays(gpu_dir, gpu, "expected")
    if set(outputs) != {criterion.output} or gpu["buffers"][criterion.output]["dtype"] != criterion.dtype:
        return [f"{name}: expected one {criterion.dtype} output '{criterion.output}', the capture has "
                f"{ {k: gpu['buffers'][k]['dtype'] for k in outputs} }"], ""
    got = outputs[criterion.output]

    source, why = "", ""
    sim_output = None
    if sim_dir is not None:
        sim, why = simulator_capture_mismatch(gpu_dir, gpu, Path(sim_dir))
        if sim is not None:
            sim_outputs = arrays(Path(sim_dir), sim, "expected")
            if set(sim_outputs) != {criterion.output}:
                return [f"{name}: the simulator capture has outputs {sorted(sim_outputs)}, not '{criterion.output}'"], ""
            sim_output, source = sim_outputs[criterion.output], "simulator capture, same PTX and inputs"
    if sim_output is None:
        if not replay:
            return [f"{name}: NOT COMPARED: {why or 'no simulator capture given'}, and replay is off"], ""
        scratch = None
        if replay_out is None:
            scratch = tempfile.TemporaryDirectory(prefix="warplens-replay-")
            replay_out = Path(scratch.name)
        target = Path(replay_out) / name
        try:
            replay_case(gpu_dir, target)  # its own verdict uses the capture's policy; ours is below
            item = gpu["buffers"][criterion.output]
            raw = (target / item["expected"]["file"]).read_bytes()
            if len(raw) != got.nbytes:
                return [f"{name}: the replay wrote {len(raw)} bytes for '{criterion.output}', expected {got.nbytes}"], ""
            sim_output = np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"]).copy()
        except Exception as error:  # a kernel the simulator refuses, a fault, a step limit
            return [f"{name}: NOT COMPARED: the simulator could not replay the GPU's capture: "
                    f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"], ""
        finally:
            if scratch:
                scratch.cleanup()
        source = "simulator replay of the GPU's PTX" + (f" ({why})" if why else "")

    errors = []
    error, summary = compare_tensors(got, sim_output, criterion)
    if error:
        errors.append(f"{name}: {error}")
    if criterion.numpy_check:
        inputs = arrays(gpu_dir, gpu, "initial")
        for side, values in (("GPU", got), ("simulator", sim_output)):
            if np.isfinite(values).all():
                error = book_reference_error(inputs, values, side)
                if error:
                    errors.append(f"{name}: {error}")
    digest = hashlib.sha256(got.tobytes()).hexdigest()[:12]
    note = "" if gpu["mode"] == "gpu" else f"; results labelled '{gpu['mode']}', not hardware"
    return errors, f"{name}: {summary or 'see error'} [{criterion.row}: {criterion.describe()}; {source}; " \
                   f"GPU output sha256 {digest}{note}]"


def compare_run(captures: Path, expected_cases: list[str], sim_captures: Path | None, *, replay: bool = True,
                replay_out: Path | None = None, require_hardware: bool = False) -> tuple[list[str], list[str]]:
    """Compare every expected case under `captures` (one end-to-end run's capture directory).
    ([errors], [one summary line per case])."""
    errors, lines = [], []
    for case in expected_cases:
        gpu_dir = Path(captures) / case
        sim_dir = Path(sim_captures) / case if sim_captures is not None else None
        try:
            if not (gpu_dir / "capture.json").is_file():
                raise ValueError(f"no capture at {gpu_dir}")
            found, summary = compare_case(gpu_dir, sim_dir, replay=replay, replay_out=replay_out,
                                          require_hardware=require_hardware)
        except (ValueError, KeyError, TypeError, OSError) as error:
            found, summary = [f"{case}: {type(error).__name__}: {error}"], ""
        errors += found
        if summary:
            lines.append(summary)
    return errors, lines
