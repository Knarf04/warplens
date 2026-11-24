"""Fail-closed completion checks shared by collection and comparison.

These checks establish that each program finished its own numerical verification.
Tensor fixture validation/replay separately compares the complete recorded arrays.
"""
import math
import re


EXPECTED_CASES = {
    **{f"e2e/cutlass_tutorials.tutorial_0{n}": [f"cutlass_tutorial_0{n}"] for n in range(1, 5)},
    "e2e/triton": ["triton_vector_add", "triton_matmul"],
    "e2e/unchanged_scripts.tirx": ["tirx_saxpy"],
    "e2e/unchanged_scripts.cute_dsl": ["cute_dsl_saxpy"],
    "e2e/book_hgemm": [f"book_hgemm_v{n}" for n in range(1, 10)],
}


def completion_errors(name: str, status: str | None, stdout: str) -> list[str]:
    if status != "ok":
        return [status or "missing status"]
    if re.search(r"(?:^|\n)\s*(?:FAILED\b|SKIPPED:|LAUNCH FAILED:|HOST ERROR:|KERNEL FAILED:)", stdout) or re.search(
            r"Execution is (?:not successful|unsuccessful|failed)", stdout, re.I):
        return ["program reported failure or skipped execution"]
    if name.startswith("e2e/cutlass_tutorials."):
        errors = re.findall(r"Relative error:\s*(\S+)", stdout)
        if "Execution is successful." not in stdout or len(errors) != 1:
            return ["missing CUTLASS numerical success result"]
        try:
            finite = math.isfinite(float(errors[0])) and float(errors[0]) >= 0
        except ValueError:
            finite = False
        return [] if finite else ["nonfinite or invalid CUTLASS relative error"]
    if name == "e2e/triton":
        for label in ("vector add", "matmul"):
            matches = re.findall(rf"^{label}: max abs error vs NumPy ([^;\s]+)", stdout, re.M)
            try:
                good = len(matches) == 1 and math.isfinite(float(matches[0])) and float(matches[0]) >= 0
            except ValueError:
                good = False
            if not good:
                return [f"missing or nonfinite Triton {label} result"]
        return []
    if name.startswith("e2e/unchanged_scripts."):
        return [] if re.search(r"^saxpy over \d+ elements(?: on compute capability \d+\.\d+)?: matches NumPy$", stdout, re.M) else ["missing saxpy verification"]
    if name == "e2e/book_hgemm":
        found = re.findall(r"^hgemm_v([1-9]) \d+x\d+x\d+: matches NumPy \(max error ([^,\s)]+)", stdout, re.M)
        if len(found) != 9 or {v for v, _ in found} != set("123456789"):
            return ["missing successful verification for one or more of hgemm_v1..v9"]
        try:
            finite = all(math.isfinite(float(error)) and float(error) >= 0 for _, error in found)
        except ValueError:
            finite = False
        return [] if finite else ["nonfinite book numerical error"]
    return [f"no completion contract for {name}"]
