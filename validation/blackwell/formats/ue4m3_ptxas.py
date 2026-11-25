#!/usr/bin/env python3
"""Does this toolkit have a conversion to or from UE4M3? Ask ptxas (matrix row NUM-05).
Needs no GPU.

    python validation/blackwell/formats/ue4m3_ptxas.py [--ptxas PROGRAM]

NUM-05 claims how the simulator rounds to UE4M3 (E4M3 without the sign bit, the scale format
of NVFP4). The row can be tested bit for bit only "where a GPU conversion exists". CUDA's
headers have none, and the PTX manual lists no `ue4m3` type for `cvt`; this script asks the
assembler itself, so that the answer is recorded for the toolkit that is installed rather
than taken from a document.

For every small format a one-instruction kernel is written as PTX for each direction
(two .f32 to a packed pair, and a packed pair to .f16x2) and given to `ptxas --gpu-name
sm_100a`. The formats the simulator already implements are the controls: they must be
accepted, or the test kernels are wrong and a rejection of UE4M3 would mean nothing.

Output: one line per format and direction, then a verdict line:

    NUM-05: no UE4M3 conversion in this toolkit; the row stays open
    NUM-05: UE4M3 CONVERSION ACCEPTED ...    a probe can now be written (formats/fp8_codes)

formats/ue4m3_ptxas.txt is this script's output with ptxas 13.5 on the development machine.
env/remote/run.sh runs it again on the rented machine with the toolkit installed there
(validation/blackwell/rows/run_rows.py). Exit status: 0 no UE4M3 conversion, 1 one was
accepted (news, not an error), 2 a control was rejected or ptxas is missing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# name -> (rounding and modifiers of the narrowing form, is it a control)
# Controls are spelled as nvcc emits them (tests/numerics/test_small_format_cvt.py has the list).
NARROW = {
    "e4m3x2": ("rn.satfinite", True),
    "e5m2x2": ("rn.satfinite", True),
    "e2m1x2": ("rn.satfinite", True),
    "ue8m0x2": ("rz.satfinite", True),
    # UE4M3 under every spelling the other formats suggest.
    "ue4m3x2": ("rn.satfinite", False),
    "ue4m3": ("rn.satfinite", False),
    "ue4m3x4": ("rn.satfinite", False),
}
# The widening form exists for these controls as `cvt.rn.<wide>.<format>`.
WIDE = {"e4m3x2": "f16x2", "e5m2x2": "f16x2", "e2m1x2": "f16x2", "ue8m0x2": "bf16x2",
        "ue4m3x2": "f16x2", "ue4m3": "f16x2", "ue4m3x4": "f16x2"}


def kernel(instruction: str) -> str:
    return f""".version 8.8
.target sm_100a
.address_size 64

.visible .entry k()
{{
    .reg .b8 %n8;
    .reg .b16 %n;
    .reg .b32 %w;
    .reg .f32 %a, %b;
    mov.f32 %a, 0f3F800000;
    mov.f32 %b, 0f40000000;
    mov.b16 %n, 0;
    {instruction}
    ret;
}}
"""


def cases():
    for name, (modifiers, control) in NARROW.items():
        target = "%n8" if name == "e2m1x2" else "%n"
        yield name, control, "from .f32", f"cvt.{modifiers}.{name}.f32 {target}, %a, %b;"
        source = "%n8" if name == "e2m1x2" else "%n"
        yield name, control, f"to .{WIDE[name]}", f"cvt.rn.{WIDE[name]}.{name} %w, {source};"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ptxas", default="ptxas")
    args = parser.parse_args()
    if not shutil.which(args.ptxas):
        print(f"error: {args.ptxas} not found (it is next to nvcc in the toolkit)", file=sys.stderr)
        return 2
    version = subprocess.run([args.ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    print("# " + ("; ".join(line.strip() for line in version[:3]) if version else "ptxas, version unknown"))
    broken_controls, accepted_ue4m3 = [], []
    with tempfile.TemporaryDirectory() as scratch:
        for index, (name, control, direction, instruction) in enumerate(cases()):
            source = Path(scratch) / f"case{index}.ptx"
            source.write_text(kernel(instruction))
            done = subprocess.run([args.ptxas, "--gpu-name", "sm_100a", str(source), "-o", "/dev/null"],
                                  capture_output=True, text=True)
            accepted = done.returncode == 0
            line = f"{name:8} {direction:10} {instruction:44} ptxas {'accepts' if accepted else 'rejects'}"
            if not accepted:  # ptxas's own reason, without the temporary file's name
                why = ((done.stderr + done.stdout).strip().splitlines() or ["no message"])[0]
                line += ": " + why.split("error", 1)[-1].lstrip(" :")[:110]
            if control and not accepted:
                broken_controls.append(instruction)
                line += "  <-- CONTROL REJECTED"
            if not control and accepted:
                accepted_ue4m3.append(instruction)
                line += "  <-- UE4M3 CONVERSION EXISTS"
            print(line)
    if broken_controls:
        print(f"NUM-05: UNDECIDED: ptxas rejected {len(broken_controls)} control conversion(s) that every sm_100a "
              "toolkit should accept, so the rejections of UE4M3 above prove nothing")
        return 2
    if accepted_ue4m3:
        print("NUM-05: UE4M3 CONVERSION ACCEPTED by this ptxas: " + " | ".join(accepted_ue4m3)
              + " -- the row can be tested: add the sweep of NUM-01 for it to formats/small_formats.cu")
        return 1
    print("NUM-05: no UE4M3 conversion in this toolkit; the row stays open (docs/validation-matrix.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
