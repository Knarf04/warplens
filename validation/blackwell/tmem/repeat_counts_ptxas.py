#!/usr/bin/env python3
"""Which (shape, repeat count) pairs of tcgen05.ld and tcgen05.st exist: ask ptxas, and
optionally the simulator (matrix row TM-06). Needs no GPU.

    python validation/blackwell/tmem/repeat_counts_ptxas.py [--ptxas PROGRAM] [--sim]

For every shape and every count in 1, 2, 3, 4, ..., 256 (powers of two, and 3 as a count
that is not one) a one-instruction kernel is written as PTX and given to `ptxas
--gpu-name sm_100a`. One line per pair says whether ptxas accepted it. With --sim the same
PTX is also launched in warplens and a second column says whether the simulator ran it.

tmem/repeat_counts_ptxas.txt is this script's output with ptxas 13.5 and --sim on the
development machine. The claim is the accepted set: 32x32b and 16x64b up to x128, 16x128b
up to x64, 16x256b up to x32, powers of two only. tmem/repeat_counts.cu holds the accepted
pairs, so the probe build proves that half on any toolkit; this script shows the other
half, that nothing beyond them is accepted. Exit status 1 if ptxas and the claim disagree.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHAPES = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}  # registers per count
COUNTS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
MAX_REGISTERS = 128


def claimed(shape: str, count: int) -> bool:
    return count & (count - 1) == 0 and SHAPES[shape] * count <= MAX_REGISTERS


def ptx(op: str, shape: str, count: int) -> str:
    registers = ", ".join(f"%v{i}" for i in range(SHAPES[shape] * count))
    transfer = (f"tcgen05.ld.sync.aligned.{shape}.x{count}.b32 {{{registers}}}, [%base];\n"
                "    tcgen05.wait::ld.sync.aligned;" if op == "ld" else
                f"tcgen05.st.sync.aligned.{shape}.x{count}.b32 [%base], {{{registers}}};\n"
                "    tcgen05.wait::st.sync.aligned;")
    moves = "\n".join(f"    mov.b32 %v{i}, {i};" for i in range(SHAPES[shape] * count)) if op == "st" else ""
    return f""".version 8.8
.target sm_100a
.address_size 64

.visible .entry k()
{{
    .shared .align 4 .b8 slot[4];
    .reg .b32 %base, {registers};
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 512;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ld.shared.u32 %base, [slot];
{moves}
    {transfer}
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, 512;
    ret;
}}
"""


def simulator_runs(path: Path) -> tuple[bool, str]:
    import warplens
    try:
        warplens.Device("B200").launch(path, "k", grid=1, block=32, args=[])
        return True, ""
    except Exception as error:  # the simulator's refusal, whatever its type
        return False, str(error).splitlines()[0][:100]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--sim", action="store_true", help="also launch each kernel in warplens")
    args = parser.parse_args()
    if not shutil.which(args.ptxas):
        sys.exit(f"error: {args.ptxas} not found (it is next to nvcc in the toolkit)")
    version = subprocess.run([args.ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    print("# " + ("; ".join(line.strip() for line in version[:3]) if version else "ptxas, version unknown"))
    wrong = 0
    with tempfile.TemporaryDirectory() as scratch:
        for op in ("ld", "st"):
            for shape in SHAPES:
                for count in COUNTS:
                    source = Path(scratch) / f"{op}_{shape}_x{count}.ptx"
                    source.write_text(ptx(op, shape, count))
                    done = subprocess.run([args.ptxas, "--gpu-name", "sm_100a", str(source), "-o", "/dev/null"],
                                          capture_output=True, text=True)
                    accepted = done.returncode == 0
                    line = (f"tcgen05.{op} {shape}.x{count:<3} ({SHAPES[shape] * count:3} registers): "
                            f"ptxas {'accepts' if accepted else 'rejects'}")
                    if args.sim:
                        ran, why = simulator_runs(source)
                        line += f"; warplens {'runs it' if ran else 'refuses it'}"
                        if ran != accepted:
                            line += f"  <-- DISAGREE{': ' + why if why else ''}"
                    if accepted != claimed(shape, count):
                        wrong += 1
                        line += "  <-- NOT AS CLAIMED"
                    print(line)
    print(f"# pairs where ptxas and the claimed set disagree: {wrong}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
