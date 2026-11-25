#!/usr/bin/env python3
"""Replay a captured launch (matrix row RT-11) in the simulator and hold it against the GPU.

    replay_launch.py DIR [--launch N] [--rewrite-out-of-scope]

DIR is what e2e/cute_dsl_tma.py leaves in `rows/e2e/cute_dsl_tma/capture/` of a rental's
archive: the kernel's PTX (dsl_dump/*.ptx), launches.json with every parameter's bytes and
the addresses found in them, and mem/*.bin, the device memory before the kernel and after
it. No GPU, no CUTLASS, no PyTorch.

What it does: gives each allocation of the capture to the simulator at an address of the
simulator's choosing, moves the pointers in the parameters accordingly (those inside a
tensor map too: the map is in the driver's layout, which warplens/host/tensor_map.py reads),
runs the kernel, and compares every byte of every allocation with the `after` dump.

The kernel of CUTLASS's fp16_gemm_0.py uses two instruction forms that are outside the v1
instruction scope (PLAN.md, section 3.5): `tcgen05.mma` with `.collector::a::discard`, and
`cvt.rn.f16x2.f32`. Without the option the replay stops at the first, as any run of that
kernel does. `--rewrite-out-of-scope` is an EXPERIMENT, not support: it edits the PTX text
before the simulator sees it, dropping the collector modifier (`discard` is what an MMA
without the modifier does) and writing each packed conversion as two `cvt.rn.f16.f32` with
the first source in the high half. With it the capture of 2026-11-25 replays with no byte
different from the B200's (docs/b200-rental-1.md).

Exit status: 0 identical, 1 different, 2 the simulator refused the kernel.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import warplens  # noqa: E402

PACKED = re.compile(r"cvt\.rn\.f16x2\.f32\s+(%\w+),\s*(%\w+),\s*(%\w+);")


def rewrite_out_of_scope(text: str) -> tuple[str, dict]:
    """The experiment of the module's docstring. Returns the text and what was changed."""
    counts = {"collector::a::discard": text.count(".collector::a::discard")}
    text = text.replace(".collector::a::discard", "")

    def unpack(match):
        d, high, low = match.groups()
        return "\n\t".join([f"cvt.rn.f16.f32 %wl_h, {high};", f"cvt.rn.f16.f32 %wl_l, {low};",
                            "cvt.u32.u16 %wl_a, %wl_h;", "cvt.u32.u16 %wl_b, %wl_l;",
                            "shl.b32 %wl_a, %wl_a, 16;", f"or.b32 {d}, %wl_a, %wl_b;"])

    text, counts["cvt.rn.f16x2.f32"] = PACKED.subn(unpack, text)
    if counts["cvt.rn.f16x2.f32"]:
        # One declaration per kernel that has a body.
        text = re.sub(r"(\.entry[^{;]*\{)", r"\1\n\t.reg .b16 %wl_h, %wl_l;\n\t.reg .b32 %wl_a, %wl_b;", text)
    return text, counts


def replay(directory: Path, index: int = 0, rewrite: bool = False):
    """Returns (stats, per-allocation list of (base, size, bytes different from the GPU's
    `after`, bytes the GPU changed), what a rewrite changed)."""
    launches = json.loads((directory / "launches.json").read_text())
    launches = launches["launches"] if isinstance(launches, dict) else launches
    launch = next(entry for entry in launches if entry["index"] == index)
    memory = launch["memory"]
    device = warplens.Device()
    arrays, dumps = {}, {}
    for allocation in memory["allocations"]:
        before = np.fromfile(directory / allocation["before"]["file"], dtype=np.uint8)
        after = np.fromfile(directory / allocation["after"]["file"], dtype=np.uint8)
        arrays[allocation["base"]] = device.from_numpy(before.copy())
        dumps[allocation["base"]] = (before, after)
    arguments = []
    for parameter in launch["parameters"]:
        raw = bytearray.fromhex(parameter["hex"])
        for pointer in memory["pointers"]:
            if pointer["parameter"] == parameter["index"]:
                (value,) = struct.unpack_from("<Q", raw, pointer["offset"])
                if value != pointer["value"]:
                    raise SystemExit(f"parameter {parameter['index']}: the capture's pointer is not in its bytes")
                moved = arrays[pointer["allocation"]].addr + pointer["offset_in_allocation"]
                struct.pack_into("<Q", raw, pointer["offset"], moved)
        arguments.append(bytes(raw))
    (source,) = sorted((directory / "dsl_dump").glob("*.ptx"))
    text, changed = source.read_text(), {}
    if rewrite:
        text, changed = rewrite_out_of_scope(text)
    stats = device.launch(warplens.parse_ptx(text, name=source.name), launch["kernel"],
                          grid=tuple(launch["grid"]), block=tuple(launch["block"]),
                          smem=launch["shared_memory_bytes"], args=arguments, max_steps=10**9)
    results = []
    for base, (before, after) in dumps.items():
        got = arrays[base].numpy().view(np.uint8).ravel()
        results.append((base, len(after), int((got != after).sum()), int((before != after).sum())))
    return stats, results, changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--launch", type=int, default=0, help="index in launches.json (default 0)")
    parser.add_argument("--rewrite-out-of-scope", action="store_true",
                        help="an experiment: see the module's docstring")
    args = parser.parse_args(argv)
    try:
        stats, results, changed = replay(args.directory, args.launch, args.rewrite_out_of_scope)
    except warplens.WarplensError as error:
        print(f"the simulator refused the kernel: {type(error).__name__}: {error}")
        return 2
    if changed:
        print("EXPERIMENT, PTX text rewritten before the run: "
              + ", ".join(f"{count} x {form}" for form, count in changed.items()))
    print(f"ran: {stats.ctas} CTA(s), {stats.instructions} warp-instructions, {len(stats.findings)} finding(s)")
    different = 0
    for base, size, wrong, written in results:
        print(f"allocation {base:#x}, {size} bytes: the GPU changed {written}; {wrong} differ from the GPU's after the kernel")
        different += wrong
    print("replay: IDENTICAL to the GPU" if not different else f"replay: {different} byte(s) DIFFER from the GPU")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
