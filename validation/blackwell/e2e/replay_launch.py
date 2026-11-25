#!/usr/bin/env python3
"""Replay a captured launch (matrix row RT-11) in the simulator and hold it against the GPU.

    replay_launch.py DIR [--launch N]

DIR is what e2e/cute_dsl_tma.py leaves in `rows/e2e/cute_dsl_tma/capture/` of a rental's
archive: the kernel's PTX (dsl_dump/*.ptx), launches.json with every parameter's bytes and
the addresses found in them, and mem/*.bin, the device memory before the kernel and after
it. No GPU, no CUTLASS, no PyTorch.

What it does: gives each allocation of the capture to the simulator at an address of the
simulator's choosing, moves the pointers in the parameters accordingly (those inside a
tensor map too: the map is in the driver's layout, which warplens/host/tensor_map.py reads),
runs the kernel as it was compiled, and compares every byte of every allocation with the
`after` dump.

The capture of 2026-11-25 (CUTLASS's fp16_gemm_0.py at --mnk 256,512,512 on a B200) replays
with no byte different from the GPU's (docs/b200-rental-1.md).
tests/test_rt11_replay.py runs this on a capture named by WARPLENS_RT11_CAPTURE.

Exit status: 0 identical, 1 different, 2 the simulator refused the kernel.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import warplens  # noqa: E402

def replay(directory: Path, index: int = 0):
    """Returns (stats, per-allocation list of (base, size, bytes different from the GPU's
    `after`, bytes the GPU changed))."""
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
    stats = device.launch(warplens.parse_ptx(source.read_text(), name=source.name), launch["kernel"],
                          grid=tuple(launch["grid"]), block=tuple(launch["block"]),
                          smem=launch["shared_memory_bytes"], args=arguments, max_steps=10**9)
    results = []
    for base, (before, after) in dumps.items():
        got = arrays[base].numpy().view(np.uint8).ravel()
        results.append((base, len(after), int((got != after).sum()), int((before != after).sum())))
    return stats, results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--launch", type=int, default=0, help="index in launches.json (default 0)")
    args = parser.parse_args(argv)
    try:
        stats, results = replay(args.directory, args.launch)
    except warplens.WarplensError as error:
        print(f"the simulator refused the kernel: {type(error).__name__}: {error}")
        return 2
    print(f"ran: {stats.ctas} CTA(s), {stats.instructions} warp-instructions, {len(stats.findings)} finding(s)")
    different = 0
    for base, size, wrong, written in results:
        print(f"allocation {base:#x}, {size} bytes: the GPU changed {written}; {wrong} differ from the GPU's after the kernel")
        different += wrong
    print("replay: IDENTICAL to the GPU" if not different else f"replay: {different} byte(s) DIFFER from the GPU")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
