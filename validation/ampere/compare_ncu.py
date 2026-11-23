"""Compare Nsight Compute shared-memory counters with warplens's bank model.

    python compare_ncu.py ncu_transpose.csv <ctas>

Reads the CSV that `ncu --csv` printed for bin/transpose_hw, runs the same
three kernels with the same grid in the simulator, and prints both side by side.
Exits non-zero if any counter differs.
"""

import csv
import sys
from pathlib import Path

import numpy as np

import warplens

HERE = Path(__file__).resolve().parent
PTX = HERE.parent.parent / "examples" / "01_smem_transpose" / "transpose.ptx"
KERNELS = ["transpose_linear", "transpose_padded", "transpose_swizzled"]


def simulate(kernel: str, ctas: int) -> dict[str, int]:
    sim = warplens.Device()
    data = np.arange(ctas * 32 * 32, dtype=np.float32).reshape(ctas, 32, 32)
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(PTX, kernel, grid=ctas, block=(32, 32), args=[sim.from_numpy(data), out])
    assert np.array_equal(out.numpy(), data.transpose(0, 2, 1))
    by_op = {"ld": [0, 0], "st": [0, 0]}  # executions, conflicts
    for e in stats.smem.instructions.values():
        by_op[e.op][0] += e.executions
        by_op[e.op][1] += e.conflicts
    return {
        "smsp__inst_executed_op_shared_ld.sum": by_op["ld"][0],
        "smsp__inst_executed_op_shared_st.sum": by_op["st"][0],
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum": stats.smem.wavefronts,
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum": stats.smem.wavefronts - stats.smem.conflicts,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": stats.smem.conflicts,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": by_op["ld"][1],
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": by_op["st"][1],
    }


def read_ncu(path: Path) -> dict[str, dict[str, float]]:
    lines = path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        sys.exit(f"{path}: no ncu CSV table found. ncu output was:\n" + "\n".join(lines[:20]))
    out: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(lines[start:]):
        kernel = row["Kernel Name"].split("(")[0]
        value = row["Metric Value"].replace(",", "")
        try:
            out.setdefault(kernel, {})[row["Metric Name"]] = float(value)
        except ValueError:
            pass  # "n/a": this GPU or ncu version does not provide the metric
    return out


def main() -> int:
    path, ctas = Path(sys.argv[1]), int(sys.argv[2])
    hw = read_ncu(path)
    differ = 0
    for kernel in KERNELS:
        sim = simulate(kernel, ctas)
        print(f"\n{kernel}  (grid {ctas}, block 32x32)")
        print(f"  {'metric':<58}{'ncu':>10}{'warplens':>10}{'ncu - warplens':>16}")
        for metric, predicted in sim.items():
            measured = hw.get(kernel, {}).get(metric)
            if measured is None:
                shown, delta = "n/a", ""
            else:
                shown, delta = f"{measured:.0f}", f"{measured - predicted:+.0f}" if measured != predicted else "="
                differ += measured != predicted
            print(f"  {metric:<58}{shown:>10}{predicted:>10}{delta:>16}")
    print(f"\n{'all counters match' if not differ else f'{differ} counter(s) differ'}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
