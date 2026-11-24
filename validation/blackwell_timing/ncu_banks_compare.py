#!/usr/bin/env python3
"""The simulator's shared-memory counters for the kernels of ncu_banks.cu, next to ncu's.

    PYTHONPATH=<repo> python ncu_banks_compare.py                the simulator's counters only
    PYTHONPATH=<repo> python ncu_banks_compare.py NCU.csv        compared with `ncu --csv` output
    PYTHONPATH=<repo> python ncu_banks_compare.py --self-check   the 32 x 32 kernels count like
                                                                 examples/01_smem_transpose

NCU.csv is what `ncu --csv --metrics <the SMEM list of run_ncu.sh> bin/hw/ncu_banks --trials 1`
printed (validation matrix TIM-16 and TIM-17). The kernels are simulated from ncu_banks.ptx
(`nvcc -arch=sm_100a -ptx -DBENCH_CLOCK=0 ncu_banks.cu`, committed), grid 64, with the block
shape in each kernel's name.

For every kernel it prints ncu's value, the model's, and the difference, and at the end the
wavefront surplus (ncu - model) per kernel next to the kernel's CTAs, warps and shared-memory
instructions executed, which is what TIM-17 asks: is the surplus per CTA, per warp or per
instruction? Exit status 1 if an instruction or conflict counter differs (TIM-16); the
wavefront totals are expected to differ (TIM-17) and do not count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import warplens

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ampere"))
import compare_ncu  # noqa: E402  (read_ncu, and the 32 x 32 reference kernels)

PTX = HERE / "ncu_banks.ptx"
CTAS = 64
SHAPES = [(32, 32), (32, 8), (16, 16)]
LAYOUTS = ["linear", "padded", "swizzled"]
WAVEFRONTS = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
EXPECTED_TO_DIFFER = (WAVEFRONTS, "l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum")


def simulate(layout: str, tx: int, ty: int, ctas: int = CTAS) -> dict[str, int]:
    sim = warplens.Device()
    data = np.arange(ctas * tx * ty, dtype=np.float32).reshape(ctas, ty, tx)
    out = sim.empty(data.shape, "f32")
    stats = sim.launch(PTX, f"banks_{layout}_{tx}x{ty}", grid=ctas, block=(tx, ty),
                       args=[sim.from_numpy(data), out])
    flat = data.reshape(ctas, -1)
    t = np.arange(tx * ty)
    assert np.array_equal(out.numpy().reshape(ctas, -1), flat[:, (t % ty) * tx + t // ty])
    by_op = {"ld": [0, 0], "st": [0, 0]}  # executions, conflicts
    for e in stats.smem.instructions.values():
        by_op[e.op][0] += e.executions
        by_op[e.op][1] += e.conflicts
    return {
        "smsp__inst_executed_op_shared_ld.sum": by_op["ld"][0],
        "smsp__inst_executed_op_shared_st.sum": by_op["st"][0],
        WAVEFRONTS: stats.smem.wavefronts,
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum": stats.smem.wavefronts - stats.smem.conflicts,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": stats.smem.conflicts,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": by_op["ld"][1],
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": by_op["st"][1],
    }


def self_check() -> int:
    bad = 0
    for layout in LAYOUTS:
        here, there = simulate(layout, 32, 32), compare_ncu.simulate(f"transpose_{layout}", CTAS)
        same = here == there
        bad += not same
        print(f"banks_{layout}_32x32 against transpose_{layout}: {'same counters' if same else 'DIFFERENT'}")
        if not same:
            print(f"  {here}\n  {there}")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-check"]:
        return self_check()
    hw = compare_ncu.read_ncu(Path(argv[1])) if len(argv) > 1 else {}
    differ, surplus = 0, []
    for tx, ty in SHAPES:
        for layout in LAYOUTS:
            kernel = f"banks_{layout}_{tx}x{ty}"
            sim = simulate(layout, tx, ty)
            print(f"\n{kernel}  (grid {CTAS}, block {tx}x{ty})")
            print(f"  {'metric':<58}{'ncu':>10}{'warplens':>10}{'ncu - warplens':>16}")
            for metric, predicted in sim.items():
                measured = hw.get(kernel, {}).get(metric)
                if measured is None:
                    shown, delta = "n/a", ""
                else:
                    shown = f"{measured:.0f}"
                    delta = f"{measured - predicted:+.0f}" if measured != predicted else "="
                    differ += measured != predicted and metric not in EXPECTED_TO_DIFFER
                    if metric == WAVEFRONTS:
                        warps = CTAS * -(-tx * ty // 32)
                        executed = sim["smsp__inst_executed_op_shared_ld.sum"] + sim["smsp__inst_executed_op_shared_st.sum"]
                        surplus.append((kernel, measured - predicted, CTAS, warps, executed))
                print(f"  {metric:<58}{shown:>10}{predicted:>10}{delta:>16}")
    if surplus:
        print("\nwavefront surplus (ncu - warplens), TIM-17")
        print(f"  {'kernel':<24}{'surplus':>9}{'CTAs':>6}{'warps':>7}{'ld+st executed':>16}{'per CTA':>9}{'per warp':>10}")
        for kernel, extra, ctas, warps, executed in surplus:
            print(f"  {kernel:<24}{extra:>+9.0f}{ctas:>6}{warps:>7}{executed:>16}{extra / ctas:>9.2f}{extra / warps:>10.3f}")
    if hw:
        print(f"\n{'instruction and conflict counters match' if not differ else f'{differ} instruction or conflict counter(s) differ'}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
