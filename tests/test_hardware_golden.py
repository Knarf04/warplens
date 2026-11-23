"""Compare the simulator with results recorded from a real GPU (see validation/README.md).

validation/ampere/shfl_golden.txt was produced by running shfl_probe on an
RTX 3080 Ti (see validation/README.md). Each line is "mode b width r0..r31".
"""

from pathlib import Path

import numpy as np

AMPERE = Path(__file__).resolve().parent.parent / "validation" / "ampere"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODES = ["__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync"]


def test_shfl_matches_real_gpu(sim):
    out = sim.empty(32, "s32")
    mismatches = []
    cases = [list(map(int, line.split())) for line in (AMPERE / "shfl_golden.txt").read_text().splitlines()]
    assert len(cases) == 4 * 5 * 36
    for mode, b, width, *expect in cases:
        stats = sim.launch(AMPERE / "shfl_probe.ptx", "shfl_probe", grid=1, block=32, args=[out, mode, b, width])
        assert not stats.findings
        if out.numpy().tolist() != expect:
            mismatches.append(f"{MODES[mode]}(b={b}, width={width})")
    assert not mismatches, f"{len(mismatches)} of {len(cases)} cases differ: {mismatches[:10]}"


def test_bank_conflict_counters_match_ncu(sim):
    """transpose_ncu_3080ti_grid64.csv is `ncu --csv` output for bin/transpose_hw 64."""
    import csv

    hw: dict[str, dict[str, int]] = {}
    with open(AMPERE / "transpose_ncu_3080ti_grid64.csv", newline="") as f:
        for row in csv.DictReader(f):
            value = row["Metric Value"].replace(",", "")
            if value.isdigit():
                hw.setdefault(row["Kernel Name"], {})[row["Metric Name"].removesuffix(".sum")] = int(value)
    assert len(hw) == 3

    ctas = 64
    data = np.arange(ctas * 1024, dtype=np.float32).reshape(ctas, 32, 32)
    src = sim.from_numpy(data)
    for kernel, counters in hw.items():
        stats = sim.launch(EXAMPLES / "01_smem_transpose" / "transpose.ptx", kernel, grid=ctas,
                           block=(32, 32), args=[src, sim.empty(data.shape, "f32")])
        by_op = {"ld": [0, 0], "st": [0, 0]}
        for e in stats.smem.instructions.values():
            by_op[e.op][0] += e.executions
            by_op[e.op][1] += e.conflicts
        prefix = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared"
        assert counters["smsp__inst_executed_op_shared_ld"] == by_op["ld"][0]
        assert counters["smsp__inst_executed_op_shared_st"] == by_op["st"][0]
        assert counters[prefix] == stats.smem.conflicts
        assert counters[prefix + "_op_ld"] == by_op["ld"][1]
        assert counters[prefix + "_op_st"] == by_op["st"][1]
        # Known gap: hardware reports a constant 4160 more wavefronts than the model.
        assert counters["l1tex__data_pipe_lsu_wavefronts_mem_shared"] - stats.smem.wavefronts == 4160


def test_printf_struct_param_and_local_array_match_real_gpu(sim):
    """printf_golden.txt is the sorted output of bin/printf_hw on the GPU."""
    import ctypes

    class Config(ctypes.Structure):
        _fields_ = [("offset", ctypes.c_int), ("scale", ctypes.c_float),
                    ("table", ctypes.c_uint64), ("tag", ctypes.c_short)]

    table = sim.from_numpy(np.array([3, 1, 4, 1], dtype=np.int32))
    out = sim.empty(8, "s32")
    stats = sim.launch(AMPERE / "printf_probe.ptx", "printf_probe", grid=1, block=8,
                       args=[Config(100, 2.5, table.addr, -7), out])
    got = ["out: " + " ".join(map(str, out.numpy()))] + stats.output.splitlines()
    assert sorted(got) == (AMPERE / "printf_golden.txt").read_text().splitlines()
    assert not stats.findings
