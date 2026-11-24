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
        # Hardware also counts two wavefronts per warp and one per CTA per barrier
        # (see test_wavefront_offset_rule): 2 * 64 * 32 + 64 = 4160 here.
        assert counters["l1tex__data_pipe_lsu_wavefronts_mem_shared"] - stats.smem.wavefronts == 4160
        assert counters["l1tex__data_pipe_lsu_wavefronts_mem_shared"] == stats.smem.ncu_wavefronts


def test_wavefront_offset_rule():
    """wavefront_probe_ncu_3080ti.csv is `ncu --csv` output for bin/wavefront_probe.

    In every launch the hardware counter is the model's count (one per
    conflict-free shared access) plus two per warp plus one per CTA per barrier.
    """
    import csv

    lines = (AMPERE / "wavefront_probe_ncu_3080ti.csv").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    hw: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(lines[start:]):
        try:
            hw.setdefault(int(row["ID"]), {})[row["Metric Name"].removesuffix(".sum")] = float(
                row["Metric Value"].replace(",", ""))
        except ValueError:
            pass
    configs = list(csv.DictReader((AMPERE / "wavefront_probe_configs_3080ti.csv").open()))
    assert len(configs) == len(hw) == 59
    for c in configs:
        h = hw[int(c["launch"])]
        grid, threads = int(c["grid"]), int(c["bx"]) * int(c["by"])
        warps = grid * -(-threads // 32)
        accesses = h["smsp__inst_executed_op_shared_ld"] + h["smsp__inst_executed_op_shared_st"]
        assert h["l1tex__data_bank_conflicts_pipe_lsu_mem_shared"] == 0
        assert h["l1tex__data_pipe_lsu_wavefronts_mem_shared"] == accesses + 2 * warps + grid * int(c["n_bar"]), c


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


def test_ldmatrix_and_bfe_match_real_gpu(sim):
    """ldmatrix_golden.txt is bin/ldmatrix_golden on an RTX 3080 Ti: all six ldmatrix forms
    with a permuted row address per lane, and bfe.u32/bfe.s32 over 726 edge cases."""
    lines = [l.split() for l in (AMPERE / "ldmatrix_golden.txt").read_text().splitlines()]
    expect = np.array([[int(x, 16) for x in l[3:]] for l in lines if l[0] == "L"], dtype=np.uint32)
    out = sim.empty((6, 32, 4), "u32")
    stats = sim.launch(AMPERE / "ldmatrix_probe.ptx", "ldmatrix_probe", grid=1, block=32, args=[out])
    np.testing.assert_array_equal(out.numpy().reshape(-1, 4), expect)
    assert not stats.findings

    cases = np.array([[int(x, 16) for x in l[1:]] for l in lines if l[0] == "B"], dtype=np.uint32)
    n = len(cases)
    assert n == 726
    result = sim.empty(2 * n, "u32")
    sim.launch(AMPERE / "ldmatrix_probe.ptx", "bfe_probe", grid=-(-n // 256), block=256,
               args=[sim.from_numpy(cases[:, 0].copy()), sim.from_numpy(cases[:, 1].copy()),
                     sim.from_numpy(cases[:, 2].copy()), result, n])
    got = result.numpy()
    wrong = [(hex(a), int(p), int(l), hex(u), hex(gu), hex(s), hex(gs))
             for (a, p, l, u, s), gu, gs in zip(cases, got[:n], got[n:]) if (u, s) != (gu, gs)]
    assert not wrong, f"{len(wrong)} of {n} differ, e.g. (a, pos, len, unsigned, got, signed, got): {wrong[:4]}"


def test_cp_async_with_a_source_size_matches_real_gpu(sim):
    """cpasync_golden.txt is bin/cpasync_golden on an RTX 3080 Ti: each lane copies 16 bytes
    but reads only its source size; the GPU zero-fills the rest."""
    rows = [l.split() for l in (AMPERE / "cpasync_golden.txt").read_text().splitlines()]
    sizes = np.array([int(r[0]) for r in rows], dtype=np.uint32)
    expect = np.array([[int(x, 16) for x in r[1:]] for r in rows], dtype=np.uint8)
    data = ((np.arange(512) * 7 + 1) % 256).astype(np.uint8)
    out = sim.empty((32, 16), "u8")
    stats = sim.launch(AMPERE / "cpasync_probe.ptx", "cpasync_probe", grid=1, block=32,
                       args=[sim.from_numpy(data), sim.from_numpy(sizes), out])
    np.testing.assert_array_equal(out.numpy(), expect)
    assert not stats.findings
    assert (expect[3, 3:] == 0).all() and (expect[3, :3] == data[48:51]).all()  # 3 read, 13 zeroed


def test_prmt_matches_real_gpu(sim):
    """prmt_golden.txt is bin/prmt_golden on an RTX 3080 Ti: 304 (a, b, control) triples."""
    from conftest import kernel
    rows = np.array([[int(x, 16) for x in l.split()] for l in (AMPERE / "prmt_golden.txt").read_text().splitlines()],
                    dtype=np.uint32)
    n = len(rows)
    mod = kernel("""
        .reg .b64 %pa, %pb, %pc, %po, %off; .reg .b32 %i, %a, %b, %c, %d, %x, %y;
        ld.param.u64 %pa, [a]; ld.param.u64 %pb, [b]; ld.param.u64 %pc, [c]; ld.param.u64 %po, [o];
        mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
        mad.lo.u32 %i, %x, 256, %y;
        mul.wide.u32 %off, %i, 4;
        add.u64 %pa, %pa, %off; add.u64 %pb, %pb, %off; add.u64 %pc, %pc, %off; add.u64 %po, %po, %off;
        ld.global.u32 %a, [%pa]; ld.global.u32 %b, [%pb]; ld.global.u32 %c, [%pc];
        prmt.b32 %d, %a, %b, %c;
        st.global.u32 [%po], %d;
        ret;
    """, ".param .u64 a, .param .u64 b, .param .u64 c, .param .u64 o")
    pad = -n % 256
    columns = [np.concatenate([rows[:, k], np.zeros(pad, np.uint32)]) for k in range(3)]
    out = sim.empty(n + pad, "u32")
    sim.launch(mod, "k", grid=(n + pad) // 256, block=256, args=[*(sim.from_numpy(c) for c in columns), out])
    np.testing.assert_array_equal(out.numpy()[:n], rows[:, 3])


def test_flush_to_zero_arithmetic_matches_real_gpu(sim):
    """ftz_golden.txt is bin/ftz_golden on an RTX 3080 Ti: add, sub, mul, div, fma and sqrt with
    `.ftz` on operands around the subnormal range. NaN results are compared as NaNs only."""
    from conftest import kernel
    rows = np.array([[int(x, 16) for x in l.split()] for l in (AMPERE / "ftz_golden.txt").read_text().splitlines()],
                    dtype=np.uint32)
    assert len(rows) == 7075
    bodies = ["add.rn.ftz.f32 %d, %a, %b;", "sub.rn.ftz.f32 %d, %a, %b;", "mul.rn.ftz.f32 %d, %a, %b;",
              "div.rn.ftz.f32 %d, %a, %b;", "fma.rn.ftz.f32 %d, %a, %b, %c;", "sqrt.rn.ftz.f32 %d, %a;"]
    for op, body in enumerate(bodies):
        cases = rows[rows[:, 0] == op]
        n = len(cases)
        mod = kernel(f"""
            .reg .b64 %pa, %pb, %pc, %po, %off; .reg .b32 %i, %x, %y; .reg .f32 %a, %b, %c, %d;
            ld.param.u64 %pa, [a]; ld.param.u64 %pb, [b]; ld.param.u64 %pc, [c]; ld.param.u64 %po, [o];
            mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
            mad.lo.u32 %i, %x, 256, %y;
            mul.wide.u32 %off, %i, 4;
            add.u64 %pa, %pa, %off; add.u64 %pb, %pb, %off; add.u64 %pc, %pc, %off; add.u64 %po, %po, %off;
            ld.global.f32 %a, [%pa]; ld.global.f32 %b, [%pb]; ld.global.f32 %c, [%pc];
            {body}
            st.global.f32 [%po], %d;
            ret;
        """, ".param .u64 a, .param .u64 b, .param .u64 c, .param .u64 o")
        pad = -n % 256
        columns = [np.concatenate([cases[:, k], np.zeros(pad, np.uint32)]) for k in (1, 2, 3)]
        out = sim.empty(n + pad, "u32")
        sim.launch(mod, "k", grid=(n + pad) // 256, block=256, args=[*(sim.from_numpy(c) for c in columns), out])
        got, expect = out.numpy()[:n], cases[:, 4]
        nan = np.isnan(expect.view(np.float32))
        assert (np.isnan(got.view(np.float32)) == nan).all(), body
        wrong = np.flatnonzero((got != expect) & ~nan)
        assert not len(wrong), (f"{body} {len(wrong)} of {n} differ, e.g. " +
                                "; ".join(f"{cases[i, 1]:08x} {cases[i, 2]:08x} {cases[i, 3]:08x} -> gpu "
                                          f"{expect[i]:08x}, sim {got[i]:08x}" for i in wrong[:6]))


def test_device_function_calls_match_real_gpu(sim):
    """call_golden.txt is bin/call_golden on an RTX 3080 Ti: 256 threads each call functions the
    compiler did not inline (early returns, nested calls, a local array, mixed argument types,
    a store through a pointer, and a call only some lanes make)."""
    expect = np.loadtxt(AMPERE / "call_golden.txt", dtype=np.int64).astype(np.int32)
    assert expect.shape == (256, 5)
    out = sim.empty(5 * 256, "s32")
    stats = sim.launch(AMPERE / "call_probe.ptx", "call_probe", grid=4, block=64, args=[out])
    assert not stats.findings
    np.testing.assert_array_equal(out.numpy().reshape(256, 5), expect)
