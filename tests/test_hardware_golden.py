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
    `.ftz` on operands around the subnormal range. Compared bit for bit, NaN results included."""
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
        assert (expect[nan] == 0x7FFFFFFF).all()  # the only NaN the GPU returned, 642 times
        wrong = np.flatnonzero(got != expect)
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


def test_call_forms_match_real_gpu(sim):
    """callforms_golden.txt is bin/callforms_golden on an RTX 3080 Ti: structs passed and returned
    by value, calls through function pointers, and recursion (see callforms_probe.cu)."""
    expect = np.loadtxt(AMPERE / "callforms_golden.txt", dtype=np.int64).astype(np.int32)
    assert expect.shape == (256, 12)
    out = sim.empty(12 * 256, "s32")
    stats = sim.launch(AMPERE / "callforms_probe.ptx", "callforms_probe", grid=4, block=64, args=[out])
    assert not stats.findings
    np.testing.assert_array_equal(out.numpy().reshape(256, 12), expect)


def test_min_max_and_float_to_integer_match_real_gpu(sim):
    """minmax_cvt_golden.txt is bin/minmax_cvt_golden on an RTX 3080 Ti: `min` and `max` on f32
    and f64 over all pairs of zeros, infinities, NaNs and a few ordinary values, and `cvt` from
    f32 and f64 to s32, u32, s64 and u64 in all four rounding modes over NaNs, infinities,
    halves and the values around every integer limit. Each line is "instruction a b result"
    with 64-bit hex patterns. Compared bit for bit, NaN results included."""
    from collections import defaultdict
    from conftest import kernel
    cases = defaultdict(list)
    for line in (AMPERE / "minmax_cvt_golden.txt").read_text().splitlines():
        name, *values = line.split()
        cases[name].append([int(x, 16) for x in values])
    assert len(cases) == 4 + 4 * 4 * 2 and sum(map(len, cases.values())) == 15844
    problems = []
    for name, rows in cases.items():
        rows = np.array(rows, dtype=np.uint64)
        n = len(rows)
        *_, src = name.split(".")
        dst, operands = (name.split(".")[2], "%a") if name.startswith("cvt") else (src, "%a, %b")
        mod = kernel(f"""
            .reg .b64 %pa, %pb, %po, %off; .reg .b32 %i, %x, %y;
            .reg .{src} %a, %b; .reg .{dst} %d;
            ld.param.u64 %pa, [a]; ld.param.u64 %pb, [b]; ld.param.u64 %po, [o];
            mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
            mad.lo.u32 %i, %x, 256, %y;
            mul.wide.u32 %off, %i, 8;
            add.u64 %pa, %pa, %off; add.u64 %pb, %pb, %off; add.u64 %po, %po, %off;
            ld.global.{src} %a, [%pa]; ld.global.{src} %b, [%pb];
            {name} %d, {operands};
            st.global.{dst} [%po], %d;
            ret;
        """, ".param .u64 a, .param .u64 b, .param .u64 o")
        pad = -n % 256
        columns = [np.concatenate([rows[:, k], np.zeros(pad, np.uint64)]) for k in (0, 1)]
        out = sim.from_numpy(np.zeros(n + pad, np.uint64))  # a 32-bit result leaves the high half zero
        stats = sim.launch(mod, "k", grid=(n + pad) // 256, block=256,
                           args=[*(sim.from_numpy(c) for c in columns), out])
        assert not stats.findings
        got, expect = out.numpy()[:n], rows[:, 2]
        wrong = np.flatnonzero(got != expect)
        if len(wrong):
            problems.append(f"{name}: {len(wrong)} of {n} differ, e.g. " + "; ".join(
                f"{rows[i, 0]:x} {rows[i, 1]:x} -> gpu {expect[i]:x}, sim {got[i]:x}" for i in wrong[:4]))
    assert not problems, "\n".join(problems)


def test_mbarrier_tokens_match_real_gpu(sim):
    """mbarrier_token_golden.txt is bin/mbarrier_token_golden on an RTX 3080 Ti: after every
    arrival, `test_wait` with the token of each arrival so far (up to eight phases old), with
    the made-up tokens 1 << b, and `test_wait.parity` with 0 to 3. The GPU answers from bit 63
    of the token alone, which holds the parity of the phase the arrival was made in."""
    rows = [l.split() for l in (AMPERE / "mbarrier_token_golden.txt").read_text().splitlines()]
    assert len(rows) == 8 * (1 + 2 + 3 + 5)
    problems = []
    for count in (1, 2, 3, 5):
        expect = np.array([[int(r[2])] + [int(x, 16) for x in r[3:]] for r in rows if int(r[0]) == count],
                          dtype=np.uint64)
        steps = len(expect)
        out = sim.from_numpy(np.zeros(4 * steps, np.uint64))
        stats = sim.launch(AMPERE / "mbarrier_token_probe.ptx", "mbarrier_token_probe", grid=1, block=1,
                           args=[np.uint32(count), np.int32(steps), out])
        assert not stats.findings
        got = out.numpy().reshape(steps, 4)
        problems += [f"count {count}, arrival {s}: gpu {[hex(v) for v in expect[s]]}, sim {[hex(v) for v in got[s]]}"
                     for s in range(steps) if (got[s] != expect[s]).any()]
    assert not problems, f"{len(problems)} arrivals differ:\n" + "\n".join(problems[:8])


def _golden_instruction_mismatches(sim, path, operands, skip=()):
    """Run every line "instruction a b [c] result" of a golden file (64-bit hex patterns, a
    narrower value in the low bits) through the simulator, one instruction per thread, and
    describe the lines whose result differs. Returns (problems, lines per instruction)."""
    from collections import defaultdict
    from conftest import kernel
    cases = defaultdict(list)
    for line in path.read_text().splitlines():
        name, *values = line.split()
        assert len(values) == operands + 1
        cases[name].append([int(x, 16) for x in values])
    reg = lambda ty: "b16" if ty == "f16" else ty  # half floats live in 16-bit registers
    problems = []
    for name, rows in cases.items():
        if name in skip:
            continue
        rows = np.array(rows, dtype=np.uint64)
        n = len(rows)
        *_, src = name.split(".")
        dst = name.split(".")[-2] if name.startswith("cvt") else src
        arity = {"fma": 3, "sqrt": 1, "neg": 1, "abs": 1, "mov": 1, "cvt": 1}.get(name.split(".")[0], 2)
        ins = [f"%a{k}" for k in range(arity)]
        mod = kernel(f"""
            .reg .b64 %p0, %p1, %p2, %po, %off; .reg .b32 %i, %x, %y;
            .reg .{reg(src)} %a0, %a1, %a2; .reg .{reg(dst)} %d;
            ld.param.u64 %p0, [a]; ld.param.u64 %p1, [b]; ld.param.u64 %p2, [c]; ld.param.u64 %po, [o];
            mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
            mad.lo.u32 %i, %x, 256, %y;
            mul.wide.u32 %off, %i, 8;
            add.u64 %p0, %p0, %off; add.u64 %p1, %p1, %off; add.u64 %p2, %p2, %off; add.u64 %po, %po, %off;
            ld.global.{reg(src)} %a0, [%p0]; ld.global.{reg(src)} %a1, [%p1]; ld.global.{reg(src)} %a2, [%p2];
            {name} %d, {", ".join(ins)};
            st.global.{reg(dst)} [%po], %d;
            ret;
        """, ".param .u64 a, .param .u64 b, .param .u64 c, .param .u64 o")
        pad = -n % 256
        columns = [np.concatenate([rows[:, k] if k < operands else np.zeros(n, np.uint64), np.zeros(pad, np.uint64)])
                   for k in range(3)]
        out = sim.from_numpy(np.zeros(n + pad, np.uint64))  # a narrow result leaves the rest zero
        stats = sim.launch(mod, "k", grid=(n + pad) // 256, block=256,
                           args=[*(sim.from_numpy(c) for c in columns), out])
        assert not stats.findings
        got, expect = out.numpy()[:n], rows[:, operands]
        wrong = np.flatnonzero(got != expect)
        if len(wrong):
            problems.append(f"{name}: {len(wrong)} of {n} differ, e.g. " + "; ".join(
                " ".join(f"{v:x}" for v in rows[i, :arity]) + f" -> gpu {expect[i]:x}, sim {got[i]:x}"
                for i in wrong[:4]))
    return problems, {name: len(rows) for name, rows in cases.items()}


def test_nan_results_match_real_gpu(sim):
    """nan_golden.txt is bin/nan_golden on an RTX 3080 Ti: add, sub, mul, div, fma, sqrt, neg,
    abs, mov and the conversions between f16, f32 and f64 over NaNs of every kind in every
    operand position and over every invalid operation. Compared bit for bit: an f32 NaN
    result is 0x7fffffff whatever went in; an f64 one is a NaN operand made quiet (which one
    depends on the instruction) or 0xfff8000000000000 for an invalid operation.
    `cvt` with `.ftz` is recorded but not implemented by the simulator."""
    problems, counts = _golden_instruction_mismatches(
        sim, AMPERE / "nan_golden.txt", 3, skip=("cvt.ftz.f64.f32", "cvt.rn.ftz.f32.f64"))
    assert len(counts) == 38 and sum(counts.values()) == 14270
    assert not problems, "\n".join(problems)


def test_integer_division_matches_real_gpu(sim):
    """intdiv_golden.txt is bin/intdiv_golden on an RTX 3080 Ti: `div` and `rem` for s16, u16,
    s32, u32, s64 and u64 over all pairs of 22 values. Division by zero gives all ones, for
    `rem` too; the minimum over -1 gives the minimum, and remainder 0."""
    problems, counts = _golden_instruction_mismatches(sim, AMPERE / "intdiv_golden.txt", 2)
    assert len(counts) == 12 and sum(counts.values()) == 5808
    assert not problems, "\n".join(problems)
