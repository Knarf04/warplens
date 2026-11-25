"""Compare the simulator with results recorded from a real GPU (see validation/README.md).

validation/ampere/shfl_golden.txt was produced by running shfl_probe on an
RTX 3080 Ti (see validation/README.md). Each line is "mode b width r0..r31".
"""

import re

import numpy as np
import pytest

from warplens.core.engine import SCHEDULE_VARIABLE

from tests.support import AMPERE, EXAMPLES

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


def _vote_lines(path):
    """Lines of a vote recording: (mode, roles, predicates, masks, the rest of the line)."""
    for line in path.read_text().splitlines():
        mode, n, roles, preds, *words = line.split()
        n = int(n)
        assert len(roles) == len(preds) == n
        yield (int(mode), np.array(list(roles), dtype=np.int32), np.array(list(preds), dtype=np.int32),
               np.array([int(x, 16) for x in words[:n]], dtype=np.uint32), words[n:])


def _vote_launch(sim, ptx, mode, roles, preds, masks):
    out = sim.from_numpy(np.full(3 * len(roles), 0xEEEEEEEE, dtype=np.uint32))
    stats = sim.launch(AMPERE / ptx, "vote_probe", grid=1, block=len(roles),
                       args=[out, sim.from_numpy(preds), sim.from_numpy(masks), sim.from_numpy(roles), mode])
    return out.numpy(), stats


@pytest.mark.parametrize("ptx, golden", [("vote_probe.ptx", "vote_golden.txt"),
                                         ("vote_probe_cuda12.ptx", "vote_golden.txt"),
                                         ("vote_legacy_probe.ptx", "vote_legacy_golden.txt")])
def test_vote_matches_real_gpu(sim, ptx, golden):
    """vote_golden.txt is bin/vote_golden on an RTX 3080 Ti: `__all_sync`, `__any_sync`,
    `__uni_sync` and `__ballot_sync` (vote.sync) in full and partial warps, with lanes that
    returned, lanes on a branch without a vote, lanes on a branch with a ballot of their own,
    several groups with their own masks at one instruction, and masks that name lanes which
    do not exist or have returned (see vote_probe.cu and vote_golden.cu). Each line is a
    launch: mode, threads, a role and a predicate per thread, the masks, three words per thread.

    The probe runs as nvcc 13.5 compiles it for sm_100a and as nvcc 12.4 does for sm_86, the
    compiler and target of the GPU binary. vote_legacy_golden.txt is the same launches with
    the forms without .sync, which ptxas only takes for targets before sm_70: the program was
    built for sm_60 and the driver compiled its PTX for the GPU."""
    text = (AMPERE / ptx).read_text()
    spellings = set(re.findall(r"^\s*(vote\.[\w.]+)", text, re.M))
    sync = {"vote.sync.all.pred", "vote.sync.any.pred", "vote.sync.uni.pred", "vote.sync.ballot.b32"}
    legacy = {"vote.all.pred", "vote.any.pred", "vote.uni.pred", "vote.ballot.b32"}
    assert spellings == (sync | legacy if "legacy" in ptx else sync)
    assert ".target sm_60" in text if "legacy" in ptx else ".target sm_60" not in text
    assert "!p" in text  # a negated source
    lines = list(_vote_lines(AMPERE / golden))
    assert len(lines) == 5 * 444
    assert {mode for mode, *_ in lines} == set(range(5, 10) if "legacy" in ptx else range(5))
    wrong = []
    for k, (mode, roles, preds, masks, results) in enumerate(lines):
        got, stats = _vote_launch(sim, ptx, mode, roles, preds, masks)
        assert not stats.findings, (k, stats.findings)
        expect = np.array([int(x, 16) for x in results], dtype=np.uint32)
        if not np.array_equal(got, expect):
            t = int(np.flatnonzero(got != expect)[0])
            wrong.append(f"line {k + 1} (mode {mode}, {len(roles)} threads): word {t % 3} of thread {t // 3}: "
                         f"gpu {expect[t]:08x}, sim {got[t]:08x}")
    assert not wrong, f"{len(wrong)} of {len(lines)} launches differ: {wrong[:6]}"
    # What the recording itself says. A lane that returned has no vote though the mask names
    # it, and a lane that does not exist has none either.
    for mode, roles, preds, masks, results in lines:
        if mode % 5 == 3 and (masks == 0xFFFFFFFF).all() and len(roles) <= 32 and set(roles.tolist()) <= {0, 1}:
            ballot = sum(int(p) << t for t, (p, r) in enumerate(zip(preds, roles)) if r == 0)
            assert {results[3 * t] for t in np.flatnonzero(roles == 0)} == {f"{ballot:08x}"}


def test_vote_masks_a_gpu_does_not_answer_are_refused_or_reported(sim):
    """vote_other_3080ti.txt is validation/ampere/run_vote_other.sh on an RTX 3080 Ti: launches
    whose member mask names a lane on another branch, or leaves out the caller. They are not
    golden results. Where the mask names a lane that is elsewhere and has to wait for the
    voters itself, the GPU gave no result (the kernel never returned), or it gave one because
    two different `vote.sync.ballot` of the source share machine code and voted as one. The
    simulator refuses each of these. A mask without its caller is reported, as the GPU ended
    some of those with an illegal-instruction error."""
    import warplens
    path = AMPERE / "vote_other_3080ti.txt"
    lines = list(_vote_lines(path))
    assert len(lines) == 100
    outcomes = {"refused": 0, "reported": 0}
    for mode, roles, preds, masks, results in lines:
        hung = results[:2] == ["no", "result"]
        left_out = not ((masks >> np.arange(len(masks), dtype=np.uint32)) & 1).all()
        try:
            got, stats = _vote_launch(sim, "vote_probe_cuda12.ptx", mode, roles, preds, masks)
        except warplens.UnsupportedError as error:
            assert "names a lane that is on another path" in str(error) and not left_out
            outcomes["refused"] += 1
            continue
        assert not hung, "the GPU did not return and the simulator did"
        assert left_out and {f.kind for f in stats.findings} == {"vote-membermask-mismatch"}
        outcomes["reported"] += 1
    assert outcomes == {"refused": 40, "reported": 60}
    text = path.read_text()
    assert text.count("no result after") == 24 and text.count("cudaErrorIllegalInstruction") == 30


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


def test_bfi_and_shf_match_real_gpu(sim):
    """bitfield_golden.txt is bin/bitfield_golden on an RTX 3080 Ti: bfi.b32, bfi.b64 and
    shf.{l,r}.{wrap,clamp}.b32 over 2925 combinations of operands, positions and lengths,
    including positions and lengths past the register and past 255."""
    rows = np.array([[int(x, 16) for x in line.split()]
                     for line in (AMPERE / "bitfield_golden.txt").read_text().splitlines()], dtype=np.uint32)
    n = len(rows)
    assert rows.shape == (2925, 11)
    out = sim.empty(7 * n, "u32")
    stats = sim.launch(AMPERE / "bitfield_probe.ptx", "bitfield_probe", grid=-(-n // 256), block=256,
                       args=[*(sim.from_numpy(rows[:, k].copy()) for k in range(4)), out, n])
    assert not stats.findings
    got = out.numpy().reshape(n, 7)
    names = ["bfi.b32", "bfi.b64 low", "bfi.b64 high", "shf.l.wrap", "shf.r.wrap", "shf.l.clamp", "shf.r.clamp"]
    wrong = [f"{names[k]} a={rows[i, 0]:x} b={rows[i, 1]:x} pos={rows[i, 2]:x} len={rows[i, 3]:x}: "
             f"gpu {rows[i, 4 + k]:x}, sim {got[i, k]:x}" for i, k in np.argwhere(got != rows[:, 4:])]
    assert not wrong, f"{len(wrong)} values differ: {wrong[:6]}"
    # What the recording says where PTX leaves it open: bfi.b32 keeps only the low 8 bits of
    # pos and len (256 acts as 0), bfi.b64 does not (a pos of 256 inserts nothing).
    case = {(int(a), int(b), int(p), int(l)): r for a, b, p, l, *r in rows.tolist()}
    assert case[0xFFFFFFFF, 0, 256, 8][0] == 0xFF and case[0xFFFFFFFF, 0, 0, 256][0] == 0
    assert case[0xFFFFFFFF, 0, 256, 8][1:3] == [1, 0] and case[0xFFFFFFFF, 0, 0, 256][1:3] == [0x61C8864F, 0xFFFFFFFF]


@pytest.mark.parametrize("ptx", ["addrwrap_probe.ptx", "addrwrap_probe_cuda12.ptx"])
def test_offset_on_a_wrapped_32_bit_address_register_matches_real_gpu(sim, ptx):
    """addrwrap_golden.txt is bin/addrwrap_golden on an RTX 3080 Ti: `[reg+offset]` on shared
    memory with a 32-bit register that has wrapped below zero, so that the sum is the address
    only when it wraps at 32 bits (see addrwrap_probe.cu). One CTA of 64 threads per kernel.

    The probe runs as two toolkits compile it. In addrwrap_probe.ptx (nvcc 13.5, sm_100a) the
    compiler writes that form itself for `fixed[63 - tid]`. In addrwrap_probe_cuda12.ptx
    (nvcc 12.4, sm_86: the compiler and target of the GPU binary that made the recording) it
    does not, and the form is there only as the inline PTX of the `ptx` kernels, which is the
    same text in both files and is what the GPU executed."""
    text = (AMPERE / ptx).read_text()
    kernels = {name: body for name, body in re.findall(r"^\.visible \.entry (\w+)\((.*?)^}", text, re.M | re.S)}
    with_offset = {name: re.findall(r"^\s*((?:ld|st)\.shared\.\w+)\s[^;]*\[(%\w+)\+(-?\d+)\]", body, re.M)
                   for name, body in kernels.items()}
    assert with_offset["addrwrap_ptx_load"] == [("ld.shared.b32", "%a", "252")]
    assert with_offset["addrwrap_ptx_store"] == [("st.shared.b32", "%a", "252")]
    assert with_offset["addrwrap_ptx_far"] == [("ld.shared.b32", "%a", "2147483644")]
    assert with_offset["addrwrap_ptx_back"] == [("ld.shared.b32", "%a", "-4")]
    compiled = [(instr, offset) for name in ("addrwrap_cxx_load", "addrwrap_cxx_store")
                for instr, register, offset in with_offset[name] if re.fullmatch(r"%r\d+", register)]
    if ptx == "addrwrap_probe.ptx":  # a 32-bit register (%r, not %rd) and the constant 4 * 63
        assert compiled == [("ld.shared.b32", "252"), ("st.shared.b32", "252")]
    else:
        assert not compiled

    lines = [line.split() for line in (AMPERE / "addrwrap_golden.txt").read_text().splitlines()]
    assert [name for name, *_ in lines] == list(kernels) and len(lines) == 6
    for name, *values in lines:
        out = sim.from_numpy(np.full(64, -99, dtype=np.int32))  # as addrwrap_golden.cu fills it
        stats = sim.launch(AMPERE / ptx, name, grid=1, block=64, args=[out])
        assert out.numpy().tolist() == [int(v) for v in values], name
        assert not stats.findings, name
    # Every kernel reads or writes its array in reverse: the recording is 63 down to 0 on a base.
    assert [[int(v) for v in values] for _, *values in lines] == [
        list(range(base + 63, base - 1, -1)) for base in (1000, 2000, 3000, 4000, 5000, 6000)]


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
    from tests.support import kernel
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
    from tests.support import kernel
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
    from tests.support import kernel
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


def test_lanes_rejoin_where_they_do_on_a_real_gpu(sim):
    """reconvergence_golden.txt is validation/ampere/reconvergence_golden.py on an RTX 3080 Ti:
    `if (tid == 0) shared = 7;` and a join that reads the word, in four layouts of the lanes'
    paths to the join, with and without `bar.sync` there, with all or half of the lanes
    reaching it. Each line is a kernel's name and out[0..63]. On the GPU every lane that
    passes the join reads 7 in every layout: lane 0 arrives with the others, not after them."""
    import importlib.util
    import warplens
    spec = importlib.util.spec_from_file_location("reconvergence_golden", AMPERE / "reconvergence_golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    expect = {name: list(map(int, values)) for name, *values in
              map(str.split, (AMPERE / "reconvergence_golden.txt").read_text().splitlines())}
    kernels = golden.kernels()
    assert sorted(kernels) == sorted(expect) and len(kernels) == 20
    problems = []
    for name, body in kernels.items():
        out = sim.from_numpy(np.full(64, 99, dtype=np.uint32))
        stats = sim.launch(warplens.parse_ptx(golden.HEADER + body, name=name + ".ptx"), "k",
                           grid=1, block=32, args=[out])
        if out.numpy().tolist() != expect[name]:
            problems.append(f"{name}: {out.numpy().tolist()}")
        # Bypassing lanes finish their work and exit before the explicit
        # rendezvous counts the remaining lanes; that needs no warning.
        kinds = sorted({f.kind for f in stats.findings})
        if kinds:
            problems.append(f"{name}: findings {kinds}")
    assert not problems, problems


@pytest.mark.parametrize("ptx", ["atom_probe.ptx", "atom_probe_cuda12.ptx"])
def test_atomic_updates_match_real_gpu(sim, ptx):
    """atom_golden.txt is bin/atom_golden on an RTX 3080 Ti: 4 CTAs of 128 threads update the
    same global words and each CTA's shared words with every atomic function (see
    atom_probe.cu). The file starts with the values the words had before.

    The probe runs as two toolkits compile it. atom_probe.ptx is nvcc 13.5 for sm_100a
    (`atom.relaxed.gpu.global.add.u32`). atom_probe_cuda12.ptx is nvcc 12.4 for sm_86, the
    compiler and target of the GPU binary that made the recording: no ordering or scope
    (`atom.global.cas.b32`), and `red` wherever the result is discarded (`red.shared.min.s64`)."""
    spellings = set(re.findall(r"^\s*((?:atom|red)\.[\w.]+)", (AMPERE / ptx).read_text(), re.M))
    if ptx == "atom_probe_cuda12.ptx":  # the forms this file is here for
        assert not [s for s in spellings if "relaxed" in s]
        assert {"atom.global.add.u32", "atom.shared.exch.b64", "atom.cas.b32", "red.global.min.s64",
                "red.shared.and.b32", "red.global.inc.u32", "red.shared.dec.u32", "red.max.u32",
                "red.global.cta.xor.b32", "red.global.sys.or.b64", "red.shared.sys.add.f64"} <= spellings
    else:
        assert "atom.relaxed.gpu.global.add.u32" in spellings and not [s for s in spellings if s.startswith("red")]
    rows = [(label, [int(x, 16) for x in values]) for label, *values in
            (line.split() for line in (AMPERE / "atom_golden.txt").read_text().splitlines())]
    init = dict(rows[:4])
    assert list(init) == ["init32", "init64", "initf", "initd"]
    ctas, block, n32, n64, nf = 4, 128, len(init["init32"]), len(init["init64"]), len(init["initf"])
    assert (n32, n64, nf) == (64, 24, 4) and len(rows) == 4 + 6 + 5 * ctas

    def unsigned(dtype):
        return np.dtype(f"u{np.dtype(dtype).itemsize}")

    def device_array(values, dtype, extra=0):  # the words, their initial values again, `extra` zeros
        words = np.array(values + values + [0] * extra, dtype=unsigned(dtype))
        return sim.from_numpy(words.view(dtype))

    g32 = device_array(init["init32"], "u4", extra=2 * ctas)
    g64, gf, gd = (device_array(init[k], t) for k, t in (("init64", "u8"), ("initf", "f4"), ("initd", "f8")))
    ticket = sim.empty(2 * ctas * block, "u32")
    shared = [sim.empty(ctas * count, ty) for count, ty in ((n32, "u32"), (n64, "u64"), (nf, "f32"), (nf, "f64"))]
    stats = sim.launch(AMPERE / ptx, "atom_probe", grid=ctas, block=block,
                       args=[g32, g64, gf, gd, ticket, *shared, 1])
    assert not stats.findings  # in particular no smem-race between the warps' atomic updates

    def words(array, start, count, exchanged=()):
        """As atom_golden.cu prints them: an exchanged word is added to the sum of what the
        exchanges returned, which follows it, because which thread came last is not fixed."""
        values = array.numpy()[start:start + count].copy()
        for at in exchanged:
            values[at:at + 1] += values[at + 1:at + 2]  # as arrays: integers wrap around
            values[at + 1] = 0
        return values.view(unsigned(values.dtype)).tolist()

    tickets = ticket.numpy().reshape(ctas, block, 2)
    got = [("g32", words(g32, 0, n32, [14])), ("g32_cta", words(g32, 2 * n32, 2 * ctas)),
           ("g64", words(g64, 0, n64, [9])), ("gf", words(gf, 0, nf, [1])), ("gd", words(gd, 0, nf)),
           ("ticket_g", sorted(tickets[:, :, 0].ravel().tolist()))]
    for b in range(ctas):
        got += [("s32", words(shared[0], b * n32, n32, [14])), ("s64", words(shared[1], b * n64, n64, [9, 14])),
                ("sf", words(shared[2], b * nf, nf, [1])), ("sd", words(shared[3], b * nf, nf)),
                ("ticket_s", sorted(tickets[b, :, 1].tolist()))]
    assert [(label, len(v)) for label, v in rows[4:]] == [(name, len(v)) for name, v in got]
    wrong = [f"{label} word {k}: gpu {e:x}, sim {g:x}" for (label, expect), (_, values) in zip(rows[4:], got)
             for k, (e, g) in enumerate(zip(expect, values)) if e != g]
    assert not wrong, f"{len(wrong)} values differ: {wrong[:8]}"
    # What the recording itself must say, whatever order the GPU ran the threads in.
    recorded = dict(rows[4:10])
    total = ctas * block
    assert recorded["ticket_g"] == list(range(total))  # 512 increments, each a different old value
    assert recorded["g32"][0] == total and recorded["g32"][13] == 11 + 3 * total  # counter, CAS loop
    assert recorded["g32"][11] == total % 100 and recorded["g32"][12] == 100 - total % 100  # inc, dec


def test_mbarrier_tokens_match_real_gpu(sim):
    """mbarrier_token_golden.txt is bin/mbarrier_token_golden on an RTX 3080 Ti: after every
    arrival, `test_wait` with the token of each arrival so far (up to eight phases old), with
    the made-up tokens 1 << b, and `test_wait.parity` with 0 to 3. The GPU answers from bit 63
    of the token alone, which holds the parity of the phase the arrival was made in.
    Older and made-up tokens are invalid PTX use: this pins an empirical fallback,
    not a portable synchronization guarantee, and findings are required."""
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
        # Stale and forged tokens are deliberate invalid-use observations. They
        # reproduce Ampere's predicate but must not acquire synchronization credit.
        assert {f.kind for f in stats.findings} == {"mbarrier-invalid-token"}
        got = out.numpy().reshape(steps, 4)
        problems += [f"count {count}, arrival {s}: gpu {[hex(v) for v in expect[s]]}, sim {[hex(v) for v in got[s]]}"
                     for s in range(steps) if (got[s] != expect[s]).any()]
    assert not problems, f"{len(problems)} arrivals differ:\n" + "\n".join(problems[:8])


def _golden_instruction_mismatches(sim, path, operands, skip=(), expected_findings=()):
    """Run every line "instruction a b [c] result" of a golden file (64-bit hex patterns, a
    narrower value in the low bits) through the simulator, one instruction per thread, and
    describe the lines whose result differs. Returns (problems, lines per instruction)."""
    from collections import defaultdict
    from tests.support import kernel
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
            .reg .b64 %p0, %p1, %p2, %po, %off; .reg .b32 %i, %x, %y; .reg .pred %outside;
            .reg .{reg(src)} %a0, %a1, %a2; .reg .{reg(dst)} %d;
            ld.param.u64 %p0, [a]; ld.param.u64 %p1, [b]; ld.param.u64 %p2, [c]; ld.param.u64 %po, [o];
            mov.u32 %x, %ctaid.x; mov.u32 %y, %tid.x;
            mad.lo.u32 %i, %x, 256, %y;
            setp.ge.u32 %outside, %i, {n};
            @%outside ret;
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
        assert {f.kind for f in stats.findings} == set(expected_findings)
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
    `cvt.ftz.f64.f32` gives 0x7fffffffe0000000 for every NaN."""
    problems, counts = _golden_instruction_mismatches(sim, AMPERE / "nan_golden.txt", 3)
    assert len(counts) == 38 and sum(counts.values()) == 14270
    assert counts["cvt.ftz.f64.f32"] == counts["cvt.rn.ftz.f32.f64"] == 22
    assert not problems, "\n".join(problems)


def test_cvt_with_ftz_matches_real_gpu(sim):
    """cvt_ftz_golden.txt is bin/cvt_ftz_golden on an RTX 3080 Ti: `cvt` with `.ftz` between
    f16, f32 and f64, from f32 to the integer types and to an integer in f32 (what fast-math
    emits for casts, floorf and the like), each next to the form without `.ftz`, over the
    subnormal range of the source and of the result. Each line is "instruction a result".
    Compared bit for bit, NaN results included."""
    path = AMPERE / "cvt_ftz_golden.txt"
    problems, counts = _golden_instruction_mismatches(sim, path, 1)
    assert len(counts) == 40 and sum(counts.values()) == 8120
    assert sum(".ftz." in name for name in counts) == 25
    assert not problems, "\n".join(problems)
    # What the recording says. An f32 result is flushed when it is tiny after rounding: below
    # 2^-126 (1 - 2^-25), although from 2^-126 (1 - 2^-24) on it rounds to the smallest normal
    # number. f16 subnormals are not flushed, as source or as result.
    case = {(name, int(a, 16)): int(r, 16) for name, a, r in map(str.split, path.read_text().splitlines())}
    assert case["cvt.rn.f32.f64", 0x380FFFFFEFFFFFFF] == 0x00800000
    assert case["cvt.rn.ftz.f32.f64", 0x380FFFFFEFFFFFFF] == 0
    assert case["cvt.rn.ftz.f32.f64", 0x380FFFFFF0000000] == 0x00800000
    assert case["cvt.rn.ftz.f32.f64", 0xB800000000000000] == 0x80000000
    assert case["cvt.ftz.f64.f32", 0x807FFFFF] == 0x8000000000000000
    assert case["cvt.rn.ftz.f16.f32", 0x33000001] == 1 and case["cvt.ftz.f32.f16", 0x0001] == 0x33800000
    assert case["cvt.rpi.f32.f32", 0x00000001] == 0x3F800000 and case["cvt.rpi.ftz.f32.f32", 0x00000001] == 0
    assert case["cvt.rmi.ftz.s32.f32", 0x807FFFFF] == 0 and case["cvt.rmi.ftz.s32.f32", 0x80800000] == 0xFFFFFFFF
    assert case["cvt.ftz.f32.f32", 0xFFC12345] == 0x7FFFFFFF and case["cvt.f32.f32", 0xFFC12345] == 0xFFC12345


def test_integer_division_matches_real_gpu(sim):
    """intdiv_golden.txt is bin/intdiv_golden on an RTX 3080 Ti: `div` and `rem` for s16, u16,
    s32, u32, s64 and u64 over all pairs of 22 values. Division by zero gives all ones, for
    `rem` too; the minimum over -1 gives the minimum, and remainder 0. Zero-divisor
    results are machine-specific in PTX, so the empirical match retains findings."""
    problems, counts = _golden_instruction_mismatches(sim, AMPERE / "intdiv_golden.txt", 2,
        expected_findings=("integer-divide-by-zero",))
    assert len(counts) == 12 and sum(counts.values()) == 5808
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("schedule", ["1", "2", "adversarial", "adversarial:3"])
@pytest.mark.parametrize("check", [
    test_shfl_matches_real_gpu,
    test_printf_struct_param_and_local_array_match_real_gpu,
    test_cp_async_with_a_source_size_matches_real_gpu,
    test_device_function_calls_match_real_gpu,
    test_lanes_rejoin_where_they_do_on_a_real_gpu,
    test_mbarrier_tokens_match_real_gpu,
], ids=lambda check: check.__name__.removeprefix("test_"))
def test_hardware_golden_results_hold_under_other_schedules(sim, monkeypatch, check, schedule):
    """All of this file passes with WARPLENS_SCHEDULE set (the order in which runnable warps are
    stepped: docs/scheduling.md); these are the quick ones, run again here under other schedules
    so that the suite keeps saying so."""
    monkeypatch.setenv(SCHEDULE_VARIABLE, schedule)
    check(sim)
