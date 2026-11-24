"""`atom` and `red`: the forms nvcc 13.5 emits for sm_100a and nvcc 12.4 for sm_80/sm_86, the
order of lanes that share an address, the shared-memory race detector, and what the timing
model charges."""

import numpy as np
import pytest

import warplens

from conftest import kernel

# What `nvcc -arch=sm_100a -ptx` (13.5) emits before the operation: atomicX(), and the
# _block and _system variants, on pointers it knows to be global or shared and on others.
PREFIXES = ["relaxed.gpu.global", "relaxed.gpu.shared", "relaxed.gpu",
            "global.cta", "global.sys", "shared.cta", "shared.sys", "cta", "sys"]
# What nvcc 12.4 emits for sm_80 and sm_86 (and 13.5 for those targets): no ordering, and a
# scope only for _block and _system. PTX gives the omitted ordering as .relaxed and the omitted
# scope as .gpu, so `atom.global.add.u32` is `atom.relaxed.gpu.global.add.u32`. "" is a
# pointer the compiler cannot place.
LEGACY_PREFIXES = ["global", "shared", "", "global.cta", "global.sys", "shared.cta", "shared.sys", "cta", "sys"]
OPERATIONS = ["add.u32", "add.u64", "add.f32", "add.f64", "and.b32", "and.b64", "or.b32", "or.b64",
              "xor.b32", "xor.b64", "exch.b32", "exch.b64", "cas.b32", "cas.b64", "inc.u32", "dec.u32",
              "min.s32", "min.u32", "min.s64", "min.u64", "max.s32", "max.u32", "max.s64", "max.u64"]
PARAMS = ".param .u64 mem, .param .u64 b, .param .u64 c, .param .u64 out"


def lane_kernel(instruction: str, bits: int, where: str, generic: bool = False):
    """Lane t: `instruction` on word t of `mem` (or of a shared copy of it) at [%a], with
    %b = b[t] and %c = c[t]; out[t] = %d, and the word goes back to mem[t]."""
    size = bits // 8
    if where == "global":
        before, after = "mov.u64 %a, %p;", ""
    else:
        before = f"""ld.global.b{bits} %v, [%p];
            mov.u64 %s, cell; add.u64 %s, %s, %off; st.shared.b{bits} [%s], %v;
            {'cvta.shared.u64 %a, %s;' if generic else 'mov.u64 %a, %s;'}"""
        after = f"ld.shared.b{bits} %v, [%s]; st.global.b{bits} [%p], %v;"
    return kernel(f"""
        .reg .b64 %p, %o, %off, %a, %s; .reg .b32 %t; .reg .b{bits} %b, %c, %d, %v;
        .shared .align 8 .b{bits} cell[32];
        mov.u32 %t, %tid.x; mul.wide.u32 %off, %t, {size};
        ld.param.u64 %p, [mem]; add.u64 %p, %p, %off;
        ld.param.u64 %o, [b]; add.u64 %o, %o, %off; ld.global.b{bits} %b, [%o];
        ld.param.u64 %o, [c]; add.u64 %o, %o, %off; ld.global.b{bits} %c, [%o];
        {before}
        {instruction}
        {after}
        ld.param.u64 %o, [out]; add.u64 %o, %o, %off; st.global.b{bits} [%o], %d;
        ret;
    """, PARAMS)


def expected(operation: str, ty: str, old: int, b: int, c: int) -> int:
    """The new contents of a word, on bit patterns, written without NumPy integer arithmetic."""
    bits = int(ty[1:])
    full = (1 << bits) - 1
    signed = lambda x: x - (1 << bits) if x >> (bits - 1) else x
    if operation == "add" and ty[0] == "f":
        f, u = (np.float32, np.uint32) if bits == 32 else (np.float64, np.uint64)
        with np.errstate(all="ignore"):
            total = np.array([old], u).view(f) + np.array([b], u).view(f)
        return int(total.view(u)[0])
    if operation in ("min", "max") and ty[0] == "s":
        return (min if operation == "min" else max)(old, b, key=signed)
    return {"add": lambda: (old + b) & full, "and": lambda: old & b, "or": lambda: old | b,
            "xor": lambda: old ^ b, "exch": lambda: b, "cas": lambda: c if old == b else old,
            "inc": lambda: 0 if old >= b else old + 1,
            "dec": lambda: b if old == 0 or old > b else old - 1,
            "min": lambda: min(old, b), "max": lambda: max(old, b)}[operation]()


def lane_inputs(operation: str, ty: str):
    bits = int(ty[1:])
    unsigned = np.dtype(f"u{bits // 8}")
    rng = np.random.default_rng(bits + len(operation))
    if ty[0] == "f":
        real = np.dtype(f"f{bits // 8}")
        old = rng.standard_normal(32).astype(real)
        b = rng.standard_normal(32).astype(real)
        old[:4] = [np.inf, 1e30 if bits == 32 else 1e300, 0.0, 1.0]
        b[:4] = [-1.0, 3e38 if bits == 32 else 1.7e308, -0.0, real.type(2) ** -30]
        return old.view(unsigned), b.view(unsigned), np.zeros(32, unsigned)
    old = rng.integers(0, 1 << bits, 32, dtype=unsigned)
    b = rng.integers(0, 1 << bits, 32, dtype=unsigned)
    c = rng.integers(0, 1 << bits, 32, dtype=unsigned)
    top = (1 << bits) - 1
    old[:8] = [0, top, 5, 99, 100, 1 << (bits - 1), top, 7]
    b[:8] = [99, 1, 5, 99, 99, top >> 1, top, 3]
    if operation == "cas":
        b[8:20] = old[8:20]  # these lanes find what they expect
    return old, b, c


# -- every form, one lane per word ------------------------------------------------------------

def check_form(sim, opcode, prefix, operation):
    """`atom` returns what it found; `red` is the same update with no destination, so %d keeps
    the marker it was given."""
    name, ty = operation.split(".")
    bits = int(ty[1:])
    old, b, c = lane_inputs(name, ty)
    want = np.array([expected(name, ty, int(o), int(x), int(y)) for o, x, y in zip(old, b, c)], old.dtype)
    spelling = ".".join(part for part in (opcode, prefix, operation) if part)
    if opcode == "atom":
        instruction = f"{spelling} %d, [%a], %b{', %c' if name == 'cas' else ''};"
    else:
        instruction = f"mov.b{bits} %d, 0x5a; {spelling} [%a], %b;"
    generic = "global" not in prefix and "shared" not in prefix
    for where in ("global", "shared") if generic else (prefix.replace("relaxed.gpu.", "").split(".")[0],):
        mem, out = sim.from_numpy(old.copy()), sim.empty(32, old.dtype)
        stats = sim.launch(lane_kernel(instruction, bits, where, generic), "k", grid=1, block=32,
                           args=[mem, sim.from_numpy(b), sim.from_numpy(c), out])
        assert not stats.findings
        np.testing.assert_array_equal(out.numpy(), old if opcode == "atom" else 0x5a)
        np.testing.assert_array_equal(mem.numpy(), want)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_every_emitted_form(sim, prefix, operation):
    check_form(sim, "atom", prefix, operation)


@pytest.mark.parametrize("prefix", ["global", "shared", ""])  # the rest are in PREFIXES
@pytest.mark.parametrize("operation", OPERATIONS)
def test_every_form_without_ordering_or_scope(sim, prefix, operation):
    check_form(sim, "atom", prefix, operation)


# nvcc 12.4 turns an atomic function whose result is not used into `red`, for every function
# but atomicExch and atomicCAS (PTX has no red.exch or red.cas). nvcc 13.5 emits no `red`.
@pytest.mark.parametrize("prefix", LEGACY_PREFIXES)
@pytest.mark.parametrize("operation", [o for o in OPERATIONS if o.split(".")[0] not in ("exch", "cas")])
def test_every_emitted_red_form(sim, prefix, operation):
    check_form(sim, "red", prefix, operation)


@pytest.mark.parametrize("prefix", ["relaxed.gpu.global", "relaxed.gpu.shared", "relaxed.gpu"])
@pytest.mark.parametrize("operation", ["add.u32", "min.s64", "xor.b64", "inc.u32"])
def test_red_with_the_ordering_and_scope_written_out(sim, prefix, operation):
    """What the legacy spelling means, as PTX defines `red`; no compiler here emits it."""
    check_form(sim, "red", prefix, operation)


@pytest.mark.parametrize("instruction, why", [
    ("atom.relaxed.gpu.global.sub.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.global.inc.u64 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.global.min.f32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.global.add.f16 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.global.exch.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.acquire.gpu.global.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.shared::cluster.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.local.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.global.cluster.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.global.shared.cta.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.relaxed.gpu.global.add.u32 %d, [%a];", "expected 3 operands"),
    ("atom.relaxed.gpu.global.cas.b32 %d, [%a], %b;", "expected 4 operands"),
    ("atom.relaxed.gpu.global.add.u32 %d, [%a], %b, %c;", "expected 3 operands"),
    ("atom.relaxed.gpu.global.add.u32 %d, %a, %b;", r"expected a \[address\] operand"),
    # Without ordering or scope the same limits hold.
    ("atom.global.sub.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.global.exch.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.local.add.u32 %d, [%a], %b;", "unsupported instruction"),
    ("atom.global.add %d, [%a], %b;", "unsupported instruction"),
    # red: PTX has no exchange or compare-and-swap that returns nothing.
    ("red.global.exch.b32 [%a], %b;", "red has no exch or cas"),
    ("red.global.cas.b32 [%a], %b, %c;", "red has no exch or cas"),
    ("red.relaxed.gpu.shared.exch.b64 [%a], %b;", "red has no exch or cas"),
    ("red.global.sub.u32 [%a], %b;", "unsupported instruction"),
    ("red.global.min.f32 [%a], %b;", "unsupported instruction"),
    ("red.global.inc.u64 [%a], %b;", "unsupported instruction"),
    ("red.global.and.u32 [%a], %b;", "unsupported instruction"),
    ("red.acquire.gpu.global.add.u32 [%a], %b;", "unsupported instruction"),
    ("red.global.cluster.or.b32 [%a], %b;", "unsupported instruction"),
    ("red.local.add.u32 [%a], %b;", "unsupported instruction"),
    ("red.global.min.u32 %d, [%a], %b;", "expected 2 operands"),
    ("red.global.min.u32 [%a];", "expected 2 operands"),
])
def test_other_forms_are_refused(sim, instruction, why):
    old = np.zeros(32, np.uint32)
    with pytest.raises(warplens.UnsupportedError, match=why) as error:
        sim.launch(lane_kernel(instruction, 32, "global"), "k", grid=1, block=32,
                   args=[sim.from_numpy(old), sim.from_numpy(old), sim.from_numpy(old), sim.empty(32, "u32")])
    assert instruction.split()[0] in str(error.value)  # the message quotes the instruction


# -- lanes of one warp on the same address ----------------------------------------------------

def one_word(body: str, space: str = "global"):
    """%a is the address of one u32 that every lane names: mem[0], or a shared copy of it."""
    if space == "global":
        setup, finish = "mov.u64 %a, %p;", ""
    else:
        setup = "ld.global.u32 %v, [%p]; mov.u64 %a, cell; st.shared.u32 [%a], %v;"
        finish = "ld.shared.u32 %v, [%a]; st.global.u32 [%p], %v;"
    return kernel(f"""
        .reg .b64 %p, %o, %off, %a; .reg .b32 %t, %d, %v, %b; .reg .pred %odd;
        .shared .align 4 .u32 cell;
        mov.u32 %t, %tid.x; mul.wide.u32 %off, %t, 4;
        and.b32 %v, %t, 1; setp.ne.u32 %odd, %v, 0;
        add.u32 %b, %t, 1;
        ld.param.u64 %p, [mem];
        {setup}
        mov.u32 %d, 0xdead;
        {body}
        {finish}
        ld.param.u64 %o, [out]; add.u64 %o, %o, %off; st.global.u32 [%o], %d;
        ret;
    """, ".param .u64 mem, .param .u64 out")


def run_one_word(sim, body, space="global", start=0):
    mem, out = sim.from_numpy(np.array([start, 0x55], np.uint32)), sim.empty(32, "u32")
    stats = sim.launch(one_word(body.replace("SPACE", space), space), "k", grid=1, block=32, args=[mem, out])
    assert not stats.findings
    assert mem.numpy()[1] == 0x55
    return int(mem.numpy()[0]), out.numpy().tolist()


@pytest.mark.parametrize("space", ["global", "shared"])
def test_lanes_on_one_word_apply_in_lane_order(sim, space):
    # 32 lanes add 1: each sees what the lanes below it left, so none of the additions is lost.
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.add.u32 %d, [%a], 1;", space, start=100)
    assert final == 132 and old == list(range(100, 132))
    # Lane t adds t + 1.
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.add.u32 %d, [%a], %b;", space)
    assert final == 32 * 33 // 2 and old == [t * (t + 1) // 2 for t in range(32)]
    # Exchange: each lane receives the value of the lane before it; the last lane's stays.
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.exch.b32 %d, [%a], %b;", space, start=77)
    assert final == 32 and old == [77] + list(range(1, 32))
    # Compare-and-swap of an expected 0: lane 0 succeeds, the others find its 1.
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.cas.b32 %d, [%a], 0, %b;", space)
    assert final == 1 and old == [0] + [1] * 31
    # atomicInc with a limit of 9 from 0: 32 increments wrap three times.
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.inc.u32 %d, [%a], 9;", space)
    assert final == 2 and old == [t % 10 for t in range(32)]
    final, old = run_one_word(sim, "atom.relaxed.gpu.SPACE.max.u32 %d, [%a], %b;", space, start=5)
    assert final == 32 and old == [5] * 6 + list(range(6, 32))


@pytest.mark.parametrize("space", ["global", "shared"])
def test_predicated_atom_updates_only_its_lanes(sim, space):
    final, old = run_one_word(sim, "@%odd atom.relaxed.gpu.SPACE.add.u32 %d, [%a], %b;", space)
    assert final == sum(range(2, 33, 2))
    assert old[::2] == [0xdead] * 16  # these lanes did not execute it: no result, no update
    assert old[1::2] == [sum(range(2, t + 1, 2)) for t in range(1, 32, 2)]


def test_some_lanes_share_a_word_and_some_do_not(sim):
    # Lane t updates word t // 4 for t < 16 and word t - 12 above: four lanes each on words
    # 0..3, one lane each on words 4..19.
    mod = kernel("""
        .reg .b64 %p, %o, %off, %w; .reg .b32 %t, %d, %i; .reg .pred %low;
        mov.u32 %t, %tid.x;
        setp.lt.u32 %low, %t, 16;
        shr.u32 %i, %t, 2;
        @!%low sub.u32 %i, %t, 12;
        mul.wide.u32 %w, %i, 4;
        ld.param.u64 %p, [mem]; add.u64 %p, %p, %w;
        atom.relaxed.gpu.global.add.u32 %d, [%p], 10;
        mul.wide.u32 %off, %t, 4;
        ld.param.u64 %o, [out]; add.u64 %o, %o, %off; st.global.u32 [%o], %d;
        ret;
    """, ".param .u64 mem, .param .u64 out")
    mem, out = sim.from_numpy(np.arange(20, dtype=np.uint32)), sim.empty(32, "u32")
    sim.launch(mod, "k", grid=1, block=32, args=[mem, out])
    assert mem.numpy().tolist() == [40, 41, 42, 43] + list(range(14, 30))
    assert out.numpy().tolist() == [t // 4 + 10 * (t % 4) for t in range(16)] + list(range(4, 20))


def test_misaligned_overlapping_atomic_words_are_rejected(sim):
    # A u32 access at byte 2 violates PTX natural alignment; do not fabricate a
    # deterministic result for undefined hardware behavior.
    mod = kernel("""
        .reg .b64 %p, %o, %off; .reg .b32 %t, %d;
        mov.u32 %t, %tid.x; mul.wide.u32 %off, %t, 2;
        ld.param.u64 %p, [mem]; add.u64 %p, %p, %off;
        atom.relaxed.gpu.global.add.u32 %d, [%p], 0x01010101;
        mul.wide.u32 %off, %t, 4;
        ld.param.u64 %o, [out]; add.u64 %o, %o, %off; st.global.u32 [%o], %d;
        ret;
    """, ".param .u64 mem, .param .u64 out")
    mem, out = sim.from_numpy(np.array([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80], np.uint8)), sim.empty(2, "u32")
    with pytest.raises(warplens.ExecutionError, match="aligned to 4 bytes"):
        sim.launch(mod, "k", grid=1, block=2, args=[mem, out])
    assert mem.numpy().tolist() == [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]


# -- warps and CTAs ---------------------------------------------------------------------------

COUNTER = """
    .reg .b64 %p, %o, %off; .reg .b32 %t, %n, %c, %d, %s; .reg .pred %first;
    .shared .align 4 .u32 cell;
    mov.u32 %t, %tid.x; mov.u32 %n, %ntid.x; mov.u32 %c, %ctaid.x;
    setp.eq.u32 %first, %t, 0;
    @%first st.shared.u32 [cell], 0;
    bar.sync 0;
    ld.param.u64 %p, [mem];
    atom.relaxed.gpu.global.add.u32 %d, [%p], 1;
    atom.relaxed.gpu.shared.add.u32 %s, [cell], 1;
    mad.lo.u32 %t, %c, %n, %t;
    mul.wide.u32 %off, %t, 8;
    ld.param.u64 %o, [out]; add.u64 %o, %o, %off;
    st.global.u32 [%o], %d; st.global.u32 [%o+4], %s;
    ret;
"""


def test_no_update_is_lost_across_warps_and_ctas(sim):
    # The scheduler runs one warp instruction at a time, so nothing comes between the read
    # and the write of an atom in another warp or CTA either.
    mem, out = sim.from_numpy(np.zeros(1, np.uint32)), sim.empty((3, 96, 2), "u32")
    stats = sim.launch(kernel(COUNTER, ".param .u64 mem, .param .u64 out"), "k", grid=3, block=96, args=[mem, out])
    assert not stats.findings and mem.numpy()[0] == 3 * 96
    got = out.numpy()
    assert sorted(got[:, :, 0].ravel().tolist()) == list(range(3 * 96))
    for cta in range(3):
        assert sorted(got[cta, :, 1].tolist()) == list(range(96))


# -- the shared-memory race detector ----------------------------------------------------------

def race_kernel(body: str, setup: str = "@%first st.shared.u32 [cell], 0;\n bar.sync 0;"):
    """%w0 / %w1 / %w2: this is warp 0, 1, 2; %first: thread 0. Warps run one instruction
    each in turn, so an earlier line runs in every warp before a later one in any."""
    return kernel(f"""
        .reg .b32 %t, %w, %d, %v; .reg .pred %first, %w0, %w1, %w2;
        .shared .align 4 .u32 cell;
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid;
        setp.eq.u32 %first, %t, 0;
        setp.eq.u32 %w0, %w, 0; setp.eq.u32 %w1, %w, 1; setp.eq.u32 %w2, %w, 2;
        {setup}
        {body}
        ret;
    """)


def races(sim, body, **setup):
    stats = sim.launch(race_kernel(body, **setup), "k", grid=1, block=96)
    return [(f.kind, f.text.strip()) for f in stats.findings]


ATOM = "atom.relaxed.gpu.shared.add.u32 %d, [cell], 1;"


def test_atoms_of_different_warps_do_not_race_with_each_other(sim):
    assert races(sim, ATOM) == []
    assert races(sim, f"{ATOM}\n atom.relaxed.gpu.shared.cas.b32 %d, [cell], 3, 4;\n"
                      "atom.shared.cta.max.u32 %d, [cell], 9;") == []


def test_a_plain_load_races_with_another_warps_atom(sim):
    load = "@%w0 ld.shared.u32 %v, [cell];"
    assert races(sim, f"{ATOM}\n {load}") == [("smem-race", load)]
    assert races(sim, f"{ATOM}\n bar.sync 0;\n {load}") == []  # ordered by the barrier
    # The warp whose atom came last is no exception: the other warps' atoms are unordered too.
    last = "@%w2 ld.shared.u32 %v, [cell];"
    assert races(sim, f"{ATOM}\n {last}") == [("smem-race", last)]


def test_a_plain_store_races_with_another_warps_atom(sim):
    store = "@%w1 st.shared.u32 [cell], 5;"
    assert races(sim, f"{ATOM}\n {store}") == [("smem-race", store)]
    assert races(sim, f"{ATOM}\n bar.sync 0;\n {store}") == []


def test_an_atom_races_with_another_warps_plain_access(sim):
    stats = sim.launch(race_kernel(f"@%w0 st.shared.u32 [cell], 5;\n @!%w0 {ATOM}"), "k", grid=1, block=96)
    (finding,) = stats.findings
    assert finding.kind == "smem-race" and finding.text.strip() == f"@!%w0 {ATOM}" and finding.count == 2
    assert "atomic update" in finding.message and "ordinary load or store" in finding.message
    assert races(sim, f"@%w0 ld.shared.u32 %v, [cell];\n @!%w0 {ATOM}") == [("smem-race", f"@!%w0 {ATOM}")]
    assert races(sim, f"@%w0 st.shared.u32 [cell], 5;\n bar.sync 0;\n @!%w0 {ATOM}") == []
    # The store stays on record when its own warp's atom follows it.
    assert races(sim, f"@%w0 st.shared.u32 [cell], 5;\n @%w0 {ATOM}\n @%w1 {ATOM}") == \
        [("smem-race", f"@%w1 {ATOM}")]


def test_one_warps_own_accesses_around_its_atom_do_not_race(sim):
    assert races(sim, f"@%w1 st.shared.u32 [cell], 5;\n @%w1 {ATOM}\n @%w1 ld.shared.u32 %v, [cell];\n"
                      "@%w1 st.shared.u32 [cell], 6;") == []


def test_atom_on_shared_memory_never_written_is_reported(sim):
    found = races(sim, ATOM, setup="")
    assert found == [("smem-uninitialized-read", ATOM)]  # once: the first atom initialized it
    stats = sim.launch(race_kernel(ATOM, setup=""), "k", grid=1, block=96)
    assert stats.findings[0].count == 1 and "atomic update" in stats.findings[0].message


# -- addresses --------------------------------------------------------------------------------

def test_one_generic_atom_reaches_shared_and_global_memory(sim):
    # Odd lanes name a shared word, even lanes a global one, through generic addresses.
    mod = kernel("""
        .reg .b64 %p, %g, %a, %o, %off; .reg .b32 %t, %d, %v; .reg .pred %odd;
        .shared .align 4 .u32 cell;
        mov.u32 %t, %tid.x; and.b32 %v, %t, 1; setp.ne.u32 %odd, %v, 0;
        st.shared.u32 [cell], 1000;
        ld.param.u64 %p, [mem];
        mov.u64 %g, cell; cvta.shared.u64 %g, %g;
        selp.u64 %a, %g, %p, %odd;
        atom.relaxed.gpu.add.u32 %d, [%a], 1;
        mul.wide.u32 %off, %t, 4;
        ld.param.u64 %o, [out]; add.u64 %o, %o, %off; st.global.u32 [%o], %d;
        ld.shared.u32 %v, [cell]; st.global.u32 [%p+4], %v;
        ret;
    """, ".param .u64 mem, .param .u64 out")
    mem, out = sim.from_numpy(np.array([7, 0], np.uint32)), sim.empty(32, "u32")
    stats = sim.launch(mod, "k", grid=1, block=32, args=[mem, out])
    assert not stats.findings and mem.numpy().tolist() == [7 + 16, 1000 + 16]
    assert out.numpy()[0::2].tolist() == list(range(7, 23))
    assert out.numpy()[1::2].tolist() == list(range(1000, 1016))


def test_generic_atom_on_another_ctas_shared_memory(sim):
    # Both CTAs of a cluster add to CTA 1's `cell`, CTA 0 through a pointer from `mapa`.
    mod = kernel("""
        .reg .b64 %g, %o, %off; .reg .b32 %r, %d, %v; .reg .pred %p;
        .shared .align 4 .u32 cell;
        mov.u32 %r, %cluster_ctarank;
        setp.eq.u32 %p, %r, 1;
        @%p st.shared.u32 [cell], 100;
        @!%p st.shared.u32 [cell], 0;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mov.u64 %g, cell; cvta.shared.u64 %g, %g;
        mapa.u64 %g, %g, 1;
        atom.relaxed.gpu.add.u32 %d, [%g], 1;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ld.shared.u32 %v, [cell];
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %r, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=2, block=32, args=[out], cluster=2)
    assert not stats.findings and out.numpy().tolist() == [0, 164]


@pytest.mark.parametrize("address, what", [
    ("mov.u64 %a, frame; cvta.local.u64 %a, %a;", "local memory"),
    ("mov.u64 %a, mem; cvta.param.u64 %a, %a;", "kernel parameters"),
])
def test_atom_outside_global_and_shared_memory_is_an_error(sim, address, what):
    mod = kernel(f"""
        .reg .b64 %a; .reg .b32 %d;
        .local .align 4 .u32 frame;
        {address}
        atom.relaxed.gpu.add.u32 %d, [%a], 1;
        ret;
    """, ".param .u64 mem")
    with pytest.raises(warplens.ExecutionError, match=f"on {what}: PTX defines atomic operations"):
        sim.launch(mod, "k", grid=1, block=4, args=[sim.empty(1, "u32")])


@pytest.mark.parametrize("instruction, message, lanes", [
    # A word per lane: lanes 2 and 3 are past the end of the 8 bytes.
    ("atom.relaxed.gpu.global.add.u32 %d, [%p], 1;", "outside every allocation", [2, 3]),
    # All active addresses are checked before the first update.
    ("atom.relaxed.gpu.shared.add.u32 %d, [cell+4], 1;", "atomic access of 4 byte", [1, 2, 3]),
])
def test_atom_out_of_bounds_faults_with_its_lanes(sim, instruction, message, lanes):
    mod = kernel(f"""
        .reg .b64 %p, %off; .reg .b32 %d, %t; .reg .pred %q;
        .shared .align 4 .u32 cell;
        mov.u32 %t, %tid.x; setp.ge.u32 %q, %t, 1;
        mul.wide.u32 %off, %t, 4;
        ld.param.u64 %p, [mem]; add.u64 %p, %p, %off;
        @%q {instruction}
        ret;
    """, ".param .u64 mem")
    mem = sim.from_numpy(np.zeros(2, np.uint32))
    with pytest.raises(warplens.MemoryFault, match=message) as error:
        sim.launch(mod, "k", grid=1, block=4, args=[mem])
    assert error.value.lanes == lanes and mem.numpy()[0] == 0


# -- TMA's view of an atom's write ------------------------------------------------------------

@pytest.mark.parametrize("fenced", [False, True])
def test_atom_is_an_ordinary_write_for_a_tma_store(sim, fenced):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 %d, %buf;
        .shared .align 16 .b8 buffer[16];
        ld.param.u64 %out, [out];
        mov.u32 %buf, buffer;
        st.shared.u32 [buffer], 1; st.shared.u32 [buffer+4], 2;
        st.shared.u32 [buffer+8], 3; st.shared.u32 [buffer+12], 4;
        fence.proxy.async.shared::cta;
        bar.sync 0;
        atom.relaxed.gpu.shared.add.u32 %d, [buffer+4], 40;
    """ + ("fence.proxy.async.shared::cta;" if fenced else "") + """
        cp.async.bulk.global.shared::cta.bulk_group [%out], [%buf], 16;
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
        ret;
    """, ".param .u64 out")
    out = sim.empty(4, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert [f.kind for f in stats.findings] == ([] if fenced else ["tma-store-missing-fence"])
    assert out.numpy().tolist() == [1, 42, 3, 4]


# -- timing and bank statistics ---------------------------------------------------------------

def timed(sim, access: str, block: int = 32):
    mod = kernel(f"""
        .reg .b64 %p, %off; .reg .b32 %t, %d, %a;
        .shared .align 4 .u32 cell[64];
        mov.u32 %t, %tid.x; mul.wide.u32 %off, %t, 4;
        shl.b32 %a, %t, 2;
        ld.param.u64 %p, [mem]; add.u64 %p, %p, %off;
        st.shared.u32 [%a], 0; st.shared.u32 [%a+128], 0;
        bar.sync 0;
        {access}
        ret;
    """, ".param .u64 mem")
    mem = sim.from_numpy(np.zeros(block, np.uint32))
    stats = sim.launch(mod, "k", grid=1, block=block, args=[mem], timing=True)
    assert not stats.findings
    return stats, mem.numpy()


@pytest.mark.parametrize("atom, load_store", [
    ("atom.relaxed.gpu.global.add.u32 %d, [%p], 1;", "ld.global.u32 %d, [%p]; st.global.u32 [%p], 1;"),
    ("atom.relaxed.gpu.shared.add.u32 %d, [%a], 1;", "ld.shared.u32 %d, [%a]; st.shared.u32 [%a], 1;"),
    # Two words in one bank: two wavefronts for the load, two for the store, four for the atom.
    ("atom.relaxed.gpu.shared.add.u32 %d, [%a], 1;".replace("%a", "%bank"),
     "ld.shared.u32 %d, [%bank]; st.shared.u32 [%bank], 1;"),
])
def test_an_atom_is_charged_as_a_load_and_a_store(sim, atom, load_store):
    bank = ".reg .b32 %bank; and.b32 %bank, %t, 1; shl.b32 %bank, %bank, 7;"  # two words, same bank
    (one, memory), (two, _) = timed(sim, bank + atom), timed(sim, bank + load_store)
    assert one.timing.time_ns == two.timing.time_ns
    assert one.instructions == two.instructions - 1
    if "global" in atom:
        assert memory.tolist() == [1] * 32
        plain, _ = timed(sim, bank + "st.global.u32 [%p], 1;")
        assert one.timing.time_ns > plain.timing.time_ns  # the wait a global load costs


def test_atom_in_the_bank_report(sim):
    stats, _ = timed(sim, "atom.relaxed.gpu.shared.add.u32 %d, [%a], 1;")
    (entry,) = [e for e in stats.smem.instructions.values() if e.op == "atom"]
    assert (entry.executions, entry.wavefronts, entry.conflicts) == (1, 1, 0)  # counted once
