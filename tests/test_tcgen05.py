"""tcgen05 MMA, tensor memory, and their checkers, on a hand-written kernel.

The test places operands A and B in shared memory itself, following CuTe's
canonical K-major layouts (cute/arch/mma_sm100_desc.hpp), builds the two
descriptors, and compares the accumulators read back from TMEM with NumPy.
"""

import numpy as np
import pytest

import warplens

from conftest import kernel

LAYOUTS = {"none": (0, 0, 16), "32B": (6, 1, 32), "64B": (4, 2, 64), "128B": (2, 3, 128)}
PARAMS = (".param .u64 img, .param .u64 out, .param .u64 da, .param .u64 db, .param .u32 idesc, "
          ".param .u32 nbytes, .param .u32 ncols, .param .u32 n, .param .u32 twice, "
          ".param .u32 wait, .param .u32 shift, .param .u32 free")


def mma_kernel(kind="f16"):
    return kernel(f"""
        .reg .b64 %img, %out, %da, %db, %off;
        .reg .b32 %idesc, %nb, %nc, %n, %twice, %wait, %shift, %free;
        .reg .b32 %t, %w, %buf, %bar, %tp, %base, %taddr, %i, %v, %x;
        .reg .pred %p, %one, %two, %q, %no, %yes, %dowait, %dofree;
        .shared .align 1024 .b8 buf[49152];
        .shared .align 8 .u64 bar;
        .shared .align 4 .u32 tptr;
        ld.param.u64 %img, [img]; ld.param.u64 %out, [out];
        ld.param.u64 %da, [da]; ld.param.u64 %db, [db]; ld.param.u32 %idesc, [idesc];
        ld.param.u32 %nb, [nbytes]; ld.param.u32 %nc, [ncols]; ld.param.u32 %n, [n];
        ld.param.u32 %twice, [twice]; ld.param.u32 %wait, [wait];
        ld.param.u32 %shift, [shift]; ld.param.u32 %free, [free];
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid;
        setp.eq.u32 %one, %t, 0; setp.eq.u32 %q, %w, 0;
        setp.ne.u32 %no, %t, %t; setp.eq.u32 %yes, %t, %t;
        setp.ne.u32 %two, %twice, 0; and.pred %two, %two, %one;
        setp.ne.u32 %dowait, %wait, 0; setp.ne.u32 %dofree, %free, 0; and.pred %dofree, %dofree, %q;
        mov.b32 %buf, buf; mov.b32 %bar, bar; mov.b32 %tp, tptr;
        @%one mbarrier.init.shared::cta.b64 [%bar], 1;
        @%q tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%tp], %nc;
        bar.sync 0;
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], %nb;
        @%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%img], %nb, [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        ld.shared.u32 %base, [%tp];
        @%one tcgen05.mma.cta_group::1.kind::{kind} [%base], %da, %db, %idesc, {{0, 0, 0, 0}}, %no;
        @%two tcgen05.mma.cta_group::1.kind::{kind} [%base], %da, %db, %idesc, {{0, 0, 0, 0}}, %yes;
        @%one tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%bar];
        @%dowait mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 1;
        shl.b32 %x, %w, 21;
        add.u32 %taddr, %base, %x;
        add.u32 %taddr, %taddr, %shift;
        mov.u32 %i, 0;
      LOOP:
        add.u32 %x, %taddr, %i;
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {{%v}}, [%x];
        mad.lo.u32 %x, %t, %n, %i;
        mul.wide.u32 %off, %x, 4;
        add.u64 %off, %out, %off;
        st.global.u32 [%off], %v;
        add.u32 %i, %i, 1;
        setp.lt.u32 %p, %i, %n;
        @%p bra LOOP;
        bar.sync 0;
        @%dofree tcgen05.dealloc.cta_group::1.sync.aligned.b32 %base, %nc;
        ret;
    """, PARAMS)


def place(matrix, layout, start, image):
    """Write a K-major operand (rows x K) into `image` the way CuTe lays it out; return its descriptor."""
    code, bits, row_bytes = LAYOUTS[layout]
    rows, k = matrix.shape
    size = matrix.dtype.itemsize
    per_chunk = 16 // size
    assert k == 2 * per_chunk
    sbo = 8 * row_bytes
    lbo = -(-rows // 8) * sbo if not bits else 16  # unswizzled: second K chunk after all rows
    raw = matrix.view(np.uint8).reshape(rows, k, size)
    for r in range(rows):
        for j in range(k):
            off = start + (r % 8) * row_bytes + (r // 8) * sbo + (j % per_chunk) * size + \
                (j // per_chunk) * (16 if bits else lbo)
            off ^= ((off >> 7) & ((1 << bits) - 1)) << 4  # Swizzle<bits, 4, 3>
            image[off:off + size] = raw[r, j]
    return (start >> 4) | ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (code << 61)


def run(sim, m=128, n=32, layout="128B", dtype=np.float16, twice=0, wait=1, shift=0, free=1,
        negate_a=False, ncols=None, seed=0, block=128):
    rng = np.random.default_rng(seed)
    k = 32 // np.dtype(dtype).itemsize
    a = rng.uniform(-2, 2, (m, k)).astype(dtype)
    b = rng.uniform(-2, 2, (n, k)).astype(dtype)
    image = np.zeros(32768, dtype=np.uint8)
    desc_a = place(a, layout, 0, image)
    desc_b = place(b, layout, 16384, image)
    fmt = 0 if dtype == np.float16 else 2
    idesc = (1 << 4) | (fmt << 7) | (fmt << 10) | (int(negate_a) << 13) | ((n >> 3) << 17) | ((m >> 4) << 24)
    out = sim.from_numpy(np.full((128, n), np.nan, dtype=np.float32))
    stats = sim.launch(mma_kernel("f16" if dtype == np.float16 else "tf32"), "k", grid=1, block=block,
                       args=[sim.from_numpy(image), out, desc_a, desc_b, idesc, len(image),
                             ncols or max(32, 1 << (n - 1).bit_length()), n, twice, wait, shift, free])
    expect = a.astype(np.float32) @ b.astype(np.float32).T * (-1 if negate_a else 1) * (2 if twice else 1)
    return out.numpy(), expect, stats


@pytest.mark.parametrize("layout", ["none", "32B", "64B", "128B"])
@pytest.mark.parametrize("n", [8, 32, 96])
def test_mma_128_rows_for_every_operand_layout(sim, layout, n):
    got, expect, stats = run(sim, m=128, n=n, layout=layout)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)  # row m is TMEM lane m
    assert not stats.findings


def test_mma_64_rows_use_the_low_16_lanes_of_each_warp(sim):
    got, expect, stats = run(sim, m=64, n=32, wait=1)
    lanes = 32 * (np.arange(64) // 16) + np.arange(64) % 16
    np.testing.assert_allclose(got[lanes], expect, rtol=1e-5, atol=1e-5)
    # The upper 16 lanes of each warp were never written.
    assert [f.kind for f in stats.findings] == ["tmem-uninitialized-read"]
    assert sorted(set(stats.findings[0].lanes)) == list(range(16, 32))


def test_accumulate_negate_and_tf32(sim):
    got, expect, stats = run(sim, twice=1)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    got, expect, _ = run(sim, negate_a=True)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    got, expect, stats = run(sim, dtype=np.float32, n=16)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    assert not stats.findings


def test_reading_accumulators_without_waiting_for_the_commit_is_caught(sim):
    got, expect, stats = run(sim, wait=0)
    kinds = {f.kind for f in stats.findings}
    assert "tmem-read-before-commit" in kinds
    first = next(f for f in stats.findings if f.kind == "tmem-read-before-commit")
    assert first.text.startswith("tcgen05.ld") and "tcgen05.mma at line" in first.message


def test_lane_rule(sim):
    # Three warps, each addressing the lanes of the next one.
    _, _, stats = run(sim, m=128, n=8, shift=32 << 16, ncols=32, block=96)
    f = next(f for f in stats.findings if f.kind == "tmem-lane-rule")
    assert "warp 0 may only access tensor-memory lanes 0-31, but this reaches lane 32" in f.message


def test_leaked_allocation_is_reported(sim):
    _, _, stats = run(sim, free=0)
    (f,) = stats.findings
    assert f.kind == "tmem-leak" and f.text.startswith("@%q tcgen05.alloc") and "32 tensor-memory columns" in f.message


def test_columns_outside_the_allocation_are_reported(sim):
    _, _, stats = run(sim, n=64, ncols=32)
    assert "tmem-unallocated" in {f.kind for f in stats.findings}


def test_allocation_size_must_be_a_power_of_two_of_at_least_32(sim):
    for bad in (16, 48, 1024):
        with pytest.raises(warplens.ExecutionError, match=f"tcgen05.alloc of {bad} columns"):
            run(sim, ncols=bad)


def test_unsupported_tcgen05_forms_are_errors(sim):
    for line, why in [
        ("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%a], 32;", "cta_group::2"),
        ("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%a}, [%a];", "only the 32x32b shape"),
        ("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%a}, [%a];", "only the 32x32b shape"),
        ("tcgen05.cp.4x256b.1x64b [%a], %d;", "tcgen05.cp"),
    ]:
        mod = kernel(f".reg .b32 %a; .reg .b64 %d;\n.shared .align 4 .u32 cell;\nmov.b32 %a, cell;\n{line}\nret;")
        with pytest.raises(warplens.UnsupportedError, match=why):
            sim.launch(mod, "k", grid=1, block=32)
