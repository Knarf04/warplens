"""Portable PTX checks for stale mbarrier state and numeric edge cases."""

import numpy as np
import pytest

from tests.support import kernel, run_lanes


@pytest.mark.parametrize("phase", [2, 3])
@pytest.mark.parametrize("wait", ["test_wait", "try_wait"])
@pytest.mark.parametrize("semantics", ["acquire", "relaxed"])
def test_stale_state_wait_is_reported_without_acquiring_newer_data(sim, wait, semantics, phase):
    # At phase 3 the stale token's fallback is true; even then it must not acquire
    # the newer TMA completion. Phase 2 exercises the false fallback too.
    extra_phase = ("mbarrier.arrive.shared::cta.b64 %advance, [bar]; "
                   "mbarrier.test_wait.shared::cta.b64 %p, [bar], %advance;" if phase == 3 else "")
    mod = kernel(f"""
        .reg .b64 %src, %old, %advance; .reg .b32 %value; .reg .pred %p;
        .shared .align 16 .b8 data[16]; .shared .align 8 .b64 bar;
        ld.param.u64 %src, [src];
        mbarrier.init.shared::cta.b64 [bar], 1;
        mbarrier.arrive.shared::cta.b64 %old, [bar];
        mbarrier.test_wait.shared::cta.b64 %p, [bar], %old;
        {extra_phase}
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 16;
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [data], [%src], 16, [bar];
        mbarrier.{wait}.{semantics}.cta.shared::cta.b64 %p, [bar], %old;
        ld.shared.u32 %value, [data];
        ret;
    """, ".param .u64 src")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[sim.from_numpy(np.arange(4, dtype=np.uint32))],
                       async_delay=0)
    kinds = [finding.kind for finding in stats.findings]
    assert "mbarrier-invalid-token" in kinds
    assert "tma-load-unsynchronized" in kinds
    invalid = next(f for f in stats.findings if f.kind == "mbarrier-invalid-token")
    assert "phase 0" in invalid.message and f"phase {phase}" in invalid.message
    assert "no synchronization" in invalid.message


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_current_and_previous_phase_tokens_remain_valid(sim, phase):
    advance = ("mbarrier.arrive.shared::cta.b64 %token, [bar]; "
               "mbarrier.arrive.shared::cta.b64 _, [bar]; "
               "mbarrier.test_wait.shared::cta.b64 %p, [bar], %token;") * phase
    mod = kernel(f"""
        .reg .b64 %out, %token; .reg .b32 %v; .reg .pred %p;
        .shared .align 8 .b64 bar;
        ld.param.u64 %out, [out];
        mbarrier.init.shared::cta.b64 [bar], 2;
        {advance}
        mbarrier.arrive.shared::cta.b64 %token, [bar];
        mbarrier.test_wait.shared::cta.b64 %p, [bar], %token;
        selp.u32 %v, 1, 0, %p; st.global.u32 [%out], %v;
        mbarrier.arrive.shared::cta.b64 _, [bar];
        mbarrier.try_wait.shared::cta.b64 %p, [bar], %token;
        selp.u32 %v, 1, 0, %p; st.global.u32 [%out+4], %v;
        mbarrier.arrive.shared::cta.b64 _, [bar];
        mbarrier.test_wait.shared::cta.b64 %p, [bar], %token;
        selp.u32 %v, 1, 0, %p; st.global.u32 [%out+8], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(3, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [0, 1, 1]
    assert not stats.findings


@pytest.mark.parametrize("ty", ["s16", "u16", "s32", "u32", "s64", "u64"])
@pytest.mark.parametrize("operation", ["div", "rem"])
@pytest.mark.parametrize("guard", ["lt", "eq"])
def test_integer_zero_divisor_finding_respects_active_lanes(sim, ty, operation, guard):
    bits = int(ty[1:])
    size = bits // 8
    signed = ty.startswith("s")
    dtype = np.dtype(f"{'int' if signed else 'uint'}{bits}")
    x = np.array([7, -7 if signed else 7, 99, 23], dtype=dtype)
    y = np.array([0, 2, 0, 0], dtype=dtype)
    mod = kernel(f"""
        .reg .b64 %a, %b, %out, %offset;
        .reg .b32 %tid; .reg .b{bits} %x, %y, %result; .reg .pred %p;
        ld.param.u64 %a, [a]; ld.param.u64 %b, [b]; ld.param.u64 %out, [out];
        mov.u32 %tid, %tid.x;
        mul.wide.u32 %offset, %tid, {size};
        add.u64 %a, %a, %offset; add.u64 %b, %b, %offset; add.u64 %out, %out, %offset;
        ld.global.{ty} %x, [%a]; ld.global.{ty} %y, [%b];
        mov.b{bits} %result, 42;
        setp.{guard}.u32 %p, %tid, {2 if guard == 'lt' else 1};
        @%p {operation}.{ty} %result, %x, %y;
        st.global.{ty} [%out], %result;
        ret;
    """, ".param .u64 a, .param .u64 b, .param .u64 out")
    out = sim.empty(4, ty)
    stats = sim.launch(mod, "k", grid=1, block=4, args=[sim.from_numpy(x), sim.from_numpy(y), out])
    assert out.numpy()[1] == ((-3 if signed else 3) if operation == "div" else (-1 if signed else 1))
    assert out.numpy()[2:].tolist() == [42, 42]
    if guard == "lt":
        finding, = stats.findings
        assert finding.kind == "integer-divide-by-zero"
        assert finding.lanes == [0]
        assert "unspecified" in finding.message
    else:
        assert not stats.findings


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "fma", "sqrt"])
def test_ftz_nan_results_are_nan_without_requiring_a_payload(sim, operation):
    # PTX specifies the NaN classification, not f32 sign/payload bits.
    values = {
        "add": ([np.inf], [-np.inf]), "sub": ([np.inf], [np.inf]),
        "mul": ([np.inf], [0.0]), "div": ([0.0], [0.0]),
        "fma": ([np.inf], [0.0], [1.0]), "sqrt": ([-1.0],),
    }[operation]
    operands = ", ".join(f"%a{i}" for i in range(len(values)))
    result = run_lanes(sim, f"{operation}.rn.ftz.f32 %d, {operands};", ["f32"] * len(values), "f32",
                       [np.asarray(value, dtype=np.float32) for value in values])
    assert np.isnan(result).all()
