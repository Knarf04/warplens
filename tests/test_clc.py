"""Cluster launch control: `clusterlaunchcontrol.try_cancel` and `.query_cancel`, the `.b128`
registers that carry a response, and the launch queue that decides what a request cancels.

The spellings are those of the PTX ISA 9.5 manual (sections 9.7.15.18 and 9.7.15.19), CUTLASS's
`sm100_tile_scheduler.hpp` and what nvcc 13.5 emits; `test_the_kernels_assemble` runs them
through ptxas when it is installed. What happens on a GPU has not been observed: see docs/clc.md.
"""

import shutil
import subprocess

import numpy as np
import pytest

import warplens

from conftest import HEADER

TRY_CANCEL = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
QUERY = "clusterlaunchcontrol.query_cancel"
DECLARE = """
    .shared .align 16 .b8 smem[64];
    .reg .b32 %x, %y, %z, %me, %bar, %resp, %ph, %v, %n, %t;
    .reg .b64 %o, %a, %off;
    .reg .pred %p, %c;
    .reg .b128 %h;
"""
SETUP = """
    mov.u32 %x, %ctaid.x;
    mov.u32 %me, %ctaid.x;
    mov.u32 %resp, smem;
    add.u32 %bar, %resp, 16;
    mbarrier.init.shared::cta.b64 [%bar], 1;
    mov.u32 %ph, 0;
    ld.param.u64 %o, [out];
"""
REQUEST = f"""
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 16;
    {TRY_CANCEL} [%resp], [%bar];
"""
WORK = """
    mul.wide.u32 %off, %x, 8;
    add.u64 %a, %o, %off;
    add.u32 %v, %me, 1;
    st.global.u32 [%a], %v;
    ld.global.u32 %n, [%a+4];
    add.u32 %n, %n, 1;
    st.global.u32 [%a+4], %n;
"""
WAIT = """
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], %ph;
    @!%p bra WAIT;
    xor.b32 %ph, %ph, 1;
"""
LOAD = "ld.shared.b128 %h, [%resp];"
# A persistent worker, one thread per CTA: do the tile of coordinate x, then take over a
# cluster that has not started, until a request fails. out[x] = (worker's ctaid.x + 1, visits).
WORKER = f"""{DECLARE}{SETUP}
LOOP:{REQUEST}{WORK}{WAIT}
    {LOAD}
    {QUERY}.is_canceled.pred.b128 %c, %h;
    @!%c bra DONE;
    {QUERY}.get_first_ctaid::x.b32.b128 %x, %h;
    fence.proxy.async.shared::cta;
    bra LOOP;
DONE:
    ret;
"""
# One request, then store what `{tail}` left in %c and %x: out = (canceled, x).
ONCE = f"""{DECLARE}{SETUP}{{request}}{WAIT}
    {{load}}
    {{query}}
    mov.u32 %v, 0;
    @%c mov.u32 %v, 1;
    st.global.u32 [%o], %v;
    st.global.u32 [%o+4], %x;
    ret;
"""
QUERY_BOTH = f"""{QUERY}.is_canceled.pred.b128 %c, %h;
    @%c {QUERY}.get_first_ctaid::x.b32.b128 %x, %h;"""
# 3-D coordinates with the .v4 form; the ::x/::y/::z forms must agree with it.
# out[z][y][x] = (worker's linear CTA index + 1, visits).
WORKER_3D = f"""{DECLARE}
    .reg .b32 %nx, %ny, %lin, %x2, %y2, %z2;
    .reg .pred %same;
{SETUP}
    mov.u32 %y, %ctaid.y;
    mov.u32 %z, %ctaid.z;
    mov.u32 %nx, %nctaid.x;
    mov.u32 %ny, %nctaid.y;
    mad.lo.u32 %me, %z, %ny, %y;
    mad.lo.u32 %me, %me, %nx, %x;
LOOP:{REQUEST}
    mad.lo.u32 %lin, %z, %ny, %y;
    mad.lo.u32 %lin, %lin, %nx, %x;
    mul.wide.u32 %off, %lin, 8;
    add.u64 %a, %o, %off;
    add.u32 %v, %me, 1;
    st.global.u32 [%a], %v;
    ld.global.u32 %n, [%a+4];
    add.u32 %n, %n, 1;
    st.global.u32 [%a+4], %n;
{WAIT}
    {LOAD}
    {QUERY}.is_canceled.pred.b128 %c, %h;
    @!%c bra DONE;
    {QUERY}.get_first_ctaid.v4.b32.b128 {{%x, %y, %z, _}}, %h;
    {QUERY}.get_first_ctaid::x.b32.b128 %x2, %h;
    {QUERY}.get_first_ctaid::y.b32.b128 %y2, %h;
    {QUERY}.get_first_ctaid::z.b32.b128 %z2, %h;
    setp.ne.u32 %same, %x, %x2;
    @%same bra BAD;
    setp.ne.u32 %same, %y, %y2;
    @%same bra BAD;
    setp.ne.u32 %same, %z, %z2;
    @%same bra BAD;
    fence.proxy.async.shared::cta;
    bra LOOP;
BAD:
    st.global.u32 [%o], 0xdead;
DONE:
    ret;
"""
# The shape of the manual's example (9.7.15.18): clusters of two CTAs, one thread each. Every
# CTA expects 16 bytes on its own mbarrier; the first CTA of the cluster makes the request,
# and with .multicast::cluster::all both receive the response. The instruction is spelled as
# CUTLASS spells it.
# out[x] = (cluster's first ctaid.x + 1, visits).
MULTICAST = f"""{DECLARE}
    .reg .b32 %rank, %first;
    .reg .pred %leader;
{SETUP}
    mov.u32 %rank, %cluster_ctaid.x;
    sub.u32 %first, %x, %rank;
    add.u32 %me, %first, 0;
    setp.eq.u32 %leader, %rank, 0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
LOOP:
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 16;
    @%leader {TRY_CANCEL.replace('.b128', '.multicast::cluster::all.b128')} [%resp], [%bar];
    add.u32 %x, %first, %rank;
{WORK}{WAIT}
    {LOAD}
    {QUERY}.is_canceled.pred.b128 %c, %h;
    @!%c bra DONE;
    {QUERY}.get_first_ctaid.v4.b32.b128 {{%first, _, _, _}}, %h;
    fence.proxy.async.shared::cta;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    bra LOOP;
DONE:
    ret;
"""
KERNELS = {"worker": WORKER, "worker_3d": WORKER_3D, "multicast": MULTICAST,
           "once": ONCE.format(request=REQUEST, load=LOAD, query=QUERY_BOTH)}


def source(body: str) -> str:
    return f"{HEADER}.visible .entry k(.param .u64 out)\n{{\n{body}\n}}\n"


def module(body: str):
    return warplens.parse_ptx(source(body), name="test.ptx")


def once(sim, *, request=REQUEST, load=LOAD, query=QUERY_BOTH, grid=2, **launch):
    out = sim.zeros(2, "u32")
    stats = sim.launch(module(ONCE.format(request=request, load=load, query=query)), "k", grid=grid,
                       block=1, args=[out], **launch)
    return out.numpy().tolist(), stats


def kinds(stats) -> list[str]:
    return [f.kind for f in stats.findings]


# -- what a request cancels ------------------------------------------------------------------

@pytest.mark.parametrize("limit, workers", [
    (None, [1] * 12),  # the first CTA takes over every other one
    (0, list(range(1, 13))),  # no request succeeds: every CTA starts and does its own tile
    (1, [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11]),
    (2, [1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10]),
    (5, [1] * 6 + [7] * 6),
])
def test_every_coordinate_is_done_exactly_once(sim, limit, workers):
    out = sim.zeros((12, 2), "u32")
    stats = sim.launch(module(WORKER), "k", grid=12, block=1, args=[out], clc_limit=limit)
    assert out.numpy()[:, 0].tolist() == workers
    assert out.numpy()[:, 1].tolist() == [1] * 12
    assert stats.ctas == 12 and stats.canceled_ctas == 12 - len(set(workers))
    assert stats.findings == []


def test_a_canceled_cta_never_runs(sim):
    seen = set()
    sim.launch(module(WORKER), "k", grid=6, block=1, args=[sim.zeros((6, 2), "u32")],
               trace=lambda warp, pc, ins, active: seen.add(warp.cta.id))
    assert seen == {(0, 0, 0)}


def test_first_ctaid_in_three_dimensions(sim):
    out = sim.zeros((2, 3, 4, 2), "u32")  # [z][y][x]
    stats = sim.launch(module(WORKER_3D), "k", grid=(4, 3, 2), block=1, args=[out], clc_limit=4)
    got = out.numpy().reshape(24, 2)
    # Clusters are taken in launch order, x fastest: each starter does itself and the next four.
    assert got[:, 0].tolist() == [1 + 5 * (i // 5) for i in range(24)]
    assert got[:, 1].tolist() == [1] * 24
    assert stats.canceled_ctas == 19 and stats.findings == []


def test_a_request_cancels_a_whole_cluster_and_multicast_reaches_every_cta(sim):
    out = sim.zeros((8, 2), "u32")
    stats = sim.launch(module(MULTICAST), "k", grid=8, block=1, args=[out], cluster=2, clc_limit=1)
    # Clusters start at CTAs 0, 2, 4, 6; the first and third each take over the next one.
    assert out.numpy()[:, 0].tolist() == [1, 1, 1, 1, 5, 5, 5, 5]
    assert out.numpy()[:, 1].tolist() == [1] * 8
    assert stats.canceled_ctas == 4 and stats.findings == []


def test_without_multicast_only_the_requesting_cta_gets_the_response(sim):
    body = MULTICAST.replace(".multicast::cluster::all", "")
    with pytest.raises(warplens.DeadlockError, match="still needs 0 arrival.s. and 16 byte"):
        sim.launch(module(body), "k", grid=4, block=1, args=[sim.zeros((4, 2), "u32")], cluster=2)


def test_the_last_request_fails(sim):
    assert once(sim, grid=1)[0] == [0, 0]
    got, stats = once(sim, grid=2)  # CTA 0 cancels CTA 1, which would have written (0, 1)
    assert got == [1, 1] and stats.canceled_ctas == 1 and stats.findings == []


def test_generic_addresses_when_no_state_space_is_given(sim):
    request = """
        .reg .b64 %g, %gb;
        cvta.shared.u64 %g, smem;
        add.u64 %gb, %g, 16;
        mbarrier.arrive.expect_tx.b64 _, [%gb], 16;
        clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128 [%g], [%gb];
    """
    assert once(sim, request=request)[0] == [1, 1]
    for operands in ("[%o], [%gb]", "[%g], [%o]"):  # a global pointer is not shared memory
        with pytest.raises(warplens.ExecutionError, match="generic address outside shared memory"):
            once(sim, request=request.replace("[%g], [%gb]", operands))


def test_each_thread_makes_its_own_request(sim):
    # Two threads, two response slots, one mbarrier that expects both arrivals and 32 bytes.
    body = f"""
        .shared .align 16 .b8 smem[64];
        .reg .b32 %t, %resp, %bar, %x; .reg .b64 %o, %off; .reg .pred %p, %c; .reg .b128 %h;
        mov.u32 %t, %tid.x;
        mov.u32 %bar, smem;
        add.u32 %bar, %bar, 32;
        mov.u32 %resp, smem;
        shl.b32 %x, %t, 4;
        add.u32 %resp, %resp, %x;
        mbarrier.init.shared::cta.b64 [%bar], 2;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 16;
        {TRY_CANCEL} [%resp], [%bar];
    WAIT:
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        @!%p bra WAIT;
        ld.shared.b128 %h, [%resp];
        {QUERY}.is_canceled.pred.b128 %c, %h;
        mov.u32 %x, 99;
        @%c {QUERY}.get_first_ctaid::x.b32.b128 %x, %h;
        ld.param.u64 %o, [out];
        mul.wide.u32 %off, %t, 4;
        add.u64 %o, %o, %off;
        st.global.u32 [%o], %x;
        ret;
    """
    out = sim.zeros(2, "u32")
    stats = sim.launch(module(body), "k", grid=3, block=2, args=[out])
    assert out.numpy().tolist() == [1, 2]  # no two successful responses name the same cluster
    assert stats.canceled_ctas == 2
    assert "clc-canceled-cluster-lost" not in kinds(stats)


def test_timed_launch(sim):
    out = sim.zeros((6, 2), "u32")
    stats = sim.launch(module(WORKER), "k", grid=6, block=1, args=[out], timing=True)
    assert out.numpy().tolist() == [[1, 1]] * 6 and stats.findings == []


# -- findings --------------------------------------------------------------------------------

def test_first_ctaid_of_a_failed_request_is_reported(sim):
    query = f"{QUERY}.is_canceled.pred.b128 %c, %h;\n{QUERY}.get_first_ctaid::x.b32.b128 %x, %h;"
    got, stats = once(sim, grid=1, query=query)
    assert got == [0, 0] and kinds(stats) == ["clc-ctaid-of-failed-cancel"]
    query = f"{QUERY}.is_canceled.pred.b128 %c, %h;\n{QUERY}.get_first_ctaid.v4.b32.b128 {{%x, _, _, _}}, %h;"
    assert kinds(once(sim, grid=1, query=query)[1]) == ["clc-ctaid-of-failed-cancel"]


def test_another_request_after_a_failure_is_reported(sim):
    body = WORKER.replace("@!%c bra DONE;", "setp.eq.u32 %p, %ph, 0;\n    @%p bra DONE;\n    @!%c bra LOOP;")
    out = sim.zeros((1, 2), "u32")
    stats = sim.launch(module(body), "k", grid=1, block=1, args=[out])  # fails, asks again, fails
    assert kinds(stats) == ["clc-try-cancel-after-failure"]
    assert out.numpy().tolist() == [[1, 2]]


def test_a_canceled_cluster_nobody_takes_over_is_reported(sim):
    got, stats = once(sim, query=f"{QUERY}.is_canceled.pred.b128 %c, %h;")
    assert got == [1, 0] and kinds(stats) == ["clc-canceled-cluster-lost"]
    assert "first CTA is (1, 0, 0)" in stats.findings[0].message


def test_exit_with_the_request_in_flight_is_reported(sim):
    body = f"{DECLARE}{SETUP}{REQUEST}\n    ret;"
    stats = sim.launch(module(body), "k", grid=2, block=1, args=[sim.zeros(2, "u32")])
    assert kinds(stats) == ["clc-not-waited", "clc-canceled-cluster-lost"]


def test_reading_the_response_without_waiting_is_reported(sim):
    body = ONCE.format(request=REQUEST, load=LOAD, query=QUERY_BOTH).replace(WAIT, "")
    out = sim.zeros(2, "u32")
    stats = sim.launch(module(body), "k", grid=2, block=1, args=[out])
    assert "clc-response-unsynchronized" in kinds(stats)
    assert "clc-query-not-a-response" in kinds(stats)  # the bytes read are not the response yet
    assert out.numpy().tolist() == [0, 0]


def test_a_second_request_into_a_response_still_in_flight_is_reported(sim):
    request = (f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 32;\n"
               + f"{TRY_CANCEL} [%resp], [%bar];\n" * 2)
    _, stats = once(sim, request=request, grid=3)
    assert "clc-response-unsynchronized" in kinds(stats)


def test_a_value_that_is_not_a_response_is_reported(sim):
    load = "mov.b128 %h, 0;"
    got, stats = once(sim, load=load)
    assert got == [0, 0] and "clc-query-not-a-response" in kinds(stats)


def test_missing_expect_tx_is_an_mbarrier_finding(sim):
    request = REQUEST.replace("arrive.expect_tx.shared::cta.b64 _, [%bar], 16", "arrive.shared::cta.b64 _, [%bar]")
    _, stats = once(sim, request=request)  # the wait passes at once; the 16 bytes arrive unannounced
    assert "mbarrier-tx-mismatch" in kinds(stats)


def test_multicast_after_a_cta_of_the_cluster_exited_is_reported(sim):
    body = f"""{DECLARE}
        .reg .pred %peer;
        {SETUP}
        barrier.cluster.arrive;
        barrier.cluster.wait;
        setp.ne.u32 %peer, %x, 0;
        @%peer bra DONE;
        mov.u32 %v, 0; mov.u32 %v, 0; mov.u32 %v, 0; mov.u32 %v, 0;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 16;
        {TRY_CANCEL.replace('.b128', '.multicast::cluster::all.b128')} [%resp], [%bar];
        {WAIT}
    DONE:
        ret;
    """
    stats = sim.launch(module(body), "k", grid=2, block=1, args=[sim.zeros(2, "u32")], cluster=2)
    assert "clc-multicast-to-exited-cta" in kinds(stats)


# -- errors ----------------------------------------------------------------------------------

def test_address_errors(sim):
    with pytest.raises(warplens.ExecutionError, match="not aligned to 16 bytes"):
        once(sim, request=REQUEST.replace("[%resp], [%bar]", "[%resp+8], [%bar]"))
    with pytest.raises(warplens.ExecutionError, match="runs past the CTA's 64 bytes"):
        once(sim, request=REQUEST.replace("[%resp], [%bar]", "[%resp+64], [%bar]"))
    with pytest.raises(warplens.ExecutionError, match="used before mbarrier.init"):
        once(sim, request=f"{TRY_CANCEL} [%resp], [%bar+8];")


def test_remote_addresses_are_rejected(sim):
    request = """
        .reg .b64 %g, %gb;
        cvta.shared.u64 %g, smem;
        mapa.u64 %g, %g, 1;
        cvta.shared.u64 %gb, smem;
        add.u64 %gb, %gb, 16;
        mbarrier.arrive.expect_tx.b64 _, [%gb], 16;
        clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128 [%g], [%gb];
    """
    with pytest.raises(warplens.ExecutionError, match="another CTA's shared memory"):
        once(sim, request=request, cluster=2)


@pytest.mark.parametrize("instruction", [
    "clusterlaunchcontrol.try_cancel.shared::cta.mbarrier::complete_tx::bytes.b128 [%resp], [%bar];",  # no .async
    "clusterlaunchcontrol.try_cancel.async.shared::cta.b128 [%resp], [%bar];",  # no completion mechanism
    "clusterlaunchcontrol.try_cancel.async.shared::cluster.mbarrier::complete_tx::bytes.b128 [%resp], [%bar];",
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b64 [%resp], [%bar];",
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%resp];",
    "clusterlaunchcontrol.query_cancel.is_canceled.b128 %c, %h;",
    "clusterlaunchcontrol.query_cancel.get_first_ctaid.b32.b128 %x, %h;",  # ptxas rejects this too
    "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%x, %y, %z}, %h;",
    "clusterlaunchcontrol.query_cancel.get_first_ctaid::w.b32.b128 %x, %h;",
    "clusterlaunchcontrol.cancel.b128 %x, %h;",
])
def test_other_spellings_are_unsupported(sim, instruction):
    with pytest.raises(warplens.UnsupportedError, match="clusterlaunchcontrol"):
        once(sim, request=instruction + REQUEST)


def test_a_response_must_be_a_b128_register(sim):
    with pytest.raises(warplens.ExecutionError, match="cannot be read as .b128"):
        once(sim, query=f"{QUERY}.is_canceled.pred.b128 %c, %x;")


def test_clc_limit_must_be_a_count(sim):
    for bad in (-1, 1.5, True, "all"):
        with pytest.raises(warplens.LaunchError, match="clc_limit"):
            sim.launch(module("ret;"), "k", grid=1, block=1, args=[0], clc_limit=bad)


# -- .b128 registers -------------------------------------------------------------------------

def test_b128_load_store_and_mov(sim):
    body = """
        .shared .align 16 .b8 smem[64];
        .reg .b64 %lo, %hi, %o; .reg .b128 %q, %r, %s;
        ld.param.u64 %o, [out];
        ld.global.v2.u64 {%lo, %hi}, [%o];
        mov.b128 %q, {%lo, %hi};
        st.shared.b128 [smem+16], %q;
        ld.acquire.cta.shared.b128 %r, [smem+16];
        ld.weak.shared.b128 %s, [smem+16];
        mov.b128 %s, %s;
        st.global.b128 [%o+16], %r;
        mov.b128 {%lo, %hi}, %s;
        st.global.v2.u64 [%o+32], {%hi, %lo};
        mov.b128 %q, 0x123456789abcdef0123456789abcdef0;
        st.global.b128 [%o+48], %q;
        ret;
    """
    out = sim.from_numpy(np.array([0x1111222233334444, 0x5555666677778888, 0, 0, 0, 0, 0, 0], dtype=np.uint64))
    stats = sim.launch(module(body), "k", grid=1, block=1, args=[out])
    assert [hex(v) for v in out.numpy()] == [
        "0x1111222233334444", "0x5555666677778888", "0x1111222233334444", "0x5555666677778888",
        "0x5555666677778888", "0x1111222233334444", "0x123456789abcdef0", "0x123456789abcdef0"]
    assert stats.findings == []


@pytest.mark.parametrize("instruction, error", [
    ("add.b128 %q, %q, %q;", warplens.UnsupportedError),
    ("add.u64 %lo, %q, %lo;", warplens.ExecutionError),
    ("mov.b128 %q, %lo;", warplens.ExecutionError),
    ("mov.b128 %q, {%lo, %lo, %lo, %lo};", warplens.UnsupportedError),
    ("ld.acquire.shared.b128 %q, [smem];", warplens.UnsupportedError),  # no scope
    ("ld.cta.shared.b128 %q, [smem];", warplens.UnsupportedError),
])
def test_b128_misuse_is_rejected(sim, instruction, error):
    body = f".shared .align 16 .b8 smem[64];\n.reg .b64 %lo; .reg .b128 %q;\n{instruction}\nret;"
    with pytest.raises(error):
        sim.launch(module(body), "k", grid=1, block=1, args=[0])


# -- the toolchain's opinion of these kernels ---------------------------------------------------

@pytest.mark.skipif(shutil.which("ptxas") is None, reason="ptxas is not installed")
@pytest.mark.parametrize("name", sorted(KERNELS))
def test_the_kernels_assemble(name, tmp_path):
    path = tmp_path / f"{name}.ptx"
    path.write_text(source(KERNELS[name]))
    done = subprocess.run(["ptxas", "-arch=sm_100a", str(path), "-o", str(tmp_path / "out.o")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
