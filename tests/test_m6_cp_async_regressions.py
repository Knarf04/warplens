"""Per-thread async-copy groups and debug-directive parsing regressions."""

import numpy as np
import pytest

import warplens
from warplens.frontend.ptx_parser import parse_ptx

from tests.support import HEADER, kernel


def copy_kernel(body, *, cluster=False):
    index = "%ctaid.x" if cluster else "%tid.x"
    return kernel(f"""
        .reg .u64 %src, %out, %offset;
        .reg .u32 %lane, %dst, %value, %size;
        .reg .pred %p0, %p1;
        .shared .align 16 .b8 buffer[128];
        ld.param.u64 %src, [src];
        ld.param.u64 %out, [out];
        mov.u32 %lane, {index};
        setp.eq.u32 %p0, %lane, 0;
        setp.eq.u32 %p1, %lane, 1;
        mul.wide.u32 %offset, %lane, 4;
        add.u64 %src, %src, %offset;
        add.u64 %out, %out, %offset;
        mov.u32 %dst, buffer;
        {"" if cluster else "mad.lo.u32 %dst, %lane, 4, %dst;"}
        {body}
        ret;
    """, ".param .u64 src, .param .u64 out")


def test_empty_commit_counts_among_the_newest_groups(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.commit_group;
        cp.async.wait_group 1;
        ld.shared.u32 %value, [%dst];
        st.global.u32 [%out], %value;
        cp.async.wait_group 0;
    """)
    out = sim.empty(1, "u32")
    stats = sim.launch(ptx, "k", grid=1, block=1,
                       args=[sim.from_numpy(np.array([73], np.uint32)), out], async_delay=100)
    assert out.numpy().tolist() == [73]
    assert not stats.findings


def test_commit_and_wait_only_track_the_executing_threads_copies(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        @%p0 cp.async.commit_group;
        @%p1 cp.async.wait_group 0;
        @%p1 ld.shared.u32 %value, [%dst];
        @%p1 st.global.u32 [%out], %value;
        cp.async.commit_group;
        cp.async.wait_group 0;
    """)
    out = sim.empty(2, "u32")
    stats = sim.launch(ptx, "k", grid=1, block=2,
                       args=[sim.from_numpy(np.array([41, 43], np.uint32)), out], async_delay=100)
    assert out.numpy().tolist() == [0, 0]
    assert any(f.kind == "tma-load-unsynchronized" for f in stats.findings)


def test_wait_larger_than_group_count_does_not_wait_an_older_prefix(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.ca.shared.global [%dst+4], [%src+4], 4;
        cp.async.commit_group;
        cp.async.wait_group 3;
        ld.shared.u32 %value, [%dst];
        st.global.u32 [%out], %value;
        cp.async.wait_group 0;
    """)
    out = sim.empty(1, "u32")
    stats = sim.launch(ptx, "k", grid=1, block=1,
                       args=[sim.from_numpy(np.array([11, 13], np.uint32)), out], async_delay=100)
    assert out.numpy().tolist() == [0]
    assert any(f.kind == "tma-load-unsynchronized" for f in stats.findings)


@pytest.mark.parametrize("timing", [False, True])
def test_cp_async_destination_is_cta_local_inside_a_cluster(sim, timing):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.wait_group 0;
        ld.shared.u32 %value, [%dst];
        st.global.u32 [%out], %value;
    """, cluster=True)
    out = sim.empty(2, "u32")
    stats = sim.launch(ptx, "k", grid=2, block=1, cluster=2, timing=timing,
                       args=[sim.from_numpy(np.array([17, 19], np.uint32)), out])
    assert out.numpy().tolist() == [17, 19]
    assert not stats.findings


@pytest.mark.parametrize("instruction,message", [
    ("cp.async.cg.shared.global [%dst], [%src], 4;", "16 bytes"),
    ("mov.u32 %size, 4; cp.async.ca.shared.global [%dst], [%src], %size;", "integer constant"),
    ("mov.u32 %size, 0; cp.async.wait_group %size;", "integer constant"),
    ("cp.async.wait_group -1;", "nonnegative"),
    ("cp.async.ca.shared.global [%dst+1], [%src], 4;", "aligned"),
    ("cp.async.ca.shared.global [%dst], [%src+1], 4;", "aligned"),
])
def test_cp_async_rejects_invalid_size_wait_and_alignment(sim, instruction, message):
    with pytest.raises(warplens.ExecutionError, match=message):
        sim.launch(copy_kernel(instruction), "k", grid=1, block=1,
                   args=[sim.from_numpy(np.arange(8, dtype=np.uint32)), sim.empty(1, "u32")])


def test_copy_fault_names_only_lanes_with_invalid_global_addresses(sim):
    with pytest.raises(warplens.MemoryFault) as exc:
        sim.launch(copy_kernel("cp.async.ca.shared.global [%dst], [%src], 4;"),
                   "k", grid=1, block=32,
                   args=[sim.from_numpy(np.arange(31, dtype=np.uint32)), sim.empty(32, "u32")])
    assert exc.value.lanes == [31]


def test_debug_sections_allow_blank_lines_and_braces_inside_comments():
    text = HEADER + """
.section .debug_info

{
  .b8 1 // a closing brace here must not terminate the section: }
  .b32 begin-end
}
.visible .entry k() {
  ret;
}
"""
    module = parse_ptx(text)
    instruction = module.functions["k"].instrs[0]
    assert instruction.opcode == "ret"
    assert instruction.line == text.splitlines().index("  ret;") + 1


def test_commented_debug_section_does_not_swallow_executable_code():
    text = HEADER + """
/*
.section .debug_info {
*/
.visible .entry k() { ret; }
"""
    assert "k" in parse_ptx(text).functions


@pytest.mark.parametrize("fenced", [False, True])
def test_cp_async_uses_generic_proxy_before_a_tma_store(sim, fenced):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 16;
        cp.async.commit_group;
        cp.async.wait_group 0;
    """ + ("fence.proxy.async.shared::cta;" if fenced else "") + """
        cp.async.bulk.global.shared::cta.bulk_group [%out], [%dst], 16;
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
    """)
    out = sim.empty(4, "u32")
    stats = sim.launch(ptx, "k", grid=1, block=1,
                       args=[sim.from_numpy(np.arange(4, dtype=np.uint32)), out])
    assert any(f.kind == "tma-store-missing-fence" for f in stats.findings) == (not fenced)
    np.testing.assert_array_equal(out.numpy(), np.arange(4, dtype=np.uint32))


def test_cp_async_overwrite_requires_waiting_for_the_previous_copy(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.wait_group 0;
    """)
    stats = sim.launch(ptx, "k", grid=1, block=1, async_delay=100,
                       args=[sim.from_numpy(np.array([3], np.uint32)), sim.empty(1, "u32")])
    assert any(f.kind == "tma-load-unsynchronized" for f in stats.findings)


def test_cp_async_detects_overlapping_destinations_in_one_warp_instruction(sim):
    ptx = copy_kernel("""
        mov.u32 %dst, buffer;
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        cp.async.wait_group 0;
    """)
    stats = sim.launch(ptx, "k", grid=1, block=2,
                       args=[sim.from_numpy(np.array([3, 5], np.uint32)), sim.empty(2, "u32")])
    assert any(f.kind == "cp-async-overlapping-writes" for f in stats.findings)


def test_predicated_wait_does_not_observe_another_lanes_copy(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4;
        cp.async.commit_group;
        @%p0 cp.async.wait_group 0;
        @%p1 ld.shared.u32 %value, [%dst];
        @%p1 st.global.u32 [%out], %value;
        cp.async.wait_group 0;
    """)
    stats = sim.launch(ptx, "k", grid=1, block=2,
                       args=[sim.from_numpy(np.array([3, 5], np.uint32)), sim.empty(2, "u32")])
    assert any(f.kind == "tma-load-unsynchronized" for f in stats.findings)


def test_timing_batches_lanes_but_only_charges_read_bytes_to_global_memory(sim):
    ptx = copy_kernel("""
        cp.async.ca.shared.global [%dst], [%src], 4, 0;
        cp.async.commit_group;
        cp.async.wait_group 0;
    """)
    stats = sim.launch(ptx, "k", grid=1, block=32, timing=True,
                       args=[sim.from_numpy(np.arange(32, dtype=np.uint32)), sim.empty(32, "u32")])
    copies = [event for event in stats.timing.slices if event.name == "cp.async copy"]
    assert len(copies) == 1
    assert copies[0].args["bytes"] == 128 and copies[0].args["from_hbm"] == 0
    assert not stats.findings
