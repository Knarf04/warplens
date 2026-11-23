"""M3's general PTX: parameters, scoped locals, vector memory and printf."""

import struct

import numpy as np
import pytest

import warplens
from warplens.core.program import Program
from conftest import HEADER, kernel


def test_nested_local_declarations_keep_distinct_storage(sim):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 %v;
        .local .u32 cell;
        ld.param.u64 %out, [out];
        st.local.u32 [cell], 11;
        {
            .local .u32 cell;
            st.local.u32 [cell], 22;
            ld.local.u32 %v, [cell];
            st.global.u32 [%out], %v;
        }
        ld.local.u32 %v, [cell];
        st.global.u32 [%out+4], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [22, 11]


def test_local_variable_shadows_an_outer_register(sim):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 cell, %v;
        ld.param.u64 %out, [out];
        mov.u32 cell, 7;
        { .local .u32 cell; st.local.u32 [cell], 21; }
        st.global.u32 [%out], cell;
        ret;
    """, ".param .u64 out")
    out = sim.empty(1, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().item() == 7


def test_local_variable_is_not_visible_after_its_block(sim):
    mod = kernel("{ .local .u32 cell; } st.local.u32 [cell], 11; ret;")
    with pytest.raises(warplens.ExecutionError, match="not a variable this kernel can address"):
        sim.launch(mod, "k", grid=1, block=1)


@pytest.mark.parametrize("space, conversion", [
    ("", "cvta.param.u64 %ptr, cfg;"),
    (".global", "cvta.param.u64 %ptr, cfg;"),
    (".param", "mov.u64 %ptr, cfg;"),
])
def test_load_by_value_parameter_through_pointer(sim, space, conversion):
    mod = kernel(f"""
        .reg .b64 %out, %ptr; .reg .b32 %a, %b;
        ld.param.u64 %out, [out];
        {conversion}
        ld{space}.v2.u32 {{%a, %b}}, [%ptr+8];
        st.global.v2.u32 [%out], {{%a, %b}};
        ret;
    """, ".param .u64 out, .param .align 8 .b8 cfg[16]")
    out = sim.empty(2, "u32")
    # The cached program must consume this launch's bytes on each call.
    for pair in [(17, 19), (23, 29)]:
        sim.launch(mod, "k", grid=1, block=1, args=[out, struct.pack("<4I", 1, 2, *pair)])
        assert out.numpy().tolist() == list(pair)


@pytest.mark.parametrize("space", ["", ".global"])
def test_parameter_pointer_load_checks_bounds_and_reports_lanes(sim, space):
    mod = kernel(f"""
        .reg .b64 %ptr; .reg .b32 %v;
        cvta.param.u64 %ptr, cfg;
        ld{space}.u32 %v, [%ptr+8];
        ret;
    """, ".param .align 8 .b8 cfg[8]")
    with pytest.raises(warplens.MemoryFault, match="parameter") as exc:
        sim.launch(mod, "k", grid=1, block=2, args=[bytes(8)])
    assert exc.value.lanes == [0, 1]


def test_parameter_blob_cache_does_not_reuse_a_previous_launch():
    mod = kernel("ret;", ".param .u32 value")
    program = Program(mod, mod.functions["k"])
    # Short-lived dictionaries may reuse an object's old id. The actual
    # launch data, rather than that id, determines which blob is returned.
    for value in range(100):
        raw = struct.pack("<I", value)
        assert program.param_blob({"value": raw}) == raw


def test_generic_load_routes_parameter_and_global_lanes_independently(sim):
    mod = kernel("""
        .reg .b64 %out, %global, %param, %ptr, %offset;
        .reg .b32 %lane, %value; .reg .pred %p;
        ld.param.u64 %out, [out];
        ld.param.u64 %global, [global];
        cvta.param.u64 %param, value;
        mov.u32 %lane, %laneid;
        setp.eq.u32 %p, %lane, 0;
        selp.u64 %ptr, %param, %global, %p;
        ld.u32 %value, [%ptr];
        mul.wide.u32 %offset, %lane, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %value;
        ret;
    """, ".param .u64 out, .param .u64 global, .param .u32 value")
    out = sim.empty(2, "u32")
    source = sim.from_numpy(np.array([29], dtype=np.uint32))
    sim.launch(mod, "k", grid=1, block=2, args=[out, source, 17])
    assert out.numpy().tolist() == [17, 29]


@pytest.mark.parametrize("space", ["", ".global"])
def test_kernel_parameter_pointer_cannot_be_stored_through(sim, space):
    mod = kernel(f"""
        .reg .b64 %ptr;
        cvta.param.u64 %ptr, value;
        st{space}.u32 [%ptr], 17;
        ret;
    """, ".param .u32 value")
    with pytest.raises(warplens.ExecutionError, match="parameter memory is read-only"):
        sim.launch(mod, "k", grid=1, block=1, args=[23])


@pytest.mark.parametrize("generic", [False, True])
def test_constant_initializers_can_be_loaded(sim, generic):
    address = "cvta.const.u64 %ptr, table;" if generic else "mov.u64 %ptr, table;"
    load = "ld.v2.u32" if generic else "ld.const.v2.u32"
    mod = warplens.parse_ptx(HEADER + f"""
        .const .align 8 .u32 table[2] = {{17, 29}};
        .entry k(.param .u64 out) {{
            .reg .b64 %out, %ptr; .reg .b32 %a, %b;
            ld.param.u64 %out, [out];
            {address}
            {load} {{%a, %b}}, [%ptr];
            st.global.v2.u32 [%out], {{%a, %b}};
            ret;
        }}
    """)
    out = sim.empty(2, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [17, 29]


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("generic", [False, True])
def test_shared_vector_access_has_one_execution_and_combined_banks(sim, width, generic):
    regs = ", ".join(f"%v{i}" for i in range(width))
    values = ", ".join(str(i + 1) for i in range(width))
    conversion = "cvta.shared.u64 %ptr, %offset;" if generic else "mov.u64 %ptr, %offset;"
    space = "" if generic else ".shared"
    mod = kernel(f"""
        .shared .align 16 .b8 tile[{32 * width * 4}];
        .reg .b64 %ptr, %offset; .reg .b32 %lane, {regs};
        mov.u32 %lane, %laneid;
        mul.wide.u32 %offset, %lane, {width * 4};
        {conversion}
        st{space}.v{width}.u32 [%ptr], {{{values}}};
        ld{space}.v{width}.u32 {{{regs}}}, [%ptr];
        ret;
    """)
    stats = sim.launch(mod, "k", grid=1, block=32)
    assert not stats.findings
    assert len(stats.smem.instructions) == 2
    for entry in stats.smem.instructions.values():
        assert entry.executions == 1
        assert entry.wavefronts == entry.ideal == width
        assert entry.conflicts == 0


@pytest.mark.parametrize("fmt, raw, expected_text, expected_return", [
    (None, b"", "", -1),
    ("plain %%", b"", "plain %", 0),
    ("value=%d", struct.pack("<i", 123), "value=123", 1),
    ("%d %.1f", struct.pack("<i4xd", 7, 2.5), "7 2.5", 2),
    ("%*d", struct.pack("<ii", 4, 7), "   7", 2),
])
def test_printf_returns_number_of_arguments(sim, fmt, raw, expected_text, expected_return):
    data = ", ".join(str(b) for b in (fmt or "").encode() + b"\0")
    mod = warplens.parse_ptx(HEADER + f"""
        .global .b8 fmt[{len(fmt or '') + 1}] = {{{data}}};
        .entry k(.param .u64 out, .param .u64 args) {{
            .reg .b64 %ptr, %args, %out; .reg .b32 %result;
            .param .b64 format_arg; .param .b64 args_arg; .param .b32 result;
            mov.u64 %ptr, {'fmt' if fmt is not None else 0};
            ld.param.u64 %args, [args];
            st.param.b64 [format_arg], %ptr;
            st.param.b64 [args_arg], %args;
            call.uni (result), vprintf, (format_arg, args_arg);
            ld.param.b32 %result, [result];
            ld.param.u64 %out, [out];
            st.global.b32 [%out], %result;
            ret;
        }}
    """)
    out = sim.empty(1, "s32")
    args = sim.from_numpy(np.frombuffer(raw or bytes(1), dtype=np.uint8))
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out, args])
    assert stats.output == expected_text
    assert out.numpy().item() == expected_return
