"""Lexical scope of shared variables in inline PTX blocks."""

import pytest

import warplens

from tests.support import HEADER, kernel


def test_nested_shared_declaration_does_not_alias_the_outer_variable(sim):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 %v;
        .shared .u32 cell;
        ld.param.u64 %out, [out];
        st.shared.u32 [cell], 11;
        {
            .shared .u32 cell;
            st.shared.u32 [cell], 22;
            ld.shared.u32 %v, [cell];
            st.global.u32 [%out], %v;
        }
        ld.shared.u32 %v, [cell];
        st.global.u32 [%out+4], %v;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [22, 11]
    assert not stats.findings


def test_nested_shared_variable_shadows_an_outer_register(sim):
    mod = kernel("""
        .reg .b64 %out; .reg .b32 value, %v;
        ld.param.u64 %out, [out];
        mov.u32 value, 11;
        {
            .shared .u32 value;
            st.shared.u32 [value], 22;
            ld.shared.u32 %v, [value];
            st.global.u32 [%out], %v;
        }
        st.global.u32 [%out+4], value;
        ret;
    """, ".param .u64 out")
    out = sim.empty(2, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [22, 11]


def test_nested_shared_variable_does_not_shadow_module_variable_after_scope_exit(sim):
    mod = warplens.parse_ptx(HEADER + """
        .shared .u32 cell;
        .entry k(.param .u64 out) {
            .reg .b64 %out; .reg .b32 %v;
            ld.param.u64 %out, [out];
            st.shared.u32 [cell], 11;
            { .shared .u32 cell; st.shared.u32 [cell], 22; }
            ld.shared.u32 %v, [cell];
            st.global.u32 [%out], %v;
            ret;
        }
    """)
    out = sim.empty(1, "u32")
    sim.launch(mod, "k", grid=1, block=1, args=[out])
    assert out.numpy().tolist() == [11]


def test_block_local_shared_symbol_is_not_visible_after_its_block(sim):
    mod = kernel("{ .shared .u32 cell; } st.shared.u32 [cell], 11; ret;")
    with pytest.raises(warplens.ExecutionError, match="not a variable this kernel can address"):
        sim.launch(mod, "k", grid=1, block=1)
