"""Regression coverage for static/dynamic shared-memory layout contracts."""

import pytest

import warplens
from warplens.units.smem import MAX_SMEM_BYTES

from conftest import HEADER, kernel


def layout_kernel(alignment, body=""):
    return warplens.parse_ptx(HEADER + f"""
        .shared .u32 prefix;
        .extern .shared .align 16 .b8 first[];
        .extern .shared .align {alignment} .b8 second[];
        .visible .entry k(.param .u64 out) {{
            .reg .b32 %a, %b, %v; .reg .b64 %o;
            mov.u32 %a, first;
            mov.u32 %b, second;
            ld.param.u64 %o, [out];
            st.global.u32 [%o], %a;
            st.global.u32 [%o+4], %b;
            {body}
            ret;
        }}
    """)


@pytest.mark.parametrize("alignment", [16, 32, 64, 128, 256])
def test_dynamic_shared_aliases_respect_strongest_alignment(sim, alignment):
    mod = layout_kernel(alignment, """
        st.shared.u8 [first+36], 99;
        ld.shared.u8 %v, [second+36];
        st.global.u32 [%o+8], %v;
    """)
    out = sim.empty(3, "u32")
    stats = sim.launch(mod, "k", grid=1, block=1, smem=37, args=[out])
    assert out.numpy().tolist() == [alignment, alignment, 99]
    assert stats.smem_bytes == alignment + 37
    assert not stats.findings


def test_dynamic_shared_padding_does_not_extend_the_requested_buffer(sim):
    mod = layout_kernel(128, "st.shared.u8 [second+37], 99;")
    with pytest.raises(warplens.MemoryFault, match="outside the CTA's 165 bytes"):
        sim.launch(mod, "k", grid=1, block=1, smem=37, args=[sim.empty(2, "u32")])


def test_dynamic_alignment_padding_counts_towards_the_shared_memory_limit(sim):
    with pytest.raises(warplens.LaunchError, match="the limit is 233472"):
        sim.launch(layout_kernel(256), "k", grid=1, block=1,
                   smem=MAX_SMEM_BYTES - 128, args=[sim.empty(2, "u32")])


@pytest.mark.parametrize("alignment", [0, -4, 3, 24, "1.5"])
def test_invalid_shared_alignment_is_rejected(alignment):
    with pytest.raises(warplens.PtxSyntaxError, match="positive power of two"):
        layout_kernel(alignment)


@pytest.mark.parametrize("literal", ["0x80", "0200", "0b10000000"])
def test_shared_alignment_accepts_ptx_integer_literals(sim, literal):
    out = sim.empty(2, "u32")
    stats = sim.launch(layout_kernel(literal), "k", grid=1, block=1, smem=1, args=[out])
    assert out.numpy().tolist() == [128, 128]
    assert stats.smem_bytes == 129


@pytest.mark.parametrize("literal", ["0x10", "020", "0b10000"])
def test_shared_array_sizes_accept_ptx_integer_literals(sim, literal):
    mod = kernel(f".shared .b8 cell[{literal}]; ret;")
    assert sim.launch(mod, "k", grid=1, block=1).smem_bytes == 16


def test_zero_sized_external_array_uses_dynamic_shared_memory(sim):
    mod = warplens.parse_ptx(HEADER + ".extern .shared .b8 dyn[0]; .entry k() { ret; }")
    assert mod.shared[0].size is None
    assert sim.launch(mod, "k", grid=1, block=1, smem=32).smem_bytes == 32


@pytest.mark.parametrize("dimension", ["0", "-1", "1.5"])
def test_invalid_static_shared_array_sizes_are_syntax_errors(dimension):
    with pytest.raises(warplens.PtxSyntaxError):
        kernel(f".shared .b8 cell[{dimension}]; ret;")
