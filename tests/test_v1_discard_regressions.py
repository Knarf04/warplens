"""PTX sink destinations discard vector memory accesses, not just results."""

import numpy as np
import pytest

import warplens
from conftest import kernel


@pytest.mark.parametrize("cache", ["", ".nc"])
@pytest.mark.parametrize("ty,value", [("u64", 0xFEDCBA9876543210), ("s64", -37), ("f64", 1.25)])
def test_vector_sink_elements_do_not_read_past_allocation(sim, cache, ty, value):
    source = sim.from_numpy(np.array([value], dtype={"u64": np.uint64, "s64": np.int64, "f64": np.float64}[ty]))
    output = sim.empty(1, ty)
    module = kernel(f"""
        .reg .b64 %src, %out, %value;
        ld.param.u64 %src, [source];
        ld.param.u64 %out, [output];
        ld.global{cache}.v4.{ty} {{%value, _, _, _}}, [%src];
        st.global.{ty} [%out], %value;
        ret;
    """, ".param .u64 source, .param .u64 output")
    stats = sim.launch(module, "k", grid=1, block=1, args=[source, output])
    assert output.numpy().tolist() == [value]
    assert not stats.findings


@pytest.mark.parametrize("cache", ["", ".nc"])
def test_retained_vector_element_still_checks_its_address(sim, cache):
    source = sim.from_numpy(np.array([17], dtype=np.uint64))
    module = kernel(f"""
        .reg .b64 %src, %value;
        ld.param.u64 %src, [source];
        ld.global{cache}.v4.u64 {{_, %value, _, _}}, [%src];
        ret;
    """, ".param .u64 source")
    with pytest.raises(warplens.MemoryFault, match="outside every allocation"):
        sim.launch(module, "k", grid=1, block=1, args=[source])


def test_elect_discarded_value_keeps_leader_predicate(sim):
    output = sim.empty(32, "u32")
    module = kernel("""
        .reg .b64 %out, %offset;
        .reg .b32 %lane, %result;
        .reg .pred %leader;
        ld.param.u64 %out, [output];
        mov.u32 %lane, %laneid;
        elect.sync _|%leader, 0xffffffff;
        selp.u32 %result, 1, 0, %leader;
        mul.wide.u32 %offset, %lane, 4;
        add.u64 %out, %out, %offset;
        st.global.u32 [%out], %result;
        ret;
    """, ".param .u64 output")
    stats = sim.launch(module, "k", grid=1, block=32, args=[output])
    assert output.numpy().sum() == 1
    assert not stats.findings
