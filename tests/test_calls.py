"""Calls to device functions defined in the module (`.func`), which are inlined before a kernel
runs. tests/test_hardware_golden.py compares compiler-generated calls with a real GPU."""

import numpy as np
import pytest

import warplens
from conftest import HEADER
from warplens.errors import UnsupportedError

KERNEL = """
.visible .entry k(.param .u64 out)
{
    .reg .b64 %po, %off; .reg .b32 %i, %r, %s;
    ld.param.u64 %po, [out];
    mov.u32 %i, %tid.x;
    mul.wide.u32 %off, %i, 4;
    add.u64 %po, %po, %off;
    { .param .b32 param0; st.param.b32 [param0], %i;
      .param .b32 retval0;
      CALL (retval0), f, (param0);
      ld.param.b32 %r, [retval0]; }
    { .param .b32 param0; st.param.b32 [param0], %r;
      .param .b32 retval0;
      call.uni (retval0), f, (param0);
      ld.param.b32 %s, [retval0]; }
    add.u32 %r, %r, %s;
    st.global.u32 [%po], %r;
    ret;
}
"""
# f(x) = x + 100 for even x, g(x) = 3 * x otherwise; the same register and label names as the kernel.
FUNCTIONS = """
.func (.param .b32 func_retval0) g(.param .b32 g_param_0)
{
    .reg .b32 %r;
    ld.param.u32 %r, [g_param_0];
    mul.lo.u32 %r, %r, 3;
    st.param.b32 [func_retval0], %r;
    ret;
}
.func (.param .b32 func_retval0) f(.param .b32 f_param_0)
{
    .reg .b32 %r, %i; .reg .pred %p;
    ld.param.u32 %i, [f_param_0];
    and.b32 %r, %i, 1;
    setp.eq.u32 %p, %r, 0;
    @%p bra even;
    { .param .b32 param0; st.param.b32 [param0], %i;
      .param .b32 retval0;
      call.uni (retval0), g, (param0);
      ld.param.b32 %r, [retval0]; }
    st.param.b32 [func_retval0], %r;
    ret;
even:
    add.u32 %r, %i, 100;
    st.param.b32 [func_retval0], %r;
    ret;
}
"""


def f(x):
    return np.where(x % 2 == 0, x + 100, 3 * x)


def module(kernel=KERNEL, functions=FUNCTIONS, call="call.uni"):
    return warplens.parse_ptx(HEADER + functions + kernel.replace("CALL", call), name="calls.ptx")


def test_nested_calls_with_divergent_returns_and_two_call_sites(sim):
    mod = module()
    assert list(mod.functions) == ["k"] and list(mod.device_functions) == ["g", "f"]
    out = sim.empty(64, "u32")
    stats = sim.launch(mod, "k", grid=1, block=64, args=[out])
    x = np.arange(64)
    np.testing.assert_array_equal(out.numpy(), f(x) + f(f(x)))
    assert not stats.findings


def test_a_declaration_without_a_body_is_not_a_function():
    mod = module(functions=".func (.param .b32 func_retval0) f(.param .b32 f_param_0)\n;\n" + FUNCTIONS)
    assert list(mod.device_functions) == ["g", "f"]


@pytest.mark.parametrize("functions, call, message", [
    (FUNCTIONS.replace("call.uni (retval0), g,", "call.uni (retval0), f,"), "call.uni", "recursive call to `f`"),
    (FUNCTIONS, "@%p call.uni", "call under a predicate"),
    (FUNCTIONS.replace("f(.param .b32 f_param_0)", "f(.param .b32 f_param_0, .param .b32 f_param_1)"),
     "call.uni", "does not match its definition"),
])
def test_unsupported_calls_are_reported_at_launch(sim, functions, call, message):
    kernel = KERNEL.replace(".reg .b32 %i, %r, %s;", ".reg .b32 %i, %r, %s; .reg .pred %p;")
    out = sim.empty(64, "u32")
    with pytest.raises(UnsupportedError, match=message):
        sim.launch(module(kernel, functions, call), "k", grid=1, block=64, args=[out])


def test_a_call_to_an_undefined_function_is_still_refused(sim):
    out = sim.empty(64, "u32")
    with pytest.raises(UnsupportedError, match="only printf is implemented"):
        sim.launch(module(functions=""), "k", grid=1, block=64, args=[out])
