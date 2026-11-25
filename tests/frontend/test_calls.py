"""Calls to device functions defined in the module (`.func`), which are inlined before a kernel
runs. tests/validation/golden_replay/test_hardware_golden.py compares compiler-generated calls with a real GPU, including
every form below except the call under a predicate, which nvcc does not generate."""

import numpy as np
import pytest

import warplens

from tests.support import EXAMPLES, HEADER
from warplens.errors import ExecutionError, UnsupportedError
from warplens.units import call as call_unit

AMPERE = EXAMPLES.parent / "validation" / "ampere"

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


def test_a_call_that_does_not_match_its_definition_is_reported_at_launch(sim):
    functions = FUNCTIONS.replace("f(.param .b32 f_param_0)", "f(.param .b32 f_param_0, .param .b32 f_param_1)")
    out = sim.empty(64, "u32")
    with pytest.raises(UnsupportedError, match="does not match its definition"):
        sim.launch(module(functions=functions), "k", grid=1, block=64, args=[out])


def test_a_call_to_an_undefined_function_is_still_refused(sim):
    out = sim.empty(64, "u32")
    with pytest.raises(UnsupportedError, match="only printf is implemented"):
        sim.launch(module(functions=""), "k", grid=1, block=64, args=[out])


# -- the forms below: four u32 results per thread -----------------------------------------

def run(sim, body: str, functions: str, threads: int = 64, globals_: str = ""):
    """Run `body` in every thread, with %i the thread index and %po its four output words."""
    text = HEADER + globals_ + functions + f"""
.visible .entry k(.param .u64 out)
{{
    .reg .b64 %po, %off, %rd<8>; .reg .b32 %i, %r<16>; .reg .pred %p<4>;
    ld.param.u64 %po, [out];
    mov.u32 %i, %tid.x;
    mul.wide.u32 %off, %i, 16;
    add.u64 %po, %po, %off;
{body}
    ret;
}}
"""
    out = sim.empty(4 * threads, "u32")
    stats = sim.launch(warplens.parse_ptx(text, name="calls.ptx"), "k", grid=1, block=threads, args=[out])
    assert not stats.findings
    return out.numpy().reshape(threads, 4).astype(np.int64)


def call(name: str, arg: str, result: str, pred: str = "", how: str = "call.uni") -> str:
    """One call with a u32 argument and result, as nvcc writes it: `call.uni` to a function, or
    `call` through a register (`name`), which also names a prototype."""
    indirect = how != "call.uni"
    proto = "proto : .callprototype (.param .b32 _) _ (.param .b32 _);" if indirect else ""
    return f"""    {{ .param .b32 param0; .param .b32 retval0;
      st.param.b32 [param0], {arg};
      {proto}
      {pred} {how} (retval0), {name}, (param0){", proto" if indirect else ""};
      {pred} ld.param.b32 {result}, [retval0]; }}
"""


# The forms nvcc generates for structs by value: storage declared as a byte array, filled and
# read with vector and offset accesses, and a callee that takes the address of a large one and
# reads it as local memory.
STRUCTS = """
.func (.param .align 16 .b8 func_retval0[8]) swap(.param .align 16 .b8 swap_param_0[8])
{
    .reg .b32 %r<4>;
    ld.param.v2.b32 {%r1, %r2}, [swap_param_0];
    add.u32 %r3, %r1, 1;
    st.param.v2.b32 [func_retval0], {%r2, %r3};
    ret;
}
.func (.param .b32 func_retval0) pick(.param .align 16 .b8 pick_param_0[16], .param .b32 pick_param_1)
{
    .reg .b32 %r<4>; .reg .b64 %rd<4>;
    mov.b64 %rd1, pick_param_0;
    ld.param.b32 %r1, [pick_param_1];
    and.b32 %r1, %r1, 3;
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.local.b32 %r2, [%rd3];
    ld.param.s16 %r3, [pick_param_0+14];
    add.u32 %r2, %r2, %r3;
    st.param.b32 [func_retval0], %r2;
    ret;
}
"""
STRUCT_BODY = """
    shl.b32 %r1, %i, 1;
    { .param .align 16 .b8 param0[8]; .param .align 16 .b8 retval0[8];
      st.param.v2.b32 [param0], {%i, %r1};
      call.uni (retval0), swap, (param0);
      ld.param.b32 %r2, [retval0]; ld.param.b32 %r3, [retval0+4]; }
    st.global.u32 [%po], %r2; st.global.u32 [%po+4], %r3;
    add.u32 %r4, %i, 10; add.u32 %r5, %i, 20; add.u32 %r6, %i, 0x10000;
    { .param .align 16 .b8 param0[16]; .param .b32 param1; .param .b32 retval0;
      st.param.v2.b32 [param0], {%i, %r4};
      st.param.b32 [param0+12], %r6;
      st.param.b32 [param1], %i;
      st.param.b32 [param0+8], %r5;
      call.uni (retval0), pick, (param0, param1);
      ld.param.b32 %r7, [retval0]; }
    st.global.u32 [%po+8], %r7;
"""


def test_structs_passed_and_returned_by_value(sim):
    got = run(sim, STRUCT_BODY, STRUCTS)
    i = np.arange(64)
    np.testing.assert_array_equal(got[:, 0], 2 * i)
    np.testing.assert_array_equal(got[:, 1], i + 1)
    # Field i & 3 of {i, i + 10, i + 20, i + 0x10000}, plus the high half of the last field: 1.
    np.testing.assert_array_equal(got[:, 2], np.choose(i & 3, [i, i + 10, i + 20, i + 0x10000]) + 1)


def test_an_access_past_the_end_of_a_struct_argument_is_an_error(sim):
    with pytest.raises(ExecutionError, match="offset 16 of the 16-byte call argument"):
        run(sim, STRUCT_BODY.replace("[param0+12], %r6", "[param0+16], %r6"), STRUCTS)


def test_a_struct_argument_of_another_size_does_not_match(sim):
    with pytest.raises(UnsupportedError, match="call to `swap` does not match its definition"):
        run(sim, STRUCT_BODY.replace("param0[8]", "param0[12]"), STRUCTS)


def test_calls_under_a_predicate(sim):
    """Only the lanes whose predicate holds make the call; `f` itself returns from two places."""
    body = ("    and.b32 %r1, %i, 3; setp.eq.u32 %p1, %r1, 0;\n    mov.u32 %r2, 7; mov.u32 %r3, 7; mov.u32 %r4, 7;\n"
            + call("f", "%i", "%r2", pred="@%p1") + call("f", "%i", "%r3", pred="@!%p1")
            + call("fact", "%r1", "%r4", pred="@!%p1")
            + "    st.global.u32 [%po], %r2; st.global.u32 [%po+4], %r3; st.global.u32 [%po+8], %r4;\n")
    got = run(sim, body, FUNCTIONS + RECURSIVE)
    i = np.arange(64)
    np.testing.assert_array_equal(got[:, 0], np.where(i % 4 == 0, f(i), 7))
    np.testing.assert_array_equal(got[:, 1], np.where(i % 4 != 0, f(i), 7))
    np.testing.assert_array_equal(got[:, 2], np.where(i % 4 != 0, np.array([1, 1, 2, 6])[i % 4], 7))


TABLE = ".global .align 8 .u64 table[2] = {g, f};\n"  # before the functions, as nvcc writes it
INDIRECT_BODY = """
    mov.b64 %rd1, table; and.b32 %r1, %i, 1; mul.wide.u32 %rd2, %r1, 8; add.u64 %rd1, %rd1, %rd2;
    ld.global.u64 %rd3, [%rd1];
    mov.b64 %rd4, g;
""" + call("%rd3", "%i", "%r2", how="call") + call("%rd4", "%i", "%r3", how="call") + """
    st.global.u32 [%po], %r2; st.global.u32 [%po+4], %r3;
"""


def test_indirect_calls_through_a_table_and_through_a_function_address(sim):
    got = run(sim, INDIRECT_BODY, FUNCTIONS, globals_=TABLE)
    i = np.arange(64)
    np.testing.assert_array_equal(got[:, 0], np.where(i % 2 == 0, 3 * i, f(i)))
    np.testing.assert_array_equal(got[:, 1], 3 * i)


def test_calltargets_limits_what_an_indirect_call_reaches(sim):
    """`.calltargets` instead of a prototype; the register of odd lanes holds `f`, which is not listed."""
    both = INDIRECT_BODY.replace("proto : .callprototype (.param .b32 _) _ (.param .b32 _);",
                                 "proto : .calltargets g, f;", 1)
    i = np.arange(64)
    np.testing.assert_array_equal(run(sim, both, FUNCTIONS, globals_=TABLE)[:, 0], np.where(i % 2 == 0, 3 * i, f(i)))
    only_g = both.replace(".calltargets g, f;", ".calltargets g;")
    with pytest.raises(ExecutionError, match="indirect call through %rd3 to 0x[0-9a-f]+, which is not") as e:
        run(sim, only_g, FUNCTIONS, globals_=TABLE)
    assert "lane 1 " in str(e.value)
    with pytest.raises(UnsupportedError, match="may target `nowhere`, which is not a function defined"):
        run(sim, both.replace(".calltargets g, f;", ".calltargets g, nowhere;"), FUNCTIONS, globals_=TABLE)


def test_an_indirect_call_to_no_function_is_an_error(sim):
    with pytest.raises(ExecutionError, match="indirect call through %rd4 to 0x0, which is not the address"):
        run(sim, INDIRECT_BODY.replace("mov.b64 %rd4, g;", "mov.b64 %rd4, 0;"), FUNCTIONS, globals_=TABLE)


def test_an_initializer_that_is_not_a_function_is_refused():
    with pytest.raises(UnsupportedError, match="initializer `elsewhere` is not a number or a function"):
        warplens.parse_ptx(HEADER + ".global .align 8 .u64 table[1] = {elsewhere};\n" + FUNCTIONS)


# fact keeps %n across its recursive call; sum recurses once per unit of its argument; forever
# has no way out.
RECURSIVE = """
.func (.param .b32 func_retval0) fact(.param .b32 fact_param_0)
{
    .reg .b32 %n, %r; .reg .pred %p;
    ld.param.u32 %n, [fact_param_0];
    setp.gt.u32 %p, %n, 1;
    @%p bra more;
    mov.u32 %r, 1;
    st.param.b32 [func_retval0], %r;
    ret;
more:
    sub.u32 %r, %n, 1;
    { .param .b32 param0; .param .b32 retval0;
      st.param.b32 [param0], %r;
      call.uni (retval0), fact, (param0);
      ld.param.b32 %r, [retval0]; }
    mul.lo.u32 %r, %r, %n;
    st.param.b32 [func_retval0], %r;
    ret;
}
.func (.param .b32 func_retval0) sum(.param .b32 sum_param_0)
{
    .reg .b32 %n, %r; .reg .pred %p;
    ld.param.u32 %n, [sum_param_0];
    mov.u32 %r, 0;
    setp.eq.u32 %p, %n, 0;
    @%p bra done;
    sub.u32 %r, %n, 1;
    { .param .b32 param0; .param .b32 retval0;
      st.param.b32 [param0], %r;
      call.uni (retval0), sum, (param0);
      ld.param.b32 %r, [retval0]; }
    add.u32 %r, %r, %n;
done:
    st.param.b32 [func_retval0], %r;
    ret;
}
.func (.param .b32 func_retval0) forever(.param .b32 forever_param_0)
{
    .reg .b32 %n;
    ld.param.u32 %n, [forever_param_0];
    add.u32 %n, %n, 1;
    { .param .b32 param0; .param .b32 retval0;
      st.param.b32 [param0], %n;
      call.uni (retval0), forever, (param0);
      ld.param.b32 %n, [retval0]; }
    st.param.b32 [func_retval0], %n;
    ret;
}
"""
STORE = "    st.global.u32 [%po], %r2; st.global.u32 [%po+4], %r3;\n"


def test_recursion_with_lanes_at_different_depths(sim):
    body = ("    rem.u32 %r1, %i, 9; mul.lo.u32 %r4, %i, 4;\n"
            + call("fact", "%r1", "%r2") + call("sum", "%r4", "%r3") + STORE)
    got = run(sim, body, RECURSIVE)
    i = np.arange(64)
    np.testing.assert_array_equal(got[:, 0], [np.prod(np.arange(1, n + 1)) for n in i % 9])
    np.testing.assert_array_equal(got[:, 1], 4 * i * (4 * i + 1) // 2)  # up to 252 calls deep


def test_recursion_through_a_function_pointer(sim):
    body = "    rem.u32 %r1, %i, 9; mov.b64 %rd1, fact; mov.u32 %r3, 0;\n" + call("%rd1", "%r1", "%r2", how="call") + STORE
    got = run(sim, body, RECURSIVE)
    np.testing.assert_array_equal(got[:, 0], [np.prod(np.arange(1, n + 1)) for n in np.arange(64) % 9])


def test_recursion_without_an_end_overflows_the_call_stack(sim, monkeypatch):
    body = "    mov.u32 %r3, 0;\n" + call("forever", "%i", "%r2") + STORE
    with pytest.raises(ExecutionError, match=r"call stack overflow: `forever` recursed past 1024 active calls") as e:
        run(sim, body, RECURSIVE)
    assert "call.uni (retval0), forever, (param0);" in str(e.value) and "lane 0 " in str(e.value)
    monkeypatch.setattr(call_unit, "MAX_CALL_DEPTH", 10)
    with pytest.raises(ExecutionError, match="recursed past 10 active calls"):
        run(sim, body, RECURSIVE)
    # The limit counts calls that are active at once, not calls made.
    deep = "    mov.u32 %r1, 10; mov.u32 %r3, 0;\n" + call("sum", "%r1", "%r2") * 3 + STORE
    np.testing.assert_array_equal(run(sim, deep, RECURSIVE)[:, 0], 55)
    with pytest.raises(ExecutionError, match="call stack overflow: `sum`"):
        run(sim, deep.replace("mov.u32 %r1, 10;", "mov.u32 %r1, 11;"), RECURSIVE)


def test_recursive_kernel_in_a_timed_launch(sim):
    """The instructions that enter and leave a recursive call are charged like any other."""
    out = sim.empty(12 * 64, "s32")
    stats = sim.launch(AMPERE / "callforms_probe.ptx", "callforms_probe", grid=1, block=64, args=[out], timing=True)
    assert stats.timing is not None and not stats.findings


# leave(n) returns 7 if it gets down to 7 and otherwise ends the thread at n == 0, {how}.
LEAVE = """
.func (.param .b32 func_retval0) leave(.param .b32 leave_param_0)
{
    .reg .b32 %n, %r; .reg .pred %p;
    ld.param.u32 %n, [leave_param_0];
    setp.eq.u32 %p, %n, 0;
    EXIT
    mov.u32 %r, 7;
    setp.eq.u32 %p, %n, 7;
    @%p bra done;
    sub.u32 %r, %n, 1;
    { .param .b32 param0; .param .b32 retval0;
      st.param.b32 [param0], %r;
      call.uni (retval0), leave, (param0);
      ld.param.b32 %r, [retval0]; }
done:
    st.param.b32 [func_retval0], %r;
    ret;
TAIL
}
"""
LEAVE_BODY = "    mov.u32 %r3, 1; st.global.u32 [%po+4], %r3;\n" + call("leave", "%i", "%r2") + "    st.global.u32 [%po], %r2;\n"


def test_threads_that_exit_inside_a_recursive_call(sim):
    got = run(sim, LEAVE_BODY, LEAVE.replace("EXIT", "@%p exit;").replace("TAIL", ""), threads=32)
    np.testing.assert_array_equal(got[:, 0], np.where(np.arange(32) >= 7, 7, 0))
    np.testing.assert_array_equal(got[:, 1], 1)


def test_recursion_whose_lanes_cannot_rejoin_is_refused(sim):
    """A branch to `exit` has no point where its lanes meet again, so the lanes of one call do
    not reach the function's end together."""
    functions = LEAVE.replace("EXIT", "@%p bra out;").replace("TAIL", "out:\n    exit;")
    with pytest.raises(UnsupportedError, match="recursive call to `leave`: lanes return separately"):
        run(sim, LEAVE_BODY, functions, threads=32)


def test_a_callee_label_named_return_does_not_alias_the_synthetic_exit(sim):
    # ptxas accepts `return` as a label. Branching there must still execute the
    # result store, rather than jumping directly out of the inlined function.
    functions = """
    .func (.param .b32 result) f(.param .b32 input) {
        .reg .b32 %r;
        ld.param.b32 %r, [input];
        bra return;
    return:
        add.u32 %r, %r, 1;
        st.param.b32 [result], %r;
        ret;
    }
    """
    out = sim.empty(64, "u32")
    sim.launch(module(functions=functions), "k", grid=1, block=64, args=[out])
    x = np.arange(64)
    np.testing.assert_array_equal(out.numpy(), (x + 1) + (x + 2))


def test_input_value_is_saved_when_the_caller_reuses_its_slot_for_the_return(sim):
    # Both sm_100a and sm_80 ptxas accept this; the sm_80 compiler preserves input
    # value 5 and returns 6 despite the earlier store of 100 to the return slot.
    functions = """
    .func (.param .b32 result) f(.param .b32 input) {
        .reg .b32 %r;
        st.param.b32 [result], 100;
        ld.param.b32 %r, [input];
        add.u32 %r, %r, 1;
        st.param.b32 [result], %r;
        ret;
    }
    """
    caller = """
    .visible .entry k(.param .u64 out) {
        .reg .b64 %p; .reg .b32 %v;
        .param .b32 arg;
        ld.param.b64 %p, [out];
        st.param.b32 [arg], 5;
        call.uni (arg), f, (arg);
        ld.param.b32 %v, [arg];
        st.global.b32 [%p], %v;
        ret;
    }
    """
    out = sim.empty(1, "u32")
    sim.launch(module(kernel=caller, functions=functions), "k", grid=1, block=1, args=[out])
    np.testing.assert_array_equal(out.numpy(), [6])
