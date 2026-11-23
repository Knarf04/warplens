import struct
import subprocess
import sys

import pytest

import warplens
from warplens.frontend.ir import Imm, Mem, Reg, Special, Sym

from conftest import EXAMPLES, HEADER, kernel


def test_parses_nvcc_vecadd():
    mod = warplens.parse_ptx_file(EXAMPLES / "00_vecadd" / "vecadd.ptx")
    assert mod.target == "sm_100a" and mod.address_size == 64
    fn = mod.functions["vecadd"]
    assert [p.type for p in fn.params] == ["u64", "u64", "u64", "u32"]
    assert fn.params[0].is_ptr and not fn.params[3].is_ptr
    assert fn.regs["%p1"] == "pred" and fn.regs["%rd10"] == "b64" and "%r9" not in fn.regs
    assert [i.opcode for i in fn.instrs].count("ld") == 6
    bra = next(i for i in fn.instrs if i.opcode == "bra")
    assert bra.pred == Reg("%p1") and not bra.pred_negated
    assert fn.instrs[fn.labels[bra.operands[0].name]].opcode == "ret"


def test_operand_forms():
    mod = kernel("""
        .reg .b32 %r<2>; .reg .b64 %rd1; .reg .pred %p;
        mov.u32 %r0, %tid.y;
        mov.b32 %r1, 0f3F800000;
        mov.b32 %r1, -0x10;
        ld.global.b32 %r0, [%rd1+8];
        ld.global.b32 %r0, [%rd1+-4];
        ld.param.u64 %rd1, [arg];
        @!%p ret;
    """, ".param .u64 arg")
    ins = mod.functions["k"].instrs
    assert ins[0].operands[1] == Special("tid.y")
    assert ins[1].operands[1] == Imm("f32", 0x3F800000)
    assert ins[2].operands[1] == Imm("int", -16)
    assert ins[3].operands[1] == Mem(Reg("%rd1"), 8)
    assert ins[4].operands[1] == Mem(Reg("%rd1"), -4)
    assert ins[5].operands[1] == Mem(Sym("arg"), 0)
    assert ins[6].pred == Reg("%p") and ins[6].pred_negated
    assert ins[5].name == "ld.param.u64" and ins[5].line == 13


def test_errors_name_the_line():
    with pytest.raises(warplens.PtxSyntaxError, match=r"test.ptx:7: undeclared register `%r9`"):
        kernel(".reg .b32 %r1;\nmov.u32 %r1, %r9;")
    with pytest.raises(warplens.PtxSyntaxError, match="undefined label"):
        kernel("bra nowhere;")
    with pytest.raises(warplens.PtxSyntaxError, match="expected ','"):
        kernel(".reg .b32 %r1;\nmov.u32 %r1 %r1;")


def test_unsupported_constructs_are_hard_errors():
    with pytest.raises(warplens.UnsupportedError, match=r"`\.local`"):
        kernel(".local .align 4 .b8 buf[64];")
    with pytest.raises(warplens.UnsupportedError, match=r"module-level directive `\.global`"):
        warplens.parse_ptx(HEADER + ".global .align 4 .u32 x;")


@pytest.mark.parametrize("literal", [
    "1e-3", "1E+3", ".5", "1.", "1.e2", ".5e-2", "0e0", "1.25",
    "-1e-3", "-.5", "-1.", "-.5e+2", "-1.E-2", "-0.", "+1e-3", "+.5", "+1.",
])
def test_decimal_float_literals(literal):
    mod = kernel(f".reg .f64 %f; mov.f64 %f, {literal}; ret;")
    imm = mod.functions["k"].instrs[0].operands[1]
    bits = int.from_bytes(struct.pack("<d", float(literal)), "little")
    assert imm == Imm("f64", bits)


@pytest.mark.parametrize("literal, expected", [
    ("0xdeadbeef", 0xDEADBEEF), ("0xDEADBEEFU", 0xDEADBEEF),
    ("+0x1e", 30), ("-0x1e", -30), ("0b101U", 5), ("077", 63),
    ("42U", 42),
])
def test_integer_literals_keep_integer_interpretation(literal, expected):
    mod = kernel(f".reg .b64 %r; mov.b64 %r, {literal}; ret;")
    assert mod.functions["k"].instrs[0].operands[1] == Imm("int", expected)


@pytest.mark.parametrize("following", ["", "\n.visible .entry actual() { ret; }"])
def test_entry_prototypes_are_rejected_before_the_next_function(following):
    text = HEADER + ".extern .entry external();" + following
    with pytest.raises(warplens.UnsupportedError,
                       match=r"test.ptx:4: entry declarations without a body"):
        warplens.parse_ptx(text, name="test.ptx")


def test_pragma_is_ignored_without_consuming_the_next_instruction():
    fn = kernel('.pragma "nounroll"; ret;').functions["k"]
    assert [ins.name for ins in fn.instrs] == ["ret"]


def test_header_pragma_is_not_mistaken_for_an_entry_declaration():
    text = HEADER + '.visible .entry k() .maxntid 32, 1, 1 .pragma "nounroll"; { ret; }'
    fn = warplens.parse_ptx(text).functions["k"]
    assert [ins.name for ins in fn.instrs] == ["ret"]


@pytest.mark.parametrize("ending", ["", "\n}"])
def test_unterminated_pragma_reports_an_error_without_hanging(ending):
    text = HEADER + '.entry k() {\n.pragma "nounroll"' + ending
    # A subprocess bounds this regression: the old parser spins forever at EOF.
    result = subprocess.run(
        [sys.executable, "-c", "import sys, warplens; warplens.parse_ptx(sys.argv[1], name='test.ptx')", text],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode != 0
    assert "PtxSyntaxError: test.ptx:5: directive `.pragma` is missing its terminating ';'" in result.stderr
