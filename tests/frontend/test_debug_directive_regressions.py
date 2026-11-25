"""Debug sections and directives a compiler emits around the code: parsed past, never as code."""

from warplens.frontend.ptx_parser import parse_ptx

from tests.support import HEADER


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
