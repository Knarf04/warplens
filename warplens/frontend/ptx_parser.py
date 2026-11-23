"""PTX text -> IR.

Covers the subset needed so far: module header directives, `.entry` functions,
`.reg` declarations, labels, and predicated instructions with register,
immediate, memory and vector operands. Anything else is reported as
UnsupportedError with its line, never skipped.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from ..errors import PtxSyntaxError, UnsupportedError
from .ir import (Function, Imm, Instr, Mem, Module, NP_TYPES, Pair, Param, Reg, SharedVar,
                 Special, Sym, Vec)

_TOKEN = re.compile(
    r"""
    (?P<ws>[ \t\r\n]+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>"[^"\n]*")
  | (?P<num>[-+]?(?:0[fF][0-9a-fA-F]{8}|0[dD][0-9a-fA-F]{16}|0[xX][0-9a-fA-F]+U?|0[bB][01]+U?
        |(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+U?))
  | (?P<word>[A-Za-z_$%.][A-Za-z0-9_$.]*(?:::[A-Za-z0-9_$.]+)*)
  | (?P<punct>[{}()\[\],;:=+<>@!|])
    """,
    re.X | re.S,
)

# Special registers and whether they have .x/.y/.z components.
SPECIAL_REGISTERS = {
    "tid": True, "ntid": True, "ctaid": True, "nctaid": True,
    "laneid": False, "warpid": False, "nwarpid": False,
}

_LINKAGE = {".visible", ".weak", ".extern", ".common"}
_IGNORED_BODY_DIRECTIVES = {".pragma"}


class _Tok:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind: str, value: str, line: int):
        self.kind, self.value, self.line = kind, value, line

    def __repr__(self) -> str:
        return f"{self.value!r}"


def _tokenize(text: str, name: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos, line = 0, 1
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise PtxSyntaxError(f"{name}:{line}: unexpected character {text[pos]!r}")
        kind = m.lastgroup
        if kind not in ("ws", "comment"):
            toks.append(_Tok(kind, m.group(), line))
        line += m.group().count("\n")
        pos = m.end()
    toks.append(_Tok("eof", "<end of file>", line))
    return toks


def _number(tok: _Tok) -> Imm:
    s = tok.value
    neg = s.startswith("-")
    body = s[1:] if s.startswith(("-", "+")) else s
    prefix = body[:2].lower()
    if prefix == "0f":
        bits = int(body[2:], 16)
        return Imm("f32", bits ^ 0x8000_0000 if neg else bits)
    if prefix == "0d":
        bits = int(body[2:], 16)
        return Imm("f64", bits ^ (1 << 63) if neg else bits)
    if prefix not in ("0x", "0b") and ("." in body or "e" in body.lower()):
        (bits,) = struct.unpack("<Q", struct.pack("<d", float(s)))
        return Imm("f64", bits)
    body = body.rstrip("U")
    if prefix == "0x":
        v = int(body, 16)
    elif prefix == "0b":
        v = int(body, 2)
    elif len(body) > 1 and body[0] == "0":
        v = int(body, 8)
    else:
        v = int(body)
    return Imm("int", -v if neg else v)


class _Parser:
    def __init__(self, text: str, name: str):
        self.name = name
        self.lines = text.splitlines()
        self.toks = _tokenize(text, name)
        self.i = 0

    # -- token helpers -----------------------------------------------------
    def peek(self) -> _Tok:
        return self.toks[self.i]

    def next(self) -> _Tok:
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    def accept(self, value: str) -> bool:
        if self.peek().value == value and self.peek().kind != "string":
            self.i += 1
            return True
        return False

    def expect(self, value: str) -> _Tok:
        t = self.next()
        if t.value != value:
            raise self.syntax(t, f"expected {value!r}, found {t.value!r}")
        return t

    def expect_kind(self, kind: str, what: str) -> _Tok:
        t = self.next()
        if t.kind != kind:
            raise self.syntax(t, f"expected {what}, found {t.value!r}")
        return t

    def source(self, line: int) -> str:
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    def syntax(self, t: _Tok, msg: str) -> PtxSyntaxError:
        return PtxSyntaxError(f"{self.name}:{t.line}: {msg}\n    {self.source(t.line)}")

    def unsupported(self, t: _Tok, msg: str) -> UnsupportedError:
        return UnsupportedError(f"{self.name}:{t.line}: {msg}\n    {self.source(t.line)}")

    # -- module ------------------------------------------------------------
    def module(self) -> Module:
        mod = Module(name=self.name)
        extern = False
        while (t := self.next()).kind != "eof":
            if t.value not in _LINKAGE and t.value != ".shared":
                extern = False
            if t.value == ".version":
                mod.version = self.expect_kind("num", "a version number").value
            elif t.value == ".target":
                mod.target = self.expect_kind("word", "a target name").value
                while self.accept(","):
                    self.next()
            elif t.value == ".address_size":
                mod.address_size = int(self.expect_kind("num", "an address size").value)
            elif t.value in _LINKAGE:
                extern = extern or t.value == ".extern"
                continue
            elif t.value == ".shared":
                mod.shared.append(self.shared_var(extern))
            elif t.value == ".entry":
                fn = self.function(t)
                mod.functions[fn.name] = fn
            else:
                what = "directive" if t.value.startswith(".") else "token"
                raise self.unsupported(t, f"unsupported module-level {what} `{t.value}`")
        return mod

    def function(self, entry_tok: _Tok) -> Function:
        fn = Function(name=self.expect_kind("word", "a function name").value, line=entry_tok.line)
        if self.accept("("):
            if not self.accept(")"):
                while True:
                    fn.params.append(self.param())
                    if self.accept(")"):
                        break
                    self.expect(",")
        # Performance-tuning directives (.maxntid, .reqntid, ...) sit before the body.
        while self.peek().value != "{":
            t = self.next()
            if t.value == ";":
                raise self.unsupported(entry_tok, "entry declarations without a body are not supported yet")
            if t.kind == "eof":
                raise self.syntax(t, f"function `{fn.name}` has no body")
            if t.value in _IGNORED_BODY_DIRECTIVES:
                self.skip_directive(t)
        self.expect("{")
        self.body(fn)
        return fn

    def skip_directive(self, directive: _Tok) -> None:
        while (end := self.next()).value != ";":
            if end.kind == "eof" or end.value in ("{", "}"):
                raise self.syntax(directive,
                                  f"directive `{directive.value}` is missing its terminating ';'")

    def param(self) -> Param:
        self.expect(".param")
        ty_tok = self.expect_kind("word", "a parameter type")
        ty = ty_tok.value.lstrip(".")
        if ty not in NP_TYPES or ty == "pred":
            raise self.unsupported(ty_tok, f"unsupported parameter type `{ty_tok.value}`")
        is_ptr = False
        while (t := self.next()).value.startswith(".") and t.kind == "word":
            if t.value == ".ptr":
                is_ptr = True
            elif t.value == ".align":
                self.expect_kind("num", "an alignment")
        if t.kind != "word":
            raise self.syntax(t, f"expected a parameter name, found {t.value!r}")
        if self.peek().value == "[":
            raise self.unsupported(self.peek(), "array parameters are not supported yet")
        return Param(name=t.value, type=ty, is_ptr=is_ptr)

    # -- function body -----------------------------------------------------
    def body(self, fn: Function) -> None:
        while True:
            t = self.next()
            if t.kind == "eof":
                raise self.syntax(t, f"function `{fn.name}` is missing its closing '}}'")
            if t.value == "}":
                break
            if t.value == "{":
                raise self.unsupported(t, "nested { } blocks are not supported yet")
            if t.kind == "word" and self.peek().value == ":":
                self.next()
                if t.value in fn.labels:
                    raise self.syntax(t, f"label `{t.value}` defined twice")
                fn.labels[t.value] = len(fn.instrs)
            elif t.value == ".reg":
                self.reg_decl(fn)
            elif t.value == ".shared":
                fn.shared.append(self.shared_var(extern=False))
            elif t.value in _IGNORED_BODY_DIRECTIVES:
                self.skip_directive(t)
            elif t.kind == "word" and t.value.startswith("."):
                raise self.unsupported(t, f"unsupported directive `{t.value}`")
            else:
                self.i -= 1
                fn.instrs.append(self.instruction(fn))
        for ins in fn.instrs:
            if ins.opcode == "bra":
                target = ins.operands[0] if ins.operands else None
                if not isinstance(target, Sym) or target.name not in fn.labels:
                    raise PtxSyntaxError(
                        f"{self.name}:{ins.line}: branch to an undefined label\n    {ins.text}")

    def shared_var(self, extern: bool) -> SharedVar:
        """`.shared [.align N] .type name[count]...;` (the `.shared` is already consumed)."""
        start = self.peek()
        align = None
        if self.accept(".align"):
            align_tok = self.expect_kind("num", "an alignment")
            alignment = _number(align_tok)
            align = alignment.value
            if alignment.kind != "int" or align <= 0 or align & (align - 1):
                raise self.syntax(align_tok, "shared-memory alignment must be a positive power of two")
        ty_tok = self.expect_kind("word", "a variable type")
        ty = ty_tok.value.lstrip(".")
        if ty not in NP_TYPES or ty == "pred":
            raise self.unsupported(ty_tok, f"unsupported variable type `{ty_tok.value}`")
        name = self.expect_kind("word", "a variable name")
        count, unsized = 1, False
        while self.accept("["):
            if self.accept("]"):
                unsized = True
            else:
                size_tok = self.expect_kind("num", "an array size")
                dimension = _number(size_tok)
                if dimension.kind != "int" or dimension.value < 0:
                    raise self.syntax(size_tok, "shared-memory array size must be a non-negative integer")
                if dimension.value == 0:
                    unsized = True
                else:
                    count *= dimension.value
                self.expect("]")
        if self.peek().value == "=":
            raise self.syntax(self.peek(), ".shared variables cannot have initializers")
        self.expect(";")
        if unsized and not extern:
            raise self.syntax(name, f"`{name.value}` needs an array size (only .extern arrays may omit it)")
        item = NP_TYPES[ty].itemsize
        return SharedVar(name.value, align or item, None if unsized else count * item, start.line)

    def reg_decl(self, fn: Function) -> None:
        ty_tok = self.expect_kind("word", "a register type")
        ty = ty_tok.value.lstrip(".")
        if ty not in NP_TYPES:
            raise self.unsupported(ty_tok, f"unsupported register type `{ty_tok.value}`")
        while True:
            name = self.expect_kind("word", "a register name")
            if not name.value.startswith("%"):
                raise self.syntax(name, f"register names start with '%', found {name.value!r}")
            if self.accept("<"):
                count = int(self.expect_kind("num", "a register count").value)
                self.expect(">")
                for k in range(count):
                    fn.regs[f"{name.value}{k}"] = ty
            else:
                fn.regs[name.value] = ty
            if self.accept(";"):
                return
            self.expect(",")

    def instruction(self, fn: Function) -> Instr:
        first = self.peek()
        pred, negated = None, False
        if self.accept("@"):
            negated = self.accept("!")
            p = self.expect_kind("word", "a predicate register")
            if fn.regs.get(p.value) != "pred":
                raise self.syntax(p, f"`{p.value}` is not a declared .pred register")
            pred = Reg(p.value)
        op = self.expect_kind("word", "an instruction")
        opcode, *mods = op.value.split(".")
        operands = []
        if not self.accept(";"):
            while True:
                operand = self.operand(fn)
                if self.accept("|"):
                    second = self.operand(fn)
                    if not (isinstance(operand, Reg) and isinstance(second, Reg)):
                        raise self.syntax(first, "both sides of a paired destination must be registers")
                    operand = Pair(operand, second)
                operands.append(operand)
                if self.accept(";"):
                    break
                self.expect(",")
        return Instr(opcode, tuple(mods), tuple(operands), pred, negated,
                     first.line, self.source(first.line))

    def operand(self, fn: Function):
        t = self.next()
        if t.kind == "num":
            return _number(t)
        if t.value == "[":
            base = self.operand(fn)
            if not isinstance(base, (Reg, Sym, Imm)):
                raise self.syntax(t, "invalid address expression")
            offset = 0
            if self.accept("+"):
                offset = _number(self.expect_kind("num", "an offset")).value
            elif self.peek().kind == "num":
                offset = _number(self.next()).value
            self.expect("]")
            return Mem(base, offset)
        if t.value == "{":
            elems = []
            while True:
                elems.append(self.operand(fn))
                if self.accept("}"):
                    break
                self.expect(",")
            return Vec(tuple(elems))
        if t.kind != "word":
            raise self.syntax(t, f"expected an operand, found {t.value!r}")
        if t.value.startswith("%"):
            if t.value in fn.regs:
                return Reg(t.value)
            base, _, comp = t.value[1:].partition(".")
            if base in SPECIAL_REGISTERS:
                if SPECIAL_REGISTERS[base] and comp not in ("x", "y", "z"):
                    raise self.unsupported(t, f"`{t.value}` without a .x/.y/.z component is not supported yet")
                return Special(t.value[1:])
            raise self.syntax(t, f"undeclared register `{t.value}`")
        return Sym(t.value)


def parse_ptx(text: str, name: str = "<ptx>") -> Module:
    """Parse PTX source text. `name` appears in error messages."""
    return _Parser(text, name).module()


def parse_ptx_file(path: str | Path) -> Module:
    path = Path(path)
    return parse_ptx(path.read_text(), name=path.name)
