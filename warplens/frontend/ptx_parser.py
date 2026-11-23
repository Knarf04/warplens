"""PTX text -> IR.

Covers the subset needed so far: module header directives, `.entry` functions,
`.reg` declarations, labels, and predicated instructions with register,
immediate, memory and vector operands. Anything else is reported as
UnsupportedError with its line, never skipped.
"""

from __future__ import annotations

import re
import struct

import numpy as np
from pathlib import Path

from ..errors import PtxSyntaxError, UnsupportedError
from .ir import (Function, GlobalVar, Imm, Instr, Mem, Module, NP_TYPES, Pair, Param, Reg,
                 SharedVar, Special, Sym, TensorRef, Vec)

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
    "clusterid": True, "nclusterid": True, "cluster_ctaid": True, "cluster_nctaid": True,
    "cluster_ctarank": False, "cluster_nctarank": False, "is_explicit_cluster": False,
}

_LINKAGE = {".visible", ".weak", ".extern", ".common"}
_IGNORED_BODY_DIRECTIVES = {".pragma"}


class _Tok:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind: str, value: str, line: int):
        self.kind, self.value, self.line = kind, value, line

    def __repr__(self) -> str:
        return f"{self.value!r}"


_DEBUG_LINE = re.compile(r"^[ \t]*\.(?:loc|file)\b[^\n]*$", re.M)
_DEBUG_SECTION = re.compile(r"^[ \t]*\.section\b[^{\n]*\n?[ \t]*\{.*?\}", re.M | re.S)


def _strip_debug_info(text: str) -> str:
    """Blank out DWARF debug directives (`.loc`, `.file`, `.section .debug_* { ... }`),
    which compilers such as Triton emit, keeping every line where it was."""
    def blank(match):
        return "\n" * match.group().count("\n")
    return _DEBUG_SECTION.sub(blank, _DEBUG_LINE.sub("", text))


def _tokenize(text: str, name: str) -> list[_Tok]:
    text = _strip_debug_info(text)
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
        # Nested { } blocks (inline assembly) have their own variables and labels.
        # Scope 0 is the function body; names declared in scope k > 0 are stored
        # as "name@k" so they cannot clash with the enclosing ones.
        self.scopes = [0]
        self.next_scope = 1
        self.branches: list[tuple[Instr, list[int], _Tok]] = []
        self.shared: dict[str, SharedVar] = {}
        self.locals: dict[str, SharedVar] = {}

    def scoped(self, name: str, scope: int) -> str:
        return name if scope == 0 else f"{name}@{scope}"

    def find(self, name: str, table) -> str | None:
        """The stored name of `name` in the innermost scope that declares it."""
        for scope in reversed(self.scopes):
            key = self.scoped(name, scope)
            if key in table:
                return key
        return None

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
            if t.value not in _LINKAGE and t.value not in (".shared", ".func"):
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
            elif t.value in (".global", ".const"):
                mod.globals.append(self.global_var(t))
            elif t.value == ".func" and extern:
                mod.extern_funcs.add(self.extern_func(t))
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
        self.scopes, self.next_scope, self.branches = [0], 1, []
        self.shared = {}
        self.locals = {}
        self.body(fn)
        for ins, scopes, tok in self.branches:
            target = ins.operands[0] if len(ins.operands) == 1 else None
            saved, self.scopes = self.scopes, scopes
            key = self.find(target.name, fn.labels) if isinstance(target, Sym) else None
            self.scopes = saved
            if key is None:
                raise PtxSyntaxError(
                    f"{self.name}:{ins.line}: branch to an undefined label\n    {ins.text}")
            ins.operands = (Sym(key),)
        return fn

    def skip_directive(self, directive: _Tok) -> None:
        while (end := self.next()).value != ";":
            if end.kind == "eof" or end.value in ("{", "}"):
                raise self.syntax(directive,
                                  f"directive `{directive.value}` is missing its terminating ';'")

    def param(self) -> Param:
        self.expect(".param")
        align = 0
        if self.accept(".align"):  # by-value structs: `.param .align 8 .b8 name[24]`
            align = int(self.expect_kind("num", "an alignment").value)
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
        size, by_value = NP_TYPES[ty].itemsize, False
        while self.accept("["):  # a struct passed by value: `.param .align 8 .b8 name[24]`
            size *= int(self.expect_kind("num", "an array size").value)
            self.expect("]")
            by_value = True
        return Param(name=t.value, type=ty, is_ptr=is_ptr, size=size, by_value=by_value,
                     align=align if by_value else 0)

    def extern_func(self, start: _Tok) -> str:
        """`.extern .func (ret) name (params);` Only the name is kept."""
        name = None
        depth = 0
        while (t := self.next()).value != ";" or depth:
            if t.kind == "eof" or t.value in ("{", "}"):
                raise self.unsupported(start, "only external function declarations are supported")
            if t.value == "(":
                depth += 1
            elif t.value == ")":
                depth -= 1
            elif depth == 0 and t.kind == "word" and name is None:
                name = t.value
        if name is None:
            raise self.syntax(start, "function declaration without a name")
        return name

    def global_var(self, start: _Tok) -> GlobalVar:
        """`.global [.align N] .type name[count] [= {v, ...}];`"""
        align = None
        if self.accept(".align"):
            align = int(self.expect_kind("num", "an alignment").value)
        ty_tok = self.expect_kind("word", "a variable type")
        ty = ty_tok.value.lstrip(".")
        if ty not in NP_TYPES or ty == "pred":
            raise self.unsupported(ty_tok, f"unsupported variable type `{ty_tok.value}`")
        dt = NP_TYPES[ty]
        name = self.expect_kind("word", "a variable name")
        count = 1
        while self.accept("["):
            count *= int(self.expect_kind("num", "an array size").value)
            self.expect("]")
        values = np.zeros(count, dtype=dt)
        if self.accept("="):
            braces = self.accept("{")
            k = 0
            while True:
                tok = self.next()
                if tok.kind != "num":
                    raise self.unsupported(tok, "only numeric initializers are supported")
                if k >= count:
                    raise self.syntax(tok, f"too many initializers for `{name.value}`")
                imm = _number(tok)
                if imm.kind == "int":
                    if dt.kind == "f":
                        values[k] = imm.value
                    else:
                        values.view(f"u{dt.itemsize}")[k] = imm.value % (1 << (8 * dt.itemsize))
                else:
                    width = 4 if imm.kind == "f32" else 8
                    as_float = np.array([imm.value], dtype=f"u{width}").view(f"f{width}")[0]
                    if dt.kind == "f":
                        values[k] = as_float
                    elif dt.itemsize == width:
                        values.view(f"u{width}")[k] = imm.value
                    else:
                        raise self.syntax(tok, "floating-point initializer for an integer of another size")
                k += 1
                if not braces or self.accept("}"):
                    break
                self.expect(",")
        self.expect(";")
        return GlobalVar(name.value, align or dt.itemsize, values.tobytes(), start.line)

    # -- function body -----------------------------------------------------
    def body(self, fn: Function) -> None:
        while True:
            t = self.next()
            if t.kind == "eof":
                raise self.syntax(t, f"function `{fn.name}` is missing its closing '}}'")
            if t.value == "}":
                break
            if t.value == "{":
                self.scopes.append(self.next_scope)
                self.next_scope += 1
                self.body(fn)
                self.scopes.pop()
                continue
            if t.kind == "word" and self.peek().value == ":":
                self.next()
                key = self.scoped(t.value, self.scopes[-1])
                if key in fn.labels:
                    raise self.syntax(t, f"label `{t.value}` defined twice")
                fn.labels[key] = len(fn.instrs)
            elif t.value == ".reg":
                self.reg_decl(fn)
            elif t.value == ".shared":
                var = self.shared_var(extern=False)
                var.name = self.scoped(var.name, self.scopes[-1])
                self.shared[var.name] = var
                fn.shared.append(var)
            elif t.value == ".local":
                var = self.shared_var(extern=False)
                var.name = self.scoped(var.name, self.scopes[-1])
                self.locals[var.name] = var
                fn.locals.append(var)
            elif t.value == ".param":  # storage for the arguments of a call
                ty_tok = self.expect_kind("word", "a parameter type")
                ty = ty_tok.value.lstrip(".")
                if ty not in NP_TYPES or ty == "pred":
                    raise self.unsupported(ty_tok, f"unsupported parameter type `{ty_tok.value}`")
                name = self.expect_kind("word", "a parameter name")
                if self.peek().value == "[" or self.peek().value.startswith("."):
                    raise self.unsupported(name, "only scalar call parameters are supported")
                self.expect(";")
                fn.call_params[self.scoped(name.value, self.scopes[-1])] = ty
            elif t.value in _IGNORED_BODY_DIRECTIVES:
                self.skip_directive(t)
            elif t.kind == "word" and t.value.startswith("."):
                raise self.unsupported(t, f"unsupported directive `{t.value}`")
            else:
                self.i -= 1
                ins = self.instruction(fn)
                fn.instrs.append(ins)
                if ins.opcode == "bra":
                    self.branches.append((ins, list(self.scopes), t))

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
            scope = self.scopes[-1]
            if self.accept("<"):
                count = int(self.expect_kind("num", "a register count").value)
                self.expect(">")
                for k in range(count):
                    fn.regs[self.scoped(f"{name.value}{k}", scope)] = ty
            else:
                fn.regs[self.scoped(name.value, scope)] = ty
            if self.accept(";"):
                return
            self.expect(",")

    def instruction(self, fn: Function) -> Instr:
        first = self.peek()
        pred, negated = None, False
        if self.accept("@"):
            negated = self.accept("!")
            p = self.expect_kind("word", "a predicate register")
            key = self.find(p.value, fn.regs)
            if key is None or fn.regs[key] != "pred":
                raise self.syntax(p, f"`{p.value}` is not a declared .pred register")
            pred = Reg(key)
        op = self.expect_kind("word", "an instruction")
        opcode, *mods = op.value.split(".")
        operands = []
        if not self.accept(";"):
            while True:
                operand = self.operand(fn)
                if self.accept("|"):
                    second = self.operand(fn)
                    discard = Sym("_")  # `_|%p`: that half of the result is not wanted
                    if not all(isinstance(x, Reg) or x == discard for x in (operand, second)):
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
            if self.accept(","):  # [tensor_map, {c0, c1, ...}]
                coords = self.operand(fn)
                if not isinstance(coords, Vec):
                    raise self.syntax(t, "expected {coordinates} after the tensor map")
                self.expect("]")
                return TensorRef(base, coords.elems)
            offset = 0
            if self.accept("+"):
                offset = _number(self.expect_kind("num", "an offset")).value
            elif self.peek().kind == "num":
                offset = _number(self.next()).value
            self.expect("]")
            return Mem(base, offset)
        if t.value in ("{", "("):  # {a, b} vector; (a, b) argument or return list of a call
            close = "}" if t.value == "{" else ")"
            elems = []
            if not self.accept(close):
                while True:
                    elems.append(self.operand(fn))
                    if self.accept(close):
                        break
                    self.expect(",")
            return Vec(tuple(elems))
        if t.kind != "word":
            raise self.syntax(t, f"expected an operand, found {t.value!r}")
        # Registers and memory variables can shadow each other. Resolve them
        # in lexical order, so an outer register cannot hide an inner variable.
        for scope in reversed(self.scopes):
            key = self.scoped(t.value, scope)
            if key in fn.regs:
                return Reg(key)
            if key in self.shared or key in self.locals:
                return Sym(key)
        if t.value.startswith("%"):
            base, _, comp = t.value[1:].partition(".")
            if base in SPECIAL_REGISTERS:
                if SPECIAL_REGISTERS[base] and comp not in ("x", "y", "z"):
                    raise self.unsupported(t, f"`{t.value}` without a .x/.y/.z component is not supported yet")
                return Special(t.value[1:])
            raise self.syntax(t, f"undeclared register `{t.value}`")
        return Sym(self.find(t.value, fn.call_params) or t.value)


def parse_ptx(text: str, name: str = "<ptx>") -> Module:
    """Parse PTX source text. `name` appears in error messages."""
    return _Parser(text, name).module()


def parse_ptx_file(path: str | Path) -> Module:
    path = Path(path)
    return parse_ptx(path.read_text(), name=path.name)
