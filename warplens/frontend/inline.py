"""Calls to device functions, removed by putting the callee's body at each call.

    .func (.param .b32 func_retval0) twice(.param .b32 twice_param_0) { ... ret; }
    ...
    st.param.b32 [param0], %r1;
    call.uni (retval0), twice, (param0);
    ld.param.b32 %r2, [retval0];

The callee reads its parameters from the storage the caller filled (`param0`) and stores its
result where the caller will load it (`retval0`), so those names replace the callee's own.
Its registers, labels and variables get a prefix that is unique to the call; its local
variables become more local variables of the caller, a frame per call site. `ret` becomes a
branch to the instruction after the call. Lanes that return at different points therefore
rejoin there, by the same analysis that handles any branch.

A struct passed or returned by value (`.param .align 16 .b8 param0[40]`) needs nothing more:
the parser makes such storage a local variable, and the renaming above points the callee at it.

A call under a predicate, `@%p call ...`, is the same body behind a branch that the other
lanes take around it.

An indirect call names a register instead of a function:

    prototype_7 : .callprototype (.param .b32 _) _ (.param .b32 _);
    call (retval0), %rd13, (param0), prototype_7;

Every function of the module that takes these arguments could be meant (or only those a
`.calltargets` label lists). Each is inlined behind a comparison of the register with that
function's address. Lanes whose register holds none of them reach the original instruction,
which reports the address.

Recursion cannot be removed this way. A call to a function whose body is already being
inlined around it becomes `callrec`, which at run time saves that copy's registers, parameter
storage and local variables, and enters the copy again; `callret`, placed after the copy,
restores them and continues after the `callrec` (units/call.py). The `Frame`s say which names
make up each such copy.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import UnsupportedError
from .ir import NP_TYPES, Frame, Function, Imm, Instr, Mem, Module, Pair, Reg, Sym, TensorRef, Vec


def _rename(operand, regs: dict[str, str], syms: dict[str, str]):
    kind = type(operand)
    if kind is Reg:
        return Reg(regs.get(operand.name, operand.name))
    if kind is Sym:
        return Sym(syms.get(operand.name, operand.name))
    if kind is Mem:
        return Mem(_rename(operand.base, regs, syms), operand.offset)
    if kind is Vec:
        return Vec(tuple(_rename(e, regs, syms) for e in operand.elems))
    if kind is Pair:
        return Pair(_rename(operand.first, regs, syms), _rename(operand.second, regs, syms))
    if kind is TensorRef:
        return TensorRef(_rename(operand.base, regs, syms), tuple(_rename(c, regs, syms) for c in operand.coords))
    return operand


def _call_parts(ins: Instr):
    """(return names, callee, argument names, table) of a call. The callee is a `Sym` for a
    direct call and a `Reg` for an indirect one, whose `table` names its `.callprototype` or
    `.calltargets`. None if the instruction has another shape."""
    ops = list(ins.operands)
    rets = ops.pop(0).elems if ops and isinstance(ops[0], Vec) else ()
    if not ops or not isinstance(ops[0], (Sym, Reg)):
        return None
    args = ops[1].elems if len(ops) > 1 and isinstance(ops[1], Vec) else ()
    table = ops[2].name if len(ops) > 2 and isinstance(ops[2], Sym) else None
    return rets, ops[0], args, table


def _storage(fn: Function) -> dict[str, tuple[int, bool]]:
    """Call-parameter storage of `fn`: name -> (bytes, whether it is an array)."""
    arrays = {v.name: v.size for v in fn.locals}
    return {name: (arrays[name], True) if name in arrays else (NP_TYPES[ty].itemsize, False)
            for name, ty in fn.call_params.items()}


def _fits(callee: Function, rets, args, storage) -> bool:
    """Whether a call with this storage for its arguments and results can be to `callee`."""
    if len(args) != len(callee.params) or len(rets) != len(callee.rets):
        return False
    for given, param in zip((*args, *rets), (*callee.params, *callee.rets)):
        if not isinstance(given, Sym) or storage.get(given.name) != (param.size, param.by_value):
            return False
    return True


def inline_calls(module: Module, fn: Function, _active: tuple[str, ...] = ()) -> Function:
    """`fn` with every call to a function defined in `module` replaced by that function's
    body. Returns `fn` itself when it makes no such call."""
    storage = _storage(fn)

    def targets(ins) -> list[str]:
        """The functions of the module that this instruction may call."""
        parts = _call_parts(ins) if ins.opcode == "call" else None
        if parts is None:
            return []
        rets, callee, args, table = parts
        if isinstance(callee, Sym):
            return [callee.name] if callee.name in module.device_functions else []
        names = module.device_functions
        if table in fn.calltargets:
            names = fn.calltargets[table]
            unknown = [n for n in names if n not in module.device_functions]
            if unknown:
                raise UnsupportedError(
                    f"indirect call may target `{unknown[0]}`, which is not a function defined in this "
                    f"module\n  {module.name}:{ins.line}: {ins.text}")
        return [n for n in names if _fits(module.device_functions[n], rets, args, storage)]

    if not any(targets(i) for i in fn.instrs):
        return fn
    out = Function(name=fn.name, params=fn.params, rets=fn.rets, regs=dict(fn.regs), shared=list(fn.shared),
                   locals=list(fn.locals), call_params=dict(fn.call_params), line=fn.line,
                   prototypes=fn.prototypes, calltargets=fn.calltargets, frames=dict(fn.frames))
    at = {index: [] for index in range(len(fn.instrs) + 1)}  # labels of the caller, by old index
    for label, index in fn.labels.items():
        at[index].append(label)

    def emit(ins, opcode, mods, operands, pred=None, negated=False):
        out.instrs.append(Instr(opcode, mods, operands, pred, negated, ins.line, ins.text))

    def expand(ins, name: str, rets, args) -> None:
        """Put the body of `name` here, reading `args` and writing `rets`."""
        where = f"{module.name}:{ins.line}: {ins.text}"
        definition = module.device_functions[name]
        if not _fits(definition, rets, args, storage):
            raise UnsupportedError(f"call to `{name}` does not match its definition\n  {where}")
        if name in _active or name == fn.name:
            # Recursion. The copy of `name` that this code is being inlined into is entered again;
            # which copy that is gets filled in when that copy is placed (below).
            emit(ins, "callrec", (), (Vec(tuple(rets)), Sym(name), Vec(tuple(args))))
            return
        callee = inline_calls(module, definition, (*_active, fn.name))

        prefix = f"{name}#{len(out.instrs)}."
        regs = {reg: "%" + prefix + reg.lstrip("%") for reg in callee.regs}
        syms = {param.name: given.name for given, param in zip((*args, *rets), (*callee.params, *callee.rets))}
        for table in (callee.labels, callee.call_params, callee.frames):
            syms.update({key: prefix + key for key in table})
        for variable in (*callee.locals, *callee.shared):
            syms[variable.name] = prefix + variable.name
        entry, after = prefix + "entry", prefix + "return"

        out.regs.update({regs[reg]: ty for reg, ty in callee.regs.items()})
        out.call_params.update({syms[key]: ty for key, ty in callee.call_params.items()})
        out.locals += [replace(v, name=syms[v.name]) for v in callee.locals]
        out.shared += [replace(v, name=syms[v.name]) for v in callee.shared]
        for inner_prefix, frame in callee.frames.items():
            out.frames[syms[inner_prefix]] = Frame(
                frame.name, syms[frame.entry], tuple(regs[r] for r in frame.regs),
                *(tuple(syms[s] for s in names) for names in (frame.params, frame.locals, frame.args, frame.rets)))
        start = len(out.instrs)
        out.labels[entry] = start
        for label, target in callee.labels.items():
            out.labels[syms[label]] = start + target
        recursive = False
        for inner in callee.instrs:
            pred = Reg(regs[inner.pred.name]) if inner.pred is not None else None
            if inner.opcode == "ret":
                emit(inner, "bra", ("uni",) if pred is None else (), (Sym(after),), pred, inner.pred_negated)
            elif inner.opcode == "callrec":
                inner_rets, target, inner_args = inner.operands
                mods = inner.mods
                if mods:  # already tied to a copy inside the callee
                    target = Sym(syms[target.name])
                elif target.name == name:  # a call back to the copy being placed
                    target, mods, recursive = Sym(prefix), ("bound",), True
                emit(inner, "callrec", mods, (_rename(inner_rets, regs, syms), target, _rename(inner_args, regs, syms)))
            else:
                emit(inner, inner.opcode, inner.mods, tuple(_rename(o, regs, syms) for o in inner.operands),
                     pred, inner.pred_negated)
        out.labels[after] = len(out.instrs)
        if recursive:
            emit(ins, "callret", (), (Sym(prefix),))
            arrays = {v.name for v in callee.locals}
            out.frames[prefix] = Frame(
                name, entry, tuple(regs.values()),
                params=tuple(syms[key] for key in callee.call_params if key not in arrays),
                locals=tuple(syms[v.name] for v in callee.locals),
                args=tuple(a.name for a in args), rets=tuple(r.name for r in rets))

    for index, ins in enumerate(fn.instrs):
        for label in at[index]:
            out.labels[label] = len(out.instrs)
        names = targets(ins)
        if not names:
            out.instrs.append(ins)
            continue
        rets, callee, args, _ = _call_parts(ins)
        site = f"call#{len(out.instrs)}."
        done = site + "done"
        if ins.pred is not None:  # lanes that do not make the call go around it
            emit(ins, "bra", (), (Sym(done),), ins.pred, not ins.pred_negated)
        if isinstance(callee, Sym):
            expand(ins, callee.name, rets, args)
        else:
            other = "%" + site + "other"
            out.regs[other] = "pred"
            for name in names:
                skip = f"{site}not.{name}"
                emit(ins, "setp", ("ne", "u64"), (Reg(other), callee, Imm("int", module.function_address(name))))
                emit(ins, "bra", (), (Sym(skip),), Reg(other))
                expand(ins, name, rets, args)
                emit(ins, "bra", ("uni",), (Sym(done),))
                out.labels[skip] = len(out.instrs)
            out.instrs.append(replace(ins, pred=None))  # no function has that address: reported there
        out.labels[done] = len(out.instrs)
    for label in at[len(fn.instrs)]:
        out.labels[label] = len(out.instrs)
    return out
