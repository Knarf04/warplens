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

Not handled: recursion, indirect calls, a call under a predicate, and arguments that are not
scalars.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import UnsupportedError
from .ir import Function, Instr, Mem, Module, Pair, Reg, Sym, TensorRef, Vec


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
    """(return names, callee name, argument names) of a direct call, else None."""
    ops = list(ins.operands)
    rets = ops.pop(0).elems if ops and isinstance(ops[0], Vec) else ()
    if not ops or not isinstance(ops[0], Sym):
        return None
    args = ops[1].elems if len(ops) > 1 and isinstance(ops[1], Vec) else ()
    return rets, ops[0].name, args


def inline_calls(module: Module, fn: Function, _active: tuple[str, ...] = ()) -> Function:
    """`fn` with every call to a function defined in `module` replaced by that function's
    body. Returns `fn` itself when it makes no such call."""
    def is_local_call(ins):
        parts = _call_parts(ins) if ins.opcode == "call" else None
        return parts is not None and parts[1] in module.device_functions

    if not any(is_local_call(i) for i in fn.instrs):
        return fn
    out = Function(name=fn.name, params=fn.params, rets=fn.rets, regs=dict(fn.regs), shared=list(fn.shared),
                   locals=list(fn.locals), call_params=dict(fn.call_params), line=fn.line)
    at = {index: [] for index in range(len(fn.instrs) + 1)}  # labels of the caller, by old index
    for label, index in fn.labels.items():
        at[index].append(label)

    for index, ins in enumerate(fn.instrs):
        for label in at[index]:
            out.labels[label] = len(out.instrs)
        if not is_local_call(ins):
            out.instrs.append(ins)
            continue
        rets, name, args = _call_parts(ins)
        where = f"{module.name}:{ins.line}: {ins.text}"
        if name in _active or name == fn.name:
            raise UnsupportedError(f"recursive call to `{name}` is not supported\n  {where}")
        if ins.pred is not None:
            raise UnsupportedError(f"a call under a predicate is not supported\n  {where}")
        callee = inline_calls(module, module.device_functions[name], (*_active, fn.name))
        if len(args) != len(callee.params) or len(rets) != len(callee.rets):
            raise UnsupportedError(f"call to `{name}` does not match its definition\n  {where}")
        for given, param in zip((*args, *rets), (*callee.params, *callee.rets)):
            if not isinstance(given, Sym) or given.name not in fn.call_params or param.by_value:
                raise UnsupportedError(f"call to `{name}`: only scalar arguments are supported\n  {where}")

        prefix = f"{name}#{len(out.instrs)}."
        regs = {reg: "%" + prefix + reg.lstrip("%") for reg in callee.regs}
        syms = {param.name: given.name for given, param in zip((*args, *rets), (*callee.params, *callee.rets))}
        for table in (callee.labels, callee.call_params):
            syms.update({key: prefix + key for key in table})
        for variable in (*callee.locals, *callee.shared):
            syms[variable.name] = prefix + variable.name
        after = prefix + "return"

        out.regs.update({regs[reg]: ty for reg, ty in callee.regs.items()})
        out.call_params.update({syms[key]: ty for key, ty in callee.call_params.items()})
        out.locals += [replace(v, name=syms[v.name]) for v in callee.locals]
        out.shared += [replace(v, name=syms[v.name]) for v in callee.shared]
        start = len(out.instrs)
        for label, target in callee.labels.items():
            out.labels[syms[label]] = start + target
        for inner in callee.instrs:
            pred = Reg(regs[inner.pred.name]) if inner.pred is not None else None
            if inner.opcode == "ret":
                out.instrs.append(Instr("bra", ("uni",) if pred is None else (), (Sym(after),), pred,
                                        inner.pred_negated, inner.line, inner.text))
            else:
                out.instrs.append(Instr(inner.opcode, inner.mods,
                                        tuple(_rename(o, regs, syms) for o in inner.operands),
                                        pred, inner.pred_negated, inner.line, inner.text))
        out.labels[after] = len(out.instrs)
    for label in at[len(fn.instrs)]:
        out.labels[label] = len(out.instrs)
    return out
