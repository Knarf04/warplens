"""Hand-written PTX: a module around a function body, and one instruction run per lane."""

import numpy as np

import warplens

HEADER = ".version 9.5\n.target sm_100a\n.address_size 64\n"


def kernel(body: str, params: str = "", name: str = "k") -> warplens.frontend.ir.Module:
    """Wrap a function body in a module with one entry called `name`."""
    return warplens.parse_ptx(f"{HEADER}.visible .entry {name}({params})\n{{\n{body}\n}}\n",
                              name="test.ptx")


def run_lanes(sim, instr: str, in_types, out_type: str, inputs):
    """Run one instruction per lane: out[i] = instr(in0[i], in1[i], ...).

    `instr` uses %a0, %a1, ... for the inputs and %d for the result, e.g.
    "div.s32 %d, %a0, %a1;". Returns the output array.
    """
    n = len(inputs[0])
    bits = lambda ty: 1 if ty == "pred" else int(ty[1:])
    params = ", ".join(f".param .u64 p{i}" for i in range(len(inputs) + 1))
    lines = [".reg .b64 %base, %addr, %off;", ".reg .b32 %t;",
             f".reg .b{bits(out_type)} %d;", "mov.u32 %t, %tid.x;"]
    for i, ty in enumerate(in_types):
        size = bits(ty) // 8
        lines += [f".reg .b{bits(ty)} %a{i};",
                  f"ld.param.u64 %base, [p{i}];",
                  f"mul.wide.u32 %off, %t, {size};",
                  "add.u64 %addr, %base, %off;",
                  f"ld.global.b{bits(ty)} %a{i}, [%addr];"]
    out_size = bits(out_type) // 8
    lines += [instr,
              f"ld.param.u64 %base, [p{len(inputs)}];",
              f"mul.wide.u32 %off, %t, {out_size};",
              "add.u64 %addr, %base, %off;",
              f"st.global.b{bits(out_type)} [%addr], %d;",
              "ret;"]
    # Declarations must come first; keep statement order otherwise.
    decls = [l for l in lines if l.startswith(".reg")]
    code = [l for l in lines if not l.startswith(".reg")]
    mod = kernel("\n".join(decls + code), params)
    arrays = [sim.from_numpy(np.asarray(x)) for x in inputs]
    out = sim.empty(n, out_type)
    sim.launch(mod, "k", grid=1, block=n, args=[*arrays, out])
    return out.numpy()
