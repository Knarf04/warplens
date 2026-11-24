"""Where do lanes of a warp rejoin when they reach one instruction by different paths?

    python validation/ampere/reconvergence_golden.py > validation/ampere/reconvergence_golden.txt

Every kernel is `if (tid == 0) shared = 7;` followed by a join that all or some lanes reach:
`[__syncthreads();] out[tid] = shared;`. One warp runs it. A lane that passes the join must
read 7, which it does only if lane 0 got there with it and not after it. The layouts differ
in how the lanes other than lane 0 get to the join:

  A  the join is the branch target
  B  lane 0 jumps to the join, the others fall into it from a block of their own
  C  as B, and the others' block ends in a conditional branch past the join
  D  the others' block is at the end of the function and branches back to the join

In C and D the first branch's immediate post-dominator is not the join but the code after it
(or the end of the function), because the second branch can go around the join. Each of the
two is recorded with the second branch taken by no lane (`all`) and by lanes 16 to 31
(`half`), and with the code after the join being a bare `ret` or a store of 5 to
out[32 + tid] (`work`). Everything is recorded with and without `bar.sync` at the join.

The kernels are PTX text because the layouts are the point and a compiler picks its own (C
and D are what nvcc 13.0 made of two probes in validation/blackwell). They are loaded with
the driver API through `cuda.bindings`, which the `warplens` conda env has; running this
needs a GPU. Each kernel is launched 100 times and must give the same 64 words every time.
One line per kernel: its name, then out[0..63] (99 where nothing was stored).

tests/test_hardware_golden.py runs the same PTX in the simulator against the recorded file.
"""
import sys

HEADER = ".version 7.8\n.target sm_86\n.address_size 64\n"
DECL = """
.visible .entry k(.param .u64 out)
{
    .reg .pred %p, %q;
    .reg .b32 %t, %v, %seven, %five, %dummy;
    .reg .b64 %base, %off, %addr;
    .shared .align 4 .b8 x[4];
    mov.u32 %t, %tid.x;
    ld.param.u64 %base, [out];
    mul.wide.u32 %off, %t, 4;
    add.u64 %addr, %base, %off;
    setp.ne.s32 %p, %t, 0;
"""
THEN = """
    mov.b32 %seven, 7;
    st.shared.u32 [x], %seven;
"""
WORK = """
    mov.b32 %five, 5;
    st.global.u32 [%addr+128], %five;
"""


def _join(barrier: bool) -> str:
    return ("\nJOIN:\n" + ("    bar.sync 0;\n" if barrier else "")
            + "    ld.shared.u32 %v, [x];\n    st.global.u32 [%addr], %v;\n")


def _after(work: bool) -> str:
    return "AFTER:\n" + (WORK if work else "") + "    ret;\n"


def kernels() -> dict[str, str]:
    """Name -> PTX body (without the header)."""
    out = {}
    for barrier in (True, False):
        tag = "bar" if barrier else "nobar"
        join = _join(barrier)
        out[f"A_{tag}"] = DECL + "    @%p bra JOIN;" + THEN + join + _after(False) + "}\n"
        out[f"B_{tag}"] = (DECL + "    @%p bra ELSE;" + THEN + "    bra.uni JOIN;\nELSE:\n    mov.b32 %dummy, 0;"
                           + join + _after(False) + "}\n")
        for who, last in (("all", 31), ("half", 15)):  # the last lane that goes to the join
            for work in (False, True):
                name = f"{who}_{'work' if work else 'ret'}_{tag}"
                out[f"C_{name}"] = (DECL + "    @%p bra ELSE;" + THEN + "    bra.uni JOIN;\nELSE:\n"
                                    f"    setp.gt.u32 %q, %t, {last};\n    @%q bra AFTER;"
                                    + join + _after(work) + "}\n")
                out[f"D_{name}"] = (DECL + "    @%p bra ELSE;" + THEN + join + _after(work) + "ELSE:\n"
                                    f"    setp.le.u32 %q, %t, {last};\n    @%q bra JOIN;\n"
                                    + ("    bra.uni AFTER;\n" if work else "    ret;\n") + "}\n")
    return out


def run_on_gpu(ptx: str, words: int = 64, fill: int = 99) -> list[int]:
    import numpy as np
    from cuda.bindings import driver

    def check(result):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"CUDA driver call failed: {error!r}")
        return values[0] if values else None

    module = check(driver.cuModuleLoadData(ptx.encode() + b"\0"))
    function = check(driver.cuModuleGetFunction(module, b"k"))
    device = check(driver.cuMemAlloc(4 * words))
    host = np.full(words, fill, dtype=np.uint32)
    check(driver.cuMemcpyHtoD(device, host.ctypes.data, host.nbytes))
    argument = np.array([int(device)], dtype=np.uint64)
    arguments = np.array([argument.ctypes.data], dtype=np.uint64)
    check(driver.cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, 0, arguments.ctypes.data, 0))
    check(driver.cuCtxSynchronize())
    check(driver.cuMemcpyDtoH(host.ctypes.data, device, host.nbytes))
    check(driver.cuMemFree(device))
    check(driver.cuModuleUnload(module))
    return host.tolist()


def main() -> int:
    from cuda.bindings import driver
    if int(driver.cuInit(0)[0]) != 0:
        print("cuInit failed: no usable GPU", file=sys.stderr)
        return 1
    error, device = driver.cuDeviceGet(0)
    try:
        error, context = driver.cuCtxCreate(None, 0, device)  # cuda-bindings 13
    except TypeError:
        error, context = driver.cuCtxCreate(0, device)
    if int(error) != 0:
        print(f"cuCtxCreate failed: {error!r}", file=sys.stderr)
        return 1
    for name, body in kernels().items():
        first = run_on_gpu(HEADER + body)
        for _ in range(99):
            again = run_on_gpu(HEADER + body)
            if again != first:
                print(f"{name}: the GPU gave different results on different launches:\n  {first}\n  {again}",
                      file=sys.stderr)
                return 1
        print(name, *first)
    return 0


if __name__ == "__main__":
    sys.exit(main())
