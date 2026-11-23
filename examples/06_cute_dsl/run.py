"""CuTe DSL (the Python front end of CUTLASS) to warplens, by way of a PTX file.

    pip install nvidia-cutlass-dsl
    python examples/06_cute_dsl/run.py

The DSL compiles for sm_100a without a GPU when told the architecture, and keeps the PTX
it generated when asked. warplens then runs that PTX. This is a spike, not an
integration: the DSL's own launch path talks to the CUDA driver, which warplens does not
emulate, so the grid, block and arguments are repeated by hand below.
"""

import os
import sys
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp(prefix="warplens_cute_dsl_"))
os.environ.update(CUTE_DSL_ARCH="sm_100a",       # compile for Blackwell, whatever GPU is present
                  CUTE_DSL_KEEP="ptx",           # keep the generated PTX ...
                  CUTE_DSL_DUMP_DIR=str(work),   # ... in this directory
                  CUDA_VISIBLE_DEVICES="")       # and prove no GPU is needed

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    sys.exit("this example needs the CuTe DSL: pip install nvidia-cutlass-dsl")

import warplens  # noqa: E402


@cute.kernel
def kernel(n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < 3:
        cute.printf("thread %d says n = %d\n", tidx, n)


@cute.jit
def host(n: cutlass.Int32):
    kernel(n).launch(grid=(1, 1, 1), block=(32, 1, 1))


cute.compile(host, 7)  # compiles; never launched through the DSL
(ptx,) = work.glob("*.ptx")
print(f"CuTe DSL wrote {ptx.name} ({len(ptx.read_text().splitlines())} lines of PTX)")

sim = warplens.Device("B200")
module = sim.load_ptx(ptx)
(name,) = module.functions
stats = sim.launch(module, name, grid=1, block=32, args=[7])  # the launch, repeated by hand
print(f"warplens ran `{name}`; the kernel printed:")
print(stats.output, end="")
