"""A CuTe DSL kernel (the Python front end of CUTLASS) that runs unchanged on a GPU and in warplens.

    pip install nvidia-cutlass-dsl
    python examples/06_cute_dsl/run.py                 # on the GPU in this machine
    warplens run python examples/06_cute_dsl/run.py    # in the simulator, as a B200

The DSL compiles `host` to machine code that loads the kernel and launches it through the
CUDA runtime. Under `warplens run` those calls reach the simulator, and the DSL is told to
compile for sm_100a and to keep the PTX it generates, because the module it loads holds only
machine code. Nothing in this script knows about any of that.
"""

import sys

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    sys.exit("this example needs the CuTe DSL: pip install nvidia-cutlass-dsl")


@cute.kernel
def kernel(n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if tidx < 3:
        cute.printf("thread %d of block %d says n = %d\n", tidx, bidx, n)


@cute.jit
def host(n: cutlass.Int32):
    kernel(n).launch(grid=(2, 1, 1), block=(32, 1, 1))


host(7)
print("launched 2 blocks of 32 threads")
