"""A CuTe DSL kernel with arrays, unchanged on a GPU and in warplens.

    python examples/06_cute_dsl/saxpy.py                 # on the GPU in this machine
    warplens run python examples/06_cute_dsl/saxpy.py    # in the simulator, as a B200

DSL examples usually hold their arrays in PyTorch tensors on the GPU. PyTorch has NVIDIA's
CUDA runtime linked in and cannot run in the simulator, so this script holds them the way the
DSL itself does underneath: device memory from the CUDA driver, through the DSL's helpers,
passed to the kernel as pointers. The driver calls reach the simulator under `warplens run`.
"""

import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.runtime import cuda as gpu
from cutlass.cute.runtime import make_ptr

N, BLOCK = 256, 64


@cute.kernel
def saxpy(a: cutlass.Float32, x: cute.Tensor, y: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * BLOCK + tidx
    y[i] = a * x[i] + y[i]


@cute.jit
def host(a: cutlass.Float32, x_ptr: cute.Pointer, y_ptr: cute.Pointer):
    layout = cute.make_layout(N)
    x = cute.make_tensor(x_ptr, layout)
    y = cute.make_tensor(y_ptr, layout)
    saxpy(a, x, y).launch(grid=(N // BLOCK, 1, 1), block=(BLOCK, 1, 1))


gpu.initialize_cuda_context()
rng = np.random.default_rng(0)
a, b = rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32)
x, y = gpu.allocate(a.nbytes), gpu.allocate(b.nbytes)
gpu.memcpy_h2d(a.ctypes.data, x, a.nbytes)
gpu.memcpy_h2d(b.ctypes.data, y, b.nbytes)
host(2.5, make_ptr(cutlass.Float32, int(x), cute.AddressSpace.gmem),
     make_ptr(cutlass.Float32, int(y), cute.AddressSpace.gmem))
out = np.empty_like(b)
gpu.memcpy_d2h(out.ctypes.data, y, out.nbytes)
gpu.deallocate(x)
gpu.deallocate(y)
np.testing.assert_allclose(out, np.float32(2.5) * a + b, rtol=1e-6)
print(f"saxpy over {N} elements: matches NumPy")
