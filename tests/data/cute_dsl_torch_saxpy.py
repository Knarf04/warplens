"""A CuTe DSL kernel over PyTorch tensors, shaped like fp16_gemm_0.py's host side: tensors made on
the CPU and moved with .to(device), from_dlpack, the launch, then a PyTorch reference on the GPU."""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

N, BLOCK = 4096, 64


@cute.kernel
def saxpy(a: cutlass.Float32, x: cute.Tensor, y: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * BLOCK + tidx
    y[i] = a * x[i] + y[i]


@cute.jit
def host(a: cutlass.Float32, x: cute.Tensor, y: cute.Tensor):
    saxpy(a, x, y).launch(grid=(N // BLOCK, 1, 1), block=(BLOCK, 1, 1))


torch.manual_seed(1111)
make = lambda: torch.empty(N, dtype=torch.int32).random_(-2, 2).to(dtype=torch.float32, device="cuda")
x, y = make(), make()
before = y.clone()
host(2.5, from_dlpack(x), from_dlpack(y))
reference = (2.5 * x.to(torch.float64) + before.to(torch.float64)).cpu()
torch.testing.assert_close(y.cpu().to(torch.float64), reference)
print("PASS")
