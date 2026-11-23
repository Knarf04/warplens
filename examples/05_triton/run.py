"""Triton kernels on a simulated Blackwell GPU: a vector add and a matmul.

    pip install triton
    python examples/05_triton/run.py

Triton compiles each kernel for sm_100a on this machine and warplens executes the PTX.
For the matmul, Triton emits a tcgen05 MMA: the same tensor core the CUTLASS examples use.
"""

import warnings

import numpy as np
import triton
import triton.language as tl

import warplens.triton_driver as wt

warnings.filterwarnings("ignore", module="triton")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def vector_add(gpu, n=1000, block=128):
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    out = gpu.empty(n, "float32")
    add_kernel[(triton.cdiv(n, block),)](gpu.tensor(a), gpu.tensor(b), out, n, BLOCK=block)
    return out.numpy(), a + b, gpu.last_stats


def matmul(gpu, M=256, N=128, K=96, block=(128, 64, 32)):
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    b = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    c = gpu.empty((M, N), "float32")
    grid = (triton.cdiv(M, block[0]), triton.cdiv(N, block[1]))
    matmul_kernel[grid](gpu.tensor(a), gpu.tensor(b), c, M, N, K, K, 1, N, 1, N, 1,
                        BLOCK_M=block[0], BLOCK_N=block[1], BLOCK_K=block[2])
    return c.numpy(), a.astype(np.float32) @ b.astype(np.float32), gpu.last_stats


if __name__ == "__main__":
    gpu = wt.install("B200")
    for name, run in (("vector add", vector_add), ("matmul", matmul)):
        got, reference, stats = run(gpu)
        print(f"{name}: max abs error vs NumPy {np.abs(got - reference).max():.3g}; "
              f"{stats.ctas} CTA(s), {stats.instructions} warp-instructions, {len(stats.findings)} finding(s)")
    wt.uninstall()
