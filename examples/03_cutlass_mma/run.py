"""Run CUTLASS's Blackwell tutorial 01 (a tcgen05 GEMM) on the CPU, unmodified.

    conda run -n warplens examples/03_cutlass_mma/build.sh    # once
    python examples/03_cutlass_mma/run.py [M N K]

The kernel computes D = alpha * A * B^T + beta * C with half-precision A (M x K)
and B (N x K) and float32 C and D, one 128 x 256 tile per CTA. warplens has no
libcuda shim yet, so this script plays the tutorial's host side: it builds the
kernel's by-value arguments by hand.
"""

import struct
import sys
from pathlib import Path

import numpy as np

import warplens

PTX = Path(__file__).parent / "tutorial_01.ptx"


def cute_tensor(array) -> bytes:
    """A CuTe Tensor with layout (rows, cols):(cols, 1) as the kernel receives it:
    shape, the one dynamic stride, then the pointer."""
    rows, cols = array.shape
    return struct.pack("<iii4xQ", rows, cols, cols, array.addr)


def run(M=128, N=256, K=64, alpha=1.0, beta=0.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (N, K)).astype(np.float16)
    C = rng.uniform(-1, 1, (M, N)).astype(np.float32)

    sim = warplens.Device("B200")
    dA, dB, dC = sim.from_numpy(A), sim.from_numpy(B), sim.from_numpy(C)
    dD = sim.empty((M, N), "f32")
    module = sim.load_ptx(PTX)
    entry = next(name for name in module.functions if "gemm_device" in name)
    # The TiledMMA argument carries the tcgen05 instruction descriptor:
    # float32 accumulators, M = 128 (stored / 16), N = 256 (stored / 8).
    idesc = (1 << 4) | ((256 >> 3) << 17) | ((128 >> 4) << 24)
    tiled_mma = struct.pack("<II", idesc, 1)
    stats = sim.launch(module, entry, grid=(-(-M // 128), -(-N // 256)), block=128, smem=64 * 1024,
                       args=[cute_tensor(dA), cute_tensor(dB), cute_tensor(dC), cute_tensor(dD),
                             b"\0", tiled_mma, b"\0", alpha, beta])
    reference = alpha * (A.astype(np.float32) @ B.astype(np.float32).T) + beta * C
    return dD.numpy(), reference, stats


if __name__ == "__main__":
    if not PTX.exists():
        sys.exit(f"{PTX.name} not found; run examples/03_cutlass_mma/build.sh first")
    dims = [int(x) for x in sys.argv[1:4]] or [256, 512, 128]
    got, reference, stats = run(*dims)
    error = np.abs(got - reference).max()
    print(f"GEMM {dims[0]} x {dims[1]} x {dims[2]}: max abs error vs NumPy {error:.3g} "
          f"(relative {error / np.abs(reference).max():.3g})")
    print(f"{stats.ctas} CTA(s), {stats.instructions} warp-instructions, "
          f"{len(stats.findings)} finding(s), {len(stats.output.splitlines())} line(s) printed by the kernel")
    for finding in stats.findings:
        print(finding)
