"""Run CUTLASS's first two Blackwell tutorials (tcgen05 GEMMs) on the CPU, unmodified.

    conda run -n warplens examples/03_cutlass_mma/build.sh    # once
    python examples/03_cutlass_mma/run.py [tutorial] [M N K]

Both kernels compute D = alpha * A * B^T + beta * C with half-precision A (M x K)
and B (N x K) and float32 C and D, one 128 x 256 tile per CTA.

    tutorial 1  01_mma_sm100.cu      threads copy A and B into shared memory
    tutorial 2  02_mma_tma_sm100.cu  TMA loads them, reporting to an mbarrier

warplens has no libcuda shim yet, so this script plays the tutorials' host side:
it builds the kernels' by-value arguments by hand.
"""

import struct
import sys
from pathlib import Path

import numpy as np

import warplens

HERE = Path(__file__).parent
TILE_M, TILE_N, TILE_K = 128, 256, 64


def ptx_path(tutorial: int) -> Path:
    return HERE / f"tutorial_0{tutorial}.ptx"


def gmem_tensor(array) -> bytes:
    """A CuTe global-memory Tensor with layout (rows, cols):(cols, 1) as the kernel
    receives it: shape, the one dynamic stride, then the pointer."""
    rows, cols = array.shape
    return struct.pack("<iii4xQ", rows, cols, cols, array.addr)


def coord_tensor(array) -> bytes:
    """What tutorial 2 receives instead for A and B: only the shape and stride.
    The data is reached through the TMA atom."""
    rows, cols = array.shape
    return struct.pack("<iii", rows, cols, cols)


def tma_atom(sim, array, tile_rows: int) -> bytes:
    """A CuTe TMA copy atom: the 128-byte tensor map, padded to the atom's 256 bytes.
    The box is one shared-memory tile, K-major with 128-byte swizzle, as the kernel
    lays out its operands for tcgen05.mma."""
    tensor_map = sim.tensor_map_tiled(array, (tile_rows, TILE_K), swizzle="128B")
    return tensor_map.tobytes().ljust(256, b"\0")


def run(tutorial=1, M=128, N=256, K=64, alpha=1.0, beta=0.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (N, K)).astype(np.float16)
    C = rng.uniform(-1, 1, (M, N)).astype(np.float32)

    sim = warplens.Device("B200")
    dA, dB, dC = sim.from_numpy(A), sim.from_numpy(B), sim.from_numpy(C)
    dD = sim.empty((M, N), "f32")
    module = sim.load_ptx(ptx_path(tutorial))
    entry = next(name for name in module.functions if "gemm_device" in name)
    # The TiledMMA argument carries the tcgen05 instruction descriptor:
    # float32 accumulators, M = 128 (stored / 16), N = 256 (stored / 8).
    idesc = (1 << 4) | ((TILE_N >> 3) << 17) | ((TILE_M >> 4) << 24)
    tiled_mma = struct.pack("<II", idesc, 1)
    if tutorial == 1:
        args = [gmem_tensor(dA), gmem_tensor(dB), gmem_tensor(dC), gmem_tensor(dD),
                b"\0", tiled_mma, b"\0", alpha, beta]
    else:
        args = [coord_tensor(dA), coord_tensor(dB), gmem_tensor(dC), gmem_tensor(dD),
                b"\0", tiled_mma, b"\0", tma_atom(sim, dA, TILE_M), tma_atom(sim, dB, TILE_N),
                alpha, beta]
    stats = sim.launch(module, entry, grid=(-(-M // TILE_M), -(-N // TILE_N)), block=128,
                       smem=64 * 1024, args=args)
    reference = alpha * (A.astype(np.float32) @ B.astype(np.float32).T) + beta * C
    return dD.numpy(), reference, stats


if __name__ == "__main__":
    numbers = [int(x) for x in sys.argv[1:]]
    tutorials = [numbers.pop(0)] if len(numbers) in (1, 4) else [1, 2]
    dims = numbers or [256, 512, 128]
    for tutorial in tutorials:
        if not ptx_path(tutorial).exists():
            sys.exit(f"{ptx_path(tutorial).name} not found; run examples/03_cutlass_mma/build.sh first")
        got, reference, stats = run(tutorial, *dims)
        error = np.abs(got - reference).max()
        print(f"tutorial {tutorial}, GEMM {dims[0]} x {dims[1]} x {dims[2]}: max abs error vs NumPy "
              f"{error:.3g} (relative {error / np.abs(reference).max():.3g})")
        print(f"  {stats.ctas} CTA(s), {stats.instructions} warp-instructions, "
              f"{len(stats.findings)} finding(s), {len(stats.output.splitlines())} line(s) printed by the kernel")
        for finding in stats.findings:
            print(finding)
