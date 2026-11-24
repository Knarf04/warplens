"""The Triton kernels of examples/05_triton, on a real GPU.

    python validation/blackwell/e2e/triton_gpu.py

examples/05_triton/run.py always runs in the simulator: it installs the warplens driver and
uses warplens tensors. This runs the SAME two kernels (imported from that file, not copied)
with the same inputs through Triton's own CUDA driver and PyTorch tensors, and prints the
same first half of each line, so the two outputs can be read side by side:

    vector add: max abs error vs NumPy 0
    matmul: max abs error vs NumPy 4.77e-06

It also prints a few result elements as raw bits. Needs PyTorch with CUDA and Triton; says
so ("SKIPPED: ...") and exits with status 2 if they are missing. The versions are pinned in
env/remote/pins.env (TORCH, TRITON); env/remote/setup.sh installs PyTorch into an env of its
own and env/remote/run.sh passes its interpreter as WARPLENS_TORCH_PYTHON.

What has run: the whole script, on the development machine's RTX 3080 Ti (sm_86) with
torch 2.15.0+cu132 and triton 3.9.0, 2026-11-23: vector add error 0, matmul error 5.72e-06.
On that GPU Triton compiles the matmul to `mma.sync`. What only a B200 can show: Triton's
sm_100a code for the same kernel, which is the `tcgen05` path the simulator executes, so
the matmul's error and result bits there are the ones to hold against the prediction.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "05_triton" / "run.py"


def capturing():
    return bool(os.environ.get("WARPLENS_CAPTURE_DIR"))


def main() -> int:
    try:
        import torch
        import triton
    except ImportError as error:
        print(f"SKIPPED: {error}")
        return 2
    if not torch.cuda.is_available():
        print("SKIPPED: PyTorch sees no CUDA device")
        return 2
    spec = importlib.util.spec_from_file_location("triton_example", EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)  # defines the kernels; installs nothing
    print(f"info: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, "
          f"torch {torch.__version__}, triton {triton.__version__}")
    capability = "%d%d" % torch.cuda.get_device_capability(0)
    if f"sm_{capability}" not in torch.cuda.get_arch_list():
        print(f"info: this PyTorch build lists {torch.cuda.get_arch_list()}, not sm_{capability}")

    n, block = 1000, 128
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    initial = np.zeros(n, dtype=np.float32)
    out = torch.from_numpy(initial).cuda() if capturing() else torch.empty(n, dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(n, block),)
    compiled = example.add_kernel[grid](torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(), out, n,
                                                 BLOCK=block)
    torch.cuda.synchronize()  # a kernel that faulted raises here, not at some later copy
    got = out.cpu().numpy()
    if capturing():
        from validation.blackwell.frontend_capture import triton_case
        triton_case("triton_vector_add", compiled, values={"n": n},
                    pointers={"x_ptr": "a", "y_ptr": "b", "out_ptr": "out"}, grid=grid,
                    inputs={"a": a, "b": b, "out": initial}, outputs={"out": got},
                    comparison={"kind": "exact"})
    np.testing.assert_array_equal(got, a + b)
    print(f"vector add: max abs error vs NumPy {np.abs(got - (a + b)).max():.3g}")

    M, N, K, blocks = 256, 128, 96, (128, 64, 32)
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    b = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    initial = np.zeros((M, N), dtype=np.float32)
    c = torch.from_numpy(initial).cuda() if capturing() else torch.empty((M, N), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(M, blocks[0]), triton.cdiv(N, blocks[1]))
    compiled = example.matmul_kernel[grid](torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(), c, M, N, K, K, 1, N, 1, N, 1,
                                BLOCK_M=blocks[0], BLOCK_N=blocks[1], BLOCK_K=blocks[2])
    torch.cuda.synchronize()
    got = c.cpu().numpy()
    reference = a.astype(np.float32) @ b.astype(np.float32)
    tolerance = float(1e-5 * np.abs(reference).max())
    if capturing():
        from validation.blackwell.frontend_capture import triton_case
        triton_case("triton_matmul", compiled,
                    values={"M": M, "N": N, "K": K, "stride_am": K, "stride_ak": 1,
                            "stride_bk": N, "stride_bn": 1, "stride_cm": N, "stride_cn": 1},
                    pointers={"a_ptr": "a", "b_ptr": "b", "c_ptr": "c"}, grid=grid,
                    inputs={"a": a, "b": b, "c": initial}, outputs={"c": got},
                    comparison={"kind": "allclose", "rtol": 0, "atol": tolerance})
    np.testing.assert_allclose(got, reference, rtol=0, atol=tolerance)
    print(f"matmul: max abs error vs NumPy {np.abs(got - reference).max():.3g}")
    bits = got.view(np.uint32)
    print("info: matmul result bits [0][0..3], [255][124..127]: "
          + " ".join(f"{x:08x}" for x in list(bits[0, :4]) + list(bits[255, 124:])))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # a compile or launch failure: one line that compare.py shows, then the traceback
        print(f"FAILED: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}", flush=True)
        raise
