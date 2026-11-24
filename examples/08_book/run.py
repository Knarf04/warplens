"""Run the GEMM kernels of the book *Modern GPU Programming for MLSys*, from your clone of it.

    git clone https://github.com/mlc-ai/modern-gpu-programming-for-mlsys book
    warplens run python examples/08_book/run.py book           # hgemm_v1 ... hgemm_v9, as a B200
    warplens run python examples/08_book/run.py book v3 v7     # only these
    python examples/08_book/run.py book                        # on a Blackwell GPU

The book's repository has no license, so none of its code is here: the kernel definitions are
read from the chapters of the clone and executed as the chapters give them. Two things differ
from the book's own cells. The arrays are TVM's instead of PyTorch's, because PyTorch cannot
run in the simulator. And the target asks for PTX, which the simulator needs and a GPU accepts.

Needs: pip install apache-tvm-tirx cuda-bindings
"""

import linecache
import re
import sys
import time
from pathlib import Path

import numpy as np
import tvm_tirx

CHAPTERS = ["chapter_gemm_basics", "chapter_gemm_async", "chapter_gemm_advanced"]
# The smallest problem each kernel does any work on: v6 on wants K to fill its pipeline, and the
# two-CTA kernels compute whole output tiles of 256 x 256 (v8) and 512 x 256 (v9), nothing else.
SIZES = {"v6": (128, 128, 256), "v7": (128, 128, 256), "v8": (256, 256, 256), "v9": (512, 256, 256)}
DEFINES = re.compile(r"^def (hgemm_v\d+)\(", flags=re.M)
IMPORTS = re.compile(r"(?:(?:import|from) .*\n|\s*\n)+")


def kernels(book: Path) -> dict:
    """The `hgemm_vN` builders, by version. Of a chapter's code cells, only those that define
    one and those that hold nothing but imports are executed; the rest are fragments of the
    explanation, or run a kernel on the GPU with PyTorch."""
    found = {}
    for chapter in CHAPTERS:
        text = (book / chapter / "index.md").read_text()
        scope: dict = {}
        for cell in re.findall(r"^```[^\n]*\n(.*?)^```", text, flags=re.S | re.M):
            names = DEFINES.findall(cell)
            if names or IMPORTS.fullmatch(cell):
                # TIRx reads a function's source text, so the cell has to be findable as a file.
                origin = f"<{chapter} cell {len(linecache.cache)}>"
                linecache.cache[origin] = (len(cell), None, cell.splitlines(keepends=True), origin)
                exec(compile(cell, origin, "exec"), scope)  # the reader's own clone
                found.update({name.removeprefix("hgemm_"): scope[name] for name in names})
    return found


def run(name: str, build, device, M: int, N: int, K: int) -> None:
    target = tvm_tirx.target.Target({"kind": "cuda", "backend_config": {"cuda": {"target_format": "ptx"}}})
    executable = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": build(M, N, K)}), target=target)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    A, B = tvm_tirx.runtime.tensor(a, device=device), tvm_tirx.runtime.tensor(b, device=device)
    D = tvm_tirx.runtime.tensor(np.zeros((M, N), dtype=np.float16), device=device)
    start = time.perf_counter()
    executable.mod(A, B, D)
    got = D.numpy()
    reference = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float16)
    np.testing.assert_allclose(got, reference, rtol=2e-2, atol=1e-2)  # the book's tolerances
    error = float(np.abs(got.astype(np.float32) - reference.astype(np.float32)).max())
    print(f"hgemm_{name} {M}x{N}x{K}: matches NumPy (max error {error:.4f}, {time.perf_counter() - start:.1f} s)",
          flush=True)


def main(argv: list[str]) -> int:
    if not argv or not (Path(argv[0]) / CHAPTERS[0] / "index.md").is_file():
        sys.exit(__doc__)
    found = kernels(Path(argv[0]))
    wanted = argv[1:] or sorted(found, key=lambda v: int(v[1:]))
    device = tvm_tirx.cuda(0)
    failed = 0
    for name in wanted:
        if name not in found:
            sys.exit(f"no hgemm_{name} in the book; it has {', '.join(sorted(found))}")
        try:
            run(name, found[name], device, *SIZES.get(name, (128, 128, 64)))
        except Exception as error:  # report every kernel, then fail
            failed += 1
            first = str(error).strip().splitlines()
            print(f"hgemm_{name}: FAILED: {type(error).__name__}: {first[0] if first else ''}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
