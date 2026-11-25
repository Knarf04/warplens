"""What the tests share: where things are in the checkout, builders of hand-written PTX, and
stand-ins for the tools and machines a test cannot have.

    from tests.support import EXAMPLES, HEADER, kernel     # the common ones, re-exported here
    from tests.support.tcgen05 import mma_kernel, place     # a builder for one area

A test module never imports from another test module: what two of them need is here.

| module | what it holds |
| --- | --- |
| `paths` | the checkout and its directories (`REPO`, `EXAMPLES`, `VALIDATION`, ...) |
| `ptx` | `HEADER`, `kernel` (a module around a function body), `run_lanes` |
| `toolchain` | `cuda_include`: the CUDA headers of this machine, if any |
| `tma` | the TMA copy example, box and mbarrier kernels, a one-buffer reuse kernel |
| `tcgen05` | the MMA kernel, CuTe's operand layouts, one MMA run against NumPy |
| `cluster` | the two-CTA MMA kernel |
| `shim` | a fat binary as nvcc embeds one, and a client of the shim protocol |
| `remote` | throwaway checkouts and stand-in tools for env/remote's scripts |
| `probes` | scripts of validation/blackwell, imported from their paths |
"""

from tests.support.paths import (AMPERE, BLACKWELL, BLACKWELL_TIMING, DOCS, ENV, EXAMPLES, PACKAGE, REPO, TESTS,
                                 TOOLS, VALIDATION)
from tests.support.ptx import HEADER, kernel, run_lanes
from tests.support.toolchain import cuda_include

__all__ = ["AMPERE", "BLACKWELL", "BLACKWELL_TIMING", "DOCS", "ENV", "EXAMPLES", "HEADER", "PACKAGE", "REPO",
           "TESTS", "TOOLS", "VALIDATION", "cuda_include", "kernel", "run_lanes"]
