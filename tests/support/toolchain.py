"""What of the CUDA toolkit this machine has."""

import shutil
from pathlib import Path


def cuda_include(header: str = "cuda.h") -> Path | None:
    """The CUDA toolkit include directory that has `header`, or None. Looks next to the nvcc on
    PATH first (a conda env keeps its headers under targets/), then in /usr/local/cuda."""
    candidates = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit = Path(nvcc).resolve().parent.parent
        candidates += [toolkit / "targets/x86_64-linux/include", toolkit / "include"]
    candidates.append(Path("/usr/local/cuda/include"))
    return next((p for p in candidates if (p / header).exists()), None)
