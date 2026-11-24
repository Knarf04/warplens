#!/usr/bin/env python3
"""Report what the installed env contains, for env/remote/setup.sh and preflight.sh.

    check_env.py versions        key=value for every package the launch paths need
    check_env.py nvrtc ARCH      compile a kernel with NVRTC for ARCH (e.g. sm_100a)

Needs no GPU. Exit status 1 when a package is missing or NVRTC refuses ARCH.
"""

import importlib
import importlib.metadata
import sys

# distribution name -> module that must import
PACKAGES = {
    "numpy": "numpy",
    "PyYAML": "yaml",
    "pytest": "pytest",
    "triton": "triton",
    "nvidia-cutlass-dsl": "cutlass",
    "apache-tvm-tirx": "tvm_tirx",
    "cuda-bindings": "cuda.bindings",
    "warplens": "warplens",
}
# reported, not imported
EXTRA = ["apache-tvm-ffi", "apache-tvm-ffi-orcjit", "cuda-pathfinder",
         "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core",
         "nvidia-cutlass-dsl-libs-cu12", "ml-dtypes", "typing-extensions", "setuptools"]

SOURCE = b"""
extern "C" __global__ void k(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= 2.0f;
}
"""


def key(name):
    return name.lower().replace("-", "_")


def versions():
    status = 0
    print(f"python={sys.version.split()[0]}")
    for dist, module in PACKAGES.items():
        try:
            version = importlib.metadata.version(dist)
            importlib.import_module(module)
            print(f"{key(dist)}={version}")
        except Exception as e:  # a missing package or one that fails to import
            print(f"{key(dist)}=MISSING ({type(e).__name__}: {e})")
            status = 1
    for dist in EXTRA:
        try:
            print(f"{key(dist)}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{key(dist)}=MISSING")
    return status


def nvrtc(arch):
    from cuda.bindings import nvrtc as rt

    def ok(result):
        err, *rest = result
        if int(err) != 0:
            raise RuntimeError(f"NVRTC error {err!r}")
        return rest[0] if len(rest) == 1 else rest

    major, minor = ok(rt.nvrtcVersion())
    print(f"nvrtc={major}.{minor}")
    program = ok(rt.nvrtcCreateProgram(SOURCE, b"check.cu", 0, [], []))
    options = [f"-arch={arch}".encode()]
    err = rt.nvrtcCompileProgram(program, len(options), options)[0]
    if int(err) != 0:
        size = ok(rt.nvrtcGetProgramLogSize(program))
        log = b" " * size
        rt.nvrtcGetProgramLog(program, log)
        print(f"error=NVRTC refused -arch={arch}: {log.decode(errors='replace').strip()}")
        return 1
    size = ok(rt.nvrtcGetPTXSize(program))
    ptx = b" " * size
    ok(rt.nvrtcGetPTX(program, ptx))
    text = ptx.decode(errors="replace")
    for line in text.splitlines():
        if line.startswith(".version"):
            print(f"nvrtc_ptx_version={line.split()[1]}")
        if line.startswith(".target"):
            print(f"nvrtc_target={line.split()[1].rstrip(',')}")
    print("ok=1")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "versions":
        return versions()
    if len(argv) == 3 and argv[1] == "nvrtc":
        try:
            return nvrtc(argv[2])
        except Exception as e:
            print(f"error={type(e).__name__}: {e}")
            return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
