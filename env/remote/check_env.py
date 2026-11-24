#!/usr/bin/env python3
"""Report what the installed env contains, for env/remote/setup.sh and preflight.sh.

    check_env.py versions        key=value for every package the launch paths need
    check_env.py nvrtc ARCH      compile a kernel with NVRTC for ARCH (e.g. sm_100a)
    check_env.py torch           (run by the PyTorch env's interpreter) torch, the CUDA
                                 release it was built for, the Triton it imports, whether
                                 it sees a GPU; and torch_pip_<name> for what else is there

Needs no GPU. Exit status 1 when a package is missing or NVRTC refuses ARCH.
"""

import importlib
import importlib.metadata
import re
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
    # Everything else pip sees, as pip_<name>, to compare with the PIP_ pins.
    main = {re.sub(r"[-_.]+", "-", name).lower() for name in PACKAGES} | {"pip"}
    others = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
        if name and name not in main:
            others.setdefault(name, dist.version)   # the first on sys.path is the one imported
    for name in sorted(others):
        print(f"pip_{key(name)}={others[name]}")
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


def torch_report():
    try:
        import torch
        import triton
    except Exception as e:
        print(f"torch=MISSING ({type(e).__name__}: {e})")
        return 1
    print(f"torch={importlib.metadata.version('torch')}")
    print(f"torch_build={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_triton={triton.__version__}")
    try:
        available = torch.cuda.is_available()
        print(f"torch_cuda_available={int(available)}")
        if available:
            major, minor = torch.cuda.get_device_capability(0)
            print(f"torch_device={torch.cuda.get_device_name(0)}")
            print(f"torch_device_cap={major}.{minor}")
            # Was this build compiled for the device? 'sm_100' for a B200.
            print(f"torch_arch_list={','.join(torch.cuda.get_arch_list())}")
    except Exception as e:  # a driver the bundled runtime cannot use
        print(f"torch_cuda_available=0 ({type(e).__name__}: {e})")
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = re.sub(r"[-_.]+", "_", dist.metadata["Name"] or "").lower()
        if name and name != "torch":
            print(f"torch_pip_{name}={dist.version}")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "versions":
        return versions()
    if len(argv) == 2 and argv[1] == "torch":
        return torch_report()
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
