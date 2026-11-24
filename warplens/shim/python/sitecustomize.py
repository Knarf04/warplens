"""Loaded by a Python program under `warplens run` (the directory is put first on PYTHONPATH).

cuda-python's `cuda.bindings.runtime` has NVIDIA's CUDA runtime linked into it, so the shim
libraries cannot stand in for it, and it cannot start without a driver. The CuTe DSL asks it
one thing before a launch: how many devices there are. This answers that; the launch itself
goes through the DSL's runtime library, which `warplens run` does replace.

Triton launches through neither: it asks its active driver. Once the program has imported
Triton, warplens's driver is installed (`warplens.cli.install_triton_driver`), so that a
Triton script needs no other command than a TIRx or CuTe DSL script does.
"""

import importlib.abc
import importlib.util
import json
import os
import sys


def _answer_device_count(module) -> None:
    module.cudaGetDeviceCount = lambda: (module.cudaError_t.cudaSuccess, 1)


def _install_triton_driver(module) -> None:
    package = None
    try:
        # warplens may not be installed for this interpreter: `warplens run` says where it is.
        package = json.loads(os.environ["WARPLENS_RUN_OPTIONS"]).get("package")
        if package:
            sys.path.append(package)
        from warplens.cli import install_triton_driver
        install_triton_driver()
    except Exception as error:  # the program may still be one that only compiles with Triton
        print(f"warplens: could not install the Triton driver, so Triton kernels will not be "
              f"simulated: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        if package in sys.path:
            sys.path.remove(package)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, ready):
        self.wrapped = wrapped
        self.ready = ready

    def __getattr__(self, name):  # what else is asked of a loader, e.g. its resources
        return getattr(self.wrapped, name)

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        self.ready(module)


class _Finder(importlib.abc.MetaPathFinder):
    """Calls `ready(module)` once the module called `target` has been imported."""

    def __init__(self, target, ready):
        self.target = target
        self.ready = ready

    def find_spec(self, name, path=None, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)  # find the real module, once
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self.ready)
        return spec


def _chain() -> None:
    """Run the sitecustomize this one hides, if there is one further along the path."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        candidate = os.path.join(entry or os.getcwd(), "sitecustomize.py")
        if os.path.abspath(entry or os.getcwd()) != here and os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("_warplens_hidden_sitecustomize", candidate)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


if os.environ.get("WARPLENS_SHIM_FD"):
    sys.meta_path.insert(0, _Finder("cuda.bindings.runtime", _answer_device_count))
    if os.environ.get("WARPLENS_RUN_OPTIONS"):
        sys.meta_path.insert(0, _Finder("triton", _install_triton_driver))
_chain()
