"""Loaded by a Python program under `warplens run` (the directory is put first on PYTHONPATH).

cuda-python's `cuda.bindings.runtime` has NVIDIA's CUDA runtime linked into it, so the shim
libraries cannot stand in for it, and it cannot start without a driver. The CuTe DSL asks it
one thing before a launch: how many devices there are. This answers that; the launch itself
goes through the DSL's runtime library, which `warplens run` does replace.
"""

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "cuda.bindings.runtime"


def _patch(module) -> None:
    module.cudaGetDeviceCount = lambda: (module.cudaError_t.cudaSuccess, 1)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)  # find the real module, once
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
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
    sys.meta_path.insert(0, _Finder())
_chain()
