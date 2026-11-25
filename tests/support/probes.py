"""The Blackwell probe suite (validation/blackwell) as tests reach it."""

import importlib.util
import sys
from pathlib import Path

from tests.support.paths import BLACKWELL

E2E = BLACKWELL / "e2e"


def reconvergence():
    """validation/blackwell/findings/reconvergence.py, imported from its path."""
    spec = importlib.util.spec_from_file_location("reconvergence", BLACKWELL / "findings" / "reconvergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_e2e(name: str, path: Path):
    """A script of the suite or of its e2e directory, imported from its path under `name`, with
    both directories importable while it loads."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(BLACKWELL), str(E2E)]
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[:2]
    return module
