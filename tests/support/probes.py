"""The Blackwell probe suite (validation/blackwell) as tests reach it."""

import importlib.util

from tests.support.paths import BLACKWELL


def reconvergence():
    """validation/blackwell/findings/reconvergence.py, imported from its path."""
    spec = importlib.util.spec_from_file_location("reconvergence", BLACKWELL / "findings" / "reconvergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
