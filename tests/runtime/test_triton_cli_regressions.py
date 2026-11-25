"""Python script CLI execution and restoration without a CUDA device."""

import importlib
import sys
from types import SimpleNamespace

import pytest

triton = pytest.importorskip("triton")
from warplens import cli
import warplens.triton_driver as wt


@pytest.fixture(autouse=True)
def deny_hardware(monkeypatch):
    module = importlib.import_module("triton.runtime.driver")
    def forbidden():
        raise AssertionError("hardware driver discovery is forbidden")
    monkeypatch.setattr(module, "_create_driver", forbidden)
    monkeypatch.setattr(triton.runtime.driver, "_active", None)
    monkeypatch.setattr(triton.runtime.driver, "_default", None)
    if hasattr(wt, "_installed"):
        monkeypatch.setattr(wt, "_installed", [])


def run_script(path):
    return cli.run_script(SimpleNamespace(program=str(path), arguments=["--example"],
                                          model="B200", timing=False, quiet=True))


def test_script_imports_sibling_and_restores_host_process(tmp_path):
    (tmp_path / "script_sibling.py").write_text("value = 37\n")
    path = tmp_path / "script.py"
    path.write_text("import script_sibling, sys\nassert sys.argv[1:] == ['--example']\n"
                    "raise SystemExit(script_sibling.value)\n")
    argv, search_path = sys.argv, sys.path[:]
    try:
        assert run_script(path) == 37
        assert sys.argv is argv
        assert sys.path == search_path
        assert triton.runtime.driver._active is None
    finally:
        sys.modules.pop("script_sibling", None)
        sys.argv = argv
        sys.path[:] = search_path


def test_script_exit_message_is_reported(tmp_path, capsys):
    path = tmp_path / "script.py"
    path.write_text("raise SystemExit('the script explains why it failed')\n")
    assert run_script(path) == 1
    assert "the script explains why it failed" in capsys.readouterr().err


def test_script_failure_preserves_original_exception_and_driver(tmp_path):
    outer = wt.WarplensDriver()
    triton.runtime.driver.set_active(outer)
    path = tmp_path / "script.py"
    path.write_text("raise ValueError('script failure')\n")
    argv, search_path = sys.argv, sys.path[:]
    try:
        with pytest.raises(ValueError, match="script failure"):
            run_script(path)
        assert triton.runtime.driver.active is outer
        assert sys.argv is argv
        assert sys.path == search_path
    finally:
        sys.argv = argv
        sys.path[:] = search_path
