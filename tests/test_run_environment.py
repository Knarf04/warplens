"""What `warplens run` arranges around the program: the settings of the compilers that run
inside it, and the start-up hook of a Python program."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from warplens import cli
from warplens.host import runtime

HOOK = Path(cli.__file__).resolve().parent / "shim" / "python"


@pytest.fixture
def no_cute_dsl(monkeypatch):
    monkeypatch.setattr(runtime, "cute_dsl_runtime", lambda shims: None)


def test_frontend_settings_default_to_the_simulated_model(no_cute_dsl, tmp_path):
    env = {}
    kept = cli.frontend_environment(env, "B300", tmp_path, tmp_path / "kept")
    assert kept == tmp_path / "kept"
    assert env["CUTE_DSL_ARCH"] == "sm_103a" and env["CUTE_DSL_KEEP"] == "ptx"
    assert env["CUTE_DSL_DUMP_DIR"] == str(kept) and "CUTE_DSL_LIBS" not in env
    assert env["PYTHONPATH"] == str(HOOK)


def test_frontend_settings_made_by_the_user_are_kept(no_cute_dsl, tmp_path):
    env = {"CUTE_DSL_ARCH": "sm_90a", "CUTE_DSL_KEEP": "ir, sass", "CUTE_DSL_DUMP_DIR": "/somewhere",
           "CUTE_DSL_LIBS": "/their/libcute_dsl_runtime.so", "PYTHONPATH": "/their/modules"}
    kept = cli.frontend_environment(env, "B200", tmp_path, tmp_path / "kept")
    assert kept == Path("/somewhere")  # where the PTX will be looked for
    assert env["CUTE_DSL_ARCH"] == "sm_90a" and env["CUTE_DSL_LIBS"] == "/their/libcute_dsl_runtime.so"
    assert env["CUTE_DSL_KEEP"] == "ir,sass,ptx"
    assert env["PYTHONPATH"] == f"{HOOK}{os.pathsep}/their/modules"


@pytest.mark.parametrize("keep", ["PTX", "all", "ir,ptx"])
def test_a_keep_setting_that_already_covers_ptx_is_left_alone(no_cute_dsl, tmp_path, keep):
    env = {"CUTE_DSL_KEEP": keep}
    cli.frontend_environment(env, "B200", tmp_path, tmp_path)
    assert env["CUTE_DSL_KEEP"] == keep


def child(code, tmp_path, **env):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": f"{HOOK}{os.pathsep}{tmp_path}", **env})


def test_start_up_hook_runs_the_sitecustomize_it_hides(tmp_path):
    (tmp_path / "sitecustomize.py").write_text("print('the hidden one ran')\n")
    done = child("print('program')", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[:2] == ["the hidden one ran", "program"]


def test_start_up_hook_answers_the_device_count_only_under_warplens_run(tmp_path):
    pytest.importorskip("cuda.bindings.runtime")
    code = ("import sys, cuda.bindings.runtime as r\n"
            "print(type(r.cudaGetDeviceCount).__name__, 'cuda.bindings.runtime' in sys.modules)")
    hooked = child(code + "\nprint(r.cudaGetDeviceCount())", tmp_path, WARPLENS_SHIM_FD="99")
    assert hooked.returncode == 0, hooked.stderr
    assert hooked.stdout.splitlines() == ["function True", "(<cudaError_t.cudaSuccess: 0>, 1)"]
    plain = child(code, tmp_path, WARPLENS_SHIM_FD="")
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout.split()[0] != "function"
