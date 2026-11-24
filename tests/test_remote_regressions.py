"""Remote orchestration checks using throwaway checkouts; no installs or GPU access."""
import os
from pathlib import Path

from test_remote_env import archive_of, bash, checkout, members, suite


def ready_suites(root, tmp_path):
    for name in ("blackwell", "blackwell_timing"):
        suite(root, name, f'echo build >> "{tmp_path}/{name}-builds"\n',
              f'echo run >> "{tmp_path}/{name}-runs"\n')


def fake_tool(tmp_path, name, body):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    tool = tools / name
    tool.write_text("#!/usr/bin/env bash\n" + body)
    tool.chmod(0o755)
    return str(tools) + os.pathsep + os.environ["PATH"]


def test_archive_failure_is_not_a_success(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    env["PATH"] = fake_tool(tmp_path, "tar", "echo archive-failed >&2\nexit 75\n")
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "archive:" not in done.stdout
    assert "collect: writing" in done.stdout


def test_rebuilding_a_suite_invalidates_its_successful_run(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    first = bash(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    second = bash(root / "env/remote/run.sh", *args, "--redo", "build-blackwell", env=env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "blackwell-runs").read_text() == "run\nrun\n"
    assert (tmp_path / "blackwell_timing-runs").read_text() == "run\n"


def test_setup_options_are_part_of_the_resume_identity(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    # Exercise setup dispatch without any installer or network access.
    setup = root / "env/remote/setup.sh"
    setup.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/setup-args"\n')
    args = [part for i in range(0, len(args), 2) if args[i + 1] != "setup" for part in args[i:i + 2]]
    for toolkit in ("pinned", "auto"):
        done = bash(root / "env/remote/run.sh", *args, "--toolkit", toolkit, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "setup-args").read_text().splitlines() == ["--toolkit pinned", "--toolkit auto"]
    assert (tmp_path / "blackwell-runs").read_text() == "run\nrun\n"


def test_collect_only_preserves_recorded_gpu_capabilities(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    run_dir = tmp_path / "home/run"
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.status").write_text("ptx-jit PASS\nncu PASS\n")
    first = bash(root / "env/remote/run.sh", *args, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    again = bash(root / "env/remote/run.sh", "--collect-only", env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    manifest = members(archive_of(tmp_path / "home"))["manifest.txt"].decode()
    assert "ptx_jit=ok\n" in manifest
    assert "ncu=ok\n" in manifest


def check_module():
    import importlib.util
    from test_remote_env import REMOTE
    spec = importlib.util.spec_from_file_location("remote_check_env", REMOTE / "check_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_e2e_checks_reject_missing_and_cpu_only_torch(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    check = check_module()
    book = tmp_path / "book/chapter_gemm_basics"
    book.mkdir(parents=True)
    (book / "index.md").touch()
    monkeypatch.setenv("WARPLENS_BOOK", str(book.parent))

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name}")

    monkeypatch.setattr(check.importlib, "import_module", missing)
    assert check.e2e() == 1
    assert "native Triton" in capsys.readouterr().out
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None))
    monkeypatch.setattr(check.importlib, "import_module", lambda name: torch)
    assert check.e2e() == 1
    assert "CPU-only" in capsys.readouterr().out
    torch.version.cuda = "13.0"
    torch.__version__ = "test"
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    assert check.e2e() == 0
    monkeypatch.delenv("WARPLENS_BOOK")
    assert check.e2e() == 1
    assert "WARPLENS_BOOK" in capsys.readouterr().out
    assert check.e2e(str(book.parent)) == 0


def test_remote_dependency_gate_tracks_selected_end_to_end_programs(checkout, tmp_path):
    root, env, args = checkout
    ready_suites(root, tmp_path)
    (root / "env/remote/preflight.sh").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/preflight-args"\n')
    args = [part for i in range(0, len(args), 2) if args[i + 1] != "preflight" for part in args[i:i + 2]]
    for selected, expected in (("", True), ("--no-e2e", False), ("--only tmem/*", False),
                               ("--book /custom/book", True)):
        env["WARPLENS_BLACKWELL_RUN_ARGS"] = selected
        done = bash(root / "env/remote/run.sh", *args, "--redo", "preflight", env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        line = (tmp_path / "preflight-args").read_text().splitlines()[-1]
        assert ("--e2e" in line) == expected
        if selected.startswith("--book"):
            assert "--book /custom/book" in line


def test_archive_checksum_failure_is_not_a_success(checkout, tmp_path):
    import shutil
    root, env, args = checkout
    ready_suites(root, tmp_path)
    real = shutil.which("sha256sum")
    env["PATH"] = fake_tool(tmp_path, "sha256sum",
        f'if [[ "${{1:-}}" == *.tar.gz ]]; then exit 75; fi\nexec "{real}" "$@"\n')
    done = bash(root / "env/remote/run.sh", *args, env=env)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "archive:" not in done.stdout
    assert "archive or checksum failed" in done.stdout


def test_refused_ptx_blocks_full_e2e_but_allows_native_probes(tmp_path):
    from test_remote_env import REMOTE, fake_smi
    runner = tmp_path / "fake-env"
    runner.write_text('''#!/usr/bin/env python3
import pathlib
import sys
args = sys.argv[1:]
if args[0] == "timeout":
    args = args[1:]
    if args[0].startswith("--kill-after="):
        args = args[1:]
    args = args[1:]
if args[0] == "nvcc":
    if "--version" in args:
        print("Cuda compilation tools, release 13.5, V13.5.37")
    else:
        pathlib.Path(args[args.index("-o") + 1]).write_text(".version 9.5\\n")
elif args[0] == "python" and args[1].endswith("check_env.py"):
    print("nvrtc=13.5\\nnvrtc_ptx_version=9.5")
elif args[0] == "python" and args[1].endswith("gpu_probe.py"):
    if args[2] == "info":
        print("compute_cap=10.0\\nname=NVIDIA B200\\ndriver_api=13.0\\ntotal_mem_mib=192000")
    elif args[2] == "vecadd" and args[3].endswith(".ptx"):
        print("error_name=CUDA_ERROR_UNSUPPORTED_PTX_VERSION\\nerror=PTX version refused")
        sys.exit(1)
    else:
        print("elements=1000\\ntmem_address=0\\nok=1")
else:
    sys.exit(1)
''')
    runner.chmod(0o755)
    env = {"WARPLENS_REMOTE_HOME": str(tmp_path / "home"), "WARPLENS_ENV_RUN": str(runner),
           "NVIDIA_SMI": str(fake_smi(tmp_path, name="NVIDIA B200", cap="10.0"))}
    for e2e in (False, True):
        status = tmp_path / "status"
        done = bash(REMOTE / "preflight.sh", *( ["--e2e"] if e2e else []), "--status", str(status), env=env)
        checks = dict(line.split() for line in status.read_text().splitlines())
        assert checks["ptx-jit"] == "WARN"
        assert checks.get("e2e-ptx") == ("FAIL" if e2e else None)
        assert done.returncode == int(e2e), done.stdout + done.stderr
