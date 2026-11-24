"""env/remote/setup.sh on machines the development machine is not: a bare image without a
download tool, a root shell, a disk that is too small, an Arm host, a container.

setup.sh runs here up to the point where it would download micromamba, in a PATH that holds
only the tools named below, so that "no curl" and "no sudo" are true whatever the machine
running the tests has. Root is `fakeroot` (the process believes it is uid 0; nothing is
privileged) and `apt-get` is a stand-in that records how it was called. That covers the
decisions the script makes as root. It does not cover a real `apt-get install`, a real root
shell or a real container: none of them can be had on the development machine without
privileges (docker needs its group, podman is not installed, and Ubuntu 24.04 denies
unprivileged user namespaces, so `unshare -r` and `bwrap` fail).
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "env" / "remote"
# What setup.sh and lib.sh call before the download, besides shell builtins.
TOOLS = ["bash", "sh", "env", "dirname", "basename", "sed", "awk", "grep", "cut", "tr", "sort", "head", "tail",
         "cat", "mkdir", "rm", "mv", "cp", "chmod", "sha256sum", "date", "id", "nproc", "mktemp", "tee", "hostname",
         "stat", "printf", "true", "false", "wc", "touch", "ls"]


def script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def bare(tmp_path):
    """PATH of a bare image: coreutils, a `df` that reports 100 GB free, an x86_64 `uname`;
    no curl, wget, python3, git, sudo, apt-get or nvidia-smi."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in TOOLS:
        found = shutil.which(name)
        assert found, name
        (tools / name).symlink_to(found)
    script(tools / "df", 'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
                         'echo "/dev/fake 209715200 104857600 ${FAKE_DF_KB:-104857600} 50% /"\n')
    script(tools / "uname", 'case "$1" in -m) echo "${FAKE_MACHINE:-x86_64}" ;; -s) echo Linux ;; '
                            '-r) echo 6.8.0-test ;; *) echo Linux ;; esac\n')
    return tmp_path


def setup(bare: Path, *args: str, root: bool = False, **env: str):
    command = ["bash", str(REMOTE / "setup.sh"), *args]
    full = {"PATH": str(bare / "tools"), "HOME": str(bare / "home"), "WARPLENS_REMOTE_HOME": str(bare / "remote"),
            "APT_LOG": str(bare / "apt.log"), **env}
    if root:
        fakeroot = shutil.which("fakeroot")
        if not fakeroot:
            pytest.skip("fakeroot is not installed")
        # fakeroot itself needs the real PATH to start; the script gets the bare one, and of
        # the environment only what is set here plus what fakeroot works through (its
        # preloaded library and the key of its daemon). Whatever else the machine running
        # the tests exports stays out: a GitHub runner has DEBIAN_FRONTEND=noninteractive.
        command = [fakeroot, "sh", "-c", 'exec env -i LD_PRELOAD="$LD_PRELOAD" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" '
                   'FAKEROOTKEY="$FAKEROOTKEY" "$@"', "sh", *(f"{k}={v}" for k, v in full.items()), *command]
        full = {**os.environ, **full, "PATH": os.environ["PATH"]}
    return subprocess.run(command, capture_output=True, text=True, env=full, timeout=120)


def fake_apt(bare: Path, body: str) -> None:
    script(bare / "tools" / "apt-get", 'echo "uid=$(id -u) frontend=${DEBIAN_FRONTEND:-unset} $*" >> "$APT_LOG"\n' + body)


FAKE_CURL = ('#!/bin/sh\n'
             'out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
             'echo "not micromamba" > "$out"\n')


def test_a_bare_image_without_root_says_what_is_missing(bare):
    done = setup(bare)
    assert done.returncode == 1
    assert "container: " in done.stdout and "(uid " in done.stdout
    assert "nvidia-smi is missing or failed" in done.stderr
    assert "no curl, wget or python3 to download with, and cannot install one (not root, or no apt-get)" in done.stderr
    assert not (bare / "remote" / "state" / "setup.done").exists()
    assert not (bare / "apt.log").exists()


def test_root_installs_a_download_tool_and_the_checksum_is_verified(bare):
    """As root without curl: apt-get is used, non-interactively; what was downloaded is not
    run unless its checksum is the pinned one."""
    fake_apt(bare, f'case "$1" in install) cat > "$(dirname "$0")/curl" <<\'EOF\'\n{FAKE_CURL}EOF\n'
                   'chmod +x "$(dirname "$0")/curl" ;; esac\n')
    done = setup(bare, root=True)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "(uid 0)" in done.stdout
    assert "installing system package(s): curl ca-certificates" in done.stdout
    calls = (bare / "apt.log").read_text().splitlines()
    assert calls[0].endswith(" update -qq") and calls[0].startswith("uid=0 ")
    assert calls[1] == "uid=0 frontend=noninteractive install -y -qq --no-install-recommends curl ca-certificates"
    assert "downloading micromamba" in done.stdout
    assert "micromamba checksum mismatch: got " in done.stderr and "pins.env has " in done.stderr
    assert not (bare / "remote" / "bin" / "micromamba").exists()      # the bad download was not installed


def test_root_whose_apt_fails_gets_the_same_plain_message(bare):
    fake_apt(bare, "exit 100\n")
    done = setup(bare, root=True)
    assert done.returncode == 1
    assert "no curl, wget or python3 to download with, and cannot install one" in done.stderr
    assert (bare / "apt.log").read_text().splitlines() == ["uid=0 frontend=unset update -qq"]


def test_a_download_that_fails_names_the_url_and_the_likely_causes(bare):
    script(bare / "tools" / "curl", "exit 6\n")
    done = setup(bare)
    assert done.returncode == 1
    assert "download failed: https://github.com/mamba-org/micromamba-releases/releases/download/" in done.stderr
    assert "is there network access? are CA certificates installed?" in done.stderr


def test_the_disk_must_hold_the_envs(bare):
    small = setup(bare, FAKE_DF_KB=str(15 * 1024 * 1024))
    assert small.returncode == 1 and "has 15 GB free" in small.stderr and "need about 18 GB" in small.stderr
    # Without the PyTorch env 15 GB is enough, and the script goes on to the next obstacle.
    without = setup(bare, "--no-torch", FAKE_DF_KB=str(15 * 1024 * 1024))
    assert "GB free" not in without.stderr and "no curl, wget or python3" in without.stderr
    tiny = setup(bare, "--no-torch", FAKE_DF_KB=str(5 * 1024 * 1024))
    assert "has 5 GB free" in tiny.stderr and "need about 12 GB" in tiny.stderr


def test_an_arm_host_is_refused_before_anything_is_downloaded(bare):
    done = setup(bare, FAKE_MACHINE="aarch64")
    assert done.returncode == 1
    assert "only x86_64 is supported and tested, not aarch64" in done.stderr
    assert "downloading" not in done.stdout


def test_a_container_is_recognized(bare):
    """Kubernetes by its environment; docker and podman by a file in /, which a test cannot
    create."""
    done = setup(bare, KUBERNETES_SERVICE_HOST="10.0.0.1")
    assert "container: kubernetes" in done.stdout
    text = (REMOTE / "lib.sh").read_text()
    assert "/.dockerenv" in text and "/run/.containerenv" in text


def test_bad_arguments(bare):
    assert setup(bare, "--toolkit").returncode != 0
    done = setup(bare, "--with-everything")
    assert done.returncode == 2 and "unknown argument" in done.stderr
    help_text = setup(bare, "--help")
    assert help_text.returncode == 0 and "--no-torch" in help_text.stdout and "--toolkit auto" in help_text.stdout
