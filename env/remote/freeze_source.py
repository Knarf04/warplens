"""Archive the exact tracked source tree used for a rental, including tracked edits.

git is asked which files are tracked. A checkout that belongs to another user than the one
running this (uploaded as one user and packed as root, or a mounted directory in a
container) makes git refuse with "dubious ownership" unless the directory is named safe.
It is named both ways there are, as repo_git in lib.sh does: on the command line, which git
2.38 and later accept, and in a global config file of its own that includes the user's,
which is the only way the patched older releases accept (Ubuntu 22.04's 2.34.1). Nothing
is written to the checkout or to the user's config. Tried under fakeroot with a checkout
of another owner and git 2.43 (tests/validation/remote/test_remote_hardening.py); not with an older git.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def git_config_quote(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def freeze(repo, out):
    given, repo, out = Path(repo).absolute(), Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        own = os.environ.get("GIT_CONFIG_GLOBAL") or str(Path.home() / ".gitconfig")
        config = Path(scratch) / "gitconfig"
        safe = sorted({str(given), str(repo)})
        config.write_text("[safe]\n" + "".join(f"\tdirectory = {git_config_quote(path)}\n" for path in safe)
                          + (f"[include]\n\tpath = {git_config_quote(own)}\n" if Path(own).is_file() else ""))
        environment = {**os.environ, "GIT_CONFIG_GLOBAL": str(config), "GIT_TERMINAL_PROMPT": "0"}
        options = [option for path in safe for option in ("-c", f"safe.directory={path}")]

        def git(*args):
            done = subprocess.run(["git", *options, "-C", str(repo), *args], capture_output=True, env=environment)
            if done.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)} failed in {repo} (exit status {done.returncode}): "
                                   f"{done.stderr.decode(errors='replace').strip()}")
            return done.stdout
        return _freeze(repo, out, git)


def _freeze(repo, out, git):
    names = git("ls-files", "-z").decode().split("\0")
    tracked = [name for name in names if name and ((repo / name).exists() or (repo / name).is_symlink())]
    status = git("status", "--porcelain=v1", "--untracked-files=no").decode()
    # Copy actual tracked bytes, not `git archive HEAD`: rental edits must be retained.
    manifest = {"schema": "warplens.source.v1", "commit": git("rev-parse", "HEAD").decode().strip(),
                "tracked_status": status, "files": {}}
    archive = out / "source.tar.gz"
    temporary = out / "source.tar.gz.part"
    with tarfile.open(temporary, "w:gz") as tar:
        for name in tracked:
            path = repo / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source snapshot requires regular tracked files: {name}")
            raw = path.read_bytes()
            manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            # Supply the same bytes we hashed even if another process edits the checkout.
            info = tar.gettarinfo(str(path), arcname=f"warplens/{name}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    temporary.replace(archive)
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "tracked.diff").write_bytes(git("diff", "HEAD", "--binary"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        freeze(args.repo, args.out)
    except (OSError, RuntimeError, ValueError) as error:   # one line that says what, for run.sh's log
        print(f"source snapshot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
