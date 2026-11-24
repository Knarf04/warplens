"""Archive the exact tracked source tree used for a rental, including tracked edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def freeze(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout
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
            import io
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
    freeze(args.repo, args.out)


if __name__ == "__main__":
    main()
