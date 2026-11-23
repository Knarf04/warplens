#!/usr/bin/env bash
# Verify the active env can compile CUDA C++ to PTX for the Blackwell targets.
# Run inside the env: conda run -n warplens env/check.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "nvcc:   $(command -v nvcc)"
nvcc --version | sed -n 's/^Cuda compilation tools, /        /p'
python -c 'import sys, numpy; print(f"python: {sys.version.split()[0]}, numpy {numpy.__version__}")'

status=0
for arch in sm_100a sm_103a; do
  out="$tmp/vecadd.$arch.ptx"
  if nvcc -arch="$arch" -ptx "$here/check_vecadd.cu" -o "$out" 2>"$tmp/err"; then
    echo "ok:     $arch -> PTX $(sed -n 's/^\.version //p' "$out"), target $(sed -n 's/^\.target //p' "$out")"
  else
    echo "FAIL:   $arch"
    sed 's/^/        /' "$tmp/err"
    status=1
  fi
done
exit $status
