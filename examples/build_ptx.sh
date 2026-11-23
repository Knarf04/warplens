#!/usr/bin/env bash
# Regenerate the checked-in .ptx files from the .cu sources.
# Run inside the env: conda run -n warplens examples/build_ptx.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for src in */*.cu; do
  out="${src%.cu}.ptx"
  nvcc -arch=sm_100a -ptx "$src" -o "$out"
  echo "$src -> $out"
done
