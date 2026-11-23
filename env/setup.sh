#!/usr/bin/env bash
# Create or update the `warplens` conda env from env/environment.yml,
# then write an exact lock file and run the toolchain check.
#
# Usage: env/setup.sh            # create or update
#        env/setup.sh --locked   # recreate exactly from the lock file
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="warplens"
lock="$here/conda-linux-64.lock"

# Plain conda on purpose: mamba 2.9.0 segfaulted on `env create` here.
conda_exe=conda
command -v conda >/dev/null 2>&1 || { echo "error: conda not found on PATH" >&2; exit 1; }

if [[ "${1:-}" == "--locked" ]]; then
  [[ -f "$lock" ]] || { echo "error: $lock not found" >&2; exit 1; }
  "$conda_exe" create -y -n "$name" --file "$lock"
else
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    "$conda_exe" env update -n "$name" -f "$here/environment.yml" --prune
  else
    "$conda_exe" env create -n "$name" -f "$here/environment.yml"
  fi
  conda list -n "$name" --explicit --sha256 > "$lock"
  echo "wrote $lock"
fi

conda run -n "$name" --no-capture-output "$here/check.sh"
