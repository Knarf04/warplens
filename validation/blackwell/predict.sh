#!/usr/bin/env bash
# Rewrite predicted/ from the simulator: every probe under `warplens run`, and the
# small-format predictions from warplens.numerics. Review the diff before committing it.
#
#   validation/blackwell/predict.sh [options of run_all.py, e.g. --only 'tma/*' or --book CLONE]
#
# Uses the `warplens` conda env if there is one (for nvcc and Python), else whatever
# nvcc and python3 are on PATH. The end-to-end predictions (predicted/e2e/) need the CUTLASS
# programs (examples/03_cutlass_mma/build.sh --programs), Triton, and for the book a clone.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null && conda env list | grep -q '^warplens '; then
  exec conda run --no-capture-output -n warplens python "$here/run_all.py" --mode predict "$@"
fi
exec python3 "$here/run_all.py" --mode predict "$@"
