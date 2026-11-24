#!/usr/bin/env bash
# The entry point env/remote/run.sh calls: every probe on the GPU, through run_all.py.
# Extra arguments go to run_all.py (see its --help).
#
#   WARPLENS_RESULTS_DIR   where the results go (default: run_all.py's own results/ directory)
#   WARPLENS_PTX_JIT       "refused" when the driver does not accept this toolkit's PTX: the
#                          programs built with machine code are run instead (build.sh --native)
#   PYTHON                 the interpreter (default: python)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=(--mode gpu)
[[ -n "${WARPLENS_RESULTS_DIR:-}" ]] && args+=(--out "$WARPLENS_RESULTS_DIR")
[[ "${WARPLENS_PTX_JIT:-}" == refused ]] && args+=(--binaries native)
exec "${PYTHON:-python}" "$here/run_all.py" "${args[@]}" "$@"
