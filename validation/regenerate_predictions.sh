#!/usr/bin/env bash
# Regenerate every committed prediction from the simulator, in one command.
#
#   validation/regenerate_predictions.sh [--book CLONE] [--no-e2e]
#
# Run this after any change to the simulator's behaviour, to the timing model, to a probe
# or benchmark, or to the compiler in the env. It rewrites, in order:
#
#   1. validation/blackwell/predicted/         every probe under `warplens run`, the
#      small-format predictions from warplens.numerics, provenance.json (the nvcc), and
#      predicted/e2e/ (the CUTLASS tutorials, Triton, the TIRx and CuTe DSL scripts, and
#      the book's kernels when a clone is given with --book or WARPLENS_BOOK)
#   2. validation/blackwell/README.md          the tables between its two markers, from
#      the manifest and the new predictions
#   3. validation/blackwell/tmem/repeat_counts_ptxas.txt   ptxas and the simulator on every
#      tcgen05.ld/st repeat count
#   4. validation/blackwell_timing/ncu_banks.ptx, then validation/blackwell_timing/predicted/
#      every benchmark under `warplens run --timing`, meta.json (config and nvcc),
#      predictions.md
#
# Nothing is committed: read `git diff` (the end-to-end predictions carry host addresses
# and times, which change on every run and mean nothing), then run the tests:
#
#   PYTHONPATH=. python -m pytest -q tests/validation/probes/test_blackwell_probes.py tests/validation/probes/test_predict_path.py \
#       tests/timing/test_timing_calibration.py tests/validation/probes/test_rental_harness.py
#   python tools/check_validation_matrix.py
#
# Uses the `warplens` conda env if there is one (for nvcc and Python), else nvcc and python3
# on PATH. Needs nvcc with sm_100a; no GPU. Takes a few minutes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
book="${WARPLENS_BOOK:-}"
e2e=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --book) book="${2:?--book needs the path of a clone}"; shift ;;
    --no-e2e) e2e=0 ;;
    -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

in_env=()
if command -v conda >/dev/null && conda env list | grep -q '^warplens '; then
  in_env=(conda run --no-capture-output -n warplens)
fi
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
say() { echo "regenerate: $*"; }

say "1/4 probe predictions (validation/blackwell/predicted)"
args=()
if (( e2e )); then
  if [[ ! -x examples/03_cutlass_mma/bin/tutorial_04 ]]; then
    say "building the CUTLASS tutorials first (needs network once)"
    "${in_env[@]}" examples/03_cutlass_mma/build.sh --programs
  fi
  if [[ -n "$book" ]]; then
    args+=(--book "$book")
  else
    say "no clone of the book (--book or WARPLENS_BOOK): predicted/e2e/book_hgemm.txt is left as it is"
  fi
else
  args+=(--no-e2e)
fi
"${in_env[@]}" python validation/blackwell/run_all.py --mode predict "${args[@]}"

say "2/4 the tables of validation/blackwell/README.md"
"${in_env[@]}" python validation/blackwell/manifest.py --update-readme

say "3/4 tcgen05 repeat counts (ptxas and the simulator)"
"${in_env[@]}" python validation/blackwell/tmem/repeat_counts_ptxas.py --sim \
  > validation/blackwell/tmem/repeat_counts_ptxas.txt

say "4/4 timing predictions (validation/blackwell_timing/predicted)"
"${in_env[@]}" env NVCC=nvcc validation/blackwell_timing/build.sh sim
"${in_env[@]}" nvcc -arch=sm_100a -ptx -DBENCH_CLOCK=0 validation/blackwell_timing/ncu_banks.cu \
  -o validation/blackwell_timing/ncu_banks.ptx
"${in_env[@]}" python validation/blackwell_timing/predict.py

say "done. Review: git status --short validation/ ; git diff --stat"
