#!/usr/bin/env bash
# Make the predictions again with the nvcc that is installed, for a run on a fallback
# toolkit. Needs no GPU: the simulator runs the PTX of the installed compiler.
#
#   env/remote/predict_toolkit.sh --out DIR [--build] [--jobs N] [--no-timing]
#
# The committed predictions are the simulator's run of PTX from one nvcc
# (validation/blackwell/predicted/provenance.json). On a machine whose driver is older,
# setup.sh installs the toolkit that matches the driver, and the probes are then another
# compilation. This script writes, without touching a committed file:
#
#   DIR/blackwell/            what validation/blackwell/predicted/ would be with this nvcc
#                             (every probe under `warplens run`; provenance.json names the
#                             nvcc). The predictions of the end-to-end programs
#                             (cutlass_tutorials, triton, unchanged_scripts, book_hgemm) are
#                             not made again but copied from the committed ones, so that
#                             compare.py has them; they are informational, and
#                             e2e/COPIED-FROM-COMMITTED.txt lists the copies.
#   DIR/timing/               what validation/blackwell_timing/predicted/ would be
#   DIR/DIFFERENCES.txt       which of them differ from the committed ones, by name
#   DIR/toolkit.txt           the nvcc, and how to compare with these predictions
#
# env/remote/run.sh runs it as its stage `predict-toolkit`, before the GPU stages, when and
# only when setup.sh reported a fallback toolkit; the directory is then in the archive as
# out/predicted-toolkit/. At home, for the toolkit a rental used or will use (RUNBOOK.md,
# "Driver older than the toolkit"):
#
#   WARPLENS_REMOTE_HOME=$HOME/warplens-13.0 env/remote/setup.sh --toolkit 13.0 --no-torch --no-ncu
#   WARPLENS_REMOTE_HOME=$HOME/warplens-13.0 env/remote/predict_toolkit.sh --build --out predicted-13.0
#
#   --build      build the programs first (validation/blackwell/build.sh and the `sim`
#                programs of validation/blackwell_timing). run.sh has built them already
#   --jobs N     probes to simulate at once (default: the CPUs, at most 8)
#   --no-timing  only the probe predictions
#
# Runs inside the env of setup.sh (WARPLENS_REMOTE_HOME), or the one WARPLENS_ENV_RUN names.
# Exit status: 0 both sets were written, 1 a step failed (what exists is kept), 2 usage.
set -uo pipefail
# shellcheck source=env/remote/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

out=""
build=0
timing=1
jobs="$(nproc 2>/dev/null || echo 4)"
(( jobs > 8 )) && jobs=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="${2:?--out needs a directory}"; shift ;;
    --build) build=1 ;;
    --jobs) jobs="${2:?--jobs needs a number}"; shift ;;
    --no-timing) timing=0 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$out" ]] || { echo "error: --out DIR is required (see --help)" >&2; exit 2; }
env_installed || { echo "error: no env in $REMOTE_HOME; run env/remote/setup.sh, or set WARPLENS_ENV_RUN" >&2; exit 1; }
mkdir -p "$out" || exit 1
out="$(cd "$out" && pwd)"
say() { echo "predict-toolkit: $*"; }
failed=0
probes="$REPO_DIR/validation/blackwell"
timing_dir="$REPO_DIR/validation/blackwell_timing"
cd "$REPO_DIR" || exit 1

if (( build )); then
  say "building the probes and the timing programs for the simulator"
  env_run env NVCC=nvcc bash "$probes/build.sh" > "$out/build-blackwell.log" 2>&1 \
    || { say "validation/blackwell/build.sh failed (see $out/build-blackwell.log)"; failed=1; }
  if (( timing )); then
    env_run env NVCC=nvcc bash "$timing_dir/build.sh" sim > "$out/build-timing.log" 2>&1 \
      || { say "validation/blackwell_timing/build.sh sim failed (see $out/build-timing.log)"; failed=1; }
  fi
fi

built="$(sed -n 's/^nvcc_version=//p' "$probes/bin/toolkit.txt" 2>/dev/null)"
committed="$(predictions_nvcc blackwell)"
say "the programs in validation/blackwell/bin were built by nvcc ${built:-unknown}; the committed predictions were made with nvcc ${committed:-unknown}"

say "probe predictions ($jobs at once) into $out/blackwell"
rm -rf "$out/blackwell" "$out/sim-run"
env_run env PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$probes/run_all.py" --mode predict --no-build --no-e2e \
    --predicted "$out/blackwell" --out "$out/sim-run" --jobs "$jobs" > "$out/predict-blackwell.log" 2>&1 \
  || { say "run_all.py --mode predict failed or some probe was not built (see $out/predict-blackwell.log)"; failed=1; }
rm -f "$out/sim-run.tar.gz"
# The end-to-end predictions are not made here (they need the CUTLASS programs, Triton and
# the book); compare.py shows them next to the GPU's result lines, so it gets the committed ones.
if [[ -d "$probes/predicted/e2e" && -d "$out/blackwell" ]]; then
  mkdir -p "$out/blackwell/e2e"
  : > "$out/blackwell/e2e/COPIED-FROM-COMMITTED.txt"
  for file in "$probes/predicted/e2e"/*; do
    [[ -e "$out/blackwell/e2e/$(basename "$file")" ]] && continue
    cp "$file" "$out/blackwell/e2e/" && basename "$file" >> "$out/blackwell/e2e/COPIED-FROM-COMMITTED.txt"
  done
fi

if (( timing )); then
  say "timing predictions into $out/timing"
  rm -rf "$out/timing"
  env_run env PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$timing_dir/predict.py" --out "$out/timing" \
      > "$out/predict-timing.log" 2>&1 \
    || { say "validation/blackwell_timing/predict.py failed (see $out/predict-timing.log)"; failed=1; }
fi

# Which predictions depend on the compiler: the names that differ from the committed files.
differences() {  # differences NEW COMMITTED GLOB... -> lines "differs NAME" / "only-here NAME" / "missing NAME"
  local new="$1" old="$2" file name; shift 2
  [[ -d "$new" ]] || return 0
  while IFS= read -r -d '' file; do
    name="${file#"$new"/}"
    if [[ ! -f "$old/$name" ]]; then echo "only-here $name"
    elif [[ "$name" == *.sim.txt ]]; then
      # How the simulator's run ended is the first line; the rest is its commentary
      # (instruction counts, first-run notices), which is not a prediction.
      [[ "$(head -n 1 "$file")" == "$(head -n 1 "$old/$name")" ]] || echo "differs $name"
    elif ! cmp -s "$file" "$old/$name"; then echo "differs $name"; fi
  done < <(find "$new" -type f \( "$@" \) -print0 | sort -z)
  while IFS= read -r -d '' file; do
    name="${file#"$old"/}"
    [[ -f "$new/$name" ]] || echo "missing $name"
  done < <(find "$old" -type f \( "$@" \) -print0 | sort -z)
}
{
  echo "# Predictions made with nvcc ${built:-unknown}, against the committed ones (nvcc ${committed:-unknown})."
  echo "# differs: another prediction with this compiler. missing: not made here. only-here: not committed."
  echo "# Probe predictions: NAME.txt is what the probe prints; NAME.sim.txt how the simulator's run ended"
  echo "# (compared by its first line only: the status)."
  echo "[blackwell]"
  differences "$out/blackwell" "$probes/predicted" -name '*.txt' | grep -v ' e2e/COPIED-FROM-COMMITTED.txt$'
  if (( timing )); then
    echo "[timing]"
    # One .out per benchmark. meta.json names the nvcc and predictions.md is made from the
    # .out files, so those two differ whenever anything does.
    differences "$out/timing" "$timing_dir/predicted" -name '*.out'
  fi
} > "$out/DIFFERENCES.txt"
n_probe="$(find "$out/blackwell" -name '*.txt' ! -name '*.sim.txt' ! -name 'COPIED-FROM-COMMITTED.txt' 2>/dev/null \
             | grep -vxFf <(sed "s|^|$out/blackwell/e2e/|" "$out/blackwell/e2e/COPIED-FROM-COMMITTED.txt" 2>/dev/null) | wc -l)"
d_probe="$(sed -n '/^\[blackwell\]/,/^\[timing\]/p' "$out/DIFFERENCES.txt" | grep '^differs ' | grep -vc '\.sim\.txt$' || true)"
s_probe="$(sed -n '/^\[blackwell\]/,/^\[timing\]/p' "$out/DIFFERENCES.txt" | grep -c '^differs .*\.sim\.txt$' || true)"
m_probe="$(sed -n '/^\[blackwell\]/,/^\[timing\]/p' "$out/DIFFERENCES.txt" | grep -c '^missing ' || true)"
d_timing="$(sed -n '/^\[timing\]/,$p' "$out/DIFFERENCES.txt" | grep -c '^differs \|^missing ' || true)"
(( timing )) || d_timing="not-made (--no-timing)"
{
  echo "nvcc_version=${built:-unknown}"
  echo "committed_predictions_nvcc_version=${committed:-unknown}"
  echo "probe_predictions=$n_probe"
  echo "probe_predictions_that_differ_from_committed=$d_probe"
  echo "probe_runs_that_end_differently_in_the_simulator=$s_probe"
  echo "probe_prediction_files_missing_here=$m_probe"
  echo "timing_prediction_files_that_differ_or_are_missing=$d_timing"
  echo "compare_probes=python validation/blackwell/compare.py <the blackwell results or archive> --predicted <this directory>/blackwell"
  echo "compare_timing=python validation/blackwell_timing/fit.py <the timing results> --predicted <this directory>/timing"
} > "$out/toolkit.txt"
say "$n_probe probe predictions written; $d_probe differ from the committed ones, $s_probe end differently in the simulator, $m_probe committed file(s) were not made here"
(( timing )) && say "timing: $d_timing prediction file(s) differ from the committed ones or are missing (instruction counts are the compiler's)"
say "names in $out/DIFFERENCES.txt; how to compare in $out/toolkit.txt"
if (( failed )); then
  say "A STEP FAILED: these predictions are incomplete (logs in $out)"
  exit 1
fi
exit 0
