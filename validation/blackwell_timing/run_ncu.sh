#!/usr/bin/env bash
# Run selected benchmarks under Nsight Compute and save the raw counter CSVs.
#
#   ./run_ncu.sh [--dry-run] [BIN_DIR] [OUT_DIR]        defaults: bin/hw, results_ncu
#
#   --dry-run   print every ncu command this script would run, one per line, and whether
#               its program is built; run nothing and write nothing. Needs no ncu and no
#               GPU: it is how the command lines are checked where counters cannot be read
#               (tests/test_ncu_scripts.py also runs the whole script against a stand-in
#               for ncu that replays recorded output; see fixture_ncu/).
#   --check-metrics CHIP[,CHIP...]
#               ask ncu which of the counter names below it lists for those chips
#               (`ncu --query-metrics --chips gb100`; `ncu --list-chips` names them), print
#               "ok" or "UNLISTED" per name, exit 1 if a name is unlisted. Needs ncu but no
#               GPU and no permission, so it can be run before the rental and as any user.
#               "UNLISTED" is a warning, not proof: ncu 2024.1 does not list
#               l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal for ga103, and on the
#               3080 Ti it accepted the name and reported "n/a" rather than failing
#               (validation/ampere/transpose_ncu_3080ti_grid64.csv).
#   NCU_BIN     the ncu to use (default: ncu on PATH)
#
# If ncu rejects a counter list, the list is tried again without the names the device does
# not list, and only then with smsp__inst_executed.sum alone; STATUS.txt says which.
#
# Modeled on validation/ampere/run_ncu.sh. First checks whether ncu can read performance
# counters at all: it needs root or a driver configured for non-admin profiling
# (https://developer.nvidia.com/ERR_NVGPUCTRPERM), and rented containers often block it.
# If it cannot, the script says so, writes the reason to OUT_DIR/STATUS.txt and exits 0, so
# that run_all.sh goes on. Never uses sudo.
#
# Under ncu every kernel launch is profiled (and replayed once per counter group), so each
# benchmark runs with one trial, 1% of its repetitions and a subset of its configurations.
# Counters are per launch; rows are in the same order as the BENCH lines saved next to them.
#
#   <bench>.csv   raw `ncu --csv` output       <bench>.out   the program's own output
#   transpose.csv the three layouts of examples/01_smem_transpose, grid 64 (validation matrix
#                 TIM-16/TIM-17), compared with the simulator if Python and warplens are there
#   ncu_banks.csv the same layouts in block shapes 32x32, 32x8 and 16x16 (TIM-17), compared
#                 by ncu_banks_compare.py
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
dry=0
check_chips=""
if [[ "${1:-}" == --dry-run ]]; then dry=1; shift; fi
if [[ "${1:-}" == --check-metrics ]]; then check_chips="${2:?--check-metrics needs a chip name, e.g. gb100}"; shift 2; fi
BIN="${1:-bin/hw}"
OUT="${2:-results_ncu}"
NCU="${NCU_BIN:-ncu}"

SMEM=smsp__inst_executed.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
CORE=smsp__inst_executed.sum,smsp__cycles_elapsed.max,smsp__cycles_active.avg,sm__cycles_elapsed.max,smsp__issue_active.avg
GLOBAL=smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum
BASIC=smsp__inst_executed.sum,smsp__cycles_elapsed.max,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum

# listed_metrics TABLE_FILE LIST -> the names of LIST whose base name (without .sum, .max,
# ...) starts a line of TABLE_FILE, which is what `ncu --query-metrics` printed.
listed_metrics() {
  local table="$1" name kept=()
  local IFS=,
  for name in $2; do
    grep -q "^${name%.*} " "$table" && kept+=("$name")
  done
  echo "${kept[*]}"
}

if [[ -n "$check_chips" ]]; then
  command -v "$NCU" > /dev/null 2>&1 || { echo "check-metrics: \`$NCU\` is not installed or not on PATH" >&2; exit 2; }
  table="$(mktemp)"
  unlisted=0
  for chip in ${check_chips//,/ }; do
    if ! "$NCU" --query-metrics --chips "$chip" > "$table" 2>&1 || ! grep -q '^Metric Name' "$table"; then
      echo "check-metrics: $NCU has no metric table for chip '$chip': $(head -n 2 "$table" | tr '\n' ' ')" >&2
      rm -f "$table"; exit 2
    fi
    for name in $(tr ',' '\n' <<<"$SMEM,$CORE,$GLOBAL,$BASIC" | sort -u); do
      if [[ -n "$(listed_metrics "$table" "$name")" ]]; then echo "$chip ok       $name"
      else echo "$chip UNLISTED $name"; unlisted=$((unlisted + 1)); fi
    done
  done
  rm -f "$table"
  echo "check-metrics: $unlisted name(s) not listed by $("$NCU" --version 2>/dev/null | sed -n 's/^Version \([0-9.]*\).*/\1/p' | head -n 1)"
  (( unlisted == 0 )); exit $?
fi
(( dry )) || mkdir -p "$OUT"

finish() { echo "$1" | tee "$OUT/STATUS.txt"; exit 0; }

if (( ! dry )); then
  command -v "$NCU" > /dev/null 2>&1 || finish "ncu not run: \`$NCU\` is not installed or not on PATH"
  [[ -x "$BIN/launch" ]] || finish "ncu not run: $BIN/launch is not built"

  # 1. Can counters be read? One empty kernel, one counter.
  timeout --kill-after=10 60 "$NCU" --csv --metrics smsp__inst_executed.sum "$BIN/launch" --trials 1 --only ctas=1 \
    > "$OUT/probe.csv" 2> "$OUT/probe.err"
  code=$?
  if grep -q "ERR_NVGPUCTRPERM" "$OUT/probe.csv" "$OUT/probe.err"; then
    finish "ncu not run: no permission to read GPU performance counters (ERR_NVGPUCTRPERM); needs root or NVreg_RestrictProfilingToAdminUsers=0"
  fi
  if [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/probe.csv"; then
    finish "ncu not run: the probe produced no counter table (exit $code): $(grep -v '^$' "$OUT/probe.csv" "$OUT/probe.err" | head -3 | tr '\n' ' ')"
  fi
  echo "ncu can read counters; profiling" >&2
  # The counters this device lists, for the retry in profile().
  timeout --kill-after=10 60 "$NCU" --query-metrics > "$OUT/metrics_table.txt" 2>&1 || true
else
  echo "dry-run: $NCU --csv --metrics smsp__inst_executed.sum $BIN/launch --trials 1 --only ctas=1   # can counters be read? $([[ -x "$BIN/launch" ]] && echo built || echo NOT BUILT)"
fi

ok=0; bad=0; reduced=()
profile() {  # name, metrics, program, arguments...
  local name="$1" metrics="$2"; shift 2
  if (( dry )); then
    echo "dry-run: $NCU --csv --metrics $metrics $*   # -> $OUT/$name.csv; $([[ -x "$1" ]] && echo built || echo NOT BUILT)"
    return
  fi
  [[ -x "$1" ]] || { echo "$name: $1 is not built; skipped" >&2; return; }
  local code
  timeout --kill-after=10 1200 "$NCU" --csv --metrics "$metrics" "$@" > "$OUT/$name.csv" 2> "$OUT/$name.err"
  code=$?
  if [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/$name.csv"; then
    # A counter name this GPU or ncu version does not have fails the whole run. First drop
    # the names the device does not list, then fall back to one counter.
    cp "$OUT/$name.csv" "$OUT/$name.first_attempt.csv" 2>/dev/null
    local retry_list=""
    grep -q '^Metric Name' "$OUT/metrics_table.txt" 2>/dev/null && retry_list="$(listed_metrics "$OUT/metrics_table.txt" "$metrics")"
    if [[ -n "$retry_list" && "$retry_list" != "$metrics" ]]; then
      echo "$name: failed with the full counter list, retrying with the listed ones: $retry_list" >&2
      timeout --kill-after=10 1200 "$NCU" --csv --metrics "$retry_list" "$@" > "$OUT/$name.csv" 2>> "$OUT/$name.err"
      code=$?
      reduced+=("$name: without $(comm -23 <(tr ',' '\n' <<<"$metrics" | sort) <(tr ',' '\n' <<<"$retry_list" | sort) | tr '\n' ' ')")
    fi
    if [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/$name.csv"; then
      echo "$name: retrying with smsp__inst_executed.sum only" >&2
      timeout --kill-after=10 1200 "$NCU" --csv --metrics smsp__inst_executed.sum "$@" > "$OUT/$name.csv" 2>> "$OUT/$name.err"
      code=$?
      reduced+=("$name: smsp__inst_executed.sum only")
    fi
  fi
  if [[ $code -eq 0 ]] && grep -q '^"ID"' "$OUT/$name.csv"; then
    ok=$((ok + 1))
    # The program's own lines are mixed into ncu's stdout before the table; keep them apart too.
    grep -E '^(DEVICE|BENCH) ' "$OUT/$name.csv" > "$OUT/$name.out"
  else
    bad=$((bad + 1)); echo "$name: no counter table" >&2
  fi
}

ARGS=(--trials 1 --scale 0.01)
profile alu_chain "$CORE" "$BIN/alu_chain" "${ARGS[@]}"
profile smem_wavefronts "$SMEM" "$BIN/smem_wavefronts" "${ARGS[@]}"
profile global_load "$GLOBAL" "$BIN/global_load" "${ARGS[@]}"
profile cp_async "$GLOBAL" "$BIN/cp_async" "${ARGS[@]}"
profile mbar "$CORE" "$BIN/mbar" "${ARGS[@]}"
profile tma "$BASIC" "$BIN/tma" "${ARGS[@]}" --only ctas=1
profile mma_rate "$CORE" "$BIN/mma_rate" "${ARGS[@]}" --only m=128/n=256
profile tmem_access "$CORE" "$BIN/tmem_access" "${ARGS[@]}"
# Requests that fail are the ones with a per-request slope (see clc_latency.cu).
profile clc_latency "$CORE" "$BIN/clc_latency" "${ARGS[@]}" --only fail

# The transpose layouts in three block shapes (validation matrix TIM-16, TIM-17): is the
# wavefront surplus of ncu over the model per CTA, per warp or per instruction?
profile ncu_banks "$SMEM" "$BIN/ncu_banks" --trials 1
if (( dry )); then
  echo "dry-run: ${PYTHON:-python3} ncu_banks_compare.py $OUT/ncu_banks.csv   # -> $OUT/ncu_banks_compare.txt"
elif [[ -s "$OUT/ncu_banks.csv" ]] && grep -q '^"ID"' "$OUT/ncu_banks.csv"; then
  PYTHONPATH="$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}" ${PYTHON:-python3} ncu_banks_compare.py "$OUT/ncu_banks.csv" \
    > "$OUT/ncu_banks_compare.txt" 2>&1 \
    || echo "ncu_banks: instruction or conflict counters differ from the simulator, or the comparison could not run; see $OUT/ncu_banks_compare.txt" >&2
fi

# The bank-counter comparison of validation/ampere, on this GPU.
TRANSPOSE=../ampere/bin/transpose_hw
if [[ ! -x "$TRANSPOSE" ]] && (( ! dry )); then
  mkdir -p bin/ncu
  TRANSPOSE=bin/ncu/transpose_hw
  ${HW_NVCC:-${NVCC:-nvcc}} -o "$TRANSPOSE" ../ampere/transpose_hw.cu > "$OUT/transpose_build.log" 2>&1 \
    || echo "transpose_hw: build failed, see $OUT/transpose_build.log" >&2
fi
profile transpose "$SMEM" "$TRANSPOSE" 64
if [[ -s "$OUT/transpose.csv" ]] && grep -q '^"ID"' "$OUT/transpose.csv"; then
  ${PYTHON:-python3} ../ampere/compare_ncu.py "$OUT/transpose.csv" 64 > "$OUT/transpose_compare.txt" 2>&1 \
    || echo "transpose: counters differ from the simulator, or the comparison could not run; see $OUT/transpose_compare.txt" >&2
fi

if (( dry )); then
  echo "dry-run: ${PYTHON:-python3} ../ampere/compare_ncu.py $OUT/transpose.csv 64   # -> $OUT/transpose_compare.txt"
  exit 0
fi
{
  echo "ncu: $ok benchmark(s) profiled, $bad failed; raw CSVs in $OUT"
  (( ${#reduced[@]} == 0 )) || printf 'ncu: reduced counter list for %s\n' "${reduced[@]}"
} | tee "$OUT/STATUS.txt"
exit 0
