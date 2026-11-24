#!/usr/bin/env bash
# Run selected benchmarks under Nsight Compute and save the raw counter CSVs.
#
#   ./run_ncu.sh [BIN_DIR] [OUT_DIR]        defaults: bin/hw, results_ncu
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
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BIN="${1:-bin/hw}"
OUT="${2:-results_ncu}"
NCU="${NCU_BIN:-ncu}"
mkdir -p "$OUT"

finish() { echo "$1" | tee "$OUT/STATUS.txt"; exit 0; }

command -v "$NCU" > /dev/null 2>&1 || finish "ncu not run: \`$NCU\` is not installed or not on PATH"
[[ -x "$BIN/launch" ]] || finish "ncu not run: $BIN/launch is not built"

# 1. Can counters be read? One empty kernel, one counter.
"$NCU" --csv --metrics smsp__inst_executed.sum "$BIN/launch" --trials 1 --only ctas=1 \
  > "$OUT/probe.csv" 2> "$OUT/probe.err"
code=$?
if grep -q "ERR_NVGPUCTRPERM" "$OUT/probe.csv" "$OUT/probe.err"; then
  finish "ncu not run: no permission to read GPU performance counters (ERR_NVGPUCTRPERM); needs root or NVreg_RestrictProfilingToAdminUsers=0"
fi
if [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/probe.csv"; then
  finish "ncu not run: the probe produced no counter table (exit $code): $(grep -v '^$' "$OUT/probe.csv" "$OUT/probe.err" | head -3 | tr '\n' ' ')"
fi
echo "ncu can read counters; profiling" >&2

SMEM=smsp__inst_executed.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
CORE=smsp__inst_executed.sum,smsp__cycles_elapsed.max,smsp__cycles_active.avg,sm__cycles_elapsed.max,smsp__issue_active.avg
GLOBAL=smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum
BASIC=smsp__inst_executed.sum,smsp__cycles_elapsed.max,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum

ok=0; bad=0
profile() {  # name, metrics, program, arguments...
  local name="$1" metrics="$2"; shift 2
  [[ -x "$1" ]] || { echo "$name: $1 is not built; skipped" >&2; return; }
  if ! timeout --kill-after=10 1200 "$NCU" --csv --metrics "$metrics" "$@" > "$OUT/$name.csv" 2> "$OUT/$name.err" \
      || ! grep -q '^"ID"' "$OUT/$name.csv"; then
    # A counter name this GPU or ncu version does not have fails the whole run: retry with one.
    echo "$name: failed with the full counter list, retrying with smsp__inst_executed.sum only" >&2
    cp "$OUT/$name.csv" "$OUT/$name.first_attempt.csv" 2>/dev/null
    timeout --kill-after=10 1200 "$NCU" --csv --metrics smsp__inst_executed.sum "$@" > "$OUT/$name.csv" 2>> "$OUT/$name.err"
  fi
  if grep -q '^"ID"' "$OUT/$name.csv"; then
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

# The bank-counter comparison of validation/ampere, on this GPU.
TRANSPOSE=../ampere/bin/transpose_hw
if [[ ! -x "$TRANSPOSE" ]]; then
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

echo "ncu: $ok benchmark(s) profiled, $bad failed; raw CSVs in $OUT" | tee "$OUT/STATUS.txt"
exit 0
