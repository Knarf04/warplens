#!/usr/bin/env bash
# Run selected benchmarks under Nsight Compute and save the raw counter CSVs.
#
#   ./run_ncu.sh [options] [BIN_DIR] [OUT_DIR]          defaults: bin/hw, results_ncu
#
#   --dry-run   print every ncu command this script would run, one per line, and whether
#               its program is built; run nothing and write nothing. Needs no ncu and no
#               GPU: it is how the command lines are checked where counters cannot be read
#               (tests/validation/probes/test_ncu_scripts.py also runs the whole script against a stand-in
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
#   --group ampere|blackwell|all
#               which programs to profile (default all, the Ampere-valid ones first):
#                 ampere     ncu_banks, transpose, alu_chain, smem_wavefronts, global_load,
#                            cp_async: the programs that run on Ampere. All of them have
#                            been profiled on the RTX 3080 Ti; the first two are the
#                            comparisons with the simulator (matrix rows TIM-16, TIM-17)
#                 blackwell  mbar, tma, mma_rate, tmem_access, clc_latency: never profiled
#   --continue  add to the results of an earlier call for the same OUT_DIR: no second
#               probe, the earlier profiles and the time they took are kept. run_all.sh
#               uses `--group ampere`, then `--group blackwell --continue`.
#   --timeout SECONDS         limit of one ncu run (default 300; NCU_PROFILE_TIMEOUT)
#   --budget SECONDS          limit of everything this script does for one OUT_DIR, both
#                             calls of a --continue pair together (default 1200; NCU_BUDGET)
#   --probe-timeout SECONDS   limit of the permission probe (default 300; NCU_PROBE_TIMEOUT)
#   NCU_BIN     the ncu to use (default: ncu on PATH)
#
# OUT_DIR/STATUS.txt is the result, and its FIRST LINE starts with exactly one of
#   ncu ok:       every profile that was attempted produced a counter table
#   ncu partial:  some did; the line names the ones that did not and why
#   ncu failed:   ncu was run and no counter table was read
#   ncu not run:  nothing was attempted: no ncu, no permission, nothing built to profile
# The lines after it are for people (reduced counter lists, programs not built, seconds).
# It is rewritten after every profile, so it is true also when the script is killed.
# OUT_DIR/profiles.tsv has one row per program: name, state, detail.
# The exit status is 0 whenever STATUS.txt could be written, whatever it says: run_all.sh
# goes on after this script and reads the first line. 143, 130 or 129 after SIGTERM,
# SIGINT or SIGHUP; 2 for a bad argument.
#
# Limits. Every ncu run is under `timeout`; one that reaches its limit (exit status 124 or
# 137) is NOT run again: only a run that ncu itself ended without a table is retried, first
# without the counter names the device does not list, then with smsp__inst_executed.sum
# alone; STATUS.txt says which. When the budget is used up the remaining programs are not
# started and STATUS.txt names them. The script therefore ends within the budget plus the
# 10 s that `timeout --kill-after` gives the last command.
#
# Stopping. Each command runs in the background under `timeout`, which makes it a process
# group of its own; on SIGTERM, SIGINT, SIGHUP or exit this script signals that group (ncu,
# its launcher and the profiled program) and after 3 s kills it, so nothing is left on the GPU.
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
group=all
resume=0
PROFILE_TIMEOUT="${NCU_PROFILE_TIMEOUT:-300}"
BUDGET="${NCU_BUDGET:-1200}"
PROBE_TIMEOUT="${NCU_PROBE_TIMEOUT:-300}"
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry=1 ;;
    --check-metrics) check_chips="${2:?--check-metrics needs a chip name, e.g. gb100}"; shift ;;
    --group) group="${2:?--group needs ampere, blackwell or all}"; shift ;;
    --continue) resume=1 ;;
    --timeout) PROFILE_TIMEOUT="${2:?--timeout needs seconds}"; shift ;;
    --budget) BUDGET="${2:?--budget needs seconds}"; shift ;;
    --probe-timeout) PROBE_TIMEOUT="${2:?--probe-timeout needs seconds}"; shift ;;
    --*) echo "run_ncu: unknown option: $1" >&2; exit 2 ;;
    *) positional+=("$1") ;;
  esac
  shift
done
BIN="${positional[0]:-bin/hw}"
OUT="${positional[1]:-results_ncu}"
NCU="${NCU_BIN:-ncu}"
case "$group" in ampere|blackwell|all) ;; *) echo "run_ncu: --group is ampere, blackwell or all, not '$group'" >&2; exit 2 ;; esac
for seconds in "$PROFILE_TIMEOUT" "$BUDGET" "$PROBE_TIMEOUT"; do
  if [[ ! "$seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "run_ncu: a limit must be a whole number of seconds above 0, not '$seconds'" >&2; exit 2
  fi
done
read -r -a python <<<"${PYTHON:-python3}"

SMEM=smsp__inst_executed.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
CORE=smsp__inst_executed.sum,smsp__cycles_elapsed.max,smsp__cycles_active.avg,sm__cycles_elapsed.max,smsp__issue_active.avg
GLOBAL=smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum
BASIC=smsp__inst_executed.sum,smsp__cycles_elapsed.max,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum

# The programs in the order they are profiled: what has been proven first.
AMPERE_GROUP=(ncu_banks transpose alu_chain smem_wavefronts global_load cp_async)
BLACKWELL_GROUP=(mbar tma mma_rate tmem_access clc_latency)
in_group() {  # in_group NAME: is it one of the programs this call profiles?
  case "$group" in
    all) return 0 ;;
    ampere) [[ " ${AMPERE_GROUP[*]} " == *" $1 "* ]] ;;
    blackwell) [[ " ${BLACKWELL_GROUP[*]} " == *" $1 "* ]] ;;
  esac
}

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

# -- children: one at a time, in the background, stopped when this script is ------------------
child=""
run_limited() {  # run_limited SECONDS COMMAND...   -> the status `timeout` returns
  local limit="$1" code
  shift
  # In the background: bash runs a trap only between commands, so a foreground child
  # would keep SIGTERM waiting until it ended by itself.
  timeout --kill-after=10 "$limit" "$@" &
  child=$!
  wait "$child"
  code=$?
  child=""
  return "$code"
}
stop_child() {  # TERM to the child's process group (timeout makes one), KILL after 3 s
  local pid="$child" i
  [[ -n "$pid" ]] || return 0
  child=""
  kill -TERM -- "-$pid" 2>/dev/null
  kill -TERM "$pid" 2>/dev/null
  for (( i = 0; i < 30; i++ )); do
    kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
}
on_signal() {  # on_signal NAME EXIT_STATUS
  trap - TERM INT HUP EXIT
  stop_child
  if (( ! dry )) && [[ -f "$OUT/profiles.tsv" ]]; then
    echo "ncu: run_ncu.sh was stopped by SIG$1" >> "$OUT/notes.txt"
    write_status > /dev/null
  elif (( ! dry )) && [[ -d "$OUT" && ! -s "$OUT/STATUS.txt" ]]; then
    echo "ncu failed: run_ncu.sh was stopped by SIG$1 before the probe had read a counter" > "$OUT/STATUS.txt"
  fi
  echo "run_ncu: stopped by SIG$1" >&2
  exit "$2"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
trap stop_child EXIT

# -- the budget ---------------------------------------------------------------------------------
used_before=0
t0=$SECONDS
elapsed() { echo $(( used_before + SECONDS - t0 )); }
remaining() { echo $(( BUDGET - used_before - (SECONDS - t0) )); }

# -- what became of each program -------------------------------------------------------------
# state: ok | failed | timeout | budget | not-built | not-selected | pending
declare -A state=() detail=()
save_state() {
  local name
  for name in "${AMPERE_GROUP[@]}" "${BLACKWELL_GROUP[@]}"; do
    printf '%s\t%s\t%s\n' "$name" "${state[$name]}" "${detail[$name]}"
  done > "$OUT/profiles.tsv.tmp" && mv "$OUT/profiles.tsv.tmp" "$OUT/profiles.tsv"
  elapsed > "$OUT/budget_used"
}
write_status() {  # STATUS.txt from the states; prints it
  local name n_ok=0 n_failed=0 n_not=0 missing=() not_built=() verdict line
  for name in "${AMPERE_GROUP[@]}" "${BLACKWELL_GROUP[@]}"; do
    case "${state[$name]}" in
      ok) n_ok=$((n_ok + 1)) ;;
      failed|timeout) n_failed=$((n_failed + 1)); missing+=("$name (${detail[$name]})") ;;
      not-built) not_built+=("$name") ;;
      *) n_not=$((n_not + 1)); missing+=("$name (${detail[$name]})") ;;
    esac
  done
  save_state
  if (( n_ok + n_failed + n_not == 0 )); then
    line="ncu not run: counters can be read (the probe passed), but none of the programs to profile is built in $BIN"
  else
    if (( n_failed + n_not == 0 )); then verdict=ok
    elif (( n_ok > 0 )); then verdict=partial
    else verdict=failed; fi
    line="ncu $verdict: $n_ok benchmark(s) profiled, $n_failed failed"
    (( n_not == 0 )) || line+=", $n_not not profiled"
    if (( ${#missing[@]} )); then
      line+="; no counter table from: $(printf '%s, ' "${missing[@]}" | sed 's/, $//')"
    fi
    line+="; raw CSVs in $OUT"
  fi
  {
    echo "$line"
    [[ ! -s "$OUT/notes.txt" ]] || cat "$OUT/notes.txt"
    (( ${#not_built[@]} == 0 )) || echo "ncu: not built in $BIN, so not profiled: ${not_built[*]}"
    echo "ncu: $(elapsed) s of the budget of $BUDGET s used; one row per program in $OUT/profiles.tsv"
  } > "$OUT/STATUS.txt.tmp" && mv "$OUT/STATUS.txt.tmp" "$OUT/STATUS.txt"
  cat "$OUT/STATUS.txt"
}

# For the endings before anything was profiled: one line, and run_all.sh goes on.
finish() { echo "$1" | tee "$OUT/STATUS.txt"; exit 0; }

# The probe: `launch` with one trial launches its spin-up kernel, a warm-up and 40 empty
# kernels; `--launch-count 1` has ncu profile the first of them only and let the rest run.
probe() {  # probe [NCU OPTION...]
  run_limited "$PROBE_TIMEOUT" "$NCU" --csv "$@" --metrics smsp__inst_executed.sum "$BIN/launch" --trials 1 --only ctas=1 \
    > "$OUT/probe.csv" 2> "$OUT/probe.err"
}

TRANSPOSE=../ampere/bin/transpose_hw
if (( dry )); then
  echo "dry-run: $NCU --csv --launch-count 1 --metrics smsp__inst_executed.sum $BIN/launch --trials 1 --only ctas=1   # can counters be read? limit $PROBE_TIMEOUT s; $([[ -x "$BIN/launch" ]] && echo built || echo NOT BUILT)"
elif (( resume )) && [[ -f "$OUT/profiles.tsv" ]]; then
  # The earlier call read a counter: keep what it profiled and the time it took.
  while IFS=$'\t' read -r name what why; do
    state[$name]="$what"; detail[$name]="$why"
  done < "$OUT/profiles.tsv"
  used_before="$(cat "$OUT/budget_used" 2>/dev/null)"
  [[ "$used_before" =~ ^[0-9]+$ ]] || used_before=0
  [[ -x "$TRANSPOSE" || ! -x bin/ncu/transpose_hw ]] || TRANSPOSE=bin/ncu/transpose_hw
elif (( resume )) && [[ -s "$OUT/STATUS.txt" ]]; then
  # The earlier call ended before profiling (no ncu, no permission, no table): that stands.
  cat "$OUT/STATUS.txt"
  exit 0
else
  rm -f "$OUT/STATUS.txt" "$OUT/profiles.tsv" "$OUT/notes.txt" "$OUT/budget_used"
  command -v "$NCU" > /dev/null 2>&1 || finish "ncu not run: \`$NCU\` is not installed or not on PATH"
  [[ -x "$BIN/launch" ]] || finish "ncu not run: $BIN/launch is not built"

  # 1. Can counters be read? One kernel, one counter.
  probe --launch-count 1
  code=$?
  if grep -q "ERR_NVGPUCTRPERM" "$OUT/probe.csv" "$OUT/probe.err"; then
    finish "ncu not run: no permission to read GPU performance counters (ERR_NVGPUCTRPERM); needs root or NVreg_RestrictProfilingToAdminUsers=0"
  fi
  if grep -q 'class=launch-failed' "$OUT/probe.csv"; then
    finish "ncu not run: $BIN/launch could not launch its kernels on this GPU ($(sed -n 's/.* error=\([A-Za-z0-9_]*\).*/\1/p' "$OUT/probe.csv" | head -n 1)), so there was nothing to profile"
  fi
  if [[ $code -ne 124 && $code -ne 137 ]] && { [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/probe.csv"; }; then
    # --launch-count has not been used where counters can be read. Before concluding that
    # they cannot, ask once more the way that has (as root on the 3080 Ti): every launch.
    cp "$OUT/probe.csv" "$OUT/probe.first_attempt.csv" 2>/dev/null
    echo "probe: no counter table with --launch-count 1 (exit $code); once more without it" >&2
    probe
    code=$?
    if [[ $code -eq 0 ]] && grep -q '^"ID"' "$OUT/probe.csv"; then
      echo "ncu: the probe read a counter only without --launch-count 1 (first attempt in $OUT/probe.first_attempt.csv)" >> "$OUT/notes.txt"
    fi
  fi
  if [[ $code -eq 124 || $code -eq 137 ]]; then
    finish "ncu failed: the probe produced no counter table (exit $code: it did not end within its limit of $PROBE_TIMEOUT s): $(grep -hv '^$' "$OUT/probe.csv" "$OUT/probe.err" | head -3 | tr '\n' ' ')"
  fi
  if [[ $code -ne 0 ]] || ! grep -q '^"ID"' "$OUT/probe.csv"; then
    finish "ncu failed: the probe produced no counter table (exit $code): $(grep -hv '^$' "$OUT/probe.csv" "$OUT/probe.err" | head -3 | tr '\n' ' ')"
  fi
  echo "ncu can read counters; profiling" >&2
  # The counters this device lists, for the retry in profile().
  run_limited 60 "$NCU" --query-metrics > "$OUT/metrics_table.txt" 2>&1 || true

  # The bank-counter comparison of validation/ampere needs its program.
  if [[ ! -x "$TRANSPOSE" ]]; then
    mkdir -p bin/ncu
    TRANSPOSE=bin/ncu/transpose_hw
    read -r -a nvcc <<<"${HW_NVCC:-${NVCC:-nvcc}}"
    # Machine code for this GPU and no PTX. nvcc's default is PTX for the driver to compile,
    # and a driver older than the toolkit refuses it: on the B200 (driver for CUDA 13.3,
    # nvcc 13.5) every launch was cudaErrorUnsupportedPtxVersion and ncu had nothing to count,
    # in both rentals. The benchmarks in bin/hw are built this way for the same reason.
    arch=()
    capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')"
    [[ "$capability" =~ ^[0-9]+$ ]] && arch=(-gencode "arch=compute_$capability,code=sm_$capability")
    run_limited 120 "${nvcc[@]}" "${arch[@]}" -o "$TRANSPOSE" ../ampere/transpose_hw.cu > "$OUT/transpose_build.log" 2>&1 \
      || echo "transpose_hw: build failed, see $OUT/transpose_build.log" >&2
  fi
  for name in "${AMPERE_GROUP[@]}" "${BLACKWELL_GROUP[@]}"; do
    program="$BIN/$name"
    [[ "$name" != transpose ]] || program="$TRANSPOSE"
    if [[ -x "$program" ]]; then state[$name]=pending; detail[$name]="not reached: run_ncu.sh was stopped, or is still running"
    else state[$name]=not-built; detail[$name]="$program is not built"; fi
  done
fi

if (( ! dry )); then
  # What this call leaves to another one is said now, so that STATUS.txt is true throughout.
  for name in "${AMPERE_GROUP[@]}" "${BLACKWELL_GROUP[@]}"; do
    if in_group "$name"; then
      [[ "${state[$name]}" == not-built ]] || { state[$name]=pending; detail[$name]="not reached: run_ncu.sh was stopped, or is still running"; }
    elif [[ "${state[$name]}" == pending ]]; then
      state[$name]=not-selected; detail[$name]="not in this run: --group $group"
    fi
  done
  write_status > /dev/null
fi

# attempt NAME METRICS PROGRAM ARGUMENTS...: one ncu run under both limits.
#   -> 0 a counter table, 1 none, 2 reached its limit, 3 not started (budget used up)
attempt_code=0
attempt_limit=0
attempt() {
  local name="$1" metrics="$2" left
  shift 2
  left="$(remaining)"
  (( left > 0 )) || return 3
  attempt_limit=$(( left < PROFILE_TIMEOUT ? left : PROFILE_TIMEOUT ))
  run_limited "$attempt_limit" "$NCU" --csv --metrics "$metrics" "$@" > "$OUT/$name.csv" 2>> "$OUT/$name.err"
  attempt_code=$?
  [[ $attempt_code -ne 124 && $attempt_code -ne 137 ]] || return 2
  [[ $attempt_code -eq 0 ]] && grep -q '^"ID"' "$OUT/$name.csv"
}

profile() {  # name, metrics, program, arguments...
  local name="$1" metrics="$2"; shift 2
  in_group "$name" || return 0
  if (( dry )); then
    echo "dry-run: $NCU --csv --metrics $metrics $*   # -> $OUT/$name.csv; limit $PROFILE_TIMEOUT s; $([[ -x "$1" ]] && echo built || echo NOT BUILT)"
    return
  fi
  [[ "${state[$name]}" != not-built ]] || { echo "$name: $1 is not built; skipped" >&2; return; }
  local result retry_list=""
  : > "$OUT/$name.err"
  attempt "$name" "$metrics" "$@"
  result=$?
  if (( result == 1 )); then
    # A counter name this GPU or ncu version does not have fails the whole run. First drop
    # the names the device does not list, then fall back to one counter. Only for a run
    # that ncu ended itself: one that reached its limit would reach it again.
    cp "$OUT/$name.csv" "$OUT/$name.first_attempt.csv" 2>/dev/null
    grep -q '^Metric Name' "$OUT/metrics_table.txt" 2>/dev/null && retry_list="$(listed_metrics "$OUT/metrics_table.txt" "$metrics")"
    if [[ -n "$retry_list" && "$retry_list" != "$metrics" ]]; then
      echo "$name: failed with the full counter list, retrying with the listed ones: $retry_list" >&2
      attempt "$name" "$retry_list" "$@"
      result=$?
      (( result > 1 )) || echo "ncu: reduced counter list for $name: without $(comm -23 <(tr ',' '\n' <<<"$metrics" | sort) <(tr ',' '\n' <<<"$retry_list" | sort) | tr '\n' ' ')" >> "$OUT/notes.txt"
    fi
    if (( result == 1 )); then
      echo "$name: retrying with smsp__inst_executed.sum only" >&2
      attempt "$name" smsp__inst_executed.sum "$@"
      result=$?
      (( result > 1 )) || echo "ncu: reduced counter list for $name: smsp__inst_executed.sum only" >> "$OUT/notes.txt"
    fi
  fi
  case "$result" in
    0) state[$name]=ok; detail[$name]="counter table in $name.csv"
       # The program's own lines are mixed into ncu's stdout before the table; keep them apart too.
       grep -E '^(DEVICE|BENCH) ' "$OUT/$name.csv" > "$OUT/$name.out" ;;
    2) state[$name]=timeout
       if (( attempt_limit < PROFILE_TIMEOUT )); then detail[$name]="stopped after $attempt_limit s, the rest of the budget of $BUDGET s; not run again"
       else detail[$name]="timed out after $attempt_limit s; not run again"; fi
       echo "$name: ${detail[$name]}" >&2 ;;
    3) state[$name]=budget; detail[$name]="not started: the budget of $BUDGET s was used up"
       echo "$name: ${detail[$name]}" >&2 ;;
    *) state[$name]=failed; detail[$name]="ncu exit $attempt_code, no counter table"
       echo "$name: no counter table" >&2 ;;
  esac
  write_status > /dev/null
}

compare() {  # compare NAME OUTPUT COMMAND...: the profiled counters against the simulator
  local name="$1" output="$2" left limit
  shift 2
  in_group "$name" || return 0
  if (( dry )); then
    echo "dry-run: $*   # -> $output"
    return
  fi
  [[ "${state[$name]}" == ok ]] || return 0
  left="$(remaining)"
  if (( left <= 0 )); then
    echo "ncu: $name was not compared with the simulator (budget used up); later: $*" >> "$OUT/notes.txt"
    return
  fi
  limit=$(( left < 120 ? left : 120 ))
  run_limited "$limit" "$@" > "$output" 2>&1 \
    || echo "$name: counters differ from the simulator, or the comparison could not run; see $output" >&2
}

ARGS=(--trials 1 --scale 0.01)

# First what has been proven on the 3080 Ti. The transpose layouts in three block shapes
# (validation matrix TIM-16, TIM-17): is the wavefront surplus of ncu over the model per
# CTA, per warp or per instruction?
profile ncu_banks "$SMEM" "$BIN/ncu_banks" --trials 1
if (( dry )); then
  compare ncu_banks "$OUT/ncu_banks_compare.txt" "${python[@]}" ncu_banks_compare.py "$OUT/ncu_banks.csv"
else
  compare ncu_banks "$OUT/ncu_banks_compare.txt" \
    env "PYTHONPATH=$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}" "${python[@]}" ncu_banks_compare.py "$OUT/ncu_banks.csv"
fi
# The bank-counter comparison of validation/ampere, on this GPU.
profile transpose "$SMEM" "$TRANSPOSE" 64
compare transpose "$OUT/transpose_compare.txt" "${python[@]}" ../ampere/compare_ncu.py "$OUT/transpose.csv" 64

# The other programs that run on Ampere (all profiled there, 2026-11-24).
profile alu_chain "$CORE" "$BIN/alu_chain" "${ARGS[@]}"
profile smem_wavefronts "$SMEM" "$BIN/smem_wavefronts" "${ARGS[@]}"
profile global_load "$GLOBAL" "$BIN/global_load" "${ARGS[@]}"
profile cp_async "$GLOBAL" "$BIN/cp_async" "${ARGS[@]}"

# Blackwell only: never profiled, and so last.
profile mbar "$CORE" "$BIN/mbar" "${ARGS[@]}"
profile tma "$BASIC" "$BIN/tma" "${ARGS[@]}" --only ctas=1
profile mma_rate "$CORE" "$BIN/mma_rate" "${ARGS[@]}" --only m=128/n=256
profile tmem_access "$CORE" "$BIN/tmem_access" "${ARGS[@]}"
# Requests that fail are the ones with a per-request slope (see clc_latency.cu).
profile clc_latency "$CORE" "$BIN/clc_latency" "${ARGS[@]}" --only fail

(( dry )) && exit 0
write_status
exit 0
