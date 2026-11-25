#!/usr/bin/env bash
# Run every timing microbenchmark once, unattended, and save the raw outputs.
#
#   ./run_all.sh [--ampere] [--bin DIR] [--out DIR] [--timeout SECONDS] [--trials N]
#                [--scale X] [--no-ncu] [--ncu-timeout SECONDS] [--ncu-budget SECONDS] [--no-fit]
#
#   default    the programs in bin/hw (sm_100a), for the B200
#   --ampere   the programs in bin/ampere: the benchmarks that are valid on Ampere, for a
#              dry run on the local GPU
#   --scale X  multiply every repetition count (longer launches, less relative noise)
#   --timeout SECONDS      limit of one benchmark (default 180, times --scale if that is
#                          above 1; the README has the measured times it rests on)
#   --ncu-timeout SECONDS  limit of one ncu run (run_ncu.sh --timeout; default 300)
#   --ncu-budget SECONDS   limit of all ncu work of this run (run_ncu.sh --budget; default 1200)
#
# Order, so that a run cut short has done what is proven first:
#   1. the benchmarks that are valid on Ampere (all of them have run on the RTX 3080 Ti)
#   2. ncu for those programs (run_ncu.sh --group ampere), TIM-16/TIM-17 first
#   3. the Blackwell-only benchmarks (never run on a GPU), then mma_rate_i8
#   4. ncu for the Blackwell-only programs (run_ncu.sh --group blackwell --continue)
#   5. fit.py, and the archive
#
# Each benchmark runs in its own process under `timeout`; a failure or a hang is recorded
# and the script goes on. Nothing is asked. Output directory (default
# results_<host>_<UTC time>): <bench>.out (stdout: DEVICE and BENCH lines), <bench>.err,
# status.txt (exit code and seconds per benchmark), env.txt, ncu/ (run_ncu.sh), fit/
# (fit.py, if Python with PyYAML is there), and <dir>.tar.gz of all of it.
#
# Counters: <out>/ncu/STATUS.txt always exists at the end, and its first line starts with
# `ncu ok:`, `ncu partial:`, `ncu failed:` or `ncu not run:` (run_ncu.sh has the meanings;
# with --no-ncu it is `ncu not run: not requested`). status.txt repeats that line.
#
# Exit status: the number of benchmarks that failed, plus 1 if the caller said that counters
# can be read on this machine (WARPLENS_NCU=ok, set by env/remote/run.sh after preflight
# read one) and the result is nevertheless `ncu failed:` or `ncu not run:`. `ncu partial:`
# adds nothing, and neither does anything under --no-ncu or without WARPLENS_NCU=ok.
# Benchmarks and fit.py run to the end either way. 143, 130 or 129 after SIGTERM, SIGINT or
# SIGHUP: the running benchmark or ncu is stopped with its whole process group (nothing is
# left on the GPU), status.txt says what was stopped and what was not run, and no archive
# is made.
#
# status.txt tells apart a benchmark that ran and failed ("FAILED"), one that did not end
# within its limit ("TIMED OUT after Ns"; the BENCH lines it had printed are kept, see
# lines=), and one whose kernels
# the driver could not load ("LAUNCH FAILED": exit status 4 and a BENCH_ERROR line with
# class=launch-failed; see common.h). TOOLKIT.txt records the driver, the nvcc that built
# the programs and the nvcc the committed predictions (predicted/meta.json) were made with;
# when the last two differ it and status.txt say "BUILT WITH FALLBACK TOOLKIT - PREDICTIONS
# NOT COMPARABLE": the predicted instruction counts are those of another compiler's code.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

BIN=bin/hw
OUT=""
TIMEOUT=""
TRIALS=5
SCALE=1
NCU=1
FIT=1
NCU_ARGS=()
NCU_BUDGET_SECONDS="${NCU_BUDGET:-1200}"
# The same two lists as build.sh (AMPERE is the first one).
AMPERE_VALID=(launch alu_chain smem_wavefronts global_load gstream cp_async ncu_banks)
BLACKWELL_ONLY=(mbar tma mma_rate tmem_access clc_latency cluster_costs pipelined_gemm)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ampere) BIN=bin/ampere; BLACKWELL_ONLY=() ;;
    --bin) BIN="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --timeout) TIMEOUT="$2"; shift ;;
    --trials) TRIALS="$2"; shift ;;
    --scale) SCALE="$2"; shift ;;
    --no-ncu) NCU=0 ;;
    --ncu-timeout) NCU_ARGS+=(--timeout "$2"); shift ;;
    --ncu-budget) NCU_ARGS+=(--budget "$2"); NCU_BUDGET_SECONDS="$2"; shift ;;
    --no-fit) FIT=0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ -z "$TIMEOUT" ]]; then
  # 180 s at --scale 1; a run asked to take X times longer gets X times the limit.
  TIMEOUT="$(awk -v s="$SCALE" 'BEGIN { t = 180 * (s > 1 ? s : 1); printf "%d", (t == int(t)) ? t : int(t) + 1 }')"
fi
if [[ ! "$TIMEOUT" =~ ^[1-9][0-9]*$ || ! "$NCU_BUDGET_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "run_all: --timeout and --ncu-budget are whole numbers of seconds above 0" >&2; exit 2
fi
[[ -n "$OUT" ]] || OUT="results_$(hostname -s 2>/dev/null || echo host)_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT/artifacts/binaries"
: > "$OUT/artifacts/SHA256SUMS"

if [[ ! -x "$BIN/launch" ]]; then
  echo "run_all: $BIN is not built; building" >&2
  case "$BIN" in bin/ampere) ./build.sh ampere ;; bin/sim) ./build.sh sim ;; *) ./build.sh hw ;; esac \
    > "$OUT/build.log" 2>&1 || echo "run_all: build failed, see $OUT/build.log" >&2
fi

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname 2>/dev/null)"
  echo "uname: $(uname -a)"
  echo "bin: $BIN  trials: $TRIALS  scale: $SCALE  timeout: $TIMEOUT"
  echo "git: $(git rev-parse HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) modified file(s)"
  echo "--- nvidia-smi"; nvidia-smi 2>&1
  echo "--- nvidia-smi query"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.max.sm,compute_mode,persistence_mode \
    --format=csv 2>&1
  echo "--- other compute processes on the GPU (other tenants or a desktop disturb the timings)"
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>&1
  echo "--- nvcc"; (${NVCC:-nvcc} --version 2>&1 | tail -2)
  echo "--- ncu"; (${NCU_BIN:-ncu} --version 2>&1 | head -3)
  echo "--- cpu"; (lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|MHz'); cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
  echo "--- sha256 of the programs"; sha256sum "$BIN"/* 2>&1
} > "$OUT/env.txt"

failed=0
: > "$OUT/status.txt"
rm -rf "$OUT/ncu"

# -- children: one at a time, in the background, stopped when this script is ------------------
child=""
current=""
current_start=0
not_run=("${AMPERE_VALID[@]}" "${BLACKWELL_ONLY[@]}")
[[ " ${BLACKWELL_ONLY[*]} " != *" mma_rate "* ]] || not_run+=(mma_rate_i8)
run_limited() {  # run_limited SECONDS COMMAND...   -> the status `timeout` returns
  local limit="$1" code
  shift
  # In the background: bash runs a trap only between commands, so a foreground child
  # would keep SIGTERM waiting until it ended by itself (up to the benchmark's limit).
  timeout --kill-after=10 "$limit" "$@" &
  child=$!
  wait "$child"
  code=$?
  child=""
  return "$code"
}
stop_child() {  # TERM to the child's process group (timeout makes one), KILL after 8 s
  local pid="$child" i
  [[ -n "$pid" ]] || return 0
  child=""
  kill -TERM -- "-$pid" 2>/dev/null
  kill -TERM "$pid" 2>/dev/null
  # 8 s: run_ncu.sh needs up to 3 s to stop its own child before it exits.
  for (( i = 0; i < 80; i++ )); do
    kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
}
on_signal() {  # on_signal NAME EXIT_STATUS
  trap - TERM INT HUP EXIT
  local was_running="$child"
  stop_child
  {
    if [[ -n "$was_running" && -n "$current" ]]; then
      echo "$current: STOPPED by SIG$1 after $((SECONDS - current_start)) s (run_all.sh was told to stop)"
    fi
    echo "run_all: stopped by SIG$1; not run: ${not_run[*]:-nothing}; no fit, no archive"
  } | tee -a "$OUT/status.txt" >&2
  exit "$2"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
trap stop_child EXIT

# Which toolkit built the programs, against the one of the committed predictions.
built_nvcc="$(sed -n 's/^nvcc_version=//p' "$BIN/toolkit.txt" 2>/dev/null | head -n 1)"
predicted_nvcc="$(sed -n 's/.*"nvcc_version": *"\([^"]*\)".*/\1/p' predicted/meta.json 2>/dev/null | head -n 1)"
{
  echo "driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  echo "driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  echo "bin=$BIN"
  echo "built_nvcc_version=${built_nvcc:-unknown}"
  echo "predictions_nvcc_version=${predicted_nvcc:-unknown}"
  if [[ "$BIN" == bin/ampere ]]; then
    echo "predictions_comparable=no"
    echo "reason=the Ampere dry-run build: another compiler and target; the predictions are for a B200"
  elif [[ -z "$built_nvcc" || "$built_nvcc" == unknown || -z "$predicted_nvcc" ]]; then
    echo "predictions_comparable=unknown"
    echo "reason=TOOLKIT NOT RECORDED: $BIN/toolkit.txt (build.sh) or predicted/meta.json (predict.py) does not name an nvcc"
  elif [[ "$built_nvcc" != "$predicted_nvcc" ]]; then
    echo "predictions_comparable=no"
    echo "reason=BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: programs built by nvcc $built_nvcc, predictions made from the code of nvcc $predicted_nvcc"
  else
    echo "predictions_comparable=yes"
    echo "reason=nvcc $built_nvcc as for the predictions"
  fi
} > "$OUT/TOOLKIT.txt"
if ! grep -q '^predictions_comparable=yes' "$OUT/TOOLKIT.txt"; then
  echo "toolkit: $(sed -n 's/^reason=//p' "$OUT/TOOLKIT.txt")" | tee -a "$OUT/status.txt" >&2
fi
run_one() {  # name, program, extra arguments...
  local name="$1" program="$2"; shift 2
  # A repeated --out must not let an absent program reuse an old measurement.
  : > "$OUT/$name.out"
  : > "$OUT/$name.err"
  not_run=("${not_run[@]:1}")
  if [[ ! -x "$program" ]]; then
    echo "$name: missing program $program" | tee -a "$OUT/status.txt" >&2
    failed=$((failed + 1)); return
  fi
  # Keep the exact program and invocation with the raw trial samples. The frozen
  # source tree in the outer rental archive supplies its build script and sources.
  mkdir -p "$OUT/artifacts/binaries" || { failed=$((failed + 1)); return; }
  cp "$program" "$OUT/artifacts/binaries/$name" || { failed=$((failed + 1)); return; }
  (cd "$OUT/artifacts" && sha256sum "binaries/$name") >> "$OUT/artifacts/SHA256SUMS"
  printf '%q ' "$program" --trials "$TRIALS" --scale "$SCALE" "$@" > "$OUT/artifacts/$name.command"
  printf '\n' >> "$OUT/artifacts/$name.command"
  local start=$SECONDS code
  current="$name"; current_start=$SECONDS
  run_limited "$TIMEOUT" "$program" --trials "$TRIALS" --scale "$SCALE" "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  code=$?
  current=""
  local note="ok"
  [[ $code -eq 0 ]] || note="FAILED"
  [[ $code -ne 124 && $code -ne 137 ]] || note="TIMED OUT after ${TIMEOUT}s"
  if [[ $code -eq 4 ]] || grep -q 'class=launch-failed' "$OUT/$name.out" 2>/dev/null; then
    note="LAUNCH FAILED (the driver could not load the kernels: $(sed -n 's/.* error=\([A-Za-z0-9_]*\).*/\1/p' "$OUT/$name.out" | head -n 1))"
  fi
  echo "$name: exit=$code seconds=$((SECONDS - start)) lines=$(grep -c '^BENCH ' "$OUT/$name.out") $note" \
    | tee -a "$OUT/status.txt" >&2
  [[ $code -eq 0 ]] || failed=$((failed + 1))
}

ncu_phase() {  # ncu_phase GROUP [--continue]: run_ncu.sh for one group of programs
  [[ $NCU -eq 1 ]] || return 0
  current="run_ncu.sh --group $1"; current_start=$SECONDS
  # run_ncu.sh keeps to its budget by itself; the limit here is a backstop.
  run_limited "$((NCU_BUDGET_SECONDS + 120))" ./run_ncu.sh --group "$@" "${NCU_ARGS[@]}" "$BIN" "$OUT/ncu" >> "$OUT/ncu.log" 2>&1
  current=""
}

# 1. Valid on Ampere: every one of these has run on the 3080 Ti.
for b in "${AMPERE_VALID[@]}"; do
  run_one "$b" "$BIN/$b"
done
# 2. Their counters, the comparisons of TIM-16 and TIM-17 first.
if [[ $NCU -eq 1 ]]; then
  : > "$OUT/ncu.log"
  ncu_phase ampere
  echo "ncu, first part (the programs valid on Ampere): $(head -n 1 "$OUT/ncu/STATUS.txt" 2>/dev/null || echo "no STATUS.txt, see $OUT/ncu.log")" \
    | tee -a "$OUT/status.txt" >&2
fi
# 3. Blackwell only: never run on a GPU. A hang costs its limit, and no longer the above.
for b in "${BLACKWELL_ONLY[@]}"; do
  run_one "$b" "$BIN/$b"
done
# Hardware-only variant with no simulator counterpart, in its own process: if the i8
# descriptor is wrong the launch fails and takes the CUDA context with it.
if [[ " ${BLACKWELL_ONLY[*]} " == *" mma_rate "* ]]; then
  run_one mma_rate_i8 "$BIN/mma_rate" --only kind=i8
fi
# 4. Their counters, added to the first part's.
ncu_phase blackwell --continue

# The verdict on the counters: <out>/ncu/STATUS.txt, its first line repeated in status.txt.
mkdir -p "$OUT/ncu"
if [[ $NCU -eq 0 ]]; then
  echo "ncu not run: not requested (run_all.sh --no-ncu)" > "$OUT/ncu/STATUS.txt"
elif [[ ! -s "$OUT/ncu/STATUS.txt" ]]; then
  echo "ncu failed: run_ncu.sh left no STATUS.txt (see $OUT/ncu.log)" > "$OUT/ncu/STATUS.txt"
fi
ncu_line="$(head -n 1 "$OUT/ncu/STATUS.txt")"
echo "$ncu_line" | tee -a "$OUT/status.txt" >&2
ncu_missed=0
if [[ $NCU -eq 1 && "${WARPLENS_NCU:-}" == ok ]] && [[ "$ncu_line" == "ncu failed:"* || "$ncu_line" == "ncu not run:"* ]]; then
  ncu_missed=1
  echo "run_all: COUNTERS REQUIRED AND NOT READ: WARPLENS_NCU=ok says preflight read one on this machine; counted as a failure in the exit status" \
    | tee -a "$OUT/status.txt" >&2
fi

if [[ $FIT -eq 1 ]]; then
  read -r -a python <<<"${PYTHON:-python3}"
  current="fit.py"; current_start=$SECONDS
  if run_limited 600 "${python[@]}" fit.py "$OUT" > "$OUT/fit.log" 2>&1; then
    echo "fit: ok, see $OUT/fit/report.md" | tee -a "$OUT/status.txt" >&2
  else
    echo "fit: fit.py failed here (see $OUT/fit.log); run it later on the saved directory" \
      | tee -a "$OUT/status.txt" >&2
  fi
  current=""
fi

tar czf "${OUT%/}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")" \
  && echo "run_all: packed ${OUT%/}.tar.gz; $failed benchmark(s) failed" >&2
exit "$((failed + ncu_missed))"
