#!/usr/bin/env bash
# Run every timing microbenchmark once, unattended, and save the raw outputs.
#
#   ./run_all.sh [--ampere] [--bin DIR] [--out DIR] [--timeout SECONDS] [--trials N]
#                [--scale X] [--no-ncu] [--no-fit]
#
#   default    the programs in bin/hw (sm_100a), for the B200
#   --ampere   the programs in bin/ampere: the benchmarks that are valid on Ampere, for a
#              dry run on the local GPU
#   --scale X  multiply every repetition count (longer launches, less relative noise)
#
# Each benchmark runs in its own process under `timeout`; a failure or a hang is recorded
# and the script goes on. Nothing is asked. Output directory (default
# results_<host>_<UTC time>): <bench>.out (stdout: DEVICE and BENCH lines), <bench>.err,
# status.txt (exit code and seconds per benchmark), env.txt, ncu/ (run_ncu.sh), fit/
# (fit.py, if Python with PyYAML is there), and <dir>.tar.gz of all of it.
# Exit status: the number of benchmarks that failed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BIN=bin/hw
OUT=""
TIMEOUT=900
TRIALS=5
SCALE=1
NCU=1
FIT=1
BENCHES=(launch alu_chain smem_wavefronts global_load gstream cp_async mbar tma mma_rate tmem_access)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ampere) BIN=bin/ampere; BENCHES=(launch alu_chain smem_wavefronts global_load gstream cp_async) ;;
    --bin) BIN="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --timeout) TIMEOUT="$2"; shift ;;
    --trials) TRIALS="$2"; shift ;;
    --scale) SCALE="$2"; shift ;;
    --no-ncu) NCU=0 ;;
    --no-fit) FIT=0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$OUT" ]] || OUT="results_$(hostname -s 2>/dev/null || echo host)_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

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
run_one() {  # name, program, extra arguments...
  local name="$1" program="$2"; shift 2
  # A repeated --out must not let an absent program reuse an old measurement.
  : > "$OUT/$name.out"
  : > "$OUT/$name.err"
  if [[ ! -x "$program" ]]; then
    echo "$name: missing program $program" | tee -a "$OUT/status.txt" >&2
    failed=$((failed + 1)); return
  fi
  local start=$SECONDS code
  timeout --kill-after=10 "$TIMEOUT" "$program" --trials "$TRIALS" --scale "$SCALE" "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  code=$?
  local note="ok"
  [[ $code -eq 0 ]] || note="FAILED"
  [[ $code -ne 124 && $code -ne 137 ]] || note="TIMED OUT after ${TIMEOUT}s"
  echo "$name: exit=$code seconds=$((SECONDS - start)) lines=$(grep -c '^BENCH ' "$OUT/$name.out") $note" \
    | tee -a "$OUT/status.txt" >&2
  [[ $code -eq 0 ]] || failed=$((failed + 1))
}

for b in "${BENCHES[@]}"; do
  run_one "$b" "$BIN/$b"
done
# Hardware-only variant with no simulator counterpart, in its own process: if the i8
# descriptor is wrong the launch fails and takes the CUDA context with it.
if [[ " ${BENCHES[*]} " == *" mma_rate "* ]]; then
  run_one mma_rate_i8 "$BIN/mma_rate" --only kind=i8
fi

if [[ $NCU -eq 1 ]]; then
  ./run_ncu.sh "$BIN" "$OUT/ncu" > "$OUT/ncu.log" 2>&1
  echo "ncu: $(tail -1 "$OUT/ncu.log")" | tee -a "$OUT/status.txt" >&2
fi

if [[ $FIT -eq 1 ]]; then
  PYTHON="${PYTHON:-python3}"
  if $PYTHON fit.py "$OUT" > "$OUT/fit.log" 2>&1; then
    echo "fit: ok, see $OUT/fit/report.md" | tee -a "$OUT/status.txt" >&2
  else
    echo "fit: fit.py failed here (see $OUT/fit.log); run it later on the saved directory" \
      | tee -a "$OUT/status.txt" >&2
  fi
fi

tar czf "${OUT%/}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")" \
  && echo "run_all: packed ${OUT%/}.tar.gz; $failed benchmark(s) failed" >&2
exit "$failed"
