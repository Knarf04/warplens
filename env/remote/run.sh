#!/usr/bin/env bash
# The whole rental in one command: check the machine, install the env, build
# and run both probe suites, pack one results archive.
#
#   env/remote/run.sh [--dry-run] [--keep-going] [--redo STAGE|all] [--skip STAGE] [--collect-only]
#                     [--toolkit auto|pinned|X.Y] [--no-ncu] [--no-torch] [--no-book] [--strict-toolkit]
#                     [--price DOLLARS_PER_HOUR]
#
# Never asks anything. Safe to run again after an interruption or a failure:
# stages that finished are skipped, the rest run, and the archive is rebuilt.
# Ctrl-C stops the running stage, writes the archive and exits with status 130.
#
# Stages, in order, and what a failure does:
#   preflight-quick  GPU identity, driver, disk          stops the run (not a B200: nothing to do)
#   setup            env/remote/setup.sh                 stops the run (nothing can be built)
#   preflight        kernels on the GPU, tensor map, a   stops the run, before anything is
#                    counter read by ncu, PyTorch, book  built or run on the GPU
#   selftest         the repository's tests, simulator   continues
#   build-blackwell  validation/blackwell/build.sh       its run stage is skipped; continues
#   build-timing     validation/blackwell_timing/build.sh  likewise
#   predict-toolkit  env/remote/predict_toolkit.sh       continues. Only with a fallback toolkit
#                    (setup.sh installed another nvcc than the committed predictions were
#                    made with): the predictions are made again with the installed nvcc,
#                    before anything runs on the GPU, into out/predicted-toolkit/ of the archive
#   run-blackwell    validation/blackwell/run_all.sh     continues. It fails, among other things,
#                    when the capture of matrix row RT-11 (a CuTe DSL kernel with TMA) is
#                    missing on a B200; the reason is then repeated as the run's last lines
#   run-timing      validation/blackwell_timing/run_all.sh  continues
#   collect          manifest, checksums, archive        always runs
#
#   --dry-run      print the plan and the time and cost estimate; touch nothing
#   --keep-going   do not stop after a failed preflight (the archive records it)
#   --redo STAGE   run STAGE again even if it finished (repeatable; `all` for every stage)
#   --skip STAGE   leave STAGE out (repeatable), e.g. --skip selftest on a second rental
#   --collect-only run nothing; write the archive from what earlier runs left
#   --toolkit, --no-torch, --no-book   passed to setup.sh
#   --no-ncu       run without performance counters. By default a run that includes the
#                  timing suite must read one for real in preflight: no ncu, no permission
#                  (ERR_NVGPUCTRPERM) or no value fails preflight and stops the run, since
#                  the timing model is held against counters. With --no-ncu that is a
#                  warning, setup.sh installs no ncu, and the archive says that no counter
#                  was read: NCU-COUNTERS-NOT-READ.txt, ncu= and ncu_required= in
#                  manifest.txt, ncu_counters= in resolved/versions.txt. A run without
#                  the timing suite (--skip run-timing) does not need counters either
#   --strict-toolkit   stop after setup if the probes would not be built by the
#                  nvcc the committed predictions were made with (a fallback toolkit).
#                  Without it the run goes on and is flagged: PREDICTIONS-NOT-COMPARABLE.txt
#                  at the top of the archive, predictions_comparable in manifest.txt, and
#                  a banner at the end
#   --price        hourly price for the cost estimate of --dry-run (default 8)
#
#   WARPLENS_REMOTE_HOME     where the env and results go (default ~/warplens-remote)
#   WARPLENS_BOOK            a clone of the book for the end-to-end step. Not set: the
#                            clone setup.sh fetches into $WARPLENS_REMOTE_HOME/book, at the
#                            commit in pins.env. The book is never copied into the archive
#   WARPLENS_BUILD_MINUTES   time limit of each build stage (default 30)
#   WARPLENS_RUN_MINUTES     time limit of each run stage (default 60)
#
# Each suite script runs inside the env with these set:
#   NVCC=nvcc PYTHON=python     the env's compiler and interpreter
#   WARPLENS_RESULTS_DIR        a directory of its own under the run directory;
#                               everything written there is archived
#   WARPLENS_PTX_JIT            ok | refused | unknown: whether the driver takes
#                               PTX from this toolkit (preflight's ptx-jit check)
#   WARPLENS_NCU                ok | unavailable: whether preflight read a counter with ncu
#   WARPLENS_BOOK               the clone of the book, when there is one
#   WARPLENS_TORCH_PYTHON       the interpreter of the PyTorch env, when there is one
#   WARPLENS_GDB                a gdb for the rows that trace a launch, when there is one
#   WARPLENS_PREDICTIONS        comparable | comparable-native | not-comparable | unknown
# Whatever a suite prints is archived too (logs/<stage>.log), and so are
# directories named results*, out or output that the run creates inside the
# suite's directory.
# Arguments: validation/blackwell/build.sh gets --native when the driver
# refuses the toolkit's PTX; validation/blackwell_timing/run_all.sh gets
# --out <its results directory>. WARPLENS_BLACKWELL_BUILD_ARGS,
# WARPLENS_BLACKWELL_RUN_ARGS, WARPLENS_TIMING_BUILD_ARGS and
# WARPLENS_TIMING_RUN_ARGS replace those when set.
#
# Exit status: 0 everything ran and passed, 1 some stage failed (the archive
# is still written), 3 the run stopped early, 130 interrupted, 2 usage.
set -uo pipefail
# shellcheck source=env/remote/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

stage_body_only=0
stage_body_args=()
if [[ "${1:-}" == --stage-body ]]; then   # internal, see stage_body below
  shift; stage_body_only=1; stage_body_args=("$@"); set --
fi
dry_run=0
keep_going=0
collect_only=0
strict_toolkit=0
no_ncu=0
price=8
redo=()
skip=()
setup_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --keep-going) keep_going=1 ;;
    --collect-only) collect_only=1 ;;
    --redo) redo+=("${2:?--redo needs a stage}"); shift ;;
    --skip) skip+=("${2:?--skip needs a stage}"); shift ;;
    --toolkit) setup_args+=(--toolkit "${2:?--toolkit needs a value}"); shift ;;
    --no-ncu) setup_args+=(--no-ncu); no_ncu=1 ;;
    --no-torch) setup_args+=(--no-torch) ;;
    --no-book) setup_args+=(--no-book) ;;
    --strict-toolkit) strict_toolkit=1 ;;
    --price) price="${2:?--price needs a number}"; shift ;;
    -h|--help) sed -n '2,78p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ "$price" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "error: --price takes a number, not '$price'" >&2; exit 2; }

STAGES=(preflight-quick setup preflight selftest build-blackwell build-timing predict-toolkit run-blackwell run-timing collect)
for stage in "${redo[@]}" "${skip[@]}"; do
  [[ "$stage" == all || " ${STAGES[*]} " == *" $stage "* ]] \
    || { echo "error: no stage '$stage' (stages: ${STAGES[*]})" >&2; exit 2; }
done
is_skipped() { [[ " ${skip[*]} " == *" $1 "* && "$1" != collect ]]; }

# Counters are what the timing suite is held against: a run that includes it must read
# one for real (preflight's ncu check), unless it was told to do without (--no-ncu).
ncu_required=1
if (( no_ncu )) || is_skipped run-timing; then ncu_required=0; fi

build_minutes="${WARPLENS_BUILD_MINUTES:-30}"
run_minutes="${WARPLENS_RUN_MINUTES:-60}"
RUN_DIR="$REMOTE_HOME/run"
SUITE_blackwell="validation/blackwell"
SUITE_timing="validation/blackwell_timing"

# Minutes, low and high, and where each number comes from. "measured" means on
# the development machine (20-core laptop, fast network), which is not the
# rented one; "guess" means nobody has run it. RUNBOOK.md has the same table.
estimate() {
  case "$1" in
    preflight-quick) echo "1|1|measured: 2 s" ;;
    setup)           echo "3|12|measured: 75 s to download and install 5.8 GB on a fast link, and 159 s for the PyTorch env (4.3 GB installed) on the same link; a slow link or disk is the upper end" ;;
    preflight)       echo "1|2|measured: 5 s without the Blackwell kernels; allows for a first-use driver compile and ncu start-up" ;;
    selftest)        echo "4|20|measured: 10.5 min on 20 cores for the whole test suite (1513 tests; it runs the whole predict path, which is what says whether the predictions hold for the installed compiler). --skip selftest saves it" ;;
    build-blackwell) echo "1|6|measured: 8 s on 20 cores for the 51 probe programs (13 s with --native); a smaller machine is the upper end" ;;
    build-timing)    echo "1|4|measured: 62 s, one compile at a time, for the 14 benchmarks" ;;
    predict-toolkit) echo "0|15|measured: 8 min on 20 cores (65 s for the 108 probe predictions, 7 min for the 14 timing benchmarks, one at a time). Runs only when setup.sh fell back to another toolkit than the predictions' (0 otherwise)" ;;
    run-blackwell)   echo "3|20|guess: short kernels, but some probes wait out a timeout on purpose; limit $run_minutes min" ;;
    run-timing)      echo "3|20|guess: repeated microbenchmarks, and ncu reruns each kernel per counter group; limit $run_minutes min" ;;
    collect)         echo "1|1|measured: under 1 s for a small archive" ;;
  esac
}

stage_command() {
  case "$1" in
    preflight-quick) echo "env/remote/preflight.sh --quick" ;;
    setup)           echo "env/remote/setup.sh ${setup_args[*]:-}" ;;
    preflight)       echo "env/remote/preflight.sh$( (( ncu_required )) || echo " --no-ncu")   (ncu must read a counter, unless --no-ncu or --skip run-timing)" ;;
    selftest)        echo "python -m pytest -q   (in the env, CUDA_VISIBLE_DEVICES empty)" ;;
    build-blackwell) echo "$SUITE_blackwell/build.sh   (adds --native if the driver refuses the toolkit's PTX)" ;;
    build-timing)    echo "$SUITE_timing/build.sh" ;;
    predict-toolkit) echo "env/remote/predict_toolkit.sh --out <run directory>/out/predicted-toolkit   (fallback toolkit only)" ;;
    run-blackwell)   echo "$SUITE_blackwell/run_all.sh" ;;
    run-timing)      echo "$SUITE_timing/run_all.sh --out <results directory>" ;;
    collect)         echo "manifest, SHA256SUMS, tar.gz under $REMOTE_HOME/results" ;;
  esac
}

commit="$(repo_commit)"
# A successful stage is reusable only with the same inputs to the orchestration.
# Changing a toolkit or suite selection must not reuse an older successful run.
# WARPLENS_BOOK here is what the caller set, not the default taken after setup: the clone
# setup.sh fetches must not make the second run differ from the first.
resume_id="$( {
  printf '%s\0' "$commit" "ncu_required=$ncu_required" "${setup_args[@]}" "${WARPLENS_ENV_RUN:-}" "${CUDA_VISIBLE_DEVICES-unset}" \
    "${WARPLENS_BLACKWELL_BUILD_ARGS:-}" "${WARPLENS_BLACKWELL_RUN_ARGS:-}" \
    "${WARPLENS_TIMING_BUILD_ARGS:-}" "${WARPLENS_TIMING_RUN_ARGS:-}" \
    "${WARPLENS_BOOK:-}" "${CUTLASS_BIN:-}"
  cat "$PINS_FILE"
} | sha256sum | cut -d' ' -f1)"
is_done() {  # finished earlier, for this commit, and not asked to be redone
  local stage="$1" r
  for r in "${redo[@]}"; do [[ "$r" == all || "$r" == "$stage" ]] && return 1; done
  [[ -f "$RUN_DIR/state/$stage.ok" && "$(cat "$RUN_DIR/state/$stage.ok")" == "$resume_id" ]]
}
suite_script() {  # suite_script build-blackwell -> path of the script it runs
  local kind="${1%%-*}" suite="SUITE_${1#*-}"
  [[ "$kind" == build ]] && echo "$REPO_DIR/${!suite}/build.sh" || echo "$REPO_DIR/${!suite}/run_all.sh"
}

# ---------------------------------------------------------------------------
# --dry-run: the plan
# ---------------------------------------------------------------------------
if (( dry_run )); then
  echo "warplens remote run: plan (dry run, nothing is executed)"
  echo "  checkout   $REPO_DIR at ${commit:0:12}"
  echo "  home       $REMOTE_HOME (env, state, results)"
  echo
  low_total=0; high_total=0; missing=0
  printf '  %-16s %-9s %-8s %s\n' STAGE MINUTES STATE COMMAND
  for stage in "${STAGES[@]}"; do
    IFS='|' read -r low high basis <<<"$(estimate "$stage")"
    state=to-run
    if is_skipped "$stage"; then state=skip; low=0; high=0
    elif [[ "$stage" != collect ]] && is_done "$stage"; then state="done"; low=0; high=0; fi
    case "$stage" in build-*|run-*)
      [[ -f "$(suite_script "$stage")" ]] || { state=MISSING; missing=1; } ;;
    esac
    low_total=$((low_total + low)); high_total=$((high_total + high))
    printf '  %-16s %-9s %-8s %s\n' "$stage" "$low-$high" "$state" "$(stage_command "$stage")"
    printf '  %-16s   basis: %s\n' "" "$basis"
  done
  echo
  echo "  estimated wall-clock time: $low_total to $high_total minutes for the stages still to run"
  awk -v lo="$low_total" -v hi="$high_total" -v p="$price" 'BEGIN {
    printf "  estimated cost at $%.2f per hour: $%.2f to $%.2f, before the time to log in, clone and copy results back\n",
           p, lo / 60 * p, hi / 60 * p }'
  echo "  The guesses dominate the total. Budget two hours; see env/remote/RUNBOOK.md."
  if (( missing )); then
    echo
    echo "  MISSING: a probe suite's script is not in this checkout. Those stages will fail with"
    echo "  that message and the rest will run. Check out a commit that has both suites."
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
say() { echo "run: $*"; }
declare -A result=()
failures=0

# run_stage NAME MINUTES COMMAND...   -> 0 ok or done earlier, else the exit status
run_stage() {
  local stage="$1" minutes="$2" start end t0 rc status
  shift 2
  if is_skipped "$stage"; then
    skip_stage "$stage" "asked to (--skip $stage)"
    return 0
  fi
  if is_done "$stage"; then
    say "$stage: finished earlier, skipping (--redo $stage to run it again)"
    result[$stage]="done"
    return 0
  fi
  rm -f "$RUN_DIR/state/$stage.ok"
  # Rebuilt programs or a changed environment invalidate dependent results.
  case "$stage" in
    setup) rm -f "$RUN_DIR/state/"{preflight,selftest,build-blackwell,build-timing,predict-toolkit,run-blackwell,run-timing}.ok ;;
    preflight) rm -f "$RUN_DIR/state/"{build-blackwell,build-timing,run-blackwell,run-timing}.ok ;;
    build-blackwell) rm -f "$RUN_DIR/state/"{predict-toolkit,run-blackwell}.ok ;;
    build-timing) rm -f "$RUN_DIR/state/"{predict-toolkit,run-timing}.ok ;;
  esac
  if [[ -f "$RUN_DIR/logs/$stage.log" ]]; then
    mkdir -p "$RUN_DIR/logs/history"
    mv "$RUN_DIR/logs/$stage.log" "$(mktemp "$RUN_DIR/logs/history/$stage.XXXXXX.log")"
  fi
  start="$(utc_now)"; t0=$SECONDS
  say "$stage: started $start (limit $minutes min, log $RUN_DIR/logs/$stage.log)"
  # In the background, so that Ctrl-C reaches the trap while the stage runs.
  timeout --kill-after=30 "${minutes}m" "$@" > "$RUN_DIR/logs/$stage.log" 2>&1 &
  stage_pid=$!
  wait "$stage_pid"
  rc=$?
  if (( interrupted )); then      # the trap has told the stage to stop; let it
    wait "$stage_pid" 2>/dev/null
    rc=130
  fi
  stage_pid=""
  end="$(utc_now)"
  case "$rc" in
    0) status=ok; echo "$resume_id" > "$RUN_DIR/state/$stage.ok" ;;
    124|137) status=timeout ;;
    130) status=interrupted ;;
    *) status=failed ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$stage" "$status" "$rc" "$start" "$end" "$((SECONDS - t0))" >> "$RUN_DIR/stages.tsv"
  result[$stage]=$status
  say "$stage: $status after $((SECONDS - t0)) s"
  if (( interrupted )); then
    say "interrupted: packing what exists. Run env/remote/run.sh again to continue from $stage."
    finish 130
  fi
  if (( rc != 0 )); then
    failures=$((failures + 1))
    [[ "$status" == timeout ]] && say "$stage: stopped at its $minutes minute limit"
    if [[ "$stage" != preflight* ]]; then   # a preflight report is printed whole by the caller
      say "$stage: last lines of its log:"
      tail -n 15 "$RUN_DIR/logs/$stage.log" | sed 's/^/    | /'
    fi
  fi
  return "$rc"
}

skip_stage() {  # skip_stage NAME REASON
  local now; now="$(utc_now)"
  printf '%s\tskipped\t-\t%s\t%s\t0\n' "$1" "$now" "$now" >> "$RUN_DIR/stages.tsv"
  result[$1]=skipped
  say "$1: skipped: $2"
}

# Run one suite script inside the env. A missing script is this stage's failure.
suite_stage() {  # suite_stage build-blackwell [args...]
  local stage="$1" script; shift
  script="$(suite_script "$stage")"
  if [[ ! -f "$script" ]]; then
    echo "error: $script not found." >&2
    echo "The probe suite '${script%/*}' is not in this checkout (commit ${commit:0:12})." >&2
    echo "Check out a commit that contains it and run env/remote/run.sh again; finished stages are kept." >&2
    return 66
  fi
  local out="$RUN_DIR/out/${stage#*-}"
  mkdir -p "$out"
  cd "$REPO_DIR" || return 1
  env_run env NVCC=nvcc PYTHON=python WARPLENS_RESULTS_DIR="$out" \
      WARPLENS_PTX_JIT="${WARPLENS_PTX_JIT:-unknown}" WARPLENS_NCU="${WARPLENS_NCU:-unavailable}" \
      WARPLENS_PREDICTIONS="${WARPLENS_PREDICTIONS:-unknown}" \
      ${WARPLENS_TORCH_PYTHON:+WARPLENS_TORCH_PYTHON="$WARPLENS_TORCH_PYTHON"} \
      ${WARPLENS_GDB:+WARPLENS_GDB="$WARPLENS_GDB"} \
      bash "$script" "$@"
}

# Stages that are not one script run through here, in a process of their own
# so that `timeout` can end them: run.sh --stage-body STAGE [args...]
stage_body() {
  local stage="$1"; shift
  case "$stage" in
    selftest)
      cd "$REPO_DIR" || return 1
      env_run env CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider ;;
    build-*|run-*) suite_stage "$stage" "$@" ;;
    predict-toolkit)
      # Not --build: build-blackwell and build-timing have made the programs it simulates.
      bash "$REMOTE_DIR/predict_toolkit.sh" --out "$RUN_DIR/out/predicted-toolkit" ;;
    *) echo "error: no body for stage '$stage'" >&2; return 2 ;;
  esac
}
if (( stage_body_only )); then
  stage_body "${stage_body_args[@]}"
  exit $?
fi
self=(bash "$REMOTE_DIR/run.sh" --stage-body)

trap '' HUP   # a dropped ssh session must not end the run
# Ctrl-C or a kill: stop the running stage, still write the archive, exit 130.
stage_pid=""
interrupted=0
# shellcheck disable=SC2329  # called by the trap
on_interrupt() {
  interrupted=1
  [[ -n "$stage_pid" ]] && kill -TERM "$stage_pid" 2>/dev/null
}
trap on_interrupt INT TERM
mkdir -p "$RUN_DIR"/{state,logs,out} "$REMOTE_HOME/results"
[[ -f "$RUN_DIR/state/first-start" ]] || utc_now > "$RUN_DIR/state/first-start"
run_start="$(utc_now)"
echo "$run_start $commit" >> "$RUN_DIR/state/attempts"
[[ -f "$RUN_DIR/stages.tsv" ]] || printf 'stage\tstatus\texit\tstart\tend\tseconds\n' > "$RUN_DIR/stages.tsv"
# Whether counters were required is the run's decision, not that of a later --collect-only.
if (( collect_only )) && [[ -f "$RUN_DIR/state/ncu-policy" ]]; then
  read -r no_ncu ncu_required < "$RUN_DIR/state/ncu-policy"
elif (( ! collect_only )); then
  echo "$no_ncu $ncu_required" > "$RUN_DIR/state/ncu-policy"
fi

collect() {
  local archive sums name first source_failed=0
  {
    echo "# warplens remote run manifest, written $(utc_now)"
    echo "warplens_commit=$commit"
    echo "warplens_branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "warplens_uncommitted_files=$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | wc -l)"
    echo "first_start=$(cat "$RUN_DIR/state/first-start")"
    echo "this_start=$run_start"
    echo "end=$(utc_now)"
    echo "attempts=$(wc -l < "$RUN_DIR/state/attempts")"
    echo "host=$(hostname)"
    echo "kernel=$(uname -r)"
    echo "os=$( (. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}") || echo unknown)"
    echo "cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)"
    echo "cpus=$(nproc 2>/dev/null)"
    echo "container=$(container_kind)"
    echo "uid=$EUID"
    echo "ptx_jit=$ptx_jit"
    echo "ncu=$ncu_state"
    echo "ncu_required=$ncu_required"
    echo "ncu_counters=$(ncu_counters)"
    echo "book=$(book_in_use)"
    echo "book_commit=$(is_book_clone "$(book_in_use)" && book_commit "$(book_in_use)" || echo none)"
    echo "driver=$(sed -n 's/^driver=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
    echo "driver_cuda=$(sed -n 's/^driver_cuda=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
    echo "toolkit_installed=$(sed -n 's/^cuda_nvcc=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
    echo "predictions_nvcc=$(predictions_nvcc blackwell)"
    echo "predictions_comparable=$predictions_state"
    echo "predictions_comparable_why=$predictions_why"
    for stage in "${STAGES[@]}"; do
      [[ "$stage" == collect ]] || echo "stage_${stage//-/_}=${result[$stage]:-not-reached}"
    done
    echo
    echo "# versions: see resolved/versions.txt (what setup.sh installed) and pins.env (what was asked for)"
    echo "# stage history, every attempt: stages.tsv. Preflight reports: preflight-quick.txt, preflight.txt"
  } > "$RUN_DIR/manifest.txt" || return 1
  rm -rf "$RUN_DIR/resolved" "$RUN_DIR/suite-dirs" || return 1
  if [[ -d "$REMOTE_HOME/resolved" ]]; then
    cp -r "$REMOTE_HOME/resolved" "$RUN_DIR/resolved" || return 1
    # versions.txt is setup.sh's, written before anything looked at the GPU. The archive's
    # copy also says what came of the counters, next to the ncu that was installed.
    if [[ -f "$RUN_DIR/resolved/versions.txt" ]]; then
      echo "ncu_counters=$(ncu_counters)" >> "$RUN_DIR/resolved/versions.txt" || return 1
    fi
  fi
  cp "$PINS_FILE" "$RUN_DIR/pins.env" || return 1
  # Preserve the implementation itself: a commit SHA alone is not an offline oracle.
  rm -f "$RUN_DIR/SOURCE-SNAPSHOT-FAILED.txt"
  if ! python3 "$REPO_DIR/env/remote/freeze_source.py" --repo "$REPO_DIR" --out "$RUN_DIR/source" \
       > "$RUN_DIR/source-snapshot.log" 2>&1; then
    source_failed=1
    echo "The frozen source snapshot is incomplete; see source-snapshot.log. This run is not a complete capture." \
      > "$RUN_DIR/SOURCE-SNAPSHOT-FAILED.txt"
  fi
  timeout --kill-after=5 15 nvidia-smi > "$RUN_DIR/nvidia-smi.txt" 2>&1 || true
  timeout --kill-after=5 15 nvidia-smi -q > "$RUN_DIR/nvidia-smi-q.txt" 2>&1 || true
  # Impossible to miss when the archive is unpacked: a file named for what it says.
  rm -f "$RUN_DIR/PREDICTIONS-NOT-COMPARABLE.txt"
  case "$predictions_state" in
    comparable|comparable-native) ;;
    *) printf '%s\n\nEvery verdict of validation/blackwell/compare.py and every predicted time in\nvalidation/blackwell_timing is to be read with this in mind. See env/remote/RUNBOOK.md,\n"Driver older than the toolkit".\n\nPredictions made on this machine with the installed nvcc, before the GPU stages ran\n(stage predict-toolkit: %s), are in out/predicted-toolkit/. Hold the results against those:\n  validation/blackwell/compare.py out/blackwell --predicted out/predicted-toolkit/blackwell\n  validation/blackwell_timing/fit.py out/timing --predicted out/predicted-toolkit/timing\nout/predicted-toolkit/DIFFERENCES.txt names the predictions that differ from the committed ones.\n' \
         "$predictions_why" "${result[predict-toolkit]:-not-reached}" > "$RUN_DIR/PREDICTIONS-NOT-COMPARABLE.txt" ;;
  esac
  # The same for counters: a run without them is a run without the timing validation.
  rm -f "$RUN_DIR/NCU-COUNTERS-NOT-READ.txt"
  if [[ "$ncu_state" != ok ]]; then
    printf 'No GPU performance counter was read in this run: ncu_counters=%s.\n\nThe preflight check "ncu" is the evidence (preflight.txt; it PASSes only when ncu printed a\ncounter value for a kernel that ran). Without it the timing suite has launch times but no\ncounters to hold the model against: read out/timing/ with that in mind, and see\nenv/remote/RUNBOOK.md, "When preflight fails", for what grants the permission.\n' \
      "$(ncu_counters)" > "$RUN_DIR/NCU-COUNTERS-NOT-READ.txt" || return 1
  fi
  # Outputs a suite left in its own directory instead of WARPLENS_RESULTS_DIR.
  # Only what this run made: directories that came with the checkout are older
  # than the run's first start.
  for suite in "$SUITE_blackwell" "$SUITE_timing"; do
    [[ -d "$REPO_DIR/$suite" ]] || continue
    while IFS= read -r -d '' made; do
      mkdir -p "$RUN_DIR/suite-dirs/$suite" || return 1
      cp -r "$made" "$RUN_DIR/suite-dirs/$suite/" || return 1
    done < <(find "$REPO_DIR/$suite" -mindepth 1 -maxdepth 1 -type d \
               \( -name 'results*' -o -name out -o -name output \) \
               -newer "$RUN_DIR/state/first-start" -print0)
  done
  (cd "$RUN_DIR" && find . -type f ! -name SHA256SUMS ! -path './state/*' -print0 | sort -z | xargs -0 sha256sum > state/sums \
     && mv state/sums SHA256SUMS) || return 1

  first="$(tr -d ':-' < "$RUN_DIR/state/first-start")"
  name="warplens-b200-${first}-${commit:0:10}"
  archive="$REMOTE_HOME/results/$name.tar.gz"
  if ! tar -czf "$archive.part" -C "$REMOTE_HOME" --transform "s|^run|$name|" --exclude='run/state' run \
     || ! mv -f "$archive.part" "$archive"; then
    say "collect: writing $archive failed"
    return 1
  fi
  sums="$archive.sha256"
  (cd "$REMOTE_HOME/results" && sha256sum "$name.tar.gz" > "$sums.part" && mv "$sums.part" "$sums") || return 1
  say "archive:  $archive"
  say "size:     $(du -h "$archive" | cut -f1) ($(stat -c %s "$archive") bytes)"
  say "sha256:   $(cut -d' ' -f1 "$sums")"
  say "fetch it from your own machine with:"
  say "  scp '<user>@<host>:$archive*' .  &&  sha256sum -c $name.tar.gz.sha256"
  return "$source_failed"
}

finish() {  # finish EXIT_STATUS
  local code="$1"
  if ! collect; then
    say "collect: archive or checksum failed; keep the raw run directory and retry --collect-only"
    (( code == 0 )) && code=1
  fi
  say "stages: $(for s in "${STAGES[@]}"; do [[ $s == collect ]] || printf '%s=%s ' "$s" "${result[$s]:-not-reached}"; done)"
  [[ "$ncu_state" == ok ]] || say "performance counters: $(ncu_counters)"
  # Written by validation/blackwell/run_rows.py when the capture of matrix row RT-11 was
  # required (a B200) and is not there; gone again once the row has been run to `ok`.
  if [[ -f "$RUN_DIR/out/blackwell/RT-11-NOT-CAPTURED.txt" ]]; then
    local missing; missing="$(printf '!%.0s' {1..100})"
    printf '%s\n' "$missing"; cat "$RUN_DIR/out/blackwell/RT-11-NOT-CAPTURED.txt"; printf '%s\n' "$missing"
    (( code == 0 )) && code=1
  fi
  case "$predictions_state" in
    comparable|comparable-native) ;;
    *) local bar; bar="$(printf '!%.0s' {1..100})"
       printf '%s\nrun: %s\n%s\n' "$bar" "$predictions_why" "$bar" ;;
  esac
  exit "$code"
}

# Only a PASS of the ncu check is a counter that was read. WARN, FAIL, SKIP, a check
# that was never reached and a preflight that never ran are all "unavailable".
read_preflight_status() {
  ptx_jit=unknown
  ncu_state=unavailable
  ncu_check=not-run
  if [[ -f "$RUN_DIR/preflight.status" ]]; then
    case "$(awk '$1 == "ptx-jit" {print $2}' "$RUN_DIR/preflight.status")" in
      PASS) ptx_jit=ok ;;
      WARN|FAIL) ptx_jit=refused ;;
    esac
    ncu_check="$(awk '$1 == "ncu" {print $2}' "$RUN_DIR/preflight.status" | tail -n 1)"
    ncu_check="${ncu_check:-not-run}"
    [[ "$ncu_check" == PASS ]] && ncu_state=ok
  fi
  return 0
}
# One line for the archive: were counters read, and if not, was that accepted beforehand.
ncu_counters() {
  if [[ "$ncu_state" == ok ]]; then echo "read (preflight's ncu check passed)"
  elif (( no_ncu )); then echo "not-read (preflight's ncu check: $ncu_check; the run was told to do without, --no-ncu)"
  elif (( ! ncu_required )); then echo "not-read (preflight's ncu check: $ncu_check; not needed, the timing suite was left out)"
  else echo "not-read (preflight's ncu check: $ncu_check; this run REQUIRED them)"
  fi
}
# The clone of the book the Blackwell suite is given: --book among its arguments, else
# WARPLENS_BOOK, else the clone setup.sh fetched. Empty when there is none.
book_in_use() {
  local words=() word next=0 found=""
  read -r -a words <<<"${WARPLENS_BLACKWELL_RUN_ARGS:-}"
  for word in "${words[@]}"; do
    if (( next )); then found="$word"; next=0; continue; fi
    case "$word" in
      --book) next=1 ;;
      --book=*) found="${word#*=}" ;;
    esac
  done
  if [[ -n "$found" ]]; then echo "$found"; else book_path; fi
}
read_preflight_status
# Whether results can be held against the committed predictions; from setup.sh's record
# until the preflight has looked at the nvcc that is really in the env.
predictions_state="$(sed -n 's/^predictions_comparable=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
predictions_why="$(sed -n 's/^predictions_comparable_why=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
predictions_state="${predictions_state:-unknown}"
predictions_why="${predictions_why:-TOOLKIT NOT RECORDED: setup.sh has not written resolved/versions.txt, so it is not known which nvcc built the probes}"
if (( collect_only )); then
  # The status each stage had when it last ran, from the history.
  for stage in "${STAGES[@]}"; do
    last="$(awk -F'\t' -v s="$stage" '$1 == s {status = $2} END {print status}' "$RUN_DIR/stages.tsv")"
    [[ -n "$last" ]] && result[$stage]="$last"
  done
  say "collect only: packing $RUN_DIR as it is"
  finish 0
fi
say "checkout $REPO_DIR at ${commit:0:12}; home $REMOTE_HOME; started $run_start"

# 1. Is this the right machine? Needs nothing installed.
run_stage preflight-quick 5 bash "$REMOTE_DIR/preflight.sh" --quick --out "$RUN_DIR/preflight-quick.txt"
rc=$?
[[ -f "$RUN_DIR/preflight-quick.txt" && "${result[preflight-quick]}" != "done" ]] && sed 's/^/    /' "$RUN_DIR/preflight-quick.txt"
if (( rc != 0 && ! keep_going )); then
  say "stopping: the quick preflight failed, so this is not a machine to spend time on."
  say "See 'When preflight fails' in env/remote/RUNBOOK.md. To go on regardless: --keep-going"
  finish 3
fi

# 2. The env.
if ! run_stage setup 45 bash "$REMOTE_DIR/setup.sh" "${setup_args[@]}"; then
  say "stopping: without the env nothing can be built. Fix the cause (log above) and run again."
  finish 3
fi
# What setup.sh found about driver, toolkit and the predictions' toolkit.
if [[ -f "$REMOTE_HOME/resolved/versions.txt" ]]; then
  predictions_state="$(sed -n 's/^predictions_comparable=//p' "$REMOTE_HOME/resolved/versions.txt")"
  predictions_why="$(sed -n 's/^predictions_comparable_why=//p' "$REMOTE_HOME/resolved/versions.txt")"
  predictions_state="${predictions_state:-unknown}"
  predictions_why="${predictions_why:-TOOLKIT NOT RECORDED: resolved/versions.txt does not say whether the installed nvcc is the one the predictions were made with}"
fi
case "$predictions_state" in
  comparable) say "toolkit: $predictions_why" ;;
  comparable-native) say "toolkit: $predictions_why" ;;
  *) bar="$(printf '!%.0s' {1..100})"
     printf '%s\nrun: %s\n%s\n' "$bar" "$predictions_why" "$bar"
     if (( strict_toolkit )); then
       say "stopping (--strict-toolkit). To build with the predictions' compiler: env/remote/run.sh --toolkit pinned --redo setup"
       finish 3
     fi ;;
esac

# The book: the clone setup.sh fetched, unless the caller named one (or said none, with an
# empty WARPLENS_BOOK). Exported for preflight, the self-test and the suites.
if [[ -z "${WARPLENS_BOOK+set}" ]] && is_book_clone "$BOOK_DIR"; then
  export WARPLENS_BOOK="$BOOK_DIR"
  say "the book: $WARPLENS_BOOK (fetched by setup.sh; WARPLENS_BOOK was not set)"
fi

# 3. Kernels on the GPU.
preflight_args=(--out "$RUN_DIR/preflight.txt" --status "$RUN_DIR/preflight.status")
(( ncu_required )) || preflight_args+=(--no-ncu)
# run_all.py omits the end-to-end programs for --no-e2e or any --only selector.
blackwell_words=()
read -r -a blackwell_words <<<"${WARPLENS_BLACKWELL_RUN_ARGS:-}"
with_e2e=1
next_book=0
for word in "${blackwell_words[@]}"; do
  if (( next_book )); then preflight_args+=(--book "$word"); next_book=0; continue; fi
  case "$word" in
    --no-e2e|--only|--only=*) with_e2e=0 ;;
    --book) next_book=1 ;;
    --book=*) preflight_args+=(--book "${word#*=}") ;;
  esac
done
if (( with_e2e )) && ! is_skipped run-blackwell; then preflight_args+=(--e2e); fi
run_stage preflight 15 bash "$REMOTE_DIR/preflight.sh" "${preflight_args[@]}"
rc=$?
[[ -f "$RUN_DIR/preflight.txt" && "${result[preflight]}" != "done" ]] && sed 's/^/    /' "$RUN_DIR/preflight.txt"
read_preflight_status
if (( rc != 0 && ! keep_going )); then
  say "stopping: the full preflight failed. See 'When preflight fails' in env/remote/RUNBOOK.md."
  if (( ncu_required )) && [[ "$ncu_state" != ok ]]; then
    say "No performance counter was read (check ncu: $ncu_check), and this run needs them for the timing suite."
    say "Grant the permission (the report above says how), or do without counters: env/remote/run.sh --no-ncu"
  fi
  say "To run the suites regardless: env/remote/run.sh --keep-going"
  finish 3
fi

# 4. The simulator on this machine, as installed.
run_stage selftest 30 "${self[@]}" selftest || true

# 5-8. Build both suites, then run what built.
# Arguments for the suite scripts. The defaults are the two things run.sh knows
# about them; WARPLENS_<SUITE>_<BUILD|RUN>_ARGS, when set, replaces a default.
declare -A stage_args=([build-blackwell]="" [build-timing]="" [run-blackwell]="" [run-timing]="--out $RUN_DIR/out/timing")
if [[ "$ptx_jit" == refused ]]; then
  say "the driver refuses this toolkit's PTX: building the Blackwell probes with machine code as well (--native)"
  stage_args[build-blackwell]="--native"
fi
[[ -n "${WARPLENS_BLACKWELL_BUILD_ARGS+set}" ]] && stage_args[build-blackwell]="$WARPLENS_BLACKWELL_BUILD_ARGS"
[[ -n "${WARPLENS_BLACKWELL_RUN_ARGS+set}" ]] && stage_args[run-blackwell]="$WARPLENS_BLACKWELL_RUN_ARGS"
[[ -n "${WARPLENS_TIMING_BUILD_ARGS+set}" ]] && stage_args[build-timing]="$WARPLENS_TIMING_BUILD_ARGS"
[[ -n "${WARPLENS_TIMING_RUN_ARGS+set}" ]] && stage_args[run-timing]="$WARPLENS_TIMING_RUN_ARGS"
export WARPLENS_PTX_JIT="$ptx_jit" WARPLENS_NCU="$ncu_state" WARPLENS_PREDICTIONS="$predictions_state"
if [[ -z "${WARPLENS_TORCH_PYTHON:-}" && -x "$TORCH_ENV/bin/python" ]]; then
  export WARPLENS_TORCH_PYTHON="$TORCH_ENV/bin/python"
fi
# gdb for the rows that trace a launch (run_rows.py); inside the env PATH may not have it.
if [[ -z "${WARPLENS_GDB:-}" && -n "$(gdb_program)" ]]; then
  WARPLENS_GDB="$(gdb_program)"; export WARPLENS_GDB
fi
suite_run_stage() {  # suite_run_stage STAGE MINUTES
  local words=()
  read -r -a words <<<"${stage_args[$1]}"
  run_stage "$1" "$2" "${self[@]}" "$1" "${words[@]}"
}
suite_run_stage build-blackwell "$build_minutes" || true
suite_run_stage build-timing "$build_minutes" || true
# A fallback toolkit: the committed predictions are another compiler's. Make them again
# with this one now, before the GPU has said anything (RUNBOOK.md, "Driver older than the toolkit").
if [[ "$predictions_state" != not-comparable ]]; then
  skip_stage predict-toolkit "not needed: $predictions_state (only a fallback toolkit needs its own predictions)"
else
  case "${result[build-blackwell]}" in
    ok|done) run_stage predict-toolkit 30 "${self[@]}" predict-toolkit || true ;;
    *) skip_stage predict-toolkit "build-blackwell did not succeed (${result[build-blackwell]})" ;;
  esac
fi
for suite in blackwell timing; do
  case "${result[build-$suite]}" in
    ok|done|skipped) suite_run_stage "run-$suite" "$run_minutes" || true ;;
    *) skip_stage "run-$suite" "build-$suite did not succeed (${result[build-$suite]})" ;;
  esac
done

if (( failures )); then
  say "$failures stage(s) failed; the archive has their logs."
  finish 1
fi
finish 0
