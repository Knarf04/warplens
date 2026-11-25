#!/usr/bin/env bash
# The whole rental in one command: check the machine, install the env, build
# and run both probe suites, pack one results archive.
#
#   env/remote/run.sh [--dry-run] [--keep-going] [--redo STAGE|all] [--skip STAGE] [--collect-only]
#                     [--toolkit auto|pinned|X.Y] [--no-ncu] [--no-torch] [--no-book] [--strict-toolkit]
#                     [--timing-on-unhealthy-gpu] [--price DOLLARS_PER_HOUR]
#
# Never asks anything. Safe to run again after an interruption or a failure:
# stages that finished are skipped, the rest run, and the archive is rebuilt.
# Ctrl-C stops the running stage and everything it started, writes the archive and
# exits with status 130. That holds when the output goes through a pipe whose reader
# Ctrl-C has killed (`run.sh | tee run.log`) or to a terminal that is gone: the run
# then goes on writing to logs/run.console.log in the run directory, which has every
# line of this script (not of the stages: those are logs/<stage>.log) in any case.
#
# Stages, in order, and what a failure does:
#   preflight-quick  GPU identity, driver, disk          stops the run (not a B200: nothing to do)
#   setup            env/remote/setup.sh                 stops the run (nothing can be built).
#                    Run again by itself when the last preflight failed for want of something
#                    it provides (the PyTorch env, the book, the CUTLASS clone, ncu, the env)
#   build-cutlass    examples/03_cutlass_mma/build.sh    continues; preflight then fails for it.
#                    --programs, from the clone setup.sh fetched: the four CUTLASS tutorials
#                    of the end-to-end step, so that nothing is fetched or compiled for them
#                    while the GPU stages run. Skipped when that step is left out
#   preflight        kernels on the GPU, tensor map, a   stops the run, before the probe suites
#                    counter read by ncu, PyTorch, book, are built or run on the GPU
#                    the CUTLASS clone and programs
#   selftest         the repository's tests, simulator   continues. HERE only with a fallback
#                    toolkit (or one that is not recorded), where its failures name the
#                    predictions that do not hold for the installed compiler. With the pinned
#                    toolkit (predictions_comparable=comparable) it runs AFTER run-timing, so
#                    that it cannot spend rented time before a measurement; the archive is
#                    packed once before it starts
#   build-blackwell  validation/blackwell/build.sh       its run stage is skipped; continues
#   build-timing     validation/blackwell_timing/build.sh  likewise
#   predict-toolkit  env/remote/predict_toolkit.sh       continues. Only with a fallback toolkit
#                    (setup.sh installed another nvcc than the committed predictions were
#                    made with): the predictions are made again with the installed nvcc,
#                    before anything runs on the GPU, into out/predicted-toolkit/ of the archive
#   run-blackwell    validation/blackwell/run_all.sh     continues. It fails, among other things,
#                    when the capture of matrix row RT-11 (a CuTe DSL kernel with TMA) is
#                    missing on a B200; the reason is then repeated as the run's last lines
#                    It also fails when the GPU stopped answering after a probe: the suite
#                    then launches nothing more and leaves GPU-UNHEALTHY.txt in its results;
#                    run-timing is NOT started (below), and the next run.sh takes the probes
#                    up where they stopped (run_all.py --no-build --resume), as it does after
#                    a run-blackwell that hit its limit or was interrupted
#   run-timing      validation/blackwell_timing/run_all.sh  continues. It fails, besides for
#                    a benchmark, when preflight read a counter and its own ncu profiles read
#                    none. Skipped while out/blackwell/GPU-UNHEALTHY.txt is there: timings on
#                    a GPU that does not answer are worthless, and each benchmark would wait
#                    out its limit. After the GPU is back, env/remote/run.sh again
#   collect          manifest, checksums, the kernel's   always runs
#                    log, archive
#
# Each stage runs in a session of its own. When it ends, at its time limit or at Ctrl-C,
# every process still in that session is stopped before the next stage or the archive:
# `timeout` inside a suite puts its child in a process group of its own, which a signal to
# the stage's group does not reach.
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
#   --timing-on-unhealthy-gpu   run the timing suite although the probe suite left
#                  GPU-UNHEALTHY.txt (you have reset the GPU and want the timings before, or
#                  without, the rest of the probes; with --skip run-blackwell for the latter)
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
#   NCU_BIN                     the ncu preflight found (it may be outside PATH), unless the
#                               caller set NCU_BIN; validation/blackwell_timing/run_ncu.sh uses it
#   WARPLENS_CUTLASS_NO_FETCH=1 examples/03_cutlass_mma/build.sh must not use the network
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
# What came of the counters is taken from the timing suite once it has run: the first
# line of <its results>/ncu/STATUS.txt (`ncu ok:`, `ncu partial:`, `ncu failed:`,
# `ncu not run:`) becomes ncu_counters= in manifest.txt and resolved/versions.txt, as
# read, partial or not-read. Before that, preflight's ncu check is all there is.
#
# Exit status: 0 everything ran and passed, 1 some stage failed or the source snapshot
# is incomplete (the archive is still written), 3 the run stopped early, 130 interrupted,
# 2 usage.
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
timing_on_unhealthy=0
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
    --timing-on-unhealthy-gpu) timing_on_unhealthy=1 ;;
    --price) price="${2:?--price needs a number}"; shift ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ {exit} NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ "$price" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "error: --price takes a number, not '$price'" >&2; exit 2; }

# The self-test's place: before the GPU stages unless the probes are built by the compiler
# the committed predictions were made with (setup.sh's verdict `comparable`). That is the
# compiler the suite was last run with at home, so there it tells nothing a measurement
# waits for: it last took 13.6 min, and nothing may spend that before the first probe runs.
order_stages() {  # order_stages PREDICTIONS_STATE: sets STAGES and selftest_last
  if [[ "$1" == comparable ]]; then
    selftest_last=1
    STAGES=(preflight-quick setup build-cutlass preflight build-blackwell build-timing predict-toolkit run-blackwell run-timing selftest collect)
  else
    selftest_last=0
    STAGES=(preflight-quick setup build-cutlass preflight selftest build-blackwell build-timing predict-toolkit run-blackwell run-timing collect)
  fi
}
recorded_predictions_state() {
  local state
  state="$(sed -n 's/^predictions_comparable=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
  echo "${state:-unknown}"
}
order_stages "$(recorded_predictions_state)"
for stage in "${redo[@]}" "${skip[@]}"; do
  [[ "$stage" == all || " ${STAGES[*]} " == *" $stage "* ]] \
    || { echo "error: no stage '$stage' (stages: ${STAGES[*]})" >&2; exit 2; }
done
is_skipped() { [[ " ${skip[*]} " == *" $1 "* && "$1" != collect ]]; }

# Counters are what the timing suite is held against: a run that includes it must read
# one for real (preflight's ncu check), unless it was told to do without (--no-ncu).
ncu_required=1
if (( no_ncu )) || is_skipped run-timing; then ncu_required=0; fi

# run_all.py omits the end-to-end programs for --no-e2e or any --only selector. With them
# the preflight also checks what they need, and the CUTLASS tutorials are built first.
blackwell_words=()
read -r -a blackwell_words <<<"${WARPLENS_BLACKWELL_RUN_ARGS:-}"
with_e2e=1
next_book=0
book_args=()
for word in "${blackwell_words[@]}"; do
  if (( next_book )); then book_args+=(--book "$word"); next_book=0; continue; fi
  case "$word" in
    --no-e2e|--only|--only=*) with_e2e=0 ;;
    --book) next_book=1 ;;
    --book=*) book_args+=(--book "${word#*=}") ;;
  esac
done
if is_skipped run-blackwell; then with_e2e=0; fi

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
    setup)           echo "3|12|measured: 75 s to download and install 5.8 GB on a fast link, 159 s for the PyTorch env (4.3 GB installed) and 6 s for the CUTLASS clone (244 MB) on the same link; a slow link or disk is the upper end" ;;
    build-cutlass)   echo "1|5|measured: 40 s on 20 cores for the four CUTLASS tutorials (PTX and program each), compiled at once (1.9 GB of memory each); one at a time, as with fewer than 4 CPUs, 2.5 min. 0 when the end-to-end programs are left out" ;;
    preflight)       echo "1|2|measured: 5 s without the Blackwell kernels; allows for a first-use driver compile and ncu start-up" ;;
    selftest)        echo "6|30|measured: 13.6 min on 20 cores for the whole test suite (3046 tests) with the pinned toolkit, 17 min with a fallback toolkit. It runs the whole predict path, which is what says whether the predictions hold for the installed compiler: so with a fallback toolkit it runs before the GPU stages, and with the pinned one after them, where it cannot delay a measurement. Limit 30 min; --skip selftest saves it" ;;
    build-blackwell) echo "1|6|measured: 8 s on 20 cores for the 51 probe programs (13 s with --native); a smaller machine is the upper end" ;;
    build-timing)    echo "1|4|measured: 62 s, one compile at a time, for the 14 benchmarks" ;;
    predict-toolkit) echo "0|15|measured: 8 min on 20 cores (65 s for the 108 probe predictions, 7 min for the 14 timing benchmarks, one at a time). Runs only when setup.sh fell back to another toolkit than the predictions' (0 otherwise)" ;;
    run-blackwell)   echo "3|20|guess: 126 probe runs of short kernels, the end-to-end programs and the RT-11 capture before the probes that may fault or hang, which wait out a timeout on purpose. Their own limits add up to far more (900 s per end-to-end step, 3000 s for RT-11), so the stage limit of $run_minutes min is what bounds it; what it cuts off, the next run.sh resumes" ;;
    run-timing)      echo "3|20|guess: 14 microbenchmarks, the 7 that ran on Ampere first, ncu after each group. Bounded by the suite: 180 s per benchmark, 300 s per ncu profile, 1200 s for all ncu work; by that arithmetic 45 min if every Blackwell-only benchmark hung and ncu used its whole budget, inside the limit of $run_minutes min" ;;
    collect)         echo "1|1|measured: under 1 s for a small archive; with the pinned toolkit it runs twice, before the self-test and at the end" ;;
  esac
}

stage_command() {
  case "$1" in
    preflight-quick) echo "env/remote/preflight.sh --quick" ;;
    setup)           echo "env/remote/setup.sh ${setup_args[*]:-}" ;;
    build-cutlass)   echo "examples/03_cutlass_mma/build.sh --programs   (no network: the clone is setup.sh's)" ;;
    preflight)       echo "env/remote/preflight.sh$( (( ncu_required )) || echo " --no-ncu")   (ncu must read a counter, unless --no-ncu or --skip run-timing)" ;;
    selftest)        echo "python -m pytest -q   (in the env, CUDA_VISIBLE_DEVICES empty; $( (( selftest_last )) && echo "after the GPU stages: pinned toolkit" || echo "before the GPU stages: the toolkit is a fallback or not recorded yet"))" ;;
    build-blackwell) echo "$SUITE_blackwell/build.sh   (adds --native if the driver refuses the toolkit's PTX)" ;;
    build-timing)    echo "$SUITE_timing/build.sh" ;;
    predict-toolkit) echo "env/remote/predict_toolkit.sh --out <run directory>/out/predicted-toolkit   (fallback toolkit only)" ;;
    run-blackwell)   echo "$SUITE_blackwell/run_all.sh" ;;
    run-timing)      echo "$SUITE_timing/run_all.sh --out <results directory>" ;;
    collect)         echo "manifest, SHA256SUMS, kernel log, tar.gz under $REMOTE_HOME/results" ;;
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
    case "$stage" in
      build-cutlass)
        if (( ! with_e2e )) || [[ -n "${CUTLASS_BIN:-}" ]]; then state=skip; low=0; high=0
        elif [[ ! -f "$CUTLASS_EXAMPLE/build.sh" ]]; then state=MISSING; missing=1; fi ;;
      build-*|run-*)
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
  if (( selftest_last )); then
    echo "  The self-test is after the GPU stages: setup.sh installed the predictions' own compiler."
  else
    echo "  The self-test is before the GPU stages: the toolkit is a fallback, or setup.sh has not recorded it yet"
    echo "  ($(recorded_predictions_state)). Once setup.sh has installed the pinned toolkit it moves after run-timing."
  fi
  if (( missing )); then
    echo
    echo "  MISSING: a probe suite's script (or examples/03_cutlass_mma/build.sh) is not in this checkout."
    echo "  Those stages will fail with that message and the rest will run. Check out a commit that has them."
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
# Every line this script says goes to the terminal and to logs/run.console.log of the run
# directory (once that exists). When the terminal is gone, or the pipe's reader (Ctrl-C
# kills `tee` in `run.sh | tee run.log`), the write fails; SIGPIPE is ignored below so that
# it does no more than fail. From then on stdout and stderr are that file.
console_log=""
stdout_dead=0
say() {
  local line="run: $*"
  if (( stdout_dead )); then echo "$line"; return 0; fi
  [[ -n "$console_log" ]] && echo "$line" >> "$console_log"
  echo "$line" 2>/dev/null || stdout_gone
  return 0
}
stdout_gone() {
  stdout_dead=1
  if [[ -n "$console_log" ]]; then exec >> "$console_log" 2>&1; else exec > /dev/null 2>&1; fi
  echo "run: [$(utc_now): the terminal or pipe this run was writing to is gone; its output continues here only]"
}
declare -A result=()
failures=0

# A stage runs in a session of its own (setsid), so that everything it starts can be found
# again: `timeout` puts its child into a new process group, and the timing suite runs each
# benchmark under one, so the stage's own group is not all of it. Without setsid (not seen
# on any image, but not required either) only the stage's own process is signalled, as before.
session_cmd=()
if setsid --wait true 2>/dev/null; then session_cmd=(setsid --wait)
elif setsid true 2>/dev/null; then session_cmd=(setsid)
fi
own_session="$(IFS= read -r line < /proc/$$/stat && rest="${line##*) }" && read -r _ _ _ sid _ <<<"$rest" && echo "$sid")"
stage_session() {  # the session of the running stage, once it has said which; else nothing
  local sid=""
  (( ${#session_cmd[@]} )) || return 0
  [[ -s "$RUN_DIR/state/stage.sid" ]] && sid="$(head -n 1 "$RUN_DIR/state/stage.sid")"
  [[ "$sid" =~ ^[0-9]+$ && "$sid" != "$own_session" ]] && echo "$sid"
  return 0
}
signal_stage() {  # signal_stage SIGNAL: to the stage's own process and to all of its session
  local signal="$1" sid pid
  [[ -n "$stage_pid" ]] && kill "-$signal" "$stage_pid" 2>/dev/null
  sid="$(stage_session)"
  [[ -n "$sid" ]] || return 0
  for pid in $(session_pids "$sid"); do kill "-$signal" "$pid" 2>/dev/null; done
  return 0
}
# After the stage's own process has ended: nothing of it may still run, on the GPU least of
# all, when the next stage starts or the archive is packed. TERM, 10 s, KILL, 5 s.
reap_stage() {  # reap_stage STAGE
  local sid left tries
  sid="$(stage_session)"
  [[ -n "$sid" ]] || return 0
  left="$(session_pids "$sid")"
  [[ -n "$left" ]] || return 0
  say "$1: $(wc -w <<<"$left") process(es) it started were still running after it ended; stopping them"
  signal_stage TERM
  for (( tries = 0; tries < 100; tries++ )); do
    left="$(session_pids "$sid")"
    [[ -n "$left" ]] || return 0
    sleep 0.1
  done
  signal_stage KILL
  for (( tries = 0; tries < 50; tries++ )); do
    left="$(session_pids "$sid")"
    [[ -n "$left" ]] || return 0
    sleep 0.1
  done
  say "$1: could not stop: pid(s) $(tr '\n' ' ' <<<"$left"). They may still hold the GPU"
}

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
    setup) rm -f "$RUN_DIR/state/"{build-cutlass,preflight,selftest,build-blackwell,build-timing,predict-toolkit,run-blackwell,run-timing}.ok ;;
    build-cutlass) rm -f "$RUN_DIR/state/run-blackwell.ok" ;;
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
  : > "$RUN_DIR/state/stage.sid"
  # In the background, so that Ctrl-C reaches the trap while the stage runs. The stage gets
  # SIGPIPE back as it was: this script ignores it, a suite's own pipelines must not.
  (
    trap - PIPE
    # shellcheck disable=SC2016  # the inner shell expands these
    exec "${session_cmd[@]}" bash -c 'echo "$$" > "$1"; shift; exec "$@"' _ "$RUN_DIR/state/stage.sid" \
      timeout --kill-after=30 "${minutes}m" "$@"
  ) > "$RUN_DIR/logs/$stage.log" 2>&1 &
  stage_pid=$!
  wait "$stage_pid"
  rc=$?
  if (( interrupted )); then      # the trap has told the stage to stop; let it
    wait "$stage_pid" 2>/dev/null
    rc=130
  fi
  reap_stage "$stage"
  stage_pid=""
  : > "$RUN_DIR/state/stage.sid"
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
  # WARPLENS_CUTLASS_NO_FETCH: the end-to-end step calls examples/03_cutlass_mma/build.sh when
  # it misses a program, and that must not clone CUTLASS while the GPU stage runs.
  env_run env NVCC=nvcc PYTHON=python WARPLENS_RESULTS_DIR="$out" \
      WARPLENS_PTX_JIT="${WARPLENS_PTX_JIT:-unknown}" WARPLENS_NCU="${WARPLENS_NCU:-unavailable}" \
      WARPLENS_PREDICTIONS="${WARPLENS_PREDICTIONS:-unknown}" WARPLENS_CUTLASS_NO_FETCH=1 \
      ${WARPLENS_TORCH_PYTHON:+WARPLENS_TORCH_PYTHON="$WARPLENS_TORCH_PYTHON"} \
      ${WARPLENS_GDB:+WARPLENS_GDB="$WARPLENS_GDB"} \
      bash "$script" "$@"
}

# Stages that are not one script run through here, in a process of their own
# so that `timeout` can end them: run.sh --stage-body STAGE [args...]
stage_body() {
  local stage="$1" missing; shift
  case "$stage" in
    selftest)
      cd "$REPO_DIR" || return 1
      env_run env CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider ;;
    build-cutlass)
      if [[ ! -f "$CUTLASS_EXAMPLE/build.sh" ]]; then
        echo "error: $CUTLASS_EXAMPLE/build.sh not found: this checkout (commit ${commit:0:12}) lacks the CUTLASS example." >&2
        return 66
      fi
      cd "$REPO_DIR" || return 1
      env_run env NVCC=nvcc WARPLENS_CUTLASS_NO_FETCH=1 bash "$CUTLASS_EXAMPLE/build.sh" --programs || return
      missing="$(cutlass_programs_missing)"
      if [[ -n "$missing" ]]; then
        echo "error: build.sh ended without error, but these programs are not in $(cutlass_bin_dir): $missing" >&2
        return 1
      fi ;;
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
# Nor a reader that went away: `run.sh 2>&1 | tee run.log` and Ctrl-C, which kills tee before
# this script has packed anything. The next line written would be the script's last (SIGPIPE).
# Ignored, the write only fails, and `say` then takes the output to a file.
trap '' PIPE
# Ctrl-C or a kill: stop the running stage and what it started, still write the archive, exit 130.
stage_pid=""
interrupted=0
# shellcheck disable=SC2329  # called by the trap
on_interrupt() {
  interrupted=1
  signal_stage TERM
}
trap on_interrupt INT TERM
mkdir -p "$RUN_DIR"/{state,logs,out,kernel} "$REMOTE_HOME/results"
console_log="$RUN_DIR/logs/run.console.log"
: > "$RUN_DIR/state/stage.sid"
[[ -f "$RUN_DIR/state/first-start" ]] || utc_now > "$RUN_DIR/state/first-start"
run_start="$(utc_now)"
echo "$run_start $commit" >> "$RUN_DIR/state/attempts"
echo "run: [$run_start: env/remote/run.sh$( (( collect_only )) && echo " --collect-only") started, attempt $(wc -l < "$RUN_DIR/state/attempts")]" >> "$console_log"
[[ -f "$RUN_DIR/stages.tsv" ]] || printf 'stage\tstatus\texit\tstart\tend\tseconds\n' > "$RUN_DIR/stages.tsv"
# Whether counters were required is the run's decision, not that of a later --collect-only.
if (( collect_only )) && [[ -f "$RUN_DIR/state/ncu-policy" ]]; then
  read -r no_ncu ncu_required < "$RUN_DIR/state/ncu-policy"
elif (( ! collect_only )); then
  echo "$no_ncu $ncu_required" > "$RUN_DIR/state/ncu-policy"
fi

# The tracked source as it is, into the archive: with the env's Python when there is one
# (a bare image has no python3; and inside the env `git` is also the one setup.sh installed
# when the machine had none), else the machine's. freeze_source.py names the checkout safe
# for git itself (a checkout that belongs to another user).
snapshot_source() {
  local python status=1
  : > "$RUN_DIR/source-snapshot.log"
  for python in env python3 python; do
    if [[ "$python" == env ]]; then
      env_installed || continue
      echo "# with the env's python" >> "$RUN_DIR/source-snapshot.log"
      env_run python "$REMOTE_DIR/freeze_source.py" --repo "$REPO_DIR" --out "$RUN_DIR/source" \
        >> "$RUN_DIR/source-snapshot.log" 2>&1 && return 0
    else
      command -v "$python" >/dev/null 2>&1 || continue
      echo "# with the machine's $python" >> "$RUN_DIR/source-snapshot.log"
      "$python" "$REMOTE_DIR/freeze_source.py" --repo "$REPO_DIR" --out "$RUN_DIR/source" \
        >> "$RUN_DIR/source-snapshot.log" 2>&1 && return 0
    fi
  done
  return "$status"
}

# The kernel's log, where the driver reports a GPU fault (Xid) or its own errors (NVRM), and
# which of those lines are new since the baseline taken before this run used the GPU.
# kernel/summary.txt says what could be read; kernel_log= in the manifest repeats its verdict.
kernel_log="not looked at"
collect_kernel_log() {
  local dir="$RUN_DIR/kernel" all new
  rm -f "$dir/dmesg-xid-nvrm.txt" "$dir/dmesg-xid-nvrm-new.txt"
  if ! save_dmesg "$dir/dmesg.txt"; then
    kernel_log="$(head -n 1 "$dir/dmesg.txt")"
  else
    grep -E 'Xid|NVRM' "$dir/dmesg.txt" > "$dir/dmesg-xid-nvrm.txt"
    all="$(wc -l < "$dir/dmesg-xid-nvrm.txt")"
    if [[ -f "$dir/dmesg-before-gpu.txt" ]] && ! grep -q '^not readable' "$dir/dmesg-before-gpu.txt"; then
      grep -vxFf "$dir/dmesg-before-gpu.txt" "$dir/dmesg-xid-nvrm.txt" > "$dir/dmesg-xid-nvrm-new.txt"
      new="$(wc -l < "$dir/dmesg-xid-nvrm-new.txt")"
      kernel_log="readable: $all line(s) with Xid or NVRM, $new of them not in the baseline from before the GPU was used"
    else
      kernel_log="readable: $all line(s) with Xid or NVRM; no readable baseline from before the GPU was used, so which are new is not known"
    fi
  fi
  {
    echo "kernel_log=$kernel_log"
    echo "dmesg.txt                 the kernel log when the archive was packed ($(utc_now)), or why it could not be read"
    echo "dmesg-xid-nvrm.txt        its lines with Xid (a GPU fault, as the driver reports it) or NVRM (the driver's own messages)"
    echo "dmesg-before-gpu.txt      the kernel log before this run's first full preflight: $(cat "$dir/dmesg-before-gpu.when" 2>/dev/null || echo "not taken (the run did not get that far, or was packed with --collect-only only)")"
    echo "dmesg-xid-nvrm-new.txt    the lines of dmesg-xid-nvrm.txt that are not in dmesg-before-gpu.txt"
    echo "A file that starts with 'not readable' says why: a container, or kernel.dmesg_restrict for a user who is not root."
  } > "$dir/summary.txt"
}
# Before the full preflight, the first thing of a run that puts a kernel on the GPU. Kept
# from the first attempt on, unless that one could not read the log and this one can.
kernel_baseline() {
  local dir="$RUN_DIR/kernel"
  if [[ -f "$dir/dmesg-before-gpu.txt" ]] && ! grep -q '^not readable' "$dir/dmesg-before-gpu.txt"; then return 0; fi
  save_dmesg "$dir/dmesg-before-gpu.txt"
  echo "taken $(utc_now), attempt $(wc -l < "$RUN_DIR/state/attempts")" > "$dir/dmesg-before-gpu.when"
  return 0
}

UNHEALTHY="$RUN_DIR/out/blackwell/GPU-UNHEALTHY.txt"
source_failed=0
archive_warning=""
collect() {
  local archive sums name first tar_status counters changed
  source_failed=0
  archive_warning=""
  counters="$(ncu_counters)"
  collect_kernel_log
  if changed="$(repo_git "$REPO_DIR" status --porcelain 2>/dev/null)"; then
    changed="$(grep -c . <<<"$changed")"
  else
    changed="unknown (git status did not run here)"
  fi
  {
    echo "# warplens remote run manifest, written $(utc_now)"
    echo "warplens_commit=$commit"
    echo "warplens_branch=$(repo_git "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "warplens_uncommitted_files=$changed"
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
    echo "ncu_counters=$counters"
    echo "ncu_bin=${NCU_BIN:-not set}"
    echo "kernel_log=$kernel_log"
    echo "gpu_unhealthy=$( [[ -f "$UNHEALTHY" ]] && head -n 1 "$UNHEALTHY" || echo no)"
    echo "book=$(book_in_use)"
    echo "book_commit=$(is_book_clone "$(book_in_use)" && book_commit "$(book_in_use)" || echo none)"
    echo "cutlass_commit=$(is_cutlass_clone "$CUTLASS_DIR" && git_head "$CUTLASS_DIR" || echo none)"
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
    echo "# this script's own lines, every attempt: logs/run.console.log. The kernel's log: kernel/summary.txt"
  } > "$RUN_DIR/manifest.txt" || return 1
  rm -rf "$RUN_DIR/resolved" "$RUN_DIR/suite-dirs" || return 1
  if [[ -d "$REMOTE_HOME/resolved" ]]; then
    cp -r "$REMOTE_HOME/resolved" "$RUN_DIR/resolved" || return 1
    # versions.txt is setup.sh's, written before anything looked at the GPU. The archive's
    # copy also says what came of the counters, next to the ncu that was installed.
    if [[ -f "$RUN_DIR/resolved/versions.txt" ]]; then
      echo "ncu_counters=$counters" >> "$RUN_DIR/resolved/versions.txt" || return 1
    fi
  fi
  cp "$PINS_FILE" "$RUN_DIR/pins.env" || return 1
  # Preserve the implementation itself: a commit SHA alone is not an offline oracle.
  # A snapshot that fails is its own warning (finish says it); the archive is still packed.
  rm -f "$RUN_DIR/SOURCE-SNAPSHOT-FAILED.txt"
  if ! snapshot_source; then
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
  if [[ "$counters" == not-read* ]]; then
    printf 'No GPU performance counter was read in this run: ncu_counters=%s.\n\nThe evidence is the timing suite'"'"'s own word once it has run (the first line of\nout/timing/ncu/STATUS.txt: ncu ok, ncu partial, ncu failed or ncu not run), and\nbefore that the preflight check "ncu" (preflight.txt; it PASSes only when ncu printed a\ncounter value for a kernel that ran). Without counters the timing suite has launch times\nbut nothing to hold the model'"'"'s counters against: read out/timing/ with that in mind, and see\nenv/remote/RUNBOOK.md, "When preflight fails", for what grants the permission.\n' \
      "$counters" > "$RUN_DIR/NCU-COUNTERS-NOT-READ.txt" || return 1
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
  # From here to the end of tar nothing may be said: logs/run.console.log is in the sums.
  (cd "$RUN_DIR" && find . -type f ! -name SHA256SUMS ! -path './state/*' -print0 | sort -z | xargs -0 sha256sum > state/sums \
     && mv state/sums SHA256SUMS) || return 1

  first="$(tr -d ':-' < "$RUN_DIR/state/first-start")"
  name="warplens-b200-${first}-${commit:0:10}"
  archive="$REMOTE_HOME/results/$name.tar.gz"
  # GNU tar exits 1 when a file changed while it was being read ("file changed as we read
  # it"): the archive is whole, with whatever that file held at the time. Something is then
  # still writing into the run directory; once more after 2 s, and if it is the same, the
  # archive stands and the run says so. Any other failure is a failed archive.
  pack() {
    tar -czf "$archive.part" -C "$REMOTE_HOME" --transform "s|^run|$name|" --exclude='run/state' run \
      2> "$RUN_DIR/state/tar.err"
  }
  pack; tar_status=$?
  if (( tar_status == 1 )); then sleep 2; pack; tar_status=$?; fi
  if (( tar_status == 1 )) && [[ -s "$archive.part" ]]; then
    archive_warning="tar: $(head -n 3 "$RUN_DIR/state/tar.err" | tr '\n' ' ')"
    tar_status=0
  fi
  if (( tar_status != 0 )) || ! mv -f "$archive.part" "$archive"; then
    say "collect: writing $archive failed (tar exit status $tar_status)"
    sed 's/^/    | /' "$RUN_DIR/state/tar.err" 2>/dev/null | head -n 10
    return 1
  fi
  sums="$archive.sha256"
  (cd "$REMOTE_HOME/results" && sha256sum "$name.tar.gz" > "$sums.part" && mv "$sums.part" "$sums") || return 1
  say "archive:  $archive"
  say "size:     $(du -h "$archive" | cut -f1) ($(stat -c %s "$archive") bytes)"
  say "sha256:   $(cut -d' ' -f1 "$sums")"
  if [[ -n "$archive_warning" ]]; then
    say "collect: WARNING: a file in $RUN_DIR changed while it was packed, twice. The archive is written and"
    say "  holds that file as it was read; SHA256SUMS inside may not match it. Is something still running?"
    say "  $archive_warning"
    say "  Pack again once nothing writes there: env/remote/run.sh --collect-only"
  fi
  say "fetch it from your own machine with:"
  say "  scp '<user>@<host>:$archive*' .  &&  sha256sum -c $name.tar.gz.sha256"
  if [[ "$archive" == /root/* ]]; then
    say "  It is under /root: only a login as root can copy it from there. For another login, first, here:"
    say "    cp '$archive'* /tmp/ && chmod a+r /tmp/$name.tar.gz*      then: scp '<user>@<host>:/tmp/$name.tar.gz*' ."
  fi
  return 0
}

finish() {  # finish EXIT_STATUS
  local code="$1" counters
  if ! collect; then
    say "collect: archive or checksum failed; keep the raw run directory and retry --collect-only"
    (( code == 0 )) && code=1
  elif (( source_failed )); then
    # Not an archive failure: the archive and its checksum are written and hold everything else.
    say "source snapshot: FAILED. The archive above is written and has the results; what it lacks is the"
    say "  frozen copy of the checkout (source/). Why: source-snapshot.log in the archive, and here:"
    tail -n 3 "$RUN_DIR/source-snapshot.log" 2>/dev/null | sed 's/^/    | /'
    say "  Keep the checkout itself (commit ${commit:0:12}), or fix the cause and: env/remote/run.sh --collect-only"
    (( code == 0 )) && code=1
  fi
  say "stages: $(for s in "${STAGES[@]}"; do [[ $s == collect ]] || printf '%s=%s ' "$s" "${result[$s]:-not-reached}"; done)"
  # Written by validation/blackwell/run_rows.py when the capture of matrix row RT-11 was
  # required (a B200) and is not there; gone again once the row has been run to `ok`.
  if [[ -f "$RUN_DIR/out/blackwell/RT-11-NOT-CAPTURED.txt" ]]; then
    local missing; missing="$(printf '!%.0s' {1..100})"
    printf '%s\n' "$missing"; cat "$RUN_DIR/out/blackwell/RT-11-NOT-CAPTURED.txt"; printf '%s\n' "$missing"
    (( code == 0 )) && code=1
  fi
  # Written by validation/blackwell/run_all.py when its canary kernel no longer ran after a
  # probe and did not come back; removed by its next run once the canary passes again.
  if [[ -f "$UNHEALTHY" ]]; then
    local sick; sick="$(printf '!%.0s' {1..100})"
    printf '%s\n' "$sick"; cat "$UNHEALTHY"
    printf 'run: the timing suite %s. After the GPU is back (a reset, a reboot; a new instance starts over):\nrun:   env/remote/run.sh        resumes the probes where they stopped, then runs the timing suite\nrun:   env/remote/run.sh --timing-on-unhealthy-gpu --skip run-blackwell     the timing suite alone\n%s\n' \
      "$( [[ "${result[run-timing]:-}" == skipped ]] && echo "was NOT started" || echo "status: ${result[run-timing]:-not-reached}")" "$sick"
    (( code == 0 )) && code=1
  fi
  case "$kernel_log" in
    readable:*", 0 of them not in the baseline"*|"readable: 0 line"*) ;;
    readable:*) say "kernel log: $kernel_log (kernel/ in the archive)" ;;
  esac
  counters="$(ncu_counters)"
  [[ "$counters" == "read "* ]] || say "performance counters: $counters"
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
# Starts with `read`, `partial` or `not-read`.
#
# Once the timing suite has run (in this invocation or an earlier one), it is the suite that
# says: the first line of ncu/STATUS.txt in its results, which it starts with `ncu ok:`
# (every profile read counters), `ncu partial:` (some did), `ncu failed:` (attempted, none
# did) or `ncu not run:`. A suite that ran and left no such file has shown no counter.
# Preflight's check alone speaks only for a run in which the timing suite did not run: it
# read one counter of one trivial kernel, which is not the suite's profiles.
ncu_counters() {
  local policy line="" file="$RUN_DIR/out/timing/ncu/STATUS.txt"
  if (( no_ncu )); then policy="the run was told to do without, --no-ncu"
  elif (( ! ncu_required )); then policy="not needed, the timing suite was left out"
  else policy="this run REQUIRED them"
  fi
  case "${result[run-timing]:-}" in
    ok|"done"|failed|timeout|interrupted)
      if [[ ! -f "$file" ]]; then
        echo "not-read (the timing suite ran (${result[run-timing]}) and left no ncu/STATUS.txt, so no profile of it is known to have read a counter; preflight's ncu check: $ncu_check; $policy)"
        return 0
      fi
      IFS= read -r line < "$file" || true
      case "$line" in
        "ncu ok:"*) echo "read (the timing suite: \"$line\")" ;;
        "ncu partial:"*) echo "partial (the timing suite: \"$line\"; $policy)" ;;
        "ncu failed:"*|"ncu not run:"*) echo "not-read (the timing suite: \"$line\"; preflight's ncu check: $ncu_check; $policy)" ;;
        *) echo "not-read (the timing suite's ncu/STATUS.txt starts with no status this script knows: \"$line\"; preflight's ncu check: $ncu_check; $policy)" ;;
      esac
      return 0 ;;
  esac
  if [[ "$ncu_state" == ok ]]; then echo "read (preflight's ncu check passed; the timing suite, whose profiles are what counts, did not run)"
  else echo "not-read (preflight's ncu check: $ncu_check; $policy)"
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

# 2. The env. setup.sh ends well when the PyTorch env, the book, the CUTLASS clone or ncu
# only gave a warning, and the stage is then done; the preflight that follows is what fails
# for them. A plain rerun would skip setup and fail the same way, so: when the last preflight
# failed a check for something setup.sh provides, setup runs again (once per invocation;
# setup.sh retries exactly those parts and only verifies the rest).
setup_owes() {  # the checks of the last preflight that failed for want of what setup.sh provides
  local owed=""
  [[ -f "$RUN_DIR/preflight.status" ]] || return 0
  owed="$(awk '$2 == "FAIL" && ($1 == "env" || $1 == "native-e2e" || $1 == "book" || $1 == "cutlass") {printf "%s ", $1}' "$RUN_DIR/preflight.status")"
  # ncu: only when none was found. One that was found and refused is not setup's to fix.
  if (( ncu_required )) && [[ "$ncu_check" == FAIL && -f "$RUN_DIR/preflight.ncu" && -z "$(head -n 1 "$RUN_DIR/preflight.ncu")" ]]; then
    owed+="ncu "
  fi
  echo "${owed% }"
}
owed="$(setup_owes)"
if [[ -n "$owed" && -f "$RUN_DIR/state/setup.ok" ]] && ! is_skipped setup && ! is_skipped preflight; then
  say "setup: running again: the last preflight failed for want of something setup.sh provides ($owed)"
  rm -f "$RUN_DIR/state/setup.ok"
fi
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
order_stages "$predictions_state"
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

# 3. The CUTLASS tutorials of the end-to-end step, from the clone setup.sh fetched: built
# now, so that the preflight can check them and the GPU stage finds them (run_e2e.py builds
# them itself otherwise, which used to mean cloning CUTLASS inside that stage).
if (( ! with_e2e )); then
  skip_stage build-cutlass "not needed: the end-to-end programs are left out (--no-e2e, --only or --skip run-blackwell)"
elif is_skipped build-cutlass; then
  skip_stage build-cutlass "asked to (--skip build-cutlass)"
elif [[ -n "${CUTLASS_BIN:-}" ]]; then
  skip_stage build-cutlass "CUTLASS_BIN names the programs ($CUTLASS_BIN); they are not built here"
elif ! is_cutlass_clone "$CUTLASS_DIR"; then
  skip_stage build-cutlass "there is no clone of CUTLASS in $CUTLASS_DIR to build from (setup.sh fetches it; its log says why not)"
else
  # A stage that finished, but whose programs are gone, did not finish.
  [[ -z "$(cutlass_programs_missing)" ]] || rm -f "$RUN_DIR/state/build-cutlass.ok"
  run_stage build-cutlass "$build_minutes" "${self[@]}" build-cutlass || true
fi

# 4. Kernels on the GPU. The kernel's log first, as it is before this run has touched the GPU.
kernel_baseline
preflight_args=(--out "$RUN_DIR/preflight.txt" --status "$RUN_DIR/preflight.status" --ncu-path "$RUN_DIR/preflight.ncu")
(( ncu_required )) || preflight_args+=(--no-ncu)
preflight_args+=("${book_args[@]}")
(( with_e2e )) && preflight_args+=(--e2e)
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
  owed="$(setup_owes)"
  [[ -z "$owed" ]] || say "Failed for want of something setup.sh provides ($owed): env/remote/run.sh again runs setup again by itself."
  say "To run the suites regardless: env/remote/run.sh --keep-going"
  finish 3
fi
# The ncu preflight found may be outside PATH (/usr/local/cuda/bin, /opt/nvidia/nsight-compute);
# the timing suite's run_ncu.sh takes NCU_BIN. One the caller named is left alone.
if [[ -z "${NCU_BIN:-}" && -s "$RUN_DIR/preflight.ncu" ]]; then
  found_ncu="$(head -n 1 "$RUN_DIR/preflight.ncu")"
  if [[ -x "$found_ncu" ]]; then
    export NCU_BIN="$found_ncu"
    say "ncu: $NCU_BIN (the one preflight found; the timing suite gets it as NCU_BIN)"
  fi
fi

# 5. The simulator on this machine, as installed. Here only when the compiler is not the
# predictions' own (order_stages); otherwise after the GPU stages, below.
selftest_stage() { run_stage selftest 30 "${self[@]}" selftest || true; }
if (( selftest_last )); then
  say "selftest: after the GPU stages (the installed nvcc is the one the committed predictions were made with)"
else
  selftest_stage
fi

# 6-9. Build both suites, then run what built.
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
# The probe suite, taken up where an earlier attempt stopped when that attempt was cut short:
# by its limit, by Ctrl-C, or by the suite itself because the GPU no longer answered
# (GPU-UNHEALTHY.txt). run_all.py --resume keeps every run that has a status and runs the
# rest. Not when the programs were built again in this invocation, when the stage was asked
# for again (--redo), or when the caller's own arguments select runs or say --resume.
blackwell_resumes() {
  local last r
  [[ "${result[build-blackwell]}" == "done" && -f "$RUN_DIR/out/blackwell/status.tsv" ]] || return 1
  for r in "${redo[@]}"; do [[ "$r" == all || "$r" == run-blackwell ]] && return 1; done
  [[ " ${stage_args[run-blackwell]} " != *" --resume "* && " ${stage_args[run-blackwell]} " != *" --only"* ]] || return 1
  last="$(awk -F'\t' '$1 == "run-blackwell" {status = $2} END {print status}' "$RUN_DIR/stages.tsv")"
  [[ -f "$UNHEALTHY" || "$last" == timeout || "$last" == interrupted ]]
}
for suite in blackwell timing; do
  if [[ "$suite" == timing && -f "$UNHEALTHY" ]] && (( ! timing_on_unhealthy )) && ! is_skipped run-timing; then
    skip_stage run-timing "the probe suite found the GPU unhealthy ($(head -n 1 "$UNHEALTHY")): timings would be worthless and every benchmark would wait out its limit. After the GPU is back: env/remote/run.sh again; or --timing-on-unhealthy-gpu"
    continue
  fi
  case "${result[build-$suite]}" in
    ok|done|skipped)
      if [[ "$suite" == blackwell ]] && ! is_skipped run-blackwell && ! is_done run-blackwell && blackwell_resumes; then
        say "run-blackwell: taking up the earlier attempt where it stopped (--no-build --resume); --redo run-blackwell starts over"
        stage_args[run-blackwell]="${stage_args[run-blackwell]} --no-build --resume"
      fi
      suite_run_stage "run-$suite" "$run_minutes" || true ;;
    *) skip_stage "run-$suite" "build-$suite did not succeed (${result[build-$suite]})" ;;
  esac
done
# The timing suite fails, with every benchmark fine, when preflight read a counter and its
# own ncu profiles read none (its run_all.sh; status.txt: COUNTERS REQUIRED AND NOT READ).
if [[ "${result[run-timing]:-}" == failed ]] && grep -qs 'COUNTERS REQUIRED AND NOT READ' "$RUN_DIR/out/timing/status.txt"; then
  say "run-timing: failed at least for the counters: preflight read one, and the suite's ncu profiles read none:"
  say "  $(head -n 1 "$RUN_DIR/out/timing/ncu/STATUS.txt" 2>/dev/null || echo "no ncu/STATUS.txt")   (out/timing/ncu.log, out/timing/ncu/)"
fi

# 10. The self-test, when it was put off. What the GPU stages measured is packed first: the
# self-test takes a quarter of an hour, and a rental that ends during it has lost nothing.
if (( selftest_last )); then
  if ! is_skipped selftest && ! is_done selftest; then
    say "the GPU stages are over. Packing what they left before the self-test starts; Ctrl-C from here on loses no measurement."
    collect || say "collect: this first archive failed; the one at the end is tried all the same"
  fi
  selftest_stage
fi

if (( failures )); then
  say "$failures stage(s) failed; the archive has their logs."
  finish 1
fi
finish 0
