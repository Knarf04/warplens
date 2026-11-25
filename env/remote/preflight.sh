#!/usr/bin/env bash
# Is this machine worth the rental? Run before anything else.
#
#   env/remote/preflight.sh [--quick] [--e2e] [--book PATH] [--no-ncu] [--out FILE] [--status FILE]
#                           [--ncu-path FILE]
#
# Prints one line per check, PASS, WARN, FAIL, INFO or SKIP, then a summary.
# Exit status 1 if any check FAILed, 0 otherwise.
#
#   --quick        only the checks that need nothing installed (GPU identity,
#                  driver, memory, other users of the GPU, disk, CPU, clock,
#                  privileges). A few seconds. Run this first.
#   (default)      also compile and run kernels on the GPU, encode a tensor map
#                  and read a performance counter with ncu. Needs the env from
#                  env/remote/setup.sh. About a minute. The counter must really
#                  be read: no ncu, no permission (ERR_NVGPUCTRPERM) or no value
#                  is a FAIL, because the timing suite is held against counters
#   --no-ncu       counters are not needed: whatever keeps ncu from reading one
#                  is a WARN. The report still says that none was read
#   --e2e          also require native Triton (CUDA PyTorch), a clone of the
#                  book and one of CUTLASS at its pinned commit with the four
#                  tutorial programs built from it, and check which commit of
#                  the book it is. Nothing is then left to fetch or compile
#                  for the end-to-end programs while the GPU stages run
#   --book PATH    book clone for --e2e (default: WARPLENS_BOOK, else the clone
#                  setup.sh made in $WARPLENS_REMOTE_HOME/book)
#   --out FILE     also write the report to FILE
#   --status FILE  write "check STATUS" lines to FILE, for run.sh
#   --ncu-path FILE  write the path of the ncu that was found to FILE (empty
#                  when none was), for run.sh, which hands it to the timing
#                  suite as NCU_BIN
#
# For tests, and nothing else:
#   preflight.sh --classify NAME COMPUTE_CAP          the GPU verdict for that card
#   preflight.sh --choose-toolkit PINNED DRIVER MIN MODE
#   NVIDIA_SMI=program            use this in place of nvidia-smi
#   PREFLIGHT_EXPECT_CAP=8.6      accept another compute capability, to exercise
#                                 the kernel checks on a development GPU
#   WARPLENS_ENV_RUN="conda run -n warplens"   use an existing env
#   PREFLIGHT_OFFLINE=1           do not ask the network for the time
#   PREFLIGHT_NCU_CANDIDATES=...  where to look for an ncu that is not in the env or on PATH
set -uo pipefail
# shellcheck source=env/remote/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if [[ "${1:-}" == --classify ]]; then
  [[ $# -eq 3 ]] || { echo "usage: preflight.sh --classify NAME COMPUTE_CAP" >&2; exit 2; }
  classify_gpu "$2" "$3"; exit 0
fi
if [[ "${1:-}" == --choose-toolkit ]]; then
  [[ $# -eq 5 ]] || { echo "usage: preflight.sh --choose-toolkit PINNED DRIVER MIN MODE" >&2; exit 2; }
  choose_toolkit "$2" "$3" "$4" "$5"; exit 0
fi

quick=0
with_e2e=0
require_ncu=1
book="$(book_path)"
out_file=""
status_file=""
ncu_path_file=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) quick=1 ;;
    --e2e) with_e2e=1 ;;
    --no-ncu) require_ncu=0 ;;
    --book) book="${2:?--book needs a path}"; shift ;;
    --out) out_file="${2:?--out needs a file}"; shift ;;
    --status) status_file="${2:?--status needs a file}"; shift ;;
    --ncu-path) ncu_path_file="${2:?--ncu-path needs a file}"; shift ;;
    -h|--help) sed -n '2,41p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

SMI="${NVIDIA_SMI:-nvidia-smi}"
expect_cap="${PREFLIGHT_EXPECT_CAP:-10.0}"
load_pins || exit 2

report="$(mktemp)"
statuses="$(mktemp)"
work=""
ncu_bin=""
# shellcheck disable=SC2329  # called by the trap
cleanup() {
  [[ -n "$out_file" ]] && cp "$report" "$out_file"
  [[ -n "$status_file" ]] && cp "$statuses" "$status_file"
  [[ -n "$ncu_path_file" ]] && echo "$ncu_bin" > "$ncu_path_file"
  rm -rf "$report" "$statuses" ${work:+"$work"}
}
trap cleanup EXIT

n_pass=0; n_warn=0; n_fail=0
emit() { echo "$*" | tee -a "$report"; }
record() {  # record STATUS CHECK DETAIL
  printf -v line '%-5s %-14s %s' "$1" "$2" "$3"
  emit "$line"
  echo "$2 $1" >> "$statuses"
}
pass() { n_pass=$((n_pass + 1)); record PASS "$1" "$2"; }
warn() { n_warn=$((n_warn + 1)); record WARN "$1" "$2"; }
fail() { n_fail=$((n_fail + 1)); record FAIL "$1" "$2"; }
info() { record INFO "$1" "$2"; }
skip() { record SKIP "$1" "$2"; }
note() { emit "      $*"; }   # continuation lines under a check

emit "warplens preflight, $(utc_now), host $(hostname), $([[ $quick == 1 ]] && echo quick || echo full)"
[[ "$expect_cap" == 10.0 ]] || emit "NOTE: PREFLIGHT_EXPECT_CAP=$expect_cap is set: this is a development dry run, not a B200 check"

# ---------------------------------------------------------------------------
# GPU identity (plan step E20)
# ---------------------------------------------------------------------------
gpu0_cap=""; driver_cuda=""; driver_version=""; have_gpu=0
if ! command -v "$SMI" >/dev/null 2>&1; then
  fail nvidia-smi "'$SMI' not found: no NVIDIA driver is visible. In a container this means it was started without GPU access"
elif ! smi_text="$("$SMI" 2>&1)"; then
  fail nvidia-smi "nvidia-smi failed: $(head -n 2 <<<"$smi_text" | tr '\n' ' ')"
else
  have_gpu=1
  driver_version="$(parse_smi_driver_version "$smi_text")"
  driver_cuda="$(parse_smi_cuda_version "$smi_text")"
  pass nvidia-smi "driver ${driver_version:-unknown}"
fi

if (( have_gpu )); then
  fields="index,name,compute_cap,memory.total,memory.used,utilization.gpu"
  if ! gpus="$("$SMI" --query-gpu="$fields" --format=csv,noheader,nounits 2>/dev/null)" || [[ -z "$gpus" ]]; then
    # An nvidia-smi too old to know compute_cap: ask for the rest.
    gpus="$("$SMI" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' -v OFS=', ' '{print $1, $2, "unknown", $3, $4, $5}')"
  fi
  n_gpus=0; n_b200=0
  while IFS=',' read -r index name cap mem_total mem_used util; do
    [[ -n "$index" ]] || continue
    index="${index// /}"; name="${name# }"; cap="${cap// /}"
    mem_total="${mem_total// /}"; mem_used="${mem_used// /}"; util="${util// /}"
    n_gpus=$((n_gpus + 1))
    IFS='|' read -r verdict why <<<"$(classify_gpu "$name" "$cap")"
    want="$(classify_gpu "expected" "$expect_cap" | cut -d'|' -f1)"
    if [[ "$cap" == "$expect_cap" && "$verdict" == "$want" ]]; then
      n_b200=$((n_b200 + 1))
      if [[ "$index" == 0 ]]; then pass gpu-identity "GPU 0: $why"; else info gpu-identity "GPU $index: $why"; fi
    elif [[ "$verdict" == unknown ]]; then
      if (( quick )); then warn gpu-identity "GPU $index: $why. The full preflight asks the driver directly"
      else info gpu-identity "GPU $index: $why; asking the driver below"; fi
    elif [[ "$index" == 0 ]]; then
      fail gpu-identity "GPU 0 is not a B200: $why"
    else
      warn gpu-identity "GPU $index is not a B200: $why. The probes use GPU 0 only"
    fi
    if [[ "$index" == 0 ]]; then
      gpu0_cap="$cap"
      # GPU memory, and whether someone else is on the card (noisy timing).
      if [[ "$mem_total" =~ ^[0-9]+$ && "$mem_used" =~ ^[0-9]+$ ]]; then
        if (( mem_used > 1024 )); then
          warn gpu-memory "GPU 0: $mem_used of $mem_total MiB already in use before anything started"
        else
          pass gpu-memory "GPU 0: $mem_total MiB, $mem_used MiB in use"
        fi
      else
        warn gpu-memory "GPU 0: memory not reported ('$mem_total', '$mem_used')"
      fi
      apps="$("$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
      if [[ -n "$apps" ]]; then
        warn gpu-idle "other compute processes are on the GPU; timing results will be noisy:"
        while IFS= read -r app; do note "$app"; done <<<"$apps"
      elif [[ "$util" =~ ^[0-9]+$ ]] && (( util > 5 )); then
        warn gpu-idle "GPU 0 is ${util}% utilized with no process visible from here (another container or tenant?); timing results will be noisy"
      else
        pass gpu-idle "no other compute process visible, utilization ${util}% (a container cannot see processes outside it)"
      fi
    fi
  done <<<"$gpus"
  (( n_gpus > 0 )) || fail gpu-identity "nvidia-smi listed no GPU"
  (( n_gpus > 1 )) && info gpu-count "$n_gpus GPUs; only GPU 0 is used and one is all the run needs"

  # Driver, and the CUDA release it supports, against the pinned toolkit.
  if [[ -z "$driver_cuda" ]]; then
    warn driver "driver $driver_version did not report a CUDA version"
  else
    installed_toolkit="$(sed -n 's/^toolkit_installed=//p' "$REMOTE_HOME/resolved/versions.txt" 2>/dev/null)"
    IFS='|' read -r tk tk_status _ <<<"$(choose_toolkit "$PIN_CUDA_VERSION" "$driver_cuda" "$PIN_CUDA_VERSION_MIN" auto)"
    described="driver $driver_version supports CUDA $driver_cuda"
    if [[ "$tk_status" == unsupported ]]; then
      fail driver "$described; sm_100a needs $PIN_CUDA_VERSION_MIN or newer, and nothing installed here can change the driver"
    elif [[ -n "$installed_toolkit" ]] && version_ge "$driver_cuda" "$installed_toolkit"; then
      if [[ "$installed_toolkit" == "$PIN_CUDA_VERSION" ]]; then
        pass driver "$described, which covers the installed toolkit $installed_toolkit (as pinned)"
      else
        warn driver "$described, which covers the installed toolkit $installed_toolkit; that is older than the pinned $PIN_CUDA_VERSION, chosen by setup.sh to match the driver"
      fi
    elif [[ -n "$installed_toolkit" ]]; then
      warn driver "$described, older than the installed toolkit $installed_toolkit: the driver will refuse PTX from it (see ptx-jit below)"
    elif [[ "$tk_status" == ok ]]; then
      pass driver "$described, which covers the pinned toolkit $PIN_CUDA_VERSION"
    else
      warn driver "$described, older than the pinned toolkit $PIN_CUDA_VERSION; setup.sh will install toolkit $tk to match (its default, --toolkit auto)"
    fi
  fi
else
  skip gpu-identity "no driver"
fi

# ---------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------
disk_dir="$REMOTE_HOME"
while [[ ! -d "$disk_dir" ]]; do disk_dir="$(dirname "$disk_dir")"; done
free_gb=$(( $(df -Pk "$disk_dir" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if env_installed; then need=2; else need=12; fi
if (( free_gb < need )); then
  fail disk "$free_gb GB free under $disk_dir; need $need GB ($([[ $need == 12 ]] && echo "the env is not installed yet" || echo "for build output and results")). Set WARPLENS_REMOTE_HOME to a larger disk"
elif (( free_gb < need + 5 )); then
  warn disk "$free_gb GB free under $disk_dir: enough, with little to spare"
else
  pass disk "$free_gb GB free under $disk_dir"
fi

cpus="$(nproc 2>/dev/null || echo 1)"
mem_gb=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 / 1024 ))
if (( cpus < 4 )); then
  warn cpu "$cpus CPU(s): the probe builds run in parallel and will be slow"
else
  pass cpu "$cpus CPUs"
fi
if (( mem_gb < 8 )); then
  warn ram "$mem_gb GB of memory available; the CUTLASS builds want several"
else
  pass ram "$mem_gb GB of memory available"
fi

# Clock: the manifest's timestamps and every TLS download depend on it.
now="$(date +%s)"
commit_time="$(repo_git "$REPO_DIR" log -1 --format=%ct 2>/dev/null || true)"
clock_problem=""
if [[ "$commit_time" =~ ^[0-9]+$ ]] && (( now < commit_time )); then
  clock_problem="the clock ($(utc_now)) is earlier than the checked-out commit ($(date -u -d "@$commit_time" +%Y-%m-%dT%H:%M:%SZ))"
fi
net_date=""
if [[ -z "${PREFLIGHT_OFFLINE:-}" ]] && command -v curl >/dev/null 2>&1; then
  net_date="$(curl -sI --max-time 8 https://github.com 2>/dev/null | tr -d '\r' | sed -n 's/^[Dd]ate: //p' | head -n 1)"
fi
if [[ -n "$clock_problem" ]]; then
  warn clock "$clock_problem"
elif [[ -n "$net_date" ]] && net_s="$(date -d "$net_date" +%s 2>/dev/null)"; then
  skew=$(( now - net_s )); skew=${skew#-}
  if (( skew > 120 )); then
    warn clock "the clock is $skew s away from github.com's ($(utc_now) here)"
  else
    pass clock "$(utc_now), within $skew s of github.com's clock"
  fi
else
  info clock "$(utc_now); no network time to compare with"
fi

# Who we are: decides whether ncu will be allowed to read counters (below).
cap_eff="$(awk '/^CapEff:/ {print $2}' /proc/self/status 2>/dev/null)"
privileges="uid $EUID"
if [[ -n "$cap_eff" ]]; then
  has_capability "$cap_eff" 21 && privileges+=", CAP_SYS_ADMIN" || privileges+=", no CAP_SYS_ADMIN"
  has_capability "$cap_eff" 38 && privileges+=", CAP_PERFMON" || privileges+=", no CAP_PERFMON"
fi
admin_only="$(awk -F': *' '/^RmProfilingAdminOnly/ {print $2}' /proc/driver/nvidia/params 2>/dev/null)"
[[ -n "$admin_only" ]] && privileges+=", driver RmProfilingAdminOnly=$admin_only"
info privileges "container: $(container_kind); $privileges"

# ---------------------------------------------------------------------------
# Kernels on the GPU. Needs the env.
# ---------------------------------------------------------------------------
gpu_checks="compile predictions driver-api run-trivial ptx-jit tcgen05 tensor-map tma nvrtc torch ncu"
if (( quick )); then
  :
elif ! env_installed; then
  fail env "the env is not installed in $REMOTE_HOME: run env/remote/setup.sh first"
  for check in $gpu_checks; do skip "$check" "no env"; done
else
  pass env "${WARPLENS_ENV_RUN:-$ENV_PREFIX}"
  if (( with_e2e )); then
    # PyTorch lives in an env of its own (lib.sh, install_torch_env); without one, the env's Python.
    e2e_py="$(torch_python)"; [[ -x "$e2e_py" ]] || e2e_py=python
    if out="$(env_run timeout --kill-after=5 60 "$e2e_py" "$REMOTE_DIR/check_env.py" e2e "$book" 2>&1)"; then
      pass native-e2e "CUDA PyTorch, Triton and the book clone are available"
    else
      fail native-e2e "native end-to-end prerequisites are missing: $(tr '\n' ' ' <<<"$out")"
      note "PyTorch and the book: env/remote/setup.sh installs the one and fetches the other (run it again,"
      note "without --no-torch or --no-book). Or clone the book by hand and set WARPLENS_BOOK (RUNBOOK.md, section 3)."
      note "For a probe-only run: WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e env/remote/run.sh"
    fi
    # Which book: the committed prediction of row RT-08 is for one commit of it (pins.env).
    if is_book_clone "$book"; then
      book_head="$(book_commit "$book")"
      if [[ "$book_head" == "$PIN_BOOK_COMMIT" ]]; then
        pass book "$book is at commit ${book_head:0:12}, the one the predictions were made against"
      elif [[ "$book" == "$BOOK_DIR" ]]; then
        fail book "$book, the clone setup.sh keeps, is at commit $book_head, not the pinned $PIN_BOOK_COMMIT"
        note "env/remote/setup.sh fetches the pinned commit again (run it; with run.sh: --redo setup)."
      else
        warn book "$book is at commit $book_head, not the pinned $PIN_BOOK_COMMIT the predictions were made against: row RT-08 runs other kernels than were predicted"
        note "Unset WARPLENS_BOOK (and drop --book) to use the clone env/remote/setup.sh fetches at the pinned commit."
      fi
    fi
    # CUTLASS, for the four tutorials (RT-06) and the CuTe DSL tutorial (RT-11): the clone
    # setup.sh fetches into the checkout, and the programs run.sh's stage build-cutlass makes
    # from it. Without them the end-to-end step would clone and compile while the GPU stages
    # run (examples/03_cutlass_mma/build.sh, called by run_e2e.py), or with
    # WARPLENS_CUTLASS_NO_FETCH, as run.sh sets it, fail there.
    if ! is_cutlass_clone "$CUTLASS_DIR"; then
      fail cutlass "no clone of CUTLASS in $CUTLASS_DIR (or not a whole one): rows RT-06 and RT-11 cannot run"
      note "env/remote/setup.sh fetches it at the pinned commit (run it again; env/remote/run.sh does that by itself"
      note "after this failure). Without network: copy a clone at commit ${PIN_CUTLASS_COMMIT:0:12} there (RUNBOOK.md, section 3)."
      note "For a probe-only run: WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e env/remote/run.sh"
    elif [[ "$(git_head "$CUTLASS_DIR")" != "$PIN_CUTLASS_COMMIT" ]]; then
      fail cutlass "$CUTLASS_DIR is at commit $(git_head "$CUTLASS_DIR"), not the pinned $PIN_CUTLASS_COMMIT"
      note "env/remote/setup.sh checks out or fetches the pinned commit (run it again; env/remote/run.sh does that by itself)."
    else
      pass cutlass "$CUTLASS_DIR is at commit ${PIN_CUTLASS_COMMIT:0:12}, the pinned one"
    fi
    not_built="$(cutlass_programs_missing)"
    if [[ -z "$not_built" ]]; then
      pass cutlass-built "the four tutorial programs are in $(cutlass_bin_dir)"
    else
      fail cutlass-built "not built in $(cutlass_bin_dir): $not_built"
      note "env/remote/run.sh builds them before this check (stage build-cutlass; its log says why it failed)."
      note "By hand, in the env: examples/03_cutlass_mma/build.sh --programs"
    fi
    # Matrix row RT-11 is captured under gdb, and on a B200 the stage run-blackwell fails
    # without that capture (validation/blackwell/run_rows.py). Better known now than then.
    tracer="$(gdb_program)"
    if [[ -z "$tracer" ]]; then
      warn rt11-tracer "no gdb: matrix row RT-11 cannot be captured, and run-blackwell will fail for it"
      note "As root: apt-get install -y gdb. Otherwise env/remote/setup.sh again, or WARPLENS_GDB=/path/to/gdb (RUNBOOK.md, section 3)."
    elif out="$(timeout --kill-after=5 60 "$tracer" -q -batch -ex "python print('gdb-python-ok')" -ex run --args /bin/true 2>&1)" \
         && grep -q 'gdb-python-ok' <<<"$out" && grep -q 'exited normally' <<<"$out"; then
      pass rt11-tracer "$tracer has Python and may trace a process"
    else
      warn rt11-tracer "$tracer cannot trace a process, or has no Python: matrix row RT-11 cannot be captured, and run-blackwell will fail for it"
      note "$(tail -n 2 <<<"$out" | tr '\n' ' ')"
      note "A container that forbids ptrace needs --cap-add=SYS_PTRACE or another machine (RUNBOOK.md, section 3)."
    fi
  fi
  work="$(mktemp -d)"
  arch="$(arch_for_cap "$expect_cap")"
  blackwell_arch=sm_100a
  [[ "$expect_cap" == 10.3 ]] && blackwell_arch=sm_103a
  probe() { env_run timeout 120 python "$REMOTE_DIR/gpu_probe.py" "$@" 2>&1; }
  field() { sed -n "s/^$1=//p" <<<"$2" | head -n 1; }

  # Compile: the trivial kernel to machine code and to PTX, and the kernels
  # that use tcgen05 and TMA to machine code.
  compiled=1
  trivial_ran=0
  nvcc_version="$(env_run nvcc --version 2>/dev/null | sed -n 's/^Cuda compilation tools, release \([0-9.]*\).*/\1/p')"
  for job in "-cubin pf_vecadd.cu vecadd.cubin $arch" "-ptx pf_vecadd.cu vecadd.ptx $arch" \
             "-cubin pf_blackwell.cu blackwell.cubin $blackwell_arch"; do
    read -r mode source output job_arch <<<"$job"
    if ! err="$(env_run nvcc -arch="$job_arch" "$mode" "$REMOTE_DIR/kernels/$source" -o "$work/$output" 2>&1)"; then
      compiled=0
      fail compile "nvcc $nvcc_version -arch=$job_arch $mode $source: $(tail -n 3 <<<"$err" | tr '\n' ' ')"
    fi
  done
  ptx_version="$(sed -n 's/^\.version //p' "$work/vecadd.ptx" 2>/dev/null | head -n 1)"
  (( compiled )) && pass compile "nvcc $nvcc_version: $arch machine code and PTX (.version $ptx_version), $blackwell_arch tcgen05 and TMA kernels"

  # Is this the compiler the committed predictions were made with? If not, every
  # comparison is between two different compilations of a probe. A WARN, so
  # that the run goes on, but run.sh carries it into the archive in capitals.
  nvcc_full="$(env_run nvcc --version 2>/dev/null | sed -n 's/^Cuda compilation tools, release [0-9.]*, V\([0-9.]*\).*/\1/p' | head -n 1)"
  IFS='|' read -r comparable comparable_why \
    <<<"$(toolkit_verdict "$nvcc_full" "$(predictions_nvcc blackwell)" "$driver_cuda")"
  case "$comparable" in
    comparable) pass predictions "$comparable_why" ;;
    comparable-native) warn predictions "$comparable_why" ;;
    *) warn predictions "$comparable_why"
       note "To build with the predictions' compiler: env/remote/setup.sh --toolkit pinned --force, then run.sh again."
       note "run.sh --strict-toolkit stops here instead of going on." ;;
  esac
  timing_nvcc="$(predictions_nvcc timing)"
  if [[ -n "$timing_nvcc" && -n "$nvcc_full" && "$timing_nvcc" != "$nvcc_full" ]]; then
    warn predictions "the timing predictions were made with nvcc $timing_nvcc, the benchmarks will be built by nvcc $nvcc_full: predicted instruction counts are another compiler's"
  fi

  if (( ! have_gpu )); then
    for check in driver-api run-trivial ptx-jit tcgen05 tensor-map tma; do skip "$check" "no driver"; done
  else
    # The driver's own answer, which is what a program gets.
    if out="$(probe info)"; then
      api_cap="$(field compute_cap "$out")"; api_name="$(field name "$out")"; api_cuda="$(field driver_api "$out")"
      IFS='|' read -r verdict why <<<"$(classify_gpu "$api_name" "$api_cap")"
      if [[ "$api_cap" != "$expect_cap" ]]; then
        fail driver-api "the driver API reports device 0 as: $why"
      elif [[ "$gpu0_cap" =~ ^[0-9] && "$gpu0_cap" != "$api_cap" ]]; then
        fail driver-api "nvidia-smi says compute capability $gpu0_cap for GPU 0 but the driver API says $api_cap for device 0: device order differs, set CUDA_VISIBLE_DEVICES"
      else
        pass driver-api "device 0: $why; driver API $api_cuda, $(field total_mem_mib "$out") MiB"
      fi
    else
      fail driver-api "cannot initialize the driver API: $(field error "$out")"
    fi

    # A trivial kernel, as machine code.
    if [[ -f "$work/vecadd.cubin" ]]; then
      if out="$(probe vecadd "$work/vecadd.cubin")"; then
        trivial_ran=1
        pass run-trivial "$arch machine code from nvcc $nvcc_version ran and added $(field elements "$out") numbers correctly"
      else
        fail run-trivial "$arch machine code did not run: $(field error "$out")"
      fi
    else
      skip run-trivial "not compiled"
    fi

    # The same kernel as PTX, which the driver compiles. The Blackwell probes
    # are built this way, and a driver refuses PTX newer than itself.
    if [[ -f "$work/vecadd.ptx" ]]; then
      if out="$(probe vecadd "$work/vecadd.ptx")"; then
        pass ptx-jit "the driver compiled and ran PTX .version $ptx_version from nvcc $nvcc_version"
      elif [[ "$(field error_name "$out")" == CUDA_ERROR_UNSUPPORTED_PTX_VERSION ]]; then
        warn ptx-jit "the driver (CUDA ${driver_cuda:-?}) refuses PTX .version $ptx_version from nvcc $nvcc_version: $(field error "$out")"
        note "Programs that carry PTX will not load. Either rerun env/remote/setup.sh --toolkit auto --force"
        note "(installs the toolkit that matches the driver), or build with machine code"
        note "(validation/blackwell/build.sh --native). run.sh does the second for probes only."
        if (( with_e2e )); then
          fail e2e-ptx "the end-to-end programs still require PTX JIT; --native only covers the probe binaries"
          note "Use a driver-compatible toolkit (--toolkit auto), or select --no-e2e for a probe-only run."
        fi
      else
        fail ptx-jit "the driver did not run PTX from nvcc $nvcc_version: $(field error "$out")"
      fi
    else
      skip ptx-jit "not compiled"
    fi

    # Blackwell-only features: tensor memory, the tensor map encoder, a TMA copy.
    if [[ -f "$work/blackwell.cubin" ]]; then
      if out="$(probe tmem "$work/blackwell.cubin")"; then
        pass tcgen05 "tcgen05.alloc returned TMEM address $(field tmem_address "$out"), dealloc completed"
      else
        fail tcgen05 "the tcgen05.alloc/dealloc kernel did not run: $(field error "$out")"
      fi
    else
      skip tcgen05 "not compiled"
    fi
    if out="$(probe tensormap)"; then
      pass tensor-map "cuTensorMapEncodeTiled succeeded"
    else
      fail tensor-map "cuTensorMapEncodeTiled failed: $(field error "$out")"
    fi
    if [[ -f "$work/blackwell.cubin" ]]; then
      if out="$(probe tma "$work/blackwell.cubin")"; then
        pass tma "a TMA load of one 16x8 box arrived complete and correct"
      else
        fail tma "the TMA copy kernel failed: $(field error "$out")"
      fi
    else
      skip tma "not compiled"
    fi
  fi

  # NVRTC for sm_100a: what TIRx compiles with.
  if out="$(env_run python "$REMOTE_DIR/check_env.py" nvrtc "$blackwell_arch" 2>&1)"; then
    pass nvrtc "NVRTC $(field nvrtc "$out") compiles for $blackwell_arch (PTX .version $(field nvrtc_ptx_version "$out"))"
  else
    warn nvrtc "NVRTC cannot compile for $blackwell_arch, so TIRx kernels will not build: $(field error "$out")"
  fi

  # PyTorch, for the Triton-on-GPU step only. A WARN: everything else runs without.
  torch_py="$(torch_python)"
  if [[ ! -x "$torch_py" ]]; then
    warn torch "no PyTorch env ($torch_py): the Triton-on-GPU step will be SKIPPED. setup.sh makes it unless --no-torch was given or the driver is of an older major release than its CUDA runtime"
  elif ! out="$(env_run timeout 120 "$torch_py" "$REMOTE_DIR/check_env.py" torch 2>&1)"; then
    warn torch "PyTorch does not import in $torch_py: $(field torch "$out"); the Triton-on-GPU step will fail"
  elif [[ "$(field torch_cuda_available "$out")" != 1 ]]; then
    warn torch "PyTorch $(field torch_build "$out") sees no CUDA device ($(field torch_cuda_available "$out")); the Triton-on-GPU step will be SKIPPED"
  elif [[ "$(field torch_triton "$out")" != "$PIN_TRITON" ]]; then
    warn torch "the PyTorch env imports Triton $(field torch_triton "$out"), not the pinned $PIN_TRITON the simulator's prediction was made with"
  elif [[ ",$(field torch_arch_list "$out")," != *",sm_${gpu0_cap/./},"* && "$gpu0_cap" =~ ^[0-9] ]]; then
    warn torch "PyTorch $(field torch_build "$out") was not built for sm_${gpu0_cap/./} (it lists $(field torch_arch_list "$out")); its own operators may not run on this GPU"
  else
    pass torch "PyTorch $(field torch_build "$out") with Triton $(field torch_triton "$out") sees $(field torch_device "$out") (capability $(field torch_device_cap "$out"))"
  fi

  # ncu and the performance counters (plan step E21). The rental is there to hold the
  # timing model against real counters, so a counter must actually be read: every other
  # outcome is a FAIL, which stops run.sh before the GPU stages. With --no-ncu (the run
  # was told to do without counters) the same outcomes are WARNs. Nothing but PASS means
  # that a counter was read; run.sh records anything else as "unavailable".
  ncu_missed() {  # ncu_missed DETAIL: counters were not read
    if (( require_ncu )); then fail ncu "$1"; else warn ncu "counters are not needed (--no-ncu) and were not read: $1"; fi
  }
  ncu_consequence() {
    if (( require_ncu )); then
      note "A full run needs the counters: the timing suite's counter comparison is what they are for."
      note "If this machine cannot grant them, env/remote/run.sh --no-ncu runs everything else;"
      note "the archive then says that no counter was read (NCU-COUNTERS-NOT-READ.txt)."
    else
      note "Counters were declared not needed (--no-ncu): the run goes on and the ncu counter comparison is lost."
    fi
  }
  ncu_bin="$(env_run bash -c 'command -v ncu' 2>/dev/null || true)"
  if [[ -z "$ncu_bin" ]]; then
    for candidate in ${PREFLIGHT_NCU_CANDIDATES:-/usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu}; do
      [[ -x "$candidate" ]] && ncu_bin="$candidate"
    done
  fi
  if [[ -z "$ncu_bin" ]]; then
    ncu_missed "no ncu in the env, on PATH, in /usr/local/cuda/bin or in /opt/nvidia/nsight-compute: counters cannot be read. Rerun env/remote/setup.sh without --no-ncu"
    ncu_consequence
  elif (( ! trivial_ran )); then
    # Not a pass and, when counters are needed, not a skip either: nothing was read.
    if (( require_ncu )); then
      fail ncu "$ncu_bin found, but the trivial kernel did not run, so there was nothing to profile and no counter was read"
    else
      skip ncu "$ncu_bin found, but the trivial kernel did not run, so there is nothing to profile (--no-ncu: not needed)"
    fi
  else
    ncu_version="$("$ncu_bin" --version 2>/dev/null | sed -n 's/^Version \([0-9.]*\).*/\1/p' | head -n 1)"
    metric=smsp__inst_executed.sum
    out="$(env_run timeout 300 "$ncu_bin" --csv --metrics "$metric" python "$REMOTE_DIR/gpu_probe.py" vecadd "$work/vecadd.cubin" 2>&1)"
    IFS='|' read -r ncu_result ncu_detail <<<"$(parse_ncu_output "$metric" "$out")"
    if [[ "$ncu_result" == permission ]]; then
      ncu_missed "ncu $ncu_version runs but may not read performance counters:"
      note "$ncu_detail"
      note "Remedies, from https://developer.nvidia.com/ERR_NVGPUCTRPERM:"
      note "  1. run as root, or with CAP_SYS_ADMIN (CAP_PERFMON also works from driver R565):"
      note "       sudo env \"PATH=\$PATH\" WARPLENS_REMOTE_HOME=$(printf '%q' "$REMOTE_HOME") env/remote/run.sh"
      note "     WARPLENS_REMOTE_HOME must be given as here: sudo resets HOME and drops the variable, and"
      note "     without it root would install a second env in /root/warplens-remote. Other WARPLENS_"
      note "     variables you set (WARPLENS_BOOK, ...) are dropped the same way: repeat them after 'env'."
      note "  2. on the host, allow it for every user (needs root on the host, and a module reload or reboot):"
      note "       modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"
      note "       or 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' in a file in /etc/modprobe.d"
      note "  3. in a container: only the host can grant it, by 2. or by starting the container with --cap-add=SYS_ADMIN"
      ncu_consequence
    elif [[ "$ncu_result" == ok ]]; then
      pass ncu "ncu $ncu_version ($ncu_bin) read a performance counter: $metric = $ncu_detail"
    else
      ncu_missed "ncu $ncu_version ($ncu_bin) did not report the counter; its last output:"
      while IFS= read -r l; do note "$l"; done <<<"$(tail -n 6 <<<"$out")"
      ncu_consequence
    fi
  fi
fi

emit ""
emit "preflight: $n_pass passed, $n_warn warning(s), $n_fail failed"
if (( n_fail )); then
  emit "preflight: FAILED. What to do for each check is in env/remote/RUNBOOK.md, 'When preflight fails'."
  exit 1
fi
exit 0
