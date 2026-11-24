# shellcheck shell=bash
# Shared by env/remote/{setup,preflight,run}.sh. Source it; it runs nothing.
# Everything that decides something from text (which GPU this is, which CUDA
# release the driver supports, which toolkit to install) is a function of its
# arguments only, so tests/test_remote_env.py can call it without a GPU.

REMOTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034  # used by the scripts that source this file
REPO_DIR="$(cd "$REMOTE_DIR/../.." && pwd)"
PINS_FILE="$REMOTE_DIR/pins.env"

# Where everything that is not the checkout lives: micromamba, the env, the
# package cache, state markers and results. Needs about 18 GB (12 without the
# PyTorch env).
REMOTE_HOME="${WARPLENS_REMOTE_HOME:-$HOME/warplens-remote}"
ENV_PREFIX="$REMOTE_HOME/env"
# A venv on top of the env, holding PyTorch and nothing else (see pins.env).
# WARPLENS_TORCH_PYTHON names another interpreter that has torch and triton.
# shellcheck disable=SC2034  # used by the scripts that source this file
TORCH_ENV="$REMOTE_HOME/torch-env"
torch_python() { echo "${WARPLENS_TORCH_PYTHON:-$TORCH_ENV/bin/python}"; }
# gdb, for the rows that trace a launch (validation/blackwell/run_rows.py: RT-10, RT-11).
# The machine's own if it has one; else the one setup.sh put in an env of its own, apart
# from the pinned env. Prints nothing when there is none.
GDB_ENV="$REMOTE_HOME/gdb-env"
gdb_program() {
  if [[ -n "${WARPLENS_GDB:-}" ]]; then echo "$WARPLENS_GDB"
  elif command -v gdb >/dev/null 2>&1; then command -v gdb
  elif [[ -x "$GDB_ENV/bin/gdb" ]]; then echo "$GDB_ENV/bin/gdb"
  fi
}
MAMBA_BIN="$REMOTE_HOME/bin/micromamba"
# The clone of the book whose kernels row RT-08 runs ("Modern GPU Programming for MLSys").
# setup.sh fetches it here at the commit pins.env names (BOOK_COMMIT). It has no license:
# it is cloned from its own repository and never copied anywhere else, so it lives beside
# the run directory, not in it (run.sh archives the run directory only).
# WARPLENS_BOOK_URL names another place to fetch it from (a mirror; the tests use a stand-in).
BOOK_DIR="$REMOTE_HOME/book"
BOOK_URL="${WARPLENS_BOOK_URL:-https://github.com/mlc-ai/modern-gpu-programming-for-mlsys}"
# What the suites take for a clone of the book (run_e2e.py, check_env.py).
is_book_clone() { [[ -n "${1:-}" && -f "$1/chapter_gemm_basics/index.md" ]]; }
# The book the suites are to use: WARPLENS_BOOK when it is set (also when set to nothing,
# which means no book), else setup.sh's clone when there is one. Prints nothing otherwise.
book_path() {
  if [[ -n "${WARPLENS_BOOK+set}" ]]; then echo "$WARPLENS_BOOK"
  elif is_book_clone "$BOOK_DIR"; then echo "$BOOK_DIR"
  fi
}

# Load pins.env into PIN_<KEY> variables. The file is parsed, not sourced: a
# line that is not KEY=VALUE is an error.
load_pins() {
  local line key value n=0
  [[ -f "$PINS_FILE" ]] || { echo "error: $PINS_FILE not found" >&2; return 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    n=$((n + 1))
    [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9._-]+)$ ]]; then
      key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
      printf -v "PIN_$key" '%s' "$value"
    else
      echo "error: $PINS_FILE line $n is not KEY=VALUE: $line" >&2
      return 1
    fi
  done < "$PINS_FILE"
}

# version_ge A B: true when version A >= version B (numeric, dot-separated).
version_ge() {
  [[ "$1" == "$2" ]] && return 0
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]]
}

# The CUDA release a driver supports, from the text `nvidia-smi` prints
# ("... Driver Version: 580.173.02     CUDA Version: 13.0 ..."). Prints X.Y,
# or nothing when the text has no such field.
parse_smi_cuda_version() {
  sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' <<<"$1" | head -n 1
}

parse_smi_driver_version() {
  sed -n 's/.*Driver Version: *\([0-9][0-9.]*\).*/\1/p' <<<"$1" | head -n 1
}

# classify_gpu NAME COMPUTE_CAP -> one line "verdict|explanation".
#
# The compute capability decides; the name only adds to the explanation, because
# marketing names mislead ("RTX PRO 6000 Blackwell Server Edition" is a
# datacenter card and is 12.0). Capabilities are NVIDIA's published table,
# https://developer.nvidia.com/cuda/gpus (read 2026-11-23): B200 and GB200 10.0,
# B300 10.3, RTX 50 series and RTX PRO Blackwell 12.0, GB10 12.1, H100/H200 9.0.
#
#   b200                10.0: sm_100a, what the probes are built for
#   b300                10.3: sm_103a, a datacenter Blackwell but not this target
#   consumer-blackwell  12.x: sm_120/sm_121, no tcgen05 or tensor memory
#   hopper              9.0:  TMA and mbarrier but no tcgen05
#   pre-hopper          below 9.0
#   unknown             no usable compute capability, or one not listed above
classify_gpu() {
  local name="$1" cap="$2" major minor
  if [[ ! "$cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "unknown|no compute capability reported for '$name' (got '$cap'); the name alone does not decide"
    return 0
  fi
  major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"
  if [[ "$major" == 10 && "$minor" == 0 ]]; then
    if [[ "$name" == *B200* ]]; then
      echo "b200|$name, compute capability 10.0 (sm_100a)"
    else
      echo "b200|compute capability 10.0 (sm_100a); the name '$name' does not say B200"
    fi
  elif [[ "$major" == 10 && "$minor" == 3 ]]; then
    echo "b300|$name, compute capability 10.3 (sm_103a): a B300-class card, not the sm_100a target the probes are built for"
  elif [[ "$major" == 12 ]]; then
    echo "consumer-blackwell|$name, compute capability $cap (sm_12$minor): a consumer or workstation Blackwell, which has no tcgen05 or tensor memory"
  elif [[ "$major" == 9 ]]; then
    echo "hopper|$name, compute capability $cap (sm_9$minor): Hopper has TMA but no tcgen05 or tensor memory"
  elif (( major < 9 )); then
    echo "pre-hopper|$name, compute capability $cap (sm_$major$minor): older than Hopper, none of the Blackwell features"
  else
    echo "unknown|$name, compute capability $cap is not one this script knows"
  fi
}

# The -arch name for a compute capability: 10.0 -> sm_100a, 10.3 -> sm_103a, 8.6 -> sm_86.
arch_for_cap() {
  case "$1" in
    10.0) echo sm_100a ;;
    10.3) echo sm_103a ;;
    *) echo "sm_${1/./}" ;;
  esac
}

# choose_toolkit PINNED DRIVER MINIMUM MODE -> "version|status|explanation".
#
# PINNED is the toolkit release in pins.env, DRIVER the CUDA release the
# driver supports (empty: no driver found), MODE one of auto, pinned, or an
# explicit X.Y.
#
# What is known about a toolkit newer than the driver, measured on the
# development machine (driver 580.173.02 = CUDA 13.0, toolkit 13.5, RTX 3080 Ti):
# machine code from the newer nvcc loads and runs; PTX from it is refused
# (CUDA_ERROR_UNSUPPORTED_PTX_VERSION: ".version 9.5; current version is 9.0").
# The Blackwell probes carry PTX, so `auto` installs the release that matches
# the driver. A driver of an older major release cannot load the newer CUDA
# runtime library at all (libcudart.so.13 on a 12.x driver).
#
# status: ok (as pinned), fallback (older release chosen), mismatch (pinned
# kept although the driver is older), unsupported (driver below MINIMUM).
choose_toolkit() {
  local pinned="$1" driver="$2" minimum="$3" mode="${4:-auto}"
  if [[ "$mode" != auto && "$mode" != pinned ]]; then
    if [[ ! "$mode" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "$pinned|unsupported|--toolkit takes auto, pinned or X.Y, not '$mode'"
    elif ! version_ge "$mode" "$minimum"; then
      echo "$mode|unsupported|toolkit $mode was asked for; the oldest that knows sm_100a is $minimum"
    else
      echo "$mode|fallback|toolkit $mode was asked for explicitly (pins.env has $pinned); its package versions are whatever conda-forge resolves and are recorded"
    fi
    return 0
  fi
  if [[ -z "$driver" ]]; then
    echo "$pinned|ok|no NVIDIA driver found, so nothing to match: installing the pinned toolkit $pinned"
  elif version_ge "$driver" "$pinned"; then
    echo "$pinned|ok|the driver supports CUDA $driver, which covers the pinned toolkit $pinned"
  elif ! version_ge "$driver" "$minimum"; then
    echo "$pinned|unsupported|the driver supports only CUDA $driver; sm_100a needs $minimum or newer, so this machine cannot run the probes whatever is installed"
  elif [[ "$mode" == pinned ]]; then
    echo "$pinned|mismatch|the driver supports CUDA $driver but the pinned toolkit $pinned is kept as asked: PTX from it will be refused by the driver, and across a major release its runtime will not load"
  else
    echo "$driver|fallback|the driver supports CUDA $driver, older than the pinned toolkit $pinned: installing toolkit $driver so the driver accepts its PTX; package versions for it are whatever conda-forge resolves and are recorded"
  fi
}

# The nvcc version the committed predictions were made with: the simulator ran PTX from
# that compiler, so a GPU run of programs built by another one is not the same programs.
# predictions_nvcc blackwell|timing -> X.Y.Z, or nothing when it is not recorded.
predictions_nvcc() {
  local file="$REPO_DIR/validation/blackwell/predicted/provenance.json"
  [[ "${1:-blackwell}" == timing ]] && file="$REPO_DIR/validation/blackwell_timing/predicted/meta.json"
  sed -n 's/.*"nvcc_version": *"\([0-9][0-9.]*\)".*/\1/p' "$file" 2>/dev/null | head -n 1
}

# toolkit_verdict INSTALLED_NVCC PREDICTIONS_NVCC DRIVER_CUDA -> "status|explanation".
#
# INSTALLED_NVCC is the version of the nvcc that will build the probes (X.Y.Z),
# PREDICTIONS_NVCC what predictions_nvcc printed, DRIVER_CUDA the CUDA release
# the driver supports (X.Y, or empty).
#
#   comparable       the same nvcc, and the driver takes its PTX
#   comparable-native  the same nvcc, but the driver is older than it: its PTX
#                    is refused, the machine-code builds (which the same nvcc
#                    produces) run, and predictions still apply
#   not-comparable   another nvcc builds the probes: the fallback toolkit
#   unknown          a version is missing
toolkit_verdict() {
  local installed="$1" predicted="$2" driver="$3" release
  if [[ -z "$installed" || "$installed" == MISSING || -z "$predicted" ]]; then
    echo "unknown|TOOLKIT NOT RECORDED: installed nvcc '${installed:-?}', predictions made with nvcc '${predicted:-?}'; cannot tell whether the predictions are comparable"
  elif [[ "$installed" != "$predicted" ]]; then
    echo "not-comparable|BUILT WITH FALLBACK TOOLKIT - PREDICTIONS NOT COMPARABLE: the probes will be built by nvcc $installed, but the committed predictions are the simulator's run of PTX from nvcc $predicted. Every verdict then compares two different compilations; a difference may be the compiler's"
  else
    release="$(cut -d. -f1,2 <<<"$installed")"
    if [[ -n "$driver" ]] && ! version_ge "$driver" "$release"; then
      echo "comparable-native|nvcc $installed as for the predictions, but the driver supports only CUDA $driver: it refuses this toolkit's PTX, so the machine-code builds run (same compiler, same code; ptxas of the toolkit instead of the driver's)"
    else
      echo "comparable|nvcc $installed as for the predictions${driver:+, and the driver (CUDA $driver) takes its PTX}"
    fi
  fi
}

# choose_torch DRIVER_CUDA TORCH_CUDA -> "install|why" or "skip|why".
# The pinned torch wheel carries the CUDA runtime of release TORCH_CUDA; a
# driver of an older major release cannot load it (the same rule as for the
# toolkit, above). Within the major release an older driver is fine: measured
# with the CUDA 13.2 wheel on a driver that supports 13.0.
choose_torch() {
  local driver="$1" torch_cuda="$2"
  if [[ -z "$driver" ]]; then
    echo "install|no NVIDIA driver found, so nothing to match: installing as pinned"
  elif [[ "${driver%%.*}" == "${torch_cuda%%.*}" ]] || version_ge "$driver" "$torch_cuda"; then
    echo "install|the driver supports CUDA $driver, and the pinned PyTorch carries the CUDA $torch_cuda runtime"
  else
    echo "skip|the driver supports only CUDA $driver; the pinned PyTorch carries the CUDA $torch_cuda runtime, which a ${driver%%.*}.x driver cannot load, and no PyTorch release pairs the pinned Triton with CUDA ${driver%%.*}. The Triton-on-GPU step (validation/blackwell/e2e/triton_gpu.py) will be SKIPPED"
  fi
}

# parse_ncu_output METRIC TEXT -> "ok|value", "permission|the error line" or "none|".
# TEXT is what `ncu --csv --metrics METRIC program` printed. A counter was read
# only if a CSV row carries METRIC with a number in its last column
# (...,"smsp__inst_executed.sum","inst","1,234"). The metric's name also
# appears in command lines and error messages, which must not count.
parse_ncu_output() {
  local metric="$1" text="$2" line value
  if line="$(grep -m1 'ERR_NVGPUCTRPERM' <<<"$text")"; then
    echo "permission|$line"
    return 0
  fi
  while IFS= read -r line; do
    [[ "$line" == \"*\",\"$metric\",\"* ]] || continue
    value="${line##*,\"}"; value="${value%\"*}"
    if [[ "$value" =~ ^[0-9][0-9,.]*$ ]]; then
      echo "ok|$value"
      return 0
    fi
  done <<<"$text"
  echo "none|"
}

# Is this process allowed to read GPU performance counters without the
# driver's admin-only restriction mattering? Bits of CapEff: 21 CAP_SYS_ADMIN,
# 38 CAP_PERFMON (https://developer.nvidia.com/ERR_NVGPUCTRPERM).
# has_capability HEX_CAPEFF BIT
has_capability() {
  local caps=$((16#$1))
  (( (caps >> $2) & 1 ))
}

# container_kind -> docker | podman | kubernetes | lxc | container | none (best effort).
container_kind() {
  if [[ -f /.dockerenv ]]; then echo docker
  elif [[ -f /run/.containerenv ]]; then echo podman
  elif [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then echo kubernetes
  elif grep -qaE 'docker|containerd|kubepods|lxc' /proc/1/cgroup 2>/dev/null; then echo container
  elif [[ "$(tr -d '\0' < /proc/1/comm 2>/dev/null)" != systemd && "$(tr -d '\0' < /proc/1/comm 2>/dev/null)" != init ]]; then echo container
  else echo none
  fi
}

# Run a command inside the installed env (activated, so nvcc finds its host compiler).
# WARPLENS_ENV_RUN names another way in, e.g. "conda run -n warplens" on the
# development machine.
env_run() {
  if [[ -n "${WARPLENS_ENV_RUN:-}" ]]; then
    # shellcheck disable=SC2086
    $WARPLENS_ENV_RUN "$@"
    return
  fi
  # PYTHONNOUSERSITE: packages in ~/.local are not part of the env.
  PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba" "$MAMBA_BIN" run -p "$ENV_PREFIX" "$@"
}

# install_torch_env BASE_CONSTRAINTS: make $TORCH_ENV, a venv of the env's Python that
# sees the env's packages (Triton, NumPy), and install the pinned PyTorch into it with
# every package it brings held to its TORCH_PIP_ pin. Needs load_pins. Returns non-zero
# when the venv or the install fails; the caller decides what that means.
install_torch_env() {
  local base="$1" constraints="$REMOTE_HOME/state/torch-constraints.txt" pin name
  mkdir -p "$REMOTE_HOME/state"
  {
    [[ -f "$base" ]] && cat "$base"
    echo "triton==$PIN_TRITON"
    echo "cuda-bindings==$PIN_CUDA_BINDINGS"
    for pin in "${!PIN_TORCH_PIP_@}"; do     # PIN_TORCH_PIP_NVIDIA_CUBLAS constrains nvidia-cublas
      name="${pin#PIN_TORCH_PIP_}"; name="${name,,}"
      echo "${name//_/-}==${!pin}"
    done
  } > "$constraints"
  rm -rf "$TORCH_ENV"
  env_run python -m venv --system-site-packages "$TORCH_ENV" || return 1
  # TMPDIR: pip unpacks 3 GB of wheels there; /tmp may be too small (it was, here).
  mkdir -p "$REMOTE_HOME/tmp"
  TMPDIR="$REMOTE_HOME/tmp" PYTHONNOUSERSITE=1 env_run "$TORCH_ENV/bin/python" -m pip install -q --no-input --disable-pip-version-check \
    --no-cache-dir -c "$constraints" "torch==$PIN_TORCH" "triton==$PIN_TRITON"
}

env_installed() {
  [[ -n "${WARPLENS_ENV_RUN:-}" ]] && return 0
  [[ -x "$MAMBA_BIN" && -x "$ENV_PREFIX/bin/python" && -f "$REMOTE_HOME/state/setup.done" ]]
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# git_head DIR -> the commit checked out in DIR, or "unknown". Without a git
# program (a bare container) it is read from DIR/.git directly.
git_head() {
  local dir="$1" head ref
  if command -v git >/dev/null 2>&1 && head="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"; then
    echo "$head"
    return 0
  fi
  if [[ -f "$dir/.git/HEAD" ]]; then
    head="$(cat "$dir/.git/HEAD")"
    if [[ "$head" == "ref: "* ]]; then
      ref="${head#ref: }"
      if [[ -f "$dir/.git/$ref" ]]; then
        head="$(cat "$dir/.git/$ref")"
      else
        head="$(awk -v ref="$ref" '$2 == ref {print $1}' "$dir/.git/packed-refs" 2>/dev/null)"
      fi
    fi
    if [[ "$head" =~ ^[0-9a-f]{40}$ ]]; then
      echo "$head"
      return 0
    fi
  fi
  echo unknown
}

# The commit of the checkout; "unknown" for a copy that has no .git at all.
repo_commit() { git_head "$REPO_DIR"; }

# The commit a clone of the book is at. Only a directory with a .git of its own counts:
# a copy of the book's files inside some other repository is not at that repository's commit.
book_commit() {
  if [[ -e "$1/.git" ]]; then git_head "$1"; else echo unknown; fi
}

# git, the machine's or the one setup.sh put into the env when the machine had none.
# Never asks for a name or password: a repository that wants one is a failed fetch.
git_run() {
  if command -v git >/dev/null 2>&1; then GIT_TERMINAL_PROMPT=0 git "$@"
  else GIT_TERMINAL_PROMPT=0 env_run git "$@"
  fi
}

# ensure_book COMMIT -> one line "status|explanation": make $BOOK_DIR a clone of the book
# at exactly COMMIT. Returns non-zero when it is not one afterwards.
#
#   present   it already was; nothing was fetched
#   fetched   fetched from $BOOK_URL (also over a clone at another commit, which is replaced)
#   failed    the fetch failed or did not give COMMIT; what was there before is left as it was
#
# The fetch goes into a directory beside $BOOK_DIR and replaces it only when it is right, so
# an interrupted or failed fetch leaves no half clone. One commit, no history (--depth 1).
ensure_book() {
  local commit="$1" part="$BOOK_DIR.part" head had="" lacks=""
  if is_book_clone "$BOOK_DIR"; then
    head="$(book_commit "$BOOK_DIR")"
    if [[ "$head" == "$commit" ]]; then
      echo "present|$BOOK_DIR is at commit $commit"
      return 0
    fi
    had="; $BOOK_DIR is left as it was, at commit $head"
  fi
  rm -rf "$part"
  # A stalled connection is given up on after 30 s below 1 kB/s: nothing here may hang the setup.
  if ! { git_run init -q "$part" \
         && git_run -C "$part" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
              fetch -q --depth 1 "$BOOK_URL" "$commit" \
         && git_run -C "$part" checkout -q --detach FETCH_HEAD; } >&2; then
    rm -rf "$part"
    echo "failed|could not fetch commit $commit from $BOOK_URL (is there network access to it? does the repository still have that commit?)$had"
    return 1
  fi
  head="$(book_commit "$part")"
  is_book_clone "$part" || lacks=" and has no chapter_gemm_basics/index.md"
  if [[ "$head" != "$commit" || -n "$lacks" ]]; then
    rm -rf "$part"
    echo "failed|what $BOOK_URL gave for $commit is at commit $head$lacks: not the book at the pinned commit$had"
    return 1
  fi
  rm -rf "$BOOK_DIR"
  mv "$part" "$BOOK_DIR" || { echo "failed|could not move $part to $BOOK_DIR"; return 1; }
  echo "fetched|$BOOK_DIR is at commit $commit, from $BOOK_URL"
}

# provide_book WITH_BOOK COMMIT -> "status|explanation": what setup.sh does about the book.
# The statuses of ensure_book, and two in which nothing is fetched or touched:
#
#   opted-out  WITH_BOOK is 0 (setup.sh --no-book)
#   given      WARPLENS_BOOK is set: that is the book the suites will use (book_path)
#
# Returns non-zero only for `failed`.
provide_book() {
  if [[ "$1" != 1 ]]; then
    echo "opted-out|not fetched (--no-book). Row RT-08 needs a clone: set WARPLENS_BOOK to one, or leave the end-to-end programs out (WARPLENS_BLACKWELL_RUN_ARGS=--no-e2e)"
  elif [[ -n "${WARPLENS_BOOK+set}" ]]; then
    echo "given|not fetched: WARPLENS_BOOK is set ('$WARPLENS_BOOK') and is what the suites will use"
  else
    ensure_book "$2"
  fi
}
