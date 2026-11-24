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
# package cache, state markers and results. Needs about 12 GB.
REMOTE_HOME="${WARPLENS_REMOTE_HOME:-$HOME/warplens-remote}"
ENV_PREFIX="$REMOTE_HOME/env"
MAMBA_BIN="$REMOTE_HOME/bin/micromamba"

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
  MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba" "$MAMBA_BIN" run -p "$ENV_PREFIX" "$@"
}

env_installed() {
  [[ -n "${WARPLENS_ENV_RUN:-}" ]] && return 0
  [[ -x "$MAMBA_BIN" && -x "$ENV_PREFIX/bin/python" && -f "$REMOTE_HOME/state/setup.done" ]]
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
