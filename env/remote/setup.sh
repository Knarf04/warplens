#!/usr/bin/env bash
# Install the pinned environment on a fresh rented Linux GPU machine.
#
#   env/remote/setup.sh [--toolkit auto|pinned|X.Y] [--no-ncu] [--force]
#
# Non-interactive and safe to run again: a finished setup is recognized and
# only re-verified. Needs no root, no conda and no system CUDA toolkit: it
# downloads one static micromamba binary and builds a private env from
# conda-forge and PyPI, at the versions in env/remote/pins.env. Root is used
# only if a download tool or git is missing and apt-get is available.
#
#   --toolkit auto     (default) the pinned CUDA toolkit, or the release that
#                      matches the driver when the driver is older (the driver
#                      of a rented machine cannot be changed; see RUNBOOK.md)
#   --toolkit pinned   the pinned toolkit whatever the driver is
#   --toolkit X.Y      that toolkit release
#   --no-ncu           do not install Nsight Compute (about 0.5 GB)
#   --force            redo every step even if setup finished before
#
#   WARPLENS_REMOTE_HOME   where everything goes (default: ~/warplens-remote).
#                          Needs about 12 GB; point it at the large disk.
#
# Writes $WARPLENS_REMOTE_HOME/resolved/: the exact conda packages, pip freeze,
# and versions.txt with what was actually installed next to what was pinned.
set -euo pipefail
# shellcheck source=env/remote/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

toolkit_mode=auto
with_ncu=1
force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolkit) toolkit_mode="${2:?--toolkit needs a value}"; shift ;;
    --toolkit=*) toolkit_mode="${1#*=}" ;;
    --no-ncu) with_ncu=0 ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { echo "setup: $*"; }
warn() { echo "setup: WARNING: $*" >&2; warnings+=("$*"); }
die()  { echo "setup: error: $*" >&2; exit 1; }
warnings=()

load_pins
mkdir -p "$REMOTE_HOME"/{bin,state,resolved,mamba}
export MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba"

# ---------------------------------------------------------------------------
# 1. The machine
# ---------------------------------------------------------------------------
[[ "$(uname -s)" == Linux ]] || die "this script is for Linux, not $(uname -s)"
# GB200 systems pair the GPU with an Arm CPU. Nothing here was tried on one:
# the pinned micromamba checksum and the PyPI wheels are for x86_64.
[[ "$(uname -m)" == x86_64 ]] || die "only x86_64 is supported and tested, not $(uname -m) (a GB200 node has Arm CPUs; rent an HGX B200 instead)"
os_name="unknown"
if [[ -r /etc/os-release ]]; then
  os_name="$(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"
fi
say "machine: $os_name, $(uname -r), $(nproc) CPUs, user $(id -un) (uid $EUID), container: $(container_kind)"
case "$os_name" in
  "Ubuntu 22.04"*|"Ubuntu 24.04"*) ;;
  *) warn "written for Ubuntu 22.04/24.04; '$os_name' was not tried" ;;
esac

free_kb="$(df -Pk "$REMOTE_HOME" | awk 'NR==2 {print $4}')"
free_gb=$((free_kb / 1024 / 1024))
if [[ ! -f "$REMOTE_HOME/state/setup.done" ]] && (( free_gb < 12 )); then
  die "$REMOTE_HOME has $free_gb GB free; the env and its package cache need about 12 GB. Set WARPLENS_REMOTE_HOME to a directory on a larger disk"
fi

# Root, or sudo that asks nothing; empty when neither.
as_root=""
if (( EUID == 0 )); then as_root="env"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then as_root="sudo -n"
fi
apt_install() {
  [[ -n "$as_root" ]] && command -v apt-get >/dev/null 2>&1 || return 1
  say "installing system package(s): $*"
  $as_root apt-get update -qq || return 1
  $as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@"
}

# ---------------------------------------------------------------------------
# 2. Driver, and which toolkit goes with it
# ---------------------------------------------------------------------------
driver_cuda=""
driver_version=""
if command -v nvidia-smi >/dev/null 2>&1 && smi_text="$(nvidia-smi 2>&1)"; then
  driver_cuda="$(parse_smi_cuda_version "$smi_text")"
  driver_version="$(parse_smi_driver_version "$smi_text")"
  say "driver $driver_version, supports CUDA ${driver_cuda:-unknown}"
  [[ -n "$driver_cuda" ]] || warn "nvidia-smi ran but printed no 'CUDA Version'; cannot match the toolkit to the driver"
else
  warn "nvidia-smi is missing or failed: no driver to match, installing as pinned. On the rented machine this means the GPU is not visible"
fi

IFS='|' read -r toolkit toolkit_status toolkit_why \
  <<<"$(choose_toolkit "$PIN_CUDA_VERSION" "$driver_cuda" "$PIN_CUDA_VERSION_MIN" "$toolkit_mode")"
case "$toolkit_status" in
  ok) say "toolkit: $toolkit_why" ;;
  fallback|mismatch) warn "toolkit: $toolkit_why" ;;
  unsupported)
    if [[ "$toolkit_mode" == auto ]]; then
      warn "toolkit: $toolkit_why"
    else
      die "toolkit: $toolkit_why"
    fi ;;
esac
if [[ "${toolkit%%.*}" != "${PIN_CUDA_VERSION%%.*}" ]]; then
  warn "toolkit $toolkit is a different major release from the pinned $PIN_CUDA_VERSION: the pinned cuda-bindings $PIN_CUDA_BINDINGS and apache-tvm-tirx $PIN_APACHE_TVM_TIRX expect CUDA ${PIN_CUDA_VERSION%%.*} libraries and will probably not run on the GPU (the simulator paths still work)"
fi

# One line that identifies this configuration; a finished setup with the same
# line is not redone.
config_id="pins=$(sha256sum "$PINS_FILE" | cut -c1-16) toolkit=$toolkit ncu=$with_ncu repo=$REPO_DIR"
if (( ! force )) && [[ -f "$REMOTE_HOME/state/setup.done" ]] \
   && [[ "$(cat "$REMOTE_HOME/state/setup.done")" == "$config_id" ]] \
   && [[ -x "$MAMBA_BIN" && -x "$ENV_PREFIX/bin/python" ]]; then
  say "already set up in $REMOTE_HOME for this configuration; verifying only (use --force to reinstall)"
  already=1
else
  already=0
  rm -f "$REMOTE_HOME/state/setup.done"
fi

# ---------------------------------------------------------------------------
# 3. micromamba
# ---------------------------------------------------------------------------
# Why micromamba and not the machine's own toolkit or conda: a rented image has
# whatever CUDA toolkit its provider chose (often none, or one too old to know
# sm_100a), and may have no conda. micromamba is a single static binary that
# needs no root and no installer, and conda-forge carries nvcc, the CUDA
# headers, NVRTC and Nsight Compute at exact versions. It is also what CI's
# `full` job uses to build the same env.
fetch() {  # fetch URL FILE
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --connect-timeout 20 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q --tries=3 -O "$2" "$1"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  else return 127
  fi
}
have_fetch_tool() { command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; }

mamba_ok() {
  [[ -x "$MAMBA_BIN" ]] && [[ "$(sha256sum "$MAMBA_BIN" | cut -d' ' -f1)" == "$PIN_MICROMAMBA_SHA256_LINUX_64" ]]
}
if ! mamba_ok; then
  if ! have_fetch_tool; then
    apt_install curl ca-certificates \
      || die "no curl, wget or python3 to download with, and cannot install one (not root, or no apt-get). Install curl, or copy micromamba $PIN_MICROMAMBA_VERSION to $MAMBA_BIN"
  fi
  url="https://github.com/mamba-org/micromamba-releases/releases/download/$PIN_MICROMAMBA_VERSION/micromamba-linux-64"
  say "downloading micromamba $PIN_MICROMAMBA_VERSION"
  fetch "$url" "$MAMBA_BIN.part" || die "download failed: $url (is there network access? are CA certificates installed?)"
  got="$(sha256sum "$MAMBA_BIN.part" | cut -d' ' -f1)"
  [[ "$got" == "$PIN_MICROMAMBA_SHA256_LINUX_64" ]] \
    || die "micromamba checksum mismatch: got $got, pins.env has $PIN_MICROMAMBA_SHA256_LINUX_64"
  chmod +x "$MAMBA_BIN.part"
  mv -f "$MAMBA_BIN.part" "$MAMBA_BIN"
fi
say "micromamba $("$MAMBA_BIN" --version)"
mamba() { "$MAMBA_BIN" --no-rc --no-env "$@"; }
# conda-forge's cuda-version package requires the solver's idea of the driver
# (the `__cuda` virtual package) to be at least as new. The driver was matched
# above, and micromamba's own detection is not reliable: on the development
# machine it reported 12.4, the version of an unrelated toolkit in
# /usr/local/cuda, for a driver that supports 13.0, and then refused to solve.
export CONDA_OVERRIDE_CUDA="$toolkit"

# ---------------------------------------------------------------------------
# 4. The conda env
# ---------------------------------------------------------------------------
if (( ! already )); then
  specs=("python==$PIN_PYTHON" "numpy==$PIN_NUMPY" "pyyaml==$PIN_PYYAML" "pytest==$PIN_PYTEST" pip
         "setuptools==$PIN_SETUPTOOLS" "cuda-version==$toolkit")
  if [[ "$toolkit" == "$PIN_CUDA_VERSION" ]]; then
    specs+=("cuda-nvcc==$PIN_CUDA_NVCC" "cuda-cudart-dev==$PIN_CUDA_CUDART_DEV" "cuda-nvrtc==$PIN_CUDA_NVRTC")
  else
    specs+=(cuda-nvcc cuda-cudart-dev cuda-nvrtc)
  fi
  # git: the probe builds clone CUTLASS, and run.sh records the commit.
  if ! command -v git >/dev/null 2>&1; then
    apt_install git ca-certificates || { say "no system git and no way to install one: adding conda-forge's git to the env (not pinned)"; specs+=(git); }
  fi
  verb=create
  [[ -x "$ENV_PREFIX/bin/python" ]] && verb=install
  say "conda-forge: $verb ${specs[*]}"
  mamba "$verb" -y -q -p "$ENV_PREFIX" --override-channels -c conda-forge --strict-channel-priority "${specs[@]}" \
    || die "conda-forge install failed (see above)"

  if (( with_ncu )); then
    ncu_spec="nsight-compute"
    [[ "$toolkit" == "$PIN_CUDA_VERSION" ]] && ncu_spec="nsight-compute==$PIN_NSIGHT_COMPUTE"
    say "conda-forge: install $ncu_spec"
    mamba install -y -q -p "$ENV_PREFIX" --override-channels -c conda-forge --strict-channel-priority \
        "$ncu_spec" "cuda-version==$toolkit" \
      || warn "Nsight Compute did not install; preflight will look for an ncu already on the machine"
  fi

  # -------------------------------------------------------------------------
  # 5. PyPI packages, then warplens from this checkout
  # -------------------------------------------------------------------------
  constraints="$REMOTE_HOME/state/pip-constraints.txt"
  cat > "$constraints" <<EOF
numpy==$PIN_NUMPY
PyYAML==$PIN_PYYAML
pytest==$PIN_PYTEST
setuptools==$PIN_SETUPTOOLS
apache-tvm-ffi==$PIN_APACHE_TVM_FFI
apache-tvm-ffi-orcjit==$PIN_APACHE_TVM_FFI_ORCJIT
cuda-pathfinder==$PIN_CUDA_PATHFINDER
nvidia-cutlass-dsl-libs-base==$PIN_NVIDIA_CUTLASS_DSL_LIBS_BASE
nvidia-cutlass-dsl-libs-core==$PIN_NVIDIA_CUTLASS_DSL_LIBS_CORE
nvidia-cutlass-dsl-libs-cu12==$PIN_NVIDIA_CUTLASS_DSL_LIBS_CU12
ml-dtypes==$PIN_ML_DTYPES
typing-extensions==$PIN_TYPING_EXTENSIONS
EOF
  say "PyPI: triton $PIN_TRITON, nvidia-cutlass-dsl $PIN_NVIDIA_CUTLASS_DSL, apache-tvm-tirx $PIN_APACHE_TVM_TIRX, cuda-bindings $PIN_CUDA_BINDINGS"
  env_run python -m pip install -q --no-input --disable-pip-version-check -c "$constraints" \
      "triton==$PIN_TRITON" "nvidia-cutlass-dsl==$PIN_NVIDIA_CUTLASS_DSL" \
      "apache-tvm-tirx==$PIN_APACHE_TVM_TIRX" "cuda-bindings==$PIN_CUDA_BINDINGS" \
    || die "pip install failed (see above)"
  say "warplens from $REPO_DIR (editable)"
  env_run python -m pip install -q --no-input --disable-pip-version-check --no-deps -e "$REPO_DIR" \
    || die "installing warplens failed (see above)"
fi

# ---------------------------------------------------------------------------
# 6. Record what was installed, and compare with the pins
# ---------------------------------------------------------------------------
resolved="$REMOTE_HOME/resolved"
mamba list -p "$ENV_PREFIX" --explicit --sha256 > "$resolved/conda-explicit.txt"
env_run python -m pip freeze --disable-pip-version-check > "$resolved/pip-freeze.txt"
# The version of one conda package, from its file name in the explicit list
# (.../cuda-nvcc-13.5.37-hcdd1206_1.conda#sha256): name-version-build.
conda_version() {
  awk -F/ -v want="$1" '
    /^https?:/ {
      file = $NF; sub(/#.*/, "", file); sub(/\.(conda|tar\.bz2)$/, "", file)
      n = split(file, part, "-"); name = part[1]
      for (i = 2; i <= n - 2; i++) name = name "-" part[i]
      if (name == want) { print part[n - 1]; found = 1 }
    }
    END { if (!found) print "MISSING" }' "$resolved/conda-explicit.txt"
}

package_status=0
{
  echo "# Written by env/remote/setup.sh at $(utc_now). pinned = env/remote/pins.env."
  echo "os=$os_name"
  echo "kernel=$(uname -r)"
  echo "driver=${driver_version:-none}"
  echo "driver_cuda=${driver_cuda:-none}"
  echo "toolkit_pinned=$PIN_CUDA_VERSION"
  echo "toolkit_installed=$toolkit"
  echo "toolkit_decision=$toolkit_status: $toolkit_why"
  echo "micromamba=$("$MAMBA_BIN" --version)"
  echo "cuda_version=$(conda_version cuda-version)"
  echo "cuda_nvcc=$(conda_version cuda-nvcc)"
  echo "cuda_cudart_dev=$(conda_version cuda-cudart-dev)"
  echo "cuda_nvrtc=$(conda_version cuda-nvrtc)"
  echo "nsight_compute=$(conda_version nsight-compute)"
  env_run python "$REMOTE_DIR/check_env.py" versions || package_status=1
} > "$resolved/versions.txt"
(( package_status == 0 )) || { grep MISSING "$resolved/versions.txt" >&2; die "a required package is missing or does not import (above)"; }

drift=()
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[a-z_0-9]+$ ]] || continue
  pin_var="PIN_${key^^}"
  [[ -n "${!pin_var:-}" ]] || continue
  [[ "${!pin_var}" == "$value" ]] || drift+=("$key: pinned ${!pin_var}, installed $value")
done < "$resolved/versions.txt"
[[ "$(conda_version python)" == "$PIN_PYTHON" ]] || drift+=("python (conda): pinned $PIN_PYTHON, installed $(conda_version python)")
for line in "${drift[@]}"; do
  # A fallback toolkit differs from the pins by design; without ncu, so does nsight_compute.
  case "$line" in
    cuda_*|nsight_compute*) [[ "$toolkit_status" == fallback || "$line" == nsight_compute*MISSING ]] && { say "differs from pins: $line"; continue; } ;;
  esac
  warn "differs from pins: $line"
done
: > "$resolved/drift.txt"
(( ${#drift[@]} == 0 )) || printf '%s\n' "${drift[@]}" > "$resolved/drift.txt"

# ---------------------------------------------------------------------------
# 7. Does it work: compile for both Blackwell targets, NVRTC, import warplens
# ---------------------------------------------------------------------------
say "toolchain check (env/check.sh)"
env_run bash "$REPO_DIR/env/check.sh" || die "nvcc cannot compile for the Blackwell targets"
nvrtc_out="$(env_run python "$REMOTE_DIR/check_env.py" nvrtc sm_100a)" \
  || die "NVRTC cannot compile for sm_100a: $nvrtc_out"
say "NVRTC: $(tr '\n' ' ' <<<"$nvrtc_out")"
echo "$nvrtc_out" | grep -E '^nvrtc' >> "$resolved/versions.txt"
env_run python -c 'import warplens; warplens.Device("B200")' || die "warplens does not import from $REPO_DIR"

# A way in for people and for other scripts.
cat > "$REMOTE_HOME/envrun" <<EOF
#!/usr/bin/env bash
# Run a command inside the warplens remote env: $REMOTE_HOME/envrun <command...>
export MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba"
exec "$MAMBA_BIN" run -p "$ENV_PREFIX" "\$@"
EOF
chmod +x "$REMOTE_HOME/envrun"

echo "$config_id" > "$REMOTE_HOME/state/setup.done"
say "done. Env: $ENV_PREFIX. Versions: $resolved/versions.txt"
say "run a command in it with: $REMOTE_HOME/envrun <command>"
if (( ${#warnings[@]} )); then
  say "${#warnings[@]} warning(s):"
  printf 'setup:   - %s\n' "${warnings[@]}"
fi
