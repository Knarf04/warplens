#!/usr/bin/env bash
# Install the pinned environment on a fresh rented Linux GPU machine.
#
#   env/remote/setup.sh [--toolkit auto|pinned|X.Y] [--no-ncu] [--no-torch] [--no-book] [--force]
#
# Non-interactive and safe to run again: a finished setup is recognized and
# only re-verified. Needs no root, no conda and no system CUDA toolkit: it
# downloads one static micromamba binary and builds a private env from
# conda-forge and PyPI, at the versions in env/remote/pins.env, and clones the
# book whose kernels the end-to-end step runs, and CUTLASS, whose tutorials it
# runs, each at the commit pinned there. Root is used only if a download tool,
# git or gdb is missing and apt-get is available.
#
#   --toolkit auto     (default) the pinned CUDA toolkit, or the release that
#                      matches the driver when the driver is older (the driver
#                      of a rented machine cannot be changed; see RUNBOOK.md)
#   --toolkit pinned   the pinned toolkit whatever the driver is
#   --toolkit X.Y      that toolkit release
#   --no-ncu           do not install Nsight Compute (about 0.5 GB). Counters can
#                      then not be read: preflight.sh and run.sh need --no-ncu too
#   --no-torch         do not make the PyTorch env (about 4.5 GB). Only the
#                      Triton-on-GPU step of the probe suite needs it
#   --no-book          do not fetch the book ("Modern GPU Programming for MLSys",
#                      matrix row RT-08) into $WARPLENS_REMOTE_HOME/book. It is
#                      not fetched either when WARPLENS_BOOK is set: that clone
#                      is the one used. Without a book a full run stops at
#                      preflight (check native-e2e)
#   --force            redo every step even if setup finished before
#
#   WARPLENS_REMOTE_HOME   where everything goes (default: ~/warplens-remote).
#                          Needs about 18 GB (12 with --no-torch); point it at
#                          the large disk.
#   WARPLENS_BOOK_URL      fetch the book from here instead of its repository
#                          on github.com (a mirror)
#   WARPLENS_CUTLASS_URL   the same for CUTLASS. It is fetched into the checkout,
#                          examples/03_cutlass_mma/.cutlass (git-ignored), one
#                          commit without history; a clone already there at the
#                          pinned commit is left alone
#
# Writes $WARPLENS_REMOTE_HOME/resolved/: the exact conda packages, pip freeze,
# and versions.txt with what was actually installed next to what was pinned.
set -euo pipefail
# shellcheck source=env/remote/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

toolkit_mode=auto
with_ncu=1
with_torch=1
with_book=1
force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolkit) toolkit_mode="${2:?--toolkit needs a value}"; shift ;;
    --toolkit=*) toolkit_mode="${1#*=}" ;;
    --no-ncu) with_ncu=0 ;;
    --no-torch) with_torch=0 ;;
    --no-book) with_book=0 ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,41p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { echo "setup: $*"; }
warn() { echo "setup: WARNING: $*" >&2; warnings+=("$*"); }
die()  { echo "setup: error: $*" >&2; exit 1; }
warnings=()

load_pins
mkdir -p "$REMOTE_HOME"/{bin,state,resolved,mamba,tmp}
# pip and mktemp unpack under TMPDIR. /tmp is often a small disk of its own: on the
# development machine the PyTorch wheels (3 GB) filled it and the install failed with
# "No space left on device". Everything goes where the env goes.
export TMPDIR="$REMOTE_HOME/tmp"
export MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba"
# Packages in ~/.local must neither satisfy pip nor be imported: a fresh rented
# machine has none, and the development machine's would hide what is missing.
export PYTHONNOUSERSITE=1

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
need_gb=$(( with_torch ? 18 : 12 ))
if [[ ! -f "$REMOTE_HOME/state/setup.done" ]] && (( free_gb < need_gb )); then
  die "$REMOTE_HOME has $free_gb GB free; the env and its package cache need about $need_gb GB (12 with --no-torch). Set WARPLENS_REMOTE_HOME to a directory on a larger disk"
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
  warn "toolkit $toolkit is a different major release from the pinned $PIN_CUDA_VERSION: the pinned cuda-bindings $PIN_CUDA_BINDINGS and apache-tvm-tirx $PIN_APACHE_TVM_TIRX are built for CUDA ${PIN_CUDA_VERSION%%.*}. Measured with toolkit 12.8: nvcc and CUDA C++ programs are unaffected, but the TIRx, CuTe DSL and driver-shim launch paths fail, in the simulator too (3 failures and 9 errors in the test suite)"
fi

# One line that identifies this configuration; a finished setup with the same
# line is not redone.
config_id="pins=$(sha256sum "$PINS_FILE" | cut -c1-16) toolkit=$toolkit ncu=$with_ncu torch=$with_torch repo=$REPO_DIR"
# An env that was installed with ncu serves a later --no-ncu as it is (run.sh --no-ncu
# after a preflight that could not read counters must not spend minutes reinstalling).
done_id="$(cat "$REMOTE_HOME/state/setup.done" 2>/dev/null || true)"
if (( ! with_ncu )) && [[ "$done_id" == "${config_id/ ncu=0 / ncu=1 }" ]]; then
  config_id="$done_id"
fi
if (( ! force )) && [[ -n "$done_id" && "$done_id" == "$config_id" ]] \
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
         "setuptools==$PIN_SETUPTOOLS" "typing_extensions==$PIN_TYPING_EXTENSIONS" "cuda-version==$toolkit")
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
  {
    echo "numpy==$PIN_NUMPY"
    echo "PyYAML==$PIN_PYYAML"
    echo "pytest==$PIN_PYTEST"
    echo "setuptools==$PIN_SETUPTOOLS"
    echo "typing-extensions==$PIN_TYPING_EXTENSIONS"
    for pin in "${!PIN_PIP_@}"; do          # PIN_PIP_ML_DTYPES becomes a constraint on ml-dtypes
      name="${pin#PIN_PIP_}"; name="${name,,}"
      echo "${name//_/-}==${!pin}"
    done
  } > "$constraints"
  say "PyPI: triton $PIN_TRITON, nvidia-cutlass-dsl $PIN_NVIDIA_CUTLASS_DSL, apache-tvm-tirx $PIN_APACHE_TVM_TIRX, cuda-bindings $PIN_CUDA_BINDINGS"
  env_run python -m pip install -q --no-input --disable-pip-version-check -c "$constraints" \
      "triton==$PIN_TRITON" "nvidia-cutlass-dsl==$PIN_NVIDIA_CUTLASS_DSL" \
      "apache-tvm-tirx==$PIN_APACHE_TVM_TIRX" "cuda-bindings==$PIN_CUDA_BINDINGS" \
    || die "pip install failed (see above)"
  say "warplens from $REPO_DIR (editable)"
  env_run python -m pip install -q --no-input --disable-pip-version-check --no-deps -e "$REPO_DIR" \
    || die "installing warplens failed (see above)"
  rm -rf "$TORCH_ENV"   # it was made for the env that has just been replaced
fi

# ---------------------------------------------------------------------------
# 5b. PyTorch, in a venv on top of the env (why not in it: pins.env)
# ---------------------------------------------------------------------------
# Nothing but the Triton-on-GPU step needs it, so a failure here is a warning:
# the step is then reported as skipped, and the rest runs. Tried again by every
# run of this script for as long as the env is missing, also after a setup that
# finished without it.
IFS='|' read -r torch_action torch_why <<<"$(choose_torch "$driver_cuda" "$PIN_TORCH_CUDA_VERSION")"
if (( ! with_torch )); then
  say "PyTorch: not installed (--no-torch); the Triton-on-GPU step will be skipped"
  rm -rf "$TORCH_ENV"
elif [[ "$torch_action" == skip ]]; then
  warn "PyTorch: not installed: $torch_why"
elif [[ -x "$TORCH_ENV/bin/python" && -f "$REMOTE_HOME/state/torch.done" ]]; then
  say "PyTorch: already in $TORCH_ENV"
else
  say "PyTorch: $torch_why"
  say "PyPI: torch $PIN_TORCH into $TORCH_ENV (a venv that sees the env's packages)"
  rm -f "$REMOTE_HOME/state/torch.done"
  if install_torch_env "$REMOTE_HOME/state/pip-constraints.txt"; then
    echo "$PIN_TORCH" > "$REMOTE_HOME/state/torch.done"
  else
    warn "PyTorch did not install (see above); the Triton-on-GPU step will be skipped. Run env/remote/setup.sh again to retry"
    rm -rf "$TORCH_ENV"
  fi
fi

# ---------------------------------------------------------------------------
# 5c. gdb, for the two rows that trace a launch
# ---------------------------------------------------------------------------
# validation/blackwell/run_rows.py runs TIRx and the CuTe DSL under gdb to log their driver
# calls (matrix row RT-10) and to save the tensor maps a CuTe DSL kernel is handed (RT-11).
# Any gdb with Python support does; it is a tool, not part of what is measured, so it is
# not pinned and not installed into the env. Without it those rows are skipped, and the
# descriptors are what a later decoder needs: hence three attempts.
if [[ -z "$(gdb_program)" ]]; then
  apt_install gdb || {
    say "no gdb and no way to install the system package: conda-forge's gdb into $GDB_ENV (not pinned)"
    mamba create -y -q -p "$GDB_ENV" --override-channels -c conda-forge gdb \
      || { rm -rf "$GDB_ENV"; warn "gdb did not install; matrix rows RT-10 and RT-11 (driver call trace, tensor maps of the CuTe DSL) will be skipped"; }
  }
fi
gdb_bin="$(gdb_program)"
if [[ -n "$gdb_bin" ]] && "$gdb_bin" -q -batch -ex "python print('gdb-python-ok')" 2>/dev/null | grep -q gdb-python-ok; then
  say "gdb: $("$gdb_bin" --version | head -n 1) ($gdb_bin)"
elif [[ -n "$gdb_bin" ]]; then
  warn "$gdb_bin has no Python support; matrix rows RT-10 and RT-11 will be skipped (set WARPLENS_GDB to another gdb)"
fi

# ---------------------------------------------------------------------------
# 5d. The book, for the end-to-end step (matrix row RT-08)
# ---------------------------------------------------------------------------
# validation/blackwell/e2e runs the book's GEMM kernels from a clone of the book's own
# repository: it has no license, so it is not in this one, and it is only ever cloned.
# The commit is the one the committed prediction was made against (pins.env). Like the
# PyTorch env, a failure is a warning here and is tried again by every run of this script;
# a full run then stops at preflight (check native-e2e), before anything is run on the GPU.
IFS='|' read -r book_state book_why <<<"$(provide_book "$with_book" "$PIN_BOOK_COMMIT")"
case "$book_state" in
  present|fetched|opted-out|given) say "the book: $book_state: $book_why" ;;
  *) warn "the book: ${book_why:-not fetched}. Row RT-08 (the book's GEMM kernels) cannot run and a full run stops at preflight (native-e2e). Run env/remote/setup.sh again to retry, or clone it by hand and set WARPLENS_BOOK (RUNBOOK.md, section 3)" ;;
esac

# ---------------------------------------------------------------------------
# 5e. CUTLASS, for the end-to-end step (matrix rows RT-06 and RT-11)
# ---------------------------------------------------------------------------
# The four Blackwell tutorials and the CuTe DSL's GEMM tutorial are run from a clone of
# CUTLASS at one commit. examples/03_cutlass_mma/build.sh used to clone it, with its whole
# history and no time limit, when the end-to-end step first asked for a program: inside the
# GPU stage of a rental. It is fetched here instead, like the book, and run.sh builds the
# programs in a stage of its own before the GPU is used (build-cutlass). A failure is a
# warning and is tried again by every run of this script; a full run then stops at preflight
# (check cutlass).
IFS='|' read -r cutlass_state cutlass_why <<<"$(ensure_cutlass "$PIN_CUTLASS_COMMIT")"
case "$cutlass_state" in
  present|fetched) say "CUTLASS: $cutlass_state: $cutlass_why" ;;
  *) warn "CUTLASS: ${cutlass_why:-not fetched}. Rows RT-06 (the CUTLASS tutorials) and RT-11 (the CuTe DSL tutorial) cannot run and a full run stops at preflight (cutlass). Run env/remote/setup.sh again to retry, or copy a clone at that commit to $CUTLASS_DIR (RUNBOOK.md, section 3)" ;;
esac

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
  (( with_ncu )) || echo "ncu_requested=no (--no-ncu): this setup was told to do without Nsight Compute, so no performance counters are expected"
  # The clone setup.sh keeps, whatever WARPLENS_BOOK says: what was fetched here.
  if is_book_clone "$BOOK_DIR"; then echo "book_commit=$(book_commit "$BOOK_DIR")"; else echo "book_commit=not-fetched"; fi
  if is_cutlass_clone "$CUTLASS_DIR"; then echo "cutlass_commit=$(git_head "$CUTLASS_DIR")"; else echo "cutlass_commit=not-fetched"; fi
  echo "gdb=$( { [[ -n "$gdb_bin" ]] && "$gdb_bin" --version 2>/dev/null | head -n 1; } || echo none)"
  env_run python "$REMOTE_DIR/check_env.py" versions || package_status=1
  if [[ -x "$TORCH_ENV/bin/python" ]]; then
    env_run "$TORCH_ENV/bin/python" "$REMOTE_DIR/check_env.py" torch || echo "torch=MISSING"
  else
    echo "torch=not-installed"
  fi
} > "$resolved/versions.txt"
(( package_status == 0 )) || { grep MISSING "$resolved/versions.txt" >&2; die "a required package is missing or does not import (above)"; }

# The three versions that decide whether a result can be held against a committed
# prediction: the driver, the nvcc installed here, and the nvcc the predictions
# were made with. Recorded, compared, and said loudly when they do not agree.
installed_nvcc="$(conda_version cuda-nvcc)"
IFS='|' read -r comparable comparable_why \
  <<<"$(toolkit_verdict "$installed_nvcc" "$(predictions_nvcc blackwell)" "$driver_cuda")"
{
  echo "predictions_nvcc=$(predictions_nvcc blackwell)"
  echo "predictions_nvcc_timing=$(predictions_nvcc timing)"
  echo "predictions_comparable=$comparable"
  echo "predictions_comparable_why=$comparable_why"
} >> "$resolved/versions.txt"
case "$comparable" in
  comparable) say "predictions: $comparable_why" ;;
  comparable-native) warn "predictions: $comparable_why" ;;
  *) warn "$comparable_why"
     banner="$comparable_why" ;;
esac
if [[ -n "$(predictions_nvcc timing)" && "$(predictions_nvcc timing)" != "$(predictions_nvcc blackwell)" ]]; then
  warn "the timing predictions were made with nvcc $(predictions_nvcc timing), the probe predictions with nvcc $(predictions_nvcc blackwell): regenerate one of them"
fi

drift=()
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[a-z_0-9]+$ ]] || continue
  pin_var="PIN_${key^^}"
  # check_env.py reports packages it does not import as pip_<name>; their pin
  # is PIP_<NAME>, or plain <NAME> when conda installs it.
  [[ -n "${!pin_var:-}" || "$key" != pip_* ]] || { plain="${key#pip_}"; pin_var="PIN_${plain^^}"; }
  [[ -n "${!pin_var:-}" ]] || continue
  [[ "${!pin_var}" == "$value" ]] || drift+=("$key: pinned ${!pin_var}, installed $value")
done < "$resolved/versions.txt"
[[ "$(conda_version python)" == "$PIN_PYTHON" ]] || drift+=("python (conda): pinned $PIN_PYTHON, installed $(conda_version python)")
# The torch env (its torch_ and torch_pip_ lines were compared with the TORCH_ pins by
# the loop above): the Triton it imports must be the main env's.
if [[ -x "$TORCH_ENV/bin/python" ]]; then
  torch_triton="$(sed -n 's/^torch_triton=//p' "$resolved/versions.txt")"
  [[ "$torch_triton" == "$PIN_TRITON" ]] || drift+=("torch env's triton: pinned $PIN_TRITON, it imports '${torch_triton:-none}'")
fi
for line in "${drift[@]}"; do
  # A fallback toolkit differs from the pins by design; without ncu, so does nsight_compute;
  # without the torch env, so does torch; without the book (said or warned of above), so does it.
  case "$line" in
    cuda_*|nsight_compute*) [[ "$toolkit_status" == fallback || "$line" == nsight_compute*MISSING ]] && { say "differs from pins: $line"; continue; } ;;
    torch:*not-installed|book_commit:*not-fetched|cutlass_commit:*not-fetched) say "differs from pins: $line"; continue ;;
  esac
  warn "differs from pins: $line"
done
: > "$resolved/drift.txt"
(( ${#drift[@]} == 0 )) || printf '%s\n' "${drift[@]}" > "$resolved/drift.txt"

# ---------------------------------------------------------------------------
# 7. Does it work: compile for the Blackwell targets, NVRTC, import warplens
# ---------------------------------------------------------------------------
# sm_100a is what the rental needs. sm_103a (B300) is what env/check.sh also asks
# for; toolkit 12.8 does not have it, so on a fallback toolkit it is a warning.
check_dir="$(mktemp -d)"
for arch in sm_100a sm_103a; do
  if err="$(env_run nvcc -arch="$arch" -ptx "$REPO_DIR/env/check_vecadd.cu" -o "$check_dir/$arch.ptx" 2>&1)"; then
    say "nvcc $(conda_version cuda-nvcc): $arch ok, PTX .version $(sed -n 's/^\.version //p' "$check_dir/$arch.ptx")"
    [[ "$arch" == sm_100a ]] && echo "nvcc_ptx_version=$(sed -n 's/^\.version //p' "$check_dir/$arch.ptx")" >> "$resolved/versions.txt"
  elif [[ "$arch" == sm_100a || "$toolkit" == "$PIN_CUDA_VERSION" ]]; then
    rm -rf "$check_dir"
    die "nvcc cannot compile for $arch: $err"
  else
    warn "nvcc of toolkit $toolkit cannot compile for $arch (B300); the B200 probes do not need it: $(tail -n 1 <<<"$err")"
  fi
done
rm -rf "$check_dir"
# NVRTC is what TIRx compiles with. The pinned cuda-bindings loads the NVRTC of
# its own major release, so with a toolkit of another major this cannot work.
if nvrtc_out="$(env_run python "$REMOTE_DIR/check_env.py" nvrtc sm_100a 2>&1)"; then
  say "NVRTC: $(tr '\n' ' ' <<<"$nvrtc_out")"
  grep -E '^nvrtc' <<<"$nvrtc_out" >> "$resolved/versions.txt"
  # cuda-bindings takes the first NVRTC it finds, and prefers one from a pip wheel
  # (nvidia-cuda-nvrtc, which PyTorch pulls in) to the toolkit's. That would change
  # the PTX of every TIRx kernel without a word, so it is checked.
  nvrtc_loaded="$(sed -n 's/^nvrtc=//p' <<<"$nvrtc_out" | head -n 1)"
  [[ "$nvrtc_loaded" == "$toolkit" ]] \
    || warn "cuda-bindings loads NVRTC $nvrtc_loaded, not the installed toolkit's $toolkit: TIRx kernels are compiled by another compiler than on the development machine. Is an nvidia-cuda-nvrtc wheel in the env (pip list)? PyTorch belongs in $TORCH_ENV, not here"
elif [[ "${toolkit%%.*}" == "${PIN_CUDA_VERSION%%.*}" ]]; then
  die "NVRTC cannot compile for sm_100a: $nvrtc_out"
else
  warn "NVRTC is not usable from cuda-bindings $PIN_CUDA_BINDINGS with toolkit $toolkit, so TIRx kernels will not compile here: $(tail -n 1 <<<"$nvrtc_out")"
fi
env_run python -c 'import warplens; warplens.Device("B200")' || die "warplens does not import from $REPO_DIR"

# A way in for people and for other scripts.
cat > "$REMOTE_HOME/envrun" <<EOF
#!/usr/bin/env bash
# Run a command inside the warplens remote env: $REMOTE_HOME/envrun <command...>
export MAMBA_ROOT_PREFIX="$REMOTE_HOME/mamba"
export PYTHONNOUSERSITE=1
exec "$MAMBA_BIN" run -p "$ENV_PREFIX" "\$@"
EOF
chmod +x "$REMOTE_HOME/envrun"

rm -rf "$REMOTE_HOME/tmp"
echo "$config_id" > "$REMOTE_HOME/state/setup.done"
say "done. Env: $ENV_PREFIX. Versions: $resolved/versions.txt"
say "run a command in it with: $REMOTE_HOME/envrun <command>"
if (( ${#warnings[@]} )); then
  say "${#warnings[@]} warning(s):"
  printf 'setup:   - %s\n' "${warnings[@]}"
fi
if [[ -n "${banner:-}" ]]; then
  bar="$(printf '!%.0s' {1..100})"
  printf '%s\nsetup: %s\nsetup: To build with the predictions'"'"' compiler instead: env/remote/setup.sh --toolkit pinned --force\nsetup: (the driver then refuses its PTX and the machine-code builds are run; see RUNBOOK.md, "Driver older than the toolkit").\n%s\n' \
    "$bar" "$banner" "$bar" >&2
fi
