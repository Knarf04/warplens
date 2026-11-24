#!/usr/bin/env bash
# Verify the active env can compile CUDA C++ to PTX for the Blackwell targets.
# Run inside the env: conda run -n warplens env/check.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "nvcc:   $(command -v nvcc)"
nvcc --version | sed -n 's/^Cuda compilation tools, /        /p'
python -c 'import sys, numpy; print(f"python: {sys.version.split()[0]}, numpy {numpy.__version__}")'

status=0

# Driver, toolkit, and the toolkit the committed predictions were made with. The
# predictions (validation/blackwell/predicted, validation/blackwell_timing/predicted)
# are the simulator's run of PTX from one nvcc; if this env's nvcc is another, they
# must be regenerated (validation/blackwell/predict.sh, validation/blackwell_timing/
# predict.py) or every later comparison is between two compilations. That is an
# error here. A driver older than the toolkit is only noted: this env compiles, and
# no GPU is needed for that.
# shellcheck source=env/remote/lib.sh
source "$here/remote/lib.sh"
nvcc_version="$(nvcc --version | sed -n 's/^Cuda compilation tools, release [0-9.]*, V\([0-9.]*\).*/\1/p' | head -n 1)"
driver_cuda=""
if command -v nvidia-smi >/dev/null 2>&1 && smi_text="$(nvidia-smi 2>&1)"; then
  driver_cuda="$(parse_smi_cuda_version "$smi_text")"
  echo "driver: $(parse_smi_driver_version "$smi_text"), supports CUDA ${driver_cuda:-unknown}"
else
  echo "driver: none visible (not needed to compile)"
fi
for suite in blackwell timing; do
  predicted="$(predictions_nvcc "$suite")"
  IFS='|' read -r verdict why <<<"$(toolkit_verdict "$nvcc_version" "$predicted" "$driver_cuda")"
  case "$verdict" in
    comparable) echo "ok:     $suite predictions: $why" ;;
    comparable-native) echo "ok:     $suite predictions: $why" ;;
    *) echo "FAIL:   $suite predictions: $why"
       status=1 ;;
  esac
done

for arch in sm_100a sm_103a; do
  out="$tmp/vecadd.$arch.ptx"
  if nvcc -arch="$arch" -ptx "$here/check_vecadd.cu" -o "$out" 2>"$tmp/err"; then
    echo "ok:     $arch -> PTX $(sed -n 's/^\.version //p' "$out"), target $(sed -n 's/^\.target //p' "$out")"
  else
    echo "FAIL:   $arch"
    sed 's/^/        /' "$tmp/err"
    status=1
  fi
done
exit $status
