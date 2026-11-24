#!/usr/bin/env bash
# Build every Blackwell probe. Needs only nvcc with sm_100a support (CUDA 12.8 or newer;
# written against 13.5); no GPU.
#
#   validation/blackwell/build.sh [--native] [--jobs N] [area/probe ...]
#   validation/blackwell/build.sh --ampere      the few probes an Ampere GPU can run, for
#                                               the local dry run (see README, "Dry run")
#
# For each <area>/<probe>.cu this produces
#
#   bin/<area>/<probe>          the program, carrying PTX (not machine code) and using the
#                               shared CUDA runtime. This one file runs on a B200, where the
#                               driver compiles the PTX, and under `warplens run`.
#   bin/<area>/<probe>.o        the same source assembled to sm_100a machine code. Never run;
#                               it proves that ptxas accepts every instruction, which the
#                               PTX-carrying build does not check.
#   bin-native/<area>/<probe>   with --native: the program with sm_100a machine code, for a
#                               machine whose driver is older than the compiler's PTX.
#
#   bin-ampere/<area>/<probe>   with --ampere: built by $HW_NVCC for $HW_ARCH with
#                               -DPROBE_AMPERE, which leaves out what needs sm_90 or newer.
#
#   NVCC      compiler to use (default: nvcc on PATH; locally `conda run -n warplens ...`)
#   HW_NVCC   --ampere: nvcc of a full toolkit (default: /usr/local/cuda/bin/nvcc)
#   HW_ARCH   --ampere: architecture of the local GPU (default: sm_86)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NVCC="${NVCC:-nvcc}"
native=0
ampere=0
# What an Ampere GPU can run: mbarrier.init / arrive / test_wait, and the host-only program.
AMPERE_PROBES=(mbarrier/script e2e/device_attributes)
jobs="$(nproc 2>/dev/null || echo 4)"
probes=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --native) native=1 ;;
    --ampere) ampere=1 ;;
    --jobs) jobs="$2"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) probes+=("${1%.cu}") ;;
  esac
  shift
done
if [[ "$ampere" == 1 ]]; then
  HW_NVCC="${HW_NVCC:-/usr/local/cuda/bin/nvcc}"
  for probe in "${AMPERE_PROBES[@]}"; do
    mkdir -p "bin-ampere/$(dirname "$probe")"
    "$HW_NVCC" -std=c++17 -O2 -w -arch="${HW_ARCH:-sm_86}" -DPROBE_AMPERE "$probe.cu" -o "bin-ampere/$probe"
    echo "built bin-ampere/$probe"
  done
  exit 0
fi
if [[ ${#probes[@]} -eq 0 ]]; then
  for source in */*.cu; do probes+=("${source%.cu}"); done
fi
command -v "$NVCC" >/dev/null || { echo "error: $NVCC not found" >&2; exit 1; }

build_one() {
  local probe="$1" native="$2" flags="-std=c++17 -O2 -w"
  mkdir -p "bin/$(dirname "$probe")"
  "$NVCC" $flags -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared \
    "$probe.cu" -o "bin/$probe" || { echo "FAILED (PTX program): $probe"; return 1; }
  "$NVCC" $flags -gencode arch=compute_100a,code=sm_100a -c "$probe.cu" -o "bin/$probe.o" \
    || { echo "FAILED (ptxas for sm_100a): $probe"; return 1; }
  if [[ "$native" == 1 ]]; then
    mkdir -p "bin-native/$(dirname "$probe")"
    "$NVCC" $flags -gencode arch=compute_100a,code=sm_100a --cudart shared "$probe.cu" -o "bin-native/$probe" \
      || { echo "FAILED (native program): $probe"; return 1; }
  fi
  echo "built $probe"
}
export -f build_one
export NVCC

status=0
printf '%s\n' "${probes[@]}" | xargs -P "$jobs" -I{} bash -c 'build_one "$1" "$2"' _ {} "$native" || status=$?
if [[ $status -ne 0 ]]; then
  echo "error: some probes did not build" >&2
  exit 1
fi
echo "built ${#probes[@]} probe(s) in $(pwd)/bin"
