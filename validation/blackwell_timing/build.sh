#!/usr/bin/env bash
# Build the timing microbenchmarks.
#
#   ./build.sh [hw] [sim] [ampere]        default: hw sim
#
#   hw      bin/hw/<bench>      machine code for sm_100a, kernels read %clock64. For the B200.
#   sim     bin/sim/<bench>     PTX for compute_100a, shared CUDA runtime, no %clock64.
#                               For `warplens run --timing` (predict.py).
#   ampere  bin/ampere/<bench>  the benchmarks that are valid on Ampere, for the local GPU.
#
#   BENCHES   build only these (space-separated), e.g. BENCHES="alu_chain tma" ./build.sh sim
#   NVCC      compiler for hw and sim; must know sm_100a (CUDA 12.8 or newer).
#             Default: the one in the `warplens` conda env. On the rented machine: NVCC=nvcc
#   HW_NVCC   compiler for ampere (default /usr/local/cuda/bin/nvcc), HW_ARCH its target (sm_86)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NVCC="${NVCC:-conda run -n warplens nvcc}"
HW_NVCC="${HW_NVCC:-/usr/local/cuda/bin/nvcc}"
HW_ARCH="${HW_ARCH:-sm_86}"
ALL=(launch alu_chain smem_wavefronts global_load gstream cp_async mbar tma mma_rate tmem_access)
AMPERE=(launch alu_chain smem_wavefronts global_load gstream cp_async)   # no mbarrier, TMA or tcgen05 before sm_90 / sm_100
FLAGS=(-O3 -std=c++17 -w)
if [[ -n "${BENCHES:-}" ]]; then
  read -r -a wanted <<< "$BENCHES"
  ALL=("${wanted[@]}")
  kept=()
  for b in "${wanted[@]}"; do [[ " ${AMPERE[*]} " == *" $b "* ]] && kept+=("$b"); done
  AMPERE=("${kept[@]}")
fi

targets=("$@")
[[ ${#targets[@]} -gt 0 ]] || targets=(hw sim)
for target in "${targets[@]}"; do
  case "$target" in
    hw)
      mkdir -p bin/hw
      for b in "${ALL[@]}"; do
        $NVCC "${FLAGS[@]}" -gencode arch=compute_100a,code=sm_100a -DBENCH_CLOCK=1 "$b.cu" -o "bin/hw/$b"
        echo "built bin/hw/$b"
      done ;;
    sim)
      mkdir -p bin/sim
      for b in "${ALL[@]}"; do
        $NVCC "${FLAGS[@]}" -gencode arch=compute_100a,code=compute_100a --no-compress --cudart shared \
          -DBENCH_CLOCK=0 "$b.cu" -o "bin/sim/$b"
        echo "built bin/sim/$b"
      done ;;
    ampere)
      mkdir -p bin/ampere
      for b in "${AMPERE[@]}"; do
        "$HW_NVCC" "${FLAGS[@]}" -arch="$HW_ARCH" -DBENCH_CLOCK=1 "$b.cu" -o "bin/ampere/$b"
        echo "built bin/ampere/$b"
      done ;;
    *) echo "unknown target: $target (hw, sim or ampere)" >&2; exit 2 ;;
  esac
done
