#!/usr/bin/env bash
# Fetch CUTLASS at a pinned commit and compile its first four Blackwell tutorials to PTX.
# CUTLASS sources and the PTX generated from them are not checked in.
#
#   conda run -n warplens examples/03_cutlass_mma/build.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
COMMIT=f8acf1a
if [[ ! -d .cutlass ]]; then
  git clone --quiet https://github.com/NVIDIA/cutlass.git .cutlass
fi
git -C .cutlass checkout --quiet "$COMMIT"
T=.cutlass/examples/cute/tutorial/blackwell
for pair in 01_mma_sm100:tutorial_01 02_mma_tma_sm100:tutorial_02 \
            03_mma_tma_multicast_sm100:tutorial_03 04_mma_tma_2sm_sm100:tutorial_04; do
  nvcc -arch=sm_100a -std=c++17 -O3 -DNDEBUG -w \
    -I.cutlass/include -I.cutlass/tools/util/include -I.cutlass/examples/common -I"$T" \
    -ptx "$T/${pair%%:*}.cu" -o "${pair##*:}.ptx"
  echo "built ${pair##*:}.ptx from CUTLASS $COMMIT"
done
