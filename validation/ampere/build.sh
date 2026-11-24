#!/usr/bin/env bash
# Build the hardware-validation programs for the local GPU, and the PTX the
# simulator runs for the same kernels.
#
#   HW_NVCC   nvcc with a full toolkit, for the GPU binaries (default: /usr/local/cuda/bin/nvcc)
#   HW_ARCH   architecture of the local GPU (default: sm_86, Ampere)
#   PTX_NVCC  nvcc that can target sm_100a (default: the one in the `warplens` conda env)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HW_NVCC="${HW_NVCC:-/usr/local/cuda/bin/nvcc}"
HW_ARCH="${HW_ARCH:-sm_86}"
PTX_NVCC="${PTX_NVCC:-conda run -n warplens nvcc}"

mkdir -p bin
"$HW_NVCC" -arch="$HW_ARCH" -o bin/shfl_golden shfl_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/transpose_hw transpose_hw.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/wavefront_probe wavefront_probe.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/barrier_exit_hw barrier_exit_hw.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/printf_hw printf_hw.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/tf32_golden tf32_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/ldmatrix_golden ldmatrix_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/cpasync_golden cpasync_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/ftz_golden ftz_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/call_golden call_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/callforms_golden callforms_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/prmt_golden prmt_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/minmax_cvt_golden minmax_cvt_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/atom_golden atom_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/bitfield_golden bitfield_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/mbarrier_token_golden mbarrier_token_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/nan_golden nan_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/intdiv_golden intdiv_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/cvt_ftz_golden cvt_ftz_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -o bin/vote_golden vote_golden.cu
# vote without .sync: ptxas refuses it for sm_70 and newer, so this one is built for sm_60 and
# the driver compiles its PTX for the GPU. CUDA 13 no longer has that target.
"$HW_NVCC" -arch=sm_60 -Wno-deprecated-gpu-targets -DVOTE_LEGACY -o bin/vote_legacy_golden vote_golden.cu
"$HW_NVCC" -arch="$HW_ARCH" -std=c++17 -o bin/launch_errors_hw launch_errors_hw.cu
"${CC:-cc}" -I"$(dirname "$(dirname "$(command -v "$HW_NVCC")")")/include" -o bin/driver_errors_hw driver_errors_hw.c -lcuda
$PTX_NVCC -arch=sm_100a -ptx shfl_probe.cu -o shfl_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx barrier_exit_hw.cu -o barrier_exit.ptx
$PTX_NVCC -arch=sm_100a -ptx printf_probe.cu -o printf_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx tf32_probe.cu -o tf32_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx ldmatrix_probe.cu -o ldmatrix_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx cpasync_probe.cu -o cpasync_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx call_probe.cu -o call_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx callforms_probe.cu -o callforms_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx atom_probe.cu -o atom_probe.ptx
# The same probe as the GPU binary's compiler spells it: CUDA 12.x writes no ordering or scope
# (atom.global.add.u32) and uses red where a result is discarded.
"$HW_NVCC" -arch="$HW_ARCH" -ptx atom_probe.cu -o atom_probe_cuda12.ptx
$PTX_NVCC -arch=sm_100a -ptx bitfield_probe.cu -o bitfield_probe.ptx
# vote: as nvcc 13.5 writes it, as the GPU binary's compiler does, and the forms without .sync
$PTX_NVCC -arch=sm_100a -ptx vote_probe.cu -o vote_probe.ptx
"$HW_NVCC" -arch="$HW_ARCH" -ptx vote_probe.cu -o vote_probe_cuda12.ptx
"$HW_NVCC" -arch=sm_60 -Wno-deprecated-gpu-targets -DVOTE_LEGACY -ptx vote_probe.cu -o vote_legacy_probe.ptx
$PTX_NVCC -arch=sm_100a -ptx mbarrier_token_probe.cu -o mbarrier_token_probe.ptx
echo "built bin/{shfl_golden,transpose_hw,barrier_exit_hw}, shfl_probe.ptx, barrier_exit.ptx"
