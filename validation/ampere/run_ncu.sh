#!/usr/bin/env bash
# Profile the transpose kernels on the local GPU with Nsight Compute and compare
# the shared-memory counters with what warplens predicts for the same launch.
#
#   validation/ampere/run_ncu.sh [ctas]
#
# ncu needs permission to read GPU performance counters. If it reports
# ERR_NVGPUCTRPERM, run this script with sudo (see validation/README.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CTAS="${1:-64}"
NCU="${NCU:-ncu}"
PYTHON="${PYTHON:-conda run -n warplens python}"

METRICS=(
  smsp__inst_executed_op_shared_ld.sum
  smsp__inst_executed_op_shared_st.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
[[ -x bin/transpose_hw ]] || ./build.sh
"$NCU" --csv --metrics "$(IFS=,; echo "${METRICS[*]}")" ./bin/transpose_hw "$CTAS" > ncu_transpose.csv
# Under sudo, hand the output back to the invoking user.
[[ -z "${SUDO_UID:-}" ]] || chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" ncu_transpose.csv
$PYTHON compare_ncu.py ncu_transpose.csv "$CTAS"
