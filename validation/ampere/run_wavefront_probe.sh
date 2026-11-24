#!/usr/bin/env bash
# Profile bin/wavefront_probe with Nsight Compute and fit the wavefront offset
# (validation matrix TIM-17).
#
#   sudo env "PATH=$PATH" validation/ampere/run_wavefront_probe.sh
#
# ncu needs permission to read GPU performance counters (root on this machine,
# see validation/README.md). Writes, next to this script:
#   ncu_metrics_available.txt     what `ncu --query-metrics` lists on this GPU
#   wavefront_probe_configs.csv   one row per launch
#   wavefront_probe_ncu.csv       raw `ncu --csv` output
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NCU="${NCU:-ncu}"
PYTHON="${PYTHON:-conda run -n warplens python}"
HW_NVCC="${HW_NVCC:-/usr/local/cuda/bin/nvcc}"
HW_ARCH="${HW_ARCH:-sm_86}"

WANTED=(
  sm__ctas_launched
  smsp__warps_launched
  smsp__inst_executed
  smsp__inst_executed_op_shared_ld
  smsp__inst_executed_op_shared_st
  smsp__inst_executed_op_global_ld
  smsp__inst_executed_op_global_st
  smsp__inst_executed_op_local_ld
  smsp__inst_executed_op_local_st
  l1tex__data_pipe_lsu_wavefronts
  l1tex__data_pipe_lsu_wavefronts_mem_shared
  l1tex__data_pipe_lsu_wavefronts_mem_shared_ideal
  l1tex__data_pipe_lsu_wavefronts_mem_lg
  l1tex__data_bank_conflicts_pipe_lsu
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st
  l1tex__data_bank_conflicts_pipe_lsu_mem_global
)

own() { [[ -z "${SUDO_UID:-}" ]] || chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$@"; }

if [[ ! -x bin/wavefront_probe || wavefront_probe.cu -nt bin/wavefront_probe ]]; then
  mkdir -p bin
  "$HW_NVCC" -arch="$HW_ARCH" -o bin/wavefront_probe wavefront_probe.cu
  own bin/wavefront_probe
fi

# Ask only for the metrics this GPU and ncu version know: an unknown name makes
# ncu stop.
"$NCU" --query-metrics > ncu_metrics_available.txt 2>&1 || true
own ncu_metrics_available.txt
if grep -q ERR_NVGPUCTRPERM ncu_metrics_available.txt; then
  echo "ncu may not read performance counters (ERR_NVGPUCTRPERM). Run with sudo." >&2
  exit 2
fi
METRICS=()
for m in "${WANTED[@]}"; do
  if grep -qE "^${m}([[:space:]]|$)" ncu_metrics_available.txt; then
    METRICS+=("$m.sum")
  else
    echo "not available here, skipped: $m" >&2
  fi
done
[[ ${#METRICS[@]} -gt 0 ]] || { echo "no metric recognised; see ncu_metrics_available.txt" >&2; exit 2; }

"$NCU" --csv --metrics "$(IFS=,; echo "${METRICS[*]}")" \
  ./bin/wavefront_probe wavefront_probe_configs.csv > wavefront_probe_ncu.csv
own wavefront_probe_configs.csv wavefront_probe_ncu.csv

# As the invoking user, so that their Python (with numpy) is the one used.
if [[ -n "${SUDO_USER:-}" ]]; then
  sudo -u "$SUDO_USER" env "PATH=$PATH" $PYTHON wavefront_fit.py wavefront_probe_ncu.csv wavefront_probe_configs.csv
else
  $PYTHON wavefront_fit.py wavefront_probe_ncu.csv wavefront_probe_configs.csv
fi
