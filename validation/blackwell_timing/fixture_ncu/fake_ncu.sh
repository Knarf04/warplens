#!/usr/bin/env bash
# A stand-in for `ncu`, for tests/test_ncu_scripts.py. NOT Nsight Compute and it measures
# nothing: it lets run_ncu.sh and compare_ncu.py be run from end to end on a machine where
# counters cannot be read (the development machine: no root, RmProfilingAdminOnly=1).
#
# What it prints is recorded from the real program where a recording exists (README.md
# here says which); the counter values it invents are all "1,234".
#
#   FAKE_NCU_MODE   ok           (default) run the program, then print a counter table
#                   permission   the recorded ERR_NVGPUCTRPERM output; exit status 1
#                   strict       like ok, but reject a list that names a counter which
#                                query_metrics_gb100.txt does not list (as ncu rejects a
#                                counter the GPU does not have)
#                   none         run the program and print no table
#   FAKE_NCU_LOG    append each command line to this file
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${FAKE_NCU_MODE:-ok}"
[[ -n "${FAKE_NCU_LOG:-}" ]] && echo "$*" >> "$FAKE_NCU_LOG"

metrics=""
csv=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      echo "NVIDIA (R) Nsight Compute Command Line Profiler"
      echo "Version 0.0.0.0 (stand-in, validation/blackwell_timing/fixture_ncu/fake_ncu.sh)"
      exit 0 ;;
    --query-metrics)
      shift
      if [[ "${1:-}" == --chips ]]; then
        [[ "$2" == gb100 ]] || { echo "==ERROR== Invalid chip $2 "; exit 1; }
      elif [[ "$mode" == permission ]]; then
        echo "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device."
        exit 0
      fi
      cat "$here/query_metrics_gb100.txt"
      exit 0 ;;
    --csv) csv=1 ;;
    --metrics) metrics="$2"; shift ;;
    --*) ;;
    *) break ;;
  esac
  shift
done
[[ $# -gt 0 && $csv == 1 ]] || { echo "fake_ncu: expected --csv --metrics LIST program [arguments]" >&2; exit 2; }
program="$1"

if [[ "$mode" == permission ]]; then
  cat "$here/permission_denied.txt"
  exit 1
fi
if [[ "$mode" == strict ]]; then
  IFS=, read -r -a names <<<"$metrics"
  for name in "${names[@]}"; do
    if ! grep -q "^${name%.*} " "$here/query_metrics_gb100.txt"; then
      echo "==ERROR== (stand-in) the counter $name is not available on this device"
      exit 1
    fi
  done
fi

echo "==PROF== Connected to process $$ ($program)"
"$@"
echo "==PROF== Disconnected from process $$"
[[ "$mode" == none ]] && { echo "==WARNING== No kernels were profiled."; exit 0; }
if [[ "$(basename "$program")" == transpose_hw && -n "${FAKE_NCU_TRANSPOSE_CSV:-}" ]]; then
  cat "$FAKE_NCU_TRANSPOSE_CSV"     # a recording of the real ncu for this very program
  exit 0
fi
echo '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'
IFS=, read -r -a names <<<"$metrics"
for name in "${names[@]}"; do
  echo "\"0\",\"$$\",\"$(basename "$program")\",\"127.0.0.1\",\"stand_in_kernel\",\"1\",\"7\",\"(32, 1, 1)\",\"(1, 1, 1)\",\"0\",\"10.0\",\"Command line profiler metrics\",\"$name\",\"\",\"1,234\""
done
