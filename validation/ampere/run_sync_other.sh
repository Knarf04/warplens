#!/usr/bin/env bash
# Record what the GPU does with shfl.sync and bar.warp.sync when the member mask names lanes
# that are elsewhere (sync_other_probe.cu). Each launch runs in a process of its own under a
# timeout, as a vote.sync of this kind never returns (run_vote_other.sh).
#
#   validation/ampere/run_sync_other.sh > validation/ampere/sync_other_3080ti.txt
#
# One line per launch: the case, then what the threads got, or "no result after N s", or the
# CUDA error the launch ended with.
#
#   SYNC_TIMEOUT  seconds to wait for a launch (default 10; one that returns takes under 1)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
limit="${SYNC_TIMEOUT:-10}"
count="$(bin/sync_other)"
errors="$(mktemp)"
for ((k = 0; k < count; ++k)); do
  line="$(timeout "$limit" bin/sync_other "$k" 2> "$errors")"
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "$line"
  elif [[ $status -eq 124 ]]; then
    echo "$line no result after $limit s"
  else
    echo "$line $(sed -n 's/.*failed: \([A-Za-z]*\).*/\1/p' "$errors" | head -n 1)"
  fi
done
rm -f "$errors"
