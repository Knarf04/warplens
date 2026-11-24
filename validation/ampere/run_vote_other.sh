#!/usr/bin/env bash
# Record what the GPU does with the vote.sync launches that are not golden results: member
# masks that name a lane on another branch, and masks that leave out the caller (see
# vote_golden.cu). Some of them never return and some end with an error, so each runs in a
# process of its own under a timeout.
#
#   validation/ampere/run_vote_other.sh > validation/ampere/vote_other_3080ti.txt
#
# One line per launch: what vote_golden prints (mode, threads, roles, predicates, masks), then
# the results, or "no result after N s", or the CUDA error the launch ended with.
#
#   VOTE_TIMEOUT  seconds to wait for a launch (default 10; one that returns takes under 1)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
limit="${VOTE_TIMEOUT:-10}"
count="$(bin/vote_golden other)"
errors="$(mktemp)"
for ((k = 0; k < count; ++k)); do
  line="$(timeout "$limit" bin/vote_golden other "$k" 2> "$errors")"
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
