#!/usr/bin/env bash
# Turn the failure summary of pytest.log into a few short annotations, so that what failed
# can be read on the run's page (and through the API) without downloading the log.
#
#   .github/report_pytest_failure.sh [LOG]      # default: pytest.log
#
# The tail of the log is included because a crashed interpreter leaves no summary lines to
# match. Never fails: it runs after a failure and must not hide it behind its own.
log="${1:-pytest.log}"
[[ -f "$log" ]] || { echo "::error title=pytest::no $log: pytest did not start"; exit 0; }
work="$(mktemp -d)"
echo "::notice title=pytest log::$(wc -l < "$log") lines"
# Which tests, first; then why. A job that may skip lists hundreds of SKIPPED lines with -rs.
{ grep -E '^(FAILED|ERROR|Fatal Python error|WARPLENS_NO_SKIPS|[0-9]+ (passed|failed))' "$log" | head -15
  grep -E '^(SKIPPED|E  |  tests/)' "$log" | head -15
  echo "--- last lines of the log ---"; tail -20 "$log"; } \
  | tr -cd '\11\12\40-\176' | cut -c1-200 | sed 's/%/%25/g; s/::/: :/g' > "$work/summary.txt"
split -l 10 "$work/summary.txt" "$work/summary.part."
for part in "$work"/summary.part.*; do
  echo "::error title=pytest ${part##*.}::$(awk '{printf "%s%%0A", $0}' "$part")"
done
rm -rf "$work"
exit 0
