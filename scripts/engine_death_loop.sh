#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# The engine-death tests (tests/unit/test_engine_death.py) run RUNS times, PARALLEL at a time, beside a busy loop on
# every core (LOAD): their timing races show here, nightly, rather than in a rented card's set-up, where the first two
# surfaced. Each run's log and JUnit file go to OUT; the script fails if any run fails.
#   bash scripts/engine_death_loop.sh [runs] [parallel] [out dir]       (defaults 30, 2, engine-death-loop)
set -uo pipefail
RUNS=${1:-30}
PAR=${2:-2}
OUT=${3:-engine-death-loop}
LOAD=${LOAD:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
mkdir -p "$OUT"
busy=()
for _ in $(seq 1 "$LOAD"); do
  (yes > /dev/null) &
  busy+=($!)
done
trap 'kill "${busy[@]}" 2> /dev/null' EXIT
run() {
  if uv run --no-sync pytest -q tests/unit/test_engine_death.py -p no:cacheprovider --junitxml="$OUT/run_$1.xml" \
    > "$OUT/run_$1.log" 2>&1; then echo "$1 pass"; else echo "$1 FAIL"; fi
}
export -f run
export OUT
seq 1 "$RUNS" | xargs -P "$PAR" -I{} bash -c 'run {}' | sort -n > "$OUT/results.txt"
fails=$(grep -c FAIL "$OUT/results.txt")
grep -h "^FAILED" "$OUT"/run_*.log | sed 's/ - .*//' | sort | uniq -c
echo "$((RUNS - fails)) of $RUNS runs passed, $PAR at a time beside $LOAD busy loops"
[ "$fails" -eq 0 ]
