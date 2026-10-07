#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# Four Decision Index 0.2.1 benchmarks (BANKING77, CLINC150+OOS, GPQA Diamond, MMLU-Pro; 20,810 requests) through the
# kit's own http engine against a running decisio server, one request at a time, the kit's own score, then the four
# benchmark values by the kit's index functions (decisio.bench.di_report) and calibration per benchmark
# (decisio.bench.di_cal). The layout of runs/2026-09-30_served-default/decision_index/<arm>/.
#   DECISION_INDEX=<kit checkout> bash benchmarks/decision_index.sh <server url> <out dir> <suite dir> [model name]
#   (LIMIT=<n>: only the first n requests, for a smoke run; the full stage is 20,810)
# <suite dir>: the kit's rebuilt suite (python -m decision_index suite rebuild --only 4 5 21 25 57, then its import);
# the four benchmarks' rows are taken from it by decisio.bench.di_rows, which checks their counts and hash.
set -euo pipefail
URL=${1:?server url}; OUT=${2:?out dir}; SUITE=${3:?the kit suite dir}
NAME=${4:-decisio-gemma-4-31b-it-letters}  # the default base's served name; pass the other bases' (docs/cli.md)
DECISION_INDEX=${DECISION_INDEX:?the Decision Index kit checkout (benchmarks/fetch_harnesses.sh)}
PY=${PY:-python}
export PYTHONPATH=$DECISION_INDEX${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"
ROWS=$OUT/rows.jsonl.gz
"$PY" -m decisio.bench.di_rows "$SUITE" "$ROWS"
"$PY" -m decision_index run --engine http --option "base_url=$URL" --option "model=$NAME" --rows "$ROWS" --out "$OUT" \
  ${LIMIT:+--limit "$LIMIT"} \
  > "$OUT/run.json" 2> "$OUT/run.log" || { echo "Decision Index run failed ($OUT/run.log)" >&2; exit 1; }
# the kit's score ends in a traceback at its 38-benchmark index step (the benchmarks not run have no tracks); the
# output that counts is benchmark-summary.json, written before it
"$PY" -m decision_index score --results "$OUT/results.jsonl" --suite-dir "$SUITE" --edition 0.2.1 --out "$OUT" \
  > "$OUT/score.json" 2> "$OUT/score_stderr.txt" || true
[ -s "$OUT/benchmark-summary.json" ] || { echo "no benchmark-summary.json ($OUT/score_stderr.txt)" >&2; exit 1; }
"$PY" -m decisio.bench.di_report "$OUT" --suite-dir "$SUITE"
"$PY" -m decisio.bench.di_cal "$OUT" --rows "$ROWS"
