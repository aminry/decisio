#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# JevBench's 231 published items through its own harness (typesafe adapter) against a running decisio server, the
# harness's own summarize per file, then the v1.5 open-set reading (decisio.bench.jevbench_v15). The layout of
# runs/2026-09-30_served-default/jevbench/<arm>/. A private reading, not a board number (EVAL_CARD.md).
#   JEVBENCH=<harness checkout> bash benchmarks/jevbench.sh <server url> <out dir> [served model name]
# Needs the bench extra (uv sync --extra bench) and benchmarks/fetch_harnesses.sh.
set -euo pipefail
URL=${1:?server url, e.g. http://127.0.0.1:8000}; OUT=${2:?out dir}
NAME=${3:-decisio-qwen3.6-35b-a3b-letters}
JEVBENCH=${JEVBENCH:?the JevBench checkout (benchmarks/fetch_harnesses.sh)}
PY=${PY:-python}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
runs=()
for file in easy original hard; do
  D=$OUT/$file
  mkdir -p "$D"
  (cd "$JEVBENCH" && "$PY" -m jevbench.cli run --tasks "datasets/public/$file.jsonl" --adapter typesafe \
     --endpoint "$URL" --key-env '' --model "$NAME" --cost-basis internal_self_hosted \
     --reserve-usd 0 --run-label "decisio-$file" \
     --results "$D/results.jsonl" --raw-dir "$D/raw" --ledger "$D/ledger.jsonl" --manifest "$D/manifest.json") \
     > "$D/run.log" 2>&1 || { echo "JevBench $file run failed ($D/run.log)" >&2; exit 1; }
  (cd "$JEVBENCH" && "$PY" -m jevbench.cli summarize --tasks "datasets/public/$file.jsonl" --results "$D/results.jsonl" \
     --ledger "$D/ledger.jsonl" --public-export "$D/summary.json") > "$D/summarize.log" 2>&1 \
     || { echo "JevBench $file summarize failed" >&2; exit 1; }
  case $file in original) tier=standard ;; *) tier=$file ;; esac
  runs+=(--run "$tier=$JEVBENCH/datasets/public/$file.jsonl:$D/results.jsonl")
done
"$PY" -m decisio.bench.jevbench_v15 "${runs[@]}" --out "$OUT/v15.json"
