#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# The two third-party harnesses the benchmark stage drives, at the commits runs/2026-09-30_served-default used, verified by
# commit (both MIT; neither is on PyPI). Prints the directory; point JEVBENCH and DECISION_INDEX at its subdirectories.
#   bash benchmarks/fetch_harnesses.sh [dir]          (default: $DECISIO_HARNESSES or ~/.cache/decisio/harnesses)
set -euo pipefail
DIR=${1:-${DECISIO_HARNESSES:-$HOME/.cache/decisio/harnesses}}
fetch() {   # fetch <name> <repository> <commit>
  local d=$DIR/$1
  [ -d "$d/.git" ] || git clone -q "$2" "$d"
  git -C "$d" checkout -q "$3"
  local got; got=$(git -C "$d" rev-parse HEAD)
  [ "${got:0:${#3}}" = "$3" ] || { echo "$d is at $got, expected $3" >&2; exit 1; }
}
mkdir -p "$DIR"
fetch jevbench https://github.com/fstandhartinger/jevbench.git bb05a335bc809e61b20c0f745d25499a82b326fc
fetch decision-index https://github.com/apolinario/decision-index.git 87d4650
echo "$DIR"
