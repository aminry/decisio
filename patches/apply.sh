#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# Apply a patch series to an installed vLLM (the `pkg/` form, paths under vllm/), as an image build or a set-up script
# does: every file is dry-run first and the script stops at the first one that does not apply cleanly.
#   bash patches/apply.sh [python] [series dir]
#   (defaults: python3 on PATH; patches/vllm-0.31.0/suffix-staging)
# Revert with `patch -p1 -R` in the reverse order. The series is inert until VLLM_SUFFIX_STAGING=1.
set -euo pipefail
PY=${1:-python3}
SERIES=${2:-$(cd "$(dirname "$0")" && pwd)/vllm-0.31.0/suffix-staging}
want=$(basename "$(dirname "$SERIES")"); want=${want#vllm-}
got=$("$PY" -c 'import vllm; print(vllm.__version__)')
[ "$got" = "$want" ] || { echo "the series is for vllm $want; $PY has vllm $got" >&2; exit 1; }
site=$("$PY" -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
for p in "$SERIES"/pkg/0*.patch; do
  (cd "$site" && patch -p1 --forward --dry-run < "$p" > /dev/null) || { echo "$p does not apply to vllm $got" >&2; exit 1; }
  (cd "$site" && patch -p1 --forward < "$p" > /dev/null)
  echo "applied $(basename "$p")"
done
