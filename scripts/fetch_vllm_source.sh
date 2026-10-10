#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# The vLLM source tree the patch series are checked against (tests/unit/test_patches.py): a shallow checkout of the tag,
# verified by commit. Not an installation: vLLM has no macOS wheels, and the check needs only the files.
#   bash scripts/fetch_vllm_source.sh [dir]          (default: $VLLM_SRC or ~/.cache/decisio/vllm-v0.31.0)
set -euo pipefail
TAG=v0.31.0; COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6
DIR=${1:-${VLLM_SRC:-$HOME/.cache/decisio/vllm-$TAG}}
mkdir -p "$(dirname "$DIR")"
[ -d "$DIR/.git" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/vllm-project/vllm.git "$DIR"
got=$(git -C "$DIR" rev-parse HEAD)
[ "$got" = "$COMMIT" ] || { echo "$DIR is at $got, expected $COMMIT ($TAG)"; exit 1; }
echo "$DIR"
