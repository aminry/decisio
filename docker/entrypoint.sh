#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# The image's entry point: check that a GPU is visible, then start the server with the served-default flags.
# Extra arguments are passed to the server after the defaults, so they win (argparse keeps the last value):
#   docker run ... ghcr.io/aminry/decisio --image-model Qwen/Qwen3.6-35B-A3B-FP8
# Environment: DECISIO_BASE (the base: gemma-4-31b by default, or qwen3.6-35b-a3b, gemma-4-12b, or a decisio repository),
# DECISIO_MODEL (a checkpoint instead, a Hugging Face id fetched into /data/hf on the first start, or a path; with it
# and no DECISIO_BASE the base is the one the checkpoint declares), DECISIO_HOST, DECISIO_PORT.
set -euo pipefail

python3 - <<'PY'
import sys
import warnings

warnings.simplefilter("ignore")
reason = ""
try:
    import torch

    count = torch.cuda.device_count()
    if count:
        for i in range(count):
            p = torch.cuda.get_device_properties(i)
            print(f"decisio: GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB", flush=True)
            if p.total_memory < 90 * 2**30:
                print(
                    "decisio: warning: the record was measured on a 96 GB card; with less memory the engine "
                    "may fail to start (see EVAL_CARD.md)",
                    file=sys.stderr,
                    flush=True,
                )
        sys.exit(0)
except Exception as e:  # no driver, a driver that is too old, a broken install
    reason = f" ({type(e).__name__}: {e})"
print(
    "decisio: no NVIDIA GPU is visible to this container" + reason + ".\n"
    "  Start it with GPU access: `docker run --gpus all ...`, or the GPU reservation in compose.yaml\n"
    "  (Docker with the NVIDIA Container Toolkit on a Linux host).\n"
    "  The host driver must support CUDA 13.0. Nothing was started.",
    file=sys.stderr,
)
sys.exit(1)
PY

# the base: DECISIO_BASE, else (no checkpoint named) the default, gemma-4-31b; a checkpoint alone names its own base
selection=()
if [ -n "${DECISIO_BASE:-}" ]; then selection+=(--base "$DECISIO_BASE"); fi
if [ -n "${DECISIO_MODEL:-}" ]; then selection+=(--model "$DECISIO_MODEL"); fi
if [ "${#selection[@]}" -eq 0 ]; then selection=(--base gemma-4-31b); fi

exec python3 -m decisio.serve.vllm_engine \
  "${selection[@]}" \
  --model-class hidden-readout \
  --host "${DECISIO_HOST:-0.0.0.0}" \
  --port "${DECISIO_PORT:-8000}" \
  "$@"
