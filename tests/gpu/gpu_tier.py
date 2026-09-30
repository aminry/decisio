# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Helpers of the GPU tier. Every test in tests/gpu needs a CUDA card, vLLM 0.30.0 (the serve extra) and a local
checkpoint.

    DECISIO_MODEL=/models/Qwen3.6-35B-A3B-FP8 DECISIO_VIEW=/models/Qwen3.6-35B-A3B-FP8-text \
        uv run pytest -m gpu tests/gpu

DECISIO_MODEL is the official checkpoint (decisio's model classes load it); DECISIO_VIEW is the text-only view built by
`python -m decisio.serve.make_text_only` (the gates' engine); DECISIO_ADAPTER=name=/path adds the LoRA gate G4. vLLM
does not reliably release a card between engines in one process, so each test runs its engine in a child process.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def env_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set (see tests/gpu/gpu_tier.py)")
    return value


def run_child(*argv, timeout=3600):
    """Run a script of this directory in a child process with DeepGEMM off; return its output (failing on exit != 0)."""
    env = {**os.environ, "VLLM_USE_DEEP_GEMM": "0"}
    r = subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True, env=env, timeout=timeout)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    return out
