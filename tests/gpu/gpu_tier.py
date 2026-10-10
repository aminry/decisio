# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Helpers of the GPU tier. Every test in tests/gpu needs a CUDA card, vLLM 0.31.0 (the serve extra) and a local
checkpoint.

    DECISIO_MODEL=<checkpoint dir> DECISIO_VIEW=<text-only view dir> uv run pytest -m gpu tests/gpu

DECISIO_MODEL is the official checkpoint (decisio's model classes load it); DECISIO_VIEW is the text-only view built by
`python -m decisio.serve.make_text_only` (the gates' engine); DECISIO_ADAPTER=name=/path adds the LoRA gate G4;
DECISIO_GEMMA_12B and DECISIO_GEMMA_31B (checkpoint dirs) add the Gemma bases to test_second_question_cached.py. vLLM
does not reliably release a card between engines in one process, so each test runs its engine in a child process.
"""

import os
import re
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


VERDICT = re.compile(r"PASS|FAIL|^\s*ok\b|^[A-Z][A-Z ]+[:(]")


def failure_report(r):
    """What a failed child said: its exit code (a negative one is the signal), the script's verdict lines, and the tails
    of stdout and stderr apart. The engine's start-up output fills both streams, so one tail of the two joined showed
    only CUDA graph capture on the first card run, hiding the verdicts (Lab 2, 2026-10-07)."""
    verdicts = [line for line in r.stdout.splitlines() if VERDICT.search(line)]
    return "\n".join(
        [f"exit code {r.returncode}", "verdicts:", *verdicts[-40:], "stdout tail:", r.stdout[-1500:], "stderr tail:"]
        + [r.stderr[-1500:]]
    )


def run_child(*argv, timeout=3600):
    """Run a script of this directory in a child process with DeepGEMM off; return its output (failing on exit != 0)."""
    env = {**os.environ, "VLLM_USE_DEEP_GEMM": "0"}
    r = subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True, env=env, timeout=timeout)
    assert r.returncode == 0, failure_report(r)
    return r.stdout + r.stderr
