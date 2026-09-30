# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The letters engine's serving gates on a card (serving_gates.py): G1 cache hit and cache effect, G2 isolation, G4 the
LoRA adapter applied and the base restored bit-exactly (with DECISIO_ADAPTER), then G3 packed against separate.
The tolerances are the ones every run in runs/ used (6.46e-2: twice the batch noise measured on this card class).

    DECISIO_VIEW=/models/Qwen3.6-35B-A3B-FP8-text uv run pytest -m gpu tests/gpu/test_serving_gates.py
"""
import json
import os

import pytest
from gpu_tier import HERE, env_path, run_child

pytestmark = pytest.mark.gpu
TOL = "0.0646"
ENGINE = json.dumps({"compilation_config": {"max_cudagraph_capture_size": 4096}})


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    return tmp_path_factory.mktemp("gates") / "separate.json"


def test_g1_g2_g4_separate(handoff):
    view = env_path("DECISIO_VIEW")
    adapter = os.environ.get("DECISIO_ADAPTER")
    out = run_child(HERE / "serving_gates.py", "--gpu", view, "--tol-g1", TOL, "--tol-g2", TOL, "--pad-where", "front",
                    "--engine", ENGINE, "--handoff", handoff, *(["--adapter", adapter] if adapter else []))
    assert "GPU GATES separate PASS" in out, out[-4000:]


def test_g3_packed(handoff):
    view = env_path("DECISIO_VIEW")
    if not handoff.exists():
        pytest.skip("needs the separate part's answers (test_g1_g2_g4_separate)")
    out = run_child(HERE / "serving_gates.py", "--gpu", view, "--part", "packed", "--handoff", handoff)
    assert "GPU GATES packed PASS" in out, out[-4000:]
