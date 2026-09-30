# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio's model classes on a card: the hidden-readout class's same-forward gate (same_forward.py). The suite identity
of the text-only class against the view, and of the hidden-readout class's plain answers, is recorded in
runs/2026-09-30_plugin-verification (1,400 of 1,400 bit-identical); it needs a private item suite and is not repeated
here.

    DECISIO_MODEL=$DECISIO_MODEL uv run pytest -m gpu tests/gpu/test_model_classes.py
"""

import pytest
from gpu_tier import HERE, env_path, run_child

pytestmark = pytest.mark.gpu


def test_hidden_readout_same_forward():
    out = run_child(HERE / "same_forward.py", "--model", env_path("DECISIO_MODEL"))
    assert "SAME_FORWARD PASS" in out, out[-4000:]
