# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The card same-forward gate's tolerance by label source (tests/gpu/same_forward.py), without a card.

V1  Qwen (recompute) keeps 1e-4, Gemma (engine) is held to 1e-6: two reads of one row under like histories are one
    computation, and the 2026-10-04 cards measured 5.6e-17
V2  one differing option fails whatever the difference
V3  every base is covered: the sources the families declare are the sources the gate knows

  uv run pytest -q tests/unit/test_same_forward_verdict.py
"""

import importlib.util
from pathlib import Path

from decisio.families import FAMILIES

PATH = Path(__file__).resolve().parents[1] / "gpu" / "same_forward.py"
spec = importlib.util.spec_from_file_location("same_forward", PATH)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


def test_v1_tolerances_by_label_source():
    assert sf.verdict(3.1e-8, 60, 60, "recompute") == (True, 1e-4)
    assert sf.verdict(5e-5, 60, 60, "recompute")[0] is True
    assert sf.verdict(5.6e-17, 60, 60, "engine") == (True, 1e-6)
    assert sf.verdict(5.5e-4, 60, 60, "engine")[0] is False  # the 31B's first-against-cached difference is not a pass
    assert sf.verdict(6.8e-2, 60, 60, "recompute")[0] is False


def test_v2_a_different_option_fails():
    assert sf.verdict(0.0, 59, 60, "engine")[0] is False


def test_v3_every_declared_source_has_a_tolerance():
    assert {f.head_label_logprobs for f in FAMILIES} <= set(sf.TOL)
