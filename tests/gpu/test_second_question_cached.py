# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""On each base's served profile, a later, different question about a 3,000-token state reads the state from the
prefix cache, and revisited states still do with the cache filled to 1.3 times its pool (second_question_cached.py;
the boundary registration, decisio.families register_state_boundary).

    DECISIO_MODEL=<Qwen checkpoint> [DECISIO_GEMMA_12B=<checkpoint>] [DECISIO_GEMMA_31B=<checkpoint>] \\
        uv run pytest -m gpu tests/gpu/test_second_question_cached.py

A base whose checkpoint variable is unset is skipped.
"""

import pytest
from gpu_tier import HERE, env_path, run_child

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("variable", "base"),
    [("DECISIO_MODEL", "qwen3.6-35b-a3b"), ("DECISIO_GEMMA_12B", "gemma-4-12b"), ("DECISIO_GEMMA_31B", "gemma-4-31b")],
)
def test_a_later_question_reads_the_state_from_the_cache(variable, base):
    out = run_child(HERE / "second_question_cached.py", env_path(variable), "--base", base, "--tokens", "3000")
    assert "PART A PASS" in out and "PART B PASS" in out, out[-4000:]
