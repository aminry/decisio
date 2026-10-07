# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""On each base's served profile, a later, different question about a 3,000-token state reads the state from the
prefix cache, and revisited states still do with the cache filled to 1.3 times its pool on the Gemma bases and to 0.9
times on the Qwen base, under its capacity (second_question_cached.py;
the boundary registration, decisio.families register_state_boundary). On the Gemma bases both orders run: after the
response (part C, a follow-up at once and after a pause reads the state from the cache) and before the question
(0.8.1's order). Part B is reported and not gated for the 31B under after, whose cache keeps fewer states in that order
(its default is before).

    DECISIO_MODEL=<Qwen checkpoint> [DECISIO_GEMMA_12B=<checkpoint>] [DECISIO_GEMMA_31B=<checkpoint>] \\
        uv run pytest -m gpu tests/gpu/test_second_question_cached.py

A base whose checkpoint variable is unset is skipped.
"""

import pytest
from gpu_tier import HERE, env_path, run_child

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("variable", "base", "register"),
    [
        ("DECISIO_MODEL", "qwen3.6-35b-a3b", "after"),
        ("DECISIO_GEMMA_12B", "gemma-4-12b", "after"),
        ("DECISIO_GEMMA_12B", "gemma-4-12b", "before"),
        ("DECISIO_GEMMA_31B", "gemma-4-31b", "after"),
        ("DECISIO_GEMMA_31B", "gemma-4-31b", "before"),
    ],
)
def test_a_later_question_reads_the_state_from_the_cache(variable, base, register):
    model = env_path(variable)
    part_b = "report" if (base, register) == ("gemma-4-31b", "after") else "gate"
    out = run_child(
        HERE / "second_question_cached.py",
        model,
        "--base",
        base,
        "--tokens",
        "3000",
        "--register-boundary",
        register,
        "--part-b",
        part_b,
    )
    assert "PART A PASS" in out, out[-4000:]
    if part_b == "gate":
        assert "PART B PASS" in out, out[-4000:]
    if register == "after" and base != "qwen3.6-35b-a3b":
        assert "PART C PASS" in out, out[-4000:]
