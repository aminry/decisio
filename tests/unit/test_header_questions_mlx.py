# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A declared question list on the MLX prefix cache (`"header_questions"`; decisio.serve.header_questions).

Q1  two requests about one state with different single questions: without a declared list the second reads the state
    again; with one list on both, the second is a prefix-cache hit

Its own module: the stand-in row text of test_header_questions.py (an autouse fixture) would change the tiny model's
rows.

  uv run pytest -q tests/unit/test_header_questions_mlx.py
"""

from test_mlx_engine import tiny  # noqa: F401  (the tiny model of the MLX engine's tests)

from decisio.serve import header_questions as hq
from decisio.serve import layout as lay

LINES = [
    "Which queue should this go to?",
    "Is the customer angry?",
    "Does the message ask for a refund?",
]
STATE = " ".join(f"word{i}" for i in range(300))


def test_q1_mlx_the_second_request_about_the_state_is_a_cache_hit(tiny):  # noqa: F811
    with lay.using("header"):
        for declared in (None, tuple(LINES)):
            tiny.prefix_cache.entries.clear()
            hits = []
            for i in (0, 1):
                qs = hq.stamp([{"kind": "noul", "instructions": LINES[i]}], declared)
                _, info = tiny.answer(STATE, qs)
                hits.append(info["prefix_cache_hits"])
            if declared is None:
                assert hits == [0, 0]  # the old behaviour: another question, another header, the state read again
            else:
                assert hits == [0, 1]
