# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Exact probability ties (docs/handoffs/tasks.md, "Exact ties").

The declared choice is the most probable option; among options with exactly equal probabilities, the one whose key sorts
first (Unicode code-point order). So the order of the keys on the wire cannot change the choice: a client that
serialises with sorted keys (imajev's harness, `canonical_bytes`) gets the answer every other client gets.

  X1  the rule: in every order of the keys (sorted, reversed, all permutations), the same per-key probabilities give
      the same choice, and the old rule (the first maximum in wire order) would not
  X2  the abstention threshold's decision (`abstention.decide`) follows the same rule
  X3  served end to end: the same request with its keys in wire order, sorted (json.dumps(sort_keys=True)) and reversed
      gives the same answer, on an engine whose probabilities depend only on each option's text (so the only thing the
      key order can move is the tie-break)

    uv run pytest -q tests/unit/test_ties.py
"""

import itertools
import json

import numpy as np
import pytest

# an exact tie seen on a served image request, with a third option
TIED = {"Tomato Soup": 0.4846947710, "Iced Tea": 0.4846947710, "Coffee": 0.0306104580}


def question(crit):
    from decisio.serve.systemone import SystemOneRequest

    return SystemOneRequest.model_validate(
        {
            "state": {},
            "questions": {"q": {"type": "choice", "instructions": "Which drink is in the photo?", "criteria": crit}},
        }
    ).questions["q"]


def test_x1_the_rule():
    from decisio.serve.systemone import to_answer

    choices, old = set(), set()
    for order in itertools.permutations(TIED):
        crit = {k: None for k in order}
        keys, p = list(order), [TIED[k] for k in order]
        a = to_answer(question(crit), keys, p)
        choices.add(a["choice"])
        assert a["probabilities"] == {k: TIED[k] for k in order}  # the probabilities are untouched
        old.add(keys[int(np.argmax(p))])
    assert choices == {"Iced Tea"}  # the tied key that sorts first
    assert old == {"Iced Tea", "Tomato Soup"}  # the old rule depended on the order
    # no tie: the most probable option, whatever the keys
    a = to_answer(question({"b": None, "a": None}), ["b", "a"], [0.6, 0.4])
    assert a["choice"] == "b"
    # code-point order, not numeric: option_10 sorts before option_2
    a = to_answer(question({"option_2": "x", "option_10": "y"}), ["option_2", "option_10"], [0.5, 0.5])
    assert a["choice"] == "option_10"


def test_x2_abstention_follows_the_rule():
    from decisio.serve.abstention import decide

    keys = ["out of scope", "Tomato Soup", "Iced Tea"]
    p = [0.1, 0.45, 0.45]
    for perm in itertools.permutations(range(3)):
        ks, ps = [keys[i] for i in perm], [p[i] for i in perm]
        ab_idx = ks.index("out of scope")
        for cfg in (None, {"applied": True, "threshold": 0.5}):
            abstain, best = decide(ps, ab_idx, cfg, keys=ks)
            assert not abstain and ks[best] == "Iced Tea"
    # a tie between the abstain option and an answer: the key that sorts first decides, in any order
    for perm in itertools.permutations(range(2)):
        ks, ps = [["Iced Tea", "can't tell"][i] for i in perm], [[0.5, 0.5][i] for i in perm]
        abstain, _ = decide(ps, ks.index("can't tell"), None, keys=ks)
        assert abstain is False  # "Iced Tea" sorts before "can't tell" (code points: capitals first)


class OrderFreeEngine:
    """Probabilities that depend only on each option's text, never on its position: the key order can move only the
    tie-break."""

    def __init__(self, weights):
        from transformers import AutoTokenizer

        self.weights, self.adapters = weights, {}
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    def answer(self, state, questions, adapter=None):
        out = []
        for q in questions:
            w = np.array([self.weights[o] for o in q["options"]], dtype=np.float64)
            out.append(w / w.sum())
        return out, {}

    def facts(self):
        return {"engine": "order-free stand-in (tests)"}


@pytest.mark.slow
def test_x3_sorted_keys_on_the_wire_cannot_change_the_answer():
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    eng = OrderFreeEngine(TIED)
    client = TestClient(make_app(eng, SystemOne(eng, "decisio-test")))
    crit = {"Tomato Soup": None, "Iced Tea": None, "Coffee": None}  # the record's own order
    body = {
        "state": {"photo": "[image 1]"},
        "model": "m",
        "questions": {"q": {"type": "choice", "instructions": "Which drink is in the photo?", "criteria": crit}},
    }
    wires = {
        "as sent": json.dumps(body),
        "sorted keys": json.dumps(body, sort_keys=True),  # imajev's canonical_bytes
        "reversed": json.dumps(
            {**body, "questions": {"q": {**body["questions"]["q"], "criteria": dict(reversed(list(crit.items())))}}}
        ),
    }
    answers = {}
    for name, raw in wires.items():
        r = client.post("/v1/systemone", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        answers[name] = r.json()["answers"]["q"]
    assert {a["choice"] for a in answers.values()} == {"Iced Tea"}
    for a in answers.values():  # same probabilities per key, any order
        assert {k: round(v, 12) for k, v in a["probabilities"].items()} == {
            k: round(v, 12) for k, v in answers["as sent"]["probabilities"].items()
        }
