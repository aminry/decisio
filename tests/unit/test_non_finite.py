# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A NaN or infinity in an engine's probabilities is a clear 500, never an unexplained 400 (`decisio.serve.systemone`).

Gemma 4 E4B served under the 31B profile on the CPU stand-in answered every request with HTTP 400 "min() iterable
argument is empty": `top_index` found no option equal to the maximum of probabilities that held a NaN.

  F1  top_index and to_answer refuse a non-finite probability with NonFiniteReadout, for every question type
  F2  a request whose engine returns NaN is answered 500 with a detail that says what happened and names the served
      model and engine, not 400, and a finite engine is answered as before
  F3  registering a task from examples whose engine returns NaN is the same 500, and nothing is stored

    uv run pytest -q tests/unit/test_non_finite.py
"""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.systemone import NonFiniteReadout, SystemOne, add_routes, to_answer, top_index
from decisio.serve.tasks import TaskStore


class Q:
    def __init__(self, type, criteria=None):
        self.type, self.criteria = type, criteria or {"a": "x", "b": "y"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_f1_the_choice_and_the_answer_refuse_a_non_finite_probability(bad):
    with pytest.raises(NonFiniteReadout):
        top_index(["a", "b"], [bad, 0.5])
    with pytest.raises(NonFiniteReadout):
        top_index(["a", "b"], [0.5, bad])
    for q, p in ((Q("choice"), [bad, 0.5]), (Q("noul"), [bad, 0.5, 0.0]), (Q("score"), [0.2, bad])):
        with pytest.raises(NonFiniteReadout):
            to_answer(q, ["a", "b"], p)
    assert top_index(["b", "a"], [0.5, 0.5]) == 1  # the tie rule is unchanged


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 50000 for w in str(text).split()]


class Engine:
    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self, bad=None):
        self.bad = bad

    def facts(self):
        return {"engine": "fake", "base": "gemma-4-31b", "quantization": None, "mode": "separate", "vllm": "1.0"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, 1.0 / k)
            if self.bad is not None:
                p[0] = self.bad
            out.append(p)
        return out, {}


CRIT = {"option_0": "billing", "option_1": "shipping", "option_2": "access"}


def wire(text, kind="choice"):
    q = {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}
    if kind == "noul":
        q = {"type": "noul", "instructions": "Is it urgent?"}
    return {"state": {"message": text}, "model": "m", "questions": {"q1": q}}


def make(bad=None):
    so = SystemOne(Engine(bad), "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app, raise_server_exceptions=False), so


def test_f2_a_request_on_a_nan_engine_is_a_clear_500_that_names_the_model():
    client, _ = make()
    assert client.post("/v1/systemone", json=wire("hello")).status_code == 200
    for bad in (float("nan"), float("inf")):
        client, _ = make(bad)
        for kind in ("choice", "noul"):
            r = client.post("/v1/systemone", json=wire("hello", kind))
            assert r.status_code == 500, (bad, kind, r.status_code, r.text)
            d = r.json()["detail"]
            assert "not finite" in d and "NaN or infinity" in d and "decisio-test" in d and "gemma-4-31b" in d
            assert "not at fault" in d and "min()" not in d


def test_f3_registration_on_a_nan_engine_is_the_same_500_and_stores_nothing():
    client, so = make(float("nan"))
    examples = [{"request": wire(f"ticket {i}"), "answer": f"option_{i % 3}"} for i in range(30)]
    r = client.post("/v1/tasks", json={"id": "t", "examples": examples})
    assert r.status_code == 500 and "not finite" in r.json()["detail"], r.text
    assert not so.task_store.by_key
