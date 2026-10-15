# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Abstention tasks that match by option keys (`POST /v1/abstention/tasks`, match `option_keys`). A fake engine.

Multiple-choice items have their own four candidate answers behind the keys A to D and an `idk` key, so no two items
share an option set and an `option_set` task cannot be registered (422), where a threshold on p(idk) is still one
threshold.

  K1  option_set refuses items with different texts and says what to use; option_keys registers them
  K2  the task applies to a question with the same keys and other texts, and not to one with other keys
  K3  examples with different key sets are refused under option_keys; an exact option_set task wins over option_keys

    uv run pytest -q tests/unit/test_abstention_keys.py
"""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.systemone import SystemOne, add_routes


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 50000 for w in str(text).split()]


KEYS = ["A", "B", "C", "D", "idk"]


class Engine:
    """The state's message is `top:conf:idk`: the engine puts `conf` on option `top`, `idk` on the last option and
    splits the rest."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def facts(self):
        return {"engine": "fake", "base": "fake-base", "quantization": None, "mode": "separate", "vllm": "1.0"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        top, conf, idk = str(state["message"]).split(":")[:3]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, 0.0)
            p[int(top)] = float(conf)
            if int(top) != k - 1:
                p[k - 1] = float(idk)
            rest = [i for i in range(k) if p[i] == 0.0]
            p[rest] = (1.0 - p.sum()) / max(len(rest), 1)
            out.append(p)
        return out, {}


def wire(msg, item, keys=KEYS):
    crit = {k: f"answer {k} of item {item}" for k in keys}  # texts that differ from item to item
    return {
        "state": {"message": msg},
        "model": "m",
        "questions": {"q": {"type": "choice", "instructions": f"Item {item}: which is right?", "criteria": crit}},
    }


def examples(n=40, keys=KEYS):
    out = []
    rng = np.random.default_rng(0)
    for i in range(n):
        gold = int(rng.integers(0, 4))
        if i % 3 == 0:  # unanswerable: the model answers anyway, with a modest p(idk) that a threshold can use
            out.append({"request": wire(f"{gold}:0.45:0.30", i, keys), "answer": None})
        else:  # answerable: right, with a small p(idk)
            out.append({"request": wire(f"{gold}:0.7:0.05", i, keys), "answer": keys[gold]})
    return out


def make():
    so = SystemOne(Engine(), "decisio-test")
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app, raise_server_exceptions=False), so


def register(client, match, ex, id="exam"):
    return client.post(
        "/v1/abstention/tasks", json={"id": id, "option": {"key": "idk"}, "match": match, "examples": ex}
    )


def test_k1_option_set_refuses_item_specific_texts_and_option_keys_takes_them():
    client, _ = make()
    r = register(client, "option_set", examples())
    assert r.status_code == 422 and "option_keys" in r.json()["detail"]
    r = register(client, "option_keys", examples())
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["match"].get("option_keys") and t["config"]["n"] == 40 and t["config"]["applied"], t["config"]
    assert "exam" in {x["id"] for x in client.get("/v1/abstention/tasks").json()["tasks"]}


def test_k2_it_applies_to_other_texts_with_the_same_keys_and_not_to_other_keys():
    client, so = make()
    assert register(client, "option_keys", examples()).status_code == 200
    thr = so.tasks["exam"]["config"]["threshold"]
    assert 0.05 < thr < 0.30
    # an unseen item (other texts), the model answers A with p(idk) 0.30 above the threshold: abstain
    a = client.post("/v1/systemone", json=wire("0:0.45:0.30", 999)).json()["answers"]["q"]
    assert a["choice"] == "idk"
    b = client.post("/v1/systemone", json=wire("0:0.7:0.05", 998)).json()["answers"]["q"]
    assert b["choice"] == "A"
    # five candidates plus idk: another key set, outside the task: the plain argmax
    six = ["A", "B", "C", "D", "E", "idk"]
    c = client.post("/v1/systemone", json=wire("0:0.45:0.30", 997, six)).json()["answers"]["q"]
    assert c["choice"] == "A"


def test_k3_mixed_key_sets_are_refused_and_an_exact_option_set_task_wins():
    client, so = make()
    mixed = examples(20) + examples(20, ["A", "B", "C", "idk"])
    r = register(client, "option_keys", mixed)
    assert r.status_code == 422 and "share one set of option keys" in r.json()["detail"]
    assert register(client, "option_keys", examples()).status_code == 200
    same = [{"request": wire(e["request"]["state"]["message"], 7), "answer": e["answer"]} for e in examples()]
    assert (
        register(client, "option_set", same, id="one-item").status_code == 200
    )  # one option list: every example is item 7
    from decisio.serve.systemone import SystemOneRequest

    req = SystemOneRequest.model_validate(wire("0:0.45:0.30", 7))
    assert so.task_for(req.questions["q"], False)["id"] == "one-item"
    req = SystemOneRequest.model_validate(wire("0:0.45:0.30", 8))
    assert so.task_for(req.questions["q"], False)["id"] == "exam"
