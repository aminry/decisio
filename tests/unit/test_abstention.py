# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-task abstention (`decisio.serve.abstention`, the `/v1/systemone` path and `POST /v1/abstention/tasks`).

  A1  the threshold rule: candidates between consecutive probabilities (log-odds midpoints), the best threshold gets
      the most examples right and ties go to the largest (fewest abstentions);
  A2  the acceptance rule: too few examples or too few of either kind, or no leave-one-out gain over the plain
      behaviour, or more false abstentions: no threshold; separable examples with a gain: a threshold between the two
      groups;
  A3  the decision: with a threshold, abstain when the option's probability exceeds it, otherwise the best other option;
      without, the argmax;
  A4  served (CPU stand-in): a declared option's task changes only `choice`, never the probabilities; an appended option
      is answered through imajev's fields only when the threshold was accepted, and a declined task leaves the request
      as it was; requests outside every task are unchanged; the endpoint scores and stores a task.

    uv run pytest -q tests/unit/test_abstention.py
"""
import pytest

from decisio.serve.abstention import best_threshold, candidates, decide, fit


def test_a1_candidates_and_ties():
    c = candidates([0.01, 0.2, 0.9])
    assert c == sorted(c) and c[-1] == 1.0 and 0.01 < c[1] < 0.2 < c[2] < 0.9 and c[0] < 0.01
    # separable: two unanswerable examples above three answerable ones
    t = best_threshold([0.01, 0.02, 0.05, 0.6, 0.8], [False, False, False, True, True], [True] * 3 + [False] * 2)
    assert 0.05 < t < 0.6
    # nothing to gain from abstaining: never abstain (1.0), the largest of the tied candidates
    assert best_threshold([0.1, 0.2, 0.3], [False] * 3, [True] * 3) == 1.0


def test_a2_acceptance_rule():
    p = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.3, 0.5, 0.7, 0.9]
    u = [False] * 8 + [True] * 4
    ok = [True] * 8 + [False] * 4
    # plain never abstains (an appended option): 8 right, 0 false
    cfg = fit(p, u, ok, [True] * 8 + [False] * 4, [False] * 12)
    assert cfg["applied"] and 0.008 < cfg["threshold"] < 0.3 and cfg["loo_right_threshold"] == 12
    # the plain behaviour already abstains right: no gain, kept plain
    cfg = fit(p, u, ok, [True] * 12, [False] * 12)
    assert not cfg["applied"] and cfg["threshold"] is None
    # too few examples, too few unanswerable ones
    assert not fit(p[:9], u[:9], ok[:9], [True] * 9, [False] * 9)["applied"]
    assert not fit(p[:9] + [0.9], u[:9] + [True], ok[:10], [True] * 10, [False] * 10)["applied"]


def test_a3_decide():
    p = [0.5, 0.3, 0.2]                        # the abstain option at index 2
    assert decide(p, 2, {"applied": True, "threshold": 0.1}) == (True, 0)
    assert decide(p, 2, {"applied": True, "threshold": 0.25}) == (False, 0)
    assert decide(p, 2, None) == (False, 0)
    assert decide([0.2, 0.3, 0.5], 2, {"applied": False}) == (True, 1)


MODEL = "Qwen/Qwen3-0.6B-Base"
CRIT = {f"option_{i}": n for i, n in enumerate(["card_arrival", "card_linking", "exchange_rate", "out of scope"])}


def req(utt, crit=CRIT, **extra):
    return {"state": {}, "model": "m", "questions": {"q1": {
        "type": "choice", "instructions": f"Classify the intent of this user request:\n{utt}",
        "criteria": crit}}, **extra}


@pytest.fixture(scope="module")
def served():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app
    eng = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(eng, "decisio-test")
    return TestClient(make_app(eng, so)), so


def test_a4_declared_option(served):
    from decisio.serve.systemone import SystemOneRequest, option_set
    client, so = served
    before = client.post("/v1/systemone", json=req("where is my card?")).json()["answers"]["q1"]
    fp = option_set(SystemOneRequest.model_validate(req("x")).questions["q1"])
    for t, want in ((0.0, "option_3"), (1.0, None)):
        so.tasks = {"t": {"id": "t", "option": {"key": "option_3"}, "match": {"option_set": fp},
                          "config": {"applied": True, "threshold": t}}}
        a = client.post("/v1/systemone", json=req("where is my card?")).json()["answers"]["q1"]
        assert a["probabilities"] == before["probabilities"]                     # probabilities untouched
        if want:
            assert a["choice"] == want
        else:
            others = {k: v for k, v in a["probabilities"].items() if k != "option_3"}
            assert a["choice"] == max(others, key=others.get)
    # a question with other options is outside the task
    other = {"a": "yes please", "b": "no thanks"}
    x = client.post("/v1/systemone", json=req("hi", crit=other)).json()
    so.tasks = {}
    assert client.post("/v1/systemone", json=req("hi", crit=other)).json()["answers"] == x["answers"]


def test_a4_appended_option(served):
    client, so = served
    plain = {"a": "pay by card", "b": "pay in cash"}
    body = req("I paid with my visa", crit=plain, images=[])            # imajev's extension, text route
    base = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
    assert base["abstained"] is False and base["unknown_probability"] == 0.0
    task = {"id": "img", "option": {"append": "can't tell"}, "match": {"imajev_extension": True}}
    so.tasks = {"img": {**task, "config": {"applied": False, "threshold": None}}}   # declined: exactly as before
    assert client.post("/v1/systemone", json=body).json()["answers"]["q1"] == base
    so.tasks = {"img": {**task, "config": {"applied": True, "threshold": 0.0}}}
    a = client.post("/v1/systemone", json=body).json()["answers"]["q1"]
    assert a["abstained"] is True and 0 < a["unknown_probability"] < 1
    so.tasks = {}


def test_a4_register_endpoint(served):
    client, so = served
    utts = ["where is my card", "link my card", "rate for euros", "tell me a joke", "what's the weather",
            "card not here yet", "connect card to app", "exchange rate today", "sing a song", "who won the game",
            "has my card shipped", "add card to account"]
    gold = ["option_0", "option_1", "option_2", None, None, "option_0", "option_1", "option_2", None, None,
            "option_0", "option_1"]
    r = client.post("/v1/abstention/tasks", json={"id": "clinc", "option": {"key": "option_3"}, "match": "option_set",
                                                  "examples": [{"request": req(u), "answer": g}
                                                               for u, g in zip(utts, gold)]})
    assert r.status_code == 200, r.text
    cfg = r.json()["config"]
    assert cfg["n"] == 12 and cfg["n_unanswerable"] == 4 and isinstance(cfg["applied"], bool)
    assert "clinc" in {t["id"] for t in client.get("/v1/abstention/tasks").json()["tasks"]}
    bad = client.post("/v1/abstention/tasks", json={"id": "x", "option": {"key": "nope"}, "examples":
                                                   [{"request": req("a"), "answer": "option_0"}]})
    assert bad.status_code == 422
    so.tasks = {}
