# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The served default's global temperature (`decisio.serve.temperature`, `--temperature`; docs/handoffs/tasks.md).

  G1  the function: softmax(log p / T); the most probable option never changes, exact ties stay ties, T = 1 returns the
      input bit for bit
  G2  served (CPU stand-in): with the flag off (T = 1) `/v1/systemone` is the raw readout (`/v1/answer`) bit for bit;
      with it on, every answer is `apply_temperature` of that readout and no choice, yes/no side or score mode changes,
      in one and two orders
  G3  composition: a registered task's own correction replaces the temperature (not stacked); a registered task whose
      correction was declined, and a question of no task, get the temperature

    uv run pytest -q tests/unit/test_temperature.py
"""
import numpy as np
import pytest

from decisio.serve.temperature import SERVED_TEMPERATURE, apply_temperature

MODEL = "Qwen/Qwen3-0.6B-Base"
T = 1.307


def test_g1_function():
    assert SERVED_TEMPERATURE == 1.307                           # the fitted value in the config
    rng = np.random.default_rng(0)
    for k in (2, 3, 4, 10, 77, 151):
        for _ in range(200):
            p = rng.dirichlet(np.full(k, rng.choice([0.05, 0.3, 1.0])))
            for t in (0.5, 1.307, 3.0):
                q = apply_temperature(p, t)
                assert abs(q.sum() - 1) < 1e-12 and int(np.argmax(q)) == int(np.argmax(p))
                assert (np.diff(q[np.argsort(p, kind="stable")]) >= 0).all()      # the whole ranking is kept
    p = np.array([0.4, 0.4, 0.2])
    q = apply_temperature(p, T)
    assert q[0] == q[1] > q[2]                                   # exact ties stay exact ties
    assert apply_temperature(p, 1.0).tolist() == p.tolist()      # off: bit for bit
    z = apply_temperature([1.0, 0.0], T)
    assert z.tolist() == [1.0, 0.0]                              # a zero probability stays zero
    assert 0.5 < apply_temperature([0.9, 0.1], T)[0] < 0.9       # T > 1 softens


CRIT = {"option_0": "card arrival", "option_1": "exchange rate", "option_2": "top up"}
REQS = [
    {"state": {"message": "where is my card"}, "questions": {
        "intent": {"type": "choice", "instructions": "Classify the intent.", "criteria": CRIT},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["not urgent", "somewhat", "very urgent"]}}},
    {"state": "I was charged twice and nobody answers my emails!", "questions": {
        "intent": {"type": "choice", "instructions": "Classify the intent.",
                   "criteria": {"a": "billing", "b": "delivery"}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"}}},
]


@pytest.fixture(scope="module")
def served():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app
    eng = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), temperature=T)
    return TestClient(make_app(eng, so)), so


def dist(a):
    return [a["noul"], 1.0 - a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def test_g2_flag_off_is_todays_output_and_on_never_changes_the_choice(served):
    from decisio.serve.systemone import SystemOneRequest, to_engine_question
    client, so = served
    assert client.get("/health").json()["systemone"]["temperature"] == T
    for body in REQS:
        req = SystemOneRequest.model_validate(body)
        raw = client.post("/v1/answer", json={"state": body["state"], "questions": [
            to_engine_question(q)[0] for q in req.questions.values()]}).json()["answers"]
        so.temperature = 1.0
        off = client.post("/v1/systemone", json=body).json()["answers"]
        off2 = client.post("/v1/systemone", json={**body, "orders": 2}).json()["answers"]
        so.temperature = T
        on = client.post("/v1/systemone", json=body).json()["answers"]
        on2 = client.post("/v1/systemone", json={**body, "orders": 2}).json()["answers"]
        for (name, a0), r in zip(off.items(), raw):
            k = 1 if a0["type"] == "noul" else None
            assert dist(a0)[:k] == r["probs"][:k]                                 # off: the raw readout, bit for bit
            assert dist(on[name])[:k] == apply_temperature(r["probs"], T).tolist()[:k]
            # two orders: the temperature after the average (1 - noul is rebuilt from the wire, hence the tolerance)
            assert np.allclose(dist(on2[name]), apply_temperature(dist(off2[name]), T), atol=1e-12)
            for x, y in ((a0, on[name]), (off2[name], on2[name])):                # the choice never changes
                if x["type"] == "choice":
                    assert x["choice"] == y["choice"]
                elif x["type"] == "noul":
                    assert (x["noul"] > 0.5) == (y["noul"] > 0.5) and (x["noul"] == 0.5) == (y["noul"] == 0.5)
                else:
                    assert int(np.argmax(dist(x))) == int(np.argmax(dist(y)))


def test_g3_a_task_correction_replaces_the_temperature(served):
    from decisio.readout import calibration
    from decisio.readout.debias import log_probs
    client, so = served
    so.debug_readout = True
    so.task_store.by_key.clear()
    utts = ["where is my card", "euro rate please", "top up my account", "card not arrived", "rate for yen"]
    ex = [{"request": {"state": {"message": u + " " + str(i)}, "questions": {"q1": {
        "type": "choice", "instructions": "Classify the intent.", "criteria": CRIT}}}, "answer": f"option_{i % 3}"}
        for i, u in enumerate(utts * 3)]
    r = client.post("/v1/tasks", json={"id": "cal", "examples": ex})
    assert r.status_code == 200, r.text
    task = so.task_store.lookup(r.json()["key"])
    body = {"state": {"message": "I want to top up"}, "questions": {"q1": ex[0]["request"]["questions"]["q1"]}}
    # the task's calibration declined: the temperature applies
    task["calibration"] = {**task["calibration"], "applied": False}
    a = client.post("/v1/systemone", json=body, headers={"x-decisio-debug": "readout"}).json()
    d = a["decisio_debug"]["q1"]
    assert d["path"] == "temperature" and dist(a["answers"]["q1"]) == apply_temperature(d["p"], T).tolist()
    # accepted: calibration on the raw readout, the temperature not stacked on it
    task["calibration"] = {**task["calibration"], "applied": True, "bias": [0.7, -0.2, -0.5]}
    a = client.post("/v1/systemone", json=body, headers={"x-decisio-debug": "readout"}).json()
    d = a["decisio_debug"]["q1"]
    want = calibration.apply_task_prior(log_probs(np.array(d["p"])), task["calibration"])
    assert d["path"] == "calibration" and dist(a["answers"]["q1"]) == want.tolist()
    # the task was fitted on the raw readout, not the tempered one
    so.task_store.by_key.clear()
    r = client.post("/v1/tasks", json={"id": "cal", "examples": ex}, headers={"x-decisio-debug": "readout"})
    raw = client.post("/v1/answer", json={"state": ex[0]["request"]["state"], "questions": [
        {"kind": "choice", "instructions": "Classify the intent.",
         "options": list(CRIT.values())}]}).json()["answers"][0]
    assert np.allclose(np.exp(r.json()["decisio_debug"]["lps"][0]), raw["probs"], atol=1e-12)
    so.task_store.by_key.clear()
    so.debug_readout = False
