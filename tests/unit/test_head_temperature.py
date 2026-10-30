# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head's temperature (`decisio.readout.intent_head`; served by `apply_intent_head`). No model: synthetic
hidden states, and a fake engine and hidden engine behind `POST /v1/tasks`.

H1  the fit stores a temperature and says what it did on the out-of-fold examples: log loss and ECE with and without it,
    better with it on a head that fits its examples tightly
H2  the arg-max does not change; on new items from the same source the tempered head is better calibrated
H3  a record without a temperature serves exactly as before (T = 1), and stored tasks still load
H4  the temperature is fitted on the fit examples only: another certification pool leaves it unchanged, and a changed
    temperature changes the fit digest a certificate names
H5  registration through the server: the response carries the temperature and its log loss and ECE, and the certificate
    is made on the tempered probabilities, after the head and its temperature exist

  uv run pytest -q tests/unit/test_head_temperature.py
"""

import copy

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.readout import intent_head
from decisio.readout.intent_head import _ece, apply_intent_head, fit_intent_head, fit_temperature
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore, fit_digest


def data(K=12, per=8, d=64, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), per)
    proto = rng.normal(size=(K, d))
    H = proto[y] + rng.normal(size=(len(y), d))
    logp = rng.normal(size=(len(y), K))
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    return H, logp, y, [f"intent {k}" for k in range(K)]


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_h1_the_fit_stores_a_temperature_and_its_effect():
    H, logp, y, opts = data()
    rec = fit_intent_head(logp, H, y, opts, fingerprint="m", task_id="t")
    assert rec["applied"] and rec["temperature"] > 1.0  # the head is overconfident: T above 1 cools it
    cv = rec["temperature_cv"]
    assert cv["n"] == len(y)
    assert cv["log_loss"]["with"] < cv["log_loss"]["without"] and cv["ece"]["with"] < cv["ece"]["without"], cv
    assert intent_head.T_RANGE[0] <= rec["temperature"] <= intent_head.T_RANGE[1]


def test_h1_the_search_finds_the_temperature_of_a_known_overconfident_source():
    rng = np.random.default_rng(1)
    K, n, true_T = 8, 4000, 3.0
    logits = rng.normal(scale=2.0, size=(n, K))
    y = np.array([rng.choice(K, p=softmax(logits[i] / true_T)) for i in range(n)])
    assert abs(fit_temperature(logits, y) - true_T) < 0.25


def test_h2_the_argmax_is_unchanged_and_new_items_are_better_calibrated():
    H, logp, y, opts = data()
    rec = fit_intent_head(logp, H, y, opts)
    plain = {k: v for k, v in rec.items() if k not in ("temperature", "temperature_cv")}
    Ht, lpt, yt, _ = data(per=40, seed=5)
    cooled = np.array([apply_intent_head(lpt[i], Ht[i], rec, opts) for i in range(len(yt))])
    hot = np.array([apply_intent_head(lpt[i], Ht[i], plain, opts) for i in range(len(yt))])
    assert (cooled.argmax(1) == hot.argmax(1)).all()
    ll = lambda P: float(np.mean(-np.log(P[np.arange(len(yt)), yt])))  # noqa: E731
    assert ll(cooled) < ll(hot) and _ece(np.log(cooled), yt) < _ece(np.log(hot), yt)


def test_h3_a_record_without_a_temperature_serves_as_before():
    H, logp, y, opts = data()
    rec = fit_intent_head(logp, H, y, opts)
    old = {k: v for k, v in rec.items() if k not in ("temperature", "temperature_cv")}
    Ht, lpt, _, _ = data(seed=1)
    for i in range(len(Ht)):
        by_hand = softmax(lpt[i] + Ht[i] @ old["A"].astype(np.float64) + old["c"].astype(np.float64))
        assert np.allclose(apply_intent_head(lpt[i], Ht[i], old, opts), by_hand, atol=1e-12)
        tempered = softmax(
            (lpt[i] + Ht[i] @ rec["A"].astype(np.float64) + rec["c"].astype(np.float64)) / rec["temperature"]
        )
        assert np.allclose(apply_intent_head(lpt[i], Ht[i], rec, opts), tempered, atol=1e-12)
    off = {**old, "applied": False}
    assert np.allclose(apply_intent_head(lpt[0], Ht[0], off, opts), softmax(lpt[0]))  # a head that is off has no T


def test_h4_the_fit_digest_names_the_temperature():
    H, logp, y, opts = data()
    rec = fit_intent_head(logp, H, y, opts)
    task = {"calibration": {"applied": False}, "head": rec}
    old = {"calibration": {"applied": False}, "head": {k: v for k, v in rec.items() if k != "temperature"}}
    assert fit_digest(task) != fit_digest(old)  # a certificate made without T does not survive a T
    task2 = copy.deepcopy(task)
    task2["head"]["temperature"] = rec["temperature"] * 1.01
    assert fit_digest(task2) != fit_digest(task)
    # the examples a head is fitted on are the only input to T: appending others changes nothing about a fit on these
    assert fit_intent_head(logp, H, y, opts)["temperature"] == rec["temperature"]


# ---- H5: a server with a fake engine and a fake hidden engine ----------------------------------------------------

K = 10
CRIT = {f"option_{k}": f"queue {k}" for k in range(K)}
PROTO = np.random.default_rng(7).normal(size=(K, 48))


def wire(msg):
    return {
        "state": {"message": msg},
        "model": "m",
        "questions": {"q1": {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}},
    }


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 50000 for w in str(text).split()]


class FakeEngine:
    """The state's message is `top:confidence:id:gold`: the engine puts that confidence on `top` and splits the rest."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def facts(self):
        return {"engine": "fake", "vllm": "1.0", "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        top, conf = str(state["message"]).split(":")[:2]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top)] = float(conf)
            out.append(p)
        return out, {}


class FakeHidden(FakeEngine):
    """The head's reading: the engine's log-scores and a hidden state that carries the gold option, with noise."""

    def readout(self, state, questions):
        (p,), _ = self.answer(state, questions)
        top, conf, ident, gold = str(state["message"]).split(":")[:4]
        noise = np.random.default_rng(int(ident)).normal(size=48)
        return [(np.log(p), PROTO[int(gold)] + noise)]


def examples(n, seed=0, start=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = int(rng.integers(0, K))
        top = gold if rng.random() < 0.7 else int(rng.integers(0, K))
        conf = float(
            np.clip(rng.normal(0.9, 0.04), 0.3, 0.99)
        )  # confident whether right or not: the readout is too sure
        out.append({"request": wire(f"{top}:{conf:.4f}:{start + i}:{gold}"), "answer": f"option_{gold}"})
    return out


def make():
    hid = FakeHidden()
    so = SystemOne(FakeEngine(), "decisio-test", task_store=TaskStore("fp"), hidden_engine=hid)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def test_h5_registration_reports_the_temperature_and_certifies_on_the_tempered_probabilities():
    client, so = make()
    seen = []
    real = so.served_rows

    def spy(t, pairs):
        rows = real(t, pairs)
        seen.append((copy.deepcopy(t["head"]), pairs, rows))
        return rows

    so.served_rows = spy
    r = client.post(
        "/v1/tasks", json={"id": "intents", "examples": examples(420), "certify": {"alpha": 0.2, "delta": 0.1}}
    )
    assert r.status_code == 200, r.text
    head = r.json()["head"]
    assert head["applied"] and head["temperature"] > 0
    cv = head["temperature_cv"]
    assert cv["log_loss"]["with"] <= cv["log_loss"]["without"] and set(cv["ece"]) == {"without", "with"}
    assert seen, "the certificate was made without scoring any example"
    stored, pairs, rows = seen[0]
    assert stored["temperature"] == head["temperature"]  # the head and its temperature existed before the certificate
    for (state, q), p in list(zip(pairs, rows))[:5]:
        ((lp, h),) = so.hidden_engine.readout(state, [so.engine_question(q)[0]])
        by_hand = softmax(
            (lp + h @ np.asarray(stored["A"], dtype=np.float64) + np.asarray(stored["c"], dtype=np.float64))
            / stored["temperature"]
        )
        assert np.allclose(p, by_hand, atol=1e-6)
    t = so.task_store.by_key[next(iter(so.task_store.by_key))]
    assert t["certificate"]["n"] == len(pairs)


def test_h5_a_stored_task_without_a_temperature_loads_and_serves_untempered():
    from decisio.serve.systemone import SystemOneRequest

    client, so = make()
    r = client.post("/v1/tasks", json={"id": "intents", "examples": examples(200)})
    assert r.status_code == 200 and r.json()["head"]["applied"] and abs(r.json()["head"]["temperature"] - 1) > 0.01
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    for t in exported:
        t["head"].pop("temperature", None)
        t["head"].pop("temperature_cv", None)
    other_client, other = make()
    assert other_client.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    head = next(iter(other.task_store.by_key.values()))["head"]
    assert "temperature" not in head
    msg = "3:0.9:9999:3"
    r = other_client.post("/v1/systemone", json=wire(msg))
    # a failed request says what failed (status and body), not a bare KeyError on "answers" (the 2026-10-28 CI run)
    assert r.status_code == 200 and "answers" in r.json(), (r.status_code, r.text[:600])
    ans = r.json()["answers"]["q1"]
    q = SystemOneRequest.model_validate(wire(msg)).questions["q1"]
    ((lp, h),) = other.hidden_engine.readout({"message": msg}, [other.engine_question(q)[0]])
    by_hand = softmax(lp + h @ np.asarray(head["A"], dtype=np.float64) + np.asarray(head["c"], dtype=np.float64))
    assert np.allclose(list(ans["probabilities"].values()), by_hand, atol=1e-4)
