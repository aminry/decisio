# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The digest of what a registration read (`inputs_sha256` on the calibration and head records, `decisio.serve.tasks`).

R1  the same examples read the same way give the same digest, in every registration; a different reading gives another
R2  the head's digest covers the hidden states as well as the log-scores
R3  a stored task exports and imports exactly, digests included

  uv run pytest -q tests/unit/test_registration_digest.py
"""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore, inputs_digest

K = 10
PROTO = np.random.default_rng(7).normal(size=(K, 48))
CRIT = {f"option_{k}": f"queue {k}" for k in range(K)}


def wire(msg):
    q = {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}
    return {"state": {"message": msg}, "model": "m", "questions": {"q1": q}}


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 50000 for w in str(text).split()]


class FakeEngine:
    """The state's message is `top:confidence:id:gold`. `jitter` moves every probability a little: another reading."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"
    jitter = 0.0

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
            p = p * (1 + self.jitter * np.arange(k) / k)
            out.append(p / p.sum())
        return out, {}


class FakeHidden(FakeEngine):
    hjitter = 0.0

    def readout(self, state, questions):
        (p,), _ = self.answer(state, questions)
        top, conf, ident, gold = str(state["message"]).split(":")[:4]
        noise = np.random.default_rng(int(ident)).normal(size=48)
        return [(np.log(p), PROTO[int(gold)] + noise + self.hjitter)]


def examples(n=200):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        gold = int(rng.integers(0, K))
        top = gold if rng.random() < 0.7 else int(rng.integers(0, K))
        conf = float(np.clip(rng.normal(0.9, 0.04), 0.3, 0.99))
        out.append({"request": wire(f"{top}:{conf:.4f}:{i}:{gold}"), "answer": f"option_{gold}"})
    return out


def make(jitter=0.0, hjitter=0.0):
    eng, hid = FakeEngine(), FakeHidden()
    eng.jitter = hid.jitter = jitter
    hid.hjitter = hjitter
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), hidden_engine=hid)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app, raise_server_exceptions=False)


def register(client):
    r = client.post("/v1/tasks", json={"id": "intents", "examples": examples()})
    assert r.status_code == 200, r.text
    return r.json()


def test_r1_same_reading_same_digest_another_reading_another():
    a, b = register(make()), register(make())
    assert a["calibration"]["inputs_sha256"] == b["calibration"]["inputs_sha256"]
    assert a["head"]["inputs_sha256"] == b["head"]["inputs_sha256"] and len(a["head"]["inputs_sha256"]) == 64
    c = register(make(jitter=1e-6))  # every probability read a little differently
    assert c["calibration"]["inputs_sha256"] != a["calibration"]["inputs_sha256"]
    assert c["head"]["inputs_sha256"] != a["head"]["inputs_sha256"]


def test_r2_the_head_digest_covers_the_hidden_states():
    a = register(make())
    d = register(make(hjitter=1e-6))  # the same log-scores, hidden states read a little differently
    assert d["calibration"]["inputs_sha256"] == a["calibration"]["inputs_sha256"]
    assert d["head"]["inputs_sha256"] != a["head"]["inputs_sha256"]


def test_r3_an_exported_task_imports_exactly():
    client = make()
    a = register(client)
    exported = client.get("/v1/tasks?full=1").json()["tasks"]
    other = make()
    assert other.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    b = other.get("/v1/tasks?full=1").json()["tasks"][0]
    assert b["head"]["inputs_sha256"] == a["head"]["inputs_sha256"]
    assert b["calibration"]["inputs_sha256"] == a["calibration"]["inputs_sha256"]
    assert inputs_digest([[1.0, 2.0]]) == inputs_digest([[1.0, 2.0]]) != inputs_digest([[1.0, 2.0000001]])
