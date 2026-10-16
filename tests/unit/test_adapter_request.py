# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A request's `"adapter"` reaches the engine (a defect found in the card session of 2026-10-15: `/v1/systemone` passed
None to `SystemOne.answer`, so a body naming an adapter was accepted and answered by the base).

R1  `/v1/systemone` passes the adapter to the engine, reports it (`adapter` in the body, `x-decisio-adapter`), and
    serves the base for a missing, null or empty field
R2  an unknown adapter is a 404 that names it and the ones that are loaded, and the engine is not asked
R3  an adapter that is not a string is a 422
R4  a question that a registered task matches is refused with an adapter (the task was fitted on the base's readout)
R5  every route that takes a body: `/v1/answer` passes and reports the adapter and 404s an unknown one; the
    registration and certification routes refuse an `adapter` field instead of ignoring it; the list of POST routes is
    pinned, so a new route has to say what it does with an adapter

  uv run pytest -q tests/unit/test_adapter_request.py
"""

import zlib

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

CRIT = {"option_0": "billing", "option_1": "shipping", "option_2": "access"}


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        text = str(text)
        if len(text.strip()) <= 2 and text.strip().isalpha():  # a letter code, with or without its space: one token
            return [1000 + sum(ord(c) * 53**i for i, c in enumerate(text.strip()))]
        return [zlib.crc32(w.encode()) % 50000 for w in text.split()]


class FakeEngine:
    """The state's message is `top:confidence`; with an adapter the engine's top option is the next one, as a tuned
    adapter moves answers. Every call is recorded."""

    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self, adapters=("metask",)):
        self.adapters = {n: object() for n in adapters}
        self.calls = []

    def facts(self):
        return {"engine": "fake", "base": "fake-base", "adapters": sorted(self.adapters)}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        self.calls.append(adapter)
        if isinstance(state, dict):
            top, conf = str(state["message"]).split(":")[:2]
        else:
            top, conf = "0", "0.9"
        top = (int(top) + (1 if adapter else 0)) % 3
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[top % k] = float(conf)
            out.append(p)
        return out, {}


def make(adapters=("metask",)):
    eng = FakeEngine(adapters)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng, app


def wire(text="0:0.9:x", **extra):
    q = {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}
    return {"state": {"message": text}, "model": "m", "questions": {"q1": q}, **extra}


def test_r1_the_adapter_reaches_the_engine_and_is_reported():
    client, so, eng, _ = make()
    base = client.post("/v1/systemone", json=wire())
    assert base.status_code == 200 and eng.calls == [None]
    assert "adapter" not in base.json() and "x-decisio-adapter" not in base.headers
    r = client.post("/v1/systemone", json=wire(adapter="metask"))
    assert r.status_code == 200 and eng.calls == [None, "metask"]
    assert r.json()["adapter"] == "metask" and r.headers["x-decisio-adapter"] == "metask"
    assert r.json()["answers"]["q1"]["choice"] != base.json()["answers"]["q1"]["choice"]  # the adapter's answer
    for none in (None, ""):  # a null or empty field is the base
        assert client.post("/v1/systemone", json=wire(adapter=none)).json().get("adapter") is None
    assert eng.calls[-2:] == [None, None]


def test_r2_an_unknown_adapter_is_a_404_that_names_it():
    client, so, eng, _ = make()
    r = client.post("/v1/systemone", json=wire(adapter="nope"))
    assert (
        r.status_code == 404 and "unknown adapter 'nope'" in r.json()["detail"] and "['metask']" in r.json()["detail"]
    )
    assert eng.calls == []
    client, so, eng, _ = make(adapters=())  # a server with none loaded says so
    r = client.post("/v1/systemone", json=wire(adapter="metask"))
    assert r.status_code == 404 and "loaded: []" in r.json()["detail"] and eng.calls == []


def test_r3_an_adapter_that_is_not_a_name_is_a_422():
    client, _, eng, _ = make()
    for bad in (1, True, ["metask"], {"name": "metask"}):
        assert client.post("/v1/systemone", json=wire(adapter=bad)).status_code == 422
    assert eng.calls == []


def exs(n, start=0):
    return [{"request": wire(f"{i % 3}:0.9:{start + i}"), "answer": f"option_{i % 3}"} for i in range(n)]


def test_r4_a_question_a_task_matches_is_refused_with_an_adapter():
    client, so, eng, _ = make()
    r = client.post("/v1/tasks", json={"id": "routing", "examples": exs(30)})
    assert r.status_code == 200, r.text
    assert so.task_store.by_key
    eng.calls.clear()
    r = client.post("/v1/systemone", json=wire(adapter="metask"))
    assert r.status_code == 400 and "routing" in r.json()["detail"] and "fitted on the base" in r.json()["detail"]
    assert eng.calls == []
    # a question no task matches is answered by the adapter, and the base still gets the task
    other = wire(adapter="metask")
    other["questions"]["q1"]["criteria"] = {"option_0": "left", "option_1": "right"}
    assert client.post("/v1/systemone", json=other).status_code == 200 and eng.calls == ["metask"]
    base = client.post("/v1/systemone", json=wire())
    assert base.status_code == 200 and base.headers.get("x-decisio-tasks") == "routing"


def test_r5_the_engines_own_answer_route_passes_and_reports_the_adapter():
    eng = FakeEngine()
    app = vllm_engine.make_app(eng)
    client = TestClient(app)
    body = {"state": "s", "questions": [{"kind": "choice", "options": ["a", "b", "c"]}]}
    r = client.post("/v1/answer", json={**body, "adapter": "metask"})
    assert r.status_code == 200 and eng.calls == ["metask"] and r.json()["adapter"] == "metask"
    assert "adapter" not in client.post("/v1/answer", json=body).json() and eng.calls == ["metask", None]
    r = client.post("/v1/answer", json={**body, "adapter": "nope"})
    assert r.status_code == 404 and "unknown adapter 'nope'" in r.json()["detail"] and len(eng.calls) == 2


def test_r5_registration_and_certification_refuse_an_adapter_field():
    client, so, eng, _ = make()
    assert client.post("/v1/tasks", json={"id": "t", "examples": exs(30), "adapter": "metask"}).status_code == 422
    bad = exs(30)
    bad[3]["request"]["adapter"] = "metask"
    r = client.post("/v1/tasks", json={"id": "t", "examples": bad})
    assert r.status_code == 422 and "example 3" in r.json()["detail"] and "adapter" in r.json()["detail"]
    assert not so.task_store.by_key and eng.calls == []
    assert client.post("/v1/tasks", json={"id": "t", "examples": exs(30)}).status_code == 200
    r = client.post("/v1/tasks/t/certify", json={"examples": exs(30, 100), "adapter": "metask"})
    assert r.status_code == 422 and "adapter" in r.json()["detail"]
    r = client.post("/v1/abstention/tasks", json={"id": "a", "examples": [], "adapter": "metask"})
    assert r.status_code == 422


def test_r5_the_post_routes_are_pinned():
    _, _, _, app = make()
    app = vllm_engine.make_app(FakeEngine(), app.state.__dict__.get("systemone"))  # the engine's own routes too
    posts = {r.path for r in app.routes if "POST" in getattr(r, "methods", ())}
    client, so, eng, app2 = make()
    posts |= {r.path for r in app2.routes if "POST" in getattr(r, "methods", ())}
    assert posts == {
        "/v1/answer",  # takes "adapter": passed, 404 when unknown, reported
        "/v1/systemone",  # takes "adapter": passed, 404 when unknown, reported, refused beside a matching task
        "/v1/tasks",  # registration: an adapter field is refused
        "/v1/tasks/{task_id}/certify",  # certification: an adapter field is refused
        "/v1/tasks/import",  # imports a task export: no request, so no adapter
        "/v1/abstention/tasks",  # registration: an adapter field is refused
    }, "a new POST route: say what it does with a request's adapter, test it, and add it here"
