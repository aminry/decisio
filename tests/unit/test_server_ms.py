# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`x-decisio-server-ms`, `x-decisio-stages` and `x-decisio-prompt` on POST /v1/systemone cover the whole request: the
route's time from its entry to the response, every read included. No model: the fake engines of the certified-answers,
fast-table and registration-digest tests, made to sleep a known time per read (off while a task registers, on for the
timed request), so the true server time is known and the header is checked against it and against the wall time.

T1  no task: server-ms covers the engine call and is under the wall time
T2  a plain (uncertified) task: the same
T3  a certified task, decisio's own read: server-ms covers the certified question's read, `task=` in the stages covers
    it too, the answer is certified (before the fix a certified-only request reported about 0 ms)
T4  a certified task whose fast table answers: server-ms covers the table's read, `task=` in the stages;
    T4b the table defers and decisio reads: server-ms covers the engine read
T5  the intent head: server-ms covers the hidden readout, `head=` in the stages covers it
T6  a mixed request, one certified question and one plain one: server-ms covers both reads together (they are
    sequential), the stages carry `task=` and `engine=`
T7  x-decisio-prompt: a certified-only request carries the prompt figures of its certified read (it used to have none)
T8  the certified rows keep the serial path on a batched server: `served_rows` is still wrapped by `_serial`, and the
    collector of a request's reads is not

  uv run pytest -q tests/unit/test_server_ms.py
"""

import time
import types

import numpy as np
import pytest
import test_certified_answers as CA
import test_registration_digest as RD
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

SLEEP = 0.05  # seconds per sleeping read
FLOOR = SLEEP * 1000 * 0.95  # milliseconds: a read's time, with headroom for the clock
PROMPT_TOKENS = 123  # what the sleeping engine reports per question it reads


class SlowEngine(CA.FakeEngine):
    """The certified-answers fake engine, sleeping `sleep` seconds per call, reporting its own time (`server_ms`, the
    stages' `engine`) and its prompt tokens per question, as the real engines do."""

    def __init__(self):
        super().__init__()
        self.sleep = 0.0
        self.calls = 0

    def answer(self, state, questions, adapter=None):
        self.calls += 1
        t0 = time.perf_counter()
        if self.sleep:
            time.sleep(self.sleep)
        probs, info = super().answer(state, questions, adapter)
        ms = (time.perf_counter() - t0) * 1000
        return probs, {**info, "server_ms": ms, "engine_prompt_tokens": [PROMPT_TOKENS] * len(questions)}


def timed(client, body):
    """The response and the wall time around the request, in milliseconds."""
    t0 = time.perf_counter()
    r = client.post("/v1/systemone", json=body)
    wall = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text
    return r, wall


def server_ms(r) -> float:
    return float(r.headers["x-decisio-server-ms"])


def stages(r) -> dict:
    raw = r.headers.get("x-decisio-stages", "")
    return {k: float(v) for k, v in (kv.split("=") for kv in raw.split(";") if kv)}


def certified_server():
    """A server on the sleeping engine with a certified task registered (the sleep off while it re-reads its 400
    examples)."""
    eng = SlowEngine()
    client, so = CA.make(engine=eng)
    r = CA.post_task(client, {"id": "cert", "examples": CA.examples(400, 0.96), "certify": {}})
    assert r.status_code == 200, r.text
    assert r.json()["certificate"]["certified"]
    eng.sleep = SLEEP
    return client, eng


# ---- T1, T2 ----------------------------------------------------------------------------------------------------------


def test_t1_no_task_server_ms_covers_the_engine_call():
    eng = SlowEngine()
    eng.sleep = SLEEP
    client, _ = CA.make(engine=eng)
    r, wall = timed(client, CA.wire("0:0.97:probe"))
    assert eng.calls == 1
    assert FLOOR <= server_ms(r) <= wall
    assert stages(r)["engine"] >= FLOOR


def test_t2_a_plain_task_server_ms_covers_the_engine_call():
    eng = SlowEngine()
    client, _ = CA.make(engine=eng)
    r = CA.post_task(client, {"id": "plain", "examples": CA.examples(60, 0.95)})
    assert r.status_code == 200, r.text
    eng.sleep, eng.calls = SLEEP, 0
    r, wall = timed(client, CA.wire("0:0.97:probe"))
    assert r.headers["x-decisio-tasks"] == "plain" and eng.calls == 1
    assert FLOOR <= server_ms(r) <= wall


# ---- T3 --------------------------------------------------------------------------------------------------------------


def test_t3_a_certified_task_server_ms_covers_the_certified_read():
    client, eng = certified_server()
    eng.calls = 0
    r, wall = timed(client, CA.wire("0:0.97:probe"))
    assert r.json()["answers"]["q1"]["certified"] is True
    assert eng.calls == 1  # the certified question's own read; there is no request engine call
    assert FLOOR <= server_ms(r) <= wall
    st = stages(r)
    assert "task" in st and st["task"] >= FLOOR
    assert st["task"] <= server_ms(r)


# ---- T4 --------------------------------------------------------------------------------------------------------------


def table_server(monkeypatch):
    """A server on the fast-table tests' engine with a fast-table task kept, registered on their virtual clock (so the
    timing guard keeps the table); the request after it runs on the real clock, the engine and the table's encoder
    sleeping `SLEEP` per read."""
    pytest.importorskip("sklearn")
    pytest.importorskip("obstore")
    import test_fast_table_task as FTT

    from decisio.serve import fast_table as FT
    from decisio.serve import systemone as SO

    sleep = {"on": False}

    class SlowTableEngine(FTT.FakeEngine):
        def answer(self, state, questions, adapter=None):
            if sleep["on"]:
                time.sleep(SLEEP)
            return super().answer(state, questions, adapter)

    def slow_embed(texts):
        if sleep["on"]:
            time.sleep(SLEEP)
        return FTT.fake_embed(texts)

    FTT.CLOCK.t, FTT.CLOCK.decisio_ms, FTT.CLOCK.table_ms = 0.0, 60.0, 5.0
    monkeypatch.setattr(FT, "embed", slow_embed)
    monkeypatch.setattr(FT, "encoder", lambda: object())
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-A")
    so = SystemOne(SlowTableEngine(), "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    client = TestClient(app)
    with monkeypatch.context() as m:  # the virtual clock for the registration only
        real = SO.time
        proxy = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("_")})
        proxy.perf_counter = FTT.CLOCK
        m.setattr(SO, "time", proxy)
        r = FTT.register(client)
    assert r.status_code == 200, r.text
    assert r.json()["fast_table_report"]["kept"]
    sleep["on"] = True
    return client, FTT


def test_t4_the_fast_table_answers_server_ms_covers_the_table_read(monkeypatch):
    client, FTT = table_server(monkeypatch)
    msg = FTT.message("billing", np.random.default_rng(1), right=False, conf=0.4)  # unsure decisio, clear text
    r, wall = timed(client, FTT.wire(msg))
    a = r.json()["answers"]["q1"]
    assert a["answered_by"] == "table" and a["certified"] is True
    assert FLOOR <= server_ms(r) <= wall
    st = stages(r)
    assert "task" in st and st["task"] >= FLOOR


def test_t4b_the_fast_table_defers_server_ms_covers_decisios_read(monkeypatch):
    client, FTT = table_server(monkeypatch)
    msg = FTT.message("billing", np.random.default_rng(1), right=True, conf=0.95, confident_text=False)
    r, wall = timed(client, FTT.wire(msg))
    a = r.json()["answers"]["q1"]
    assert a["answered_by"] == "decisio" and a["certified"] is True
    # the table's read and then decisio's: both sleep
    assert 2 * FLOOR <= server_ms(r) <= wall
    st = stages(r)
    assert "task" in st and st["task"] >= 2 * FLOOR


# ---- T5 --------------------------------------------------------------------------------------------------------------


class SlowHidden(RD.FakeHidden):
    """The registration-digest tests' hidden engine, sleeping `sleep` seconds per readout."""

    sleep = 0.0

    def readout(self, state, questions):
        if self.sleep:
            time.sleep(self.sleep)
        return super().readout(state, questions)


def test_t5_the_intent_head_server_ms_covers_the_hidden_readout():
    hid = SlowHidden()
    so = SystemOne(RD.FakeEngine(), "decisio-test", task_store=TaskStore("fp"), hidden_engine=hid)
    app = FastAPI()
    add_routes(app, so)
    client = TestClient(app)
    r = client.post("/v1/tasks", json={"id": "intents", "examples": RD.examples()})
    assert r.status_code == 200, r.text
    assert r.json()["head"]["applied"]
    hid.sleep = SLEEP
    r, wall = timed(client, RD.wire("3:0.9:9999:3"))
    assert r.headers["x-decisio-tasks"] == "intents"
    assert FLOOR <= server_ms(r) <= wall
    st = stages(r)
    assert "head" in st and st["head"] >= FLOOR


# ---- T6 --------------------------------------------------------------------------------------------------------------


def test_t6_a_mixed_request_server_ms_covers_the_certified_and_the_plain_read():
    client, eng = certified_server()
    body = CA.wire("0:0.97:probe")
    body["questions"]["q2"] = {
        "type": "choice",
        "instructions": "Which colour is it?",
        "criteria": {"option_0": "red", "option_1": "blue"},
    }  # no task matches it: the request's own engine call
    eng.calls = 0
    r, wall = timed(client, body)
    answers = r.json()["answers"]
    assert answers["q1"]["certified"] is True
    assert eng.calls == 2  # the certified read, then the request's engine call: one after the other
    assert 2 * FLOOR <= server_ms(r) <= wall
    st = stages(r)
    assert st["task"] >= FLOOR and st["engine"] >= FLOOR
    assert st["task"] + st["engine"] <= server_ms(r)


# ---- T7 --------------------------------------------------------------------------------------------------------------


def test_t7_a_certified_only_request_carries_its_reads_prompt_figures():
    client, eng = certified_server()
    eng.sleep = 0.0  # no timing here
    r, _ = timed(client, CA.wire("0:0.97:probe"))
    assert r.json()["answers"]["q1"]["certified"] is True
    prompt = r.headers.get("x-decisio-prompt")
    assert prompt is not None, "a certified-only request lost its prompt figures"
    assert f"tokens={PROMPT_TOKENS}" in prompt and "cached=0" in prompt


def test_t8_served_rows_keeps_the_serial_path():
    from decisio.serve.systemone import SystemOne

    assert getattr(SystemOne.served_rows, "__wrapped__", None) is not None  # functools.wraps of _serial
    assert getattr(SystemOne.collect_read, "__wrapped__", None) is None
