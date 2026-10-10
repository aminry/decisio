# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The wait a request spent before it took the engine is on the wire (decisio #132): `wait=` in x-decisio-stages next to
`engine=`, 0 on an idle server, and x-decisio-wait-behind only when a registration was running as the request arrived
(tests/unit/test_boundary_after_response.py H8 covers the held registration).

W1  /v1/systemone on the CPU stand-in: the stages header ends with engine=... ;wait=..., the wait is 0 and no
    x-decisio-wait-behind header is sent
W2  one request waiting for another request's engine call (not a registration) reports the wait without the header
"""

import threading
import time

import pytest

MODEL = "Qwen/Qwen3-0.6B-Base"
Q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    eng = HFLettersEngine(MODEL, pad_to=None)
    return TestClient(make_app(eng, SystemOne(eng, "decisio-test"))), eng


def stages(r):
    assert r.status_code == 200, r.text
    return dict(kv.split("=") for kv in r.headers["x-decisio-stages"].split(";"))


def test_w1_an_idle_server_waits_zero_and_says_so(client):
    c, _ = client
    r = c.post("/v1/systemone", json={"state": "Ticket one: the site is down.", "questions": Q})
    s = stages(r)
    assert list(s)[-2:] == ["engine", "wait"] and float(s["wait"]) < 50
    assert "x-decisio-wait-behind" not in r.headers


def test_w2_a_request_behind_another_requests_call_reports_its_wait_but_not_a_registration(client):
    c, eng = client
    out = {}
    eng._lock.acquire()  # another request holds the engine
    t = threading.Thread(
        target=lambda: out.update(
            r=c.post("/v1/systemone", json={"state": "Ticket two: refund please.", "questions": Q})
        )
    )
    t.start()
    time.sleep(0.6)
    eng._lock.release()
    t.join(60)
    s = stages(out["r"])
    assert float(s["wait"]) >= 500
    assert "x-decisio-wait-behind" not in out["r"].headers
