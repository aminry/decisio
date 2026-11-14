# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A `/v1/systemone` body over 32 MiB is a 413 before it is parsed (`decisio.serve.body_limit`).

B1  the middleware: a Content-Length over the limit is refused without the app reading a byte; a body of exactly the
    limit passes; other routes and methods are not touched;
B2  a body without a Content-Length (chunked), or one that sends more than it declared, is counted as it arrives and
    refused once it passes the limit, and the app never sees more than the limit;
B3  on the served app: 33 MiB of JSON or of multipart is a 413 in System One's error shape and the engine is not
    called; a request under the limit is answered.

  uv run pytest -q tests/unit/test_body_limit.py
"""

import json

import numpy as np
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from decisio.serve.body_limit import MAX_BODY_BYTES, BodyLimit
from decisio.serve.systemone import SystemOne
from decisio.serve.vllm_engine import make_app

SHAPE = {"detail": {"error_type": "api_usage_error", "message": "Request exceeds size limits."}}


def small_app(limit=1000):
    app = FastAPI()
    app.add_middleware(BodyLimit, limit=limit)
    seen = {"read": 0}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        n = 0
        async for chunk in request.stream():
            n += len(chunk)
        seen["read"] = n
        return {"n": n}

    @app.post("/v1/answer")
    async def answer(request: Request):
        return {"n": len(await request.body())}

    return TestClient(app), seen


def test_b1_content_length_over_the_limit_is_refused_unread():
    c, seen = small_app()
    assert MAX_BODY_BYTES == 33_554_432
    r = c.post("/v1/systemone", content=b"x" * 1001)
    assert r.status_code == 413 and r.json() == SHAPE and seen["read"] == 0
    assert c.post("/v1/systemone", content=b"x" * 1000).json() == {"n": 1000}
    assert c.post("/v1/answer", content=b"x" * 5000).json() == {"n": 5000}  # another route: not this middleware's
    assert c.get("/v1/systemone").status_code == 405  # a GET is passed on


def test_b2_a_body_with_no_length_is_counted_as_it_arrives():
    c, seen = small_app()

    def chunks(n, size=300):
        for _ in range(n):
            yield b"x" * size

    ok = c.post("/v1/systemone", content=chunks(3))  # 900 bytes, chunked: no Content-Length
    assert ok.status_code == 200 and ok.json() == {"n": 900}
    over = c.post("/v1/systemone", content=chunks(5))  # 1,500 bytes
    assert over.status_code == 413 and over.json() == SHAPE and seen["read"] == 900  # never saw the 5th chunk's end


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [1] * max(1, len(str(text).split()))


class Engine:
    tok = Tok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters = {}

    def __init__(self):
        self.calls = 0

    def facts(self):
        return {"engine": "fake"}

    def answer(self, state, questions, adapter=None):
        self.calls += 1
        return [np.full(len(q.get("options") or [1, 2]), 0.5) for q in questions], {}


def test_b3_the_served_app():
    eng = Engine()
    client = TestClient(make_app(eng, SystemOne(eng, "decisio-test")))
    q = {"q": {"type": "noul", "instructions": "Is it x?"}}
    big = {"state": "x" * (MAX_BODY_BYTES + 1), "questions": q}
    r = client.post("/v1/systemone", content=json.dumps(big), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json() == SHAPE and eng.calls == 0
    r = client.post(
        "/v1/systemone",
        data={"request": json.dumps({"state": "s", "questions": q})},
        files=[("p", ("p.png", b"\0" * (MAX_BODY_BYTES + 1), "image/png"))],
        headers={"x-typesafe-sdk": "t"},
    )
    assert r.status_code == 413 and r.json() == SHAPE and eng.calls == 0
    ok = {"state": "x" * 1_000_000, "questions": q}
    r = client.post("/v1/systemone", content=json.dumps(ok), headers={"content-type": "application/json"})
    assert r.status_code == 200 and eng.calls == 1
