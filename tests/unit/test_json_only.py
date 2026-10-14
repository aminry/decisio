# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A POST that is not JSON is refused before any work (`decisio.serve.content_type`, SECURITY.md).

A web page can send a "simple" cross-origin POST (Content-Type text/plain, application/x-www-form-urlencoded or
multipart/form-data) to a server on the visitor's machine without a preflight; the page cannot read the answer, but the
server would do the work. FastAPI parses a JSON body whatever the Content-Type says.

J1  the middleware: application/json (any case, with parameters) passes; every other type, and none, is 415 with
    `accept-post: application/json`; GET and OPTIONS are not touched;
J2  on every POST route of the served app, a non-JSON POST is 415 and the engine is not called; the same POST as JSON
    reaches it;
J3  POST /v1/systemone takes multipart/form-data when the request carries an X-Decisio-Client header (any value), and
    nothing else gets through with it: no other route, no other type;
J4  the header is only a barrier if a browser cannot add it: the server answers a preflight that asks for it with 405
    and no Access-Control-Allow-* header (a CORS layer added later fails this test).

  uv run pytest -q tests/unit/test_json_only.py
"""

import zlib

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decisio.serve.content_type import JsonOnly, media_type
from decisio.serve.systemone import SystemOne
from decisio.serve.vllm_engine import make_app

NOT_JSON = [
    "text/plain",
    "text/plain;charset=UTF-8",
    "application/x-www-form-urlencoded",
    "multipart/form-data; boundary=x",
    "application/jsonx",
    "application/vnd.api+json",
    "application/json-patch+json",
    "",
]


def test_j1_the_middleware_by_content_type():
    app = FastAPI()
    app.add_middleware(JsonOnly)

    @app.post("/echo")
    async def echo(request: __import__("fastapi").Request):
        return {"ok": True, "n": len(await request.body())}

    @app.get("/status")
    def status():
        return {"ok": True}

    client = TestClient(app)
    for ok in ("application/json", "application/json; charset=utf-8", "APPLICATION/JSON", " application/json ;q=1"):
        assert client.post("/echo", content=b"{}", headers={"content-type": ok}).status_code == 200, ok
    for bad in NOT_JSON:
        r = client.post("/echo", content=b"{}", headers={"content-type": bad})
        assert r.status_code == 415 and r.headers["accept-post"] == "application/json", bad
        assert "application/json only" in r.json()["detail"]
    assert client.get("/status").status_code == 200 and client.options("/echo").status_code == 405
    assert media_type([(b"content-type", b"Text/Plain; charset=x")]) == "text/plain" and media_type([]) == ""


class Tok:
    def encode(self, text, add_special_tokens=False):
        text = str(text)
        if len(text.strip()) <= 2 and text.strip().isalpha():  # a letter code, with or without its space: one token
            return [1000 + sum(ord(c) * 53**i for i, c in enumerate(text.strip()))]
        return [zlib.crc32(w.encode()) % 50000 for w in text.split()]


class Engine:
    """Counts how often the server asks the engine for anything."""

    tok = Tok()
    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self):
        self.calls = 0

    def facts(self):
        return {"engine": "fake"}

    def answer(self, state, questions, adapter=None):
        self.calls += 1
        return [np.full(len(q.get("options") or [1, 2]), 0.5) for q in questions], {}


def served(**kw):
    eng = Engine()
    return TestClient(make_app(eng, SystemOne(eng, "decisio-test", **kw))), eng


def post_routes(client):
    return sorted(
        {r.path for r in client.app.routes if "POST" in (getattr(r, "methods", None) or ())},
    )


def test_j2_every_post_route_refuses_a_non_json_body_before_any_work():
    client, eng = served()
    routes = post_routes(client)
    assert {"/v1/systemone", "/v1/answer", "/v1/tasks", "/v1/tasks/import", "/v1/abstention/tasks"} <= set(routes)
    body = b'{"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it x?"}}}'
    for path in routes:
        for kind in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=b", ""):
            r = client.post(path.replace("{task_id}", "t"), content=body, headers={"content-type": kind})
            assert r.status_code == 415, (path, kind, r.status_code)
    assert eng.calls == 0
    # the same body as JSON reaches the application: a plain 200 on the route that answers
    r = client.post("/v1/systemone", content=body, headers={"content-type": "application/json; charset=utf-8"})
    assert r.status_code == 200 and eng.calls == 1, r.text


def test_j3_multipart_on_systemone_needs_the_client_header():
    client, eng = served()
    form = {"content-type": "multipart/form-data; boundary=b"}
    without = client.post("/v1/systemone", content=b"--b--", headers=form)
    assert without.status_code == 415 and "X-Decisio-Client" in without.json()["detail"]
    for value in ("my-app", "", "1"):
        r = client.post("/v1/systemone", content=b"--b--", headers={**form, "x-decisio-client": value})
        assert r.status_code != 415, value  # reached the route (it then refuses the empty form)
    # the header opens nothing else: not another route, not another type
    for path in post_routes(client):
        if path != "/v1/systemone":
            r = client.post(path.replace("{task_id}", "t"), content=b"x", headers={**form, "x-decisio-client": "a"})
            assert r.status_code == 415, path
    for kind in ("text/plain", "application/x-www-form-urlencoded", ""):
        r = client.post("/v1/systemone", content=b"x", headers={"content-type": kind, "x-decisio-client": "a"})
        assert r.status_code == 415, kind
    assert eng.calls == 0


def test_j4_a_preflight_for_the_client_header_is_refused():
    client, _ = served()
    for path in ("/v1/systemone", "/v1/answer", "/v1/tasks"):
        r = client.options(
            path,
            headers={
                "origin": "https://example.com",
                "access-control-request-method": "POST",
                "access-control-request-headers": "x-decisio-client, content-type",
            },
        )
        assert r.status_code == 405, path
        assert not [h for h in r.headers if h.lower().startswith("access-control-")], (path, dict(r.headers))
