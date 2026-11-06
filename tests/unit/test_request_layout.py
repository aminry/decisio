# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A layout per request (`"layout"` on /v1/systemone and /v1/answer; decisio.serve.layout.using; docs/running.md, "A
layout per request"; docs/tasks.md, "A task under a layout").

R1   the field: absent or null is the server's layout, anything else is a 422 that names the two; /v1/answer too
R2   the echo: x-decisio-prompt says the layout read, usage counts the header only when it was read; /v1/answer's timing
R3   token for token: a per-request plain on a header server builds the plain server's rows, and the reverse, through
     the vLLM engine's own request path and row builder (inherited by the HF and MLX engines); the CPU stand-in through
     the route, and a task registered under a layout fits what a server of that layout fits (slow)
R4   a task is fitted under one layout and applies to its requests only: layout_mismatch, the reads, the dynamic and
     abstention routes, an example of another layout refused
R5   fast tables: a labelled table and one that learns never answer, nor log, a request of the other layout; a learning
     task's stack is its layout's
R6   the notes: an adapter, a tuned repository, the temperatures file, a flavor; never a refusal; what the start records
R7   the image route refuses the field; packed mode refuses the header
R8   --concurrency batched: one layout per engine call, built under it
R9   the header limit applies to header requests only
R10  at start: tasks of both layouts load and apply; the held default stays and says requests may ask for the header
R11  import across layouts: readout and abstention tasks of either layout load; another model's are still refused
R12  certificates are per layout's stack: a header certificate holds for header requests on a plain server
R13  think and two orders read under the request's layout
R14  one question, a task under each layout: both are kept, each answers its own layout's requests, and neither
     replaces the other, whichever is registered or loaded first

  uv run pytest -q tests/unit/test_request_layout.py
"""

import json
import os
import types
import zlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine as CertEngine
from test_certified_answers import examples, post_task, wire
from test_default_layout import launcher  # noqa: F401  (the start with the stand-in engine, for R10)
from test_layout_flags import BASE, STATE, StubTok

from decisio.readout.letters import PromptFormat, layout_header
from decisio.serve import layout as lay
from decisio.serve import vllm_engine
from decisio.serve.combine import Combiner, FairLock, Waiting
from decisio.serve.systemone import SystemOne, SystemOneRequest, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import PAD_TOKEN, LettersEngine

MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
FP = {"plain": "fp-plain", "header": "fp-header"}  # the task store's fingerprint of each layout


class RowEngine(LettersEngine):
    """The vLLM engine's request path and row builder (`LettersEngine`), the forward pass replaced: each row's
    distribution is a function of its token ids, and every row scored is kept."""

    def __init__(self, layout="header", mode="separate", limit=None):
        self.tok = StubTok()
        self.fmt = PromptFormat(**BASE, layout=layout)
        self.mode, self.pad_token, self.pad_where = mode, PAD_TOKEN, "front"
        self.adapters, self.block_size, self.match_unit, self.pad_unit = {}, 16, 16, None
        self.pad_policy = "none"
        self.over_limit_tokens = limit
        self._lock = FairLock()
        self.scored = []

    def facts(self):
        return {"engine": "rows", "vllm": "test", "base": "fake-base", "quantization": None, "mode": self.mode}

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        probs = []
        for ids, lab in rows:
            self.scored.append(list(ids))
            h = zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes())
            z = np.array([(h >> (3 * i)) % 7 for i in range(len(lab))], dtype=np.float64)
            p = np.exp(z - z.max())
            probs.append(p / p.sum())
        n = [len(ids) for ids, _ in rows]
        return probs, {
            "warm_ms": 0.0,
            "questions_ms": 0.0,
            "readout_ms": 0.0,
            "cached_tokens_mean": 0.0,
            "engine_timing": [],
            "prompt_tokens_mean": float(np.mean(n)),
            "prompt_tokens": sum(n),
            "engine_prompt_tokens": n,
            "cached_tokens": [0] * len(n),
        }


class FollowsLayout:
    """A test engine whose prompt format follows the layout in effect, as every real text engine's does (the
    `LettersEngine.fmt` property), and that keeps the layout of every read."""

    fmt = LettersEngine.fmt

    def follows(self, layout):
        self.fmt = PromptFormat(**BASE, layout=layout)
        self.reads = []
        return self

    def _prepare_separate(self, state, questions):  # the probe's rows differ by layout, as the real rows do
        return ([1, 2, 3, len(questions), lay.LAYOUT_VALUES.index(self.fmt.layout)],)

    def answer(self, state, questions, adapter=None):
        self.reads.append(self.fmt.layout)
        return super().answer(state, questions, adapter)


class CertFake(FollowsLayout, CertEngine):
    pass


def server(engine, layout="header", **kw):
    so = SystemOne(engine, "decisio-test", task_store=TaskStore(FP[layout], layouts=FP), **kw)
    app = vllm_engine.make_app(engine, so) if isinstance(engine, LettersEngine) else FastAPI()
    if not isinstance(engine, LettersEngine):
        add_routes(app, so)
    return TestClient(app), so


def cert_server(layout="header", **kw):
    eng = CertFake().follows(layout)
    client, so = server(eng, layout, **kw)
    return client, so, eng


def ask(client, text="0:0.99:x", layout=None, **extra):
    body = {**wire(text), **({"layout": layout} if layout is not None else {}), **extra}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    return r


TWO_QUESTIONS = {
    "q1": {"type": "choice", "instructions": "Which queue should this go to?", "criteria": {"a": "billing", "b": "x"}},
    "q2": {"type": "noul", "instructions": "Is the customer angry?"},
}


def request(layout=None, questions=TWO_QUESTIONS):
    return {"state": STATE, "model": "m", "questions": questions, **({"layout": layout} if layout else {})}


# ---- R1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["footer", "Plain", 3, ["plain"], {"v": "plain"}, ""])
def test_r1_a_value_other_than_the_two_is_a_422_that_names_them(bad):
    client, _, _ = cert_server()
    r = client.post("/v1/systemone", json={**wire("0:0.9:x"), "layout": bad})
    assert r.status_code == 422 and "'plain', 'header'" in r.json()["detail"]
    with pytest.raises(lay.LayoutRefused, match="plain.*header"):
        lay.request_value(bad)


def test_r1_absent_or_null_is_the_servers_layout():
    client, so, eng = cert_server("header")
    ask(client)
    assert client.post("/v1/systemone", json={**wire("0:0.99:x"), "layout": None}).status_code == 200
    assert eng.reads == ["header", "header"] and so.layout == "header"
    ask(client, layout="plain")
    assert eng.reads[-1] == "plain" and so.layout == "header"  # the server's own layout does not move
    assert eng.fmt.layout == "header" and lay.current() is None


def test_r1_the_engines_route_takes_it_too():
    eng = RowEngine("header")
    client = TestClient(vllm_engine.make_app(eng))
    q = [{"kind": "choice", "instructions": "Which?", "options": ["a", "b"]}]
    assert client.post("/v1/answer", json={"state": "s", "questions": q, "layout": "footer"}).status_code == 422
    r = client.post("/v1/answer", json={"state": "s", "questions": q, "layout": "plain"})
    assert r.status_code == 200 and r.json()["timing"]["layout"] == "plain"
    assert client.post("/v1/answer", json={"state": "s", "questions": q}).json()["timing"]["layout"] == "header"


# ---- R2 -------------------------------------------------------------------------------------------------------------


def test_r2_the_response_says_the_layout_read_and_usage_counts_the_header_only_when_read():
    header = RowEngine("header")
    client, so = server(header, "header")
    plain_r = client.post("/v1/systemone", json=request("plain"))
    head_r = client.post("/v1/systemone", json=request())
    assert plain_r.status_code == head_r.status_code == 200
    assert plain_r.headers["x-decisio-prompt"].startswith("layout=plain;")
    assert head_r.headers["x-decisio-prompt"].startswith("layout=header;")
    listed = layout_header([q["instructions"] for q in TWO_QUESTIONS.values()])
    gap = head_r.json()["usage"]["input_tokens"] - plain_r.json()["usage"]["input_tokens"]
    assert gap == len(header.tok.encode(listed))
    # the plain server's count for the same request, token for token
    pc, _ = server(RowEngine("plain"), "plain")
    assert pc.post("/v1/systemone", json=request()).json()["usage"] == plain_r.json()["usage"]
    assert "layout_note" not in plain_r.json()  # nothing recorded another layout


# ---- R3 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("server_layout, asked", [("header", "plain"), ("plain", "header")])
def test_r3_a_request_of_the_other_layout_reads_the_other_servers_rows_token_for_token(server_layout, asked):
    mine, other = RowEngine(server_layout), RowEngine(asked)
    c_mine, _ = server(mine, server_layout)
    c_other, _ = server(other, asked)
    a = c_mine.post("/v1/systemone", json=request(asked))
    b = c_other.post("/v1/systemone", json=request())
    assert a.status_code == b.status_code == 200
    assert mine.scored == other.scored and mine.scored  # the token ids of every row
    assert a.json()["answers"] == b.json()["answers"] and a.json()["usage"] == b.json()["usage"]
    # the row builder itself, as the engines call it
    with lay.using(asked):
        rows, n = mine._prepare_separate(STATE, [{"kind": "noul", "instructions": "Is it?"}])
    assert (rows, n) == other._prepare_separate(STATE, [{"kind": "noul", "instructions": "Is it?"}])
    # and the engine's route
    q = [{"kind": "choice", "instructions": "Which?", "options": ["a", "b"]}]
    e1 = TestClient(vllm_engine.make_app(mine)).post("/v1/answer", json={"state": "s", "questions": q, "layout": asked})
    e2 = TestClient(vllm_engine.make_app(other)).post("/v1/answer", json={"state": "s", "questions": q})
    assert e1.json()["answers"] == e2.json()["answers"] and mine.scored[-1] == other.scored[-1]


def test_r3_the_format_follows_the_layout_in_effect_and_the_template_and_labels_do_not():
    eng = RowEngine("header")
    with lay.using("plain"):
        assert eng.fmt == PromptFormat(**BASE) and lay.server_fmt(eng).layout == "header"
        tail, labels = eng._template_tail(), eng._labels([" A", " B"])
    assert eng.fmt.layout == "header" and eng._template_tail() == tail and eng._labels([" A", " B"]) == labels
    with lay.using("header"), lay.using(None):
        assert lay.current() is None and eng.fmt.layout == "header"


# ---- R4 -------------------------------------------------------------------------------------------------------------


def test_r4_a_task_applies_only_to_the_requests_of_its_layout_and_says_layout_mismatch_otherwise():
    client, so, eng = cert_server("header")
    r = post_task(client, {"id": "p", "layout": "plain", "examples": examples(400, 0.96), "certify": {}})
    assert r.status_code == 200, r.text
    t = r.json()
    assert set(eng.reads) == {"plain"}  # every registration read, the served check's too, under the task's layout
    assert t["fingerprint"] == FP["plain"] and so.task_store.layout_of(t) == "plain"
    hit = ask(client, layout="plain")
    assert hit.headers["x-decisio-tasks"] == "p" and hit.json()["answers"]["q1"]["certified"] is True
    miss = ask(client)  # the server's layout: the header
    a = miss.json()["answers"]["q1"]
    assert a["certified"] is False and a["certified_reason"] == "layout_mismatch" and a["certificate"]
    assert "x-decisio-tasks" not in miss.headers and eng.reads[-1] == "header"
    assert ask(client, layout="header").json()["answers"]["q1"]["certified_reason"] == "layout_mismatch"
    # a task of another model is still serving_stack_changed
    so.task_store.by_key[so.task_store.slot(t)]["fingerprint"] = "another model"
    assert ask(client, layout="plain").json()["answers"]["q1"]["certified_reason"] == "serving_stack_changed"


def test_r4_registration_takes_the_servers_layout_by_default_and_refuses_an_example_of_another():
    client, so, eng = cert_server("plain")
    body = {"id": "h", "examples": examples(40, 0.9)}
    r = post_task(client, {**body, "layout": "footer"})
    assert r.status_code == 422 and "'plain', 'header'" in r.json()["detail"]
    mixed = [{**e, "request": {**e["request"], "layout": "header"}} for e in examples(40, 0.9)]
    r = post_task(client, {"id": "h", "examples": mixed})
    assert r.status_code == 422 and "fitted under 'plain'" in r.json()["detail"] and not so.task_store.by_key
    r = post_task(client, {"id": "h", "layout": "header", "examples": mixed})  # an example that names the task's
    assert r.status_code == 200 and r.json()["fingerprint"] == FP["header"]
    assert post_task(client, {**body, "id": "d"}).json()["fingerprint"] == FP["plain"]  # the server's


def test_r4_the_dynamic_and_abstention_routes_register_under_a_layout():
    import test_dynamic_task as DT
    from test_task_persistence import abstention_body

    class DynFake(FollowsLayout, DT.FakeEngine):
        pass

    eng = DynFake().follows("header")
    client, so = server(eng, "header")
    r = client.post("/v1/tasks", json={"id": "brand", **DT.DYN, "examples": DT.examples(500), "certify": DT.CERT,
                                       "layout": "plain"})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["fingerprint"] == FP["plain"] and set(eng.reads) == {"plain"}
    c, so2, _ = cert_server("header")
    r = c.post("/v1/abstention/tasks", json={**abstention_body(), "layout": "plain"})
    assert r.status_code == 200 and "layout" not in so2.tasks["abs"]  # plain is written as it always was
    with lay.using("plain"):
        assert so2.abstention_applies(so2.tasks["abs"])
    assert not so2.abstention_applies(so2.tasks["abs"])  # not to the server's header requests
    r = c.post("/v1/abstention/tasks", json={**abstention_body("abs2")})
    assert r.status_code == 200 and so2.tasks["abs2"]["layout"] == "header"
    assert c.post("/v1/abstention/tasks", json={**abstention_body("x"), "layout": 1}).status_code == 422


# ---- R5 -------------------------------------------------------------------------------------------------------------


@pytest.fixture
def tables(monkeypatch):
    """The fast-table stand-ins of test_fast_table_task, test_fast_table_kinds and test_fast_table_learn (the virtual
    clock, the students, refits by hand), on a fake engine that follows the layout in effect."""
    pytest.importorskip("sklearn")
    pytest.importorskip("obstore")
    pytest.importorskip("tokenizers")
    import time as real_time

    import test_fast_table_kinds as K
    import test_fast_table_task as H

    from decisio.serve import fast_table as FT
    from decisio.serve import fast_table_learn as FTL
    from decisio.serve import systemone as SO

    H.CLOCK.t, H.CLOCK.decisio_ms, H.CLOCK.table_ms = 0.0, 60.0, 5.0
    proxy = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("_")})
    proxy.perf_counter = H.CLOCK
    monkeypatch.setattr(SO, "time", proxy)
    monkeypatch.setattr(FT, "encoder", lambda: object())
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-A")
    K.CALLS["static"].clear()
    K.CALLS["encoder"].clear()
    K.STATIC[0], K.STATIC_BLIND[0] = 1.0, False
    monkeypatch.setattr(FT, "static_embed", K.fake_static_embed)
    monkeypatch.setattr(FT, "static_model", lambda name: object())
    monkeypatch.setattr(FT, "embed", K.counting_embed)
    monkeypatch.setattr(FTL, "BACKGROUND", False)
    monkeypatch.setattr(FTL, "SEED", 0)

    class TableFake(FollowsLayout, H.FakeEngine):
        pass

    eng = TableFake().follows("plain")
    client, so = server(eng, "plain")
    return types.SimpleNamespace(H=H, client=client, so=so, eng=eng)


def table_ask(t, msg, layout=None):
    r = t.client.post("/v1/systemone", json={**t.H.wire(msg), **({"layout": layout} if layout else {})})
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


def test_r5_a_labelled_table_never_answers_a_request_of_the_other_layout(tables):
    t = tables
    r = t.client.post(
        "/v1/tasks",
        json={"id": "queue", "layout": "header", "examples": t.H.examples(800), "certify": {"alpha": 0.1, "delta": 0.1},
              "fast_table": {"kind": "encoder"}},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["fast_table_report"]["kept"]
    sure = t.H.message("billing", np.random.default_rng(1), right=False, conf=0.4)  # the table is sure of it
    assert table_ask(t, sure, "header")["answered_by"] == "table"
    calls = type(t.eng).calls
    a = table_ask(t, sure)  # the server's plain: decisio's own read, no table
    assert "answered_by" not in a and a["certified_reason"] == "layout_mismatch"
    assert type(t.eng).calls == calls + 1 and t.eng.reads[-1] == "plain"


def test_r5_a_table_that_learns_logs_no_request_of_the_other_layout(tables):
    t = tables
    q = {"type": "choice", "instructions": t.H.INS, "criteria": t.H.CRIT}
    r = t.client.post(
        "/v1/tasks", json={"id": "queue", "question": q, "layout": "header", "fast_table": {"learn": True}}
    )
    assert r.status_code == 200, r.text
    assert r.json()["fingerprint"] == FP["header"]

    def logged():
        return t.client.get("/v1/tasks/queue").json()["status"]["fast_table"]["rows"]["logged"]

    rng = np.random.default_rng(0)
    for _ in range(5):
        assert "answered_by" not in table_ask(t, t.H.message("billing", rng))  # plain: not the task's
    assert logged() == 0
    for _ in range(5):
        assert table_ask(t, t.H.message("billing", rng), "header")["answered_by"] == "decisio"
    assert logged() == 5
    L = t.so.learning.learners["queue"]
    assert L.layout == "header" and L.stack_sha() == t.so.learning.stack_sha("header")
    assert t.so.learning.stack_sha("header") != t.so.learning.stack_sha("plain") == t.so.learning.stack_sha()


# ---- R6 -------------------------------------------------------------------------------------------------------------


RECORDS = {
    "adapters": {"n1": "header", "n2": None},
    "repository": None,
    "temperatures_file": "plain",
    "flavor": {"name": "calibrated", "layout": "header"},
}


def test_r6_the_notes_say_what_was_trained_or_fitted_under_another_layout():
    assert lay.notes(RECORDS, "header", "n1") == [
        "the tuned repository records no layout: it was trained under the plain layout",
        "the temperatures (--temperatures-file) were fitted under the plain layout",
    ]
    assert lay.notes(RECORDS, "plain", "n1") == [
        "the adapter n1 was trained, and its temperatures fitted, under the header layout",
        "the flavor calibrated was fitted under the header layout",
    ]
    assert (
        lay.notes(RECORDS, "header", "n2")[0]
        == "the adapter n2 records no layout: it was trained under the plain layout"
    )
    assert lay.notes({"repository": "header"}, "header") == [] and lay.notes({}, "plain") == []


def test_r6_a_request_of_another_layout_is_served_as_asked_with_a_note():
    client, so, eng = cert_server("plain", layout_records={"adapters": {"n1": "header"}})
    eng.adapters = {"n1": object()}
    a = ask(client, adapter="n1")
    assert "the adapter n1 was trained, and its temperatures fitted, under the header layout" in a.json()["layout_note"]
    assert "layout_note" not in ask(client, adapter="n1", layout="header").json()
    assert "layout_note" not in ask(client).json() and "layout_note" not in ask(client, layout="header").json()


def test_r6_the_start_records_what_each_thing_recorded():
    args = types.SimpleNamespace(
        layout_tuned=True, layout_recorded="header", temperatures_file_facts={"layout": "plain"}, flavor_report=None
    )
    out = vllm_engine.layout_records(args, {"n1": "p", "n2": "q"}, {"n1": {"layout": "header"}})
    assert out == {"adapters": {"n1": "header", "n2": None}, "repository": "header", "temperatures_file": "plain"}
    base = types.SimpleNamespace(layout_tuned=False, flavor_report={"name": "commit"}, temperatures_file_facts=None)
    assert vllm_engine.layout_records(base, {}, {}) == {
        "adapters": {},
        "flavor": {"name": "commit", "layout": "header"},
    }


# ---- R7 -------------------------------------------------------------------------------------------------------------


def test_r7_the_image_route_refuses_the_field_and_says_why():
    client, so, _ = cert_server("header")
    r = client.post("/v1/systemone", json={**wire("0:0.9:x"), "layout": "plain"}, headers={"x-decisio-route": "image"})
    assert r.status_code == 422 and "image route builds its own prompt" in r.json()["detail"]
    with pytest.raises(lay.LayoutRefused, match="no layout"):
        so.answer(SystemOneRequest.model_validate(wire("0:0.9:x")), images=[object()], layout="header")


def test_r7_packed_mode_refuses_the_header():
    eng = RowEngine("plain", mode="packed")
    q = [{"kind": "choice", "instructions": "Which?", "options": ["a", "b"]}]
    r = TestClient(vllm_engine.make_app(eng)).post(
        "/v1/answer", json={"state": "s", "questions": q, "layout": "header"}
    )
    assert r.status_code == 422 and "packed mode reads the plain layout only" in r.json()["detail"]
    with lay.using("header"), pytest.raises(lay.LayoutRefused):
        eng.answer("s", q)


# ---- R8 -------------------------------------------------------------------------------------------------------------


def test_r8_the_combiner_takes_one_layout_per_engine_call():
    c = Combiner(types.SimpleNamespace(_lock=FairLock()), max_requests=16)
    for layout in ("plain", "header", "plain", "header", None):
        c._waiting.append(Waiting([("s", [])], None, None, False, layout))
    assert [w.layout for w in c.take()] == ["plain", "plain"]
    assert [w.layout for w in c.take()] == ["header", "header"]
    assert [w.layout for w in c.take()] == [None]


def test_r8_an_engine_call_builds_its_rows_under_its_requests_layout():
    eng, plain = RowEngine("header"), RowEngine("plain")
    q = [{"kind": "noul", "instructions": "Is it?"}]
    ((probs, _),) = eng._combined_call([Waiting([(STATE, q)], None, None, False, "plain")])
    assert eng.scored == [plain._prepare_separate(STATE, q)[0][0][0]] and lay.current() is None
    # through the engine thread: a batched server's answer is the serial plain server's
    eng.concurrency = "batched"
    with lay.using("plain"):
        got, _ = eng.answer(STATE, q)
    assert [p.tolist() for p in got] == [p.tolist() for p in plain.answer(STATE, q)[0]]
    eng.combiner.close()


# ---- R9 -------------------------------------------------------------------------------------------------------------


def test_r9_the_header_limit_applies_to_header_requests_only():
    many = {
        f"q{i}": {"type": "noul", "instructions": f"Question {i}: " + " ".join(f"d{i}x{j}" for j in range(20)) + "?"}
        for i in range(12)
    }
    eng = RowEngine("header")
    with lay.using("plain"):
        plain_row = max(len(ids) for ids, _ in eng._prepare_separate(STATE, [{"kind": "noul", "instructions": "x"}])[0])
    eng.over_limit_tokens = plain_row + 120  # the plain rows fit, the header's do not
    client, so = server(eng, "header")
    limited = client.post("/v1/systemone", json=request(questions=many))
    assert limited.status_code == 200 and limited.json()["header_limited"]["listed"] < 12
    plain = client.post("/v1/systemone", json=request("plain", many))
    assert plain.status_code == 200 and "header_limited" not in plain.json() and "x-decisio-layout" not in plain.headers
    assert so.header_limited["requests"] == 1
    with lay.using("plain"):
        assert eng.header_plan(STATE, [{"kind": "noul", "instructions": "x"}]) is None


# ---- R10 ------------------------------------------------------------------------------------------------------------


def test_r10_tasks_of_both_layouts_load_and_apply_at_start(launcher, tmp_path, capsys):  # noqa: F811
    from test_default_layout import plain_task_record

    from decisio.serve import task_backends as TB

    d, plain = plain_task_record(launcher, tmp_path)
    _, header_so = launcher("--base", str(d), "--layout", "header")
    header = {**plain, "id": "triage", "key": "another question", "fingerprint": header_so.task_store.fingerprint}
    backend = TB.DirBackend(tmp_path / "tasks")
    backend.save("readout", plain)
    backend.save("readout", header)
    backend.close()
    capsys.readouterr()
    engine, so = launcher("--base", str(d), "--tasks-dir", str(tmp_path / "tasks"))
    out = capsys.readouterr().out
    # the plain task still holds the default (its requests name no layout), and the start says the header can be asked
    assert engine.fmt.layout == "plain" and 'A request may ask for the header ("layout": "header")' in out
    assert "WARNING: tasks fitted under another model" not in out
    assert so.task_store_report()["readout"] == {"loaded": 2, "applied": 2, "other_stack": 0}
    assert {t["id"]: so.task_store.layout_of(t) for t in so.task_store.by_key.values()} == {
        "routing": "plain",
        "triage": "header",
    }
    with lay.using("header"):
        assert so.task_store.lookup("another question")["id"] == "triage" and so.task_store.lookup(plain["key"]) is None
    assert so.task_store.lookup(plain["key"])["id"] == "routing" and so.task_store.lookup("another question") is None
    so.task_backend.close()


# ---- R11 ------------------------------------------------------------------------------------------------------------


def test_r11_tasks_of_either_layout_import_and_another_models_are_refused():
    a, _, _ = cert_server("header")
    assert post_task(a, {"id": "h", "examples": examples(40, 0.9)}).status_code == 200
    exported = a.get("/v1/tasks?full=1").json()["tasks"]
    b, so_b, _ = cert_server("plain")
    r = b.post("/v1/tasks/import", json={"tasks": exported})
    assert r.status_code == 200 and r.json()["loaded"] == ["h"]
    assert ask(b, layout="header").headers["x-decisio-tasks"] == "h" and "x-decisio-tasks" not in ask(b).headers
    other = [{**exported[0], "fingerprint": "another model"}]
    r = b.post("/v1/tasks/import", json={"tasks": other})
    assert r.status_code == 422 and "another model or rendering" in r.json()["detail"]
    from test_task_persistence import abstention_body

    assert a.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    record = a.get("/v1/abstention/tasks").json()["tasks"][0]
    assert record["layout"] == "header"
    r = b.post("/v1/abstention/tasks/import", json={"tasks": [record]})
    assert r.status_code == 200 and r.json()["loaded"] == ["abs"]
    assert so_b.task_store_report()["readout"]["applied"] == 1


# ---- R12 ------------------------------------------------------------------------------------------------------------


def test_r12_a_header_certificate_holds_for_header_requests_on_a_plain_server():
    a, so_a, _ = cert_server("header")
    assert post_task(a, {"id": "c", "examples": examples(400, 0.96), "certify": {}}).status_code == 200
    exported = a.get("/v1/tasks?full=1").json()["tasks"]
    assert ask(a).json()["answers"]["q1"]["certified"] is True
    b, so_b, _ = cert_server("plain")
    assert b.post("/v1/tasks/import", json={"tasks": exported}).status_code == 200
    assert b.get("/v1/tasks/c").json()["status"]["state"] == "ok"
    assert ask(b, layout="header").json()["answers"]["q1"]["certified"] is True
    assert ask(b).json()["answers"]["q1"]["certified_reason"] == "layout_mismatch"
    # the stack is the layout's: the same on either server for one layout, and different between the layouts
    assert so_a.stack_under("header") == so_b.stack_under("header")
    assert so_b.stack_under("header") != so_b.stack_under("plain") == so_b.stack_facts()


# ---- R13 ------------------------------------------------------------------------------------------------------------


def test_r13_thinking_and_two_orders_read_under_the_requests_layout():
    client, so, eng = cert_server("plain")
    thought = []

    def think(state, questions, adapter, budget, subset):
        thought.append(eng.fmt.layout)
        return [np.log(np.full(len(questions[0]["options"]), 1 / 3))], [5], {"generate_ms": 1.0, "read_ms": 1.0}

    eng.can_think, eng.think = (lambda: True), think
    out = ask(client, "0:0.4:x", layout="header", think=True).json()
    assert out["thinking"]["questions"] == 1 and thought == ["header"] and eng.reads[-1] == "header"
    ask(client, "0:0.4:x", layout="header", orders=2)
    assert eng.reads[-1] == "header"


# ---- R3, on the CPU stand-in (slow) -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def served():
    from decisio.serve.hf_letters import HFLettersEngine

    return HFLettersEngine(MODEL)


@pytest.mark.parametrize("server_layout, asked", [("header", "plain"), ("plain", "header")])
def test_r3_the_stand_in_reads_the_other_servers_rows_token_for_token(served, server_layout, asked):
    eng = served
    rows = {}
    for name, fmt_layout, body_layout in (("asked", server_layout, asked), ("other", asked, None)):
        eng.fmt = PromptFormat(layout=fmt_layout)
        client = TestClient(vllm_engine.make_app(eng, SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))))
        start = len(eng.scored_rows)
        r = client.post("/v1/systemone", json=request(body_layout))
        assert r.status_code == 200, r.text
        rows[name] = (eng.scored_rows[start:], r.json())
    assert rows["asked"][0] == rows["other"][0] and rows["asked"][0]  # token ids and labels of every row
    assert rows["asked"][1] == rows["other"][1]  # answers and usage
    eng.fmt = PromptFormat()


def test_r3_a_task_registered_under_a_layout_on_the_stand_in_is_the_one_a_server_of_that_layout_fits(served):
    eng = served
    crit = {"a": "billing", "b": "shipping", "c": "access"}
    q = SystemOneRequest.model_validate(
        {"state": "", "questions": {"q": {"type": "choice", "instructions": "Which queue?", "criteria": crit}}}
    ).questions["q"]
    texts = ["my invoice is wrong", "where is my parcel", "I cannot log in", "refund my payment"] * 3
    ex = [(text, q, "abc"[i % 3]) for i, text in enumerate(texts)]
    fitted = {}
    for name, server_layout, kw in (("asked", "header", {"layout": "plain"}), ("plain", "plain", {})):
        eng.fmt = PromptFormat(layout=server_layout)
        so = SystemOne(eng, "decisio-test", task_store=TaskStore(FP[server_layout], layouts=FP))
        start = len(eng.scored_rows)
        task, _ = so.register_readout_task("t", ex, **kw)
        calibration = json.dumps(task["calibration"], sort_keys=True, default=lambda x: np.asarray(x).tolist())
        fitted[name] = (eng.scored_rows[start:], task["fingerprint"], calibration)
    assert fitted["asked"] == fitted["plain"] and fitted["plain"][1] == FP["plain"] and fitted["plain"][0]
    eng.fmt = PromptFormat()


def test_r10_the_start_line_of_a_held_server_names_the_field():
    d = lay.decide_layout(explicit=None, default="header", tasks=["routing"])
    assert d["value"] == "plain" and 'A request may ask for the header ("layout": "header")' in lay.describe(d)
    d = lay.decide_layout(explicit="header", default="header", tasks=["routing"])
    assert any('apply to the requests that ask for "layout": "plain"' in n for n in d["notes"])


def test_r12_the_task_store_without_layouts_has_its_one_fingerprint():
    store = TaskStore("fp")
    with lay.using("header"):
        assert store.fingerprint == "fp" and store.layout_of({"fingerprint": "fp"}) is None
    assert store.applies({"fingerprint": "fp"}) and not store.applies({"fingerprint": "other"})
    both = TaskStore(FP["plain"], layouts=FP)
    with lay.using("header"):
        assert both.fingerprint == FP["header"]
    assert both.fingerprint == FP["plain"] and both.layout_of({"fingerprint": FP["header"]}) == "header"


# ---- R14 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["plain", "header"])
def test_r14_one_question_under_each_layout_is_two_tasks(first):
    second = "header" if first == "plain" else "plain"
    client, so, eng = cert_server("header")
    for layout in (first, second):
        r = post_task(client, {"id": f"t-{layout}", "layout": layout, "examples": examples(400, 0.96), "certify": {}})
        assert r.status_code == 200, r.text
    tasks = sorted(t["id"] for t in so.task_store.by_key.values())
    assert tasks == ["t-header", "t-plain"]
    for layout in ("plain", "header"):
        hit = ask(client, layout=layout)
        assert hit.headers["x-decisio-tasks"] == f"t-{layout}"
        assert hit.json()["answers"]["q1"]["certified"] is True
    # a restart loads both records, in either order, and both apply again
    kept = list(so.task_store.by_key.values())
    for order in (kept, kept[::-1]):
        store = TaskStore(FP["header"], layouts=FP)
        for t in order:
            store.put(dict(t), persist=False)
        assert store.lookup(kept[0]["key"])["id"] == "t-header"  # the server's own layout
        with lay.using("plain"):
            assert store.lookup(kept[0]["key"])["id"] == "t-plain"
        with lay.using("header"):
            assert store.lookup(kept[0]["key"])["id"] == "t-header"
    # removing one leaves the other
    assert client.delete("/v1/tasks/t-plain").status_code == 200
    assert ask(client, layout="header").headers["x-decisio-tasks"] == "t-header"
    miss = ask(client, layout="plain").json()["answers"]["q1"]
    assert miss["certified"] is False and miss["certified_reason"] == "layout_mismatch"
