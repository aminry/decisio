# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task's context (`"context"` on `POST /v1/tasks`; `"task"` on `/v1/systemone`; docs/tasks.md, "Task context").

X1  the prompt, token for token: a request of a task with a context reads the rows of the same request with the context
    put before its state (a blank line between), under both layouts, the header before the context; the registration's
    examples are read the same way; a task without one reads exactly as before
X2  the field: a string as given, an object or array as compact JSON; empty or of another type refused; a task without
    one is stored and keyed as before
X3  identity: the context's sha256 is in the record, the slot and the fit digest; the same question with another context
    is another task beside it; a request naming no task gets the one without a context, and is refused when several
    with contexts ask its question; a certificate made under one context does not hold under another
X4  the selector `"task"`: it picks the task, says so in x-decisio-tasks; an unknown id, a task of another question, a
    task of the other layout and /v1/answer are refused; an example names none
X5  one context per request: questions whose tasks have different contexts are refused
X6  limits: a context over the ceiling before any state refuses the registration; a row over it is refused with where
    its tokens are (plain and header); truncation cuts the state, never the context
X7  status: the task, the listing and /health give the context's length and sha256, never its text; the export holds it
X8  persistence: a task directory and a shared store (obstore's in-memory store) keep the context across a restart; an
    import without the text, or with a text that is not its sha256's, is refused
X9  the fast table encodes the state only, a learning task logs the state only, and the certificate is read with the
    context (the served check through the serving path applies the task)
X10 usage counts the context
X11 a state that already starts with the context is served, with x-decisio-warning, logged once per task

  uv run pytest -q tests/unit/test_task_context.py
"""

import hashlib
import json
import logging
import re
import time
import types

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import CRIT, examples, post_task, wire
from test_layout_flags import StubTok
from test_request_layout import FP, RowEngine, server

from decisio.readout.letters import CONTEXT_SEP, QUESTIONS_HEAD, fmt_state
from decisio.serve import fast_table as FT
from decisio.serve import fast_table_learn as FTL
from decisio.serve import task_backends as TB
from decisio.serve.systemone import CONTEXT_ELSEWHERE, CONTEXT_TWICE, SystemOne, SystemOneRequest, add_routes
from decisio.serve.tasks import TaskStore, fit_digest

CONTEXT = "Refund policy: a refund is due within thirty days of purchase, for any reason, once per customer."
OTHER = "Escalation policy: route every message that names a lawyer to the legal queue before anything else."


class CtxEngine:
    """The fake engine of tests/unit/test_certified_answers.py (the state's `top:confidence` sets the answer), with a
    chat tokenizer and an optional prompt ceiling, keeping every state it reads."""

    multi_question = "sequential"
    pad_policy = "always"

    def __init__(self, limit=None):
        self.tok = StubTok()
        self.limit = limit
        self.states = []

    def facts(self):
        return {"engine": "fake", "vllm": "1.0", "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def prompt_limit(self):
        return self.limit

    def answer(self, state, questions, adapter=None):
        self.states.append(state)
        top, conf = re.findall(r"(\d+):(\d\.\d+)", fmt_state(state))[-1]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top) % k] = float(conf)
            out.append(p)
        return out, {}


def make(engine=None, store=None, backend=None, report=None):
    eng = engine or CtxEngine()
    ts = store or TaskStore("fp")
    so = SystemOne(eng, "decisio-test", task_store=ts, task_backend=backend, task_report=report)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def ask(client, text="0:0.95:x", **extra):
    return client.post("/v1/systemone", json={**wire(text), **extra})


def pieces(engine, ids):
    """A row's tokens as text pieces: each stand-in tokenizer numbers the pieces it meets in its own order."""
    return [engine.tok.rev[i] for i in ids]


def read_with(context, state):
    """The state as the documented rule reads it: the context, a blank line, the rendered state."""
    return f"{context}{CONTEXT_SEP}{fmt_state(state)}"


# ---- X1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["plain", "header"])
def test_x1_a_context_task_reads_the_rows_of_the_context_put_before_the_state_token_for_token(layout):
    eng = RowEngine(layout)
    client, so = server(eng, layout)
    eng.scored.clear()
    body = {"id": "t", "examples": examples(12, 0.9), "context": CONTEXT}
    assert post_task(client, body).status_code == 200
    registered = list(eng.scored)
    state = {"message": "0:0.9:x"}
    r = client.post("/v1/systemone", json={**wire("0:0.9:x")})
    assert r.status_code == 200 and r.headers["x-decisio-tasks"] == "t"
    served = eng.scored[-1]

    bare = RowEngine(layout)
    plain_client, _ = server(bare, layout)
    first = examples(12, 0.9)[0]["request"]
    r0 = plain_client.post("/v1/systemone", json={**first, "state": read_with(CONTEXT, first["state"])})
    # the registration read its examples the same way
    assert r0.status_code == 200 and pieces(bare, bare.scored[-1]) == pieces(eng, registered[0])
    r1 = plain_client.post("/v1/systemone", json={**wire("0:0.9:x"), "state": read_with(CONTEXT, state)})
    assert r1.status_code == 200 and pieces(bare, bare.scored[-1]) == pieces(eng, served)
    text = eng.tok.decode(served)
    at = [text.find(CONTEXT), text.find("0:0.9:x"), text.rfind("Which queue?")]  # the question after the state
    assert -1 not in at and at == sorted(at)
    assert CONTEXT + CONTEXT_SEP + "message: 0:0.9:x" in text
    if layout == "header":
        assert -1 < text.find(QUESTIONS_HEAD) < at[0]  # [questions][context][state][question]
    else:
        assert QUESTIONS_HEAD not in text


def test_x1_a_task_without_a_context_reads_exactly_as_before():
    eng = RowEngine("plain")
    client, so = server(eng, "plain")
    assert post_task(client, {"id": "t", "examples": examples(12, 0.9)}).status_code == 200
    client.post("/v1/systemone", json=wire("0:0.9:x"))
    bare = RowEngine("plain")
    plain_client, _ = server(bare, "plain")
    plain_client.post("/v1/systemone", json=wire("0:0.9:x"))
    assert pieces(eng, eng.scored[-1]) == pieces(bare, bare.scored[-1])


# ---- X2 -------------------------------------------------------------------------------------------------------------


def test_x2_a_string_as_given_an_object_or_array_as_compact_json():
    client, so, _ = make()
    obj = {"policy": "refunds within thirty days", "réf": [1, 2]}
    r = post_task(client, {"id": "o", "examples": examples(40, 0.9), "context": obj})
    assert r.status_code == 200, r.text
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    assert so.task_store.by_id("o")["context"]["text"] == text == '{"policy":"refunds within thirty days","réf":[1,2]}'
    assert r.json()["context"]["chars"] == len(text) and "text" not in r.json()["context"]
    for bad in ("", "   ", 3, True):
        r = post_task(client, {"id": "b", "examples": examples(40, 0.9), "context": bad})
        assert r.status_code == 422 and "context" in r.json()["detail"]
    assert so.task_store.by_id("b") is None


def test_x2_the_dynamic_option_registration_takes_it_and_reads_its_examples_with_it():
    client, so, eng = make()
    body = {"id": "d", "options": "dynamic", "examples": examples(400, 0.96), "certify": {}, "context": CONTEXT}
    r = post_task(client, body)
    assert r.status_code == 200, r.text
    assert r.json()["dynamic"] and r.json()["context"]["sha256"] == hashlib.sha256(CONTEXT.encode()).hexdigest()
    assert eng.states and all(s == read_with(CONTEXT, {"message": s.rsplit(": ", 1)[1]}) for s in eng.states)
    assert ask(client).headers["x-decisio-tasks"] == "d" and eng.states[-1] == read_with(
        CONTEXT, {"message": "0:0.95:x"}
    )


def test_x2_a_task_without_one_is_stored_and_keyed_as_before():
    client, so, _ = make()
    t = post_task(client, {"id": "t", "examples": examples(40, 0.9), "context": None}).json()
    assert "context" not in t and list(so.task_store.by_key) == [t["key"]]
    stored = so.task_store.by_id("t")
    assert fit_digest(stored) == fit_digest({k: v for k, v in stored.items() if k != "context"})


# ---- X3 -------------------------------------------------------------------------------------------------------------


def test_x3_the_same_question_with_another_context_is_another_task_beside_it():
    client, so, eng = make()
    for tid, ctx in (("a", CONTEXT), ("b", OTHER), ("c", None)):
        body = {"id": tid, "examples": examples(400, 0.96), "certify": {}, **({"context": ctx} if ctx else {})}
        assert post_task(client, body).status_code == 200
    store = so.task_store
    a, b, c = (store.by_id(x) for x in "abc")
    assert a["key"] == b["key"] == c["key"] and len(store.by_key) == 3
    assert {store.slot(a), store.slot(b), store.slot(c)} == {
        f"{a['key']}~{a['context']['sha256']}",
        f"{a['key']}~{b['context']['sha256']}",
        a["key"],
    }
    assert len({fit_digest(a), fit_digest(b), fit_digest(c)}) == 3
    assert a["certificate"]["stack"]["calibration"]["task_fit"] == fit_digest(a)
    r = ask(client)  # no task named: the one without a context, as before contexts existed
    assert r.headers["x-decisio-tasks"] == "c" and eng.states[-1] == {"message": "0:0.95:x"}
    for tid, ctx in (("a", CONTEXT), ("b", OTHER)):
        r = ask(client, task=tid)
        assert r.status_code == 200 and r.headers["x-decisio-tasks"] == tid
        assert eng.states[-1] == read_with(ctx, {"message": "0:0.95:x"})
        assert r.json()["answers"]["q1"]["certificate"] == store.by_id(tid)["certificate"]["id"]
    client.delete("/v1/tasks/c")
    r = ask(client)  # two tasks with a context and none without: which one cannot be guessed
    assert r.status_code == 422 and "'a', 'b'" in r.json()["detail"] and '"task"' in r.json()["detail"]
    client.delete("/v1/tasks/b")
    assert ask(client).headers["x-decisio-tasks"] == "a"  # one task asks it: found by its question
    assert eng.states[-1] == read_with(CONTEXT, {"message": "0:0.95:x"})


def test_x3_registering_an_id_again_with_another_context_moves_it_and_its_certificate():
    client, so, _ = make()
    one = post_task(client, {"id": "a", "examples": examples(400, 0.96), "certify": {}, "context": CONTEXT}).json()
    two = post_task(client, {"id": "a", "examples": examples(400, 0.96), "certify": {}, "context": OTHER}).json()
    assert len(so.task_store.by_key) == 1 and two["context"]["sha256"] != one["context"]["sha256"]
    assert two["certificate"]["id"] != one["certificate"]["id"]


def test_x3_a_certificate_made_under_one_context_does_not_hold_under_another():
    client, so, _ = make()
    assert (
        post_task(client, {"id": "a", "examples": examples(400, 0.96), "certify": {}, "context": CONTEXT}).status_code
        == 200
    )
    rec = client.get("/v1/tasks?full=1").json()["tasks"][0]
    rec["context"] = {**rec["context"], "text": OTHER}
    rec["context"]["sha256"] = hashlib.sha256(OTHER.encode()).hexdigest()
    other, so2, _ = make()
    assert other.post("/v1/tasks/import", json={"tasks": [rec]}).status_code == 200
    a = ask(other, "0:0.99:x").json()["answers"]["q1"]
    assert a["certified"] is False and a["certified_reason"] == "serving_stack_changed"
    st = other.get("/v1/tasks/a").json()["status"]
    assert st["state"] == "suspended" and "calibration.task_fit" in st["changed"]
    same, _, _ = make()  # the record as exported: the certificate holds
    same.post("/v1/tasks/import", json={"tasks": client.get("/v1/tasks?full=1").json()["tasks"]})
    assert ask(same, "0:0.99:x").json()["answers"]["q1"]["certified"] is True


# ---- X4 -------------------------------------------------------------------------------------------------------------


def test_x4_the_selector_refuses_what_it_cannot_apply():
    client, so, _ = make()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    r = ask(client, task="nope")
    assert r.status_code == 422 and "no registered task 'nope'" in r.json()["detail"]
    other = {"option_0": "billing", "option_1": "shipping"}
    r = client.post("/v1/systemone", json={**wire("0:0.9:x", crit=other), "task": "a"})
    assert r.status_code == 422 and "asks another question" in r.json()["detail"]
    body = {"id": "x", "examples": [{**e, "request": {**e["request"], "task": "a"}} for e in examples(40, 0.9)]}
    r = post_task(client, body)
    assert r.status_code == 422 and "example 0" in r.json()["detail"] and "task" in r.json()["detail"]


def test_x4_a_task_of_the_other_layout_is_refused_and_says_which_layout():
    eng = CtxEngine()
    client, so, _ = make(eng, store=TaskStore(FP["plain"], layouts=FP))
    assert (
        post_task(
            client, {"id": "h", "layout": "header", "examples": examples(40, 0.9), "context": CONTEXT}
        ).status_code
        == 200
    )
    r = ask(client, task="h")
    assert r.status_code == 422 and '"layout": "header"' in r.json()["detail"]
    r = ask(client, task="h", layout="header")
    assert r.status_code == 200 and r.headers["x-decisio-tasks"] == "h"
    assert eng.states[-1] == read_with(CONTEXT, {"message": "0:0.95:x"})
    r = ask(client)  # the plain request finds no task of its layout: decisio's own read, no context, and a warning
    assert r.status_code == 200 and "x-decisio-tasks" not in r.headers
    assert r.headers["x-decisio-warning"] == CONTEXT_ELSEWHERE.format(id="h", layout="header")
    assert eng.states[-1] == {"message": "0:0.95:x"}


def test_x4_the_engines_route_takes_no_task():
    eng = RowEngine("plain")
    client, so = server(eng, "plain")
    q = [{"kind": "choice", "instructions": "Which?", "options": ["a", "b"]}]
    r = client.post("/v1/answer", json={"state": "s", "questions": q, "task": "a"})
    assert r.status_code == 422 and "/v1/systemone" in r.json()["detail"]


# ---- X5 -------------------------------------------------------------------------------------------------------------


def test_x5_questions_of_one_request_share_one_context():
    client, so, eng = make()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    body = wire("0:0.9:x")
    body["questions"]["q2"] = {"type": "noul", "instructions": "Is the customer angry?"}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422 and "one context" in r.json()["detail"] and "'q2': no task" in r.json()["detail"]
    body["questions"]["q2"] = {**body["questions"]["q1"]}  # the same task twice: one context
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200 and eng.states[-1] == read_with(CONTEXT, {"message": "0:0.9:x"})


# ---- X6 -------------------------------------------------------------------------------------------------------------


def test_x6_a_context_over_the_ceiling_before_any_state_refuses_the_registration():
    client, so, _ = make(CtxEngine(limit=60))
    r = post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": " ".join(["word"] * 80)})
    assert r.status_code == 422 and "context: 80 tokens" in r.json()["detail"] and "60" in r.json()["detail"]
    assert so.task_store.by_id("a") is None


def test_x6_a_plain_row_over_the_ceiling_says_where_its_tokens_are():
    eng = CtxEngine()
    client, so, _ = make(eng)
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    eng.limit = 120
    r = ask(client, "0:0.9:x " + " ".join(["filler"] * 200))
    detail = r.json()["detail"]
    assert r.status_code == 422 and f"the task's context {len(CONTEXT.split())}" in detail
    assert "the state 20" in detail and "shorten the state, the task's context or the question" in detail
    assert ask(client).status_code == 200  # a short state fits


def test_x6_a_header_row_over_the_ceiling_with_the_limit_off_counts_the_context_apart():
    eng = RowEngine("header")
    client, so = server(eng, "header")
    assert post_task(client, {"id": "a", "examples": examples(12, 0.9), "context": CONTEXT}).status_code == 200
    eng.over_limit_tokens, eng.header_limit_setting = 100, ("off", None)
    r = client.post("/v1/systemone", json=wire("0:0.9:x " + " ".join(["filler"] * 200)))
    detail = r.json()["detail"]
    assert r.status_code == 422 and "the header" in detail and "the task's context" in detail


def test_x6_truncation_cuts_the_state_and_never_the_context():
    eng = CtxEngine()
    client, so, _ = make(eng)
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    eng.limit = 160
    r = ask(client, "0:0.9:x " + " ".join(["filler"] * 300), over_limit="truncate")
    assert r.status_code == 200, r.text
    cut = r.json()["state_truncated"]
    read = eng.states[-1]
    assert read.startswith(CONTEXT + CONTEXT_SEP) and cut["kept_tokens"] < cut["of_tokens"]
    eng.limit = 30  # the context and the question alone are over: refused, and the context is named
    r = ask(client, "0:0.9:x " + " ".join(["filler"] * 300), over_limit="truncate")
    assert r.status_code == 422 and "context" in r.json()["detail"]


# ---- X7 -------------------------------------------------------------------------------------------------------------


def test_x7_status_gives_the_length_and_digest_never_the_text_and_the_export_holds_it():
    eng = RowEngine("plain")
    client, so = server(eng, "plain")
    t = post_task(client, {"id": "a", "examples": examples(12, 0.9), "context": CONTEXT}).json()
    want = {"sha256": hashlib.sha256(CONTEXT.encode()).hexdigest(), "chars": len(CONTEXT)}
    for got in (
        t["context"],
        client.get("/v1/tasks/a").json()["context"],
        client.get("/v1/tasks").json()["tasks"][0]["context"],
    ):
        assert {k: got[k] for k in want} == want and got["tokens"] == len(eng.tok.encode(CONTEXT))
    for path in ("/v1/tasks/a", "/v1/tasks"):
        assert CONTEXT not in client.get(path).text
    health = client.get("/health")
    assert health.status_code == 200 and CONTEXT not in health.text
    assert health.json()["systemone"]["readout_tasks"]["a"]["context"]["sha256"] == want["sha256"]
    assert client.get("/v1/tasks?full=1").json()["tasks"][0]["context"]["text"] == CONTEXT


# ---- X8 -------------------------------------------------------------------------------------------------------------


def test_x8_a_task_directory_keeps_the_context_across_a_restart(tmp_path):
    def start():
        backend = TB.DirBackend(tmp_path)
        ts = TaskStore("fp")
        _, report = TB.load_at_start(ts, backend)
        return make(store=ts, backend=backend, report=report)

    client, first, _ = start()
    assert (
        post_task(client, {"id": "a", "examples": examples(400, 0.96), "certify": {}, "context": CONTEXT}).status_code
        == 200
    )
    first.task_backend.close()  # the server stops: one directory belongs to one server
    again, so, eng = start()
    assert so.task_store.by_id("a")["context"]["text"] == CONTEXT
    a = ask(again, "0:0.99:x").json()["answers"]["q1"]
    assert a["certified"] is True and eng.states[-1] == read_with(CONTEXT, {"message": "0:0.99:x"})


def test_x8_a_shared_store_keeps_the_context_across_a_restart():
    pytest.importorskip("obstore")
    from obstore.store import MemoryStore

    from decisio.serve import task_store_shared as S

    mem = MemoryStore()

    def start():
        backend = S.ObjectBackend(mem, "memory://t")
        ts = TaskStore("fp")
        _, report = TB.load_at_start(ts, backend)
        return make(store=ts, backend=backend, report=report)

    client, _, _ = start()
    assert (
        post_task(client, {"id": "a", "examples": examples(400, 0.96), "certify": {}, "context": CONTEXT}).status_code
        == 200
    )
    again, so, eng = start()
    assert so.task_store.by_id("a")["context"]["text"] == CONTEXT
    assert ask(again, "0:0.99:x").json()["answers"]["q1"]["certified"] is True
    assert eng.states[-1] == read_with(CONTEXT, {"message": "0:0.99:x"})


def test_x8_an_import_without_the_text_or_with_another_text_is_refused():
    client, so, _ = make()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    listed = client.get("/v1/tasks").json()["tasks"][0]  # the listing: no text
    other, _, _ = make()
    r = other.post("/v1/tasks/import", json={"tasks": [listed]})
    assert r.status_code == 422 and "without its text" in r.json()["detail"]
    full = client.get("/v1/tasks?full=1").json()["tasks"][0]
    full["context"]["text"] = OTHER
    r = other.post("/v1/tasks/import", json={"tasks": [full]})
    assert r.status_code == 422 and "sha256" in r.json()["detail"]


# ---- X9 -------------------------------------------------------------------------------------------------------------


@pytest.fixture
def table(monkeypatch):
    """The fast table's stand-ins and virtual clock (tests/unit/test_fast_table_task.py), and every text the students
    featurize."""
    pytest.importorskip("sklearn")
    import test_fast_table_task as F

    from decisio.serve import systemone as SO

    F.CLOCK.t, F.CLOCK.decisio_ms, F.CLOCK.table_ms = 0.0, 60.0, 5.0
    proxy = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
    proxy.perf_counter = F.CLOCK
    monkeypatch.setattr(SO, "time", proxy)
    monkeypatch.setattr(FT, "embed", F.fake_embed)
    monkeypatch.setattr(FT, "encoder", lambda: object())
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-A")
    seen = []
    real = FT.featurize

    def featurize(f, texts):
        seen.extend(texts)
        return real(f, texts)

    monkeypatch.setattr(FT, "featurize", featurize)

    class TableEngine(F.FakeEngine):
        """test_fast_table_task's engine, reading the state after the context."""

        def __init__(self):
            self.states = []

        def answer(self, state, questions, adapter=None):
            self.states.append(state)
            if isinstance(state, str) and CONTEXT_SEP in state:
                state = state.rsplit(CONTEXT_SEP, 1)[1]
            return super().answer(state, questions, adapter)

    eng = TableEngine()
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return F, TestClient(app), so, eng, seen


def test_x9_the_table_encodes_the_state_only_and_the_certificate_is_read_with_the_context(table):
    F, client, so, eng, seen = table
    r = F.register(client, context=CONTEXT)
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["fast_table_report"]["kept"] and t["certificate"]["certified"]
    assert seen and not any(CONTEXT in s for s in seen) and "warm" in seen  # the examples' states alone
    assert eng.states and all(s.startswith(CONTEXT + CONTEXT_SEP) for s in eng.states)  # decisio read the context
    assert t["certificate"]["served_check"]["task_applied"] == t["certificate"]["served_check"]["n"]
    seen.clear()
    msg = F.message("billing", np.random.default_rng(1), right=False, conf=0.4)
    a = client.post("/v1/systemone", json=F.wire(msg)).json()["answers"]["q1"]
    assert a["answered_by"] == "table" and seen == [msg]


def test_x9_a_learning_task_logs_the_state_only(table, monkeypatch):
    F, client, so, eng, seen = table
    monkeypatch.setattr(FTL, "BACKGROUND", False)
    logged = []
    real = FTL.Learner.record

    def record(self, text, *a, **k):
        logged.append(text)
        return real(self, text, *a, **k)

    monkeypatch.setattr(FTL.Learner, "record", record)
    body = {
        "id": "learn",
        "question": {"type": "choice", "instructions": F.INS, "criteria": F.CRIT},
        "fast_table": {"learn": True, "kind": "encoder"},
        "context": CONTEXT,
    }
    r = client.post("/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["context"]["sha256"] and "text" not in r.json()["context"]
    msgs = [F.message("billing", np.random.default_rng(i)) for i in range(3)]
    for m in msgs:
        assert client.post("/v1/systemone", json=F.wire(m)).status_code == 200
    assert logged == msgs and eng.states[-1] == read_with(CONTEXT, msgs[-1])


# ---- X10 ------------------------------------------------------------------------------------------------------------


def test_x10_usage_counts_the_context():
    client, so, eng = make()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    with_ctx = ask(client).json()["usage"]["input_tokens"]
    bare, _, _ = make()
    as_state = bare.post(
        "/v1/systemone", json={**wire("0:0.95:x"), "state": read_with(CONTEXT, {"message": "0:0.95:x"})}
    )
    without = bare.post("/v1/systemone", json=wire("0:0.95:x")).json()["usage"]["input_tokens"]
    assert with_ctx == as_state.json()["usage"]["input_tokens"] and with_ctx > without


# ---- X11 ------------------------------------------------------------------------------------------------------------


def test_x11_a_state_that_starts_with_the_context_is_served_warned_and_logged_once(caplog):
    client, so, eng = make()
    assert post_task(client, {"id": "a", "examples": examples(40, 0.9), "context": CONTEXT}).status_code == 200
    twice = read_with(CONTEXT, {"message": "0:0.9:x"})
    with caplog.at_level(logging.WARNING, logger="decisio.serve.systemone"):
        for _ in range(3):
            r = client.post("/v1/systemone", json={**wire("0:0.9:x"), "state": "  " + twice})
            assert r.status_code == 200 and r.headers["x-decisio-warning"] == CONTEXT_TWICE
            assert eng.states[-1] == read_with(CONTEXT, "  " + twice)  # served as sent: read twice
    assert sum("starts with the task's context" in m for m in caplog.messages) == 1
    assert "x-decisio-warning" not in ask(client).headers


def test_x4_the_selector_is_an_extension_field_and_must_be_a_string():
    req = SystemOneRequest.model_validate({**wire("0:0.9:x"), "task": "a"})
    assert req.model_extra["task"] == "a" and "task" not in SystemOneRequest.model_fields  # System One's fields stay
    client, so, _ = make()
    r = ask(client, task=3)
    assert r.status_code == 422 and '"task" must be' in r.json()["detail"]
    assert ask(client, task=None).status_code == 200 and CRIT
