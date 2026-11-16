# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A declared question list (`"header_questions"`; decisio.serve.header_questions; docs/running.md, "The header
layout").

Q1  the prefix: two requests about one state with different single questions and one declared list build rows that
    share the header and the state; without it they share neither (the MLX prefix cache: test_header_questions_mlx.py)
Q2  no list: the rows are the ones the request's own questions make; a list equal to them makes the same rows
Q3  the rules: a question not in the list, an empty list, a duplicate, a non-string, an empty line: a 422
Q4  the plain layout: the field is checked and has no effect, the rows are the same with and without it
Q5  determinism: the same request, the same rows, whatever came before it
Q6  the header limit: a declared header is read whole or refused, never cut
Q7  a task's context: the boundary is built from the declared header
Q8  usage counts the declared header; /v1/answer; the image route and a registration example refuse the field

  uv run pytest -q tests/unit/test_header_questions.py
"""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import examples, post_task
from test_header_limit import HeaderEngine
from test_over_limit import WordTok, plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio.readout.letters import DECLARED_KEY, context_state, header_instructions, layout_header
from decisio.serve import header_limit as hl
from decisio.serve import header_questions as hq
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

LE = vllm_engine.LettersEngine
STATE = {"message": "my invoice was charged twice and nobody answered my first mail"}
LINES = [
    "Which queue should this go to?",
    "Is the customer angry?",
    "Does the message ask for a refund?",
]


def wire_q(i):
    """The wire question that has LINES[i] as its instructions."""
    return {"type": "noul", "instructions": LINES[i]}


def engine_q(i):
    return {"kind": "noul", "instructions": LINES[i]}


def body(which, declared=None, state=STATE, **extra):
    out = {"state": state, "model": "m", "questions": {f"q{i}": wire_q(i) for i in which}, **extra}
    if declared is not None:
        out[hq.FIELD] = declared
    return out


class Recording(HeaderEngine):
    """The stand-in engine with the real row builder; every read builds its rows and keeps them."""

    def __init__(self, limit=100_000, layout="header", **kw):
        super().__init__(limit=limit, layout=layout, **kw)
        self.rows, self.declared = [], []

    def answer(self, state, questions, adapter=None, images=None):
        rows, n, _ = self._build_rows(state, questions)
        self.rows.append(([ids for ids, _ in rows], n))
        self.declared.append([q.get(DECLARED_KEY) for q in questions])
        return [np.array([0.5, 0.5]) for _ in questions], {}


def make(layout="header", **kw):
    eng = Recording(layout=layout, **kw)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# ---- Q1 -------------------------------------------------------------------------------------------------------------


def test_q1_one_declared_list_gives_every_request_about_a_state_the_same_prefix():
    client, _, eng = make()
    for which in ([0], [1], [2]):
        assert client.post("/v1/systemone", json=body(which, LINES)).status_code == 200
    (r0, n0), (r1, n1), (r2, n2) = eng.rows
    assert n0 == n1 == n2  # the header and the state, up to the question
    assert r0[0][:n0] == r1[0][:n1] == r2[0][:n2]
    assert layout_header(LINES) in eng.tok.decode(r0[0][:n0])


def test_q1_without_the_list_the_second_request_shares_only_the_text_before_the_header_differs():
    # the old behaviour, which the field exists to fix: each request lists its own question, so the prefix breaks in
    # the header and the state is read again
    client, _, eng = make()
    client.post("/v1/systemone", json=body([0]))
    client.post("/v1/systemone", json=body([1]))
    (r0, n0), (r1, n1) = eng.rows
    assert common(r0[0], r1[0]) < n0 and common(r0[0], r1[0]) < n1


def test_q1_the_header_lists_exactly_the_declared_lines_in_the_declared_order_on_every_row():
    client, _, eng = make()
    declared = [LINES[2], LINES[0], LINES[1]]  # not the request's order
    assert client.post("/v1/systemone", json=body([0, 1], declared)).status_code == 200
    rows, n = eng.rows[0]
    head = eng.tok.decode(rows[0][:n])
    assert layout_header(declared) in head and head.index(LINES[2]) < head.index(LINES[0]) < head.index(LINES[1])
    assert rows[0][:n] == rows[1][:n]
    assert eng.declared[0] == [declared, declared]


# ---- Q2 -------------------------------------------------------------------------------------------------------------


def test_q2_without_the_list_the_rows_are_the_requests_own():
    eng = Recording()
    qs = [engine_q(0), engine_q(1)]
    rows, n, _ = LE._build_rows(eng, STATE, qs)
    ref, ref_n, _ = LE._rows(eng, STATE, qs, None)
    assert rows == ref and n == ref_n
    assert header_instructions(qs) == LINES[:2]
    assert layout_header(LINES[:2]) in eng.tok.decode(rows[0][0][:n])


def test_q2_a_list_equal_to_the_requests_questions_makes_the_same_rows():
    eng = Recording()
    qs = [engine_q(0), engine_q(1)]
    plain = LE._build_rows(eng, STATE, qs)[:2]
    declared = LE._build_rows(eng, STATE, hq.stamp(qs, tuple(LINES[:2])))[:2]
    assert plain == declared


def test_q2_a_request_without_the_field_reaches_the_engine_unchanged():
    client, _, eng = make()
    assert client.post("/v1/systemone", json=body([0, 1])).status_code == 200
    assert eng.declared == [[None, None]]


# ---- Q3 -------------------------------------------------------------------------------------------------------------


def test_q3_a_question_that_is_not_in_the_list_is_a_422_that_names_it():
    client, _, eng = make()
    r = client.post("/v1/systemone", json=body([0, 1], [LINES[0]]))
    assert r.status_code == 422 and "'q1'" in r.json()["detail"] and LINES[1] in r.json()["detail"]
    assert eng.rows == []  # refused before any read


@pytest.mark.parametrize(
    "bad,says",
    [
        ([], "non-empty list"),
        ("Which queue?", "non-empty list"),
        ({"a": 1}, "non-empty list"),
        ([LINES[0], 3], "must be a string"),
        ([LINES[0], None], "must be a string"),
        ([LINES[0], "   "], "is empty"),
        ([LINES[0], LINES[1], LINES[0]], "repeats line 0"),
        ([LINES[0], f"  {LINES[0]} "], "repeats line 0"),  # the same once stripped
    ],
)
def test_q3_a_list_that_is_not_one_is_a_422(bad, says):
    client, _, eng = make()
    r = client.post("/v1/systemone", json={**body([0]), hq.FIELD: bad})
    assert r.status_code == 422 and says in r.json()["detail"], r.text
    assert eng.rows == []


def test_q3_null_is_no_list_and_a_line_is_matched_once_stripped():
    client, _, eng = make()
    assert client.post("/v1/systemone", json={**body([0]), hq.FIELD: None}).status_code == 200
    assert client.post("/v1/systemone", json=body([0], [f"  {LINES[0]}\n", LINES[1]])).status_code == 200
    assert eng.declared == [[None], [[LINES[0], LINES[1]]]]


def test_q3_a_yes_no_question_with_criteria_is_declared_by_its_instructions_as_sent():
    client, _, eng = make()
    q = {"type": "noul", "instructions": LINES[1], "criteria": {"true": "shouting", "false": "calm"}}
    r = client.post("/v1/systemone", json={"state": STATE, "model": "m", "questions": {"q": q}, hq.FIELD: [LINES[1]]})
    assert r.status_code == 200, r.text


# ---- Q4 -------------------------------------------------------------------------------------------------------------


def test_q4_under_the_plain_layout_the_field_changes_nothing():
    client, _, eng = make("plain")
    client.post("/v1/systemone", json=body([0]))
    client.post("/v1/systemone", json=body([0], LINES))
    assert eng.rows[0] == eng.rows[1] and eng.declared[1] == [None]
    # and it is still checked: the same request is a 422 under both layouts
    assert client.post("/v1/systemone", json=body([0, 1], [LINES[0]])).status_code == 422


def test_q4_a_header_server_asked_for_plain_per_request_ignores_it_too():
    client, _, eng = make("header")
    client.post("/v1/systemone", json=body([0], layout="plain"))
    client.post("/v1/systemone", json=body([0], LINES, layout="plain"))
    assert eng.rows[0] == eng.rows[1]


# ---- Q5 -------------------------------------------------------------------------------------------------------------


def test_q5_the_same_request_makes_the_same_rows_whatever_came_before():
    client, _, eng = make()
    client.post("/v1/systemone", json=body([1], LINES))
    client.post("/v1/systemone", json=body([0, 2], [LINES[2], LINES[0], LINES[1]]))
    client.post("/v1/systemone", json=body([0], LINES))
    other, _, eng2 = make()
    other.post("/v1/systemone", json=body([0], LINES))
    assert eng.rows[2] == eng2.rows[0]
    client.post("/v1/systemone", json=body([0], LINES))
    assert eng.rows[2] == eng.rows[3]


# ---- Q6 -------------------------------------------------------------------------------------------------------------


def long_lines(n=12, words=14):
    return [f"Question {i}: " + " ".join(f"detail{i}x{j}" for j in range(words)) + "?" for i in range(n)]


def test_q6_a_declared_header_over_the_context_is_a_422_never_a_cut_one():
    lines = long_lines()
    qs = {"q0": {"type": "noul", "instructions": lines[0]}}
    client, so, eng = make(limit=300)
    long_state = {"message": " ".join(f"w{i}" for i in range(150))}
    ask = {"state": long_state, "model": "m", "questions": qs}
    # the request's own header (one line) fits, as ever
    assert client.post("/v1/systemone", json=ask).status_code == 200
    r = client.post("/v1/systemone", json={**ask, hq.FIELD: lines})
    assert r.status_code == 422 and "never cut" in r.json()["detail"] and "12 lines" in r.json()["detail"]
    # without the field the many-question request is cut as before (the old behaviour is untouched)
    many = {f"q{i}": {"type": "noul", "instructions": t} for i, t in enumerate(lines)}
    r = client.post("/v1/systemone", json={"state": long_state, "model": "m", "questions": many})
    assert r.status_code == 200 and r.json()["header_limited"]["listed"] < 12
    # a declared header that fits is read whole
    assert client.post("/v1/systemone", json={**ask, hq.FIELD: lines[:2]}).status_code == 200


def test_q6_a_header_cap_refuses_a_declared_header_over_it():
    lines = long_lines(8)
    eng = Recording(limit=100_000, setting=(hl.AUTO, 40))
    so = SystemOne(eng, "d", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    client = TestClient(app)
    qs = {"q0": {"type": "noul", "instructions": lines[0]}}
    r = client.post("/v1/systemone", json={"state": STATE, "model": "m", "questions": qs, hq.FIELD: lines})
    assert r.status_code == 422 and "never cut" in r.json()["detail"]


# ---- Q7 -------------------------------------------------------------------------------------------------------------


def test_q7_the_context_boundary_is_built_from_the_declared_header():
    from test_context_boundary import CONTEXT, StubTok, stub_engine  # noqa: F401

    eng = stub_engine(layout="header")
    read = context_state(CONTEXT, STATE)
    qs = hq.stamp([engine_q(0)], tuple(LINES))
    rows, n, _ = LE._build_rows(eng, read, qs)
    cb = LE._context_boundary(eng, read, qs, rows, n)
    assert cb is not None and cb.fallback is None and cb.tokens is not None
    head = LE._template_head(eng) + layout_header(LINES) + CONTEXT
    assert rows[0][0][: cb.tokens] == eng.tok.encode(head + "\n\n", add_special_tokens=False)[: cb.tokens]
    # the boundary is the same for another question of the list: the context's blocks are shared
    other = hq.stamp([engine_q(1)], tuple(LINES))
    rows2, n2, _ = LE._build_rows(eng, read, other)
    assert LE._context_boundary(eng, read, other, rows2, n2).tokens == cb.tokens
    assert rows[0][0][:n] == rows2[0][0][:n2]
    # without the list the two questions' headers differ, and so do the boundaries' tokens
    a, b = [engine_q(0)], [engine_q(1)]
    ra, na, _ = LE._build_rows(eng, read, a)
    rb, nb, _ = LE._build_rows(eng, read, b)
    assert ra[0][0][:na] != rb[0][0][:nb]


# ---- Q8 -------------------------------------------------------------------------------------------------------------


def test_q8_usage_counts_the_declared_header():
    client, _, eng = make()
    own = client.post("/v1/systemone", json=body([0])).json()["usage"]["input_tokens"]
    decl = client.post("/v1/systemone", json=body([0], LINES)).json()["usage"]["input_tokens"]
    extra = len(eng.tok.encode(layout_header(LINES))) - len(eng.tok.encode(layout_header(LINES[:1])))
    assert extra > 0 and decl - own == extra
    plain, _, _ = make("plain")
    assert plain.post("/v1/systemone", json=body([0], LINES)).json()["usage"]["input_tokens"] == own - len(
        eng.tok.encode(layout_header(LINES[:1]))
    )


def test_q8_the_engines_route_takes_it_too():
    eng = Recording()
    client = TestClient(vllm_engine.make_app(eng))
    q = [{"kind": "choice", "instructions": LINES[0], "options": ["a", "b"]}]
    for bad in ([], [LINES[1]], [LINES[0], LINES[0]]):
        r = client.post("/v1/answer", json={"state": "s", "questions": q, hq.FIELD: bad})
        assert r.status_code == 422, r.text
    for which in (0, 1):
        q = [{"kind": "choice", "instructions": LINES[which], "options": ["a", "b"]}]
        r = client.post("/v1/answer", json={"state": "s", "questions": q, hq.FIELD: LINES[:2]})
        assert r.status_code == 200, r.text
    (first, n1), (second, n2) = eng.rows
    assert n1 == n2 and first[0][:n1] == second[0][:n2] and layout_header(LINES[:2]) in eng.tok.decode(first[0][:n1])


def test_q8_a_registration_example_refuses_the_field():
    client, _, _ = make()
    ex = examples(4, 0.9)
    ex[0]["request"][hq.FIELD] = [LINES[0]]
    r = post_task(client, {"id": "routing", "examples": ex})
    assert r.status_code == 422 and hq.FIELD in r.text and "example 0" in r.text


def test_q8_the_image_route_refuses_the_field():
    from decisio.serve import layout as lay
    from decisio.serve.systemone import SystemOneRequest

    _, so, _ = make()
    with pytest.raises(lay.LayoutRefused, match="header"):
        so.answer(SystemOneRequest.model_validate(body([0], LINES)), None, None, False, "image")
