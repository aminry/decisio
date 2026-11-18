# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator in the request path (decisio.serve.calculator_serving; docs/calculator.md).

S1  a server whose model records the format shows the block after the state, in every read, and says so
S2  a model that records none never sees the block, even when a request asks for it or sends amounts: a warning instead
S3  "calculator": false reads the state without the block, and applies no registered task
S4  a state the calculator does not read gets no block, with the reason, and /health counts it
S5  calculator_input: the block from the request's amounts; invalid input is a 422 naming the field, on any server
S6  the block never causes a refusal: a block that would push the row over the ceiling is left out (over-limit)
S7  determinism, the context before the state, task registration reads, and the fingerprints
S8  defaults.calculator in a tuned repository's file

  uv run pytest -q tests/unit/test_calculator_serving.py
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_calculator import FREIGHT, FREIGHT_BLOCK
from test_calculator_input import FREIGHT_INPUT
from test_header_limit import HeaderEngine
from test_over_limit import plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio.calculator import Annotation, annotate, from_input
from decisio.names import task_fingerprint
from decisio.readout.letters import CONTEXT_SEP
from decisio.serve import calculator_serving as calcs
from decisio.serve.systemone import SystemOne
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import make_app

QUESTION = {"q": {"type": "noul", "instructions": "Is the total charge above $1,000?"}}


class Engine(HeaderEngine):
    """The stand-in engine with the real row builder: answers evenly and keeps the state text of every read."""

    def answer(self, state, questions, adapter=None, images=None):
        self.seen.append(str(state))
        return [np.full(len(q.get("options") or [0, 0]), 1 / len(q.get("options") or [0, 0])) for q in questions], {}


def make(calculator="calc-v1", limit=10_000_000):
    eng = Engine(limit=limit, layout="plain")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), calculator=calculator)
    return TestClient(make_app(eng, so)), so, eng


def ask(client, state=FREIGHT, **extra):
    return client.post("/v1/systemone", json={"state": state, "model": "m", "questions": QUESTION, **extra})


# ---- S1 -------------------------------------------------------------------------------------------------------------


def test_s1_the_block_follows_the_state():
    client, so, eng = make()
    r = ask(client)
    assert r.status_code == 200, r.text
    assert eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    assert r.headers["x-decisio-calculator"] == "shown; lines=9; version=calc-v1"
    assert "x-decisio-warning" not in r.headers
    plain_client, _, _ = make(calculator=None)
    assert r.json()["usage"]["input_tokens"] > ask(plain_client).json()["usage"]["input_tokens"]
    health = client.get("/health").json()["systemone"]["calculator"]
    assert health == {"version": "calc-v1", "requests": 1, "shown": 1, "from_input": 0, "none": {}}


# ---- S2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [{"calculator": True}, {"calculator_input": FREIGHT_INPUT}])
def test_s2_a_model_without_the_record_never_sees_the_block(extra):
    client, so, eng = make(calculator=None)
    r = ask(client, **extra)
    assert r.status_code == 200, r.text
    assert eng.seen[-1] == FREIGHT
    assert r.headers["x-decisio-warning"] == calcs.NOT_TRAINED
    assert r.headers["x-decisio-calculator"] == "none; reason=not-trained"
    assert "calculator" not in client.get("/health").json()["systemone"]


def test_s2_a_request_that_asks_nothing_says_nothing():
    client, _, eng = make(calculator=None)
    r = ask(client)
    assert eng.seen[-1] == FREIGHT
    assert "x-decisio-calculator" not in r.headers and "x-decisio-warning" not in r.headers


# ---- S3 -------------------------------------------------------------------------------------------------------------


def test_s3_a_request_can_turn_it_off():
    client, so, eng = make()
    r = ask(client, calculator=False)
    assert r.status_code == 200 and eng.seen[-1] == FREIGHT
    assert r.headers["x-decisio-calculator"] == "none; reason=off-by-request"
    r = ask(client, calculator=False, task="some-task")
    assert r.status_code == 422 and "applies no registered task" in r.text


def test_s3_the_field_is_true_false_or_null():
    client, _, _ = make()
    r = ask(client, calculator="yes")
    assert r.status_code == 422 and r.json()["detail"] == '"calculator" must be true, false or null'


# ---- S4 -------------------------------------------------------------------------------------------------------------


def test_s4_a_state_it_does_not_read_gets_no_block():
    client, so, eng = make()
    state = "The customer writes that the parcel arrived late and asks for a call back."
    r = ask(client, state=state)
    assert r.status_code == 200 and eng.seen[-1] == state
    assert r.headers["x-decisio-calculator"].startswith("none; reason=")
    why = r.headers["x-decisio-calculator"].split("reason=")[1]
    assert client.get("/health").json()["systemone"]["calculator"]["none"] == {why: 1}


# ---- S5 -------------------------------------------------------------------------------------------------------------


def test_s5_the_block_from_the_request_amounts():
    client, so, eng = make()
    state = "Shipment SH-310: see the amounts sent with the request. Is the total right?"
    r = ask(client, state=state, calculator_input=FREIGHT_INPUT)
    assert r.status_code == 200, r.text
    assert eng.seen[-1] == f"{state}\n{from_input(FREIGHT_INPUT).text}"
    assert client.get("/health").json()["systemone"]["calculator"]["from_input"] == 1


@pytest.mark.parametrize("calculator", ["calc-v1", None])
def test_s5_invalid_input_is_a_422_naming_the_field(calculator):
    client, _, eng = make(calculator=calculator)
    bad = {**FREIGHT_INPUT, "items": [{**FREIGHT_INPUT["items"][0], "rate": 41.0}, *FREIGHT_INPUT["items"][1:]]}
    r = ask(client, calculator_input=bad)
    assert r.status_code == 422 and "calculator_input.items[0]" in r.text
    assert eng.seen == []


# ---- S6 -------------------------------------------------------------------------------------------------------------


def test_s6_a_block_that_would_not_fit_is_left_out():
    client, so, eng = make()
    ask(client)
    full = len(so.engine._rows(f"{FREIGHT}\n{FREIGHT_BLOCK}", [{"kind": "noul", **QUESTION["q"]}])[0][0][0])
    plain = len(so.engine._rows(FREIGHT, [{"kind": "noul", **QUESTION["q"]}])[0][0][0])
    assert plain < full
    tight, so2, eng2 = make(limit=(plain + full) // 2)
    r = ask(tight)
    assert r.status_code == 200, r.text
    assert eng2.seen[-1] == FREIGHT
    assert r.headers["x-decisio-calculator"] == "none; reason=over-limit"


# ---- S7 -------------------------------------------------------------------------------------------------------------


def test_s7_the_same_request_reads_the_same_bytes():
    client, _, eng = make()
    ask(client)
    ask(client, calculator=False)
    ask(client)
    assert eng.seen[0] == eng.seen[2]


def test_s7_the_context_comes_first_and_the_block_last():
    _, so, _ = make()
    context = "These are the billing rules of the carrier."
    read = so.read_state(context, FREIGHT, examples=True)
    ann = annotate(FREIGHT, context=context)
    assert isinstance(ann, Annotation)
    assert str(read) == f"{context}{CONTEXT_SEP}{FREIGHT}\n{ann.text}" and read.context == context


def test_s7_registration_reads_follow_the_server():
    _, on, _ = make()
    _, off, _ = make(calculator=None)
    assert str(on.read_state(None, FREIGHT, examples=True)) == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    assert off.read_state(None, FREIGHT, examples=True) == FREIGHT


def test_s7_fingerprints_carry_the_format_only_when_on():
    common = dict(served_name="s", model="m", pad_to=None, pad_where="front", hide_index_keys=True, desnake_labels=True)
    assert task_fingerprint(**common) == task_fingerprint(**common, calculator=None)
    assert '"calculator": "calc-v1"' in task_fingerprint(**common, calculator="calc-v1")
    _, on, _ = make()
    _, off, _ = make(calculator=None)
    assert on.stack_facts()["serving"]["calculator"] == "calc-v1"
    assert "calculator" not in off.stack_facts()["serving"]


# ---- S8 -------------------------------------------------------------------------------------------------------------


def test_s8_the_repository_records_a_known_format():
    from decisio.hub import check_defaults

    base = {"temperatures": {"choice": 1.0}, "noul_commit": False, "fit": {"method": "log loss"}}
    check_defaults({**base, "calculator": "calc-v1"})
    check_defaults({**base, "calculator": "calc-v9"})  # a newer format: a warning, and served without the block
    with pytest.raises(ValueError, match="defaults.calculator must be the name"):
        check_defaults({**base, "calculator": 9})


# ---- S9 review: packed mode, input bounds, a real tie, every read of a request, concurrency --------------------------


def test_s9_packed_mode_shows_no_block_anywhere():
    eng = Engine(limit=10_000_000, layout="plain")
    eng.mode = "packed"
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), calculator="calc-v1")
    assert so.calculator is None
    assert so.read_state(None, FREIGHT, examples=True) == FREIGHT
    assert "calculator" not in so.stack_facts()["serving"]


def test_s9_packed_mode_turns_the_record_off_at_start():
    import types

    from decisio.serve.vllm_engine import apply_repository_defaults

    defaults = {"temperatures": {"choice": 2.0}, "noul_commit": False, "fit": {"method": "x"}, "calculator": "calc-v1"}
    repo = types.SimpleNamespace(defaults=defaults, name="r", revision="abc", defaults_sha256="d")
    for mode, want in (("separate", "calc-v1"), ("packed", None)):
        args = types.SimpleNamespace(mode=mode, calculator="auto")
        apply_repository_defaults(args, repo)
        assert args.calculator_version == want
    assert "--mode packed" in args.repository_defaults["calculator_set_by"]
    args = types.SimpleNamespace(mode="separate", calculator="off")
    apply_repository_defaults(args, repo)
    assert args.calculator_version is None


def test_s9_reads_outside_a_request_get_no_block_unless_they_read_examples():
    _, so, _ = make()
    assert so.read_state(None, FREIGHT) == FREIGHT
    assert str(so.read_state(None, FREIGHT, examples=True)) == f"{FREIGHT}\n{FREIGHT_BLOCK}"


def test_s9_a_half_even_tie_in_a_computed_line():
    def tax_line(price):
        data = {
            "currency": "$",
            "items": [{"kind": "count", "label": "Widgets", "quantity": 1, "price": price}],
            "tax": {"rate": "10", "kind": "charged"},
        }
        return from_input(data).text.splitlines()[-1]

    assert tax_line("100.000005") == "- Tax charged: $10.00"  # 10.0000005: a tie, to the even digit (0)
    assert tax_line("100.000015") == "- Tax charged: $10.000002"  # 10.0000015: a tie, to the even digit (2)
    assert tax_line("100.000025") == "- Tax charged: $10.000002"


def test_s9_header_layout_reads_the_block_after_the_state():
    eng = Engine(limit=10_000_000, layout="header")
    client = TestClient(make_app(eng, SystemOne(eng, "d", task_store=TaskStore("fp"), calculator="calc-v1")))
    assert ask(client).status_code == 200 and eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    rows = eng._rows(eng.seen[-1], [{"kind": "noul", **QUESTION["q"]}])[0]
    plain = eng._rows(FREIGHT, [{"kind": "noul", **QUESTION["q"]}])[0]
    assert len(rows[0][0]) > len(plain[0][0])
    assert ask(client, calculator=False).status_code == 200 and eng.seen[-1] == FREIGHT


class ThinkingEngine(Engine):
    """Answers evenly (so every yes/no question is unsure) and keeps the state each second read is given."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.thought = []

    def can_think(self):
        return True

    def think(self, state, questions, adapter=None, budget=512, subset=None):
        subset = list(range(len(questions))) if subset is None else subset
        self.thought.append(str(state))
        return [np.log(np.full(2, 0.5)) for _ in subset], [10 for _ in subset], {"generate_ms": 1.0, "read_ms": 1.0}


@pytest.mark.parametrize("on", [True, False])
def test_s9_the_thinking_second_read_sees_the_same_state(on):
    eng = ThinkingEngine(limit=10_000_000, layout="plain")
    so = SystemOne(eng, "d", task_store=TaskStore("fp"), calculator="calc-v1", think=True, think_factor=1.0)
    client = TestClient(make_app(eng, so))
    r = ask(client, **({} if on else {"calculator": False}))
    assert r.status_code == 200, r.text
    want = f"{FREIGHT}\n{FREIGHT_BLOCK}" if on else FREIGHT
    assert eng.thought == [want] and eng.seen[-1] == want


def test_s9_read_twice_doubles_the_state_with_its_block():
    from dataclasses import replace

    eng = Engine(limit=10_000_000, layout="plain")
    eng.fmt = replace(eng.fmt, read_twice_under=10_000_000)
    client = TestClient(make_app(eng, SystemOne(eng, "d", task_store=TaskStore("fp"), calculator="calc-v1")))
    assert ask(client).status_code == 200 and eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    q = [{"kind": "noul", **QUESTION["q"]}]
    twice = eng._rows(eng.seen[-1], q, with_plan=True)[0]
    once = Engine(limit=10_000_000, layout="plain")._rows(eng.seen[-1], q)[0]
    assert len(twice[0][0]) > 2 * len(once[0][0]) - 40  # the row holds the state and its block twice


def test_s9_requests_with_images_never_carry_the_block():
    d = calcs.decide("calc-v1", True, None, images=True, packed=False)
    assert not d.on and d.reason == "images" and d.warning == calcs.WITH_IMAGES
    assert calcs.decide("calc-v1", None, None, images=True, packed=False).warning is None


def test_s9_a_context_task_with_the_calculator_off():
    from test_certified_answers import CRIT

    client, so, eng = make()
    context = "Queue rules: billing questions go to billing."
    exs = [
        {"request": {"state": f"ticket {i}: please send a copy", "model": "m",
                     "questions": {"q": {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}}},
         "answer": "option_0"}
        for i in range(40)
    ]  # fmt: skip
    r = client.post("/v1/tasks", json={"id": "queues", "examples": exs, "context": context})
    assert r.status_code == 200, r.text
    body = {"state": "ticket x: please send a copy", "model": "m",
            "questions": {"q": {"type": "choice", "instructions": "Which queue?", "criteria": CRIT}}}  # fmt: skip
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert eng.seen[-1].startswith(context)  # the task applies: its context is read first
    r = client.post("/v1/systemone", json={**body, "calculator": False})
    assert r.status_code == 200 and eng.seen[-1] == body["state"]  # no task, no context, no block
    assert r.headers["x-decisio-calculator"] == "none; reason=off-by-request"


def test_s9_concurrent_requests_on_and_off_each_read_their_own_state():
    from concurrent.futures import ThreadPoolExecutor

    from decisio.serve.systemone import SystemOneRequest

    _, so, eng = make()

    def one(i):
        extra = {} if i % 2 == 0 else {"calculator": False}
        req = SystemOneRequest.model_validate({"state": FREIGHT, "model": "m", "questions": QUESTION, **extra})
        return so.answer(req)["_timing"]["calculator"]

    with ThreadPoolExecutor(8) as ex:
        headers = list(ex.map(one, range(64)))
    assert headers.count("shown; lines=9; version=calc-v1") == 32
    assert headers.count("none; reason=off-by-request") == 32
    assert sum(s == f"{FREIGHT}\n{FREIGHT_BLOCK}" for s in eng.seen) == 32 and eng.seen.count(FREIGHT) == 32
