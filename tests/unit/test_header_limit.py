# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A header that does not outgrow the context (decisio.serve.header_limit; docs/running.md, "The header layout").

H1  the setting, the choice of lines (a prefix, as many as fit) and the refusal's words
H2  a request whose rows fit is built exactly as before: no plan is made, the rows are byte for byte the same
H3  a request over the context lists the first lines that fit, the rows fit, two limits nest, k = 0 is the plain row
H4  the setting off refuses, and says where the tokens are; a request the plain layout cannot hold is refused too
H5  a number is a cap on the header's tokens
H6  the response: header_limited, x-decisio-layout, usage, /health counts; absent on a request that fits
H7  a certified task's answer in a limited request is not certified (reason header_limited)

  uv run pytest -q tests/unit/test_header_limit.py
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import examples, post_task, wire
from test_over_limit import LimitEngine, WordTok, plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio.readout.letters import PromptFormat, layout_header
from decisio.serve import header_limit as hl
from decisio.serve import over_limit as over
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

LE = vllm_engine.LettersEngine


class HeaderEngine(LimitEngine):
    """The stand-in engine with the real row builder, the header layout and the header plan."""

    pad_unit = 0
    over_limit_tokens = None
    header_limit_setting = (hl.AUTO, None)

    def __init__(self, limit=300, layout="header", setting=(hl.AUTO, None)):
        super().__init__(limit)
        self.fmt = PromptFormat(layout=layout)
        self.header_limit_setting = setting
        self.labels = {}

    def _labels(self, cands):
        return tuple(cands)

    def _prepare_separate(self, state, questions):
        rows, n, _ = self._build_rows(state, questions)
        return rows, n


for _name in ("_build_rows", "_rows", "header_plan", "_header_plan_exact", "_template_tail", "read_plan"):
    setattr(HeaderEngine, _name, getattr(LE, _name))


def qs_many(n=12, words=14):
    return [
        {
            "instructions": f"Question {i}: " + " ".join(f"detail{i}x{j}" for j in range(words)) + "?",
            "options": ["yes", "no"],
        }
        for i in range(n)
    ]


def state_of(words):
    return {"message": " ".join(f"w{i}" for i in range(words))}


def longest(eng, state, qs, k=None):
    rows, _, _ = LE._rows(eng, state, qs, k)
    return max(len(ids) for ids, _ in rows)


def header_text(eng, qs, state, rows_k):
    return layout_header([q["instructions"] for q in qs][:rows_k])


# ---- H1 -------------------------------------------------------------------------------------------------------------


def test_h1_the_setting():
    assert hl.parse(None) == (hl.AUTO, None) and hl.parse("auto") == (hl.AUTO, None)
    assert hl.parse("OFF") == (hl.OFF, None) and hl.parse("4000") == (hl.AUTO, 4000)
    for bad in ("0", "-3", "some", "1.5"):
        with pytest.raises(ValueError, match="--header-limit"):
            hl.parse(bad)
    assert [hl.describe(*hl.parse(x)) for x in (None, "off", "250")] == ["auto", "off", "250"]


def test_h1_the_lines_are_a_prefix_as_many_as_fit():
    assert hl.choose_lines([10, 10, 10, 10], 5, 100) == 4
    assert hl.choose_lines([10, 10, 10, 10], 5, 30) == 2  # 5 + 10 + 10 = 25; a third would be 35
    assert hl.choose_lines([10, 500, 10, 10], 5, 100) == 1  # a line that does not fit ends the list: no skipping ahead
    assert (
        hl.choose_lines([10, 10], 5, 12) == 0 and hl.choose_lines([10], 50, 40) == 0 and hl.choose_lines([], 5, 9) == 0
    )


def test_h1_the_refusal_says_where_the_tokens_are_and_the_ways_out():
    pieces = {"header": 1200, "state": 300, "question": 80, "template": 20, "row": 1600}
    text = hl.refusal(1000, pieces, header_off=True)
    for part in (
        "1,600",
        "the header 1,200",
        "the state 300",
        "the longest question 80",
        "1,000",
        "--header-limit auto",
        "over_limit",
        "--max-model-len",
    ):
        assert part in text, part
    assert "--header-limit" not in hl.refusal(1000, {**pieces, "header": 0}, header_off=False)


# ---- H2 -------------------------------------------------------------------------------------------------------------


def test_h2_a_request_that_fits_is_built_as_before():
    eng, qs, state = HeaderEngine(limit=2000), qs_many(6), state_of(50)
    assert not eng.header_plan(state, qs)["limited"]  # the rows were measured and fit
    assert HeaderEngine(limit=65_535).header_plan(state, qs) is None  # the bytes fit: not even tokenized
    rows, n, plan = eng._build_rows(state, qs)
    ref, ref_n, _ = LE._rows(eng, state, qs, None)
    assert rows == ref and n == ref_n and "header_limited" not in plan and plan["header"] is True


def test_h2_a_request_just_under_the_context_is_not_limited_either():
    eng, qs, state = HeaderEngine(limit=10_000), qs_many(6), state_of(50)
    exact = longest(eng, state, qs)
    eng.limit = exact + 1  # the longest row fits by one token: bytes say maybe, the rows say yes
    hp = eng.header_plan(state, qs)
    assert hp is None or not hp["limited"]
    rows, _, plan = eng._build_rows(state, qs)
    assert rows == LE._rows(eng, state, qs, None)[0] and "header_limited" not in plan


# ---- H3 -------------------------------------------------------------------------------------------------------------


def test_h3_a_request_over_the_context_lists_the_first_lines_that_fit():
    eng, qs, state = HeaderEngine(limit=300), qs_many(12), state_of(150)
    assert longest(eng, state, qs) > 300 >= longest(eng, state, qs, 0)  # the whole header does not fit; plain does
    hp = eng.header_plan(state, qs)
    k = hp["listed"]
    assert hp["limited"] and 0 < k < 12 and hp["of"] == 12 and hp["fits"]
    rows, n, plan = eng._build_rows(state, qs)
    assert max(len(ids) for ids, _ in rows) <= 300
    assert plan["header_limited"]["listed"] == k and plan["header"] is True
    assert longest(eng, state, qs, k + 1) > 300  # one more line would not have fit: as many as fit
    assert rows == LE._rows(eng, state, qs, k)[0]  # the rows are the first k lines' header, then the state


def test_h3_two_limits_nest():
    qs, state = qs_many(12), state_of(150)
    small = HeaderEngine(limit=230).header_plan(state, qs)["listed"]
    large = HeaderEngine(limit=330).header_plan(state, qs)["listed"]
    assert 0 < small < large < 12


def test_h3_with_no_line_that_fits_the_row_is_the_plain_layouts():
    qs, state = qs_many(3, words=40), state_of(150)
    plain_eng = HeaderEngine(limit=300, layout="plain")
    eng = HeaderEngine(limit=longest(plain_eng, state, qs) + 3)  # three tokens beside the plain row: no line fits
    hp = eng.header_plan(state, qs)
    assert hp["limited"] and hp["listed"] == 0 and hp["report"]["layout"] == "plain" and hp["fits"]
    rows, n, plan = eng._build_rows(state, qs)
    plain_rows_, plain_n, _ = plain_eng._build_rows(state, qs)
    assert rows == plain_rows_ and n == plain_n and plan["header"] is False


def test_h3_the_plan_is_made_once_per_request():
    eng, qs, state = HeaderEngine(limit=300), qs_many(12), state_of(150)
    first = eng.header_plan(state, qs)
    assert eng.header_plan(state, qs) is first  # the same object: from the small cache
    assert eng.header_plan(state_of(61), qs) is not first


# ---- H4 -------------------------------------------------------------------------------------------------------------


def test_h4_off_refuses_a_row_over_the_context_and_says_where_the_tokens_are():
    eng, qs, state = HeaderEngine(limit=300, setting=(hl.OFF, None)), qs_many(12), state_of(150)
    hp = eng.header_plan(state, qs)
    assert hp["limited"] is False and hp["fits"] is False and hp["off"] is True and hp["pieces"]["header"] > 0
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    with pytest.raises(over.OverLimit, match="--header-limit auto"):
        so.header_plan(state, qs, eng)
    assert so.header_limited["refused"] == 1


def test_h4_a_request_the_plain_layout_cannot_hold_is_refused_whatever_the_setting():
    eng, qs, state = HeaderEngine(limit=100), qs_many(4), state_of(200)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    with pytest.raises(over.OverLimit, match=r"longest row of the request is [\d,]+ tokens") as e:
        so.header_plan(state, qs, eng)
    assert "--max-model-len" in str(e.value) and "over_limit" in str(e.value)


# ---- H5 -------------------------------------------------------------------------------------------------------------


def test_h5_a_number_caps_the_header_even_where_the_context_would_hold_more():
    qs, state = qs_many(12), state_of(30)
    open_eng = HeaderEngine(limit=5000)
    assert open_eng.header_plan(state, qs) is None
    eng = HeaderEngine(limit=5000, setting=(hl.AUTO, 120))
    hp = eng.header_plan(state, qs)
    assert hp["limited"] and 0 < hp["listed"] < 12 and hp["report"]["reason"] == "cap"
    assert hp["report"]["header_tokens"] <= 120 < hp["report"]["of_header_tokens"]


# ---- H6 -------------------------------------------------------------------------------------------------------------


MESSAGE = "1:0.9:x " + " ".join(f"w{i}" for i in range(150))


def post_many(client, n=12, message=MESSAGE, words=14, **extra):
    from test_certified_answers import CRIT

    qs = {
        f"q{i}": {"type": "choice", "instructions": q["instructions"], "criteria": CRIT}
        for i, q in enumerate(qs_many(n, words))
    }
    return client.post("/v1/systemone", json={"state": {"message": message}, "model": "m", "questions": qs, **extra})


def make(limit, setting=(hl.AUTO, None)):
    eng = HeaderEngine(limit=limit, setting=setting)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def test_h6_the_response_says_what_was_listed():
    client, so, eng = make(300)
    r = post_many(client)
    assert r.status_code == 200, r.text
    body = r.json()
    rep = body["header_limited"]
    assert rep["layout"] == "header" and 0 < rep["listed"] < rep["of"] == 12 and rep["reason"] == "context"
    assert rep["row_limit"] == 300 and rep["header_tokens"] <= rep["allowed_tokens"] < rep["of_header_tokens"]
    assert r.headers["x-decisio-layout"] == f"header; listed={rep['listed']}; of=12; reason=context"
    assert so.header_limited["requests"] == 1 and so.header_limited["lines_dropped"] == 12 - rep["listed"]
    # the usage counts the header that was read, not the one that was asked for
    whole = SystemOne(HeaderEngine(limit=10_000_000), "d", task_store=TaskStore("fp"))
    qs = [{"instructions": q["instructions"], "options": q["options"], "kind": "choice"} for q in qs_many(12)]
    state = {"message": MESSAGE}
    assert body["usage"]["input_tokens"] < whole.usage_tokens(state, qs)


def test_h6_a_request_that_fits_carries_nothing_extra():
    client, so, eng = make(10_000)
    r = post_many(client)
    assert r.status_code == 200 and "header_limited" not in r.json() and "x-decisio-layout" not in r.headers
    assert so.header_limited == {"requests": 0, "plain": 0, "refused": 0, "lines_dropped": 0}
    eng.limit = 300  # the same request over a smaller context
    assert "header_limited" in post_many(client).json()


def test_h6_a_plain_row_says_plain_and_is_counted():
    plain = HeaderEngine(limit=300, layout="plain")
    state = {"message": MESSAGE}
    qs = [{"instructions": q["instructions"], "options": q["options"], "kind": "choice"} for q in qs_many(3, words=40)]
    client, so, eng = make(longest(plain, state, qs) + 3)
    r = post_many(client, n=3, words=40)
    rep = r.json()["header_limited"]
    assert (
        rep["layout"] == "plain" and rep["listed"] == 0 and r.headers["x-decisio-layout"].startswith("plain; listed=0")
    )
    assert so.header_limited["plain"] == 1


def test_h6_a_row_nothing_can_hold_is_a_422_with_the_pieces():
    client, so, eng = make(60)
    r = post_many(client, n=3)
    assert r.status_code == 422 and "longest row of the request" in r.text and "--max-model-len" in r.text
    off_client, off_so, _ = make(300, setting=(hl.OFF, None))
    r = post_many(off_client)
    assert r.status_code == 422 and "the header" in r.text and "--header-limit auto" in r.text


def test_h6_health_counts_the_limited_requests():
    _, so, _ = make(300)
    so.header_limited.update(requests=3, plain=1, lines_dropped=14)
    assert so.header_limited == {"requests": 3, "plain": 1, "refused": 0, "lines_dropped": 14}


# ---- H7 -------------------------------------------------------------------------------------------------------------


def test_h7_a_certified_task_is_not_certified_in_a_limited_request():
    client, so, eng = make(10_000)
    r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {"alpha": 0.05, "delta": 0.1}})
    assert r.status_code == 200 and r.json()["certificate"]["certified"], r.text
    cert = r.json()["certificate"]
    whole = client.post("/v1/systemone", json=wire("1:0.99:x short")).json()["answers"]["q1"]
    assert whole["certified"] is True and whole["certificate"] == cert["id"]
    plain = HeaderEngine(limit=300, layout="plain")
    row = longest(plain, {"message": "1:0.99:x short"}, [{"instructions": "Which queue?", "options": ["a", "b", "c"]}])
    eng.limit = row + 2  # the header does not fit beside the plain row
    cut = client.post("/v1/systemone", json=wire("1:0.99:x short")).json()
    a = cut["answers"]["q1"]
    assert cut["header_limited"]["listed"] == 0
    assert a["certified"] is False and a["certificate"] == cert["id"] and a["certified_reason"] == "header_limited"


# ---- H8 -------------------------------------------------------------------------------------------------------------


def test_h8_with_gemma_4s_tokenizer_a_many_question_request_with_one_huge_instruction():
    """The shape of the failing request: 32 questions, one of whose instructions is a very long document, which the
    header repeats on every row (skipped where the tokenizer is not cached)."""
    from test_truncate_header import Stub, gemma_tokenizer

    tok = gemma_tokenizer()
    fmt = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")
    qs = [{"kind": "noul", "instructions": f"Does tool number {i} answer the request?"} for i in range(32)]
    qs[7] = {"kind": "noul", "instructions": "Does this tool answer the request? " + "Tool description. " * 900}
    state = {"request": "Book a table for two on Friday."}
    context = 4096

    class GemmaStub(Stub):
        header_limit_setting = (hl.AUTO, None)

    for name in ("header_plan", "_header_plan_exact", "_rows"):
        setattr(GemmaStub, name, getattr(LE, name))
    eng = GemmaStub(tok, fmt, context - 1)
    whole = max(len(ids) for ids, _ in LE._rows(eng, state, qs, None)[0])
    plain = max(len(ids) for ids, _ in LE._rows(eng, state, qs, 0)[0])
    assert whole > context > plain  # the header alone pushes the request over the context
    rows, n, plan = eng._build_rows(state, qs)
    rep = plan["header_limited"]
    assert max(len(ids) for ids, _ in rows) <= context - 1 and rep["reason"] == "context"
    assert rep["listed"] in range(1, 32)
    assert rep["listed"] <= 7  # the huge line is the eighth: it does not fit, and ends the list before it
    first = LE._rows(eng, state, qs, rep["listed"])[0]
    assert rows == first
    # one more line would not have fit
    nxt = max(len(ids) for ids, _ in LE._rows(eng, state, qs, rep["listed"] + 1)[0])
    assert nxt > context - 1
