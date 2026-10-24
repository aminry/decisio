# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Read twice (`--read-twice-under N`): an opt-in that reads short requests twice, never the default.

The oracle is an independent renderer of the doubled layout (the user turn written again after
`Read the same input again before answering:`), copied below, so the served flag and the A/B read the same tokens. Which
requests are written twice is decided here by counted input tokens (the unit of usage.input_tokens) and by the
question cap, not by the A/B's row count.

R1  the format: off by default, the facts carry the settings only when on, bad values are refused
R2  the rows equal the A/B's A2 token for token whenever a request is written twice, on both sides of the token limit,
    with and without the header; the request read once is counted in input tokens
R2b the question cap: a request with more questions than the cap is read once, whatever its size
R3  usage.input_tokens counts what is read: the second copy of every row
R4  tasks and certificates are per setting
R5  the flag, and what the start prints

  uv run pytest -q tests/unit/test_read_twice.py
"""

import pytest
from test_layout_flags import BASE, ONE, STATE, TWO, StubTok, engine, gemma_tokenizer, layout_header_text, q

from decisio.readout import letters as L
from decisio.serve import vllm_engine
from decisio.serve.vllm_engine import counted_tokens, question_text

HEAD = "Questions that will be asked about the context:\n"
AGAIN = "\n\nRead the same input again before answering:\n"


def oracle(tok, state, questions, fmt, header, twice):
    """The oracle on engine questions, written twice or not as `twice` says: the token rows."""
    body = L.fmt_state(state)
    bodies = [L.question_body(tok, e["kind"], e["instructions"], e.get("options"), fmt)[0] for e in questions]
    head = HEAD + "".join(f"- {e['instructions']}\n" for e in questions) + "\n" if header else ""
    turns = [head + body + "\n\n" + t for t in bodies]
    ids = [tok.encode(L.chat_turn(tok, t, fmt), add_special_tokens=False) for t in turns]

    if twice:
        ids = [tok.encode(L.chat_turn(tok, t + AGAIN + t, fmt), add_special_tokens=False) for t in turns]
    return ids


# ---- R1 -------------------------------------------------------------------------------------------------------------


def test_r1_off_by_default_and_in_the_facts_only_when_on():
    base = L.PromptFormat(**BASE)
    assert base.read_twice_under == 0 and "read_twice_under" not in base.facts()
    on = L.PromptFormat(**BASE, read_twice_under=1500)
    assert on.facts() == {**base.facts(), "read_twice_under": 1500} and not on.is_default()
    assert on.core() == base  # a row's own text and labels are not changed by it
    both = L.PromptFormat(**BASE, layout="header", read_twice_under=3000)
    assert both.facts() == {**base.facts(), "layout": "header", "read_twice_under": 3000}
    for bad in (-1, True, 2.5, "3000"):
        with pytest.raises(ValueError):
            L.PromptFormat(**BASE, read_twice_under=bad)
    capped = L.PromptFormat(**BASE, read_twice_under=2000, read_twice_max_questions=8)
    assert capped.facts() == {**base.facts(), "read_twice_under": 2000, "read_twice_max_questions": 8}
    assert capped.core() == base and "read_twice_max_questions" not in on.facts()
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            L.PromptFormat(**BASE, read_twice_under=2000, read_twice_max_questions=bad)
    with pytest.raises(ValueError, match="limits read twice"):
        L.PromptFormat(**BASE, read_twice_max_questions=8)


# ---- R2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("questions", [ONE, TWO])
@pytest.mark.parametrize("header", [True, False])
def test_r2_the_rows_are_the_ab_arm_a2_on_both_sides_of_the_limit(questions, header):
    plain = L.PromptFormat(**BASE)
    layout = "header" if header else "plain"
    counted = counted_tokens(StubTok(), L.PromptFormat(**BASE, layout=layout), STATE, questions)["total"]
    for limit in (counted - 1, counted, counted + 50):
        fmt = L.PromptFormat(**BASE, layout=layout, read_twice_under=limit)
        eng = engine(fmt, StubTok())
        rows, n = eng._prepare_separate(STATE, questions)
        twice = limit >= counted
        assert [r[0] for r in rows] == oracle(eng.tok, STATE, questions, plain, header, twice)
        assert eng.read_plan(STATE, questions) == {
            "header": header,
            "twice": twice,
            "once_tokens": counted,
            "questions": len(questions),
        }
        assert all(r[0][:n] == rows[0][0][:n] for r in rows)  # the cacheable prefix is the first copy's


def test_r2_counted_tokens_are_the_header_the_state_and_each_questions_text():
    tok = StubTok()
    fmt = L.PromptFormat(**BASE, layout="header")
    c = counted_tokens(tok, fmt, STATE, TWO)
    assert c["header"] == len(tok.encode(layout_header_text(TWO))) and c["state"] == len(tok.encode(L.fmt_state(STATE)))
    assert c["asked"] == [len(tok.encode(question_text(tok, e, fmt)[0])) for e in TWO]
    assert c["total"] == c["header"] + c["state"] + sum(c["asked"])
    assert counted_tokens(tok, L.PromptFormat(**BASE), STATE, TWO)["header"] == 0


def test_r2_a_request_over_the_limit_is_read_once_and_the_plan_says_so():
    fmt = L.PromptFormat(**BASE, layout="header", read_twice_under=5)
    eng = engine(fmt)
    plan = eng.read_plan(STATE, TWO)
    assert plan["twice"] is False and plan["once_tokens"] > 5 and plan["header"] is True
    rows, _ = eng._prepare_separate(STATE, TWO)
    assert eng.tok.decode(rows[0][0]).count("Read the same input again") == 0
    off = engine(L.PromptFormat(**BASE, layout="header"))
    assert off.read_plan(STATE, TWO) == {"header": True, "twice": False, "once_tokens": None}


# ---- R2b ------------------------------------------------------------------------------------------------------------


def test_r2b_a_request_with_more_questions_than_the_cap_is_read_once():
    many = [q(f"Is statement {i} true?", kind="noul") for i in range(5)]
    for cap, questions, twice in ((2, TWO, True), (1, TWO, False), (1, ONE, True), (4, many, False), (5, many, True)):
        fmt = L.PromptFormat(**BASE, layout="header", read_twice_under=10_000, read_twice_max_questions=cap)
        plan = engine(fmt).read_plan(STATE, questions)
        assert plan["twice"] is twice and plan["questions"] == len(questions), (cap, len(questions))
    # no cap: size alone decides; and a cap does not make a large request small
    assert engine(L.PromptFormat(**BASE, read_twice_under=10_000)).read_plan(STATE, many)["twice"] is True
    big = L.PromptFormat(**BASE, read_twice_under=5, read_twice_max_questions=10)
    assert engine(big).read_plan(STATE, ONE)["twice"] is False


def test_r2_the_second_copy_is_the_whole_user_turn_of_each_row():
    tok, fmt = StubTok(), L.PromptFormat(**BASE, layout="header", read_twice_under=10_000)
    rows, _ = engine(fmt, tok)._prepare_separate(STATE, TWO)
    for (ids, _), e in zip(rows, TWO):
        text = tok.decode(ids)
        turn = layout_header_text(TWO) + L.fmt_state(STATE) + "\n\n" + question_text(tok, e, fmt)[0]
        assert turn + AGAIN + turn in text and text.count(layout_header_text(TWO)) == 2


# ---- R3 -------------------------------------------------------------------------------------------------------------


class UsageEngine:
    def __init__(self, fmt):
        self.tok, self.fmt = StubTok(), fmt
        self.eng = engine(fmt, self.tok)

    def read_plan(self, state, qs):
        return self.eng.read_plan(state, qs)


def usage(fmt, questions=TWO):
    from decisio.serve.systemone import SystemOne

    so = SystemOne.__new__(SystemOne)
    so.engine = UsageEngine(fmt)
    return so.usage_tokens(STATE, questions), so.engine.tok


def test_r3_usage_counts_the_second_copy_of_every_row():
    header, tok = usage(L.PromptFormat(**BASE, layout="header"))
    twice, tok2 = usage(L.PromptFormat(**BASE, layout="header", read_twice_under=10_000))
    fmt = L.PromptFormat(**BASE)
    header_n = len(tok2.encode(layout_header_text(TWO)))
    state_n = len(tok2.encode(L.fmt_state(STATE)))
    asked = [len(tok2.encode(question_text(tok2, e, fmt)[0])) for e in TWO]
    again = len(tok2.encode(AGAIN))
    assert twice == header + sum(header_n + state_n + a + again for a in asked)
    assert twice > 1.9 * header  # roughly doubles
    short, _ = usage(L.PromptFormat(**BASE, layout="header", read_twice_under=5))
    assert short == header  # a request over the limit is read once, and counted once
    plain_twice, _ = usage(L.PromptFormat(**BASE, read_twice_under=10_000), ONE)
    plain_once, _ = usage(L.PromptFormat(**BASE), ONE)
    assert plain_twice > plain_once + len(tok2.encode(AGAIN))  # without the header too


# ---- R4 -------------------------------------------------------------------------------------------------------------


def test_r4_tasks_and_certificates_are_per_setting():
    from test_layout_flags import fingerprint

    from decisio.names import same_fingerprint

    header = fingerprint(L.PromptFormat(**BASE, layout="header"))
    t3000 = fingerprint(L.PromptFormat(**BASE, layout="header", read_twice_under=3000))
    t1500 = fingerprint(L.PromptFormat(**BASE, layout="header", read_twice_under=1500))
    capped = fingerprint(L.PromptFormat(**BASE, layout="header", read_twice_under=1500, read_twice_max_questions=8))
    assert len({header, t3000, t1500, capped}) == 4
    assert not same_fingerprint(header, t3000) and not same_fingerprint(t3000, t1500)
    assert header == fingerprint(L.PromptFormat(**BASE, layout="header", read_twice_under=0))  # off: unchanged


# ---- R5 -------------------------------------------------------------------------------------------------------------


def test_r5_the_flag_reaches_the_format_and_the_start_says_so(capsys):
    from test_hub import args_for

    parser = vllm_engine.build_parser()
    a = parser.parse_args(["--read-twice-under", "2000", "--read-twice-max-questions", "12"])
    assert (a.read_twice_under, a.read_twice_max_questions) == (2000, 12)
    d = parser.parse_args([])
    assert (d.read_twice_under, d.read_twice_max_questions) == (None, None)
    fam, fmt = vllm_engine.resolve_base(
        args_for("gemma-4-12b", layout="header", read_twice_under=2000, read_twice_max_questions=12)
    )
    assert (fmt.read_twice_under, fmt.read_twice_max_questions, fmt.layout) == (2000, 12, "header")
    _, plain = vllm_engine.resolve_base(args_for("gemma-4-12b"))
    assert plain.read_twice_under == 0 and plain.read_twice_max_questions == 0  # never the default
    with pytest.raises(SystemExit):
        vllm_engine.resolve_base(args_for("gemma-4-12b", read_twice_under=-5))
    with pytest.raises(SystemExit, match="limits --read-twice-under"):
        vllm_engine.resolve_base(args_for("gemma-4-12b", read_twice_max_questions=8))
    with pytest.raises(SystemExit):
        vllm_engine.resolve_base(args_for("gemma-4-12b", read_twice_under=2000, read_twice_max_questions=-1))


def test_r5_the_real_tokenizer_reads_the_same_tokens_as_the_ab_arm_a2():
    tok = gemma_tokenizer()
    plain = L.PromptFormat(**BASE)
    for questions in (ONE, TWO):
        counted = counted_tokens(tok, L.PromptFormat(**BASE, layout="header"), STATE, questions)["total"]
        for limit, twice in ((counted - 1, False), (counted, True)):
            fmt = L.PromptFormat(**BASE, layout="header", read_twice_under=limit)
            rows, _ = engine(fmt, tok)._prepare_separate(STATE, questions)
            assert [r[0] for r in rows] == oracle(tok, STATE, questions, plain, True, twice), (len(questions), limit)
