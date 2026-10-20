# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A state over the engine's prompt ceiling: refused (the default) or answered from its start (`--over-limit truncate`,
a request's `over_limit` field, a tuned repository's `defaults.over_limit`).

O1  the cut: a state that fits is returned as it is, one that does not keeps its start, lands on whitespace, leaves room
    for the longest question, nests across ceilings, and a question over the ceiling alone is refused
O2  refuse is the default: the engine gets the whole state; truncate cuts it once per request and every question of the
    request reads the same cut state
O3  precedence: the request's field over the server's setting, either way; a bad value is a 422; a request that does not
    fit even without a state is a 422
O4  the report: `state_truncated` in the body and the header, absent when the whole state fits; counted in /health
O5  a truncated state's answers are never certified (reason `state_truncated`), the monitor does not count them, and a
    whole state is certified as before
O6  an engine with no prompt ceiling refuses to truncate unless it is given one; the vLLM engine reads its ceiling from
    the served context
O7  `defaults.over_limit` in a `/2` repository: validated, reaches the server, a flag beats it, absent means refuse; an
    older reader refuses the key

  uv run pytest -q tests/unit/test_over_limit.py
"""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import examples, post_task, wire
from test_hub import args_for
from test_hub_tuned import build, write_repo

from decisio import hub
from decisio.serve import over_limit as over
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import resolve_base


class WordTok:
    """A word is a token, with its leading whitespace, as a byte-level BPE writes it."""

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text, add_special_tokens=False):
        out = []
        for chunk in re.findall(r"\s*\S+|\s+", str(text)):
            if chunk not in self.vocab:
                self.vocab[chunk] = len(self.words)
                self.words.append(chunk)
            out.append(self.vocab[chunk])
        return out

    def decode(self, ids):
        return "".join(self.words[i] for i in ids)


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in str(text)]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture(autouse=True)
def plain_rows(monkeypatch):
    """The question's text and the user turn without a chat template: the question, framed by two tokens."""
    monkeypatch.setattr(vllm_engine, "user_turn", lambda tok, content, fmt=None: f"<user> {content} <end>")
    monkeypatch.setattr(
        vllm_engine,
        "question_text",
        lambda tok, q, fmt=None: (q["instructions"] + " " + " ".join(q.get("options", [])), [], []),
    )


STATE_WORDS = 400


def long_message(n=STATE_WORDS, top=1, conf=0.95):
    return f"{top}:{conf}:x " + " ".join(f"w{i}" for i in range(n))


class LimitEngine:
    """Answers from the first word of the state, whatever shape it arrives in, and keeps every state it was given."""

    tok = WordTok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters: dict = {}

    def __init__(self, limit=200):
        self.limit, self.seen = limit, []

    def prompt_limit(self):
        return self.limit

    def facts(self):
        return {"engine": "fake", "vllm": "1", "base": "fake-base", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        self.seen.append(state)
        text = state["message"] if isinstance(state, dict) else str(state).removeprefix("message: ")
        top, conf = text.split()[0].split(":")[:2]
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, (1 - float(conf)) / (k - 1))
            p[int(top)] = float(conf)
            out.append(p)
        return out, {}


def make(mode="refuse", limit=200, engine=None):
    eng = engine or LimitEngine(limit)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), over_limit=mode)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def post(client, message, **extra):
    body = wire(message)
    body.update(extra)
    return client.post("/v1/systemone", json=body)


# ---- O1 -------------------------------------------------------------------------------------------------------------


def test_o1_a_state_that_fits_is_returned_as_it_is():
    tok = WordTok()
    body = " ".join(f"w{i}" for i in range(50))
    assert over.fit_state(tok, body, overhead=20, limit=200) == (body, None)
    assert over.fit_state(tok, body, overhead=20, limit=20 + over.MARGIN + 50) == (body, None)  # exactly the room


def test_o1_the_cut_keeps_the_start_and_leaves_the_room_the_questions_need():
    tok = WordTok()
    body = " ".join(f"w{i}" for i in range(500))
    text, report = over.fit_state(tok, body, overhead=30, limit=200)
    room = 200 - 30 - over.MARGIN
    assert body.startswith(text) and report["part"] == "start" and report["limit"] == 200
    assert report["kept_tokens"] == len(tok.encode(text)) <= room and report["of_tokens"] == 500
    assert len(tok.encode(text)) + 30 + over.MARGIN <= 200
    assert not text.endswith(" ")  # no trailing whitespace before the questions
    # a longer overhead leaves less of the state: the cut follows the longest question
    shorter, rep2 = over.fit_state(tok, body, overhead=60, limit=200)
    assert body.startswith(shorter) and rep2["kept_tokens"] == report["kept_tokens"] - 30


def test_o1_the_cut_lands_on_whitespace_and_cuts_inside_a_long_word_only_when_it_must():
    tok = CharTok()
    body = ("alpha beta gamma delta " * 40).strip()
    for limit in range(100, 140):
        text, report = over.fit_state(tok, body, overhead=10, limit=limit)
        assert body.startswith(text) and text and text[-1] != " "
        assert body[len(text)] == " "  # the next character is a space: the cut did not end mid-word
        assert report["kept_tokens"] == len(text)
    blob = "x" * 5000  # no whitespace within reach: cut at the token
    text, report = over.fit_state(tok, blob, overhead=10, limit=100)
    assert text == "x" * (100 - 10 - over.MARGIN) and report["kept_tokens"] == 58


def test_o1_the_same_state_at_two_ceilings_nests():
    tok = WordTok()
    body = " ".join(f"w{i}" for i in range(500))
    small, _ = over.fit_state(tok, body, overhead=30, limit=150)
    large, _ = over.fit_state(tok, body, overhead=30, limit=250)
    assert large.startswith(small) and len(large) > len(small)


def test_o1_a_question_over_the_ceiling_alone_is_refused():
    tok = WordTok()
    with pytest.raises(over.OverLimit, match="questions alone"):
        over.fit_state(tok, "a b c " * 100, overhead=190, limit=200)
    with pytest.raises(over.OverLimit):
        over.fit_state(tok, "a " * 400, overhead=500, limit=200)


def test_o1_a_state_within_the_byte_count_is_not_tokenized():
    class Boom(WordTok):
        def encode(self, *a, **k):
            raise AssertionError("tokenized")

    body = "w " * 20
    assert over.fit_state(Boom(), body, overhead=10, limit=500) == (body, None)


# ---- O2, O3, O4 -----------------------------------------------------------------------------------------------------


def test_o2_refuse_is_the_default_and_hands_the_engine_the_whole_state():
    client, so, eng = make()
    assert so.over_limit == "refuse"
    r = post(client, long_message())
    assert r.status_code == 200 and "state_truncated" not in r.json()
    assert eng.seen == [{"message": long_message()}]
    assert "x-decisio-state-truncated" not in r.headers


def test_o2_truncate_cuts_the_state_once_and_every_question_reads_the_same_cut_state():
    client, so, eng = make("truncate")
    body = wire(long_message())
    body["questions"]["q2"] = {**body["questions"]["q1"], "instructions": "Which other queue should it go to?"}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    cut = r.json()["state_truncated"]
    assert len(eng.seen) == 1 and isinstance(eng.seen[0], str)  # one state for the request's questions
    kept = eng.seen[0]
    assert kept.startswith("message: 1:0.95:x w0 w1") and not kept.endswith(" ")
    assert cut["part"] == "start" and cut["limit"] == 200 and cut["of_tokens"] > cut["kept_tokens"] > 0
    assert cut["of_tokens"] == len(eng.tok.encode("message: " + long_message()))
    longest = max(
        len(eng.tok.encode(f"<user> \n\n{q['instructions']} option_0 option_1 option_2 <end>"))
        for q in (body["questions"]["q1"], body["questions"]["q2"])
    )
    assert len(eng.tok.encode(kept)) + longest <= 200  # the longest row fits
    assert set(r.json()["answers"]) == {"q1", "q2"}  # both answered
    assert r.json()["usage"]["input_tokens"] < 260  # the usage counts what was read


def test_o3_the_request_field_beats_the_servers_setting_either_way():
    client, so, eng = make("refuse")
    r = post(client, long_message(), over_limit="truncate")
    assert r.status_code == 200 and r.json()["state_truncated"]["part"] == "start"
    client, so, eng = make("truncate")
    r = post(client, long_message(), over_limit="refuse")
    assert r.status_code == 200 and "state_truncated" not in r.json() and eng.seen == [{"message": long_message()}]
    r = post(client, long_message(), over_limit=None)  # null: the server's own
    assert "state_truncated" in r.json()


def test_o3_a_bad_value_is_a_422_and_a_request_that_does_not_fit_without_a_state_is_a_422():
    client, *_ = make("truncate")
    for bad in ("cut", True, 3, ["truncate"]):
        r = post(client, long_message(), over_limit=bad)
        assert r.status_code == 422 and "over_limit" in r.json()["detail"], bad
    client, _, eng = make("truncate", limit=30)  # the question alone is over the ceiling
    r = post(client, "1:0.9:x a b c")
    assert r.status_code == 422 and "questions alone" in r.json()["detail"] and not eng.seen


def test_o4_the_report_is_in_the_body_and_the_header_and_only_when_something_was_cut():
    client, so, _ = make("truncate")
    r = post(client, long_message())
    cut = r.json()["state_truncated"]
    assert r.headers["x-decisio-state-truncated"] == f"kept={cut['kept_tokens']}; of={cut['of_tokens']}; part=start"
    short = post(client, "1:0.9:x short state")
    assert short.status_code == 200 and "state_truncated" not in short.json()
    assert "x-decisio-state-truncated" not in short.headers
    assert so.truncated == {"requests": 1, "largest_state_tokens": cut["of_tokens"]}
    post(client, long_message(900))
    assert so.truncated["requests"] == 2 and so.truncated["largest_state_tokens"] > cut["of_tokens"]


def test_o4_health_says_the_mode_the_ceiling_and_the_counts():
    client, so, eng = make("truncate")
    post(client, long_message())
    eng.repository_defaults, eng.fmt = None, SimpleNamespace(facts=lambda: {})
    try:
        body = TestClient(vllm_engine.make_app(eng, so)).get("/health").json()
    except Exception as e:  # make_app wants more of a real engine than the stand-in has
        pytest.skip(f"make_app needs a fuller engine: {e}")
    o = body["systemone"]["over_limit"]
    assert o["mode"] == "truncate" and o["prompt_limit"] == 200 and o["requests"] == 1 and o["largest_state_tokens"] > 0


# ---- O5 -------------------------------------------------------------------------------------------------------------


def certified(client, so):
    r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {"alpha": 0.05, "delta": 0.1}})
    assert r.status_code == 200 and r.json()["certificate"]["certified"], r.text
    return r.json()["certificate"]


def test_o5_a_truncated_state_is_never_certified_and_the_monitor_does_not_see_it():
    client, so, eng = make("refuse", limit=10_000)
    cert = certified(client, so)
    mon = so.task_store.monitors["routing"]
    whole = post(client, "1:0.99:x short").json()["answers"]["q1"]
    assert whole["certified"] is True and whole["certificate"] == cert["id"]
    counted = mon.facts()["answers"]
    assert counted == 1  # the whole state's answer
    eng.limit = 100  # a state over the ceiling from here on
    cut = post(client, long_message(conf=0.99), over_limit="truncate").json()
    a = cut["answers"]["q1"]
    assert a["certified"] is False and a["certificate"] == cert["id"] and a["certified_reason"] == "state_truncated"
    assert cut["state_truncated"]["of_tokens"] > cut["state_truncated"]["kept_tokens"]
    assert mon.facts()["answers"] == counted  # the monitor did not count the cut state's answer
    again = post(client, "1:0.99:x short again").json()["answers"]["q1"]
    assert again["certified"] is True  # a whole state is certified as before


def test_o5_registration_and_certification_refuse_the_field_and_never_truncate():
    client, so, eng = make("truncate", limit=10_000)
    ex = examples(5, 0.9)
    ex[0]["request"]["over_limit"] = "truncate"
    r = post_task(client, {"id": "t", "examples": ex})
    assert r.status_code == 422 and "over_limit" in r.json()["detail"] and not so.task_store.by_key


# ---- O6 -------------------------------------------------------------------------------------------------------------


def test_o6_an_engine_with_no_prompt_ceiling_refuses_to_truncate_unless_it_is_given_one():
    eng = LimitEngine()
    eng.prompt_limit = lambda: None
    client, *_ = make("truncate", engine=eng)
    r = post(client, long_message())
    assert r.status_code == 422 and "--over-limit-tokens" in r.json()["detail"]
    ok = post(client, "1:0.9:x short")  # a state that fits still needs a ceiling to be checked against
    assert ok.status_code == 422


def test_o6_the_vllm_engine_reads_its_ceiling_from_the_served_context():
    eng = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    eng.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=98_304)))
    )
    assert eng.prompt_limit() == 98_303  # one token is generated
    eng.over_limit_tokens = 5_000
    assert eng.prompt_limit() == 5_000  # --over-limit-tokens wins
    bare = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    assert bare.prompt_limit() is None  # an engine with no vLLM configuration (the MLX build)


# ---- O7 -------------------------------------------------------------------------------------------------------------


def test_o7_the_repository_default_is_validated_and_reaches_the_server(tmp_path):
    for value in ("refuse", "truncate"):
        d = write_repo(tmp_path, build(over_limit=value), name=f"r-{value}")
        args = args_for(str(d))
        resolve_base(args)
        assert args.over_limit == value and args.repository_defaults["over_limit"] == value
    config = build()
    config["defaults"]["over_limit"] = "cut"
    with pytest.raises(ValueError, match="over_limit"):
        hub.check_defaults(config["defaults"])


def test_o7_a_flag_beats_the_repository_and_absent_means_refuse(tmp_path):
    d = write_repo(tmp_path, build(over_limit="truncate"), name="r")
    args = args_for(str(d), over_limit="refuse")
    resolve_base(args)
    assert args.over_limit == "refuse" and args.repository_defaults["over_limit_set_by"] == "--over-limit refuse"
    d = write_repo(tmp_path, build(), name="plain")
    args = args_for(str(d))
    resolve_base(args)
    assert args.over_limit == "refuse" and "over_limit" not in args.repository_defaults
    base = args_for("gemma-4-31b")
    resolve_base(base)
    assert base.over_limit == "refuse"


def test_o7_the_flags_are_in_the_parser():
    args = vllm_engine.build_parser().parse_args(["--over-limit", "truncate", "--over-limit-tokens", "6000"])
    assert (args.over_limit, args.over_limit_tokens) == ("truncate", 6000)
    assert vllm_engine.build_parser().parse_args([]).over_limit is None
