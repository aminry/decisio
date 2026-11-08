# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A task context's boundary in the prefix cache (docs/tasks.md, "The context in the prefix cache";
vllm_engine.LettersEngine._prepare_separate and _answer_separate), on the request builder with a recording engine; no
model. The MLX engine's two-step prefix is in tests/unit/test_mlx_engine.py (it needs a tiny model).

B1  the padding: a state read with its task's context (`context_state`) is padded from the context's end, so the
    context ends on the block and the pad is the same for every state of the task (under always, shared and row, one
    question or several); the rows are the context request's text, the pad aside; none never pads, between pads after
    the state and leaves the context in place; a base that does not pad builds the all-in-state rows
B2  unchanged without a context: rows, pads and warm-ups as before, no context facts; a plain string that happens to
    start with the context is a state like any other
B3  the registration: the context's warm-up (its tokens and one more) is sent once per context and adapter, in an engine
    call of its own ahead of the request's rows under `before`, after the response under `after`, never under `off`;
    a later request of the task finds it; one found evicted is registered again; on the Gemma bases and on the Qwen
    base (which needs no state registration); two requests of a call send one warm-up; each caller of a batched call
    gets its own facts
B4  the fallback: a state whose tokens do not start with the context's own (the stand-in tokenizer merges a leading
    space with the blank line) is built as without the boundary, counted in the facts and logged once per context
B5  the header layout: the boundary includes the header the rows carry
B6  through the server: a task's registration examples and its requests are read with the boundary, one pad for all;
    `context=` is in `x-decisio-stages` for every request with a context, a certified task's too
B7  eviction, on a modelled prefix cache: a context found registered is checked on the first engine read of the request
    that starts from its boundary (the state's warm-up when one is sent, which would otherwise make the question look
    cached), so a context evicted from the cache is registered again by the next request, under before and after
B8  the intent head's read on the serving engine (SingleEngineHidden) uses the same context boundary: the rows of the
    letters path, the context's warm-up once per context, found on the next read, registered again after eviction

    uv run pytest -q tests/unit/test_context_boundary.py
"""

import os
import pickle

import numpy as np
import pytest
from test_certified_answers import examples, post_task, wire
from test_layout_flags import StubTok
from test_request_layout import RowEngine, server
from test_state_boundary import OTHER, Q1, Q2, STATE, Recording

from decisio.readout.letters import CONTEXT_SEP, QUESTIONS_HEAD, ContextState, PromptFormat, context_state
from decisio.serve import boundary
from decisio.serve import vllm_engine as sv
from decisio.serve.hidden_engine import SingleEngineHidden
from decisio.serve.vllm_engine import ContextBoundary, split_info

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
CONTEXT = (
    "Refund policy: a refund is due within thirty days of purchase, for any reason, once per customer. "
    "Items bought on sale are refunded at the price paid. Shipping is refunded only when the item arrived damaged. "
) * 3
CONTEXT = CONTEXT.strip()
# as long as CONTEXT, so it passes a cache hit unit on every base, as CONTEXT does
OTHER_CONTEXT = (
    "Escalation policy: route every message that names a lawyer to the legal queue before anything else. "
    "A message that threatens a chargeback goes to billing disputes, and one that names the press goes to the lead. "
) * 3
OTHER_CONTEXT = OTHER_CONTEXT.strip()
STATES = [STATE, OTHER, "short", {"message": "an object state", "order": 7}]
QWEN, G12, G31 = "qwen3.6-35b-a3b", "gemma-4-12b", "gemma-4-31b"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def pads(ids):
    """The pad tokens at the front of a row."""
    n = 0
    while n < len(ids) and ids[n] == sv.PAD_TOKEN:
        n += 1
    return n


def ctx_warm(eng, state, questions=Q1):
    """The context's warm-up of a request: its rows' first context-boundary tokens and one more."""
    prepared = eng._prepare_separate(state, questions)
    C = prepared.context.tokens
    return prepared[0][0][0][: C + 1]


def stub_engine(pad_unit=None, policy="none", layout="plain", where="front"):
    """The row builder on the stand-in tokenizer (whitespace runs are one token)."""
    from test_layout_flags import engine

    eng = engine(PromptFormat(layout=layout), StubTok(), policy)
    eng.pad_unit, eng.pad_where = pad_unit, where
    return eng


# ---- B1 -------------------------------------------------------------------------------------------------------------


def test_b1_the_context_state_is_the_all_in_state_text_and_says_where_the_context_ends():
    cs = context_state(CONTEXT, {"message": "hello", "_meta": "never shown"})
    assert isinstance(cs, ContextState) and cs.context == CONTEXT
    assert cs == f"{CONTEXT}{CONTEXT_SEP}message: hello"
    assert context_state(None, "plain") == "plain" and not isinstance(context_state("", "plain"), ContextState)
    again = pickle.loads(pickle.dumps(cs))
    assert again == cs and again.context == CONTEXT


@pytest.mark.parametrize("policy", ["always", "shared", "row"])
@pytest.mark.parametrize("questions", [Q1, Q1 + Q2], ids=["one", "two"])
def test_b1_the_pad_aligns_the_contexts_end_the_same_for_every_state(tok, policy, questions):
    eng = Recording(tok, QWEN)
    eng.pad_policy = policy
    bare = Recording(tok, QWEN)
    bare.pad_policy = "none"
    ks, cs = set(), set()
    for state in STATES:
        read = context_state(CONTEXT, state)
        prepared = eng._prepare_separate(read, questions)
        rows, P = prepared
        C = prepared.context.tokens
        assert C % 1056 == 0 and 0 < C < P and prepared.context.fallback is None
        k = pads(rows[0][0])
        unpadded = bare._prepare_separate(read, questions)
        assert unpadded.context.tokens == C - k  # the context's own tokens, then the pad sized from them
        assert P == unpadded[1] + k
        # the rows are the context request's text: the all-in-state request's tokens with the pad in front
        plain_rows, _ = bare._prepare_separate(str(read), questions)
        assert [ids for ids, _ in rows] == [[sv.PAD_TOKEN] * k + ids for ids, _ in plain_rows]
        ks.add(k)
        cs.add(C)
    assert len(ks) == 1 and len(cs) == 1  # one pad, one boundary for every state of the task


def test_b1_none_never_pads_and_between_leaves_the_context_in_place(tok):
    none = Recording(tok, QWEN)
    none.pad_policy = "none"
    read = context_state(CONTEXT, STATE)
    rows, P = prepared = none._prepare_separate(read, Q1)
    _, n, _ = none._build_rows(read, Q1)
    assert pads(rows[0][0]) == 0 and P == n
    between = Recording(tok, QWEN)
    between.pad_where = "between"
    b = between._prepare_separate(read, Q1)
    c = prepared.context.tokens
    assert b.context.tokens == c and b[0][0][0][:c] == rows[0][0][:c]  # the pad goes after the state
    _, n, _ = between._build_rows(read, Q1)
    assert b[1] == n + (-n % 1056)  # sized from the state's end, as without a context


def test_b1_a_base_that_does_not_pad_builds_the_all_in_state_rows(tok):
    eng = Recording(tok, G31)  # pad_to none
    read = context_state(CONTEXT, STATE)
    prepared = eng._prepare_separate(read, Q1 + Q2)
    assert prepared == eng._prepare_separate(str(read), Q1 + Q2)
    assert prepared.context.tokens < prepared[1] and pads(prepared[0][0][0]) == 0


# ---- B2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", [QWEN, G12])
def test_b2_a_request_without_a_context_is_unchanged(tok, base):
    for state in (STATE, str(context_state(CONTEXT, STATE))):  # a plain string is a state like any other
        eng = Recording(tok, base)
        prepared = eng._prepare_separate(state, Q1)
        rows, P = prepared
        _, n, _ = eng._build_rows(state, Q1)
        k = (-n % eng.pad_unit) if eng.pad_unit else 0
        assert prepared.context is None and P == n + k and pads(rows[0][0]) == k
        assert isinstance(prepared, tuple) and tuple(prepared) == (rows, P)
        _, info = eng.answer(state, Q1)
        assert "context_boundary" not in info and "context_ms" not in info
        # the warm-ups as before: the state's on a Gemma base (before), none on the Qwen base
        assert eng.sent == [{"rows": [rows[0][0]], "warm": [rows[0][0][: P + 1]] if base == G12 else []}]


# ---- B3 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", [G12, G31, QWEN])
def test_b3_before_the_contexts_warm_up_goes_once_ahead_of_the_rows(tok, base):
    eng = Recording(tok, base)  # --register-boundary before
    first = context_state(CONTEXT, STATE)
    rows, P = prepared = eng._prepare_separate(first, Q1)
    C = prepared.context.tokens
    _, info = eng.answer(first, Q1)
    assert eng.sent[0] == {"rows": [], "warm": [rows[0][0][: C + 1]]}  # an engine call of its own, first
    assert eng.sent[1]["rows"] == [rows[0][0]]
    assert info["context_boundary"] == {"registered": 1, "found": 0, "deferred": 0, "unaligned": 0, "tokens": C}
    assert isinstance(info["context_ms"], float)
    if base == QWEN:
        assert eng.sent[1]["warm"] == [] and "state_boundary" not in info  # the Qwen base registers no state
    else:  # the state's own boundary as without a context, now reading the context from the cache
        assert eng.sent[1]["warm"] == [rows[0][0][: P + 1]] and info["state_boundary"]["registered"] == 1
    for state in (OTHER, "short"):  # the next requests of the task: the context is found, no warm-up of it
        n = len(eng.sent)
        _, info = eng.answer(context_state(CONTEXT, state), Q1)
        assert info["context_boundary"]["found"] == 1 and info["context_boundary"]["registered"] == 0
        assert all(e["rows"] for e in eng.sent[n:])  # no engine call of warm-ups alone
        assert all(w != rows[0][0][: C + 1] for e in eng.sent[n:] for w in e["warm"])
    # another context is registered on its own; another adapter too
    _, info = eng.answer(context_state(OTHER_CONTEXT, STATE), Q1)
    assert info["context_boundary"]["registered"] == 1
    n = len(eng.sent)
    _, info = eng.answer(first, Q1, adapter="a")
    assert info["context_boundary"]["registered"] == 1 and eng.sent[n] == {"rows": [], "warm": [rows[0][0][: C + 1]]}


def test_b3_a_context_found_evicted_is_registered_again(tok):
    eng = Recording(tok, G12)
    eng.answer(context_state(CONTEXT, STATE), Q1)
    eng.cached = [0]  # the next request finds the boundary gone
    assert eng.answer(context_state(CONTEXT, OTHER), Q1)[1]["context_boundary"]["found"] == 1
    eng.cached = None
    n = len(eng.sent)
    _, info = eng.answer(context_state(CONTEXT, "short"), Q1)  # so the one after registers it again
    assert info["context_boundary"]["registered"] == 1
    assert eng.sent[n] == {"rows": [], "warm": [ctx_warm(eng, context_state(CONTEXT, "short"))]}


def test_b3_after_the_contexts_warm_up_follows_the_response(tok):
    eng = Recording(tok, G12, register="after")
    first = context_state(CONTEXT, STATE)
    _, info = eng.answer(first, Q1)
    assert all(e["rows"] for e in eng.sent) and eng.sent[0]["warm"] == []  # answered without any warm-up
    assert info["context_boundary"]["deferred"] == 1 and info["state_boundary"]["deferred"] == 1
    assert eng.registrar.facts()["pending"] == 2  # the state's and the context's
    _, info = eng.answer(context_state(CONTEXT, OTHER), Q1)  # the next request sends both first, then finds it
    assert info["state_boundary"]["ran_before"] == 2
    assert ctx_warm(eng, first) in eng.sent[1]["warm"] and eng.sent[1]["rows"] == []
    assert info["context_boundary"]["found"] == 1 and info["context_boundary"]["deferred"] == 0


def test_b3_off_never_registers_the_context(tok):
    eng = Recording(tok, QWEN, register="off")
    for state in (STATE, OTHER):
        _, info = eng.answer(context_state(CONTEXT, state), Q1)
        assert info["context_boundary"] == {"registered": 0, "found": 0, "deferred": 0, "unaligned": 0} | {
            "tokens": info["context_boundary"]["tokens"]
        }
    assert all(e["rows"] for e in eng.sent) and eng.registrar.facts()["pending"] == 0


def test_b3_an_engine_without_a_hit_unit_or_a_cache_registers_nothing(tok):
    no_hit = Recording(tok, G31, hit=None)  # the CPU stand-in and MLX report no hit unit
    no_cache = Recording(tok, G31)
    no_cache.prefix_caching = False
    for eng in (no_hit, no_cache):
        _, info = eng.answer(context_state(CONTEXT, STATE), Q1)
        assert info["context_boundary"]["registered"] == 0 and all(e["rows"] for e in eng.sent)


def test_b3_two_requests_of_one_call_send_one_warm_up_and_each_gets_its_own_facts(tok):
    eng = Recording(tok, QWEN)
    a, b = context_state(CONTEXT, STATE), context_state(CONTEXT, OTHER)
    probs, info = eng._answer_separate([(a, Q1), (b, Q1), (STATE, Q1)], None, together=True)
    assert eng.sent[0] == {"rows": [], "warm": [ctx_warm(eng, a)]} and len(eng.sent) == 2
    assert info["context_boundary"] == {"registered": 2, "found": 0, "deferred": 0, "unaligned": 0}
    rows_total = sum(len(p) for p in probs)
    first, last = split_info(info, 0, 1, rows_total), split_info(info, 2, 3, rows_total)
    assert first["context_boundary"]["registered"] == 1 and first["context_boundary"]["tokens"] is not None
    assert "context_boundary" not in last and "context_ms" not in last  # the request without a context
    serial = Recording(tok, QWEN)
    serial.answer_many([(a, Q1), (b, Q1)])  # a serial call of several requests sends one too
    assert [e for e in serial.sent if not e["rows"]] == [{"rows": [], "warm": [ctx_warm(serial, a)]}]


def test_b3_a_pending_context_registration_is_sent_ahead_by_a_request_before_the_response(tok):
    eng = Recording(tok, G12, register="after")
    first = context_state(CONTEXT, STATE)
    ticket, token = boundary.open_ticket()  # a server request whose response is not out yet
    try:
        eng.answer(first, Q1)
        n = len(eng.sent)
        _, info = eng.answer(context_state(CONTEXT, OTHER), Q1)  # the same context: sent ahead, as for a state
    finally:
        boundary._TICKET.reset(token)
    assert eng.sent[n] == {"rows": [], "warm": [ctx_warm(eng, first)]} and info["context_boundary"]["registered"] == 1
    ticket.release()


# ---- B4 -------------------------------------------------------------------------------------------------------------


def test_b4_a_state_whose_tokens_merge_with_the_blank_line_is_built_as_without_the_boundary(capsys):
    eng = stub_engine(pad_unit=16, policy="always")
    merged = context_state(CONTEXT, " starts with a space")  # "\n\n " is one token of the stand-in
    prepared = eng._prepare_separate(merged, Q1)
    assert prepared.context == ContextBoundary(None, "merge")
    _, n, _ = eng._build_rows(merged, Q1)
    assert prepared == eng._prepare_separate(str(merged), Q1) and prepared[1] == n + (-n % 16)
    eng._prepare_separate(context_state(CONTEXT, " again"), Q1)
    assert capsys.readouterr().out.count("merges with the context's blank line") == 1  # once per context
    fine = eng._prepare_separate(context_state(CONTEXT, "no space"), Q1)
    assert fine.context.tokens % 16 == 0 and fine.context.fallback is None


def test_b4_the_fallback_is_counted_and_registers_nothing():
    eng = Recording(StubTok(), G12)
    _, info = eng.answer(context_state(CONTEXT, " starts with a space"), Q1)
    assert info["context_boundary"] == {"registered": 0, "found": 0, "deferred": 0, "unaligned": 1, "tokens": None}
    assert all(e["rows"] for e in eng.sent)


# ---- B5 -------------------------------------------------------------------------------------------------------------


def test_b5_under_the_header_layout_the_boundary_includes_the_header():
    eng = stub_engine(layout="header")
    seen = {}
    for questions in (Q1, Q1 + Q2):
        for state in ("one state", "another state entirely"):
            prepared = eng._prepare_separate(context_state(CONTEXT, state), questions)
            C = prepared.context.tokens
            head = eng.tok.decode(prepared[0][0][0][:C])
            assert QUESTIONS_HEAD in head and head.endswith(CONTEXT + CONTEXT_SEP)
            assert all(q["instructions"] in head for q in questions)
            seen.setdefault(len(questions), set()).add(C)
    assert all(len(v) == 1 for v in seen.values()) and seen[1] != seen[2]  # per question set, the same for states


# ---- B6 -------------------------------------------------------------------------------------------------------------


def test_b6_the_server_reads_a_tasks_examples_and_requests_with_the_boundary():
    eng = RowEngine("plain")
    eng.pad_unit, eng.pad_policy = 16, "always"
    client, _ = server(eng, "plain")
    eng.scored.clear()
    assert post_task(client, {"id": "t", "examples": examples(12, 0.9), "context": CONTEXT}).status_code == 200
    for text in ("0:0.9:x", "1:0.8:a much longer message that moves the state's end"):
        r = client.post("/v1/systemone", json=wire(text))
        assert r.status_code == 200 and r.headers["x-decisio-tasks"] == "t"
        assert "context=" in r.headers["x-decisio-stages"]
    k = {pads(ids) for ids in eng.scored}
    assert len(k) == 1 and k != {0}  # one pad for the examples and the requests alike
    (k,) = k
    c = len(eng._prefix_ids(eng._template_head() + CONTEXT + CONTEXT_SEP))
    assert (k + c) % 16 == 0 and len({tuple(ids[: k + c]) for ids in eng.scored}) == 1  # one prefix, on the block


def test_b6_a_certified_tasks_request_says_context_in_its_stages():
    eng = RowEngine("plain")
    client, _ = server(eng, "plain")
    body = {"id": "t", "examples": examples(400, 0.96), "context": CONTEXT, "certify": {}}
    r = post_task(client, body)
    assert r.status_code == 200, r.text
    assert r.json()["certificate"]  # every question of the request is read by the certificate's function
    r = client.post("/v1/systemone", json=wire("0:0.9:x"))
    assert r.status_code == 200 and r.headers["x-decisio-tasks"] == "t"
    assert "task=" in r.headers["x-decisio-stages"] and "context=" in r.headers["x-decisio-stages"]
    assert "engine=" not in r.headers["x-decisio-stages"]  # no question left for the request's own engine call


# ---- B7 -------------------------------------------------------------------------------------------------------------


class Modelled(Recording):
    """The recording engine with a prefix cache like vLLM's on these bases, modelled: every read (a warm-up or a
    question row) keeps its latest checkpoint, all its tokens but the last; a read takes from the cache the whole hit
    units of the longest kept checkpoint it starts with, and says so per warm-up (`warm_cached_tokens`) and per row.
    `evict()` empties the cache."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.kept = set()

    def read(self, ids):
        hit = max((len(p) for p in self.kept if tuple(ids[: len(p)]) == p), default=0)
        self.kept.add(tuple(ids[:-1]))
        return (hit // self.cache_hit_unit) * self.cache_hit_unit

    def evict(self):
        self.kept.clear()

    def score_prompts(self, rows, adapter=None, warm=(), **kw):
        self.sent.append({"rows": [ids for ids, _ in rows], "warm": list(warm)})
        warm_cached = [self.read(w) for w in warm]
        cached = [self.read(ids) for ids, _ in rows]
        info = {"cached_tokens": cached, **({"warm_cached_tokens": warm_cached} if warm else {})}
        return [np.ones(len(lab)) / len(lab) for _, lab in rows], info

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        ms = super().send_warm(warm, adapter, mm, mm_uuids)
        for w in warm:
            self.read(w)
        return ms


def context_warm_ups(eng, n=0):
    """The engine calls from the n-th on that were a context's warm-up alone."""
    return [e for e in eng.sent[n:] if not e["rows"]]


@pytest.mark.parametrize(
    "base,register,questions",
    [(G12, "before", Q1), (G31, "before", Q1), (G12, "before", Q1 + Q2), (G12, "after", Q1 + Q2), (G12, "after", Q1),
     (QWEN, "before", Q1 + Q2)],
    ids=["12b-before-one", "31b-before-one", "12b-before-two", "12b-after-two", "12b-after-one", "qwen-two"],
)  # fmt: skip
def test_b7_a_context_evicted_from_the_cache_is_registered_again(tok, base, register, questions):
    eng = Modelled(tok, base, register=register)
    states = (f"state number {i} of the task" for i in range(100))  # a new state on every request

    def ask():
        _, info = eng.answer(context_state(CONTEXT, next(states)), questions)
        return info["context_boundary"]

    ask()
    if register == "after":
        ask()  # the deferred warm-up went ahead of this request's rows
    cb = ask()
    assert cb["found"] == 1 and cb["registered"] == 0 and cb["deferred"] == 0  # registered and in the cache
    eng.evict()
    assert ask()["found"] == 1  # this request reads the context again, and finds it gone
    n = len(eng.sent)
    cb = ask()  # so this one registers it again
    if register == "after":
        assert cb["deferred"] == 1
        cb = ask()
        assert cb["found"] == 1
    else:
        assert cb["registered"] == 1 and len(context_warm_ups(eng, n)) == 1
    cb = ask()
    assert cb["found"] == 1 and cb["registered"] == 0 and cb["deferred"] == 0  # and it stays found


# ---- B8 -------------------------------------------------------------------------------------------------------------


class Head(SingleEngineHidden):
    """The intent head's reader on the serving engine (`SingleEngineHidden`) over a modelled engine: its label read is
    the engine's own (`engine`, the Gemma bases) or computed from h (`recompute`, the Qwen base); the hidden-state
    chunks of a row are one read of the modelled cache, which says what it took from the cache as the vLLM reader
    does."""

    def __init__(self, engine, label_source):
        self.engine, self.tok, self._lock, self.label_source = engine, engine.tok, engine._lock, label_source
        self.hidden = []

    def hidden_rows(self, token_lists):
        self.hidden += [list(ids) for ids in token_lists]
        self._cached = [self.engine.read(ids) for ids in token_lists]
        return [np.ones(4, dtype=np.float32) for _ in token_lists]

    def label_lp(self, h, lab):
        return np.log(np.ones(len(lab)) / len(lab))


@pytest.mark.parametrize(
    "base,source,register", [(G12, "engine", "before"), (G12, "engine", "after"), (QWEN, "recompute", "before")]
)
def test_b8_the_intent_heads_read_registers_the_context_as_the_letters_path_does(tok, base, source, register):
    eng = Modelled(tok, base, register=register)
    head = Head(eng, source)
    states = (f"message number {i} for the head" for i in range(100))

    def read():
        state = context_state(CONTEXT, next(states))
        n, h = len(eng.sent), len(head.hidden)
        out = head.readout(state, Q2)
        assert len(out) == 1
        assert head.hidden[h:] == [eng._prepare_separate(state, Q2)[0][0][0]]  # the letters path's row, as before
        return head.read_info()["context_boundary"], eng.sent[n:], state

    cb, sent, state = read()
    if register == "after":
        assert cb["deferred"] == 1 and not context_warm_ups(eng)  # the read did not wait for it
        cb, sent, state = read()  # the registration went ahead of this read
        assert sent[0] == {"rows": [], "warm": [ctx_warm(eng, context_state(CONTEXT, "message number 0 for the head"),
                                                         Q2)]}  # fmt: skip
        assert cb["found"] == 1
    else:
        assert cb["registered"] == 1 and sent[0] == {"rows": [], "warm": [ctx_warm(eng, state, Q2)]}
        assert isinstance(head.read_info()["context_ms"], float)
    cb, sent, _ = read()
    assert cb["found"] == 1 and not [e for e in sent if not e["rows"]]  # found: no warm-up of it
    eng.evict()
    assert read()[0]["found"] == 1  # this read finds it gone
    cb, sent, _ = read()
    if register == "after":
        assert cb["deferred"] == 1
    else:
        assert cb["registered"] == 1  # so the next one registers it again
    # the letters path and the head share the registration: a letters request of the task finds it
    _, info = eng.answer(context_state(CONTEXT, "a letters request"), Q1)
    assert info["context_boundary"]["found"] == 1
    # a state without a context: no context facts
    head.readout("no context here", Q2)
    assert head.read_info() is None
