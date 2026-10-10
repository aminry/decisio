# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A request with one question registers its state's boundary in the prefix cache on the Gemma bases
(decisio.families register_state_boundary; vllm_engine.LettersEngine._answer_separate), on the request builder with a
recording engine; no model.

vLLM keeps only the latest sliding-window checkpoint of a finished request. On the Gemma bases it lies inside the
question, so a later, different question about a state read the whole state again (on the 31B, 531 ms instead of 47 at
3,000 tokens; RLCD experiments/2026-10-05_t7_card_e2_e3_e4). The warm-up the multi-question path sends, the state and
one token, registers the boundary; vLLM's retention interval would keep it too, but evicts states sooner (same record).

S1  the profiles: the Gemma bases register, the Qwen base does not need to (its padded state already ends on a single
    question's latest checkpoint)
S2  on a Gemma base a single question on a new state sends the warm-up (the state's P tokens and one more) before it;
    a later question on the same state sends none while the engine reports the state read from the cache; after a
    request finds the boundary gone (evicted), the next request on that state sends the warm-up again
S3  unchanged: a single question on the Qwen base, any multi-question request, --multi-question batch, and an engine
    without a cache hit unit (the CPU stand-in, MLX)
S4  the engine keeps at most BOUNDARIES_KEPT states, the least recently used dropped first
S1-S4 hold with `--register-boundary before` (0.8.1's order); with `after`, the default, the warm-up follows the
response (decisio.serve.boundary):
A1  a single question on a new state is sent without the warm-up, which is kept pending; once the call has returned,
    the next request sends it before its own questions, and a later question on the state finds the boundary
A2  while the response is not out (the server's ticket not released), the warm-up is not sent for other requests; a
    second question about the same state sends it itself, ahead of its question, and nothing is left to send after
A3  without another request, the registrar's thread sends it once the call has returned
A4  at most PENDING_KEPT registrations wait, the oldest dropped first and counted
A5  a warm-up that fails is counted and dropped; the request that sent it is answered, and the state is deferred again
A6  a registered boundary found gone (evicted) is registered again after the next response, as in S2
O1  --register-boundary off: no warm-up, before or after, and nothing pending
D1  the default: before on both Gemma bases (after on the 12B from 0.9.0 to 0.11.0 made a request that follows
    another wait behind the previous state's registration), the flag overriding
D2  one-question requests on new states back to back, on the 12B's default: nothing is left pending behind a response
X1  at exit the registrar waits for the warm-up in progress, drops and counts the pending ones, and starts none after

The GPU tier checks the effect on a card (tests/gpu/test_second_question_cached.py): a second, different question on a
3,000-token state reads it from the cache, and revisited states still do with the cache filled past its pool.

    uv run pytest -q tests/unit/test_state_boundary.py
"""

import os
import threading

import numpy as np
import pytest

from decisio.families import BASES
from decisio.serve import boundary
from decisio.serve import vllm_engine as sv

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
STATE = "Ticket 8842. Customer writes: our API integration started returning 500 errors on every request. " * 3
OTHER = "Ticket 9001. Customer writes: the invoice for September lists a seat we cancelled in August. " * 3
Q1 = [{"kind": "noul", "instructions": "Is the customer reporting an outage?"}]
Q2 = [{"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales", "account"]}]


class Recording(sv.LettersEngine):
    """The request path with a tokenizer, a base's profile and a scorer that records what it would send; `cached` is
    what the engine reports as read from the cache for each question row (all of it by default)."""

    def __init__(self, tok, base, hit=32, multi_question=None, register="before", background=False):
        fam = BASES[base]
        self.register_boundary = register
        self._registrar = boundary.Registrar(self, background=background)
        self.warmed, self.warm_started, self.gate = threading.Event(), threading.Event(), threading.Event()
        self.gate.set()
        self.tok, self.family, self.pad_token, self.pad_where = tok, fam, sv.PAD_TOKEN, "front"
        self.pad_unit = 1056 if fam.pad_to == "block" else None
        self.mode, self.adapters = "separate", {}
        self.match_unit = 1056 if fam.pad_to == "block" else 16
        self.cache_hit_unit = 1056 if fam.pad_to == "block" else hit
        self._lock = threading.Lock()
        if multi_question:
            self.multi_question = multi_question
        self.sent, self.cached = [], None

    def score_prompts(self, rows, adapter=None, warm=(), **kw):
        self.sent.append({"rows": [ids for ids, _ in rows], "warm": list(warm)})
        cached = self.cached if self.cached is not None else [len(ids) for ids, _ in rows]
        return [np.ones(len(lab)) / len(lab) for _, lab in rows], {"cached_tokens": list(cached)}

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        """A deferred registration's own engine call (decisio.serve.boundary.Registrar.run_due); held while `gate` is
        clear."""
        self.warm_started.set()
        assert self.gate.wait(30)
        self.sent.append({"rows": [], "warm": list(warm)})
        self.warmed.set()
        return 0.0

    def ask(self, state, questions):
        """The request's first engine call (the warm-up goes with it; several questions follow one call each)."""
        n = len(self.sent)
        self.answer(state, questions)
        return self.sent[n]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_s1_the_gemma_bases_register_and_the_qwen_base_needs_not():
    assert BASES["gemma-4-12b"].register_state_boundary and BASES["gemma-4-31b"].register_state_boundary
    assert not BASES["qwen3.6-35b-a3b"].register_state_boundary and BASES["qwen3.6-35b-a3b"].pad_to == "block"


@pytest.mark.parametrize("base", ["gemma-4-12b", "gemma-4-31b"])
def test_s2_a_single_question_registers_its_states_boundary(tok, base):
    eng = Recording(tok, base)
    rows, P = eng._prepare_separate(STATE, Q1)
    first = eng.ask(STATE, Q1)
    assert first["warm"] == [rows[0][0][: P + 1]] and first["rows"] == [rows[0][0]]  # the state and one token, first
    second = eng.ask(STATE, Q2)  # a different question on the same state: the boundary is registered
    assert second["warm"] == []
    repeat = eng.ask(STATE, Q1)
    assert repeat["warm"] == []  # a repeated question costs no extra engine call
    eng.cached = [0]  # the next request finds the boundary gone (evicted)
    assert eng.ask(STATE, Q2)["warm"] == []
    eng.cached = None
    again = eng.ask(STATE, Q1)  # so the one after registers it again
    assert again["warm"] == [rows[0][0][: P + 1]]
    other = eng.ask(OTHER, Q1)  # another state is registered on its own
    assert len(other["warm"]) == 1 and other["warm"][0] != rows[0][0][: P + 1]


def test_s3_unchanged_elsewhere(tok):
    qwen = Recording(tok, "qwen3.6-35b-a3b")
    assert qwen.ask(STATE, Q1)["warm"] == [] and qwen.ask(STATE, Q2)["warm"] == []  # its padding registers the boundary
    two = Q1 + Q2
    for base in ("qwen3.6-35b-a3b", "gemma-4-31b"):  # several questions: the warm-up as before, every request
        eng = Recording(tok, base)
        rows, P = eng._prepare_separate(STATE, two)
        assert eng.ask(STATE, two)["warm"] == [rows[0][0][: P + 1]] and eng.ask(STATE, two)["warm"] == [
            rows[0][0][: P + 1]
        ]
    batch = Recording(tok, "gemma-4-31b", multi_question="batch")
    assert batch.ask(STATE, Q1)["warm"] == [] and batch.ask(STATE, two)["warm"] == []
    no_cache = Recording(tok, "gemma-4-31b", hit=None)  # the CPU stand-in and MLX report no hit unit
    assert no_cache.ask(STATE, Q1)["warm"] == []


def test_s4_the_registry_is_bounded(tok, monkeypatch):
    monkeypatch.setattr(Recording, "BOUNDARIES_KEPT", 2)
    eng = Recording(tok, "gemma-4-31b")
    states = [f"Ticket {i}. The customer asks about order {i}. " * 4 for i in range(3)]
    for s in states:
        assert len(eng.ask(s, Q1)["warm"]) == 1
    assert len(eng._boundaries) == 2
    assert len(eng.ask(states[0], Q1)["warm"]) == 1  # the least recently used was dropped: registered again
    assert eng.ask(states[2], Q1)["warm"] == []


@pytest.mark.parametrize("base", ["gemma-4-12b", "gemma-4-31b"])
def test_a1_after_the_response_the_warm_up_follows_the_answer(tok, base):
    eng = Recording(tok, base, register="after")
    rows, P = eng._prepare_separate(STATE, Q1)
    _, info = eng.answer(STATE, Q1)
    assert eng.sent == [{"rows": [rows[0][0]], "warm": []}]  # answered without the warm-up
    assert info["state_boundary"] == {"registered": 0, "found": 0, "deferred": 1}
    assert eng.registrar.facts()["pending"] == 1
    _, info = eng.answer(OTHER, Q1)  # the next request, about another state, sends it first
    assert eng.sent[1] == {"rows": [], "warm": [rows[0][0][: P + 1]]}
    assert info["state_boundary"]["ran_before"] == 1 and info["state_boundary"]["deferred"] == 1
    n = len(eng.sent)
    _, info = eng.answer(STATE, Q2)  # a later, different question about the first state finds the boundary
    assert info["state_boundary"]["found"] == 1 and info["state_boundary"]["deferred"] == 0
    other, Po = eng._prepare_separate(OTHER, Q1)
    assert (
        eng.sent[n:]
        == [  # the other state's warm-up, now due, then the question alone
            {"rows": [], "warm": [other[0][0][: Po + 1]]},
            {"rows": [eng._prepare_separate(STATE, Q2)[0][0][0]], "warm": []},
        ]
    )


def test_a2_a_second_question_before_the_response_sends_the_warm_up_itself(tok):
    eng = Recording(tok, "gemma-4-12b", register="after")
    rows, P = eng._prepare_separate(STATE, Q1)
    ticket, token = boundary.open_ticket()  # a server request whose response is not out yet
    try:
        eng.answer(STATE, Q1)
        eng.answer(OTHER, Q1)  # not due: the other state's request does not send it
        assert all(e["warm"] == [] for e in eng.sent)
        second = eng.ask(STATE, Q2)  # the same state: sent ahead of the question, as 0.8.1 did
    finally:
        boundary._TICKET.reset(token)
    rows2, _ = eng._prepare_separate(STATE, Q2)  # the warm-up is the state and its own question's first token
    assert second["warm"] == [rows2[0][0][: P + 1]] and len(second["rows"]) == 1
    ticket.release()  # the first response goes out: nothing is left to send for that state
    n = len(eng.sent)
    _, info = eng.answer(STATE, Q1)
    assert info["state_boundary"]["found"] == 1
    assert [e for e in eng.sent[n:] if e["warm"] == [rows[0][0][: P + 1]]] == []


def test_a3_the_registrar_thread_sends_it_without_another_request(tok):
    eng = Recording(tok, "gemma-4-31b", register="after", background=True)
    rows, P = eng._prepare_separate(STATE, Q1)
    eng.answer(STATE, Q1)
    assert eng.warmed.wait(10)
    assert eng.sent[-1] == {"rows": [], "warm": [rows[0][0][: P + 1]]}
    with eng._lock:  # the registrar records it under the engine's lock
        assert eng.registrar.facts() == {"pending": 0, "deferred": 1, "registered": 1, "dropped": 0, "failed": 0}


def test_a4_the_pending_registrations_are_bounded(tok, monkeypatch):
    monkeypatch.setattr(boundary.Registrar, "PENDING_KEPT", 2)
    eng = Recording(tok, "gemma-4-31b", register="after")
    states = [f"Ticket {i}. The customer asks about order {i}. " * 4 for i in range(3)]
    ticket, token = boundary.open_ticket()
    try:
        for s in states:
            eng.answer(s, Q1)
    finally:
        boundary._TICKET.reset(token)
    assert eng.registrar.facts()["pending"] == 2 and eng.registrar.facts()["dropped"] == 1
    ticket.release()
    _, info = eng.answer(OTHER, Q1)
    assert info["state_boundary"]["ran_before"] == 2
    assert eng.answer(states[0], Q1)[1]["state_boundary"]["deferred"] == 1  # the dropped one: deferred again
    assert eng.answer(states[2], Q1)[1]["state_boundary"]["found"] == 1


def test_a5_a_failed_warm_up_is_dropped_and_counted(tok):
    eng = Recording(tok, "gemma-4-12b", register="after")

    def fail(*a, **k):
        raise RuntimeError("an injected failure of the warm-up")

    eng.send_warm = fail
    eng.answer(STATE, Q1)
    probs, info = eng.answer(OTHER, Q1)  # sends the failing warm-up first, and is still answered
    assert len(probs) == 1 and "ran_before" not in info["state_boundary"]
    assert eng.registrar.facts()["failed"] == 1 and eng.registrar.facts()["pending"] == 1  # OTHER's own
    assert eng.answer(STATE, Q2)[1]["state_boundary"]["deferred"] == 1  # not registered: deferred again


def test_a6_an_evicted_boundary_is_registered_again_after_the_response(tok):
    eng = Recording(tok, "gemma-4-31b", register="after")
    eng.answer(STATE, Q1)
    eng.answer(OTHER, Q1)  # sends STATE's warm-up
    assert eng.answer(STATE, Q2)[1]["state_boundary"]["found"] == 1
    eng.cached = [0]  # the next request finds it gone
    assert eng.answer(STATE, Q1)[1]["state_boundary"]["found"] == 1
    eng.cached = None
    assert eng.answer(STATE, Q2)[1]["state_boundary"]["deferred"] == 1


def test_o1_off_never_registers(tok):
    eng = Recording(tok, "gemma-4-12b", register="off")
    for q in (Q1, Q2, Q1):
        _, info = eng.answer(STATE, q)
        assert "state_boundary" not in info
    assert all(e["warm"] == [] for e in eng.sent) and eng.registrar.facts()["pending"] == 0


def test_d1_the_default_order_per_base():
    from decisio.serve.vllm_engine import resolve_register_boundary

    g12, g31 = BASES["gemma-4-12b"], BASES["gemma-4-31b"]
    assert resolve_register_boundary(g12, None) == "before" and resolve_register_boundary(g31, None) == "before"
    for choice in ("after", "before", "off"):
        assert resolve_register_boundary(g12, choice) == resolve_register_boundary(g31, choice) == choice


def test_d2_back_to_back_one_question_requests_leave_nothing_pending_on_the_12b_default(tok):
    """Lab 2's grid (12B, `after`, one question on each new 3,000-token state in a row): each later request took about
    twice a read, because it waited behind the previous state's registration. On the default (`before`) the warm-up is
    inside the request, so nothing is pending when a response goes out and no request runs another's registration."""
    from decisio.serve.vllm_engine import resolve_register_boundary

    eng = Recording(tok, "gemma-4-12b", register=resolve_register_boundary(BASES["gemma-4-12b"], None))
    states = [STATE, OTHER, STATE + " Please answer today.", OTHER + " It is urgent."]
    for state in states:
        sent = eng.ask(state, Q1)
        assert len(sent["warm"]) == 1  # registered in the request, ahead of its question
        facts = eng.registrar.facts()
        assert facts["pending"] == 0 and facts["deferred"] == 0
    # the same sequence on `after` leaves each state's registration pending behind its response (the regression)
    late = Recording(tok, "gemma-4-12b", register="after")
    ticket, token = boundary.open_ticket()
    try:
        late.answer(STATE, Q1)
    finally:
        boundary._TICKET.reset(token)
    assert late.registrar.facts()["pending"] == 1
    ticket.release()


def test_x1_at_exit_the_registrar_waits_for_the_warm_up_in_progress(tok):
    eng = Recording(tok, "gemma-4-31b", register="after", background=True)
    ticket, token = boundary.open_ticket()  # a registration pending behind a response not yet out
    try:
        eng.answer(OTHER, Q1)
    finally:
        boundary._TICKET.reset(token)
    eng.gate.clear()
    eng.answer(STATE, Q1)  # its warm-up starts on the registrar's thread and is held
    assert eng.warm_started.wait(10)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (eng.registrar.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.5)  # waits for the warm-up in progress
    eng.gate.set()
    assert closed.wait(10)
    n = len(eng.sent)
    ticket.release()  # due after close: never sent
    assert eng.registrar.facts()["dropped"] == 1 and eng.registrar.facts()["registered"] == 1
    assert len(eng.sent) == n and not eng.registrar._thread
