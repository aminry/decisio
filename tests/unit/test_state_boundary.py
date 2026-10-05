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

The GPU tier checks the effect on a card (tests/gpu/test_second_question_cached.py): a second, different question on a
3,000-token state reads it from the cache, and revisited states still do with the cache filled past its pool.

    uv run pytest -q tests/unit/test_state_boundary.py
"""

import os
import threading

import numpy as np
import pytest

from decisio.families import BASES
from decisio.serve import vllm_engine as sv

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
STATE = "Ticket 8842. Customer writes: our API integration started returning 500 errors on every request. " * 3
OTHER = "Ticket 9001. Customer writes: the invoice for September lists a seat we cancelled in August. " * 3
Q1 = [{"kind": "noul", "instructions": "Is the customer reporting an outage?"}]
Q2 = [{"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales", "account"]}]


class Recording(sv.LettersEngine):
    """The request path with a tokenizer, a base's profile and a scorer that records what it would send; `cached` is
    what the engine reports as read from the cache for each question row (all of it by default)."""

    def __init__(self, tok, base, hit=32, multi_question=None):
        fam = BASES[base]
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
