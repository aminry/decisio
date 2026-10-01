# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fresh-state serving flags (decisio.serve.vllm_engine, --pad-policy and --multi-question), on the request builder
with the served model's tokenizer and a recording engine; no model.

  F1  the defaults are the served default, token for token: front padding on every request, a warm-up of P+1
      tokens before the questions of a multi-question request
  F2  --pad-policy shared leaves a single-question request unpadded and pads a multi-question request exactly as
      the default does
  F3  --multi-question batch sends the questions of a multi-question request in one engine call, with no warm-up,
      the rows unchanged
  F4  the engine reports where its time went (prepare, warm-up, questions, readout)

    uv run pytest -q tests/unit/test_fresh_state_flags.py
"""

import os

import numpy as np
import pytest

from decisio.serve import vllm_engine as sv

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
STATE = "Ticket 8842. Customer writes: our API integration started returning 500 errors on every request. " * 3
ONE = [{"kind": "noul", "instructions": "Is the customer reporting an outage?"}]
FOUR = ONE + [
    {"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales", "account"]},
    {"kind": "noul", "instructions": "Is this urgent?"},
    {"kind": "score", "instructions": "How urgent is this?", "options": ["low", "medium", "high", "critical"]},
]


class Recording(sv.LettersEngine):
    """The engine's request path with a tokenizer and a scorer that records what it would send."""

    def __init__(self, tok, pad_policy=None, multi_question=None):
        self.tok, self.pad_unit, self.pad_token, self.pad_where = tok, 1056, sv.PAD_TOKEN, "front"
        self.mode, self.match_unit, self.adapters = "separate", 1056, {}
        import threading

        self._lock = threading.Lock()
        if pad_policy:
            self.pad_policy = pad_policy
        if multi_question:
            self.multi_question = multi_question
        self.sent = []

    def score_prompts(self, rows, adapter=None, warm=(), **kw):
        self.sent.append({"rows": [ids for ids, _ in rows], "warm": list(warm)})
        return [np.ones(len(lab)) / len(lab) for _, lab in rows], {
            "warm_ms": 0.0,
            "questions_ms": 1.0,
            "readout_ms": 0.1,
        }


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def sent(tok, questions, **flags):
    eng = Recording(tok, **flags)
    _, info = eng.answer(STATE, questions)
    return eng.sent[0], info


def test_f1_defaults_are_the_served_default(tok):
    base = Recording(tok)
    rows, P = base._prepare_separate(STATE, ONE)
    assert P % 1056 == 0 and rows[0][0][: P - 300].count(sv.PAD_TOKEN) > 0  # front-padded to the block
    one, _ = sent(tok, ONE)
    four, _ = sent(tok, FOUR)
    assert one["warm"] == [] and len(one["rows"][0]) > 1056
    assert len(four["warm"]) == 1 and len(four["warm"][0]) == P + 1 and len(four["rows"]) == 4


def test_f2_pad_policy_shared(tok):
    default_one, _ = sent(tok, ONE)
    shared_one, _ = sent(tok, ONE, pad_policy="shared")
    assert len(shared_one["rows"][0]) < 1056 and shared_one["rows"][0][0] != sv.PAD_TOKEN
    assert default_one["rows"][0][-len(shared_one["rows"][0]) :] == shared_one["rows"][0]  # the same prompt, unpadded
    assert sent(tok, FOUR, pad_policy="shared")[0] == sent(tok, FOUR)[0]  # multi-question: unchanged


def test_f3_multi_question_batch(tok):
    warm, _ = sent(tok, FOUR)
    batch, _ = sent(tok, FOUR, multi_question="batch")
    assert batch["warm"] == [] and batch["rows"] == warm["rows"]
    assert sent(tok, ONE, multi_question="batch")[0] == sent(tok, ONE)[0]  # single question: unchanged


def test_f4_stages_reported(tok):
    _, info = sent(tok, FOUR)
    for k in ("prepare_ms", "warm_ms", "questions_ms", "readout_ms", "server_ms"):
        assert isinstance(info[k], float) and info[k] >= 0, k
