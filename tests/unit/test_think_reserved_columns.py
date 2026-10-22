# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Thinking must not sample decisio's reserved readout columns.

The served default engine runs decisio's hidden-readout class, which overwrites the logits of d + 1 reserved token ids
with [0, h_1..h_d] on every row. A read allows only its labels and never sees them; greedy writing of reasoning may
sample any token, the hidden-state values beat the real logits, and it wrote reserved ids for ever (finish "length",
text of unrelated words) from 0.21.0 until this fix. A probe of what think() asked vLLM and got back found it.

R1  the columns: on, a greedy argmax over a realistic logits row lands in the reserved range; off, the row is as
    the model computed it; nesting and an exception leave them on afterwards
R2  think() writes with the columns off, restores them, and a read after it sees them on
R3  the tripwire: reserved ids among the written tokens raise instead of returning a read after junk
R4  an engine whose class is hidden-readout and whose engine runs in another process cannot think, and says why; every
    other engine is unaffected

  uv run pytest -q tests/unit/test_think_reserved_columns.py
"""

import sys
import threading
import types

import numpy as np
import pytest
import torch
from test_think import FakeVLLM

from decisio.serve import think as T
from decisio.serve import vllm_engine
from decisio.vllm_plugin import hidden

START, D = 170_000, 64


def logits_row(seed=0, vocab=170_000 + 4096):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, vocab, generator=g) * 3.0).to(torch.bfloat16)  # real logits: a few units around zero


def test_r1_with_the_columns_on_greedy_sampling_picks_reserved_ids_and_off_it_does_not():
    h = torch.randn(1, D) * 40  # a final-norm hidden state: tens, far above a real logit
    on = hidden.write_hidden_columns(logits_row().clone(), h, start=START)
    assert START <= int(on.argmax()) <= START + D  # the bug: every greedy step picks a reserved id
    base = logits_row()
    with hidden.columns_off():
        off = hidden.write_hidden_columns(base.clone(), h, start=START)
    assert torch.equal(off, base) and int(off.argmax()) < START  # as the model computed them
    assert hidden.columns_on()


def test_r1_nesting_and_an_exception_leave_the_columns_on():
    assert hidden.columns_on()
    with hidden.columns_off():
        with hidden.columns_off():
            assert not hidden.columns_on()
        assert not hidden.columns_on()
    assert hidden.columns_on()
    with pytest.raises(RuntimeError):
        with hidden.columns_off():
            raise RuntimeError("engine died")
    assert hidden.columns_on()


class Tok:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return {T.PREFIX: [50, 51], T.SUFFIX: [60]}[text]

    def get_vocab(self):
        return {"<turn|>": 105}

    def convert_tokens_to_ids(self, name):
        return 105


def make_engine(monkeypatch, written=None, hidden_class=True, engine_process="in"):
    fake = FakeVLLM()
    seen = {}
    inner = fake.generate

    def generate(prompts, sp, lora_request=None, use_tqdm=False):
        writing = (sp[0] if isinstance(sp, list) else sp).max_tokens > 1
        if writing:
            seen["columns_on_while_writing"] = hidden.columns_on()
        out = inner(prompts, sp, lora_request=lora_request, use_tqdm=use_tqdm)
        if writing and written is not None:
            for o in out:
                o.outputs[0].token_ids = list(written)
        return out

    fake.generate = generate
    mod = types.ModuleType("vllm")
    mod.SamplingParams = FakeVLLM.SamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = FakeVLLM.TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    eng = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    eng.llm, eng.tok, eng.adapters, eng._lock = fake, Tok(), {}, threading.Lock()
    eng.hidden_class, eng.engine_process = hidden_class, engine_process
    eng._prepare_separate = lambda state, questions: ([([1, 2, 3], ((11,), (12,), (13,)))], 3)
    return eng, seen


def test_r2_think_writes_with_the_columns_off_and_restores_them(monkeypatch):
    eng, seen = make_engine(monkeypatch)
    assert eng.can_think()
    lps, counts, _ = eng.think("s", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)
    assert seen["columns_on_while_writing"] is False  # the call that writes ran with the reserved columns off
    assert hidden.columns_on() and counts == [3] and np.exp(lps[0]).argmax() == 2


def test_r2_columns_are_restored_when_the_engine_fails_while_writing(monkeypatch):
    eng, _ = make_engine(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("engine died")

    eng.llm.generate = boom
    with pytest.raises(RuntimeError):
        eng.think("s", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)
    assert hidden.columns_on()


def test_r3_reserved_ids_among_the_written_tokens_raise_instead_of_returning_a_read(monkeypatch):
    start = hidden.readout_start()
    eng, _ = make_engine(monkeypatch, written=[7, start + 801, start + 12, 105])
    with pytest.raises(RuntimeError, match="reserved readout token ids"):
        eng.think("s", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)
    assert hidden.columns_on()
    plain, _ = make_engine(monkeypatch, written=[7, start + 801], hidden_class=False)  # no such class: nothing to guard
    assert plain.think("s", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)[1] == [2]


def test_r4_a_hidden_readout_engine_in_another_process_cannot_think_and_says_why(monkeypatch):
    eng, _ = make_engine(monkeypatch, engine_process="separate")
    assert not eng.can_think() and "--engine-process in" in eng.think_blocked()
    with pytest.raises(ValueError, match="--engine-process in"):
        eng.think("s", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)
    other, _ = make_engine(monkeypatch, hidden_class=False, engine_process="separate")  # a text-only class: unaffected
    assert other.can_think() and other.think_blocked() is None
    assert vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine).think_blocked() == "think needs the vLLM engine"
