# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""No state-boundary warm-up when vLLM's prefix cache is off (vllm_engine.resolve_register_boundary, send_warm,
boundary_needs_cache); no model.

A warm-up prefills a state so a later question reads it from the prefix cache. With the cache turned off by
`--engine '{"enable_prefix_caching": false}'` it can never be reused: on the 12B it cost about 0.8 times a question's
own read on every new state, and the hit check, seeing no cached tokens, forgot the state, so every other repeat paid
it again (a station read of 214 rows: 128 ms first, 65 ms repeated, no cached tokens counted).

N1  the default resolves to off when the engine has no prefix cache, on both Gemma bases; with the cache on it is the
    base's own; a flag given stands (the start refuses the contradictory ones, N2)
N2  --register-boundary after or before with --engine turning the cache off is refused before the weights load; off,
    no flag, the cache on or not named in --engine are served
N3  send_warm sends nothing and takes no time without a prefix cache, whatever the path that calls it
N4  answers unchanged: with the cache off the question rows sent are byte-identical to those with registration on,
    request after request, and so are the probabilities; only the warm-ups, a separate engine call, are gone

    uv run pytest -q tests/unit/test_boundary_without_cache.py
"""

import json

import numpy as np
import pytest
from test_state_boundary import OTHER, Q1, Q2, STATE, Recording, tok  # noqa: F401  (tok: the shared fixture)

from decisio.families import BASES
from decisio.serve import vllm_engine as sv

GEMMA = ["gemma-4-12b", "gemma-4-31b"]


@pytest.mark.parametrize("base", GEMMA)
def test_n1_the_default_is_off_without_a_prefix_cache(base):
    fam = BASES[base]
    assert sv.resolve_register_boundary(fam, None, prefix_caching=False) == "off"
    assert sv.resolve_register_boundary(fam, None, prefix_caching=True) == fam.register_boundary == "before"
    assert sv.resolve_register_boundary(fam, None) == "before"  # an engine that does not say: as before
    for choice in ("after", "before", "off"):
        assert sv.resolve_register_boundary(fam, choice, prefix_caching=False) == choice


def test_n2_a_registration_asked_for_without_a_cache_is_refused():
    off = json.dumps({"enable_prefix_caching": False, "max_num_batched_tokens": 16384})
    for choice in ("after", "before"):
        why = sv.boundary_needs_cache(choice, off)
        assert why and f"--register-boundary {choice}" in why and "enable_prefix_caching" in why
    assert sv.boundary_needs_cache("off", off) is None
    assert sv.boundary_needs_cache(None, off) is None  # the default resolves to off once the engine is built
    for engine in ("{}", json.dumps({"enable_prefix_caching": True}), None):
        assert sv.boundary_needs_cache("before", engine) is None


class _NoGenerate:
    def generate(self, *a, **k):
        raise AssertionError("a warm-up was sent to an engine without a prefix cache")


def test_n3_send_warm_sends_nothing_without_a_prefix_cache(monkeypatch):
    eng = sv.LettersEngine.__new__(sv.LettersEngine)
    eng.llm, eng.adapters, eng.prefix_caching = _NoGenerate(), {}, False
    assert eng.send_warm([[1, 2, 3], [4, 5, 6, 7]]) == 0.0
    # with the cache on the same call reaches the engine (vLLM stubbed; the engine-death guard needs a live engine)
    import sys
    import types

    vllm, inputs = types.ModuleType("vllm"), types.ModuleType("vllm.inputs")
    vllm.SamplingParams, inputs.TokensPrompt = (lambda **k: k), (lambda **k: k)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(sv, "guarded", lambda engine, fn, *a, **k: fn(*a, **k))
    eng.prefix_caching = True
    with pytest.raises(AssertionError, match="a warm-up was sent"):
        eng.send_warm([[1, 2, 3]])


class _Scored(Recording):
    """The request path's rows scored by a fixed function of their tokens, so a changed row changes its answer."""

    def score_prompts(self, rows, adapter=None, warm=(), **kw):
        self.sent.append({"rows": [ids for ids, _ in rows], "warm": list(warm)})
        out = []
        for ids, lab in rows:
            x = np.cos(np.arange(1, len(lab) + 1) * (sum(ids) % 997 + 1))
            out.append(np.exp(x) / np.exp(x).sum())
        cached = [0] * len(rows)  # the cache is off: vLLM reports no cached tokens
        return out, {"cached_tokens": cached}


@pytest.mark.parametrize("base", GEMMA)
def test_n4_answers_unchanged_without_the_warm_up(tok, base):  # noqa: F811
    traffic = [(STATE, Q1), (STATE, Q2), (STATE, Q1), (OTHER, Q1), (STATE, Q1), (OTHER, Q2), (STATE, Q1 + Q2)]
    before = _Scored(tok, base, register="before")
    off = _Scored(tok, base, register=sv.resolve_register_boundary(BASES[base], None, prefix_caching=False))
    assert off.register_boundary == "off"
    for state, questions in traffic:
        a, _ = before.answer_many([(state, questions)])
        b, _ = off.answer_many([(state, questions)])
        assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    rows = lambda eng: [r for call in eng.sent for r in call["rows"]]  # noqa: E731
    assert rows(before) == rows(off)  # every question row byte-identical, in the same order
    assert any(call["warm"] for call in before.sent)  # registration on sent warm-ups ...
    single = [c for c, (_, q) in zip(off.sent, traffic) if len(q) == 1]
    assert single and not any(call["warm"] for call in single)  # ... the single questions without a cache send none
