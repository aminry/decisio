# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The default context: the largest length that fits this card, up to the base's ceiling (decisio.serve.context).

A1  the selection rule: a multiple of 8,192, never above vLLM's fit, never above the ceiling, never below 32,768, and
    a refusal (ContextDoesNotFit) only when 32,768 itself does not fit
A2  the reasons /health gives: "auto: fits 81,920 of 98,304 on this card"
A3  a fake vLLM that reports a fit: the engine asks for -1, settles on the rule's length, lowers vLLM's own limit to it,
    and a length in --engine or a plain length is never touched
A4  the serving-stack fingerprint holds the context length served (not its source text or the ceiling), so a
    certificate made at one length is suspended at another

  uv run pytest -q tests/unit/test_context_auto.py
"""

import sys
import types

import pytest

from decisio.families import BASES
from decisio.serve import stack, vllm_engine
from decisio.serve.context import FLOOR, STEP, AutoContext, ContextDoesNotFit


@pytest.mark.parametrize(
    "ceiling,fit,want",
    [
        (98304, 141865, 98304),  # a 96 GB card: vLLM's fit is above the ceiling
        (98304, 103000, 98304),  # an 80 GB card with room for the ceiling
        (98304, 98304, 98304),
        (98304, 98303, 90112),
        (98304, 90000, 81920),
        (98304, 81920, 81920),
        (98304, 40000, 32768),
        (98304, 32768, 32768),
        (98304, 33000, 32768),
        (65536, 612140, 65536),
        (65536, 70000, 65536),
        (65536, 50000, 49152),
    ],
)
def test_a1_the_largest_clean_step_that_fits_up_to_the_ceiling(ceiling, fit, want):
    assert AutoContext(ceiling).choose(fit) == want


def test_a1_the_rule_holds_for_every_fit():
    for ceiling in (65536, 98304):
        auto = AutoContext(ceiling)
        for fit in range(FLOOR, 300000, 997):
            n = auto.choose(fit)
            assert n % STEP == 0 and FLOOR <= n <= ceiling and n <= fit
            assert n == ceiling or n + STEP > fit  # the largest step that fits, unless the ceiling stops it


def test_a1_below_the_measured_length_it_does_not_start():
    with pytest.raises(ContextDoesNotFit, match=r"24,096 tokens, less than the 32,768"):
        AutoContext(98304).choose(24096)
    assert issubclass(ContextDoesNotFit, ValueError)


def test_a2_the_reason():
    auto = AutoContext(98304)
    assert auto.describe(81920, 90000) == "auto: fits 81,920 of 98,304 on this card"
    assert auto.describe(98304, 141865) == "auto: the base's ceiling of 98,304 fits on this card"
    assert "(the floor)" in auto.describe(32768, 33000)


class FakeLLM:
    """vLLM's LLM as far as LettersEngine reads it: the keyword arguments, a model config that carries the fit vLLM
    found for max_model_len=-1, a block size and a tokenizer."""

    fit = 90000
    kw = None

    def __init__(self, **kw):
        FakeLLM.kw = kw
        n = FakeLLM.fit if kw["max_model_len"] == -1 else kw["max_model_len"]
        model_config = types.SimpleNamespace(max_model_len=n)
        cache_config = types.SimpleNamespace(block_size=16, prefix_match_unit=None)
        self.llm_engine = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(model_config=model_config, cache_config=cache_config)
        )

    def get_tokenizer(self):
        return object()

    def collective_rpc(self, name):
        return [None]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr(vllm_engine.LettersEngine, "_warm_up", lambda self: None)
    FakeLLM.kw = None
    return FakeLLM


def build(**kw):
    return vllm_engine.LettersEngine("stand-in", family=BASES["gemma-4-31b"], **kw)


def test_a3_vllm_is_asked_for_the_fit_and_the_rule_picks_the_length(fake_vllm, capsys):
    fake_vllm.fit = 90000
    e = build(max_model_len=AutoContext(98304))
    assert fake_vllm.kw["max_model_len"] == -1
    assert e.context_source == "auto: fits 81,920 of 98,304 on this card"
    assert e.llm.llm_engine.vllm_config.model_config.max_model_len == 81920  # a longer prompt is refused (422)
    assert "CONTEXT 81,920 tokens" in capsys.readouterr().out


def test_a3_a_card_that_holds_the_ceiling_serves_the_ceiling(fake_vllm):
    fake_vllm.fit = 141865
    e = build(max_model_len=AutoContext(98304))
    assert e.llm.llm_engine.vllm_config.model_config.max_model_len == 98304
    assert e.context_source == "auto: the base's ceiling of 98,304 fits on this card"


def test_a3_vllms_fit_is_kept_when_it_is_already_a_clean_step(fake_vllm):
    fake_vllm.fit = 65536
    e = build(max_model_len=AutoContext(98304))
    assert e.llm.llm_engine.vllm_config.model_config.max_model_len == 65536
    assert e.context_source == "auto: fits 65,536 of 98,304 on this card"


def test_a3_32768_that_does_not_fit_is_a_failed_start_as_before(fake_vllm):
    fake_vllm.fit = 24096
    with pytest.raises(ContextDoesNotFit, match="24,096"):
        build(max_model_len=AutoContext(98304))


def test_a3_a_length_in_engine_json_or_a_plain_length_is_left_alone(fake_vllm):
    e = build(max_model_len=AutoContext(98304), engine_kw={"max_model_len": 49152})
    assert fake_vllm.kw["max_model_len"] == 49152 and e.context_source == "set by --engine"
    e = build(max_model_len=65536, context_source="set by --max-model-len")
    assert fake_vllm.kw["max_model_len"] == 65536 and e.context_source == "set by --max-model-len"
    e = build()
    assert fake_vllm.kw["max_model_len"] == 32768 and e.context_source == "32,768"


def test_a4_the_context_length_is_in_the_serving_stack_fingerprint():
    assert "max_model_len" in stack.ENGINE_KEYS
    assert "max_model_len_source" not in stack.ENGINE_KEYS and "max_model_len_ceiling" not in stack.ENGINE_KEYS

    class E:
        def __init__(self, n):
            self.n = n

        def facts(self):
            return {"engine": "fake", "max_model_len": self.n, "max_model_len_source": f"source of {self.n}"}

    def fp(e):
        return stack.stack_facts(
            e, decisio_version="x", serving={}, temperatures={}, task_fingerprint=None, probe_sha256="p"
        )

    assert stack.differences(fp(E(98304)), fp(E(81920))) == ["engine.max_model_len"]
    assert stack.differences(fp(E(98304)), fp(E(98304))) == []  # the reason text is not part of it
