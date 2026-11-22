# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The hidden-state tap (a study prototype; decisio.vllm_plugin.tap): the final-norm state and chosen middle layers at
the answer position, in the same request as the letter probabilities.

H1  DECISIO_TAP_LAYERS parses to sorted unique layers; a stored row splits into the final state and its layers
H2  the model class (every base's hidden-readout class): its layers come from the engine's additional_config (the
    compiled graph's cache key), not the environment; forward concatenates [final | layers] from vLLM's auxiliary
    states; compute_logits computes the logits from the contiguous final part, equal to the stock class's on the same
    final state with the reserved columns written, and keeps the rows for the processor
H3  the logits processor stores the rows of the requests that carry a key, by their slot, never changes the logits,
    and declines (and forgets) a request without a key
H4  score_prompts gives each row's extra_args to its request, and none when not asked
H5  the reader: one score_prompts call for the question rows, each with its own key; the label log-probabilities
    are that call's, h and the layers are the stored rows'; a missing row is an error naming the setup

  uv run pytest -q tests/unit/test_hidden_tap.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402

D, V, LAYERS = 8, 64, (2, 5)


def test_h1_layers_and_split():
    from decisio.vllm_plugin import tap

    assert tap.tap_layers({"DECISIO_TAP_LAYERS": "36, 12,24,12"}) == (12, 24, 36)
    assert tap.tap_layers({}) == ()
    with pytest.raises(ValueError):
        tap.tap_layers({"DECISIO_TAP_LAYERS": "-1"})
    row = torch.arange(3 * D, dtype=torch.float32)
    h, mids = tap.split(row, D, LAYERS)
    assert np.array_equal(h, np.arange(D)) and np.array_equal(mids[5], np.arange(2 * D, 3 * D))
    with pytest.raises(ValueError, match="not 3 states"):
        tap.split(torch.zeros(2 * D), D, LAYERS)


def _cfg(layers):
    from decisio.vllm_plugin import tap

    return types.SimpleNamespace(**tap.engine_config(layers))


def test_h2_the_class_concatenates_and_computes_the_logits_from_the_final_state(monkeypatch):
    monkeypatch.setenv("DECISIO_TAP_LAYERS", "7")  # the environment is never read by the class
    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", "40")
    with stub_vllm(vocab=V, hidden=D):
        from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration as Base

        from decisio.vllm_plugin import gemma, tap

        set_layers = []
        monkeypatch.setattr(
            Base, "set_aux_hidden_state_layers", lambda self, layers: set_layers.append(layers), raising=False
        )
        g = torch.Generator().manual_seed(1)
        final, l2, l5 = (torch.randn(6, D, generator=g) for _ in range(3))
        monkeypatch.setattr(Base, "forward", lambda self, *a, **k: (final, [l2, l5]), raising=False)

        model = gemma.DecisioGemma4UnifiedHiddenReadout(vllm_config=_cfg([5, 2]))
        assert set_layers == [(2, 5)] and model.tap_layers == (2, 5)
        wide = model.forward(None, None)
        assert torch.equal(wide, torch.cat([final, l2, l5], dim=-1))

        rows = wide[[1, 4]]  # the runner's answer-position rows
        logits = model.compute_logits(rows)
        assert torch.equal(tap.take(), rows) and tap.take() is None
        reference = gemma.DecisioGemma4UnifiedHiddenReadout()  # no layers: the final state's own logits
        reference.load_state_dict(model.state_dict())
        assert torch.equal(logits, reference.compute_logits(final[[1, 4]].contiguous()))

        plain = gemma.DecisioGemma4UnifiedHiddenReadout(vllm_config=_cfg([]))  # final state only
        monkeypatch.setattr(Base, "forward", lambda self, *a, **k: final, raising=False)
        assert plain.forward() is final
        plain.load_state_dict(model.state_dict())
        assert torch.equal(plain.compute_logits(final[[1]]), reference.compute_logits(final[[1]]))
        orig_init = Base.__init__

        def with_layers(self, **k):
            orig_init(self, **k)
            self.config = types.SimpleNamespace(hidden_size=D, num_hidden_layers=4)

        monkeypatch.setattr(Base, "__init__", with_layers)
        with pytest.raises(ValueError, match="this model has 4 layers"):
            gemma.DecisioGemma4UnifiedHiddenReadout(vllm_config=_cfg([5]))
        assert gemma.DecisioGemma4UnifiedHiddenReadout().tap_layers == ()  # no config: none, whatever the env


@pytest.mark.parametrize(
    "module, base_module, base, cls",
    [
        ("gemma", "gemma4", "Gemma4ForCausalLM", "DecisioGemma4HiddenReadout"),  # 31B (bf16, FP8), E4B
        ("models", "qwen3_5", "Qwen3_5MoeForCausalLM", "DecisioQwen3_5MoeHiddenReadout"),  # Qwen3.6-35B-A3B
        ("models", "qwen3_5", "Qwen3_5ForCausalLM", "DecisioQwen3_5HiddenReadout"),  # Qwen3.5 4B
    ],
)
def test_h2_every_base_class_carries_the_tap(monkeypatch, module, base_module, base, cls):
    """H2 on the other bases' hidden-readout classes: the same tap, the same columns as without it."""
    import importlib

    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", "40")
    with stub_vllm(vocab=V, hidden=D):
        Base = getattr(importlib.import_module(f"vllm.model_executor.models.{base_module}"), base)
        from decisio.vllm_plugin import tap
        from decisio.vllm_plugin.hidden import write_hidden_columns

        Cls = getattr(importlib.import_module(f"decisio.vllm_plugin.{module}"), cls)
        assert Cls.__mro__[1] is tap.TapReadout and Cls.__name__ == cls  # the Hub profiles' class name
        set_layers = []
        monkeypatch.setattr(
            Base, "set_aux_hidden_state_layers", lambda self, layers: set_layers.append(layers), raising=False
        )
        g = torch.Generator().manual_seed(2)
        final, l2, l5 = (torch.randn(6, D, generator=g) for _ in range(3))
        monkeypatch.setattr(Base, "forward", lambda self, *a, **k: (final, [l2, l5]), raising=False)
        model = Cls(vllm_config=_cfg([5, 2]))
        assert set_layers == [(2, 5)]
        rows = model.forward(None, None)[[1, 4]]
        logits = model.compute_logits(rows)
        assert torch.equal(tap.take(), rows)
        stock = Base.compute_logits(model, final[[1, 4]].contiguous())  # the stock class's logits, columns written
        assert torch.equal(logits, write_hidden_columns(stock, final[[1, 4]].contiguous()))

        plain = Cls(vllm_config=_cfg([]))  # final state only: forward and logits as the class without the tap
        monkeypatch.setattr(Base, "forward", lambda self, *a, **k: final, raising=False)
        assert plain.forward() is final and set_layers == [(2, 5)]
        plain.load_state_dict(model.state_dict())
        assert torch.equal(plain.compute_logits(final), write_hidden_columns(Base.compute_logits(plain, final), final))
        assert torch.equal(tap.take(), final) and tap.take() is None


def test_h3_the_processor_stores_keyed_rows_by_slot_and_leaves_the_logits():
    from decisio.vllm_plugin import tap

    proc = tap.HiddenTap()
    sp = lambda key=None: types.SimpleNamespace(extra_args={tap.TAP_KEY: key} if key else None)  # noqa: E731
    assert proc.add_request(7, sp("a")) is True
    assert proc.add_request(3, sp()) is False
    assert proc.add_request(9, sp("b")) is True
    rows = torch.arange(3 * 2 * D, dtype=torch.float32).reshape(3, 2 * D)
    logits = torch.randn(3, V)
    before = logits.clone()
    tap.stash(rows)
    ctx = types.SimpleNamespace(idx_mapping_np=np.array([9, 3, 7]))  # batch position -> slot
    out = proc.apply(logits, ctx)
    assert out is logits and torch.equal(out, before)
    assert torch.equal(tap.pop("b"), rows[0]) and torch.equal(tap.pop("a"), rows[2])
    assert tap.pop("a") is None and tap.pending() == 0
    # a recycled slot without a key forgets the old one; a step with no stashed rows stores nothing
    assert proc.add_request(9, sp()) is False and 9 not in proc.keys
    assert proc.apply(logits, ctx) is logits and tap.pending() == 0
    with pytest.raises(ValueError):
        tap.HiddenTap.validate_params(types.SimpleNamespace(extra_args={tap.TAP_KEY: 5}))


def test_h4_score_prompts_gives_each_row_its_extra_args(monkeypatch):
    import sys

    from decisio.serve import vllm_engine as sv

    seen = []
    vllm, inputs = types.ModuleType("vllm"), types.ModuleType("vllm.inputs")
    vllm.SamplingParams, inputs.TokensPrompt = (lambda **k: seen.append(k) or k), (lambda **k: k)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(sv, "guarded", lambda engine, fn, *a, **k: fn(*a, **k))

    def generate(prompts, sps, **kw):
        lp = {t: types.SimpleNamespace(logprob=-1.0) for t in (1, 2)}
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(logprobs=[lp])],
                num_cached_tokens=0,
                prompt_token_ids=p["prompt_token_ids"],
            )
            for p in prompts
        ]

    eng = sv.LettersEngine.__new__(sv.LettersEngine)
    eng.llm, eng.adapters = types.SimpleNamespace(generate=generate), {}
    rows = [([5, 6], [1, 2]), ([7], [1, 2])]
    eng.score_prompts(rows, extra_args=[{"decisio_tap": "k1"}, None])
    assert seen[0]["extra_args"] == {"decisio_tap": "k1"} and "extra_args" not in seen[1]
    seen.clear()
    eng.score_prompts(rows)
    assert all("extra_args" not in k for k in seen)


class _Engine:
    """A serving engine whose readout request stores each keyed row, as the processor does in a step."""

    def __init__(self, probs, H, store=True):
        import threading

        self.probs, self.H, self.store, self.calls = probs, H, store, []
        self.llm = self.tok = object()
        self._lock = threading.Lock()

    def _prepare_separate(self, state, questions):
        return [([int(q["i"])], [0, 1, 2]) for q in questions], 1

    def score_prompts(self, rows, extra_args=None, **kw):
        from decisio.vllm_plugin import tap

        self.calls.append([a[tap.TAP_KEY] for a in extra_args])
        if self.store:
            for (ids, _), a in zip(rows, extra_args):
                tap.put(a[tap.TAP_KEY], torch.from_numpy(self.H[ids[0]]))
        return [self.probs[ids[0]] for ids, _ in rows], {}


def test_h5_the_reader_reads_one_request_per_question_row():
    from decisio.serve.hidden_engine import TapHidden
    from decisio.vllm_plugin import tap

    rng = np.random.default_rng(0)
    H = rng.normal(size=(2, 3 * D)).astype(np.float32)
    probs = {0: np.array([0.7, 0.2, 0.1]), 1: np.array([0.1, 0.3, 0.6])}
    eng = _Engine(probs, H)
    reader = TapHidden(eng, layers=LAYERS)
    out = reader.readout(None, [{"i": 0}, {"i": 1}])
    assert len(eng.calls) == 1 and len(set(eng.calls[0])) == 2  # one call, a key per row
    for i, (lp, h) in enumerate(out):
        assert np.array_equal(lp, np.log(probs[i])) and np.array_equal(h, H[i, :D])
        assert np.array_equal(reader.last_layers()[i][5], H[i, 2 * D :])
    assert tap.pending() == 0 and reader.facts()["requests_per_question"] == 1
    with pytest.raises(RuntimeError, match="DecisioGemma4UnifiedHiddenReadout"):
        TapHidden(_Engine(probs, H, store=False), layers=LAYERS).readout(None, [{"i": 0}])
