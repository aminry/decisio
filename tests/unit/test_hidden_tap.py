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
H6  a state-span job (the mean of a tapped layer over the request's state tokens): the forward adds the rows at the
    span's positions of each step, padding left out; a span computed in part, or a step shared with another request,
    is an error
H7  the reader's state-span mean: the state's text at its last occurrence in the shared prefix, one skip-cache
    request up to the span's end, the mean equal to the reference within float tolerance, kept per state

  uv run pytest -q tests/unit/test_hidden_tap.py
"""

import os
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


def _rows_at(positions, slots, d=D):
    """A stand-in forward's [final | layers] rows: row t, part j is t * 10 + j + arange(d) / d."""
    pos = torch.as_tensor(positions, dtype=torch.float32)[:, None]
    return torch.cat([pos * 10 + j + torch.arange(d) / d for j in range(1 + slots)], dim=-1)


def test_h6_the_state_span_mean_is_summed_in_the_forward(monkeypatch):
    """H6 the class's forward, during a state-span job, adds the rows at the span's positions of each step (a prompt
    read in chunks), leaves padding rows out (the forward context's mask), and the job's mean is the layer's mean over
    the span; a span computed in part, or a step with another request's tokens, is an error."""
    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", "40")
    with stub_vllm(vocab=V, hidden=D):
        from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration as Base

        from decisio.vllm_plugin import gemma, tap

        monkeypatch.setattr(Base, "set_aux_hidden_state_layers", lambda self, layers: None, raising=False)
        steps = iter([np.arange(0, 6), np.arange(6, 12)])  # one request's prompt of 12 tokens in two chunks
        pad = {"n": 0}

        def forward(self, *a, positions=None, **k):
            p = positions.numpy()
            wide = _rows_at(p, 2)
            return wide[:, :D], [wide[:, D : 2 * D], wide[:, 2 * D :]]

        monkeypatch.setattr(Base, "forward", forward, raising=False)
        fc = types.ModuleType("vllm.forward_context")
        fc.get_forward_context = lambda: types.SimpleNamespace(is_padding=pad.get("mask"))
        monkeypatch.setitem(sys.modules, "vllm.forward_context", fc)
        model = gemma.DecisioGemma4UnifiedHiddenReadout(vllm_config=_cfg([5, 2]))  # layers (2, 5): slots 0 and 1

        tap.span_begin(3, 10, {5: 1})
        for p in steps:
            if p[0] == 6:  # the second step padded with two rows, at stale positions inside the span
                pad["mask"] = torch.tensor([False] * 6 + [True] * 2)
                p = np.concatenate([p, [4, 5]])
            model.forward(None, positions=torch.as_tensor(p))
        pad.pop("mask")
        means = tap.span_end()
        want = _rows_at(np.arange(3, 10), 2)[:, 2 * D :].mean(0).numpy()
        assert set(means) == {5} and np.allclose(means[5], want, atol=1e-5) and means[5].dtype == np.float32

        tap.span_begin(3, 10, {5: 1})
        model.forward(None, positions=torch.arange(0, 6))  # the read stopped before the span's end
        with pytest.raises(tap.SpanError, match="computed for 3 of its 7 tokens"):
            tap.span_end()
        tap.span_begin(3, 10, {5: 1})
        model.forward(None, positions=torch.tensor([0, 1, 2, 3, 0, 1]))  # two requests in one step
        with pytest.raises(tap.SpanError, match="another request"):
            tap.span_end()
        model.forward(None, positions=torch.arange(0, 4))  # no job: nothing is kept
        with pytest.raises(tap.SpanError, match="no state-span read"):
            tap.span_end()


class _SpanEngine(_Engine):
    """`_Engine` with a tokenizer and prompt rows built from text, whose skip-cache request runs the stand-in forward
    over its whole prompt (in two steps), as the class's forward sums it during a state-span job."""

    def __init__(self, tok, prompt, shared_text, H):
        super().__init__({0: np.array([0.5, 0.5])}, H)
        self.tok, self.ids = tok, tok(prompt, add_special_tokens=False)["input_ids"]
        self.shared = len(tok(shared_text, add_special_tokens=False)["input_ids"])
        self.span_reads, self.salts = [], []

    def _prepare_separate(self, state, questions):
        return [(self.ids, [0, 1]) for _ in questions], self.shared

    def score_prompts(self, rows, extra_args=None, skip_cache=False, cache_salt=None, **kw):
        from decisio.vllm_plugin import tap

        if extra_args is None:  # the state-span read
            assert skip_cache and len(rows) == 1
            ids = rows[0][0]
            self.span_reads.append(len(ids))
            self.salts.append(cache_salt)
            half = len(ids) // 2
            for p in (np.arange(0, half), np.arange(half, len(ids))):
                tap.span_add(_rows_at(p, 2), torch.as_tensor(p), D)
            return [np.array([0.5, 0.5])], {}
        for (ids, _), a in zip(rows, extra_args):
            tap.put(a[tap.TAP_KEY], torch.from_numpy(self.H[0]))
        return [self.probs[0] for _ in rows], {}


def test_h7_the_reader_reads_the_state_span_mean_once_per_state():
    """H7 the reader's state-span mean: the span is the state's text at its last occurrence inside the shared prefix
    (not the option that repeats it after), read in one skip-cache request up to the span's end, its mean equal to the
    reference (the layer's rows averaged over the span) within float tolerance; kept per state, so the next question
    sends none; a reader asked for a layer the tap does not read is refused."""
    from transformers import AutoTokenizer

    from decisio.readout.state_span import state_span
    from decisio.serve.hidden_engine import TapHidden

    tok = AutoTokenizer.from_pretrained(os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base"))
    state = "The parcel arrived three days late, and the box was crushed."
    head = "Read the text and answer.\n\n"
    shared = head + state + "\n\n"
    prompt = shared + "Was the delivery on time? Options: A. " + state + " B. No\nAnswer:"
    eng = _SpanEngine(tok, prompt, shared, np.zeros((1, 3 * D), dtype=np.float32))
    reader = TapHidden(eng, layers=LAYERS, state_mean_layers=(5,))
    reader.readout(state, [{"i": 0}, {"i": 0}])
    start, end = state_span(tok, eng.ids, state, within=eng.shared, occurrence="last")
    dec = lambda a, b: tok.decode(eng.ids[a:b])  # noqa: E731
    assert state in dec(start, end) and state not in dec(start + 1, end) and state not in dec(start, end - 1)
    assert end <= eng.shared and eng.span_reads == [end]  # the read stops at the span's end
    means = reader.last_state_means()
    want = _rows_at(np.arange(start, end), 2)[:, 2 * D :].mean(0).numpy()
    assert len(means) == 2 and np.allclose(means[0][5], want, atol=1e-5)
    reader.readout(state, [{"i": 0}])
    assert eng.span_reads == [end] and np.array_equal(reader.last_state_means()[0][5], means[0][5])  # kept
    assert reader.facts()["state_mean_layers"] == [5] and reader.facts()["state_mean_cache"] == "isolated"
    # the read's blocks are cached apart from every served request's (a salt), unless asked to share them
    from decisio.vllm_plugin import tap

    assert eng.salts == [tap.STATE_MEAN_SALT]
    eng2 = _SpanEngine(tok, prompt, shared, np.zeros((1, 3 * D), dtype=np.float32))
    TapHidden(eng2, layers=LAYERS, state_mean_layers=(5,), state_mean_cache="shared").readout(state, [{"i": 0}])
    assert eng2.salts == [None]
    with pytest.raises(ValueError, match="the tap reads only"):
        TapHidden(eng, layers=LAYERS, state_mean_layers=(36,))
    with pytest.raises(ValueError, match="one of"):
        TapHidden(eng, layers=LAYERS, state_mean_layers=(5,), state_mean_cache="nowhere")
