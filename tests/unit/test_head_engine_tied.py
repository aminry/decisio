# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The second engine for the head (--head-engine) on a base with tied embeddings (Gemma 4), whose checkpoint has no
`lm_head.weight`: the output layer's rows are the input embedding.

T1  `HiddenEngine` loads its label rows where the served base keeps them: on a tied-embedding checkpoint as
    Hugging Face writes it (no lm_head.weight), the input embedding, equal to the model's output layer; with no base
    it reads lm_head.weight alone and refuses such a checkpoint, naming it
T2  under the base's `head_label_logprobs="engine"` (Gemma) a head question's label log-probabilities are the serving
    engine's readout of the same row, read under the engine's lock, and h comes from the pooling engine; without a
    serving engine, or on a base that recomputes (Qwen), they are recomputed from h

  uv run pytest -q tests/unit/test_head_engine_tied.py
"""

import sys
import threading
import types

import numpy as np
import pytest

from decisio.families import GEMMA4, QWEN
from decisio.serve import hidden_engine as he


class _LLM:
    def __init__(self, **kw):
        self.llm_engine = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
        )

    def get_tokenizer(self):
        return object()


def _fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.__version__ = _LLM, "0.31.0"
    cfg = types.ModuleType("vllm.config")
    cfg.PoolerConfig = lambda **kw: kw
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: types.SimpleNamespace(prompt_token_ids=prompt_token_ids)
    vllm.__path__ = []  # a package, for its submodules
    for name, mod in (("vllm", vllm), ("vllm.config", cfg), ("vllm.inputs", inputs)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(he.HiddenEngine, "readout", lambda self, *a, **k: None)  # the warm-up


@pytest.fixture(scope="module")
def tied_checkpoint(tmp_path_factory):
    """A small Gemma 4 text model with tied embeddings, saved as Hugging Face saves it."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from safetensors import safe_open

    cfg = transformers.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(cfg)
    out = tmp_path_factory.mktemp("gemma4-tied")
    model.save_pretrained(out)
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        assert "lm_head.weight" not in set(f.keys())  # the case: the output layer is not in the checkpoint
    return out, model.lm_head.weight.detach().clone()


def test_t1_the_second_engine_reads_the_tied_embedding(monkeypatch, tied_checkpoint):
    import torch

    path, W = tied_checkpoint
    _fake_vllm(monkeypatch)
    eng = he.HiddenEngine(str(path), family=GEMMA4)
    assert torch.equal(eng._W, W)
    assert eng.facts()["lm_head"] == [64, 16]
    with pytest.raises(ValueError, match=r"has no lm_head\.weight \(tied embeddings\?\)"):
        he.HiddenEngine(str(path))  # no base: lm_head.weight alone, as 0.51.1 read every base


class _TextEngine:
    """The serving engine's plain readout of a row: fixed probabilities per row, and whether its lock was held."""

    def __init__(self, probs):
        self.probs, self._lock, self.locked = probs, threading.Lock(), []

    def score_prompts(self, rows):
        self.locked.append(self._lock.locked())
        return [self.probs[rows[0][0][0]]], {}


def _engine(monkeypatch, family, text_engine, H, W):
    import torch

    _fake_vllm(monkeypatch)
    monkeypatch.setattr(he.HiddenEngine, "_load_lm_head", staticmethod(lambda model, names=(), revision=None: W))
    eng = he.HiddenEngine("stand-in", family=family, text_engine=text_engine)
    monkeypatch.delattr(he.HiddenEngine, "readout")  # after the warm-up: the real readout, HiddenReadout's
    eng._prepare_separate = lambda state, questions: ([([int(q["i"])], [0, 1, 2]) for q in questions], 1)
    eng.llm = types.SimpleNamespace(
        encode=lambda prompts, **kw: [
            types.SimpleNamespace(outputs=types.SimpleNamespace(data=torch.from_numpy(H[p.prompt_token_ids[0]])))
            for p in prompts
        ]
    )
    return eng


def test_t2_gemma_label_logprobs_are_the_serving_engines(monkeypatch):
    import torch

    rng = np.random.default_rng(0)
    H = rng.normal(size=(2, 4)).astype(np.float32)
    W = torch.from_numpy(rng.normal(size=(3, 4)))
    probs = {0: [0.7, 0.2, 0.1], 1: [0.1, 0.3, 0.6]}
    text = _TextEngine(probs)
    eng = _engine(monkeypatch, GEMMA4, text, H, W)
    assert eng.label_source == "engine" and eng.facts()["label_logprobs"] == "engine"
    out = eng.readout(None, [{"i": 0}, {"i": 1}])
    for i, (lp, h) in enumerate(out):
        assert np.array_equal(lp, np.log(np.asarray(probs[i], dtype=np.float64)))
        assert np.array_equal(h, H[i])
    assert text.locked == [True, True]

    # recomputed from h (with the base's soft cap): no serving engine given, or a base that recomputes
    for family, text_engine in ((GEMMA4, None), (QWEN, _TextEngine(probs))):
        e = _engine(monkeypatch, family, text_engine, H, W)
        assert e.label_source == "recompute"
        got = e.readout(None, [{"i": 0}, {"i": 1}])
        assert all(np.array_equal(lp, e.label_lp(H[i], [0, 1, 2])) for i, (lp, _) in enumerate(got))
        assert text_engine is None or text_engine.locked == []


class _Built(Exception):
    pass


def test_t3_the_server_gives_the_second_engine_its_base_and_the_serving_engine(monkeypatch):
    """main() on the vLLM backend with --head-engine, until the second engine would be built: it is given the served
    base and the text engine (0.51.1 gave neither, and read lm_head.weight on every base)."""
    from decisio.serve import vllm_engine

    seen, text = {}, types.SimpleNamespace(pad_unit=16, facts=lambda: {})

    def second(model, **kw):
        seen.update(kw, model=model)
        raise _Built

    monkeypatch.setattr(vllm_engine, "LettersEngine", lambda *a, **k: text)
    monkeypatch.setattr(vllm_engine, "engine_kwargs", lambda args: {})
    monkeypatch.setattr(he, "HiddenEngine", second)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    argv = ["decisio", "--base", "gemma-4-12b", "--head-engine"]
    argv += ["--gpu-memory-utilization", "0.47", "--head-gpu-memory-utilization", "0.47"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Built):
        vllm_engine.main()
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    assert seen["family"] is GEMMA4 and seen["text_engine"] is text
    assert seen["model"] == GEMMA4.model and seen["revision"] == GEMMA4.revision
