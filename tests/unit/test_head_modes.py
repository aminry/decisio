# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head's two modes (`decisio.serve.vllm_engine.resolve_head_mode`; docs/handoffs/tasks.md).

single-engine   the default: the text engine runs decisio's hidden-readout class and the head's hidden state is read
                from it (`hidden_engine.SingleEngineHidden`)
second-engine   --head-engine: a second copy of the model in vLLM's pooling mode reads it
                (`hidden_engine.HiddenEngine`)

M1  the flags select the mode: the default is single-engine on the hidden-readout class; --head-engine selects the
    second engine and the text-only class; the view and the text-only class alone serve no head; contradictory
    flags are refused
M2  the server's start-up builds the selected mode's engine (the CPU stand-in's classes, replaced by recorders)
M3  both modes give the same declared choice on the stored intent readouts (tests/data/intent_readouts, BANKING77 and
    CLINC150): each mode's own code reads the stored hidden states (the single engine through vLLM's masked
    log-probabilities of the reserved columns and bf16 label logits, the second engine through a pooled h and
    float64 label logits), fits its head on its own example readouts and serves the evaluation items

  uv run pytest -q tests/unit/test_head_modes.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402
from test_hidden_readout import EmulatedEngine, stored  # noqa: E402

from decisio.serve.vllm_engine import resolve_head_mode  # noqa: E402


@pytest.mark.parametrize(
    "model_class,head_engine,want",
    [
        (None, False, ("hidden-readout", "single-engine")),
        (None, True, ("text-only", "second-engine")),
        ("text-only", True, ("text-only", "second-engine")),
        ("view", True, ("view", "second-engine")),
        ("text-only", False, ("text-only", None)),
        ("view", False, ("view", None)),
        ("hidden-readout", False, ("hidden-readout", "single-engine")),
    ],
)
def test_m1_flags_select_the_mode(model_class, head_engine, want):
    assert resolve_head_mode(model_class, head_engine) == want


def test_m1_contradictions_are_refused():
    with pytest.raises(ValueError, match="choose one"):
        resolve_head_mode("hidden-readout", True)
    with pytest.raises(ValueError, match="one-engine"):
        resolve_head_mode(None, True, one_engine=True)
    assert resolve_head_mode(None, False, one_engine=True) == (None, None)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_head_mode("multimodal", False)


class Recorder:
    """Stands in for the CPU stand-in's engines: records how it was built, builds identical token rows."""

    built: list = []

    def __init__(self, model, pad_to=None, pad_where="front", **kw):
        self.model, self.pad_unit, self.pad_where, self.adapters = model, 64, pad_where, {}
        Recorder.built.append(type(self).__name__)

    def facts(self):
        return {"engine": type(self).__name__}

    def _prepare_separate(self, state, questions):
        return [([1, 2, 3], [4, 5]) for _ in questions], 3


def start(monkeypatch, *argv):
    """Run the server's main() on the CPU stand-in up to the point it would listen; return the SystemOne it serves."""
    import uvicorn

    import decisio.serve.hf_letters as hf
    import decisio.serve.hidden_engine as he
    from decisio.serve import vllm_engine

    classes = {
        name: type(name, (Recorder,), {}) for name in ("HFLettersEngine", "HFReservedHiddenEngine", "HFHiddenEngine")
    }
    monkeypatch.setattr(hf, "HFLettersEngine", classes["HFLettersEngine"])
    monkeypatch.setattr(he, "HFReservedHiddenEngine", classes["HFReservedHiddenEngine"])
    monkeypatch.setattr(he, "HFHiddenEngine", classes["HFHiddenEngine"])
    served = {}
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: served.update(engine=engine, so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(sys, "argv", ["decisio", "--backend", "hf", "--model", "stand-in", *argv])
    Recorder.built = []
    vllm_engine.main()
    return served["so"]


def test_m2_start_up_builds_the_selected_mode(monkeypatch):
    so = start(monkeypatch)
    assert type(so.hidden_engine).__name__ == "HFReservedHiddenEngine"  # single-engine, the default
    assert Recorder.built == ["HFLettersEngine", "HFReservedHiddenEngine"]
    so = start(monkeypatch, "--head-engine")
    assert type(so.hidden_engine).__name__ == "HFHiddenEngine"  # the second engine
    for argv in (["--model-class", "view"], ["--model-class", "text-only"]):
        assert start(monkeypatch, *argv).hidden_engine is None  # no head: calibration only
    for argv in (["--model-class", "hidden-readout", "--head-engine"], ["--head-engine", "--image-model", "x"]):
        with pytest.raises(SystemExit):
            start(monkeypatch, *argv)


START = 256  # reserved ids above the stand-in's label ids 0..K-1 (small: no 100,000-row matrix)


def single_engine_readout(monkeypatch, H, W):
    """(lp, h) per stored state through `SingleEngineHidden`: the hidden-readout class writes [0, h] into the reserved
    columns, the emulated engine returns vLLM's masked float32 log-probabilities one chunk at a time."""
    import torch

    from decisio.serve import hidden_engine

    K, D = W.shape
    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", str(START))
    monkeypatch.setattr(
        hidden_engine.HiddenEngine,
        "_load_lm_head",
        staticmethod(lambda model: torch.from_numpy(np.concatenate([W, np.zeros((START + D + 1 - K, D))]))),
    )
    monkeypatch.setattr("decisio.readout.letters.label_token_ids", lambda tok, labels: list(range(K)))
    monkeypatch.setattr("decisio.readout.letters.letter_labels", lambda tok, k: [])
    with stub_vllm("0.30.0", vocab=START + D + 1, hidden=D, zero_head=True) as registry:
        import decisio.vllm_plugin as p

        p.register()
        engine = EmulatedEngine(registry.resolve(p.HIDDEN_READOUT)(vllm_config=None), H, K)
        single = hidden_engine.SingleEngineHidden(engine, "unused", start=START)
        return single.readout(None, [{"i": i} for i in range(len(H))])


def second_engine_readout(H, W):
    """(lp, h) per stored state through `HiddenEngine`'s own methods: the pooling engine returns h in float32, the
    label log-probabilities come from the output layer's label rows in float64."""
    import threading

    import torch

    from decisio.serve.hidden_engine import HiddenEngine

    K = W.shape[0]
    eng = HiddenEngine.__new__(HiddenEngine)
    eng._W, eng._lock = torch.from_numpy(W), threading.Lock()
    eng._prepare_separate = lambda state, questions: ([([int(q["i"])], list(range(K))) for q in questions], 1)

    def pooled(prompts, **kw):
        return [
            types.SimpleNamespace(outputs=types.SimpleNamespace(data=torch.from_numpy(H[pr.prompt_token_ids[0]])))
            for pr in prompts
        ]

    eng.llm = types.SimpleNamespace(encode=pooled)
    with stub_vllm("0.30.0"):
        return eng.readout(None, [{"i": i} for i in range(len(H))])


@pytest.mark.parametrize("task", ["banking77", "clinc150"])
def test_m3_both_modes_declare_the_same_choice(monkeypatch, task):
    from decisio.readout import intent_head
    from decisio.serve.systemone import top_index

    ex_h, ex_lp, y, ev_h, ev_lp = stored(task)
    H = np.concatenate([ex_h, ev_h]).astype(np.float32)
    # stand-in label rows (no checkpoint here): the minimum-norm W with H @ W^T = the stored log-scores
    W = np.linalg.lstsq(H.astype(np.float64), np.concatenate([ex_lp, ev_lp]), rcond=None)[0].T
    n, K = len(ex_h), W.shape[0]
    options = [f"option_{k}" for k in range(K)]
    served, heads = {}, {}
    for mode, readout in (
        ("single-engine", single_engine_readout(monkeypatch, H, W)),
        ("second-engine", second_engine_readout(H, W)),
    ):
        lp = [a for a, _ in readout]
        h = [b for _, b in readout]
        rec = intent_head.fit_intent_head(lp[:n], h[:n], y, options)
        assert rec["applied"], (mode, rec["reason"])
        heads[mode] = rec
        served[mode] = np.array([intent_head.apply_intent_head(lp[i], h[i], rec, options) for i in range(n, len(H))])
    assert heads["single-engine"]["lambda"] == heads["second-engine"]["lambda"]
    choices = {m: [top_index(options, p) for p in served[m]] for m in served}
    assert choices["single-engine"] == choices["second-engine"]  # every evaluation item
    assert np.abs(served["single-engine"] - served["second-engine"]).max() < 1e-2
