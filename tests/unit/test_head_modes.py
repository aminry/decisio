# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head's hidden-state readers (`decisio.serve.vllm_engine.resolve_head_mode`, `single_engine_reader`;
docs/handoffs/tasks.md). Two-engine mode (--head-engine) is removed.

tap        on the Gemma 4 12B with an in-process engine: the head question's hidden state from its own request
           (`hidden_engine.TapHidden`, decisio.vllm_plugin.tap)
reserved   otherwise: the hidden-readout class writes h into reserved logit columns (`hidden_engine.SingleEngineHidden`)

M1  the flags select the mode: the default is the single engine on the hidden-readout class; the view and the text-only
    class serve no head; --one-engine serves none; an unknown class is refused
M2  the server's start-up builds the single engine's reader (the CPU stand-in's classes, replaced by recorders); the
    removed flags --head-engine and --head-gpu-memory-utilization are refused with the reason
M3  the two readers give the same declared choice on the stored intent readouts (tests/data/intent_readouts, BANKING77
    and CLINC150): each reads the stored hidden states its own way (the reserved columns through vLLM's masked
    log-probabilities and bf16 label logits, the tap as the forward's own rows and the engine's own letters), fits its
    head on its own example readouts and serves the evaluation items

  uv run pytest -q tests/unit/test_head_modes.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402
from test_hidden_readout import EmulatedEngine, stored  # noqa: E402

from decisio.serve.vllm_engine import resolve_head_mode  # noqa: E402


@pytest.mark.parametrize(
    "model_class,want",
    [
        (None, ("hidden-readout", "single-engine")),
        ("hidden-readout", ("hidden-readout", "single-engine")),
        ("text-only", ("text-only", None)),
        ("view", ("view", None)),
    ],
)
def test_m1_flags_select_the_mode(model_class, want):
    assert resolve_head_mode(model_class) == want


def test_m1_one_engine_and_unknown_classes():
    assert resolve_head_mode(None, one_engine=True) == (None, None)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_head_mode("multimodal")


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


def test_m2_start_up_builds_the_single_engine_reader(monkeypatch, capsys):
    so = start(monkeypatch)
    assert type(so.hidden_engine).__name__ == "HFReservedHiddenEngine"  # the single engine, the default
    assert Recorder.built == ["HFLettersEngine", "HFReservedHiddenEngine"]
    for argv in (["--model-class", "view"], ["--model-class", "text-only"]):
        assert start(monkeypatch, *argv).hidden_engine is None  # no head: calibration only
    for argv in (["--head-engine"], ["--head-gpu-memory-utilization", "0.47"]):
        with pytest.raises(SystemExit):
            start(monkeypatch, *argv)
        assert "removed with two-engine mode" in capsys.readouterr().err


START = 256  # reserved ids above the stand-in's label ids 0..K-1 (small: no 100,000-row matrix)


def single_engine_readout(monkeypatch, H, W):
    """(lp, h) per stored state through `SingleEngineHidden`: the hidden-readout class writes [0, h] into the reserved
    columns, the emulated engine returns vLLM's masked float32 log-probabilities one chunk at a time."""
    import torch

    from decisio.serve import hidden_engine

    K, D = W.shape
    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", str(START))
    monkeypatch.setattr(
        hidden_engine,
        "load_output_rows",
        lambda model, names=(), revision=None: torch.from_numpy(np.concatenate([W, np.zeros((START + D + 1 - K, D))])),
    )
    monkeypatch.setattr("decisio.readout.letters.label_token_ids", lambda tok, labels: list(range(K)))
    monkeypatch.setattr("decisio.readout.letters.letter_labels", lambda tok, k: [])
    with stub_vllm("0.31.0", vocab=START + D + 1, hidden=D, zero_head=True) as registry:
        import decisio.vllm_plugin as p

        p.register()
        engine = EmulatedEngine(registry.resolve(p.HIDDEN_READOUT)(vllm_config=None), H, K)
        single = hidden_engine.SingleEngineHidden(engine, "unused", start=START)
        return single.readout(None, [{"i": i} for i in range(len(H))])


def tap_readout(H, W):
    """(lp, h) per stored state through `TapHidden`: the engine's readout request returns the letters (softmax of the
    label logits, float64) and the tap's processor stores the forward's own row, h in float32."""
    import threading

    import torch

    from decisio.serve.hidden_engine import TapHidden
    from decisio.vllm_plugin import tap

    K = W.shape[0]

    class Engine:
        llm = tok = None
        _lock = threading.Lock()

        def _prepare_separate(self, state, questions):
            return [([int(q["i"])], list(range(K))) for q in questions], 1

        def score_prompts(self, rows, extra_args=None, **kw):
            out = []
            for (ids, _), a in zip(rows, extra_args):
                z = H[ids[0]].astype(np.float64) @ W.T
                e = np.exp(z - z.max())
                out.append(e / e.sum())
                tap.put(a[tap.TAP_KEY], torch.from_numpy(H[ids[0]]))
            return out, {"cached_tokens": [0] * len(rows)}

    return TapHidden(Engine(), layers=()).readout(None, [{"i": i} for i in range(len(H))])


@pytest.mark.parametrize("task", ["banking77", "clinc150"])
@pytest.mark.slow
def test_m3_both_readers_declare_the_same_choice(monkeypatch, task):
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
        ("tap", tap_readout(H, W)),
    ):
        lp = [a for a, _ in readout]
        h = [b for _, b in readout]
        rec = intent_head.fit_intent_head(lp[:n], h[:n], y, options)
        assert rec["applied"], (mode, rec["reason"])
        heads[mode] = rec
        served[mode] = np.array([intent_head.apply_intent_head(lp[i], h[i], rec, options) for i in range(n, len(H))])
    assert heads["single-engine"]["lambda"] == heads["tap"]["lambda"]
    choices = {m: [top_index(options, p) for p in served[m]] for m in served}
    assert choices["single-engine"] == choices["tap"]  # every evaluation item
    assert np.abs(served["single-engine"] - served["tap"]).max() < 1e-2
