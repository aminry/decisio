# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The head's hidden state from the serving engine (decisio's hidden-readout model class and the server's recovery
path; docs/design/hidden-state-readout.md), without a GPU: the vLLM stand-in (stub_vllm.py) and stored intent
readouts (tests/data/intent_readouts: the hidden states an intent head was fitted and served on, BANKING77 with 10
examples per intent).

  H1  the model class: the reserved columns hold [0, h], every other logit is vLLM's own; the reserved ids are checked
      against the vocabulary and the label tokens
  H2  recovery: from the float32 log-probabilities vLLM returns for the reserved ids, h comes back within 1e-4 on every
      stored vector (the float32 rounding at the normaliser's magnitude), and a request that does not allow the reserved
      ids gets vLLM's own masked log-probabilities, bit for bit
  H3  exact arithmetic, end to end through `SingleEngineHidden` on an emulated engine (chunk requests one at a time):
      the label log-probabilities are the engine's own arithmetic (bf16 logits) on the recovered state; the readout,
      fitted with `fit_intent_head` and served with `apply_intent_head`, equals the reference functions on that same
      readout bit for bit; against the same arithmetic on the exact stored states the head picks the same option on
      every evaluated item and its probabilities agree within 1e-4

    uv run pytest -q tests/unit/test_hidden_readout.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402

from decisio.vllm_plugin.hidden import (  # noqa: E402
    MAX_ALLOWED,
    check_reserved,
    recover_hidden_chunks,
    reserved_chunks,
    reserved_ids,
)

READOUTS = Path(__file__).resolve().parents[1] / "data" / "intent_readouts"
START, D = 100_000, 2048


def stored(task="banking77"):
    """(example h, example lp, example labels, evaluation h, evaluation lp) of one intent set."""
    ex, ev = np.load(READOUTS / f"{task}_examples.npz"), np.load(READOUTS / f"{task}_eval.npz")
    return ex["h"], ex["lp"], ex["y"], ev["h"], ev["lp"]


def test_h1_model_class_writes_only_the_reserved_columns(monkeypatch):
    import torch
    monkeypatch.setenv("DECISIO_HIDDEN_READOUT_START", "20")
    with stub_vllm("0.30.0", vocab=64, hidden=8) as registry:
        import decisio.vllm_plugin as p
        assert p.register()
        cls = registry.resolve(p.HIDDEN_READOUT)
        assert issubclass(cls, registry.resolve(p.TEXT_ONLY))             # it loads the official checkpoint too
        m = cls(vllm_config=None)
        h = torch.randn(5, 8)
        base = sys.modules["vllm.model_executor.models.qwen3_5"].Qwen3_5MoeForCausalLM.compute_logits(m, h)
        for fn in (m.compute_logits, m.compute_logits_local):
            z = fn(h)
            ids = reserved_ids(8)
            assert ids == list(range(20, 29))
            assert torch.equal(z[:, 20], torch.zeros(5)) and torch.equal(z[:, 21:29], h)
            other = [i for i in range(64) if i not in ids]
            assert torch.equal(z[:, other], base[:, other])
    assert check_reserved(8, 64, [1, 2, 3], start=20) == list(range(20, 29))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        check_reserved(8, 25, [1], start=20)
    with pytest.raises(ValueError, match="label tokens"):
        check_reserved(8, 64, [22], start=20)


def masked_logprobs(logits, allowed):
    """vLLM's processed log-probabilities of a one-token request: float32 log-softmax over the allowed ids."""
    import torch
    return torch.log_softmax(logits[allowed].float(), -1)


def test_h2_recovery_on_the_stored_hidden_states():
    import torch

    from decisio.serve.hidden_engine import reserved_logprobs
    ex_h, _, _, ev_h, _ = stored()
    H = np.concatenate([ex_h, ev_h])
    assert H.shape[1] == D and len(H) == 770 + 150
    worst = 0.0
    for h in H:
        back = recover_hidden_chunks(reserved_logprobs(h))
        worst = max(worst, float(np.abs(back - h.astype(np.float64)).max()))
    assert worst < 1e-4, worst
    # vLLM's sampler allows at most 1,024 ids per request: 2,048 dimensions are read in 3 chunks, each with the
    # reference
    chunks = reserved_chunks(D, START)
    assert [len(c) for c in chunks] == [1024, 1024, 3] and all(len(c) <= MAX_ALLOWED and c[0] == START for c in chunks)
    assert sorted({i for c in chunks for i in c}) == reserved_ids(D, START)
    # the emulation is torch's float32 log-softmax of each chunk's columns of [0, h]
    full = torch.cat([torch.zeros(1), torch.from_numpy(H[0])])
    for c, got in zip(reserved_chunks(D, 0), reserved_logprobs(H[0])):
        assert np.allclose(got, torch.log_softmax(full[c], -1).numpy(), atol=2e-4, rtol=0)
    # a request without the reserved ids: the class changes nothing it can see
    with stub_vllm("0.30.0", vocab=64, hidden=8) as registry:
        import decisio.vllm_plugin as p
        p.register()
        import os
        os.environ["DECISIO_HIDDEN_READOUT_START"] = "20"
        try:
            m = registry.resolve(p.HIDDEN_READOUT)(vllm_config=None)
            h = torch.randn(1, 8)
            base = sys.modules["vllm.model_executor.models.qwen3_5"].Qwen3_5MoeForCausalLM.compute_logits(m, h)[0]
            labels = [3, 7, 40, 41]
            assert torch.equal(masked_logprobs(m.compute_logits(h)[0], labels), masked_logprobs(base, labels))
        finally:
            del os.environ["DECISIO_HIDDEN_READOUT_START"]


class EmulatedEngine:
    """The slice of LettersEngine the head path uses, over stored hidden states: a prompt is one token, the index of
    its stored vector; `llm.generate` runs decisio's model class and returns vLLM's masked float32 log-probabilities."""

    def __init__(self, model, H, K):
        import threading
        self.model, self.H, self.K = model, H, K
        self.tok, self._lock = None, threading.Lock()
        self.llm = types.SimpleNamespace(generate=self.generate)

    def _prepare_separate(self, state, questions):
        return [([int(q["i"])], list(range(self.K))) for q in questions], 1

    def generate(self, prompts, sps, use_tqdm=False):
        import torch
        outs = []
        assert len(sps) == len(prompts)
        for pr, sp in zip(prompts, sps):
            h = torch.from_numpy(self.H[pr.prompt_token_ids[0]])[None]
            if len(sp.allowed_token_ids) > MAX_ALLOWED:                     # vLLM 0.30.0 kills the engine here
                raise ValueError(f"Too many allowed token IDs: {len(sp.allowed_token_ids)}. The max size is 1024.")
            lp = masked_logprobs(self.model.compute_logits(h)[0], sp.allowed_token_ids)
            assert sp.logprobs == len(sp.allowed_token_ids) and sp.max_tokens == 1 and sp.detokenize is False
            row = {t: types.SimpleNamespace(logprob=float(v)) for t, v in zip(sp.allowed_token_ids, lp)}
            outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[row])]))
        return outs


def test_h3_exact_arithmetic_on_the_stored_readouts(monkeypatch):
    import torch

    from decisio.readout import intent_head
    from decisio.serve import hidden_engine
    ex_h, ex_lp, y, ev_h, ev_lp = stored()
    K = ex_lp.shape[1]
    H = np.concatenate([ex_h, ev_h]).astype(np.float32)
    # stand-in label rows (no checkpoint here): the minimum-norm W with H @ W^T = stored log-scores
    W = np.linalg.lstsq(H.astype(np.float64), np.concatenate([ex_lp, ev_lp]), rcond=None)[0].T          # K x d
    assert np.abs(H.astype(np.float64) @ W.T - np.concatenate([ex_lp, ev_lp])).max() < 1e-6
    monkeypatch.setattr(hidden_engine.HiddenEngine, "_load_lm_head", staticmethod(lambda model: torch.from_numpy(
        np.concatenate([W, np.zeros((START + D + 1 - K, D))]))))            # label token ids are 0..K-1 here
    monkeypatch.setattr("decisio.readout.letters.label_token_ids", lambda tok, labels: list(range(K)))
    monkeypatch.setattr("decisio.readout.letters.letter_labels", lambda tok, k: [])
    with stub_vllm("0.30.0", vocab=START + D + 1, hidden=D, zero_head=True) as registry:
        import decisio.vllm_plugin as p
        p.register()
        engine = EmulatedEngine(registry.resolve(p.HIDDEN_READOUT)(vllm_config=None), H, K)
        single = hidden_engine.SingleEngineHidden(engine, "unused")
        assert single.reserved == reserved_ids(D, START)
        out = single.readout(None, [{"i": i} for i in range(len(H))])
    lp = np.array([a for a, _ in out])
    h = np.array([b for _, b in out])
    n = len(ex_h)
    assert h.dtype == np.float32 and np.abs(h - H).max() < 1e-4                 # h, within float32 rounding

    def bf16_lp(hh):                                                           # the engine's own label arithmetic
        z = torch.from_numpy(hh.astype(np.float64) @ W.T).float().to(torch.bfloat16).double().numpy()
        return z - np.logaddexp.reduce(z, 1)[:, None]
    assert np.abs(lp - bf16_lp(H)).max() < 1e-4                               # the same forward's lp (bf16 logits)
    options = [f"option_{k}" for k in range(K)]
    # fit on the single-engine readout; serve; the reference functions on the same readout, bit for bit
    rec = intent_head.fit_intent_head(list(lp[:n]), list(h[:n]), y, options)
    assert rec["applied"]
    served = np.array([intent_head.apply_intent_head(a, b, rec, options) for a, b in out[n:]])
    again = intent_head.fit_intent_head(list(lp[:n]), list(h[:n]), y, options)
    assert np.array_equal(again["A"], rec["A"]) and np.array_equal(again["c"], rec["c"])
    ref = np.array([intent_head.apply_intent_head(lp[n + i], h[n + i], again, options) for i in range(len(ev_h))])
    assert np.array_equal(served, ref)
    # against the same arithmetic on the exact stored states: the recovery changes no option, probabilities within 1e-4
    ex_lpb, ev_lpb = bf16_lp(ex_h), bf16_lp(ev_h)
    rec2 = intent_head.fit_intent_head(list(ex_lpb), list(ex_h), y, options)
    two = np.array([intent_head.apply_intent_head(ev_lpb[i], ev_h[i], rec2, options) for i in range(len(ev_h))])
    assert rec2["lambda"] == rec["lambda"]
    assert (served.argmax(1) == two.argmax(1)).all() and np.abs(served - two).max() < 1e-4


@pytest.mark.slow
def test_h4_served_on_the_cpu_stand_in():
    """The whole route on the CPU stand-in (`--backend hf --model-class hidden-readout`): a head registered and served
    through /v1/tasks and /v1/systemone on the recovered hidden state equals the reference arithmetic on the request's
    own readout, and the recovered state is the direct one within float32 rounding."""
    from fastapi.testclient import TestClient
    from test_tasks import CRIT10, EXAMPLES10, MODEL, wire

    from decisio.readout import intent_head
    from decisio.serve.hf_letters import HFLettersEngine
    from decisio.serve.hidden_engine import HFHiddenEngine, HFReservedHiddenEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app
    eng = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    hid = HFReservedHiddenEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), hidden_engine=hid, debug_readout=True)
    client = TestClient(make_app(eng, so))
    r = client.post("/v1/tasks", json={"id": "intent", "examples": EXAMPLES10}, headers={"x-decisio-debug": "hidden"})
    assert r.status_code == 200, r.text
    task = so.task_store.lookup(r.json()["key"])
    if not task["head"]["applied"]:
        pytest.skip("cross-validation declined the head on the CPU stand-in's 50 examples")
    body = wire("please top up my balance", CRIT10)
    a = client.post("/v1/systemone", json=body, headers={"x-decisio-debug": "readout"}).json()
    d = a["decisio_debug"]["q1"]
    assert d["path"] == "head"
    want = intent_head.apply_intent_head(np.array(d["hidden_lp"]), np.array(d["h"], dtype=np.float32), task["head"],
                                         list(CRIT10))
    assert list(a["answers"]["q1"]["probabilities"].values()) == want.tolist()
    direct = HFHiddenEngine.hidden_rows(hid, [hid._prepare_separate(body["state"], [
        {"kind": "choice", "instructions": "Classify the intent of the user's message.",
         "options": list(CRIT10.values())}])[0][0][0]])[0]
    assert np.abs(np.array(d["h"]) - direct).max() < 1e-4
