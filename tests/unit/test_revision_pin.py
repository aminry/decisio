# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The pinned revision reaches every load of a base's checkpoint, without a model or the network:

R1  --mode packed loads its tokenizer at the revision (and vLLM at `revision` and `tokenizer_revision`)
R2  the second-engine head (HiddenEngine) starts its vLLM engine and reads the output layer at the revision
R3  the hf stand-in engines (text, hidden, reserved-column) load the tokenizer and the model at the revision
R4  from the command line to each engine: resolve_base's revision is what main() hands the engines
"""

import sys
import types

import pytest

from decisio.families import FAMILIES, QWEN
from decisio.serve import hidden_engine as he
from decisio.serve import vllm_engine as sv

REV = "0123456789abcdef0123456789abcdef01234567"


class _Stop(Exception):
    pass


def _fake_vllm(monkeypatch, llm):
    """A `vllm` with just the names the engines import: LLM (built by `llm`) and config.PoolerConfig."""
    mod = types.ModuleType("vllm")
    mod.LLM = llm
    cfg = types.ModuleType("vllm.config")
    cfg.PoolerConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.config = cfg
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.config", cfg)


def _recording_tokenizer(monkeypatch, calls):
    import transformers

    class Tok:
        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

    def from_pretrained(model, **kw):
        calls.append((model, kw))
        return Tok()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(from_pretrained))


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
@pytest.mark.parametrize("revision", [REV, None])
def test_r1_packed_tokenizer_at_the_revision(fam, revision, monkeypatch):
    tok_calls, llm_kw = [], []

    def llm(**kw):
        llm_kw.append(kw)
        raise _Stop

    _fake_vllm(monkeypatch, llm)
    _recording_tokenizer(monkeypatch, tok_calls)
    monkeypatch.setattr(sv, "label_token_ids", lambda t, forms: [1, 2])
    monkeypatch.setattr(sv, "letter_labels", lambda t, n: ["a"])
    with pytest.raises(_Stop):
        sv.LettersEngine(fam.model, mode="packed", family=fam, revision=revision)
    assert tok_calls == [(fam.model, {"revision": revision})]
    kw = llm_kw[0]
    assert kw.get("revision") == revision and kw.get("tokenizer_revision") == revision
    assert ("revision" in kw) is (revision is not None)


@pytest.mark.parametrize("revision", [REV, None])
def test_r2_second_engine_head_at_the_revision(revision, monkeypatch):
    llm_kw, head_calls = [], []

    class LLM:
        def __init__(self, **kw):
            llm_kw.append(kw)
            self.llm_engine = types.SimpleNamespace(
                vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
            )

        def get_tokenizer(self):
            return object()

    _fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(
        he.HiddenEngine,
        "_load_lm_head",
        staticmethod(lambda model, names=(), revision=None: head_calls.append((model, revision))),
    )
    monkeypatch.setattr(he.HiddenEngine, "readout", lambda self, *a, **k: None)
    he.HiddenEngine(QWEN.model, revision=revision)
    kw = llm_kw[0]
    assert kw["model"] == QWEN.model and kw.get("revision") == revision and kw.get("tokenizer_revision") == revision
    assert ("revision" in kw) is (revision is not None)
    assert head_calls == [(QWEN.model, revision)]
    # --engine's JSON may still add to the engine's arguments but the revision stays
    llm_kw.clear()
    he.HiddenEngine(QWEN.model, engine_kw={"enforce_eager": True}, revision=revision)
    assert llm_kw[0]["enforce_eager"] is True and llm_kw[0].get("revision") == revision


def _record_hf_loads(monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    calls = []

    class Model:
        def eval(self):
            return self

    def tok(model, **kw):
        calls.append(("tokenizer", model, kw))
        return object()

    def lm(model, **kw):
        calls.append(("model", model, kw))
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(tok))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lm))
    return calls


@pytest.mark.parametrize("revision", [REV, None])
def test_r3_hf_stand_in_engines_load_at_the_revision(revision, monkeypatch):
    from decisio.serve.hf_letters import HFLettersEngine

    calls = _record_hf_loads(monkeypatch)
    engine = HFLettersEngine("m", warm_up=False, revision=revision)
    assert engine.revision == revision
    for cls in (he.HFHiddenEngine, he.HFReservedHiddenEngine):
        built = cls("m", revision=revision)
        assert not hasattr(built, "revision") or built.revision == revision
    assert len(calls) == 6
    assert [c[2]["revision"] for c in calls] == [revision] * 6
    assert {c[0] for c in calls} == {"tokenizer", "model"}


def test_r4_main_hands_the_engines_the_resolved_revision(monkeypatch):
    """On the hf backend, what --base resolves to (or --revision gives) is what main() builds the text engine and either
    head engine with: the reserved-column head by default, the second engine with --head-engine."""
    import decisio.serve.hf_letters as hl

    seen = {}

    def text_engine(model, **kw):
        seen["text"] = (model, kw.get("revision", "absent"))
        return types.SimpleNamespace(pad_unit=None, facts=lambda: {})

    def head(name):
        def build(model, **kw):
            seen[name] = (model, kw.get("revision", "absent"))
            raise _Stop

        return build

    monkeypatch.setattr(hl, "HFLettersEngine", text_engine)
    monkeypatch.setattr(he, "HFReservedHiddenEngine", head("reserved"))
    monkeypatch.setattr(he, "HFHiddenEngine", head("second"))
    monkeypatch.setattr(sv, "deep_gemm_guard", lambda *a, **k: None)
    for extra, want in (([], QWEN.revision), (["--revision", REV], REV)):
        for flags, name in (([], "reserved"), (["--head-engine"], "second")):
            seen.clear()
            argv = ["vllm_engine", "--backend", "hf", "--base", "qwen3.6-35b-a3b", *extra, *flags]
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(_Stop):
                sv.main()
            assert seen == {"text": (QWEN.model, want), name: (QWEN.model, want)}, argv
