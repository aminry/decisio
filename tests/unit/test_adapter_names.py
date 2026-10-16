# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""An adapter in the other tensor-name layout is mapped, not refused (decisio.serve.adapters).

N1  the form of an adapter's names, and the form each served class needs
N2  mapping from the multimodal layout to the text layout and back: the tensors' bytes are unchanged, the names are the
    other layout's, the small files are copied as they are, the original folder is untouched
N3  an adapter that is in neither layout, mixes them, or has no weights file is refused with the reason
N4  the engine checks and maps the adapters before it builds vLLM's engine (a refusal costs no model load), serves the
    mapped copy, and reports the mapping
    uv run pytest -q tests/unit/test_adapter_names.py
"""

import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from decisio.families import BASES
from decisio.serve import adapters as ad
from decisio.serve import vllm_engine

TEXT = "base_model.model.model.layers."
MM = "base_model.model.model.language_model.layers."


def adapter(tmp_path, prefix, name="a", extra=None):
    d = tmp_path / name
    d.mkdir()
    t = {}
    for i in range(2):
        for m in ("self_attn.q_proj", "mlp.up_proj"):
            g = torch.Generator().manual_seed(7 * i + len(m))
            t[f"{prefix}{i}.{m}.lora_A.weight"] = torch.randn(4, 6, generator=g).to(torch.bfloat16)
            t[f"{prefix}{i}.{m}.lora_B.weight"] = torch.randn(6, 4, generator=g).to(torch.bfloat16)
    t.update(extra or {})
    save_file(t, str(d / ad.WEIGHTS), metadata={"format": "pt"})
    (d / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj", "up_proj"]}))
    return d, t


def read(d):
    with safe_open(d / ad.WEIGHTS, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def test_n1_forms_and_wanted_forms():
    assert ad.form_of([TEXT + "0.mlp.up_proj.lora_A.weight"]) == "text"
    assert ad.form_of([MM + "0.mlp.up_proj.lora_A.weight"]) == "multimodal"
    assert (
        ad.form_of([TEXT + "0.x", MM + "0.x"]) is None
        and ad.form_of(["something.else"]) is None
        and ad.form_of([]) is None
    )
    assert ad.wanted_form("DecisioGemma4TextOnlyForCausalLM") == "text" and ad.wanted_form("ForCausalLM") == "text"
    assert ad.wanted_form("Gemma4ForConditionalGeneration") == "multimodal"
    assert ad.wanted_form("SomethingElse") is None and ad.wanted_form(None) is None


@pytest.mark.parametrize("have,want", [(MM, "text"), (TEXT, "multimodal")])
def test_n2_the_mapped_copy_has_the_other_names_and_the_same_bytes(tmp_path, have, want):
    src, tensors = adapter(tmp_path, have)
    before = (src / ad.WEIGHTS).read_bytes()
    paths, report = ad.prepare(
        {"a": src}, "ForCausalLM" if want == "text" else "Gemma4ForConditionalGeneration", workdir=tmp_path / "work"
    )
    out = Path(paths["a"])
    assert out != src and (src / ad.WEIGHTS).read_bytes() == before  # the original is untouched
    got = read(out)
    expected = {ad.rename(k, want): v for k, v in tensors.items()}
    assert sorted(got) == sorted(expected) and ad.form_of(got) == want
    assert all(torch.equal(got[k], expected[k]) for k in got)  # every tensor's bytes
    assert (out / "adapter_config.json").read_text() == (src / "adapter_config.json").read_text()
    n = int.from_bytes((out / ad.WEIGHTS).read_bytes()[:8], "little")
    assert (8 + n) % 8 == 0  # the tensor data starts on an 8-byte boundary
    r = report["a"]
    assert r["mapped"] is True and r["names"] == ("multimodal" if want == "text" else "text") and r["mapped_to"] == want
    assert r["tensors"] == len(tensors) and "->" in r["mapping"]


def test_n2_an_adapter_already_in_the_wanted_layout_is_served_as_given(tmp_path):
    src, _ = adapter(tmp_path, TEXT)
    paths, report = ad.prepare({"a": src}, "ForCausalLM", workdir=tmp_path / "work")
    assert paths["a"] == str(src) and report["a"]["mapped"] is False and report["a"]["names"] == "text"
    assert not (tmp_path / "work").exists()
    paths, report = ad.prepare({"a": src}, "SomeUnknownClass")  # a class not known to need one: as given
    assert paths["a"] == str(src) and report["a"]["mapped"] is False


def test_n3_refusals_name_the_adapter_and_the_reason(tmp_path):
    mixed, _ = adapter(
        tmp_path, TEXT, name="mixed", extra={MM + "9.x.lora_A.weight": torch.zeros(1, dtype=torch.bfloat16)}
    )
    with pytest.raises(ValueError, match=r"adapter mixed: its tensor names are neither .* or they mix the two"):
        ad.prepare({"mixed": mixed}, "ForCausalLM")
    odd, _ = adapter(tmp_path, "weird.prefix.", name="odd")
    with pytest.raises(ValueError, match="adapter odd: its tensor names are neither"):
        ad.prepare({"odd": odd}, "Gemma4ForConditionalGeneration")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="adapter empty: no adapter_model.safetensors"):
        ad.prepare({"empty": empty}, "ForCausalLM")


class FakeLLM:
    built = 0

    def __init__(self, **kw):
        FakeLLM.built += 1
        self.llm_engine = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                cache_config=types.SimpleNamespace(block_size=16, prefix_match_unit=None),
                model_config=types.SimpleNamespace(max_model_len=32768),
            )
        )

    def get_tokenizer(self):
        return object()

    def collective_rpc(self, name):
        return [None]


@pytest.fixture
def fake_vllm(monkeypatch):
    requests = []

    class LoRARequest:
        def __init__(self, name, i, path):
            requests.append((name, i, path))

    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.lora": types.ModuleType("vllm.lora"),
        "vllm.lora.request": types.ModuleType("vllm.lora.request"),
    }
    mods["vllm"].LLM = FakeLLM
    mods["vllm.lora.request"].LoRARequest = LoRARequest
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(vllm_engine.LettersEngine, "_warm_up", lambda self: None)
    FakeLLM.built = 0
    return requests


TEXT_CLASS = {"hf_overrides": {"architectures": ["DecisioGemma4TextOnly"]}}  # what a decisio class gives vLLM


def test_n4_a_refused_adapter_is_refused_before_the_engine_is_built(tmp_path, fake_vllm):
    bad, _ = adapter(tmp_path, "weird.prefix.", name="bad")
    with pytest.raises(ValueError, match="adapter bad"):
        vllm_engine.LettersEngine("stand-in", family=BASES["gemma-4-31b"], adapters={"bad": bad}, engine_kw=TEXT_CLASS)
    assert FakeLLM.built == 0


def test_n4_the_engine_serves_the_mapped_copy_and_reports_it(tmp_path, fake_vllm):
    src, tensors = adapter(tmp_path, MM, name="metask")
    e = vllm_engine.LettersEngine(
        "stand-in", family=BASES["gemma-4-31b"], adapters={"metask": src}, engine_kw=TEXT_CLASS
    )
    assert FakeLLM.built == 1
    ((name, i, path),) = fake_vllm
    assert name == "metask" and i == 1 and path != str(src) and ad.form_of(read(Path(path))) == "text"
    assert sorted(e.adapters) == ["metask"]  # `adapters` in /health stays the list of names
    assert e.adapter_names["metask"]["mapped"] is True and e.adapter_names["metask"]["names"] == "multimodal"
    # an adapter already in the class's layout is served from where it is
    own, _ = adapter(tmp_path, TEXT, name="own")
    fake_vllm.clear()
    e = vllm_engine.LettersEngine("stand-in", family=BASES["gemma-4-31b"], adapters={"own": own}, engine_kw=TEXT_CLASS)
    assert fake_vllm == [("own", 1, str(own))] and e.adapter_names["own"]["mapped"] is False


def test_n4_the_multimodal_class_gets_the_multimodal_names(tmp_path, fake_vllm, monkeypatch):
    # no decisio class: the checkpoint's own architecture decides (read from its config.json)
    monkeypatch.setattr(
        "decisio.families.read_config",
        lambda model, revision=None: {"architectures": ["Gemma4ForConditionalGeneration"]},
    )
    src, _ = adapter(tmp_path, TEXT, name="t")
    vllm_engine.LettersEngine("stand-in", family=BASES["gemma-4-31b"], adapters={"t": src})
    ((_, _, path),) = fake_vllm
    assert ad.form_of(read(Path(path))) == "multimodal"
