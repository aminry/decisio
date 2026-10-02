# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio's vLLM plugin entry point (`decisio.vllm_plugin.register`), against the vLLM stand-in (tests/stub_vllm.py).

P1  on the supported version it registers exactly the declared architectures, each in vLLM's lazy form
P2  re-entrancy: vLLM calls the entry point in every process and may call it again; a second call registers nothing
P3  version guard: on any other vLLM version (and without vLLM) nothing is registered, and it says so once
P4  the entry point is declared in pyproject.toml under `vllm.general_plugins` and resolves to `register`
P5  no name a decisio class defines shadows an attribute of torch.nn.Module instances or of the vLLM base class

  uv run pytest -q tests/unit/test_vllm_plugin.py
"""

import importlib
import logging
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from stub_vllm import stub_vllm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def plugin():
    return importlib.import_module("decisio.vllm_plugin")


def test_p1_registers_the_declared_models():
    with stub_vllm("0.30.0") as registry:
        p = plugin()
        assert p.register() is True
        assert dict(registry.calls) == p.MODELS and len(registry.calls) == len(p.MODELS)
        for arch, target in registry.calls:
            module, _, cls = target.partition(":")  # the lazy form, never a class object
            assert isinstance(target, str) and module.startswith("decisio.") and cls


def test_p1_config_hook_registered_for_every_class():
    """vLLM looks up its per-architecture config hook by architecture name; every decisio class must get the hook of
    the class it subclasses (without it a card ran 544-token blocks instead of 1,056, 2026-09-30)."""
    with stub_vllm("0.30.0") as registry:
        p = plugin()
        before = dict(registry.config_map)
        assert p.register()
        for arch in p.MODELS:
            assert registry.config_map[arch] == before[p.BASE_ARCHS[arch]]
        assert registry.config_map[p.GEMMA4_HIDDEN_READOUT] == "Gemma4Config"
        assert {k: v for k, v in registry.config_map.items() if k not in p.MODELS} == before  # nothing else touched
    with stub_vllm("0.30.0", config_map={"Qwen3ForCausalLM": "other"}) as registry:  # no hook to reuse
        p = plugin()
        assert p.register() is False and registry.calls == []
    # a vLLM without Gemma 4's hook: the Qwen classes register, the Gemma class does not
    with stub_vllm("0.30.0", config_map={"Qwen3_5MoeForCausalLM": "Qwen3_5ForCausalLMConfig"}) as registry:
        p = plugin()
        assert p.register() is True
        assert sorted(a for a, _ in registry.calls) == sorted([p.TEXT_ONLY, p.HIDDEN_READOUT])


def test_p2_reentrant():
    with stub_vllm("0.30.0") as registry:
        p = plugin()
        assert p.register() and p.register() and p.register()
        assert len(registry.calls) == len(p.MODELS)


@pytest.mark.parametrize("version", ["0.29.2", "0.30.1", "0.31.0", "dev"])
def test_p3_version_guard(version, caplog):
    with stub_vllm(version) as registry, caplog.at_level(logging.WARNING, logger="decisio.vllm_plugin"):
        p = plugin()
        assert p.register() is False and registry.calls == []
        assert "nothing registered" in caplog.text and version in caplog.text


def test_p3_without_vllm(monkeypatch):
    with stub_vllm("0.30.0"):
        p = plugin()
    monkeypatch.setitem(sys.modules, "vllm", None)  # import vllm -> ImportError
    sys.modules.pop("decisio.vllm_plugin", None)
    p = plugin()
    assert p.vllm_version() is None and p.register() is False
    sys.modules.pop("decisio.vllm_plugin", None)


def test_p4_entry_point_declared():
    eps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["entry-points"]["vllm.general_plugins"]
    assert eps == {"decisio": "decisio.vllm_plugin:register"}
    module, _, name = eps["decisio"].partition(":")
    assert callable(getattr(importlib.import_module(module), name))


def test_p5_no_class_shadows_a_torch_or_vllm_attribute():
    """A name a decisio class defines must not collide with an attribute torch.nn.Module sets on every instance (a
    method called `_buffers` is shadowed by the module's buffer registry) or with one of the vLLM base class."""
    import torch.nn as nn

    instance_attrs = set(vars(nn.Module()))
    with stub_vllm("0.30.0") as registry:
        p = plugin()
        assert p.register()
        base = sys.modules["vllm.model_executor.models.qwen3_5"].Qwen3_5MoeForCausalLM
        inherited = {n for c in base.__mro__ for n in vars(c)}
        for arch in p.MODELS:
            cls = registry.resolve(arch)
            own = {n for c in cls.__mro__ if c.__module__.startswith("decisio.") for n in vars(c)} - {
                "__module__",
                "__doc__",
                "__qualname__",
                "__firstlineno__",
                "__static_attributes__",
                "__init__",
                "forward",
                "compute_logits",
                "compute_logits_local",
                "load_weights",
            }
            assert not own & instance_attrs, (arch, sorted(own & instance_attrs))
            assert not own & inherited, (arch, sorted(own & inherited))


# ---- the text-only model class ------------------------------------------------------------------------------------


def test_m1_text_only_class_skips_the_vision_tower():
    import torch

    with stub_vllm("0.30.0") as registry:
        p = plugin()
        assert p.register()
        cls = registry.resolve(p.TEXT_ONLY)  # what vLLM imports for the architecture
        base = sys.modules["vllm.model_executor.models.qwen3_5"].Qwen3_5MoeForCausalLM
        assert issubclass(cls, base) and cls.__name__ == p.TEXT_ONLY
        m = cls(vllm_config=None)
        names = [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.merger.mlp.0.weight",
            "visual.patch_embed.proj.weight",
            "mtp.layers.0.weight",
        ]
        loaded = m.load_weights((n, torch.zeros(1)) for n in names)
        assert m.loaded == [n for n in names if "visual" not in n] and loaded == set(m.loaded)
        # nothing else differs from vLLM's class: the same logits
        h = torch.randn(3, 8)
        assert torch.equal(m.compute_logits(h), base.compute_logits(m, h))
        assert set(vars(cls)) - {
            "__module__",
            "__doc__",
            "__qualname__",
            "__firstlineno__",
            "__static_attributes__",
        } == {"load_weights"}


def test_m1_vision_rule_is_make_text_onlys():
    from decisio.vllm_plugin.weights import is_vision_weight

    # the rule decisio.serve.make_text_only applies to the checkpoint's index
    ref = lambda k: ".visual." in f".{k}" or k.startswith("visual.")  # noqa: E731
    for k in (
        "model.visual.blocks.1.norm1.weight",
        "visual.pos_embed",
        "model.language_model.layers.3.mlp.gate.weight",
        "lm_head.weight",
        "model.language_model.visualizer.weight",
        "xvisual.y",
    ):
        assert is_vision_weight(k) == ref(k), k


def test_m1_engine_kwargs_and_entry_point_check():
    with stub_vllm("0.30.0"):
        p = plugin()
        assert p.engine_kwargs(p.TEXT_ONLY) == {"hf_overrides": {"architectures": ["DecisioQwen3_5MoeTextOnly"]}}
        with pytest.raises(KeyError):
            p.engine_kwargs("Qwen3_5MoeForCausalLM")
        assert isinstance(p.installed_entry_point(), bool)


# ---- the Gemma 4 hidden-readout class -----------------------------------------------------------------------------


def test_g1_gemma_hidden_readout_writes_after_the_soft_cap(monkeypatch):
    """DecisioGemma4UnifiedHiddenReadout is vLLM's Gemma 4 unified class with [0, h] in the reserved columns, written
    after the class's own logits (soft cap included); every other column is the stock class's, bit for bit."""
    import torch

    from decisio.vllm_plugin.hidden import ENV_START, recover_hidden

    monkeypatch.setenv(ENV_START, "40")
    with stub_vllm("0.30.0", vocab=64, hidden=8) as registry:
        p = plugin()
        assert p.register()
        cls = registry.resolve(p.GEMMA4_HIDDEN_READOUT)
        base = sys.modules["vllm.model_executor.models.gemma4_unified"].Gemma4UnifiedForConditionalGeneration
        assert issubclass(cls, base) and cls.__name__ == p.GEMMA4_HIDDEN_READOUT
        m = cls(vllm_config=None)
        h = torch.randn(3, 8)
        with torch.no_grad():
            got, stock = m.compute_logits(h), base.compute_logits(m, h)
        keep = [i for i in range(64) if not 40 <= i <= 48]
        assert torch.equal(got[:, keep], stock[:, keep])
        lp = torch.log_softmax(got[:, 40:49].double(), -1).numpy()
        assert abs(recover_hidden(lp) - h.double().numpy()).max() < 1e-6
        dunder = {"__module__", "__doc__", "__qualname__", "__firstlineno__", "__static_attributes__"}
        assert set(vars(cls)) - dunder == {"compute_logits"}
