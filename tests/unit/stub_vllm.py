# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A stand-in for the `vllm` package, for tests on machines where vLLM cannot be installed (it has no macOS wheels).

`stub_vllm(version)` is a context manager that puts fake `vllm` modules into `sys.modules`: the package with its
`__version__`, a `ModelRegistry` that records registrations (and resolves the lazy `<module>:<class>` form on
request), and `vllm.model_executor.models.qwen3_5` with a minimal `Qwen3_5MoeForCausalLM` (a torch module with an
`lm_head`, `compute_logits` and a `load_weights` that records the names it is given), Gemma 4's unified and text
classes in the same spirit, plus `SamplingParams` and
`vllm.inputs.TokensPrompt` as plain records. Only what decisio's plugin and the server's head path touch is modelled;
anything that loads a real model needs a card.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types


class RecordingRegistry:
    def __init__(self):
        self.models: dict[str, str | type] = {}
        self.calls: list[tuple[str, str | type]] = []

    def register_model(self, model_arch, model_cls):
        if isinstance(model_cls, str) and len(model_cls.split(":")) != 2:
            raise ValueError("Expected a string in the format `<module>:<class>`")
        self.calls.append((model_arch, model_cls))
        self.models[model_arch] = model_cls

    def resolve(self, model_arch):
        """The class vLLM would import for an architecture (the lazy form imported now)."""
        target = self.models[model_arch]
        if isinstance(target, str):
            module, name = target.split(":")
            return getattr(importlib.import_module(module), name)
        return target


def _qwen3_5_module(vocab=64, hidden=8, zero_head=False):
    import torch
    from torch import nn

    class Qwen3_5MoeForCausalLM(nn.Module):
        """The slice of vLLM's class the plugin subclasses: lm_head, compute_logits, load_weights."""

        def __init__(self, *, vllm_config=None, prefix: str = ""):
            super().__init__()
            self.vllm_config = vllm_config
            self.config = types.SimpleNamespace(vocab_size=vocab, hidden_size=hidden)
            torch.manual_seed(0)
            # zero_head: all-zero vocabulary logits without a vocab x hidden matrix (for real hidden sizes)
            self.lm_head = None if zero_head else nn.Linear(hidden, vocab, bias=False)
            self.loaded: list[str] = []

        def compute_logits(self, hidden_states):
            if self.lm_head is None:
                return torch.zeros(*hidden_states.shape[:-1], vocab, dtype=hidden_states.dtype)
            return self.lm_head(hidden_states)

        def compute_logits_local(self, hidden_states):
            return self.compute_logits(hidden_states)

        def load_weights(self, weights):
            self.loaded = [name for name, _ in weights]
            return set(self.loaded)

    mod = types.ModuleType("vllm.model_executor.models.qwen3_5")
    mod.Qwen3_5MoeForCausalLM = Qwen3_5MoeForCausalLM
    return mod


def _gemma4_unified_module(vocab=64, hidden=8, softcap=30.0):
    import torch
    from torch import nn

    class Gemma4UnifiedForConditionalGeneration(nn.Module):
        """The slice of vLLM's Gemma 4 unified class the plugin subclasses: compute_logits with the final soft cap."""

        def __init__(self, *, vllm_config=None, prefix: str = ""):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab, hidden_size=hidden)
            torch.manual_seed(0)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def compute_logits(self, hidden_states):
            z = self.lm_head(hidden_states)
            return torch.tanh(z / softcap) * softcap

    mod = types.ModuleType("vllm.model_executor.models.gemma4_unified")
    mod.Gemma4UnifiedForConditionalGeneration = Gemma4UnifiedForConditionalGeneration
    return mod


def _gemma4_module(vocab=64, hidden=8, softcap=30.0):
    import torch
    from torch import nn

    class Gemma4ForCausalLM(nn.Module):
        """The slice of vLLM's Gemma 4 text class the plugin subclasses: compute_logits with the final soft cap, and a
        load_weights that leaves the multimodal towers out (as vLLM's does) and records the names it kept."""

        def __init__(self, *, vllm_config=None, prefix: str = ""):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab, hidden_size=hidden)
            torch.manual_seed(0)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.loaded: list[str] = []

        def compute_logits(self, hidden_states):
            z = self.lm_head(hidden_states)
            return torch.tanh(z / softcap) * softcap

        def load_weights(self, weights):
            towers = ("vision_tower.", "embed_vision.", "audio_tower.", "embed_audio.")
            self.loaded = [n.replace("language_model.", "") for n, _ in weights if not any(t in n for t in towers)]
            return set(self.loaded)

    mod = types.ModuleType("vllm.model_executor.models.gemma4")
    mod.Gemma4ForCausalLM = Gemma4ForCausalLM
    return mod


@contextlib.contextmanager
def stub_vllm(version="0.30.0", with_models=True, vocab=64, hidden=8, zero_head=False, config_map=None):
    names = [
        "vllm",
        "vllm.inputs",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.config",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.gemma4_unified",
        "vllm.model_executor.models.gemma4",
    ]
    saved = {n: sys.modules.get(n) for n in names}
    ours = [n for n in list(sys.modules) if n.startswith("decisio.vllm_plugin")]
    saved_ours = {n: sys.modules.pop(n) for n in ours}  # a fresh plugin state per stub
    registry = RecordingRegistry()
    vllm = types.ModuleType("vllm")
    vllm.__version__ = version
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kw: types.SimpleNamespace(**kw)
    sys.modules["vllm.inputs"] = inputs
    me = types.ModuleType("vllm.model_executor")
    models = types.ModuleType("vllm.model_executor.models")
    models.ModelRegistry = registry
    config = types.ModuleType("vllm.model_executor.models.config")  # the per-architecture config hooks
    config.MODELS_CONFIG_MAP = dict(
        config_map
        if config_map is not None
        else {
            "Qwen3_5MoeForCausalLM": "Qwen3_5ForCausalLMConfig",
            "Qwen3ForCausalLM": "other",
            "Gemma4UnifiedForConditionalGeneration": "Gemma4Config",
            "Gemma4ForCausalLM": "Gemma4Config",
        }
    )
    registry.config_map = config.MODELS_CONFIG_MAP
    sys.modules["vllm.model_executor.models.config"] = config
    sys.modules.update({"vllm": vllm, "vllm.model_executor": me, "vllm.model_executor.models": models})
    if with_models:
        sys.modules["vllm.model_executor.models.qwen3_5"] = _qwen3_5_module(vocab, hidden, zero_head)
        sys.modules["vllm.model_executor.models.gemma4_unified"] = _gemma4_unified_module(vocab, hidden)
        sys.modules["vllm.model_executor.models.gemma4"] = _gemma4_module(vocab, hidden)
    try:
        yield registry
    finally:
        for n in [m for m in list(sys.modules) if m.startswith("decisio.vllm_plugin")]:
            del sys.modules[n]
        sys.modules.update(saved_ours)
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
