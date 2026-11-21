# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio's Gemma 4 class for vLLM 0.31.0, registered by `decisio.vllm_plugin.register` in the lazy form, so this
module is imported only inside vLLM, when it builds a model of this architecture.

  DecisioGemma4UnifiedHiddenReadout
      vLLM's own `Gemma4UnifiedForConditionalGeneration` (google/gemma-4-12B-it: encoder-free, the language model is
      vLLM's `Gemma4ForCausalLM`) that also returns the final-norm hidden state at the answer position through d + 1
      reserved logit columns (`decisio.vllm_plugin.hidden`), written after the class's own final-logit soft cap and
      special-token suppression, so the intent head needs no second copy of the weights. The text path is the stock
      class with every multimodal input off (`decisio.families.GEMMA4.limit_mm`); there is no text-only subclass,
      because Gemma has no M-RoPE and the stock class reads a text prompt exactly as its language model does. A
      serving-only class, like the Qwen one: only the decision server's head-capable engine uses it.

  DecisioGemma4UnifiedHiddenReadoutTap
      a study prototype (`decisio.vllm_plugin.tap`): the class above that also returns, in the same request, the
      final-norm state and the residual stream after the layers in DECISIO_TAP_LAYERS at the answer position. Its
      forward returns [final | layers ...] for every token (vLLM's EAGLE-3 auxiliary states of the language model,
      concatenated after the compiled region); `compute_logits` computes the logits from the contiguous final part,
      as the class above does, and keeps the answer-position rows for `tap.HiddenTap`.

  DecisioGemma4TextOnly
      the text model of a `Gemma4ForConditionalGeneration` checkpoint (google/gemma-4-31B-it): vLLM's own
      `Gemma4ForCausalLM`, which loads the
      checkpoint's `model.language_model.*` tensors and leaves the vision and audio towers out at load time, so the
      engine neither builds nor profiles them.

  DecisioGemma4HiddenReadout
      the text-only class that also returns the final-norm hidden state through the reserved logit columns, written
      after the final-logit soft cap, as on the 12B.
"""

from __future__ import annotations

from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration

from decisio.vllm_plugin import tap
from decisio.vllm_plugin.hidden import write_hidden_columns


class DecisioGemma4UnifiedHiddenReadout(Gemma4UnifiedForConditionalGeneration):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        return logits if logits is None else write_hidden_columns(logits, hidden_states)


class DecisioGemma4UnifiedHiddenReadoutTap(DecisioGemma4UnifiedHiddenReadout):
    def __init__(self, *, vllm_config=None, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = getattr(self.config, "text_config", self.config)
        self.tap_d = int(cfg.hidden_size)
        self.tap_layers = tap.tap_layers()
        n = getattr(cfg, "num_hidden_layers", None)
        if n is not None and any(layer > n for layer in self.tap_layers):
            raise ValueError(f"DECISIO_TAP_LAYERS {self.tap_layers}: this model has {n} layers")
        if self.tap_layers:
            self.set_aux_hidden_state_layers(self.tap_layers)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, tuple):  # (final-norm state, [layer states]) once auxiliary layers are set
            import torch

            final, layers = out
            return torch.cat([final, *layers], dim=-1)
        return out

    def compute_logits(self, hidden_states):
        tap.stash(hidden_states)
        final = hidden_states
        if hidden_states.shape[-1] != self.tap_d:
            final = hidden_states[..., : self.tap_d].contiguous()
        return super().compute_logits(final)


class DecisioGemma4TextOnly(Gemma4ForCausalLM):
    pass


class DecisioGemma4HiddenReadout(DecisioGemma4TextOnly):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        return logits if logits is None else write_hidden_columns(logits, hidden_states)
