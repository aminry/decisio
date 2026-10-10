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

from decisio.vllm_plugin.hidden import write_hidden_columns


class DecisioGemma4UnifiedHiddenReadout(Gemma4UnifiedForConditionalGeneration):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        return logits if logits is None else write_hidden_columns(logits, hidden_states)


class DecisioGemma4TextOnly(Gemma4ForCausalLM):
    pass


class DecisioGemma4HiddenReadout(DecisioGemma4TextOnly):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        return logits if logits is None else write_hidden_columns(logits, hidden_states)
