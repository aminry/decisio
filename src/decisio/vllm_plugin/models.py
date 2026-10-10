# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""decisio's model classes for vLLM 0.31.0, registered by `decisio.vllm_plugin.register` in the lazy form, so this
module is imported only inside vLLM, when it builds a model of one of these architectures.

  DecisioQwen3_5MoeTextOnly   the text model of the official Qwen3.6-35B-A3B checkpoint, loaded without the vision
                              tower: vLLM's `Qwen3_5MoeForCausalLM` with the checkpoint's `visual` tensors left out at
                              load time. It replaces the rewritten checkpoint directory
                              (`decisio.serve.make_text_only`); the text class uses plain RoPE, where the multimodal
                              class builds M-RoPE positions for every prompt token.

  DecisioQwen3_5MoeHiddenReadout
                              the text-only class that also returns the hidden state at the answer position through
                              d + 1 reserved logit columns (`decisio.vllm_plugin.hidden`), so the intent head needs no
                              second copy of the weights. A serving-only class: a text generation that sampled a
                              reserved token would be wrong, so only the decision server's head-capable engine uses it.

A class is selected per engine with `hf_overrides={"architectures": ["<name>"]}` (`decisio.vllm_plugin.engine_kwargs`).
"""

from __future__ import annotations

from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

from decisio.vllm_plugin.hidden import write_hidden_columns
from decisio.vllm_plugin.weights import without_vision


class DecisioQwen3_5MoeTextOnly(Qwen3_5MoeForCausalLM):
    def load_weights(self, weights):
        return super().load_weights(without_vision(weights))


class DecisioQwen3_5MoeHiddenReadout(DecisioQwen3_5MoeTextOnly):
    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        return logits if logits is None else write_hidden_columns(logits, hidden_states)

    def compute_logits_local(self, hidden_states):
        return write_hidden_columns(super().compute_logits_local(hidden_states), hidden_states)
