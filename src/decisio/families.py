# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The model families decisio serves, and the few things that differ between them.

  qwen3.6-moe     Qwen3.6-35B-A3B (the served default; a hybrid MoE, Gated DeltaNet plus attention). Everything here
                  is what decisio did before families existed, so a Qwen server is unchanged token for token.
  gemma4-unified  google/gemma-4-12B-it (dense attention, sliding plus full layers; encoder-free multimodal class in
                  vLLM 0.30.0, read through its text path). No front padding: an attention model's prefix cache hits
                  at any block boundary, so the 1,056-token rule of the hybrid does not apply. Tied embeddings and a
                  final-logit softcap of 30, which the head's recomputation of label logits must follow.

A family is detected from the checkpoint's config.json (`model_type`); anything that is not a known family is served
as Qwen, as before (the CPU stand-in's small Qwen models included).

Standard library only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from decisio.names import SERVED_NAME


@dataclass(frozen=True)
class Family:
    key: str
    model_types: tuple[str, ...]
    served_name: str
    # --pad-to's default: "block" (front padding to vLLM's block) or "none"
    pad_to: str
    # LLM(limit_mm_per_prompt=...): every multimodal input off on the text route
    limit_mm: dict = field(default_factory=dict)
    # --model-class -> the architecture name the engine is built under (None: the checkpoint's own, stock class)
    classes: dict = field(default_factory=dict)
    # where the output layer's rows are, in the checkpoint, for the head's recomputation of label logits
    head_rows: tuple[str, ...] = ("lm_head.weight",)
    # tanh softcap on the final logits (vLLM's LogitsProcessor soft_cap), or None
    softcap: float | None = None
    # the first reserved token id of the hidden-state readout (decisio.vllm_plugin.hidden): d + 1 ordinary tokens that
    # are no label form in any prompt format (checked at start-up, and per family in tests/unit/test_gemma_prompts.py)
    hidden_start: int = 100_000
    # the served default temperature, or None: the server then needs --temperature (no fit has shipped)
    temperature: float | None = None
    # the prompt flags' defaults (decisio.readout.letters.PromptFormat)
    system_prompt: str = "none"
    prompt_tail: str = "decisio"
    answer_slot: str = "prefill"
    label_variants: str = "single"
    # --noul-rendering's default (decisio.serve.systemone.NOUL_RENDERINGS)
    noul_rendering: str = "words"


QWEN = Family(
    key="qwen3.6-moe",
    model_types=("qwen3_5_moe",),
    served_name=SERVED_NAME,
    pad_to="block",
    limit_mm={"image": 0, "video": 0},
    classes={
        "text-only": "DecisioQwen3_5MoeTextOnly",
        "hidden-readout": "DecisioQwen3_5MoeHiddenReadout",
    },
    head_rows=("lm_head.weight",),
    softcap=None,
    temperature=1.307,
)

GEMMA4 = Family(
    key="gemma4-unified",
    model_types=("gemma4_unified",),
    served_name="decisio-gemma-4-12b-it-letters",
    pad_to="none",
    limit_mm={"image": 0, "video": 0, "audio": 0},
    # text-only is vLLM's own class (Gemma4UnifiedForConditionalGeneration) with every multimodal input off: Gemma has
    # no M-RoPE, so its text path gives a text prompt the text model's own positions, and it is the class Cygnet's
    # figures were measured with
    classes={"text-only": None, "hidden-readout": "DecisioGemma4UnifiedHiddenReadout"},
    # tied embeddings: the output layer is the input embedding
    head_rows=(
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ),
    softcap=30.0,
    # 100,000 to 103,840 holds three of Gemma's two-letter label tokens (found at start-up on a card, 2026-10-02)
    hidden_start=170_000,
    temperature=None,
    # the primary arm of experiments/2026-10-03_t7_gemma_base (PREREG.md, G1): Cygnet's system prompt and tail, the
    # template's own answer slot, every single-token form of a label summed
    system_prompt="cygnet",
    prompt_tail="cygnet",
    answer_slot="template",
    label_variants="summed",
    noul_rendering="letters",
)

FAMILIES = (QWEN, GEMMA4)


def read_config(model: str, revision: str | None = None) -> dict:
    """The checkpoint's config.json: from a local directory, else from the Hugging Face hub (the file alone)."""
    p = Path(model)
    if (p / "config.json").is_file():
        return json.loads((p / "config.json").read_text())
    if p.exists():
        return {}
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text())
    except Exception:
        return {}


def family_of(model: str | None, revision: str | None = None) -> Family:
    """The family of a checkpoint (a local directory or a hub id); Qwen for anything unknown. DECISIO_FAMILY, when set
    to a family's key, overrides the detection (for tokenizer-only tests)."""
    forced = os.environ.get("DECISIO_FAMILY")
    if forced:
        for f in FAMILIES:
            if f.key == forced:
                return f
        raise ValueError(f"DECISIO_FAMILY={forced!r} is not one of {[f.key for f in FAMILIES]}")
    if model is None:
        return QWEN
    mt = read_config(model, revision).get("model_type")
    for f in FAMILIES:
        if mt in f.model_types:
            return f
    return QWEN
