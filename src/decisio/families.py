# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The two bases decisio serves, chosen with `--base`, and what differs between them.

  qwen3.6-35b-a3b  Qwen/Qwen3.6-35B-A3B-FP8 at a pinned revision, the default: a hybrid MoE (Gated DeltaNet plus
                   attention). Its defaults are the served defaults as they were before bases existed, so a Qwen
                   server is unchanged.
  gemma-4-12b      google/gemma-4-12B-it at a pinned revision: dense attention (sliding plus full layers), read
                   through vLLM's encoder-free multimodal class with every multimodal input off. No front padding (an
                   attention model's prefix cache needs none); tied embeddings and a final-logit soft cap of 30, which
                   the intent head follows; a system turn, the spaced layout read at the template's own answer slot,
                   every single-token form of a label summed, yes/no as a two-option letter choice, and its own
                   temperature, all measured as one configuration (EVAL_CARD.md).

Without `--base`, the base is detected from the checkpoint's config.json (`model_type`); anything unknown is served as
the Qwen base, as before (the CPU stand-in's small models included).

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
    # the checkpoint --base serves when --model is not given, and its pinned revision (every base pins one, so a run
    # record names the bytes it was measured on; --revision overrides it)
    model: str
    revision: str
    served_name: str
    # --pad-to's default: "block" (front padding to vLLM's block) or "none"
    pad_to: str
    # LLM(limit_mm_per_prompt=...): every multimodal input off on the text route
    limit_mm: dict = field(default_factory=dict)
    # --model-class -> the architecture the engine is built under (None: the checkpoint's own class, unmodified)
    classes: dict = field(default_factory=dict)
    # where the output layer's rows are in the checkpoint, for the head (tied embeddings name the input embedding)
    head_rows: tuple[str, ...] = ("lm_head.weight",)
    # tanh soft cap on the final logits (vLLM's LogitsProcessor soft_cap), or None
    softcap: float | None = None
    # where the intent head takes the label log-probabilities from: "recompute" (from the hidden state and the output
    # layer's rows, the engine's arithmetic reproduced) or "engine" (the engine's own readout of the same prompt, one
    # more request; for a base whose logits the recomputation cannot reproduce exactly, Gemma's soft cap)
    head_label_logprobs: str = "recompute"
    # the first reserved token id of the hidden-state readout (decisio.vllm_plugin.hidden): d + 1 ordinary tokens that
    # are no label form in any prompt format (checked at start-up and per base in tests)
    hidden_start: int = 100_000
    # the served temperatures: global, and for choice questions (None: the global one)
    temperature: float = 1.0
    choice_temperature: float | None = None
    # the prompt's defaults (decisio.readout.letters.PromptFormat) and the yes/no rendering
    system_prompt: bool = False
    prompt_tail: str = "spaced"
    answer_slot: str = "prefill"
    label_variants: str = "single"
    noul_rendering: str = "letters-keys"


QWEN = Family(
    key="qwen3.6-35b-a3b",
    model_types=("qwen3_5_moe",),
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    # the revision behind every Qwen record (runs/); the repository's head was the same commit when it was pinned
    revision="95a723d08a9490559dae23d0cff1d9466213d989",
    served_name=SERVED_NAME,
    pad_to="block",
    limit_mm={"image": 0, "video": 0},
    classes={
        "text-only": "DecisioQwen3_5MoeTextOnly",
        "hidden-readout": "DecisioQwen3_5MoeHiddenReadout",
    },
    temperature=1.506,
    choice_temperature=1.370,
)

GEMMA4 = Family(
    key="gemma-4-12b",
    model_types=("gemma4_unified",),
    model="google/gemma-4-12B-it",
    revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    served_name="decisio-gemma-4-12b-it-letters",
    pad_to="none",
    limit_mm={"image": 0, "video": 0, "audio": 0},
    # text-only is vLLM's own class with every multimodal input off: Gemma has no M-RoPE, so a text prompt gets the
    # language model's own positions
    classes={"text-only": None, "hidden-readout": "DecisioGemma4UnifiedHiddenReadout"},
    head_rows=(
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ),
    softcap=30.0,
    head_label_logprobs="engine",
    # 100,000 to 103,840 holds three of Gemma's two-letter label tokens
    hidden_start=170_000,
    temperature=3.592,
    choice_temperature=None,
    system_prompt=True,
    prompt_tail="spaced",
    answer_slot="template",
    label_variants="summed",
    noul_rendering="letters",
)

FAMILIES = (QWEN, GEMMA4)
BASES = {f.key: f for f in FAMILIES}


def read_config(model: str, revision: str | None = None) -> dict:
    """The checkpoint's config.json: from a local directory, else from the Hugging Face hub (that file alone)."""
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
    """The base a checkpoint (a local directory or a hub id) belongs to; Qwen for anything unknown. DECISIO_BASE, when
    set to a base's key, overrides the detection (tokenizer-only tests)."""
    forced = os.environ.get("DECISIO_BASE")
    if forced:
        if forced not in BASES:
            raise ValueError(f"DECISIO_BASE={forced!r} is not one of {sorted(BASES)}")
        return BASES[forced]
    if model is None:
        return QWEN
    mt = read_config(model, revision).get("model_type")
    return next((f for f in FAMILIES if mt in f.model_types), QWEN)
