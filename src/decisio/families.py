# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The bases decisio serves, chosen with `--base`, and what differs between them.

  qwen3.6-35b-a3b  Qwen/Qwen3.6-35B-A3B-FP8 at a pinned revision, the default: a hybrid MoE (Gated DeltaNet plus
                   attention). Its defaults are the served defaults as they were before bases existed, so a Qwen
                   server is unchanged.
  gemma-4-12b      google/gemma-4-12B-it at a pinned revision: dense attention (sliding plus full layers), read
                   through vLLM's encoder-free multimodal class with every multimodal input off. No front padding (an
                   attention model's prefix cache needs none); tied embeddings and a final-logit soft cap of 30, which
                   the intent head follows; a system turn, the spaced layout read at the template's own answer slot,
                   every single-token form of a label summed, yes/no as a two-option letter choice, and its own
                   temperature, all measured as one configuration (EVAL_CARD.md).

  gemma-4-31b      google/gemma-4-31B-it at a pinned revision: dense attention, read through vLLM's text class of the
                   multimodal checkpoint with the vision tower left out at load, quantized to FP8 on load; the 12B's
                   prompt (system turn, spaced layout at the template's answer slot, summed label forms, yes/no as
                   letters) with its own temperatures (EVAL_CARD.md).

Without `--base`, the base is detected from the checkpoint's config.json (`model_type`, and for `gemma4` checkpoints the
mixture-of-experts flag: only the dense one is the 31B base); anything unknown is served as the Qwen base, as before
(the CPU stand-in's small models included).

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
    # record names the bytes it was measured on; it applies whenever the checkpoint is this one, however it was named,
    # and --revision overrides it: see pinned_revision)
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
    # LLM(quantization=...): the precision the checkpoint is quantized to on load (None: as the checkpoint stores it)
    quantization: str | None = None
    # --multi-question's default when the engine runs in the server's process (vllm_engine.resolve_multi_question)
    multi_question: str = "sequential"
    # register a state's boundary in the prefix cache for a request with one question too, with the warm-up the
    # multi-question path sends (vllm_engine.LettersEngine._answer_separate): vLLM keeps only the latest sliding-window
    # checkpoint of a finished request, so without it a later, different question about the state reads it again
    register_state_boundary: bool = False
    # detection without --base where two checkpoints share a model type: the config's mixture-of-experts flag must
    # equal this (None: either)
    moe: bool | None = None


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
    # not needed: the state is padded to end on the 1,056-token block, also the cache's hit unit, and a question is
    # shorter than a block, so the state's end is already the latest checkpoint of a request with one question
    register_state_boundary=False,
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
    # a request's latest checkpoint lies inside its question (hit unit 64), so a second, different question about a
    # state read it again (RLCD experiments/2026-10-05_t7_card_e2_e3_e4, on the 31B: 531 ms instead of 47 at 3,000
    # tokens); vLLM's retention interval would keep it too, but evicts states sooner (the same record)
    register_state_boundary=True,
)

GEMMA4_31B = Family(
    key="gemma-4-31b",
    model_types=("gemma4",),
    model="google/gemma-4-31B-it",
    revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    served_name="decisio-gemma-4-31b-it-letters",
    pad_to="none",
    limit_mm={"image": 0, "video": 0},
    # vLLM's text class of the Gemma4ForConditionalGeneration checkpoint, the vision tower left out at load
    classes={"text-only": "DecisioGemma4TextOnly", "hidden-readout": "DecisioGemma4HiddenReadout"},
    head_rows=GEMMA4.head_rows,
    softcap=30.0,
    head_label_logprobs="engine",
    # 210,000 to 215,376: the 12B's 170,000 would put a label form (174,960) inside 5,376 dimensions
    hidden_start=210_000,
    temperature=5.252,
    choice_temperature=4.672,
    system_prompt=True,
    prompt_tail="spaced",
    answer_slot="template",
    label_variants="summed",
    noul_rendering="letters",
    # FP8 on load (vLLM 0.30.0's online FP8): at bf16 the weights leave too little of one 96 GB card for a 32,768-token
    # context; a pinned FP8 checkpoint replaces this when one exists
    quantization="fp8",
    moe=False,
    # as the 12B (hit unit 32): RLCD experiments/2026-10-05_t7_card_e2_e3_e4, arms R0 and amendment 1
    register_state_boundary=True,
    # warm: four questions in one batch after the state's prefill, repeating exactly in-process; gated against
    # sequential (runs/2026-10-05_engine-death-gates: no choice changed, the largest probability difference 0.0073)
    multi_question="warm",
)

FAMILIES = (QWEN, GEMMA4, GEMMA4_31B)
BASES = {f.key: f for f in FAMILIES}


def pinned_revision(model: str, revision: str | None = None) -> str | None:
    """The revision to load `model` at: the one given, else the pin of the base whose checkpoint `model` is (however it
    was named: --base, --model, the container's entrypoint), else None. A local directory that happens to be called
    like a base's repository is not that repository and gets no pin."""
    if revision is not None:
        return revision
    if Path(model).exists():
        return None
    return next((f.revision for f in FAMILIES if f.model == model), None)


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
    cfg = read_config(model, revision)
    mt = cfg.get("model_type")
    moe = bool((cfg.get("text_config") or cfg).get("enable_moe_block"))
    return next((f for f in FAMILIES if mt in f.model_types and (f.moe is None or f.moe == moe)), QWEN)
