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

  gemma-4-12b-fp8  the 12B in FP8, a second precision beside the bf16 one (which stays the 12B's default and the one
                   every 12B record was measured on): the same checkpoint and prompt, its linear layers quantized to
                   FP8 on load (per-tensor weight scales, per-token activations, as the 31B), or a stored FP8
                   conversion of it (`python -m decisio.hub_fp8 convert`), which loads to the same tensors. Its own key
                   and served name, so its tasks, certificates and stack are its own; its temperatures are the bf16
                   12B's until they are refitted on FP8 reads.

  gemma-4-31b      google/gemma-4-31B-it at a pinned revision: dense attention, read through vLLM's text class of the
                   multimodal checkpoint with the vision tower left out at load, quantized to FP8 on load; the 12B's
                   prompt (system turn, spaced layout at the template's answer slot, summed label forms, yes/no as
                   letters) with its own temperatures (EVAL_CARD.md).

Without `--base`, the base is detected from the checkpoint's config.json (`model_type`, and for `gemma4` checkpoints the
mixture-of-experts flag: only the dense one is the 31B base); where two bases share a checkpoint in two precisions (the
12B), the one whose precision the checkpoint declares: FP8 only for a checkpoint whose config.json declares FP8 weights,
bf16 otherwise. Anything unknown is refused (the CPU stand-in keeps the Qwen base for its small models).

Standard library only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
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
    # the longest `--max-model-len` the documents state for this base (vLLM server only): the server refuses a longer
    # one. 32,768 is the length every record was measured at; a longer ceiling is an opt-in whose card record is
    # runs/ (docs/running.md, "The context length")
    context_ceiling: int = 32768
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
    # --register-boundary's default where register_state_boundary is set (vllm_engine.resolve_register_boundary):
    # before, ahead of the question, or after, once the response is out (decisio.serve.boundary). The 12B's was after
    # from 0.9.0 to 0.11.0: a request that follows another then waits behind the previous state's registration (about
    # one extra read of the state at 3,000 and 6,000 tokens; docs/running.md), so both Gemma bases default to before
    register_boundary: str = "before"
    # --backend mlx: the prefix cache's budget in MiB when --prefix-cache-mb is not given (vllm_engine.main). Per base,
    # because an entry holds a state's whole cache and the sliding-window layers keep 1.05 to 1.25 GB of it at long
    # states; the figure is what stays under two thirds of a 32 GiB Mac's memory beside the base's peak at 32k tokens
    # (RLCD experiments/2026-10-06_ls_mlx_cache_budget)
    mlx_prefix_cache_mb: int = 2048
    # --backend mlx: the base's peak memory above its weights at a 32k-token state, in GB (1e9 bytes), as measured
    # (docs/running.md: Qwen 4-bit 19.5 -> 22.7 GB, Gemma 4 12B 9.7 -> 15.1, Gemma 4 31B 24.9 -> 33.7), rounded up;
    # decisio.serve.mlx_memory sizes MLX's memory limit from it (0: the default, 6)
    mlx_working_gb: float = 0
    # detection without --base where two checkpoints share a model type: the config's mixture-of-experts flag must
    # equal this (None: either)
    moe: bool | None = None
    # where one checkpoint is served in two precisions (the 12B: "bf16" and "fp8"), which this base is; detection
    # without --base takes "fp8" only for a checkpoint that declares FP8 weights (declares_fp8), "bf16" otherwise.
    # None: the base's only precision (as `quantization` and the checkpoint say)
    precision: str | None = None


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
    mlx_working_gb=4,
)

GEMMA4 = Family(
    key="gemma-4-12b",
    model_types=("gemma4_unified",),
    model="google/gemma-4-12B-it",
    revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    served_name="decisio-gemma-4-12b-it-letters",
    pad_to="none",
    limit_mm={"image": 0, "video": 0, "audio": 0},
    context_ceiling=65536,
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
    # the default 2,048 MiB kept 5 states of 1,000 tokens (Qwen's 25); 7,400 MiB keeps 20 of 1,000, 10 of 3,000 and 7 of
    # 8,000: two thirds of a 32 GiB Mac (22.906 GB) less the 15.125 GB peak at 32k tokens leaves 7.781 GB = 7,421 MiB
    mlx_prefix_cache_mb=7400,
    mlx_working_gb=6,
    precision="bf16",
)

GEMMA4_FP8 = replace(
    GEMMA4,
    key="gemma-4-12b-fp8",
    served_name="decisio-gemma-4-12b-it-fp8-letters",
    # FP8 on load, as the 31B (vLLM 0.31.0's online per-tensor FP8); a stored conversion that declares FP8 is loaded
    # as stored (as_loaded) and holds the same tensors (RLCD experiments/2026-10-21_serve_12b_fp8/results/convert_12b:
    # 192 fused groups, the loader changes 0 weights)
    quantization="fp8",
    precision="fp8",
)

GEMMA4_31B = Family(
    key="gemma-4-31b",
    model_types=("gemma4",),
    model="google/gemma-4-31B-it",
    revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    served_name="decisio-gemma-4-31b-it-letters",
    pad_to="none",
    limit_mm={"image": 0, "video": 0},
    context_ceiling=98304,
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
    # before: registering after the response cost the 31B 39% and 26% of its throughput at 32 and 64 clients on new
    # 1,000-token states (p95 1.69 and 1.62 times), past the rule's 95% and 110%; on the 12B it held (0.99 and 0.98)
    # (runs/2026-10-06_latency-585w)
    register_boundary="before",
    # warm: four questions in one batch after the state's prefill, repeating exactly in-process; gated against
    # sequential (runs/2026-10-05_engine-death-gates: no choice changed, the largest probability difference 0.0073)
    multi_question="warm",
    mlx_working_gb=9,
)

FAMILIES = (QWEN, GEMMA4, GEMMA4_FP8, GEMMA4_31B)
BASES = {f.key: f for f in FAMILIES}
# the base a vLLM server serves when neither --base nor --model is given (0.10.0; before, the server refused to start
# and the container served the Qwen base); a --model alone still brings the base its checkpoint declares
DEFAULT_BASE = "gemma-4-31b"


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


def declares_fp8(config: dict) -> bool:
    """Whether a checkpoint's config.json declares FP8 weights: a `quantization_config` (at the top or in `text_config`)
    whose method is an FP8 one, or compressed-tensors with an 8-bit float scheme. An MLX conversion declares its own
    `quantization` (bits, group size) instead."""
    for c in (config, config.get("text_config") or {}):
        q = c.get("quantization_config") or {}
        method = str(q.get("quant_method", "")).lower()
        if "fp8" in method:
            return True
        if method == "compressed-tensors":
            for group in (q.get("config_groups") or {}).values():
                w = group.get("weights") or {}
                if w.get("type") == "float" and w.get("num_bits") == 8:
                    return True
    return False


def family_of(model: str | None, revision: str | None = None, lenient: bool = False) -> Family:
    """The base a checkpoint (a local directory or a hub id) belongs to. No checkpoint named is the default base
    (Qwen). A checkpoint whose config.json names a model type no base serves, or has no readable config.json, is
    refused with the supported bases named: it used to be given the Qwen profile silently, whose classes, token
    ranges and temperatures were fitted for another model. `--base <key>` with `--model` is the explicit way to serve a
    checkpoint as a base's profile. `lenient` (the CPU stand-in, which serves a small Qwen3 model to develop the routes
    and is not a measured system) keeps the old answer for them, the Qwen profile. DECISIO_BASE, when set to a base's
    key, overrides the detection (tokenizer-only tests)."""
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
    found = [f for f in FAMILIES if mt in f.model_types and (f.moe is None or f.moe == moe)]
    if len(found) > 1:  # one checkpoint in two precisions (the 12B): the one the checkpoint declares, bf16 unless FP8
        precision = "fp8" if declares_fp8(cfg) else "bf16"
        found = [f for f in found if f.precision == precision]
    found = found[0] if found else None
    if found is None and lenient:
        return QWEN
    if found is None:
        what = f"its config.json names the model type {mt!r}" if mt else "its config.json could not be read"
        raise ValueError(
            f"--model {model}: {what}, which no supported base serves (the bases: {', '.join(sorted(BASES))}). "
            "Give --base <key> with --model to serve a checkpoint as that base's profile, if it is one."
        )
    return as_loaded(found, cfg)


def as_loaded(fam: Family, config: dict) -> Family:
    """The profile a checkpoint is loaded under: a base that quantizes to FP8 on load does not quantize a checkpoint
    whose config.json already declares FP8 weights (a stored conversion, `python -m decisio.hub_fp8 convert`); vLLM
    loads it as stored. Everything else is the base's."""
    if fam.quantization == "fp8" and declares_fp8(config):
        return replace(fam, quantization=None)
    return fam
