# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The small base, Qwen/Qwen3.5-4B (`qwen3.5-4b`), and the Gemma 4 E4B migration.

S1  detection: Gemma 4 12B in bf16 and FP8, Gemma 4 31B, Qwen MoE, and Qwen 3.5 4B resolve from their config.json
S2  Gemma 4 E4B is refused with a migration hint, even if requested under the 31B profile; other sizes of these
    model types (Gemma 4 E2B, other Qwen3.5 sizes) are also refused
S3  `--base` with a checkpoint of the other kind is refused both ways; with its own checkpoint it starts
S4  Qwen 3.5 4B uses the 12B's prompt, vLLM, a pinned revision, temperature 1.0, and its own served name
S5  with Qwen's cached tokenizer: the chat template renders an empty think block and two forms per answer letter
S6  Qwen's real cached config.json resolves to its base

  uv run pytest -q tests/unit/test_small_bases.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisio.families import BASES, FAMILIES, QWEN35_4B, family_of
from decisio.serve.vllm_engine import resolve_base

# the fields detection reads, as in the real config.json files (top level and text_config)
REAL = {
    "gemma-4-12b": {"model_type": "gemma4_unified", "text_config": {"hidden_size": 3840, "enable_moe_block": False,
                                                                    "hidden_size_per_layer_input": 0}},
    "gemma-4-31b": {"model_type": "gemma4", "text_config": {"hidden_size": 5376, "enable_moe_block": False,
                                                            "hidden_size_per_layer_input": 0}},
    "qwen3.6-35b-a3b": {"model_type": "qwen3_5_moe", "text_config": {"hidden_size": 2048}},
    "qwen3.5-4b": {"model_type": "qwen3_5", "text_config": {"hidden_size": 2560}},
    "gemma-4-e4b": {"model_type": "gemma4", "text_config": {"hidden_size": 2560, "enable_moe_block": False,
                                                            "hidden_size_per_layer_input": 256}},
}  # fmt: skip
FP8_12B = {
    **REAL["gemma-4-12b"],
    "quantization_config": {"quant_method": "fp8"},
}


def checkpoint(tmp_path, config, name):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return str(d)


@pytest.fixture(autouse=True)
def no_forced_base(monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)


def args_for(**kw):
    a = dict(
        backend="vllm", base=None, model=None, revision=None, prompt_tail=None, answer_slot=None,
        label_variants=None, system_prompt=None, pad_to=None, served_name=None, noul_rendering=None,
        temperature=None, temperature_choice=None,
    )  # fmt: skip
    return SimpleNamespace(**{**a, **kw})


@pytest.mark.parametrize(
    ("base", "with_model", "backend"),
    [
        ("gemma-4-e4b", False, "vllm"),
        (None, True, "vllm"),
        (None, True, "hf"),
        ("gemma-4-31b", True, "vllm"),
    ],
    ids=["base-key", "inferred", "cpu-stand-in", "explicit-other-profile"],
)
def test_removed_e4b_is_refused_with_migration_hint(tmp_path, base, with_model, backend):
    model = checkpoint(tmp_path, REAL["gemma-4-e4b"], "local-checkpoint") if with_model else None
    with pytest.raises(ValueError, match=r"Gemma 4 E4B.*qwen3\.5-4b"):
        resolve_base(args_for(base=base, model=model, backend=backend))


def test_removed_e4b_repository_is_refused_with_migration_hint(tmp_path):
    repository = {"schema": "decisio.decision_config/1", "base": "gemma-4-e4b"}
    (tmp_path / "decision_config.json").write_text(json.dumps(repository))
    with pytest.raises(ValueError, match=r"Gemma 4 E4B.*qwen3\.5-4b"):
        resolve_base(args_for(base=str(tmp_path)))


@pytest.mark.parametrize("key", ["gemma-4-12b", "gemma-4-31b", "qwen3.6-35b-a3b", "qwen3.5-4b"])
def test_s1_every_base_is_detected_from_its_config(tmp_path, key):
    assert family_of(checkpoint(tmp_path, REAL[key], key)).key == key


def test_s1_the_bases_served_before_are_detected_exactly_as_before(tmp_path):
    assert family_of(checkpoint(tmp_path, REAL["gemma-4-12b"], "g12")).key == "gemma-4-12b"
    assert family_of(checkpoint(tmp_path, FP8_12B, "g12fp8")).key == "gemma-4-12b-fp8"
    assert family_of(checkpoint(tmp_path, REAL["gemma-4-31b"], "g31")).key == "gemma-4-31b"
    assert family_of(checkpoint(tmp_path, REAL["qwen3.6-35b-a3b"], "q35")).key == "qwen3.6-35b-a3b"
    # the bare forms the earlier tests use (no text_config, no sizes): as before
    assert family_of(checkpoint(tmp_path, {"model_type": "gemma4"}, "bare31")).key == "gemma-4-31b"
    assert family_of(checkpoint(tmp_path, {"model_type": "qwen3_5_moe"}, "bareq")).key == "qwen3.6-35b-a3b"
    assert family_of(checkpoint(tmp_path, {"model_type": "gemma4_unified"}, "bare12")).key == "gemma-4-12b"


@pytest.mark.parametrize("ple", [0, None, "absent"])
def test_s2_no_per_layer_inputs_is_the_31b(tmp_path, ple):
    text = {"hidden_size": 2560, "enable_moe_block": False}
    if ple != "absent":
        text["hidden_size_per_layer_input"] = ple
    # even at the E4B's hidden size: without per-layer inputs a gemma4 checkpoint is never the E4B
    assert family_of(checkpoint(tmp_path, {"model_type": "gemma4", "text_config": text}, "c")).key == "gemma-4-31b"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "gemma4", "text_config": {"hidden_size": 1536, "hidden_size_per_layer_input": 256}},  # E2B
        {"model_type": "qwen3_5", "text_config": {"hidden_size": 1024}},  # Qwen3.5-0.8B
        {"model_type": "qwen3_5", "text_config": {"hidden_size": 5120}},  # a larger dense Qwen3.5
    ],
)
def test_s2_other_sizes_are_refused(tmp_path, config):
    with pytest.raises(ValueError, match="in a size or variant"):
        family_of(checkpoint(tmp_path, config, "other"))


def test_s3_base_with_the_other_kind_of_checkpoint_is_refused_both_ways(tmp_path):
    q4 = checkpoint(tmp_path, REAL["qwen3.5-4b"], "q4")
    q35 = checkpoint(tmp_path, REAL["qwen3.6-35b-a3b"], "q35")
    with pytest.raises(ValueError, match="is a qwen3.6-35b-a3b checkpoint"):
        resolve_base(args_for(base="qwen3.5-4b", model=q35))
    with pytest.raises(ValueError, match="is a qwen3.5-4b checkpoint"):
        resolve_base(args_for(base="qwen3.6-35b-a3b", model=q4))


@pytest.mark.parametrize("key", ["qwen3.5-4b", "gemma-4-31b", "gemma-4-12b"])
def test_s3_base_with_its_own_checkpoint_resolves(tmp_path, key):
    model = checkpoint(tmp_path, REAL[key], key)
    fam, _ = resolve_base(args_for(base=key, model=model))[:2]
    assert fam.key == key


def test_s4_the_small_profile():
    fam = QWEN35_4B
    assert (fam.system_prompt, fam.layout, fam.answer_slot, fam.prompt_tail) == (True, "header", "template", "spaced")
    assert (fam.label_variants, fam.noul_rendering) == ("summed", "letters")
    assert fam.vllm_only and fam.quantization is None and fam.pad_to == "none"
    assert fam.temperature == 1.0 and fam.choice_temperature is None and not fam.temperatures
    assert len(fam.revision) == 40 and BASES[fam.key] is fam
    assert fam.classes["text-only"] and fam.classes["hidden-readout"]
    assert [f.served_name for f in FAMILIES].count(fam.served_name) == 1


def test_s4_the_revision_is_the_measured_one():
    assert QWEN35_4B.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _tokenizer(repo, revision):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
    except Exception as e:
        pytest.skip(f"{repo} is not cached: {e}")


def _fmt(fam):
    from decisio.readout.letters import PromptFormat

    return PromptFormat(
        tail=fam.prompt_tail, slot=fam.answer_slot, variants=fam.label_variants, system_prompt=fam.system_prompt,
        layout=fam.layout,
    )  # fmt: skip


def test_s5_own_template_with_thinking_off_and_the_label_forms():
    from decisio.readout.letters import chat_turn, label_groups

    fam = QWEN35_4B
    tok = _tokenizer(fam.model, fam.revision)
    text = chat_turn(tok, "STATE AND QUESTION", _fmt(fam))
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "<|channel>thought" not in text
    assert "You are a calibration engine." in text  # the system turn
    groups = label_groups(tok, [" A", " B", " C"], _fmt(fam))
    assert [len(g) for g in groups] == [2, 2, 2]


def test_s6_the_real_cached_config_is_detected():
    fam = QWEN35_4B
    snap = Path.home() / ".cache/huggingface/hub" / ("models--" + fam.model.replace("/", "--")) / "snapshots"
    cfg = snap / fam.revision / "config.json"
    if not cfg.is_file():
        pytest.skip(f"{fam.model} at {fam.revision[:8]} is not cached")
    assert family_of(str(cfg.parent)) is fam
