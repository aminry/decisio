# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The 12B in FP8 (`--base gemma-4-12b-fp8`, decisio.families), a second precision beside the bf16 12B, without a model:

P1  detection without --base: the 12B's checkpoint is the bf16 base unless its config.json declares FP8 weights; a
    stored FP8 conversion is the FP8 base, loaded as stored (not quantized again); the 31B's stored FP8 is loaded as
    stored too
P2  the command line: --base gemma-4-12b-fp8 alone serves Google's bf16 checkpoint at its pin, quantized on load; with
    --model a stored conversion, as stored; --base gemma-4-12b with an FP8 checkpoint is refused naming the FP8 base;
    MLX refuses the FP8 base
P3  the start-up line says the base and the precision, for every way of starting
P4  the FP8 base is its own stack: its own key, served name and task fingerprint; the bf16 12B's profile and
    temperatures otherwise (until they are refitted on FP8 reads)
P5  the engine asks vLLM for FP8 on load only where the checkpoint is not FP8 already

  uv run pytest -q tests/unit/test_12b_fp8.py
"""

import dataclasses
import json
import sys
import types
from types import SimpleNamespace

import pytest

from decisio import hub
from decisio.families import BASES, FAMILIES, GEMMA4, GEMMA4_31B, GEMMA4_FP8, as_loaded, declares_fp8, family_of
from decisio.serve import vllm_engine as sv

# what `python -m decisio.hub_fp8 convert` writes into config.json (compressed-tensors, FP8 weights per tensor)
FP8_QUANT = {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "tensor"}}},
}


def checkpoint(tmp_path, name, model_type="gemma4_unified", fp8=False):
    d = tmp_path / name
    d.mkdir()
    cfg = {"model_type": model_type, "text_config": {"enable_moe_block": False}}
    if fp8:
        cfg["quantization_config"] = FP8_QUANT
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def args(**kw):
    a = dict(
        backend="vllm", base=None, model=None, revision=None, prompt_tail=None, answer_slot=None,
        label_variants=None, system_prompt=None, pad_to=None, served_name=None, noul_rendering=None,
        temperature=None, temperature_choice=None,
    )  # fmt: skip
    return SimpleNamespace(**{**a, **kw})


@pytest.fixture(autouse=True)
def no_forced_base(monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)


def test_p1_detection_keeps_bf16_unless_the_checkpoint_declares_fp8(tmp_path):
    bf16, fp8 = checkpoint(tmp_path, "bf16"), checkpoint(tmp_path, "fp8", fp8=True)
    assert family_of(bf16) is GEMMA4  # the default precision: every 12B record is bf16
    got = family_of(fp8)
    assert got.key == "gemma-4-12b-fp8" and got.precision == "fp8"
    assert got.quantization is None  # a stored conversion is loaded as stored
    assert got == dataclasses.replace(GEMMA4_FP8, quantization=None)
    # the 31B: one precision; a stored FP8 checkpoint is not quantized again on load either
    g31 = checkpoint(tmp_path, "g31", model_type="gemma4", fp8=True)
    assert family_of(g31).key == "gemma-4-31b" and family_of(g31).quantization is None
    assert family_of(checkpoint(tmp_path, "g31bf16", model_type="gemma4")) is GEMMA4_31B
    # the rule in one place
    assert declares_fp8({"quantization_config": FP8_QUANT}) and not declares_fp8({})
    assert as_loaded(GEMMA4, {"quantization_config": FP8_QUANT}) is GEMMA4  # bf16 is never changed by it
    # exactly two bases share the 12B's checkpoint, one per precision
    twelve = [f for f in FAMILIES if f.model == GEMMA4.model]
    assert sorted(f.precision for f in twelve) == ["bf16", "fp8"]


def test_p2_the_command_line(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "decisio.families.read_config",
        lambda model, revision=None: calls.append(model) or {"model_type": "gemma4_unified"},
    )
    a = args(base="gemma-4-12b-fp8")  # alone: Google's bf16 checkpoint at its pin, quantized to FP8 on load
    fam, fmt = sv.resolve_base(a)
    assert fam is GEMMA4_FP8 and fam.quantization == "fp8"
    assert (a.model, a.revision) == (GEMMA4.model, GEMMA4.revision)
    assert a.served_name == "decisio-gemma-4-12b-it-fp8-letters" and a.temperature == GEMMA4.temperature
    assert fmt == sv.resolve_base(args(base="gemma-4-12b"))[1]  # the 12B's prompt
    monkeypatch.undo()
    monkeypatch.delenv("DECISIO_BASE", raising=False)

    stored, bf16 = checkpoint(tmp_path, "stored", fp8=True), checkpoint(tmp_path, "bf16")
    fam, _ = sv.resolve_base(args(base="gemma-4-12b-fp8", model=stored))
    assert fam.key == "gemma-4-12b-fp8" and fam.quantization is None
    fam, _ = sv.resolve_base(args(base="gemma-4-12b-fp8", model=bf16))  # a bf16 copy: quantized on load
    assert fam is GEMMA4_FP8
    with pytest.raises(ValueError, match="declares FP8 weights: serve it with --base gemma-4-12b-fp8"):
        sv.resolve_base(args(base="gemma-4-12b", model=stored))
    assert sv.resolve_base(args(base="gemma-4-12b", model=bf16))[0] is GEMMA4
    # detection without --base, as P1
    assert sv.resolve_base(args(model=stored))[0].key == "gemma-4-12b-fp8"
    assert sv.resolve_base(args(model=bf16))[0] is GEMMA4
    # another base's checkpoint is still refused under the FP8 base
    qwen = checkpoint(tmp_path, "qwen", model_type="qwen3_5_moe")
    with pytest.raises(ValueError, match="is a qwen3.6-35b-a3b checkpoint"):
        sv.resolve_base(args(base="gemma-4-12b-fp8", model=qwen))
    with pytest.raises(ValueError, match="--backend mlx does not serve gemma-4-12b-fp8"):
        sv.resolve_base(args(base="gemma-4-12b-fp8", model=bf16, backend="mlx"))


def test_p3_the_start_up_line_says_the_precision():
    stored = dataclasses.replace(GEMMA4_FP8, quantization=None)
    note = "; temperatures 3.592 fitted on the bf16 12B's reads, not refitted on FP8"
    assert sv.base_line(GEMMA4_FP8, GEMMA4.model) == (
        "BASE gemma-4-12b-fp8 (fp8, quantized to FP8 on load) from google/gemma-4-12B-it" + note
    )
    assert sv.base_line(stored, "/c/fp8") == (
        "BASE gemma-4-12b-fp8 (fp8, stored FP8, loaded as stored) from /c/fp8" + note
    )
    assert sv.base_line(GEMMA4, GEMMA4.model) == (
        "BASE gemma-4-12b (bf16, as the checkpoint stores it) from google/gemma-4-12B-it"
    )
    assert sv.base_line(GEMMA4_31B, "x").startswith("BASE gemma-4-31b (quantized to FP8 on load)")
    assert sv.base_line(GEMMA4, "/m", backend="mlx") == "BASE gemma-4-12b (bf16, as mlx loads it) from /m"
    # a tuned repository's own temperatures: no note
    assert "temperatures" not in sv.base_line(stored, "/t", tuned=True)


def test_p4_its_own_stack_otherwise_the_12b():
    fp8, bf16 = hub.profile_of(GEMMA4_FP8), hub.profile_of(GEMMA4)
    assert fp8["prompt"]["fingerprint"]["base"] == "gemma-4-12b-fp8"  # tasks fitted on bf16 are not applied to it
    assert fp8["prompt"]["fingerprint"] != bf16["prompt"]["fingerprint"]
    assert fp8["serving"]["served_name"] != bf16["serving"]["served_name"]
    same = {k: v for k, v in fp8.items() if k not in ("prompt", "serving")}
    assert same == {k: v for k, v in bf16.items() if k not in ("prompt", "serving")}  # temperatures included
    assert fp8["prompt"]["format"] == bf16["prompt"]["format"]
    differ = [f.name for f in dataclasses.fields(GEMMA4) if getattr(GEMMA4, f.name) != getattr(GEMMA4_FP8, f.name)]
    assert sorted(differ) == ["key", "precision", "quantization", "served_name"]
    assert BASES["gemma-4-12b-fp8"] is GEMMA4_FP8
    w = hub.weights_of("gemma-4-12b-fp8")
    assert (w["dtype"], w["quantization"], w["quantized_on_load"]) == ("fp8_e4m3", "compressed-tensors", False)
    assert hub.REPOSITORIES["gemma-4-12b-fp8"] == "tachara-ai/decisio-gemma-4-12b-fp8"
    assert "gemma-4-12b-fp8" not in hub.DEFAULT_SOURCES  # the bare key quantizes Google's weights on load


class FakeLLM:
    kw = None

    def __init__(self, **kw):
        FakeLLM.kw = kw
        model_config = types.SimpleNamespace(max_model_len=kw["max_model_len"])
        cache_config = types.SimpleNamespace(block_size=16, prefix_match_unit=None)
        self.llm_engine = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(model_config=model_config, cache_config=cache_config)
        )

    def get_tokenizer(self):
        return object()

    def collective_rpc(self, name):
        return [None]


def test_p5_the_engine_quantizes_on_load_only_a_bf16_checkpoint(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr(sv.LettersEngine, "_warm_up", lambda self: None)
    sv.LettersEngine("stand-in", family=GEMMA4_FP8, max_model_len=32768)
    assert FakeLLM.kw["quantization"] == "fp8"
    sv.LettersEngine("stand-in", family=dataclasses.replace(GEMMA4_FP8, quantization=None), max_model_len=32768)
    assert "quantization" not in FakeLLM.kw
    sv.LettersEngine("stand-in", family=GEMMA4, max_model_len=32768)
    assert "quantization" not in FakeLLM.kw


def test_p6_the_31b_with_a_local_fp8_directory_is_loaded_as_stored_and_its_default_is_unchanged(tmp_path, monkeypatch):
    """as_loaded also covers the 31B: `--model <a stored FP8 directory>` is not handed to vLLM with quantization=fp8
    again (vLLM 0.31.0 refuses a compressed-tensors checkpoint with that argument); a bf16 checkpoint keeps quantizing
    on load; the default start (its pinned repository) is what it was: no quantization on load."""
    assert as_loaded(GEMMA4_31B, {"quantization_config": FP8_QUANT}).quantization is None
    assert as_loaded(GEMMA4_31B, {"model_type": "gemma4"}) is GEMMA4_31B and GEMMA4_31B.quantization == "fp8"
    stored = checkpoint(tmp_path, "s31", model_type="gemma4", fp8=True)
    assert sv.resolve_base(args(base="gemma-4-31b", model=stored))[0].quantization is None
    g31 = tmp_path / "repo31"
    g31.mkdir()
    (g31 / hub.FILE).write_text(json.dumps(hub.export_decision_config(GEMMA4_31B, "0.10.0")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda name, filename, revision=None: str(g31 / filename))
    a = args(base="gemma-4-31b")
    fam, _ = sv.resolve_base(a)
    assert fam.quantization is None and a.repository["name"] == hub.REPOSITORIES["gemma-4-31b"]
    assert sv.base_line(fam, a.model) == f"BASE gemma-4-31b (as the checkpoint stores it) from {a.model}"


def test_p7_the_vision_embedder_is_left_unquantized_on_the_12b_only():
    """vLLM builds the 12B's vision_embedder.patch_dense with the checkpoint's quantization config, so a 12B conversion
    names it in the ignore list; a 31B conversion writes the list its published repository holds."""
    from decisio import hub_fp8

    twelve = hub_fp8.quantization_config("compressed-tensors", hub_fp8.ignored_for({"model_type": "gemma4_unified"}))
    thirty_one = hub_fp8.quantization_config("compressed-tensors", hub_fp8.ignored_for({"model_type": "gemma4"}))
    assert twelve["ignore"] == hub_fp8.IGNORED + ["re:.*vision_embedder.*"]
    assert (
        thirty_one["ignore"]
        == hub_fp8.IGNORED
        == [
            "lm_head",
            "re:.*vision_tower.*",
            "re:.*audio_tower.*",
            "re:.*embed_vision.*",
            "re:.*embed_audio.*",
        ]
    )
    assert hub_fp8.quantization_config("fp8")["modules_to_not_convert"] == hub_fp8.IGNORED
