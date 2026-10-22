# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A tuned 12B in FP8 (`gemma-4-12b-fp8`): the `/2` schema for the 12B's second precision, and `hub_fp8 merge` on a
tiny checkpoint with the 12B's layout (no model, no card).

D1  the file: FP8 weights (fp8_e4m3, compressed-tensors, not quantized on load) made by `hub_fp8 merge` (fp32, quantized
    once), the FP8 12B's profile unchanged, the 12B's checkpoint as the tuned base; the bf16 12B's and the 31B's weights
    sections are unchanged; the three tuned bases are listed
D2  mixed facts are refused, each naming the base they belong to: a bf16 merge report (output bf16, stochastic
    rounding, a seed) under the FP8 12B, an FP8 merge report under the bf16 12B, either weights dtype under the other
    base, the bf16 12B's profile under the FP8 key
D3  the 12B's layout in miniature (gemma4_unified, attention_k_eq_v, sliding layers with q, k and v and full-attention
    layers with a wider q and no v_proj, the vision embedder and the two embedding projections, two shards): `hub_fp8
    merge` fuses what vLLM fuses ([q; k; k] on the full layers), merges in fp32 and quantizes once (the scale of each
    group is the merged group's amax / 448 and every byte is the fp32 merge quantized with it), leaves out the adapter's
    v_proj copies of k_proj pairs on the full layers and counts them, refuses a v_proj pair that differs, copies every
    other tensor byte for byte, predicts no weight changed by the loader, and writes a checkpoint detected as the FP8
    12B
D4  on that checkpoint `merge --null` equals `convert` byte for byte (every weight file, the index, the config and the
    tokenizer files), and the conversion is what quantizing on load holds (the 12B's identity claim, in miniature)
D5  `tuned-config --base gemma-4-12b-fp8` from the real merge report writes a file `check --weights-dir` accepts, and
    finds a changed file; the bf16 12B refuses the FP8 directory and the FP8 12B the bf16 one, at `tuned-config` and at
    `check`
D6  served from the directory: the FP8 12B loaded as stored, its own identity (not the bf16 12B's with the same
    defaults); a repository that says FP8 12B over a checkpoint that declares no FP8 is refused, as is the bf16 12B's
    file over an FP8 checkpoint

  uv run pytest -q tests/unit/test_hub_tuned_12b_fp8.py
"""

import hashlib
import json
import os
import re

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_hub import args_for
from test_hub_fp8 import bf16
from test_hub_fp8_merge import adapter, read_dir, targets
from test_hub_tuned import ADAPTER_SHA, BASIS, CONFIG_SHA, TEMPS
from test_hub_tuned_12b import MERGE as MERGE_BF16
from test_hub_tuned_12b import build12, write12

from decisio import hub, hub_bf16, hub_fp8
from decisio.families import BASES, declares_fp8, family_of
from decisio.serve.vllm_engine import base_line, resolve_base

FAM = BASES["gemma-4-12b-fp8"]
L = "model.language_model.layers"
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
FULL = [i for i, k in enumerate(LAYER_TYPES) if k == "full_attention"]
MERGE = {
    "precision": "fp32",
    "output": "fp8 compressed-tensors",
    "tool": "decisio.hub_fp8 merge",
    "adapter_sha256": ADAPTER_SHA,
    "adapted_tensors": 336,
    "dropped_k_eq_v_pairs": 8,
}
# not part of the comparison of a null merge with `convert`: the reports each writes about itself
REPORTS = {"fp8_conversion.json", "MERGE_FP8.json", "expected_written.json", "expected_loaded.json"}


@pytest.fixture(autouse=True)
def no_forced_base(monkeypatch):
    monkeypatch.delenv("DECISIO_BASE", raising=False)


def build12fp8(**over):
    kw = {
        "temperatures": dict(TEMPS["temperatures"]),
        "choice_bands": dict(TEMPS["choice_bands"]),
        "noul_commit": True,
        "noul_commit_basis": BASIS,
        "fit": {"method": "minimum log loss"},
        "adapter": {
            "repository": "tachara-ai/adapter-12b",
            "revision": "abc",
            "weights_sha256": ADAPTER_SHA,
            "config_sha256": CONFIG_SHA,
            "rank": 16,
            "alpha": 32,
        },
        "merge": dict(MERGE),
        "files": {"model-00001-of-00004.safetensors": hashlib.sha256(b"w").hexdigest()},
        "decisio_version": "0.23.0",
    }
    kw.update(over)
    return hub.tuned_config(FAM, **kw)


def checkpoint12(tmp_path, name="bf16-12b"):
    """The 12B's layout in miniature (google/gemma-4-12B-it: 48 layers, every sixth full attention with one KV head
    of twice the width and no v_proj): two sliding layers with q, k and v, two full-attention layers with a wider q, k
    and o and no v_proj; q and k norms, the four layer norms and the layer scalar; the embedding, the final norm, the
    vision embedder and the vision and audio embedding projections (the 12B has no tower weights); two shards and an
    index; the tokenizer, chat template, generation and processor files."""
    t = {}
    for i, kind in enumerate(LAYER_TYPES):
        full, p, s = kind == "full_attention", f"{L}.{i}", 10 * i
        head = 8 if full else 4
        t[f"{p}.self_attn.q_proj.weight"] = bf16(2 * head, 6, seed=s + 1)
        t[f"{p}.self_attn.k_proj.weight"] = bf16(head, 6, seed=s + 2)
        if not full:
            t[f"{p}.self_attn.v_proj.weight"] = bf16(head, 6, seed=s + 3)
        t[f"{p}.self_attn.o_proj.weight"] = bf16(6, 2 * head, seed=s + 4)
        t[f"{p}.mlp.gate_proj.weight"] = bf16(10, 6, seed=s + 5)
        t[f"{p}.mlp.up_proj.weight"] = bf16(10, 6, seed=s + 6)
        t[f"{p}.mlp.down_proj.weight"] = bf16(6, 10, seed=s + 7)
        t[f"{p}.self_attn.q_norm.weight"] = bf16(head, scale=1.0, seed=s + 8)
        t[f"{p}.self_attn.k_norm.weight"] = bf16(head, scale=1.0, seed=s + 9)
        for norm in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            t[f"{p}.{norm}_layernorm.weight"] = torch.ones(6, dtype=torch.bfloat16)
        t[f"{p}.layer_scalar"] = torch.ones(1, dtype=torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = bf16(20, 6, seed=90)
    t["model.language_model.norm.weight"] = torch.ones(6, dtype=torch.bfloat16)
    t["model.vision_embedder.patch_dense.weight"] = bf16(5, 12, seed=91)
    t["model.vision_embedder.patch_dense.bias"] = bf16(5, seed=92)
    t["model.embed_vision.embedding_projection.weight"] = bf16(6, 5, seed=93)
    t["model.embed_audio.embedding_projection.weight"] = bf16(6, 4, seed=94)
    d = tmp_path / name
    d.mkdir()
    names = sorted(t)
    weight_map = {}
    for n, part in enumerate((names[: len(names) // 2], names[len(names) // 2 :]), start=1):
        f = f"model-0000{n}-of-00002.safetensors"
        save_file({k: t[k] for k in part}, str(d / f), metadata={"format": "pt"})
        weight_map.update({k: f for k in part})
    (d / hub_fp8.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    cfg = {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "text_config": {
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "num_hidden_layers": len(LAYER_TYPES),
            "layer_types": LAYER_TYPES,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
        },
        "vision_config": {"hidden_size": 5},
        "audio_config": {"hidden_size": 4},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text('{"tokenizer": true}')
    (d / "tokenizer_config.json").write_text('{"model_max_length": 131072}')
    (d / "chat_template.jinja").write_text("{{ messages }}")
    (d / "generation_config.json").write_text('{"do_sample": false}')
    (d / "processor_config.json").write_text('{"processor_class": "Gemma4Processor"}')
    return d, t


def with_k_copies(tmp_path, t, seed=0, differ=False):
    """A text-layout adapter on every projection of the text model, plus a v_proj pair on each full-attention layer
    that is bit for bit the layer's k_proj pair (what the trainer's export writes for a K=V layer), or (differ) is not
    on the last one."""
    d = adapter(tmp_path, targets(t), seed=seed)
    tensors = load_file(str(d / "adapter_model.safetensors"))
    for i in FULL:
        for half in ("A", "B"):
            k = tensors[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_{half}.weight"].clone()
            if differ and i == FULL[-1]:
                k = k + 0.001
            tensors[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_{half}.weight"] = k
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "adapter_model.safetensors"))
    return d


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d1_the_file_is_fp8_merged_in_fp32_and_quantized_once():
    c = build12fp8()
    assert c["schema"] == hub.SCHEMA_TUNED and c["base"] == "gemma-4-12b-fp8"
    assert hub.check_profile(c, FAM) == []
    w = c["weights"]
    assert (w["modified"], w["dtype"], w["quantization"], w["quantized_on_load"]) == (
        True,
        "fp8_e4m3",
        "compressed-tensors",
        False,
    )
    assert "12B's bf16 base in fp32 and quantized once to FP8" in w["method"] and "hub_fp8 merge" in w["method"]
    assert c["tuned"]["base"] == {"repository": "google/gemma-4-12B-it", "revision": FAM.revision}
    assert c["tuned"]["merge"]["dropped_k_eq_v_pairs"] == 8
    assert hub.weights_of_tuned("gemma-4-12b-fp8", "fp8")["quantization"] == "fp8"  # vLLM's own format, if chosen
    # the other two bases' sections are what they were
    assert hub.weights_of_tuned("gemma-4-12b")["dtype"] == "bfloat16"
    assert hub.weights_of_tuned("gemma-4-31b")["method"].startswith("a LoRA adapter merged into the bf16 base in fp32")
    assert hub.TUNED_BASES == ("gemma-4-31b", "gemma-4-12b", "gemma-4-12b-fp8")


@pytest.mark.parametrize(
    ("make", "mutate", "message"),
    [
        (build12fp8, lambda c: c["tuned"].update(merge=dict(MERGE_BF16)), "a bf16 merge of the 12B"),
        (build12fp8, lambda c: c["tuned"]["merge"].update(output="bf16"), "is the base gemma-4-12b"),
        (build12fp8, lambda c: c["tuned"]["merge"].update(rounding="stochastic"), "no stochastic rounding or seed"),
        (build12fp8, lambda c: c["tuned"]["merge"].update(seed=7), "no stochastic rounding or seed"),
        (build12, lambda c: c["tuned"].update(merge=dict(MERGE)), "a tuned 12B in FP8 is the base gemma-4-12b-fp8"),
        (build12fp8, lambda c: c["weights"].update(dtype="bfloat16"), "a tuned gemma-4-12b-fp8 is stored as fp8_e4m3"),
        (build12, lambda c: c["weights"].update(dtype="fp8_e4m3"), "a tuned gemma-4-12b is stored as bfloat16"),
    ],
)
def test_d2_mixed_facts_are_refused_naming_the_other_base(make, mutate, message):
    c = make()
    mutate(c)
    with pytest.raises(ValueError, match=re.escape(message)):
        hub.check_sections(c)


def test_d2_the_bf16_12bs_profile_is_not_the_fp8_12bs():
    c = build12fp8()
    bf16_profile = hub.profile_of(BASES["gemma-4-12b"])
    c["prompt"], c["serving"] = bf16_profile["prompt"], bf16_profile["serving"]
    assert hub.check_profile(c, FAM) == [
        "prompt.fingerprint.base",
        "prompt.fingerprint.served_name",
        "serving.served_name",
    ]


def test_d3_the_merge_on_the_12bs_layout(tmp_path):
    base, t = checkpoint12(tmp_path)
    ad = with_k_copies(tmp_path, t, seed=3)
    out = tmp_path / "merged"
    meta = hub_fp8.merge(base, ad, out)
    # the groups vLLM fuses: [q; k; k] on the full-attention layers, which ship no v_proj
    groups = {g.key: g for g in hub_fp8.find_groups(t, k_eq_v=True)}
    assert len(groups) == 4 * len(LAYER_TYPES) == meta["groups"]
    for i in FULL:
        assert groups[f"layers.{i}.self_attn.qkv_proj"].fused == [f"{L}.{i}.self_attn.{p}_proj.weight" for p in "qkk"]
    assert meta["adapted_tensors"] == len(targets(t)) and meta["dropped_k_eq_v_pairs"] == len(FULL)
    assert meta["loader_changed_elements"] == 0 and meta["written_vs_loaded_differ"] == []
    assert meta["null"] is False and meta["adapter_layout"] == "text" and meta["adapter_scale"] == 2.0
    merged = read_dir(out)
    assert not any(f"{L}.{i}.self_attn.v_proj.weight" in merged for i in FULL)
    # fp32, then one quantization: each group's scale is the merged group's amax / 448, and every byte is the fp32
    # merge quantized with it (no bf16 between)
    deltas, scale, _, _, _ = hub_fp8.load_adapter(ad, hub_fp8.Source(base).map)
    for g in groups.values():
        w32 = {n: hub_fp8.merged32(t[n], deltas.get(n), scale, "cpu") for n in dict.fromkeys(g.fused)}
        amax = max(float(x.abs().max()) for x in w32.values())
        for n in g.names:
            sc = merged[n[: -len(".weight")] + ".weight_scale"]
            assert float(sc) == pytest.approx(amax / 448, rel=1e-6), n
            assert torch.equal(merged[n].view(torch.uint8), hub_fp8.quantize(w32[n], sc).view(torch.uint8)), n
    # the adapter is in the weights: the merge differs from the base's own conversion
    hub_fp8.convert(base, tmp_path / "converted")
    plain = read_dir(tmp_path / "converted")
    names = [n for g in groups.values() for n in g.names]
    assert any(not torch.equal(plain[n].view(torch.uint8), merged[n].view(torch.uint8)) for n in names)
    # every other tensor, the vision embedder and the embedding projections included, is the source's
    quantized = {n for g in groups.values() for n in g.names}
    for n in set(t) - quantized:
        assert merged[n].dtype == t[n].dtype and torch.equal(merged[n], t[n]), n
    # the checkpoint declares FP8 and is detected as the FP8 12B, loaded as stored
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "gemma4_unified" and cfg["text_config"]["attention_k_eq_v"] is True
    assert declares_fp8(cfg) and cfg["quantization_config"]["quant_method"] == "compressed-tensors"
    got = family_of(str(out))
    assert got.key == "gemma-4-12b-fp8" and got.quantization is None
    assert hub_fp8.compare(json.loads((out / "expected_written.json").read_text()), hub_fp8.expected_loaded(out)) == []


def test_d3_a_v_proj_pair_that_differs_from_k_is_refused(tmp_path):
    base, t = checkpoint12(tmp_path)
    other = with_k_copies(tmp_path, t, seed=4, differ=True)
    with pytest.raises(ValueError, match=rf"1 adapter pairs .*layers\.{FULL[-1]}\.self_attn\.v_proj: A differs"):
        hub_fp8.merge(base, other, tmp_path / "refused")


def test_d4_a_null_merge_is_convert_byte_for_byte_and_convert_is_quantizing_on_load(tmp_path):
    base, t = checkpoint12(tmp_path)
    ad = with_k_copies(tmp_path, t, seed=5)
    conv, null = tmp_path / "converted", tmp_path / "null"
    report = hub_fp8.convert(base, conv)
    meta = hub_fp8.merge(base, ad, null, null=True)
    assert meta["null"] is True and meta["adapter_scale"] == 0.0 and meta["dropped_k_eq_v_pairs"] == len(FULL)
    files = {p.name for p in conv.iterdir()} - REPORTS
    assert files == {p.name for p in null.iterdir()} - REPORTS
    assert {"model-00001-of-00001.safetensors", "config.json", hub_fp8.INDEX, "tokenizer.json"} <= files
    assert {f: sha(conv / f) for f in files} == {f: sha(null / f) for f in files}  # every file, byte for byte
    groups = json.loads((null / "fp8_conversion.json").read_text())["groups"]
    assert groups == report["groups"]
    # the stored conversion loads to what quantizing on load holds, and the loader changes nothing
    assert report["loader_changed_elements"] == 0 and report["tensors"] == len(read_dir(conv))
    assert hub_fp8.compare(hub_fp8.expected_online(base), hub_fp8.expected_loaded(conv)) == []
    assert hub_fp8.compare(hub_fp8.expected_online(base), json.loads((null / "expected_loaded.json").read_text())) == []


def tuned_config_argv(base, ad, out, report, key="gemma-4-12b-fp8"):
    return [
        "tuned-config", "--base", key, "--temperatures", str(base.parent / "temps.json"), "--noul-commit",
        "--noul-commit-basis", json.dumps(BASIS), "--adapter-dir", str(ad), "--adapter-repository", "o/a",
        "--adapter-revision", "0" * 40, "--merge-report", str(out / report), "--weights-dir", str(out),
        "--out", str(out / hub.FILE),
    ]  # fmt: skip


@pytest.fixture
def merged12(tmp_path):
    base, t = checkpoint12(tmp_path)
    (tmp_path / "temps.json").write_text(json.dumps(TEMPS))
    ad = with_k_copies(tmp_path, t, seed=6)
    out = tmp_path / "tuned-12b-fp8"
    hub_fp8.merge(base, ad, out)
    return base, t, ad, out


def test_d5_tuned_config_writes_a_file_check_accepts(merged12, capsys):
    base, _, ad, out = merged12
    assert hub.main(tuned_config_argv(base, ad, out, "MERGE_FP8.json")) == 0
    config = json.loads((out / hub.FILE).read_text())
    assert config["base"] == "gemma-4-12b-fp8" and config["weights"]["dtype"] == "fp8_e4m3"
    merge = config["tuned"]["merge"]
    assert (merge["output"], merge["tool"], merge["dropped_k_eq_v_pairs"]) == (
        "fp8 compressed-tensors",
        "decisio.hub_fp8 merge",
        len(FULL),
    )
    assert "rounding" not in merge and "seed" not in merge
    assert {"model-00001-of-00001.safetensors", "config.json"} <= set(config["tuned"]["files"])
    capsys.readouterr()
    assert hub.main(["check", str(out / hub.FILE), "--weights-dir", str(out)]) == 0
    assert "agrees with the installed profile" in capsys.readouterr().out
    (out / "tokenizer.json").write_text("changed")
    assert hub.main(["check", str(out / hub.FILE), "--weights-dir", str(out)]) == 1
    assert "tokenizer.json" in capsys.readouterr().out


def test_d5_each_12b_refuses_the_others_directory(merged12, tmp_path, capsys):
    base, t, ad, out = merged12
    capsys.readouterr()
    # the FP8 directory as the bf16 12B: refused by its config.json before the merge report is read
    assert hub.main(tuned_config_argv(base, ad, out, "MERGE_FP8.json", key="gemma-4-12b")) == 1
    assert "declares FP8 weights, so it is a gemma-4-12b-fp8 checkpoint, not gemma-4-12b" in capsys.readouterr().out
    # a bf16 merge of the same base as the FP8 12B
    bf = tmp_path / "tuned-12b-bf16"
    hub_bf16.merge(base, ad, bf, seed=11)
    assert hub.main(tuned_config_argv(base, ad, bf, "MERGE_BF16.json")) == 1
    assert "declares no FP8 weights, so it is a gemma-4-12b checkpoint, not gemma-4-12b-fp8" in capsys.readouterr().out
    # its report alone under the FP8 base is refused too, by the merge facts
    refused = re.escape("a bf16 merge of the 12B (`hub_bf16 merge`) is the base gemma-4-12b")
    with pytest.raises(ValueError, match=refused):
        build12fp8(merge=hub.merge_facts(json.loads((bf / "MERGE_BF16.json").read_text())))
    # check: an FP8 12B file over the bf16 directory
    assert hub.main(tuned_config_argv(base, ad, out, "MERGE_FP8.json")) == 0
    capsys.readouterr()
    assert hub.main(["check", str(out / hub.FILE), "--weights-dir", str(bf)]) == 1
    assert "declares no FP8 weights" in capsys.readouterr().out


def test_d6_served_from_the_directory(merged12, tmp_path):
    base, _, ad, out = merged12
    assert hub.main(tuned_config_argv(base, ad, out, "MERGE_FP8.json")) == 0
    args = args_for(str(out))
    fam, _ = resolve_base(args)
    assert fam.key == "gemma-4-12b-fp8" and fam.precision == "fp8" and fam.quantization is None  # stored FP8
    assert args.served_name == FAM.served_name and args.temperature_choice == TEMPS["temperatures"]["choice"]
    assert args.model_identity.startswith(f"{os.path.basename(out)}@")
    assert args.repository["tuned"]  # its own temperatures, fitted on its reads: the start line has no bf16 note
    assert base_line(fam, args.model, tuned=True) == (
        f"BASE gemma-4-12b-fp8 (fp8, stored FP8, loaded as stored) from {out}"
    )
    # the bf16 12B with the same defaults is another model
    d12 = write12(tmp_path, build12(), name=os.path.basename(out) + "-bf16")
    a12 = args_for(str(d12))
    assert resolve_base(a12)[0].key == "gemma-4-12b" and a12.model_identity != args.model_identity


def test_d6_a_repository_and_its_checkpoint_must_agree_on_the_precision(tmp_path):
    fp8_cfg = {"model_type": "gemma4_unified", "quantization_config": {"quant_method": "fp8"}}
    for name, config in (("tuned", build12fp8()), ("plain", hub.export_decision_config(FAM, "0.23.0"))):
        d = write12(tmp_path, config, name=f"says-fp8-{name}")  # its config.json declares no FP8
        with pytest.raises(ValueError, match="declares no FP8 weights: a bf16 checkpoint is served as gemma-4-12b"):
            resolve_base(args_for(str(d)))
        (d / "config.json").write_text(json.dumps(fp8_cfg))
        assert resolve_base(args_for(str(d)))[0].key == "gemma-4-12b-fp8"
    d = write12(tmp_path, build12(), name="says-bf16")
    (d / "config.json").write_text(json.dumps(fp8_cfg))
    with pytest.raises(ValueError, match="declares FP8 weights: serve it with --base gemma-4-12b-fp8"):
        resolve_base(args_for(str(d)))
