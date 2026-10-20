# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`python -m decisio.hub_fp8 merge`: a LoRA adapter merged into the bf16 base in fp32 and quantized once, on the tiny
checkpoint of test_hub_fp8 (no model, no card).

M1  a null merge (scale 0) equals `convert` of the base, byte for byte, and writes the report
M2  an adapter in either layout (text-only `model.layers.N`, multimodal `model.language_model.layers.N`) lands on the
    base's tensors; every pair is applied, and the count is in the report
M3  the merged FP8 weights are the fp32 merge quantized once: dequantized, they equal W + scale * B @ A to the FP8 grid
    of their group, and the adapter's change is inside the amax (the scale of a group moves with it)
M4  the merge keeps the adapter where a bf16 round trip loses it: on weights at the bf16 grid with an adapter 0.05
    percent of a weight, the fp32 merge differs from the base in far more entries than (W + delta).to(bf16) does
M5  refusals: a pair that finds no tensor, a tensor outside the quantized groups, a missing half, an unexpected key,
    DoRA / rank patterns
M6  the report is what `decisio.hub tuned-config` reads: its adapter hash is the file's, its counts are the merge's

  uv run pytest -q tests/unit/test_hub_fp8_merge.py
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_hub_fp8 import L, checkpoint

from decisio import hub, hub_fp8

R = 2


def adapter(tmp_path, names, layout="text", scale_alpha=4, seed=0, drop=None, extra=None, cfg=None):
    """A PEFT directory with a rank-2 pair for each base weight in `names` (the 2-D projections)."""
    g = torch.Generator().manual_seed(seed)
    d = tmp_path / f"adapter-{layout}-{seed}-{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    tensors = {}
    for name, shape in names.items():
        module = name[: -len(".weight")]
        if layout == "text":
            module = module.replace("model.language_model.layers.", "model.layers.", 1)
        out_f, in_f = shape
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.randn(R, in_f, generator=g) * 0.02
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.randn(out_f, R, generator=g) * 0.02
    for k in drop or []:
        tensors.pop(k)
    tensors.update(extra or {})
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": scale_alpha, **(cfg or {})}))
    return d


def targets(t):
    return {n: tuple(w.shape) for n, w in t.items() if f"{L}." in n and n.endswith("_proj.weight")}


def read_dir(directory):
    return hub_fp8_read(directory)


def hub_fp8_read(directory):
    out = {}
    src = hub_fp8.Source(directory)
    for n in src.map:
        out[n] = src.get(n)
    return out


def test_m1_a_null_merge_is_convert_byte_for_byte(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t))
    hub_fp8.convert(base, tmp_path / "converted")
    meta = hub_fp8.merge(base, ad, tmp_path / "null", null=True)
    a, b = read_dir(tmp_path / "converted"), read_dir(tmp_path / "null")
    assert set(a) == set(b) and all(
        torch.equal(
            a[k].view(torch.uint8) if a[k].dtype == hub_fp8.FP8 else a[k],
            b[k].view(torch.uint8) if b[k].dtype == hub_fp8.FP8 else b[k],
        )
        for k in a
    )
    assert meta["null"] is True and meta["adapter_scale"] == 0.0 and meta["loader_changed_elements"] == 0
    assert json.loads((tmp_path / "null" / "MERGE_FP8.json").read_text())["kind"] == "fp32 merge to FP8"
    assert (tmp_path / "null" / "tokenizer.json").exists() and (tmp_path / "null" / "expected_loaded.json").exists()


@pytest.mark.parametrize("layout", ["text", "multimodal"])
def test_m2_either_layout_lands_on_the_bases_tensors(tmp_path, layout):
    base, t = checkpoint(tmp_path)
    want = targets(t)
    ad = adapter(tmp_path, want, layout=layout)
    meta = hub_fp8.merge(base, ad, tmp_path / "out")
    assert meta["adapted_tensors"] == len(want) and meta["adapter_layout"] == layout and meta["null"] is False
    assert meta["adapter_scale"] == 2.0 and meta["written_vs_loaded_differ"] == []


def test_m3_the_merge_is_fp32_then_one_quantization(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=3)
    hub_fp8.merge(base, ad, tmp_path / "out")
    merged = read_dir(tmp_path / "out")
    deltas, scale, _, _, _ = hub_fp8.load_adapter(ad, hub_fp8.Source(base).map)
    name = f"{L}.0.mlp.down_proj.weight"  # a group of its own (down_proj is not fused)
    want = t[name].float() + scale * (deltas[name][1] @ deltas[name][0])
    sc = merged[name.replace(".weight", ".weight_scale")]
    got = merged[name].float() * sc  # dequantized
    # FP8 e4m3 keeps 3 mantissa bits: a value is within 2**-4 of itself relative to the group's largest
    assert torch.allclose(got, want, atol=float(want.abs().max()) * 0.07)
    assert float(sc) == pytest.approx(float(want.abs().max()) / 448, rel=1e-6)  # the change is inside the amax


def test_m4_fp32_keeps_the_adapter_where_a_bf16_round_trip_loses_it(tmp_path):
    _, t = checkpoint(tmp_path)
    w = t[f"{L}.0.mlp.down_proj.weight"]  # on the bf16 grid already
    g = torch.Generator().manual_seed(7)
    delta = torch.randn(w.shape, generator=g) * 0.0005 * float(w.float().abs().mean())  # 0.05 percent of a weight
    through_bf16 = (w.float() + delta).to(torch.bfloat16)
    assert int((through_bf16 != w).sum()) < 0.5 * w.numel()  # most of the adapter is rounded away in bf16
    assert int(((w.float() + delta) != w.float()).sum()) > 0.9 * w.numel()  # and kept in fp32
    a, b = torch.randn(R, w.shape[1], generator=g), torch.randn(w.shape[0], R, generator=g)
    assert hub_fp8.merged32(w, (a, b), 1.0, "cpu").dtype == torch.float32  # the merge itself adds in fp32


def test_m5_refusals(tmp_path):
    base, t = checkpoint(tmp_path)
    want = targets(t)
    src = hub_fp8.Source(base)
    ok = adapter(tmp_path, want, seed=1)
    stray = adapter(tmp_path, {**want, "model.language_model.layers.9.mlp.up_proj.weight": (4, 4)}, seed=2)
    with pytest.raises(ValueError, match="layers.9.mlp.up_proj: no such tensor in the base"):
        hub_fp8.merge(base, stray, tmp_path / "o1")
    outside = adapter(tmp_path, {**want, "model.language_model.embed_tokens.weight": (20, 6)}, seed=3)
    with pytest.raises(ValueError, match="not in a quantized group"):
        hub_fp8.merge(base, outside, tmp_path / "o2")
    half = next(
        k
        for k in __import__("safetensors").safe_open(str(ok / "adapter_model.safetensors"), "pt").keys()
        if k.endswith("lora_B.weight")
    )  # noqa: SIM118
    missing = adapter(tmp_path, want, seed=4, drop=[half])
    with pytest.raises(ValueError, match="missing one half"):
        hub_fp8.load_adapter(missing, src.map)
    odd = adapter(tmp_path, want, seed=5, extra={"base_model.model.something": torch.zeros(1)})
    with pytest.raises(ValueError, match="unexpected adapter tensor"):
        hub_fp8.load_adapter(odd, src.map)
    for cfg, message in (({"use_dora": True}, "plain LoRA"), ({"rank_pattern": {"x": 4}}, "one rank and one alpha")):
        with pytest.raises(ValueError, match=message):
            hub_fp8.load_adapter(adapter(tmp_path, want, seed=6, cfg=cfg), src.map)
    mixed = adapter(tmp_path, want, seed=8)
    from safetensors.torch import load_file

    tensors = load_file(str(mixed / "adapter_model.safetensors"))
    k = next(x for x in tensors if "layers.0.self_attn.q_proj" in x)
    tensors[k.replace("model.layers.", "model.language_model.layers.", 1)] = tensors.pop(k)
    save_file(tensors, str(mixed / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="mixes the text-only and the multimodal layout"):
        hub_fp8.load_adapter(mixed, src.map)


def test_m6_the_report_feeds_tuned_config(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=9)
    meta = hub_fp8.merge(base, ad, tmp_path / "out")
    facts = hub.merge_facts(json.loads((tmp_path / "out" / "MERGE_FP8.json").read_text()))
    assert facts["adapter_sha256"] == meta["adapter_sha256"] and facts["adapted_tensors"] == len(targets(t))
    assert facts["adapter_scale"] == 2.0 and facts["precision"] == "fp32"
    adapter_facts = hub.adapter_facts(ad, "x/adapter", "rev")
    assert adapter_facts["weights_sha256"] == meta["adapter_sha256"] and adapter_facts["rank"] == R


def test_m7_the_command_line(tmp_path, capsys):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=10)
    rc = hub_fp8.main(["merge", "--source", str(base), "--adapter", str(ad), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0 and "applied (text layout, 0 v_proj copies of k_proj pairs left out)" in out and "identical" in out
    rc = hub_fp8.main(["merge", "--source", str(base), "--adapter", str(ad), "--out", str(tmp_path / "n"), "--null"])
    assert rc == 0


def with_v_proj(tmp_path, want, seed, differ=False):
    """The adapter of `want` plus a `v_proj` pair for layer 1 (a full-attention layer with no v_proj in the base): a
    copy of the layer's k_proj pair, as the trainer's export writes it, or (differ) a different pair."""
    from safetensors import safe_open

    base_dir = adapter(tmp_path, want, seed=seed)
    with safe_open(str(base_dir / "adapter_model.safetensors"), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    for half in ("A", "B"):
        src = tensors[f"base_model.model.model.layers.1.self_attn.k_proj.lora_{half}.weight"]
        tensors[f"base_model.model.model.layers.1.self_attn.v_proj.lora_{half}.weight"] = src + (0.001 if differ else 0)
    d = tmp_path / f"adapter-v-{seed}-{differ}"
    d.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text((base_dir / "adapter_config.json").read_text())
    return d


def test_m8_v_proj_copies_of_k_eq_v_layers_are_left_out_and_counted_and_a_different_pair_is_refused(tmp_path):
    """Layer 1 of the tiny checkpoint is a full-attention layer with no v_proj (attention_k_eq_v). The N1 pilot's
    adapter carries a v_proj pair there, bit for bit its k_proj pair: merging the k_proj delta already gives V that
    delta, so the copy is left out and counted; a pair that differs asks for a V that differs from K and is refused."""
    base, t = checkpoint(tmp_path)
    want = targets(t)
    copy = with_v_proj(tmp_path, want, seed=11)
    meta = hub_fp8.merge(base, copy, tmp_path / "dropped")
    assert meta["dropped_k_eq_v_pairs"] == 1 and meta["adapted_tensors"] == len(want)
    facts = hub.merge_facts(json.loads((tmp_path / "dropped" / "MERGE_FP8.json").read_text()))
    assert facts["dropped_k_eq_v_pairs"] == 1
    # the same merge without the copy is the same checkpoint: leaving the copy out changes nothing else
    plain = hub_fp8.merge(base, adapter(tmp_path, want, seed=11), tmp_path / "plain")
    assert plain["dropped_k_eq_v_pairs"] == 0
    a, b = read_dir(tmp_path / "dropped"), read_dir(tmp_path / "plain")
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    # a pair that is not the k_proj pair is refused, with the difference
    other = with_v_proj(tmp_path, want, seed=11, differ=True)
    with pytest.raises(ValueError, match=r"not copies of the layer's k_proj pair.*A differs by up to 0.001"):
        hub_fp8.merge(base, other, tmp_path / "refused")
    # a v_proj pair on a layer that DOES ship v_proj is an ordinary pair, and a made-up layer is still an error
    stray = adapter(tmp_path, {**want, f"{L}.7.self_attn.v_proj.weight": (4, 6)}, seed=12)
    with pytest.raises(ValueError, match="layers.7.self_attn.v_proj: no such tensor in the base"):
        hub_fp8.merge(base, stray, tmp_path / "stray")
