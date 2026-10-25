# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`--keep-bf16` of `python -m decisio.hub_fp8 convert|merge`: projections left in bf16 inside an FP8 checkpoint, on the
tiny checkpoint of test_hub_fp8 (no model, no card).

K1  the specs: layers (a range, a list), projections, a group; every malformation names the spec
K2  which groups a spec keeps; naming one shard of a fused group keeps the whole group and says so
K3  convert: kept tensors are the source's own bf16, byte for byte, with no weight_scale; the rest is FP8 as before; the
    ignore list of the quantization config names every shard of a kept group (v_proj included where the layer has none)
K4  the fingerprints skip the kept groups; written and loaded agree on the rest
K5  merge: a kept tensor is the fp32 merge rounded once to bf16 stochastically (and carries the adapter where round to
    nearest would give it back); the quantized rest is the merge without a keep list, byte for byte
K6  a null merge with a keep list equals convert with the same list, byte for byte
K7  refusals: everything kept, a layer or group the checkpoint does not have
K8  the report and the tuned-config facts say what was kept; the command line takes the flag

  uv run pytest -q tests/unit/test_hub_fp8_keep.py
"""

import json

import pytest
import torch
from test_hub_fp8 import FP8, L, checkpoint, read_all
from test_hub_fp8_merge import adapter, targets

from decisio import hub, hub_fp8
from decisio.hub_bf16 import stochastic_bf16

SCALE = 2.0  # the adapter() helper: alpha 4 over rank 2


def same_bytes(a, b):
    f = lambda x: x.view(torch.uint8) if x.dtype == FP8 else x  # noqa: E731
    return a.dtype == b.dtype and torch.equal(f(a), f(b))


def test_k1_the_specs():
    k = hub_fp8.parse_keep(["layers:1-3", "layers:7,9", "proj:down_proj,o_proj", "group:layers.2.mlp.gate_up_proj"])
    assert (
        k.layers == {1, 2, 3, 7, 9} and k.projs == {"down_proj", "o_proj"} and k.groups == {"layers.2.mlp.gate_up_proj"}
    )
    assert bool(k) and not hub_fp8.parse_keep([]) and not hub_fp8.parse_keep(None)
    for bad, msg in [
        ("down_proj", "expected layers:"),
        ("layers:", "expected layers:"),
        ("layers:4-2", "numbers or ranges"),
        ("layers:x", "numbers or ranges"),
        ("proj:lm_head", "not a projection"),
        ("group:layers.3.mlp.q_proj", "not a group key"),
        ("expert:3", "expected layers:, proj: or group:"),
    ]:
        with pytest.raises(ValueError, match=msg):
            hub_fp8.parse_keep([bad])


def test_k2_which_groups_a_spec_keeps(tmp_path):
    src, t = checkpoint(tmp_path)
    groups = hub_fp8.find_groups(t, k_eq_v=True)
    keys = {g.key for g in groups}
    assert keys == {
        f"layers.{i}.{m}"
        for i in (0, 1)
        for m in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    }
    pick = lambda specs: {g.key for g in hub_fp8.kept_groups(groups, hub_fp8.parse_keep(specs))}  # noqa: E731
    assert pick(["layers:1"]) == {k for k in keys if k.startswith("layers.1.")}
    assert pick(["proj:down_proj", "proj:o_proj"]) == {
        f"layers.{i}.{m}" for i in (0, 1) for m in ("mlp.down_proj", "self_attn.o_proj")
    }
    assert pick(["group:layers.0.mlp.gate_up_proj"]) == {"layers.0.mlp.gate_up_proj"}
    keep = hub_fp8.parse_keep(["proj:q_proj"])  # one shard of a fused group
    assert {g.key for g in hub_fp8.kept_groups(groups, keep)} == {
        "layers.0.self_attn.qkv_proj",
        "layers.1.self_attn.qkv_proj",
    }
    assert len(keep.extended) == 2 and "q_proj named, so the whole fused group is kept" in keep.extended[0]
    assert pick(["proj:qkv_proj"]) == {"layers.0.self_attn.qkv_proj", "layers.1.self_attn.qkv_proj"}


@pytest.mark.parametrize("scheme", ["compressed-tensors", "fp8"])
def test_k3_convert_leaves_the_kept_tensors_as_the_source_has_them(tmp_path, scheme):
    src, t = checkpoint(tmp_path)
    out = tmp_path / "out"
    report = hub_fp8.convert(src, out, scheme, keep=hub_fp8.parse_keep(["layers:1", "proj:down_proj"]))
    written = read_all(out)
    kept = [n for n in t if n.startswith(f"{L}.1.") and n.endswith("_proj.weight")] + [f"{L}.0.mlp.down_proj.weight"]
    assert len(kept) == 6 + 1  # layer 1 has no v_proj
    for n in kept:
        assert same_bytes(written[n], t[n]), n
        assert n[: -len(".weight")] + ".weight_scale" not in written
    for n in (f"{L}.0.self_attn.q_proj.weight", f"{L}.0.self_attn.o_proj.weight", f"{L}.0.mlp.gate_proj.weight"):
        assert written[n].dtype == FP8 and n[: -len(".weight")] + ".weight_scale" in written
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    ignore = qc.get("ignore", qc.get("modules_to_not_convert"))
    for name in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        assert rf"re:.*layers\.1\.{name.replace('.', chr(92) + '.')}$" in ignore  # v_proj of layer 1 too: all-or-none
    assert r"re:.*layers\.0\.mlp\.down_proj$" in ignore and r"re:.*layers\.0\.self_attn\.q_proj$" not in ignore
    import re

    names = [f"{L}.1.self_attn.v_proj", f"{L}.0.mlp.down_proj"]  # a pattern matches the checkpoint's own module names
    assert all(any(re.match(p[3:], n) for p in ignore if p.startswith("re:")) for n in names)
    assert not any(re.match(p[3:], f"{L}.0.self_attn.q_proj") for p in ignore if p.startswith("re:"))
    assert "re:.*vision_tower.*" in ignore  # the base list stays
    assert report["kept_bf16"]["groups"] == [
        "layers.1.self_attn.qkv_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.mlp.gate_up_proj",
        "layers.1.mlp.down_proj",
        "layers.0.mlp.down_proj",
    ] or set(report["kept_bf16"]["groups"]) == {
        "layers.1.self_attn.qkv_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.mlp.gate_up_proj",
        "layers.1.mlp.down_proj",
        "layers.0.mlp.down_proj",
    }
    assert "rounding" not in report["kept_bf16"]  # a conversion rounds nothing


def test_k4_the_fingerprints_skip_the_kept_groups(tmp_path):
    src, t = checkpoint(tmp_path)
    out = tmp_path / "out"
    hub_fp8.convert(src, out, keep=hub_fp8.parse_keep(["layers:1"]))
    loaded = hub_fp8.expected_loaded(out)
    online = hub_fp8.expected_online(src, hub_fp8.parse_keep(["layers:1"]))
    assert set(loaded) == set(online) == {k for k in hub_fp8.expected_online(src) if k.startswith("layers.0.")}
    assert hub_fp8.compare(loaded, online) == []
    assert hub_fp8.expected_online(src, None).keys() > online.keys()  # without the list, every group


def test_k4_a_group_with_some_scales_and_not_others_is_refused(tmp_path):
    src, t = checkpoint(tmp_path)
    out = tmp_path / "out"
    hub_fp8.convert(src, out)
    from safetensors.torch import load_file, save_file

    for f in out.glob("*.safetensors"):
        d = load_file(str(f))
        if f"{L}.0.self_attn.k_proj.weight_scale" in d:
            del d[f"{L}.0.self_attn.k_proj.weight_scale"]
            save_file(d, str(f), metadata={"format": "pt"})
            idx = json.loads((out / hub_fp8.INDEX).read_text())
            idx["weight_map"].pop(f"{L}.0.self_attn.k_proj.weight_scale")
            (out / hub_fp8.INDEX).write_text(json.dumps(idx))
    with pytest.raises(ValueError, match="some of its shards have a weight_scale and some do not"):
        hub_fp8.expected_loaded(out)


def test_k5_a_merge_keeps_the_adapter_in_the_bf16_tensors_and_the_rest_is_the_plain_merge(tmp_path):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=3)
    plain = tmp_path / "plain"
    hub_fp8.merge(src, ad, plain)
    out = tmp_path / "keep"
    meta = hub_fp8.merge(src, ad, out, keep=hub_fp8.parse_keep(["layers:1"]), seed=5)
    a, b = read_all(plain), read_all(out)
    from safetensors import safe_open

    with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    for n in (n for n in t if n.startswith(f"{L}.1.") and n.endswith("_proj.weight")):
        module = n[: -len(".weight")].replace("model.language_model.layers.", "model.layers.", 1)
        w32 = t[n].to(torch.float32) + SCALE * (
            lora[f"base_model.model.{module}.lora_B.weight"] @ lora[f"base_model.model.{module}.lora_A.weight"]
        )
        assert b[n].dtype == torch.bfloat16 and torch.equal(b[n], stochastic_bf16(w32, 5, n)), n
        assert not torch.equal(b[n], t[n])  # the adapter is in it
        assert n[: -len(".weight")] + ".weight_scale" not in b
    for n in a:
        if (
            not n.startswith(f"{L}.1.")
            and a[n].dtype != torch.bfloat16
            or (n.startswith(f"{L}.0.") and a[n].dtype == FP8)
        ):
            assert same_bytes(a[n], b[n]), n  # layer 0's FP8 tensors and scales: the same as without a keep list
    assert meta["quantized_groups"] == 4 and meta["groups"] == 8
    assert meta["kept_bf16"]["rounding"] == "stochastic" and meta["kept_bf16"]["seed"] == 5
    assert "rounding" not in meta  # the top level stays an FP8 merge's: hub.merge_facts reads it
    assert meta["loader_changed_elements"] == 0 and meta["written_vs_loaded_differ"] == []


def test_k5_round_to_nearest_would_have_given_the_adapter_back(tmp_path):
    """An adapter whose change is below a bf16 step of the weight (here 5% of the tiny one): round to nearest returns
    most entries to the base, the stochastic rounding changes more of them (and keeps the change in expectation,
    which test_hub_bf16_merge measures)."""
    from safetensors.torch import load_file, save_file

    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=4)
    tensors = load_file(str(ad / "adapter_model.safetensors"))
    tensors = {k: (v * 0.05 if "lora_B" in k else v).contiguous() for k, v in tensors.items()}
    save_file(tensors, str(ad / "adapter_model.safetensors"))
    out = tmp_path / "keep"
    hub_fp8.merge(src, ad, out, keep=hub_fp8.parse_keep(["layers:0"]))
    b = read_all(out)
    n, module = f"{L}.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj"
    delta = SCALE * (
        tensors[f"base_model.model.{module}.lora_B.weight"] @ tensors[f"base_model.model.{module}.lora_A.weight"]
    )
    w32 = t[n].to(torch.float32) + delta
    nearest = int((w32.to(torch.bfloat16) != t[n]).sum())
    stochastic = int((b[n] != t[n]).sum())
    assert stochastic > nearest


def test_k6_a_null_merge_with_a_keep_list_is_convert_with_the_same_list(tmp_path):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=6)
    keep = ["layers:1", "proj:down_proj"]
    hub_fp8.convert(src, tmp_path / "c", keep=hub_fp8.parse_keep(keep))
    hub_fp8.merge(src, ad, tmp_path / "n", null=True, keep=hub_fp8.parse_keep(keep))
    a, b = read_all(tmp_path / "c"), read_all(tmp_path / "n")
    assert set(a) == set(b) and all(same_bytes(a[k], b[k]) for k in a)


def test_k7_refusals(tmp_path):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=7)
    with pytest.raises(ValueError, match="keeps every projection"):
        hub_fp8.convert(src, tmp_path / "a", keep=hub_fp8.parse_keep(["layers:0-1"]))
    with pytest.raises(ValueError, match="keeps every projection"):
        hub_fp8.merge(src, ad, tmp_path / "b", keep=hub_fp8.parse_keep(["layers:0,1"]))
    with pytest.raises(ValueError, match=r"layers \[5\] are not in this checkpoint"):
        hub_fp8.convert(src, tmp_path / "c", keep=hub_fp8.parse_keep(["layers:5"]))
    with pytest.raises(ValueError, match="group 'layers.9.mlp.down_proj' is not in this checkpoint"):
        hub_fp8.convert(src, tmp_path / "d", keep=hub_fp8.parse_keep(["group:layers.9.mlp.down_proj"]))


def test_k8_the_report_and_the_tuned_config_facts_and_the_command_line(tmp_path, capsys):
    src, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=8)
    rc = hub_fp8.main(
        [
            "merge",
            "--source",
            str(src),
            "--adapter",
            str(ad),
            "--out",
            str(tmp_path / "m"),
            "--keep-bf16",
            "layers:1",
            "--keep-bf16",
            "proj:down_proj",
            "--seed",
            "3",
        ]
    )
    assert rc == 0 and "identical" in capsys.readouterr().out
    report = json.loads((tmp_path / "m" / "MERGE_FP8.json").read_text())
    facts = hub.merge_facts(report)
    assert facts["tool"] == "decisio.hub_fp8 merge" and facts["output"] == "fp8 compressed-tensors"
    assert facts["kept_bf16"]["specs"] == ["layers:1", "proj:down_proj"] and facts["kept_bf16"]["seed"] == 3
    assert hub.merge_facts(json.loads((tmp_path / "m" / "MERGE_FP8.json").read_text()))["kept_bf16"]["groups"]
    rc = hub_fp8.main(["convert", "--source", str(src), "--out", str(tmp_path / "c"), "--keep-bf16", "proj:o_proj"])
    assert rc == 0 and json.loads((tmp_path / "c" / "fp8_conversion.json").read_text())["kept_bf16"]["specs"] == [
        "proj:o_proj"
    ]
    with pytest.raises(ValueError, match="--keep-bf16 'o_proj'"):
        hub_fp8.main(["convert", "--source", str(src), "--out", str(tmp_path / "x"), "--keep-bf16", "o_proj"])
    plain = hub.merge_facts(json.loads((tmp_path / "m" / "MERGE_FP8.json").read_text()) | {"kept_bf16": None})
    assert "kept_bf16" not in plain  # a merge with no exceptions has no such key
