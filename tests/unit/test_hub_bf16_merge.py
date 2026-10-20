# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`python -m decisio.hub_bf16 merge`: a LoRA adapter merged into a bf16 checkpoint in fp32 and rounded once to
bf16, stochastically (the 12B's repository), on the tiny checkpoint of test_hub_fp8 (no model, no card).

R1  stochastic rounding: exact bf16 values are fixed points, every result is one of the two neighbours, the expected
    value
    is the fp32 value (unbiased), the noise is seeded per tensor name (repeatable, other name or seed differs)
R2  a null merge equals the source byte for byte, in every tensor and every file
R3  the merge carries the adapter: on weights at the bf16 grid with an adapter a small fraction of a grid step, the
    stochastic merge keeps about all of the change in weight space where rounding to nearest keeps about half or less
R4  every other tensor is copied byte for byte; the result does not depend on the shard size or the tensor order
R5  the K=V rule: a v_proj pair that is bit for bit its layer's k_proj pair is left out and counted, one that differs is
    refused; a pair that finds no tensor, or a tensor outside the text model's projections, is an error
R6  the report is what `decisio.hub tuned-config` reads for a 12B: output bf16, rounding, seed, counts
R7  the command line

  uv run pytest -q tests/unit/test_hub_bf16_merge.py
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_hub_fp8 import L, checkpoint
from test_hub_fp8_merge import adapter, read_dir, targets, with_v_proj

from decisio import hub, hub_bf16, hub_fp8
from decisio.families import BASES


def grid_neighbours(x):
    """The two bf16 neighbours of float32 values (by magnitude: truncated, and the next one up)."""
    bits = x.contiguous().view(torch.int32)
    a = (bits & torch.tensor(-65536, dtype=torch.int32)).view(torch.float32)
    b = ((bits & torch.tensor(-65536, dtype=torch.int32)) + 65536).view(torch.float32)
    return a, b


def test_r1_exact_values_are_fixed_points_and_every_result_is_a_neighbour():
    w = (torch.randn(64, 64) * 0.05).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(hub_bf16.stochastic_bf16(w, 0, "t").to(torch.float32), w)  # on the grid: unchanged
    x = w + torch.randn(64, 64) * 1e-4
    r = hub_bf16.stochastic_bf16(x, 0, "t").to(torch.float32)
    lo, hi = grid_neighbours(x)
    assert bool(((r == lo) | (r == hi)).all())
    assert r.dtype == torch.float32 and hub_bf16.stochastic_bf16(x, 0, "t").dtype == torch.bfloat16


def test_r1_the_rounding_is_unbiased():
    w = (torch.randn(256, 256) * 0.05).to(torch.bfloat16).to(torch.float32)
    delta = torch.randn(256, 256) * 3e-5  # a fraction of a grid step (the grid at 0.05 is about 2e-4)
    x = w + delta
    errors = torch.stack([hub_bf16.stochastic_bf16(x, s, "t").to(torch.float32) - x for s in range(40)])
    assert abs(float(errors.mean())) < 2e-7  # unbiased: the mean error vanishes
    nearest = (x.to(torch.bfloat16).to(torch.float32) - w).double()
    kept_nearest = float((nearest * delta.double()).sum() / (delta.double() ** 2).sum())
    stoch = (
        torch.stack([hub_bf16.stochastic_bf16(x, s, "t").to(torch.float32) for s in range(40)]).mean(0) - w
    ).double()
    kept_stoch = float((stoch * delta.double()).sum() / (delta.double() ** 2).sum())
    assert kept_nearest < 0.6 and 0.97 < kept_stoch < 1.03, (kept_nearest, kept_stoch)


def test_r1_the_noise_is_seeded_per_tensor_name():
    x = torch.randn(32, 32) * 0.05 + 1e-4
    a = hub_bf16.stochastic_bf16(x, 0, "layers.0.q")
    assert torch.equal(a, hub_bf16.stochastic_bf16(x, 0, "layers.0.q"))
    assert not torch.equal(a, hub_bf16.stochastic_bf16(x, 0, "layers.0.k"))
    assert not torch.equal(a, hub_bf16.stochastic_bf16(x, 1, "layers.0.q"))


def test_r2_a_null_merge_equals_the_source_byte_for_byte(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=1)
    meta = hub_bf16.merge(base, ad, tmp_path / "null", null=True)
    assert (
        meta["null"] is True and meta["adapted_tensors"] == len(targets(t)) and meta["weight_space_effect_kept"] is None
    )
    a, b = read_dir(base), read_dir(tmp_path / "null")
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
    for f in base.iterdir():  # the config, tokenizer and other files of the base are copied
        if not f.name.endswith(".safetensors") and f.name != hub_fp8.INDEX:
            assert (tmp_path / "null" / f.name).read_bytes() == f.read_bytes(), f.name


def test_r3_the_merge_carries_the_adapter_where_rounding_to_nearest_loses_it(tmp_path):
    base, t = checkpoint(tmp_path)
    names = targets(t)
    ad = adapter(tmp_path, names, seed=2, scale_alpha=0.02)  # a delta far below a grid step
    meta = hub_bf16.merge(base, ad, tmp_path / "m", seed=3)
    assert 0.5 < meta["weight_space_effect_kept"] < 1.5 and abs(meta["rounding_bias_over_change"]) < 0.5
    out = read_dir(tmp_path / "m")
    deltas, scale, *_ = hub_fp8.load_adapter(ad, hub_fp8.Source(base).map)
    num = den = nearest_num = 0.0
    for n, ab in deltas.items():
        exact = hub_fp8.merged32(t[n], ab, scale, "cpu")
        d = (exact - t[n].float()).double()
        num += float(((out[n].float() - t[n].float()).double() * d).sum())
        nearest_num += float(((exact.to(torch.bfloat16).float() - t[n].float()).double() * d).sum())
        den += float((d * d).sum())
    assert nearest_num / den + 0.15 < num / den and 0.5 < num / den < 1.5, (nearest_num / den, num / den)


def test_r4_other_tensors_are_the_sources_and_the_result_does_not_depend_on_the_shards(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=4, scale_alpha=0.02)
    hub_bf16.merge(base, ad, tmp_path / "one", seed=5, shard_bytes=1 << 30)
    hub_bf16.merge(base, ad, tmp_path / "many", seed=5, shard_bytes=200)  # a shard per few tensors
    a, b, src = read_dir(tmp_path / "one"), read_dir(tmp_path / "many"), read_dir(base)
    assert set(a) == set(b) == set(src) and all(torch.equal(a[k], b[k]) for k in a)
    adapted = set(targets(t))
    assert all(torch.equal(a[k], src[k]) for k in set(src) - adapted)  # norms, embeddings, the vision tower: copied
    assert any(not torch.equal(a[k], src[k]) for k in adapted)
    assert len(list((tmp_path / "many").glob("model-*.safetensors"))) > len(
        list((tmp_path / "one").glob("model-*.safetensors"))
    )
    again = hub_bf16.merge(base, ad, tmp_path / "again", seed=5)
    assert all(torch.equal(a[k], v) for k, v in read_dir(tmp_path / "again").items()) and again["seed"] == 5


def test_r5_the_k_eq_v_rule_is_the_fp8_merges(tmp_path):
    base, t = checkpoint(tmp_path)
    want = targets(t)
    copy = with_v_proj(tmp_path, want, seed=11)
    meta = hub_bf16.merge(base, copy, tmp_path / "copy")
    assert meta["dropped_k_eq_v_pairs"] == 1 and meta["adapted_tensors"] == len(want)
    plain = hub_bf16.merge(base, adapter(tmp_path, want, seed=11), tmp_path / "plain")
    a, b = read_dir(tmp_path / "copy"), read_dir(tmp_path / "plain")
    assert plain["dropped_k_eq_v_pairs"] == 0 and all(torch.equal(a[k], b[k]) for k in a)
    other = with_v_proj(tmp_path, want, seed=11, differ=True)
    with pytest.raises(ValueError, match=r"not copies of the layer's k_proj pair.*A differs by up to 0.001"):
        hub_bf16.merge(base, other, tmp_path / "refused")


def test_r5_a_pair_without_a_tensor_or_outside_the_projections_is_an_error(tmp_path):
    base, t = checkpoint(tmp_path)
    want = targets(t)
    stray = adapter(tmp_path, {**want, f"{L}.7.self_attn.v_proj.weight": (4, 6)}, seed=12)
    with pytest.raises(ValueError, match="layers.7.self_attn.v_proj: no such tensor in the base"):
        hub_bf16.merge(base, stray, tmp_path / "stray")
    vision = {**want, "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight": (4, 6)}
    ad = adapter(tmp_path, vision, seed=13)
    with pytest.raises(ValueError, match="not projections of the text model"):
        hub_bf16.merge(base, ad, tmp_path / "vision")
    half = adapter(tmp_path, want, seed=14, drop=["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"])
    with pytest.raises(ValueError, match="missing one half"):
        hub_bf16.merge(base, half, tmp_path / "half")


def test_r6_the_report_is_what_tuned_config_reads_for_a_12b(tmp_path):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=6, scale_alpha=0.02)
    hub_bf16.merge(base, ad, tmp_path / "m", seed=9)
    facts = hub.merge_facts(json.loads((tmp_path / "m" / "MERGE_BF16.json").read_text()))
    assert facts["output"] == "bf16" and facts["rounding"] == "stochastic" and facts["seed"] == 9
    assert (
        facts["tool"] == "decisio.hub_bf16 merge"
        and facts["precision"] == "fp32"
        and facts["adapted_tensors"] == len(targets(t))
    )
    assert facts["adapter_sha256"] == hub.adapter_facts(ad, "o/a", "r")["weights_sha256"]
    null = hub_bf16.merge(base, ad, tmp_path / "n", null=True)
    with pytest.raises(ValueError, match="null merge"):
        hub.merge_facts(null)
    assert BASES["gemma-4-12b"].key in hub.TUNED_BASES


def test_r7_the_command_line(tmp_path, capsys):
    base, t = checkpoint(tmp_path)
    ad = adapter(tmp_path, targets(t), seed=7, scale_alpha=0.02)
    rc = hub_bf16.main(
        ["merge", "--source", str(base), "--adapter", str(ad), "--out", str(tmp_path / "o"), "--seed", "4"]
    )
    out = capsys.readouterr().out
    assert (
        rc == 0
        and "adapter pairs applied (text layout, 0 v_proj copies of k_proj pairs left out)" in out
        and "seed 4" in out
    )
    assert json.loads((tmp_path / "o" / "MERGE_BF16.json").read_text())["seed"] == 4
    assert save_file is not None
