# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calibration pass of error-compensating FP8 rounding (decisio.fp8_gptq_pass) on a tiny Gemma 4 text model built
with transformers (two layers: sliding and full attention with attention_k_eq_v, so the second has no v_proj):

P1  the pass returns codes and a scale for every group of the checkpoint, the scale is hub_fp8's, and the groups kept
    bf16 are not in it
P2  every group's compensated output error is below round to nearest's (the pass's own report)
P3  end to end: the last hidden states of the model with the compensated weights are closer to the original's than with
    round to nearest, on sequences the pass did not see
P4  the codes are the stored format of hub_fp8 (fp8, one scale per fused group, the loader's requantization is the
    identity), and a merged adapter is the weight the pass rounds
P5  refusals: no sequences, a model type the pass does not read

  uv run pytest -q tests/unit/test_fp8_gptq_pass.py
"""

import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")
from transformers.models.gemma4 import modeling_gemma4 as M  # noqa: E402
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig  # noqa: E402

from decisio import fp8_gptq_pass as P  # noqa: E402
from decisio import hub_fp8  # noqa: E402

VOCAB, H = 96, 32


def tiny_config():
    return Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, global_head_dim=16, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"], sliding_window=8, attention_k_eq_v=True,
        hidden_size_per_layer_input=0, enable_moe_block=False, num_kv_shared_layers=0, max_position_embeddings=128,
    )  # fmt: skip


def checkpoint(tmp_path, seed=0):
    """The tiny model as a decisio-named checkpoint, with a low-rank embedding so that the layers' inputs correlate."""
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = M.Gemma4TextModel(cfg).eval()
    basis = torch.randn(4, H)
    model.embed_tokens.weight.data = (torch.randn(VOCAB, 4) @ basis) * 0.5
    for p in model.parameters():
        if p.dim() == 2 and p is not model.embed_tokens.weight:
            p.data = p.data * 3.0  # larger weights: the FP8 grid is coarse relative to them, so rounding matters
    d = tmp_path / "ckpt"
    d.mkdir()
    state = {f"model.language_model.{k}": v.clone().to(torch.bfloat16) for k, v in model.state_dict().items()}
    state = {k: v for k, v in state.items() if "rotary_emb" not in k}
    save_file(state, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": cfg.to_dict()}))
    return d, cfg, model


def sequences(n, length, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (length,), generator=g).tolist() for _ in range(n)]


def model_with(cfg, source, codes_by_group=None):
    """A text model whose projections are the dequantized codes (or the originals when None), in fp32."""
    m = M.Gemma4TextModel(cfg).eval()
    state = {k[len("model.language_model.") :]: source.get(k).to(torch.float32) for k in source.map}
    m.load_state_dict({k: v for k, v in state.items()}, strict=False)
    for group in (codes_by_group or {}).values():
        for name, c in group["codes"].items():
            key = name[len("model.language_model.") :]
            m.get_parameter(key).data = c.to(torch.float32) * group["scale"]
    return m


def last_hidden(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([s])).last_hidden_state for s in seqs]


def rtn(source, merged):
    out = {}
    for g in hub_fp8.find_groups(source.map, source.k_eq_v()):
        t = {n: merged(n) for n in dict.fromkeys(g.fused)}
        scale = hub_fp8.scale_of(torch.stack([x.abs().max() for x in t.values()]).max())
        out[g.key] = {"codes": {n: hub_fp8.quantize(t[n], scale) for n in g.names}, "scale": scale}
    return out


@pytest.fixture
def tiny(tmp_path):
    d, cfg, model = checkpoint(tmp_path)
    src = hub_fp8.Source(d)
    merged = lambda n: src.get(n).to(torch.float32)  # noqa: E731
    return d, cfg, src, merged


def test_p1_every_group_gets_codes_and_hub_fp8s_scale(tiny):
    _, cfg, src, merged = tiny
    res = P.gptq_pass(src, merged, sequences(12, 20, 1), dtype=torch.float32)
    groups = hub_fp8.find_groups(src.map, src.k_eq_v())
    assert set(res) == {g.key for g in groups} and len(groups) == 8
    for g in groups:
        r = res[g.key]
        assert set(r["codes"]) == set(g.names) and all(c.dtype == hub_fp8.FP8 for c in r["codes"].values())
        tensors = {n: merged(n) for n in dict.fromkeys(g.fused)}
        want = hub_fp8.scale_of(torch.stack([t.abs().max() for t in tensors.values()]).max())
        assert torch.equal(r["scale"], want)
    kept = P.gptq_pass(src, merged, sequences(6, 20, 1), keep=hub_fp8.parse_keep(["layers:1"]), dtype=torch.float32)
    assert set(kept) == {g.key for g in groups if g.key.startswith("layers.0.")}


def test_p2_each_groups_error_is_below_round_to_nearests(tiny):
    _, _, src, merged = tiny
    res = P.gptq_pass(src, merged, sequences(32, 24, 2), dtype=torch.float32)
    for key, r in res.items():
        assert r["info"]["error_gptq"] <= r["info"]["error_rtn"] * 1.0001, key
    assert sum(r["info"]["error_gptq"] for r in res.values()) < 0.9 * sum(r["info"]["error_rtn"] for r in res.values())


def test_p3_end_to_end_closer_to_the_original_than_round_to_nearest(tiny):
    _, cfg, src, merged = tiny
    res = P.gptq_pass(src, merged, sequences(48, 24, 3), dtype=torch.float32)
    held_out = sequences(24, 24, 77)
    ref = last_hidden(model_with(cfg, src), held_out)
    err = lambda m: sum(float(((a - b) ** 2).mean()) for a, b in zip(ref, last_hidden(m, held_out)))  # noqa: E731
    e_rtn, e_gptq = err(model_with(cfg, src, rtn(src, merged))), err(model_with(cfg, src, res))
    assert e_gptq < e_rtn


def test_p4_the_stored_format_and_a_merged_adapter(tiny):
    _, _, src, merged = tiny
    res = P.gptq_pass(src, merged, sequences(8, 20, 4), dtype=torch.float32)
    for r in res.values():
        for c in r["codes"].values():
            again = hub_fp8.loader_requantize(c, r["scale"], r["scale"].max())
            assert torch.equal(again.view(torch.uint8), c.view(torch.uint8))
    bumped = lambda n: merged(n) * 1.5 if "mlp.down_proj" in n else merged(n)  # noqa: E731  (the adapter's weight)
    res2 = P.gptq_pass(src, bumped, sequences(8, 20, 4), dtype=torch.float32)
    key = "layers.0.mlp.down_proj"
    assert float(res2[key]["scale"]) == pytest.approx(1.5 * float(res[key]["scale"]), rel=1e-5)


def test_p5_refusals(tiny, tmp_path):
    d, _, src, merged = tiny
    with pytest.raises(ValueError, match="no calibration sequences"):
        P.gptq_pass(src, merged, [])
    cfg = json.loads((d / "config.json").read_text())
    cfg["model_type"] = "qwen3"
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not 'qwen3'"):
        P.gptq_pass(hub_fp8.Source(d), merged, sequences(2, 8, 1))
