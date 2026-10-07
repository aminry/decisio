# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The FP8 checkpoint of the Gemma 4 31B (decisio.hub_fp8) and the engine's fingerprint of it, on a tiny checkpoint with
the 31B's structure; no model, no card (the identity gate itself runs on one: RLCD
experiments/2026-10-07_sd_hf_model_repos/PLAN.md).

G1  the groups vLLM fuses: q, k and v; gate and up; o_proj and down_proj alone; a layer with no v_proj repeats k
    (attention_k_eq_v) and is refused without that flag; the towers and the norms are not quantized
Q1  the arithmetic of vLLM's online per-tensor FP8: scale = float32(amax) / 448 over the fused weight, then the weight
    times the fp32 reciprocal of the scale, clamped and rounded to nearest even
C1  the converted checkpoint: every projection FP8 with its scale beside it under the source's names, every other
    tensor byte for byte, the config naming the scheme, the tokenizer files, files of the size asked, a report
R1  loading it (vLLM's requantize_with_max_scale): exact for ordinary scales, and a layer of tiny scale shows up as
    changed weights in the report and as a differing module in the fingerprints
E1  the engine's fingerprint method puts a transposed FP8 weight back to [out, in] and ignores what is not FP8

    uv run pytest -q tests/unit/test_hub_fp8.py
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from decisio import hub_fp8
from decisio.vllm_plugin.worker import DecisioWorkerExtension

L = "model.language_model.layers"
FP8 = torch.float8_e4m3fn


def bf16(*shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


def checkpoint(tmp_path, small_layer=False):
    """Layer 0 sliding (q, k, v), layer 1 full attention with no v_proj; MLPs; norms; a vision tower; two shards."""
    t = {}
    for i, (has_v, s) in enumerate([(True, 0.05), (False, 0.05)]):
        s_mlp = 1e-4 if (small_layer and i == 1) else 0.05
        t[f"{L}.{i}.self_attn.q_proj.weight"] = bf16(8, 6, scale=s, seed=10 * i + 1)
        t[f"{L}.{i}.self_attn.k_proj.weight"] = bf16(4, 6, scale=s, seed=10 * i + 2)
        if has_v:
            t[f"{L}.{i}.self_attn.v_proj.weight"] = bf16(4, 6, scale=s, seed=10 * i + 3)
        t[f"{L}.{i}.self_attn.o_proj.weight"] = bf16(6, 8, scale=s, seed=10 * i + 4)
        t[f"{L}.{i}.mlp.gate_proj.weight"] = bf16(10, 6, scale=s_mlp, seed=10 * i + 5)
        t[f"{L}.{i}.mlp.up_proj.weight"] = bf16(10, 6, scale=s_mlp, seed=10 * i + 6)
        t[f"{L}.{i}.mlp.down_proj.weight"] = bf16(6, 10, scale=s_mlp, seed=10 * i + 7)
        t[f"{L}.{i}.input_layernorm.weight"] = bf16(6, scale=1.0, seed=10 * i + 8)
        t[f"{L}.{i}.layer_scalar"] = torch.ones(1, dtype=torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = bf16(20, 6, seed=90)
    t["model.vision_tower.encoder.layers.0.self_attn.q_proj.weight"] = bf16(4, 6, seed=91)
    d = tmp_path / "bf16"
    d.mkdir()
    names = sorted(t)
    halves = [names[: len(names) // 2], names[len(names) // 2 :]]
    weight_map = {}
    for n, part in enumerate(halves, start=1):
        f = f"model-0000{n}-of-00002.safetensors"
        save_file({k: t[k] for k in part}, str(d / f), metadata={"format": "pt"})
        weight_map.update({k: f for k in part})
    (d / hub_fp8.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    cfg = {"model_type": "gemma4", "text_config": {"attention_k_eq_v": True, "num_hidden_layers": 2}}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text('{"tokenizer": true}')
    (d / "chat_template.jinja").write_text("{{ x }}")
    return d, t


def read_all(directory):
    out = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118
    return out


def test_g1_the_groups_are_what_vllm_fuses(tmp_path):
    d, t = checkpoint(tmp_path)
    groups = hub_fp8.find_groups(t, k_eq_v=True)
    assert [g.key for g in groups] == [
        "layers.0.self_attn.qkv_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_up_proj",
        "layers.0.mlp.down_proj",
        "layers.1.self_attn.qkv_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.mlp.gate_up_proj",
        "layers.1.mlp.down_proj",
    ]
    by = {g.key: g for g in groups}
    assert by["layers.0.self_attn.qkv_proj"].fused == [f"{L}.0.self_attn.{p}_proj.weight" for p in "qkv"]
    # a layer with no v_proj: V is a copy of K, so the fused weight is [q; k; k] and k_proj is one tensor in the file
    full = by["layers.1.self_attn.qkv_proj"]
    assert full.fused == [f"{L}.1.self_attn.{p}_proj.weight" for p in "qkk"]
    assert full.names == [f"{L}.1.self_attn.q_proj.weight", f"{L}.1.self_attn.k_proj.weight"]
    assert by["layers.0.mlp.gate_up_proj"].fused == [f"{L}.0.mlp.gate_proj.weight", f"{L}.0.mlp.up_proj.weight"]
    quantized = {n for g in groups for n in g.names}
    assert not any("vision_tower" in n or "layernorm" in n or "embed" in n for n in quantized)
    with pytest.raises(ValueError, match="attention_k_eq_v"):
        hub_fp8.find_groups(t, k_eq_v=False)


def test_g1_an_incomplete_mlp_is_refused():
    names = {f"{L}.0.mlp.gate_proj.weight", f"{L}.0.mlp.down_proj.weight"}
    with pytest.raises(ValueError, match="expected"):
        hub_fp8.find_groups(names)


def test_q1_the_arithmetic_of_online_per_tensor_fp8():
    w = torch.tensor([[2.0, -1.0, 0.0, 0.5, -2.0]], dtype=torch.bfloat16)
    amax = hub_fp8.amax_of([w])
    assert float(amax) == 2.0
    scale = hub_fp8.scale_of(amax)
    assert scale.dtype == torch.float32 and scale.shape == (1,) and float(scale) == pytest.approx(2.0 / 448.0)
    q = hub_fp8.quantize(w, scale)
    assert q.dtype == FP8
    assert q.to(torch.float32).tolist() == [[448.0, -224.0, 0.0, 112.0, -448.0]]  # exact powers of two times the scale
    # clamped, not wrapped: a value above the amax the scale was made for saturates
    big = hub_fp8.quantize(torch.tensor([[1000.0]], dtype=torch.bfloat16), scale)
    assert float(big.to(torch.float32)) == 448.0
    # one amax over several tensors: the fused weight's
    assert float(hub_fp8.amax_of([bf16(3, 3), torch.full((2, 2), -7.0, dtype=torch.bfloat16)])) == 7.0


def test_q1_the_reciprocal_is_taken_in_float32():
    scale = torch.tensor([0.123456789], dtype=torch.float32)
    w = bf16(16, 16, scale=0.3, seed=5)
    inv = torch.reciprocal(scale)
    expected = (w.to(torch.float32) * inv).clamp(-448, 448).to(FP8)
    assert torch.equal(hub_fp8.quantize(w, scale).view(torch.uint8), expected.view(torch.uint8))
    assert inv.dtype == torch.float32


@pytest.mark.parametrize("scheme", ["compressed-tensors", "fp8"])
def test_c1_the_converted_checkpoint(tmp_path, scheme):
    src, t = checkpoint(tmp_path)
    out = tmp_path / f"out-{scheme}"
    report = hub_fp8.convert(src, out, scheme)
    written = read_all(out)
    groups = hub_fp8.find_groups(t, k_eq_v=True)
    quantized = {n for g in groups for n in g.names}
    # every projection FP8 with its scale beside it; the layer with no v_proj has none here either
    for n in quantized:
        s = n[: -len(".weight")] + ".weight_scale"
        assert written[n].dtype == FP8 and written[n].shape == t[n].shape
        assert written[s].dtype == torch.float32 and written[s].shape == (1,)
    assert f"{L}.1.self_attn.v_proj.weight" not in written and f"{L}.1.self_attn.v_proj.weight_scale" not in written
    # every other tensor, the vision tower included, is the source's, byte for byte
    for n in set(t) - quantized:
        assert written[n].dtype == t[n].dtype and torch.equal(written[n], t[n]), n
    assert set(written) == set(t) | {n[: -len(".weight")] + ".weight_scale" for n in quantized}
    # the index names every tensor, in a file that holds it
    index = json.loads((out / hub_fp8.INDEX).read_text())["weight_map"]
    assert set(index) == set(written) and all((out / f).exists() for f in set(index.values()))
    # the config names the scheme and keeps the source's own
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "gemma4" and cfg["text_config"]["attention_k_eq_v"] is True
    qc = cfg["quantization_config"]
    assert qc["quant_method"] == scheme
    assert ("activation_scheme" in qc and qc["activation_scheme"] == "dynamic") or scheme == "compressed-tensors"
    assert "re:.*vision_tower.*" in qc.get("ignore", qc.get("modules_to_not_convert"))
    for name in ("tokenizer.json", "chat_template.jinja"):
        assert (out / name).read_text() == (src / name).read_text()
    assert report["scheme"] == scheme and report["tensors"] == len(written)
    assert json.loads((out / "fp8_conversion.json").read_text())["tensors"] == len(written)


def test_c1_the_scale_is_one_per_fused_layer_and_the_values_follow_from_it(tmp_path):
    src, t = checkpoint(tmp_path)
    out = tmp_path / "out"
    hub_fp8.convert(src, out)
    w = read_all(out)
    for i, parts in ((0, "qkv"), (1, "qk")):
        names = [f"{L}.{i}.self_attn.{p}_proj.weight" for p in parts]
        amax = max(float(t[n].abs().max()) for n in names)
        scales = [w[n[: -len(".weight")] + ".weight_scale"] for n in names]
        assert all(torch.equal(s, scales[0]) for s in scales) and float(scales[0]) == pytest.approx(amax / 448.0)
        for n in names:
            assert torch.equal(w[n].view(torch.uint8), hub_fp8.quantize(t[n], scales[0]).view(torch.uint8))
    # the layer with no v_proj: the scale is over q and k, which is what [q; k; k] gives
    rep = json.loads((out / "fp8_conversion.json").read_text())["groups"]["layers.1.self_attn.qkv_proj"]
    assert rep["shape"] == [8 + 4 + 4, 6]
    # gate and up share one scale, o_proj and down_proj have their own
    g, u = (w[f"{L}.0.mlp.{p}_proj.weight_scale"] for p in ("gate", "up"))
    assert torch.equal(g, u) and not torch.equal(g, w[f"{L}.0.mlp.down_proj.weight_scale"])


def test_c1_files_are_the_size_asked(tmp_path):
    src, t = checkpoint(tmp_path)
    out = tmp_path / "out"
    hub_fp8.convert(src, out, shard_bytes=200)
    files = sorted(out.glob("model-*.safetensors"))
    n = len(files)
    assert n >= 4 and [f.name for f in files] == [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    biggest = max(t[k].numel() * 2 for k in t) + 200  # a file stops growing once it holds `limit` bytes
    assert all(f.stat().st_size < biggest + 2000 for f in files)
    assert not list(out.glob("part-*"))
    assert set(read_all(out)) == set(json.loads((out / hub_fp8.INDEX).read_text())["weight_map"])


def test_r1_loading_is_exact_for_ordinary_scales(tmp_path):
    src, _ = checkpoint(tmp_path)
    out = tmp_path / "out"
    report = hub_fp8.convert(src, out)
    assert report["loader_changed_elements"] == 0 and report["smallest_amax"] > 0.02
    online, loaded = hub_fp8.expected_online(src), hub_fp8.expected_loaded(out)
    assert set(online) == set(loaded) and len(online) == 8
    assert hub_fp8.compare(online, loaded) == []
    assert online["layers.0.self_attn.qkv_proj"]["shape"] == [16, 6]
    assert all(g["sha256_online"] == g["sha256_loaded"] for g in report["groups"].values())
    assert online["layers.1.mlp.down_proj"]["sha256"] == report["groups"]["layers.1.mlp.down_proj"]["sha256_online"]


def test_r1_a_layer_of_tiny_scale_shows_up(tmp_path, capsys):
    src, _ = checkpoint(tmp_path, small_layer=True)
    out = tmp_path / "out"
    report = hub_fp8.convert(src, out)
    changed = {k: g["loader_changed"] for k, g in report["groups"].items() if g["loader_changed"]}
    assert changed and set(changed) <= {"layers.1.mlp.gate_up_proj", "layers.1.mlp.down_proj"}
    assert report["loader_changed_elements"] == sum(changed.values()) and report["smallest_amax"] < 0.01
    differ = hub_fp8.compare(hub_fp8.expected_online(src), hub_fp8.expected_loaded(out))
    assert differ == sorted(changed)  # the fingerprints name exactly the layers the report does
    # the command line says so with its exit code
    assert hub_fp8.main(["convert", "--source", str(src), "--out", str(tmp_path / "again")]) == 1
    assert "would change" in capsys.readouterr().out


def test_r1_the_loader_step_is_the_identity_where_float16_holds_every_product():
    # every FP8 value at a scale of an ordinary layer round-trips (the property the gate relies on) ...
    bits = torch.arange(256, dtype=torch.uint8).view(FP8)
    q = bits[torch.isfinite(bits.to(torch.float32))]
    for amax in (0.05, 0.5, 5.0):
        scale = hub_fp8.scale_of(torch.tensor(amax))
        assert torch.equal(hub_fp8.loader_requantize(q, scale, scale).view(torch.uint8), q.view(torch.uint8))
    # ... and at a very small one it does not
    scale = hub_fp8.scale_of(torch.tensor(1e-3))
    assert not torch.equal(hub_fp8.loader_requantize(q, scale, scale).view(torch.uint8), q.view(torch.uint8))


def test_r1_the_command_line(tmp_path, capsys):
    src, _ = checkpoint(tmp_path)
    out = tmp_path / "out"
    assert hub_fp8.main(["convert", "--source", str(src), "--out", str(out), "--scheme", "fp8"]) == 0
    a, b = tmp_path / "online.json", tmp_path / "loaded.json"
    assert hub_fp8.main(["expected", "--source", str(src), "--out", str(a)]) == 0
    assert hub_fp8.main(["expected", "--checkpoint", str(out), "--out", str(b)]) == 0
    capsys.readouterr()
    assert hub_fp8.main(["compare", str(a), str(b)]) == 0 and "identical" in capsys.readouterr().out
    changed = json.loads(b.read_text())
    changed["layers.0.mlp.down_proj"]["sha256"] = "0" * 64
    b.write_text(json.dumps(changed))
    assert hub_fp8.main(["compare", str(a), str(b)]) == 1 and "layers.0.mlp.down_proj" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        hub_fp8.main(["expected", "--out", str(a)])


class Linear(torch.nn.Module):
    def __init__(self, weight, scale, n_in, n_out):
        super().__init__()
        self.weight, self.weight_scale = weight, scale
        self.input_size, self.output_size = n_in, n_out


def engine_with(modules):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    for path, m in modules.items():
        owner, name = path.split(".")
        setattr(getattr(layer, owner), name, m)
    ext = DecisioWorkerExtension()
    ext.model_runner = SimpleNamespace(get_model=lambda: model)
    return ext


def test_e1_the_engine_fingerprint_puts_the_weight_back_to_out_by_in():
    q = hub_fp8.quantize(bf16(12, 5, scale=0.1, seed=3), torch.tensor([0.001]))  # [out, in] as the checkpoint has it
    scale = torch.tensor([0.001])
    ext = engine_with(
        {
            "self_attn.qkv_proj": Linear(q.t(), scale, 5, 12),  # vLLM keeps it transposed ([in, out])
            "mlp.down_proj": Linear(
                q.clone(), scale, 12, 5
            ),  # [out, in] already is [5, 12]? no: shape (12, 5) is [in, out]
            "mlp.gate_up_proj": Linear(torch.zeros(3, 3, dtype=torch.bfloat16), scale, 3, 3),  # not FP8: ignored
        }
    )
    got = ext.decisio_fp8_fingerprints()
    assert set(got) == {"layers.0.self_attn.qkv_proj", "layers.0.mlp.down_proj"}
    want = hub_fp8.fingerprint(q, scale)
    assert got["layers.0.self_attn.qkv_proj"] == want
    # down_proj holds (12, 5) with in = 12 and out = 5, i.e. [in, out] again: it is put back to [5, 12]
    assert got["layers.0.mlp.down_proj"] == hub_fp8.fingerprint(q.t(), scale)
    assert got["layers.0.mlp.down_proj"]["shape"] == [5, 12]


def test_e1_without_a_model_the_engine_says_nothing():
    assert DecisioWorkerExtension().decisio_fp8_fingerprints() is None
    ext = DecisioWorkerExtension()
    ext.model_runner = SimpleNamespace(get_model=lambda: None)
    assert ext.decisio_fp8_fingerprints() is None
