# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The MLX engine (decisio.serve.mlx_engine) on a tiny random model of the served architecture (qwen3_5_moe: Gated
DeltaNet layers, full attention every fourth layer, a switch MoE), saved in MLX's format with the official tokenizer and
loaded through the engine's own constructor. Apple silicon only: skipped where mlx is not installed. The server's flag
checks need no MLX and run everywhere.

    uv run pytest -q tests/unit/test_mlx_engine.py     (downloads the tokenizer of Qwen/Qwen3.6-35B-A3B once)
"""

import json
import shutil
import sys

import numpy as np
import pytest
from test_prompts import QUESTIONS, STATE, TOKENIZER, _Tokonly

TINY_TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "vocab_size": 248320,
    "tie_word_embeddings": False,
}


def _main(monkeypatch, *argv):
    from decisio.serve import vllm_engine

    monkeypatch.setattr(sys, "argv", ["vllm_engine", "--model", "m", *argv])
    with pytest.raises(SystemExit) as e:
        vllm_engine.main()
    return e.value.code


@pytest.mark.parametrize(
    "flags",
    [["--image-model", "x"], ["--head-engine"], ["--adapter", "a=/p"], ["--mode", "packed"], ["--one-engine"]],
)
def test_mlx_backend_refuses_what_it_does_not_serve(monkeypatch, flags, capsys):
    assert _main(monkeypatch, "--backend", "mlx", *flags) == 2
    assert "--backend mlx serves the text route" in capsys.readouterr().err


def test_mlx_backend_serves_the_31b_however_it_is_named(tmp_path):
    # gemma-4-31b on MLX: by --base, and by a conversion whose config names it (a dense gemma4), with its own settings
    from types import SimpleNamespace

    from decisio.serve.vllm_engine import resolve_base

    d = tmp_path / "gemma-4-31b-it-6bit"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}))
    for base in ("gemma-4-31b", None):
        args = SimpleNamespace(
            backend="mlx",
            base=base,
            model=str(d),
            revision=None,
            prompt_tail=None,
            answer_slot=None,
            label_variants=None,
            system_prompt=None,
            pad_to=None,
            served_name=None,
            noul_rendering=None,
            temperature=None,
            temperature_choice=None,
        )
        fam, fmt = resolve_base(args)
        assert fam.key == "gemma-4-31b" and fmt.system_prompt
        assert (args.temperature, args.temperature_choice, args.noul_rendering) == (5.252, 4.672, "letters")


def _checkpoint_with(tmp_path, name, **config):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}, **config})
    )
    return str(d)


def _mlx_args(model, base=None, backend="mlx"):
    from types import SimpleNamespace

    return SimpleNamespace(
        backend=backend,
        base=base,
        model=model,
        revision=None,
        prompt_tail=None,
        answer_slot=None,
        label_variants=None,
        system_prompt=None,
        pad_to=None,
        served_name=None,
        noul_rendering=None,
        temperature=None,
        temperature_choice=None,
    )


FP8_CONFIGS = {
    "qwen-fp8": {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}},
    "compressed-tensors-fp8": {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float"}}},
        }
    },
    "in-text-config": {"text_config": {"model_type": "gemma4_text", "quantization_config": {"quant_method": "fp8"}}},
}


@pytest.mark.parametrize("name", sorted(FP8_CONFIGS))
def test_mlx_backend_refuses_a_checkpoint_that_declares_fp8(tmp_path, name):
    from decisio.serve.vllm_engine import resolve_base

    model = _checkpoint_with(tmp_path, name, **FP8_CONFIGS[name])
    for base in (None, "gemma-4-31b"):  # --model alone, and --model with --base
        with pytest.raises(ValueError, match="declares FP8 quantization"):
            resolve_base(_mlx_args(model, base))
    resolve_base(_mlx_args(model, "gemma-4-31b", backend="vllm"))  # vLLM is where FP8 is served


def test_mlx_backend_takes_what_does_not_declare_fp8(tmp_path):
    from decisio.families import declares_fp8
    from decisio.serve.vllm_engine import resolve_base

    mlx = _checkpoint_with(tmp_path, "mlx-6bit", quantization={"group_size": 64, "bits": 6})  # an MLX conversion
    _checkpoint_with(
        tmp_path,
        "int4",
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}},
        },
    )
    for model in (mlx, _checkpoint_with(tmp_path, "bf16")):
        assert resolve_base(_mlx_args(model))[0].key == "gemma-4-31b"  # a dense gemma4 checkpoint
    assert not declares_fp8(json.loads((tmp_path / "int4" / "config.json").read_text()))  # int4 is not FP8
    assert declares_fp8({"quantization_config": {"quant_method": "fbgemm_fp8"}})
    assert not declares_fp8({})


def test_mlx_backend_scores_every_question_on_its_own(monkeypatch, capsys):
    # the MLX engine continues each question from a copy of the shared prefix, so its multi-question mode is
    # sequential whatever the base's profile says (gemma-4-31b's is warm), and a flag asking for another is refused
    from decisio.families import BASES
    from decisio.serve.vllm_engine import resolve_multi_question

    assert resolve_multi_question(BASES["gemma-4-31b"], None, None, backend="mlx") == "sequential"
    assert resolve_multi_question(BASES["gemma-4-31b"], "sequential", None, backend="mlx") == "sequential"
    for mode in ("warm", "batch"):
        assert _main(monkeypatch, "--backend", "mlx", "--multi-question", mode) == 2
        assert "--multi-question" in capsys.readouterr().err


def test_pad_policy_defaults_by_backend():
    from decisio.serve.vllm_engine import resolve_pad_policy

    assert resolve_pad_policy("mlx") == "none"  # gated at 6 bits (runs/2026-10-02_mlx-backend, 6bit_none)
    assert resolve_pad_policy("vllm") == "always" and resolve_pad_policy("hf") == "always"
    assert resolve_pad_policy("mlx", "always") == "always" and resolve_pad_policy("vllm", "shared") == "shared"


def test_pad_policy_none_never_pads_and_alone_is_inside():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    padded = _Tokonly(tok, 1056, "front")
    none = _Tokonly(tok, 1056, "front")
    none.pad_policy = "none"
    rows, P = none._prepare_separate(STATE, QUESTIONS)
    d_rows, dP = padded._prepare_separate(STATE, QUESTIONS)
    k = dP - P
    assert k > 0 and all(d[0][:k] == [none.pad_token] * k and d[0][k:] == r[0] for r, d in zip(rows, d_rows))
    for q, row in zip(QUESTIONS, rows):  # a question alone is the same prompt as inside the request
        assert none._prepare_separate(STATE, [q])[0][0] == row


def test_prefix_cache_flag_is_for_mlx_only(monkeypatch, capsys):
    assert _main(monkeypatch, "--backend", "hf", "--prefix-cache-mb", "0") == 2
    assert "--prefix-cache-mb is for --backend mlx" in capsys.readouterr().err


def test_tokenizer_flag_is_for_mlx_only(monkeypatch, capsys):
    assert _main(monkeypatch, "--backend", "hf", "--tokenizer", "t") == 2
    assert "--tokenizer is for --backend mlx" in capsys.readouterr().err


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5_moe

    from decisio.serve.mlx_engine import MLXLettersEngine

    mx.random.seed(0)
    config = {"model_type": "qwen3_5_moe", "text_config": TINY_TEXT}
    model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
    d = tmp_path_factory.mktemp("tiny_mlx")
    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (d / "config.json").write_text(json.dumps(config))
    tok_dir = snapshot_download(TOKENIZER, allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*.jinja"])
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        shutil.copy(f"{tok_dir}/{f}", d / f)
    return MLXLettersEngine(str(d), tokenizer=str(d))


@pytest.fixture(scope="module")
def tiny_no_prefix_cache(tiny):
    from decisio.serve.mlx_engine import MLXLettersEngine

    return MLXLettersEngine(tiny.model_name, tokenizer=tiny.tokenizer_name, prefix_cache_mb=0)


def test_rows_are_the_served_rows(tiny):
    # the served row builder on the same tokenizer, with the served padding (front, to the 1,056-token block)
    assert tiny._prepare_separate(STATE, QUESTIONS) == _Tokonly(tiny.tok, 1056, "front")._prepare_separate(
        STATE, QUESTIONS
    )


def test_answers_are_isolated_and_repeat_bit_for_bit(tiny):
    together = tiny.answer(STATE, QUESTIONS)[0]
    alone = [tiny.answer(STATE, [q])[0][0] for q in QUESTIONS]
    filler = [{"kind": "noul", "instructions": f"Is the ticket id greater than {i}?"} for i in range(16)]
    filled = tiny.answer(STATE, QUESTIONS + filler)[0][: len(QUESTIONS)]
    again = tiny.answer(STATE, QUESTIONS)[0]
    for other in (alone, filled, again):
        assert all(np.array_equal(a, b) for a, b in zip(together, other))
    for p, q in zip(together, QUESTIONS):
        assert len(p) == (2 if q["kind"] == "noul" else len(q["options"])) and abs(p.sum() - 1) < 1e-12


def test_every_question_continues_from_the_padded_state_prefix(tiny):
    probs, info = tiny.answer(STATE, QUESTIONS)
    rows, P = tiny._prepare_separate(STATE, QUESTIONS)
    assert P % 1056 == 0 and info["shared_prefix_tokens"] == P and info["cached_tokens"] == [P] * len(QUESTIONS)


def test_hidden_readout_is_the_served_readout(tiny):
    from decisio.serve.mlx_engine import MLXHiddenReadout

    hidden = MLXHiddenReadout(tiny).readout(STATE, QUESTIONS)
    served = tiny.answer(STATE, QUESTIONS)[0]
    for (lp, h), p in zip(hidden, served):
        assert np.array_equal(np.exp(lp) / np.exp(lp).sum(), p)
        assert h.shape == (TINY_TEXT["hidden_size"],) and h.dtype == np.float32


def test_the_probe_answers_and_changes_no_answer(tiny):
    """The engine's probe (decisio.serve.engine_health): a one-token forward pass into a new cache."""
    from decisio.serve.mlx_engine import MLXHiddenReadout

    before = tiny.answer(STATE, QUESTIONS)[0]
    tiny.probe()
    MLXHiddenReadout(tiny).probe()
    assert all(np.array_equal(a, b) for a, b in zip(before, tiny.answer(STATE, QUESTIONS)[0]))


def test_copy_cache_leaves_the_prefix_untouched(tiny):
    import mlx.core as mx

    from decisio.serve.mlx_engine import copy_cache

    rows, P = tiny._prepare_separate(STATE, QUESTIONS[:1])
    base = tiny.model.make_cache()
    tiny._feed(rows[0][0][:P], base)

    def arrays():
        return [
            np.array(a.astype(mx.float32))
            for c in base
            for a in (c.cache if hasattr(c, "cache") else (c.keys, c.values))
        ]

    before = arrays()
    for _ in range(2):
        tiny._feed(rows[0][0][P:], copy_cache(base))
    after = arrays()
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    assert all(c.offset == P for c in base if hasattr(c, "offset"))


def test_the_engine_refuses_images_adapters_and_packed_mode(tiny):
    rows, _ = tiny._prepare_separate(STATE, QUESTIONS[:1])
    with pytest.raises(ValueError, match="text route"):
        tiny.score_prompts(rows, mm={"image": [1]})
    with pytest.raises(ValueError, match="no adapters"):
        tiny.answer(STATE, QUESTIONS[:1], adapter="a")
    with pytest.raises(ValueError, match="separate mode"):
        tiny._answer_packed([(STATE, QUESTIONS[:1])])


def test_prefix_cache_answers_are_bit_identical_with_and_without_it(tiny, tiny_no_prefix_cache):
    from decisio.serve.mlx_engine import MLXHiddenReadout

    states = [STATE, "Another state, for the cache to hold two entries. " * 20]
    plain = [tiny_no_prefix_cache.answer(st, QUESTIONS)[0] for st in states]
    hits_before = tiny.prefix_cache.hits
    first = [tiny.answer(st, QUESTIONS) for st in states]  # misses (or hits from earlier tests): either way exact
    again = [tiny.answer(st, QUESTIONS) for st in states]  # hits
    assert tiny.prefix_cache.hits >= hits_before + len(states)
    assert all(info["prefix_cache_hits"] == 1 for _, info in again)
    for p, (f, _), (g, _) in zip(plain, first, again):
        assert all(np.array_equal(x, y) and np.array_equal(x, z) for x, y, z in zip(p, f, g))
    for st in states:  # the head's hidden readout through the cache is the uncached one, bit for bit
        a = MLXHiddenReadout(tiny).readout(st, QUESTIONS)
        b = MLXHiddenReadout(tiny_no_prefix_cache).readout(st, QUESTIONS)
        assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert tiny_no_prefix_cache.facts()["prefix_cache"] is None and tiny.facts()["prefix_cache"]["entries"] >= 2


def test_prefix_cache_evicts_least_recent_and_never_returns_other_tokens(tiny, monkeypatch):
    from decisio.serve.mlx_engine import PrefixCache

    rows, P = tiny._prepare_separate(STATE, QUESTIONS[:1])
    base = tiny.model.make_cache()
    tiny._feed(rows[0][0][:P], base)
    from decisio.serve.mlx_engine import _cache_arrays

    n = sum(a.nbytes for a in _cache_arrays(base))
    cache = PrefixCache(2 * n)
    for k in range(3):
        cache.put([k, 1, 2], base)
    assert len(cache.entries) == 2 and cache.get([0, 1, 2]) is None and cache.get([2, 1, 2]) is base
    monkeypatch.setattr(PrefixCache, "_key", staticmethod(lambda ids: "same"))  # a hash collision
    collide = PrefixCache(2 * n)
    collide.put([1, 2, 3], base)
    assert collide.get([4, 5, 6]) is None and collide.get([1, 2, 3]) is base


def test_official_tokenizer_by_base():
    from decisio.families import FAMILIES, GEMMA4, QWEN
    from decisio.serve.mlx_engine import OFFICIAL_TOKENIZER, TOKENIZER_SHA256, split_revision

    # every base has its tokenizer's sha256 (the two Gemma 4 bases' files are identical); the tokenizer is the base's
    # own checkpoint at its revision
    served = sorted(f.key for f in FAMILIES)
    assert sorted(TOKENIZER_SHA256) == served and OFFICIAL_TOKENIZER == QWEN.model
    assert (GEMMA4.model, GEMMA4.revision) == ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
    assert split_revision("google/gemma-4-12B-it@707f0a3b") == ("google/gemma-4-12B-it", "707f0a3b")
    assert split_revision("Qwen/Qwen3.6-35B-A3B-FP8") == ("Qwen/Qwen3.6-35B-A3B-FP8", None)


@pytest.mark.parametrize("steps", [[1, 1, 1], [5], [3, 1]])
def test_copy_cache_rotating_leaves_the_original_untouched(steps):
    # a sliding-window layer (Gemma 4) past its window: continuing a copy by one token (written in place) or several
    # (concatenated) leaves the original's arrays, offset and write position as they were, and two copies agree
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import RotatingKVCache

    from decisio.serve.mlx_engine import copy_cache

    mx.random.seed(0)
    base = RotatingKVCache(max_size=8)
    pre = mx.random.normal((1, 2, 11, 4))
    base.update_and_fetch(pre, pre)
    base.update_and_fetch(pre[..., :1, :], pre[..., :1, :])  # a one-token step: the buffer now rotates in place
    before = (np.array(base.keys), np.array(base.values), base.offset, base._idx)
    outs = []
    for _ in range(2):
        (c,) = copy_cache([base])
        for n in steps:
            x = mx.random.normal((1, 2, n, 4))
            k, v = c.update_and_fetch(x, x)
        outs.append((c.offset, c._idx))
    after = (np.array(base.keys), np.array(base.values), base.offset, base._idx)
    assert all(np.array_equal(a, b) for a, b in zip(before[:2], after[:2])) and before[2:] == after[2:]
    assert outs[0] == outs[1] == (base.offset + sum(steps), outs[0][1])


def _served_budget(monkeypatch, tmp_path, base, *argv):
    """The prefix cache budget (MiB) that `--backend mlx --base <base>` hands the MLX engine (stubbed)."""
    import uvicorn

    import decisio.serve.mlx_engine as mlx
    from decisio.families import BASES
    from decisio.serve import vllm_engine

    got = {}

    class Engine:
        adapters = {}
        pad_unit = None

        def __init__(self, model, **kw):
            got.update(kw)

        def facts(self):
            return {}

    d = tmp_path / base
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": BASES[base].model_types[0]}))
    monkeypatch.setattr(mlx, "MLXLettersEngine", Engine)
    monkeypatch.setattr(mlx, "MLXHiddenReadout", lambda engine: type("H", (), {"facts": lambda self: {}})())
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(sys, "argv", ["vllm_engine", "--backend", "mlx", "--base", base, "--model", str(d), *argv])
    vllm_engine.main()
    return got["prefix_cache_mb"]


def test_prefix_cache_budget_is_the_bases_and_the_flag_overrides_it(monkeypatch, tmp_path):
    from decisio.families import BASES

    # Qwen's 2,048 MiB keeps 25 states of 1,000 tokens; the 12B's 7,400 keeps 20
    # (RLCD experiments/2026-10-06_ls_mlx_cache_budget)
    assert BASES["qwen3.6-35b-a3b"].mlx_prefix_cache_mb == 2048 and BASES["gemma-4-12b"].mlx_prefix_cache_mb == 7400
    assert _served_budget(monkeypatch, tmp_path, "qwen3.6-35b-a3b") == 2048
    assert _served_budget(monkeypatch, tmp_path, "gemma-4-12b") == 7400
    assert _served_budget(monkeypatch, tmp_path, "gemma-4-12b", "--prefix-cache-mb", "1024") == 1024
    assert _served_budget(monkeypatch, tmp_path, "gemma-4-12b", "--prefix-cache-mb", "0") == 0  # off, not the default


def test_the_12b_budget_stays_under_two_thirds_of_a_32_gib_mac():
    # the arithmetic behind 7,400: 32 GiB x 2/3 = 22.906 GB (decimal), less the 12B's 15.125 GB peak at 32k tokens
    from decisio.families import BASES

    line_gb = 32 * 2**30 * 2 / 3 / 1e9
    peak_gb = 15.125
    budget_gb = BASES["gemma-4-12b"].mlx_prefix_cache_mb * 2**20 / 1e9
    assert round(line_gb, 3) == 22.906 and budget_gb < line_gb - peak_gb < 7.8
