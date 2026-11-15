# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images on a Mac (`--backend mlx --images on`, decisio.serve.mlx_images) on a tiny random model of the 12B's
architecture (`gemma4_unified`, built here with random weights: no weights are downloaded), under the real Gemma 4
tokenizer. Apple silicon with the `mlx-images` extra only: the engine tests are skipped where mlx-vlm is not
installed; the refusals that need no MLX run everywhere.

  M1  placement, scope and usage are the vLLM route's: an image stands at its reference, a question's image is in its
      row alone, the usage counts come from the processor's arithmetic, the engine's prompts are the expected ones
  M2  the image token types reach the model: the blocks are bidirectional, which changes the answer, and the answer
      equals the one of mlx-vlm's language model given the same token types (the library's own overlay)
  M3  a question's image after the cached state prefix is read as a cold read of the whole prompt reads it
  M4  the prefix cache keeps a state with its images: a second question reads from it with the same answer as a cold
      read, and another picture of the same size is another entry
  M5  chunked prefill never cuts an image block, and a chunked read equals an unchunked one
  M6  refusals: a conversion without vision weights, the 31B, adapters, more images than --max-images, Box
  M7  `generate`: the top-k log-probabilities of a greedy decode on the image prompt

  uv run pytest -q tests/unit/test_mlx_images.py     (the Gemma 4 tokenizer downloads once)
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from decisio.families import GEMMA4
from decisio.readout.letters import PromptFormat
from decisio.serve import image_marks as marks

FORMAT = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")
TAG = "0123456789abcdef"
STATE = "Photo:" + marks.mark(1, TAG) + "Title: Blue ceramic mug, 350 ml\nPrice: 12 EUR"
QS = [
    {"kind": "noul", "instructions": "Does the photo show the listed item?"},
    {"kind": "choice", "instructions": "Which colour is the mug?", "options": ["blue", "red", "white"]},
]


def picture(w=96, h=96, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def vision_dir(tmp_path, model_type="gemma4_unified", vision=True, index=True, name="conv"):
    d = tmp_path / name
    d.mkdir()
    names = ["language_model.model.embed_tokens.weight"]
    if vision:
        names += ["vision_embedder.patch_dense.weight", "embed_vision.embedding_projection.weight"]
    cfg = {"model_type": model_type, **({"vision_config": {"model_type": "x"}} if vision else {})}
    (d / "config.json").write_text(json.dumps(cfg))
    if index:
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "m.safetensors" for n in names}}))
    else:
        header = json.dumps({n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 0]} for n in names}).encode()
        (d / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    return d


# ---- M6 (no MLX needed) ----------------------------------------------------------------------------------------------


def test_m6_a_conversion_without_vision_weights_is_refused_naming_what_is_missing(tmp_path):
    from decisio.serve.mlx_images import vision_refusal

    d = vision_dir(tmp_path, vision=False)
    why = vision_refusal(d)
    assert "no vision weights" in why and "vision_embedder, embed_vision" in why
    assert "mlx-community/gemma-4-12B-it-6bit" in why


@pytest.mark.parametrize("index", [True, False])
def test_m6_a_conversion_with_the_weights_is_accepted(tmp_path, index):
    from decisio.serve.mlx_images import vision_refusal

    assert vision_refusal(vision_dir(tmp_path, index=index)) is None  # by the index file or the safetensors header


def test_m6_the_31b_without_a_tower_is_refused(tmp_path):
    from decisio.serve.mlx_images import vision_refusal

    why = vision_refusal(vision_dir(tmp_path, model_type="gemma4", vision=False))
    assert "31B" in why and "without its vision tower" in why


def test_m6_the_31b_with_a_tower_is_not_served_yet(tmp_path):
    from decisio.serve.mlx_images import vision_refusal

    assert "serves the 12B" in vision_refusal(vision_dir(tmp_path, model_type="gemma4"))


def test_m6_another_architecture_is_refused(tmp_path):
    from decisio.serve.mlx_images import vision_refusal

    assert "Gemma 4 bases" in vision_refusal(vision_dir(tmp_path, model_type="qwen3_5_moe", vision=False))


def _main(monkeypatch, *argv):
    from decisio.serve import vllm_engine

    monkeypatch.setattr(sys, "argv", ["vllm_engine", *argv])
    with pytest.raises(SystemExit) as e:
        vllm_engine.main()
    return e.value.code


def test_m6_the_31b_refuses_images_on_a_mac(monkeypatch, capsys):
    assert _main(monkeypatch, "--backend", "mlx", "--base", "gemma-4-31b", "--model", "m", "--images", "on") == 2
    err = capsys.readouterr().err
    assert "serves the 12B" in err and "no vision tensors" in err


def test_m6_the_qwen_base_refuses_images_on_a_mac(monkeypatch, capsys):
    assert _main(monkeypatch, "--backend", "mlx", "--base", "qwen3.6-35b-a3b", "--model", "m", "--images", "on") == 2
    assert "Gemma bases" in capsys.readouterr().err


def test_m6_a_server_that_is_neither_vllm_nor_mlx_refuses_images(monkeypatch, capsys):
    assert _main(monkeypatch, "--backend", "hf", "--base", "gemma-4-12b", "--model", "m", "--images", "on") == 2
    assert "vLLM server and --backend mlx" in capsys.readouterr().err


def test_m6_the_12b_gets_its_own_working_set_for_images():
    from decisio.families import GEMMA4, GEMMA4_31B

    assert GEMMA4.mlx_working_images_gb > GEMMA4.mlx_working_gb > 0
    assert GEMMA4_31B.mlx_working_images_gb == 0


# ---- the engine on a tiny random model -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_stack():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("mlx_lm")
    return mx


def make(tmp_path_factory, mx, budget=70, max_images=4, prefill_step=2048, cache_mb=256, seed=0):
    from decisio.serve.mlx_images import MLXGemmaImageLettersEngine

    d = tmp_path_factory.mktemp("tiny_images")
    e = MLXGemmaImageLettersEngine(
        str(d),
        image_budget=budget,
        max_images=max_images,
        random={"seed": seed},
        family=GEMMA4,
        prefill_step=prefill_step,
        prefix_cache_mb=cache_mb,
        memory_limit_gb=0,
    )
    e.fmt, e.family, e.pad_policy = FORMAT, GEMMA4, "none"
    return e


@pytest.fixture(scope="module")
def engine(tmp_path_factory, mlx_stack):
    return make(tmp_path_factory, mlx_stack)


def reference_letters(e, ids_exp, images, labels, native=True):
    """The letters of a cold read of the expanded ids by mlx-vlm's own language model (the library's overlay, given the
    token types) when `native`, else with every image causal (no token types)."""
    mx = e._mx
    types = np.array([[int(t == e.image_token_id) for t in ids_exp]])
    emb = e._embed(mx.array([ids_exp])) * e._embed_scale
    pieces, at = [], 0
    arr = np.array(ids_exp)
    k, i = 0, 0
    while i < len(arr):
        if arr[i] == e.image_token_id:
            j = i
            while j < len(arr) and arr[j] == e.image_token_id:
                j += 1
            px, pos, _ = e._processed(images[k])
            pieces += [emb[:, at:i], e.model.get_image_features(px, pos).astype(emb.dtype)[None]]
            at, i, k = j, j, k + 1
        else:
            i += 1
    pieces.append(emb[:, at:])
    x = mx.concatenate(pieces, axis=1)
    kw = {"mm_token_type_ids": mx.array(types)} if native else {}
    h = e.model.language_model.model(None, inputs_embeds=x, cache=e.model.make_cache(), **kw)[0, -1]
    from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax

    z = np.array(e._logits(h, allowed_ids(labels)).astype(mx.float32), dtype=np.float64)
    lp = label_log_softmax(z, labels) if is_grouped(labels) else z - z.max()
    p = np.exp(lp)
    return p / p.sum()


def one_row(e, state, question, images):
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(state, images, [question], TAG)
    return rows, p_sent, row_found, counts


def test_m1_placement_scope_and_usage(engine):
    e = engine
    imgs = [picture(), picture(w=200, h=120, seed=1)]
    qs = [
        {"kind": "noul", "instructions": f"Does the photo match {marks.mark(2, TAG)}?"},
        {"kind": "noul", "instructions": "Is the mug blue?"},
    ]
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(STATE, imgs, qs, TAG)
    assert state_found == [1] and row_found == [[1, 2], [1]]
    assert rows[0][0][:p_sent] == rows[1][0][:p_sent] and rows[0][0][:p_sent].count(e.image_token_id) == 1
    assert rows[0][0][p_sent:].count(e.image_token_id) == 1 and e.image_token_id not in rows[1][0][p_sent:]
    ids = rows[0][0]
    at = ids.index(e.image_token_id)
    assert e.tok.decode(ids[:at]).rstrip().endswith("Photo:")  # the image stands where its reference stands
    probs, info = e.answer(STATE, qs, images=imgs, tag=TAG)
    assert info["image_tokens"] == [counts[0], counts[1]]  # the state's once, question A's own once
    assert info["engine_prompt_tokens"] == info["expected_prompt_tokens"]
    assert info["engine_prompt_tokens"][0] - info["engine_prompt_tokens"][1] > counts[1] - 5
    assert info["shared_prefix_tokens"] == p_exp and info["state_image_end"] > 0
    assert all(abs(float(np.sum(p)) - 1) < 1e-6 for p in probs)


def test_m1_the_cache_boundary_is_after_the_state_images_end_token(engine):
    e = engine
    for state in (marks.mark(1, TAG), "Photo: " + marks.mark(1, TAG) + " A short note.", STATE):
        rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(state, [picture()], QS, TAG)
        assert 0 < end <= p_exp
        expanded = e._expand(rows[0][0][:p_sent], [counts[0] - 2])
        assert expanded[end - 1] == e.eoi and len(expanded) == p_exp  # the prefix ends past the end-of-image token


def test_m1_a_text_request_reads_as_text(engine):
    probs, info = engine.answer("A plain state.", QS)
    assert len(probs) == 2 and "image_tokens" not in info
    again, _ = engine.answer("A plain state.", QS)
    assert all(np.array_equal(a, b) for a, b in zip(probs, again))


def test_m2_the_token_types_reach_the_model(engine):
    e = engine
    img = picture()
    rows, p_sent, row_found, counts = one_row(e, STATE, QS[1], [img])
    ids, lab = rows[0]
    exp = e._expand(ids, [counts[0] - 2])
    got, _ = e.answer(STATE, [QS[1]], images=[img], tag=TAG)
    bidirectional = reference_letters(e, exp, [img], lab, native=True)
    causal = reference_letters(e, exp, [img], lab, native=False)
    # the engine is the library's own overlay (given the token types), and the overlay matters
    assert np.allclose(got[0], bidirectional, atol=5e-4)  # the engine reads the prefix and the question in two calls
    assert np.abs(bidirectional - causal).max() > 5e-3
    assert np.abs(got[0] - causal).max() > 5e-3


def test_m3_a_question_image_after_the_cached_prefix_equals_a_cold_read(engine):
    e = engine
    imgs = [picture(seed=0), picture(w=140, h=100, seed=1)]
    q = {"kind": "noul", "instructions": f"Does the photo match {marks.mark(2, TAG)}?"}
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(STATE, imgs, [q], TAG)
    ids, lab = rows[0]
    exp = e._expand(ids, [c - 2 for c in counts])
    assert len(exp) > p_exp  # the question's image follows the prefix
    e.prefix_cache.entries.clear()
    e.prefix_cache.bytes = 0
    cached, info = e.answer(STATE, [q], images=imgs, tag=TAG)
    again, info2 = e.answer(STATE, [q], images=imgs, tag=TAG)
    cold = reference_letters(e, exp, imgs, lab, native=True)
    assert not info["prefix_cache_hit"] and info2["prefix_cache_hit"]
    assert np.allclose(cached[0], cold, atol=5e-4) and np.allclose(again[0], cold, atol=5e-4)
    assert np.array_equal(cached[0], again[0])


def test_m4_the_prefix_cache_keeps_a_state_with_its_images(engine):
    e = engine
    img, other = picture(seed=0), picture(seed=5)  # the same size, other pixels: the same token ids
    e.prefix_cache.entries.clear()
    e.prefix_cache.bytes = 0
    # the header layout lists the questions before the state, so the same state and questions share a prefix
    first, i1 = e.answer(STATE, QS, images=[img], tag=TAG)
    second, i2 = e.answer(STATE, QS, images=[img], tag=TAG)
    assert not i1["prefix_cache_hit"] and i2["prefix_cache_hit"]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))  # a cached read is the cold read, bit for bit
    rows, _, _, _, _, counts, _ = e._prepare_images(STATE, [img], QS, TAG)
    exp = e._expand(rows[1][0], [counts[0] - 2])
    assert np.allclose(second[1], reference_letters(e, exp, [img], rows[1][1]), atol=5e-4)
    third, i3 = e.answer(STATE, QS, images=[other], tag=TAG)
    assert not i3["prefix_cache_hit"] and not np.allclose(third[0], second[0], atol=1e-6)
    assert e.prefix_cache.facts()["entries"] == 2
    _, i4 = e.answer(STATE, QS, images=[img], tag=TAG)
    assert i4["prefix_cache_hit"] and e.prefix_cache.facts()["entries"] == 2  # the first image's entry was still there


def test_m5_chunked_prefill_never_cuts_an_image_block(tmp_path_factory, mlx_stack):
    small = make(tmp_path_factory, mlx_stack, prefill_step=40)
    t = small.image_token_id
    ids = [5] * 30 + [small.boi] + [t] * 64 + [small.eoi] + [7] * 50
    chunks = small._chunks(ids)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(ids) and all(a < b for a, b in chunks)
    assert all(not (ids[b - 1] == t and ids[b] == t) for _, b in chunks[:-1])  # no cut inside the 64 image tokens
    assert any(b - a > 40 for a, b in chunks)  # the block was kept whole, past the step
    big = make(tmp_path_factory, mlx_stack, prefill_step=2048)
    img = picture()
    chunked, _ = small.answer(STATE, QS, images=[img], tag=TAG)
    whole, _ = big.answer(STATE, QS, images=[img], tag=TAG)
    assert all(np.allclose(a, b, atol=1e-4) for a, b in zip(chunked, whole))


def test_m6_the_engine_refuses_adapters_box_and_too_many_images(engine):
    e = engine
    with pytest.raises(ValueError, match="serves no adapters"):
        e.answer(STATE, QS, adapter="a", images=[picture()], tag=TAG)
    with pytest.raises(ValueError, match="Box is not served with --backend mlx"):
        e.refuse_box()
    assert e.box_supported is False
    many = [picture(seed=i) for i in range(e.max_images + 1)]
    state = "".join(marks.mark(i + 1, TAG) for i in range(len(many)))
    with pytest.raises(ValueError, match=f"at most {e.max_images} images"):
        e.answer(state, QS, images=many, tag=TAG)
    with pytest.raises(ValueError, match="image budget|--image-budget"):
        make_budget = type(e)
        make_budget("x", image_budget=100)


def test_m6_the_engine_refuses_a_conversion_without_the_vision_weights(tmp_path, mlx_stack):
    from decisio.serve.mlx_images import ImagesRefused, MLXGemmaImageLettersEngine

    d = vision_dir(tmp_path, vision=False)
    with pytest.raises(ImagesRefused, match="no vision weights"):
        MLXGemmaImageLettersEngine(str(d), family=GEMMA4)


def test_m7_generate_returns_the_top_k_log_probabilities_of_a_greedy_decode(engine):
    e = engine
    rows, p_sent, row_found, counts = one_row(e, STATE, QS[1], [picture()])
    ids, lab = rows[0]
    steps = e.generate(ids, [picture()], max_new_tokens=4, top_k=5, prefix=p_sent)
    assert len(steps) == 4
    for s in steps:
        lps = [lp for _, lp in s["top"]]
        assert len(lps) == 5 and lps == sorted(lps, reverse=True) and all(lp <= 0 for lp in lps)
        assert s["top"][0][0] == s["token"]
    # the first step is the model's distribution at the answer position: the labels' log-probabilities agree with it
    exp = e._expand(ids, [counts[0] - 2])
    letters = reference_letters(e, exp, [picture()], lab)
    top = dict(steps[0]["top"])
    full = e.generate(ids, [picture()], max_new_tokens=1, top_k=len(lab) + 200)
    allp = dict(full[0]["top"])
    got = np.array([allp.get(t, -1e9) for t in lab])
    if (got > -1e8).all():
        assert np.allclose(np.exp(got) / np.exp(got).sum(), letters, atol=1e-4)
    assert top  # the cached-prefix call and the cold call agree on the first token
    assert steps[0]["token"] == full[0]["token"]


def test_copy_cache_of_the_image_models_entries_leaves_the_original_alone(engine):
    from decisio.serve.mlx_engine import copy_cache

    e = engine
    cache = e.model.make_cache()
    e._feed_images(e._expand([5, 6, e.image_token_id, 7], [e._count(picture())]), cache, [picture()])
    before = [(np.array(c.keys).copy(), c.offset) for c in cache]
    twin = copy_cache(cache)
    e._feed(list(range(10, 20)), twin)
    after = [(np.array(c.keys), c.offset) for c in cache]
    assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(before, after))
    assert all(type(a) is type(b) for a, b in zip(cache, twin))
