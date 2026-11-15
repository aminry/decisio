# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images on the Gemma 4 bases (`--images on`, decisio.serve.gemma_images) on the CPU stand-in.

The engine is the CPU stand-in (decisio.serve.hf_letters.HFGemmaImageLettersEngine): transformers' Gemma 4 classes with
a tiny random model built here (no checkpoint is loaded) under the real Gemma 4 tokenizer, so the checks are about the
prompt, the scoping, the token counts and the wire format, not about any number:

  G1  the image token count equals the Gemma 4 processor's at every budget, on both image processors
  G2  an image stands where its reference stands, and a request with images differs from the same request without them
      by the image tokens only (system turn, header, tail, answer slot and labels as the text route's)
  G3  scope: an image of the state is in the shared prefix of every row; an image of a question's text is in that
      question's row alone, and no other question's prompt or token count carries it
  G4  the cached prefix ends after the state's last image's end token, never inside an image block
  G5  usage.input_tokens counts the text and every image (state images once, question images once per question that
      carries them), against the processor, at each budget
  G6  a text-only request answers bit for bit the same with images on as without
  G7  an adapter serves images: the name reaches the engine, the answer differs from the base's
  G8  the token types reach the forward pass, so the image blocks are bidirectional (and it matters)
  G9  refusals: more images than --max-images, a budget that is not allowed, --images on where it cannot be served;
      text that looks like an image mark is plain text, and every request has its own tag
  G10 the wire: bind_assets keeps where each image is; images of every encoding reach the engine
  G11 the temperature of a request with images: the base's text temperature, or the base's `image_temperature`
  G12 /health reports images on or off, the budget and max_images
  G13 (needs DECISIO_GEMMA_E2B_CORNERS=1 and the local Gemma 4 E2B checkpoint) a square in each corner is placed:
      top and bottom, left and right are answered right

  uv run pytest -q tests/unit/test_gemma_images.py      (the Gemma 4 tokenizer downloads once)
"""

import base64
import io
import json
import os
import types

import numpy as np
import pytest
from PIL import Image

from decisio.families import GEMMA4, GEMMA4_31B, GEMMA4_FP8, QWEN
from decisio.readout.letters import PromptFormat
from decisio.serve import image_marks as marks
from decisio.serve import vllm_engine as sv
from decisio.serve.systemone import bind_assets, extract_state_images

FORMAT = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")
TAG = "0123456789abcdef"  # a request's marks carry its own tag
BUDGETS = (70, 140, 280, 560, 1120)
SIZES = [(512, 512), (768, 1024), (1568, 1568), (280, 280), (1080, 1920), (3, 900), (900, 3), (33, 17), (4000, 5000)]


def png(w, h, seed=0):
    rng = np.random.default_rng(seed)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB").save(b, "PNG")
    return b.getvalue()


def asset(w=96, h=96, seed=0):
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(png(w, h, seed)).decode()}


def picture(w=96, h=96, seed=0):
    return Image.open(io.BytesIO(png(w, h, seed))).convert("RGB")


def make_engine(kind="unified", budget=280, max_images=4, **kw):
    from decisio.serve.hf_letters import HFGemmaImageLettersEngine

    engine = HFGemmaImageLettersEngine(
        GEMMA4.model,
        kind=kind,
        budget=budget,
        max_images=max_images,
        config={"vision": {"position_embedding_size": 128}} if kind == "tower" else {},
        revision=GEMMA4.revision,
        **kw,
    )
    engine.fmt, engine.family = FORMAT, GEMMA4
    return engine


@pytest.fixture(scope="module")
def engine():
    return make_engine()


@pytest.fixture(scope="module", params=["unified", "tower"])
def both(request):
    return make_engine(request.param)


def plain_engine(engine):
    """The text stand-in over the same model and tokenizer: the text route, no image code."""
    from decisio.serve.combine import FairLock
    from decisio.serve.hf_letters import HFLettersEngine

    p = object.__new__(HFLettersEngine)
    p.mode, p.pad_token, p.pad_where, p.adapters = "separate", engine.pad_token, engine.pad_where, {}
    p.tok, p.model, p.model_name, p.revision = engine.tok, engine.model, engine.model_name, None
    p.block_size = p.match_unit = engine.block_size
    p.pad_unit = None
    p._lock, p.scored_rows = FairLock(), []
    p.fmt, p.family = FORMAT, GEMMA4
    return p


# no text merges across the mark, so the same state without the image tokenizes to the same tokens around it
STATE = "Photo:" + marks.mark(1, TAG) + "Title: Blue ceramic mug, 350 ml\nPrice: 12 EUR"
# a tail after the image long enough that the prefix needs no padding to end past it (G4)
LONG = STATE + "\nDescription: " + "Hand-made, dishwasher safe, with a matching saucer. " * 8
QS = [
    {"kind": "noul", "instructions": "Does the photo show the listed item?"},
    {"kind": "choice", "instructions": "Which colour is the mug?", "options": ["blue", "red", "white"]},
]


def without_placeholders(ids, engine):
    return [t for t in ids if t != engine.image_token_id]


# ---- G1 --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
def test_g1_token_count_matches_the_processor(budget):
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor

    for P in (Gemma4UnifiedImageProcessor, Gemma4ImageProcessor):
        ip = P()
        for w, h in SIZES:
            got = ip(images=[Image.new("RGB", (w, h))], max_soft_tokens=budget, return_tensors="pt")
            assert marks.soft_tokens(h, w, budget) == int(got["num_soft_tokens_per_image"][0]), (P.__name__, w, h)
            assert marks.image_tokens(h, w, budget) == int(got["num_soft_tokens_per_image"][0]) + 2


def test_g1_the_documented_figures():
    # the table in docs/api.md: 280 is the default; a 512 by 512 picture is 256 soft tokens
    assert marks.soft_tokens(512, 512, 280) == 256 and marks.soft_tokens(768, 1024, 280) == 266
    assert marks.soft_tokens(512, 512, 70) == 64 and marks.soft_tokens(512, 512, 1120) == 1089
    with pytest.raises(ValueError, match="image budget"):
        marks.soft_tokens(10, 10, 100)


# ---- G2 --------------------------------------------------------------------------------------------------------------


def test_g2_an_image_stands_where_its_reference_stands(both):
    e = both
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(STATE, [picture()], QS, TAG)
    ids = rows[0][0]
    at = ids.index(e.image_token_id)
    before = e.tok.decode(ids[:at])
    after = e.tok.decode(ids[at + 1 : at + 8])
    assert before.rstrip().endswith("Photo:") and after.lstrip().startswith("Title")  # inside the state, not first
    assert state_found == [1] and row_found == [[1], [1]] and ids.count(e.image_token_id) == 1
    assert p_exp == p_sent + counts[0] - 1


def test_g2_with_and_without_images_differ_by_the_image_tokens_only(both):
    e = both
    text = plain_engine(e)
    rows = e._prepare_images(LONG, [picture()], QS, TAG)[0]
    bare = text._prepare_separate(marks.without_marks(LONG, TAG), QS)[0]
    assert e.fmt.layout == "header" and e.fmt.system_prompt and e.fmt.slot == "template"
    for (ids, lab), (tids, tlab) in zip(rows, bare):
        assert without_placeholders(ids, e) == tids and lab == tlab
    # the header lists the questions as the text route's does, and the system turn is there
    assert "Questions that will be asked" in e.tok.decode(rows[0][0])


def test_g2_the_header_names_a_question_image_and_holds_no_pixels():
    e = make_engine()
    qs = [
        {"kind": "noul", "instructions": f"Is this the same as {marks.mark(2, TAG)}?"},
        {"kind": "noul", "instructions": "Is the mug blue?"},
    ]
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(
        STATE, [picture(), picture(seed=1)], qs, TAG
    )
    header = e.tok.decode(rows[0][0][:p_sent])
    assert "[image 2]" in header and "- Is the mug blue?" in header
    assert rows[0][0][:p_sent].count(e.image_token_id) == 1  # the state's image alone is in the prefix


# ---- G3 --------------------------------------------------------------------------------------------------------------


def test_g3_a_question_image_is_in_its_own_row_only(both):
    e = both
    imgs = [picture(seed=0), picture(w=200, h=120, seed=1)]
    qs = [
        {"kind": "noul", "instructions": f"Does the photo match {marks.mark(2, TAG)}?"},
        {"kind": "noul", "instructions": "Is the mug blue?"},
    ]
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(STATE, imgs, qs, TAG)
    assert state_found == [1] and row_found == [[1, 2], [1]]
    assert rows[0][0][:p_sent] == rows[1][0][:p_sent]  # the prefix is shared
    assert rows[0][0].count(e.image_token_id) == 2 and rows[1][0].count(e.image_token_id) == 1
    assert rows[0][0][p_sent:].count(e.image_token_id) == 1  # the question's own part holds its image
    assert e.image_token_id not in rows[1][0][p_sent:]  # question B's own part has no image
    probs, info = e.answer(STATE, qs, images=imgs, tag=TAG)
    # state image once, question A's image once; question B's tokens do not count it
    assert info["image_tokens"] == [counts[0], counts[1]]
    lens = info["engine_prompt_tokens"]
    assert lens[0] - lens[1] > counts[1] - 5  # A's prompt carries the second image, B's does not
    assert lens == info["expected_prompt_tokens"]
    # A's second image reaches the model only in A's forward pass
    imgs_in_forward = [f["images"] for f in e.forwards[-2:]]
    assert imgs_in_forward == [2, 1]


def test_g3_images_in_criteria_are_in_that_questions_row(both):
    e = both
    q = {"kind": "choice", "instructions": "Which is it?", "options": [f"like {marks.mark(2, TAG)}", "something else"]}
    rows, p_sent, _, _, row_found, _, _ = e._prepare_images(STATE, [picture(), picture(seed=3)], [q, QS[0]], TAG)
    assert row_found == [[1, 2], [1]]
    listing = e.tok.decode(rows[0][0][p_sent:])
    assert "A. like" in listing and "B. something else" in listing


def test_g3_an_image_in_the_state_is_shared_not_repeated(both):
    e = both
    probs, info = e.answer(STATE, QS, images=[picture()], tag=TAG)
    c = e.image_tokens(picture())
    assert info["image_tokens"] == [c]  # once for the request, however many questions read it
    assert info["engine_prompt_tokens"] == info["expected_prompt_tokens"]


# ---- G4 --------------------------------------------------------------------------------------------------------------


def test_g4_the_prefix_ends_after_the_last_state_images_end_token():
    e = make_engine()
    unit = e.match_unit
    img = picture()
    for state in (marks.mark(1, TAG), "Photo: " + marks.mark(1, TAG), marks.mark(1, TAG) + " " + "A short note. " * 2):
        rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(state, [img], QS, TAG)
        assert end > 0 and p_exp >= end
        assert (p_exp // unit) * unit >= end, state  # the last whole block of the prefix is after the image
    # a long tail after the image needs no padding: the prefix is the state's own tokens
    long_state = "Photo:" + marks.mark(1, TAG) + "Details about the item. " * 40
    rows, p_sent, p_exp, *_ = e._prepare_images(long_state, [img], QS, TAG)
    shared = plain_engine(e)._prepare_separate(marks.without_marks(long_state, TAG), QS)[1]
    assert p_sent == shared + 1  # the text route's prefix and the image's placeholder, no pad tokens


def test_g4_a_prefix_that_is_too_short_is_padded_to_the_block_boundary():
    e = make_engine()
    rows, p_sent, p_exp, state_found, row_found, counts, end = e._prepare_images(
        marks.mark(1, TAG), [picture()], QS, TAG
    )
    assert p_exp % e.match_unit == 0 and p_exp >= end
    text = plain_engine(e)._prepare_separate("", QS)
    assert p_sent > text[1]  # newline tokens were added after the state's blank line


def test_g4_no_prefix_image_means_no_pad_for_the_boundary():
    e = make_engine()
    q = {"kind": "noul", "instructions": f"Is this {marks.mark(1, TAG)} a mug?"}
    rows, p_sent, p_exp, state_found, *_ = e._prepare_images("A short state.", [picture()], [q], TAG)
    assert state_found == [] and p_exp == p_sent


# ---- G5 --------------------------------------------------------------------------------------------------------------


def make_client(engine, **kw):
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne

    so = SystemOne(engine, "decisio-test", **kw)
    return TestClient(sv.make_app(engine, so), headers={"x-decisio-client": "test"}), so


def request_body(**kw):
    return {
        "state": {"listing": "ITEM", "title": "Blue ceramic mug, 350 ml"},
        "questions": {
            "matches": {"type": "noul", "instructions": "Does the photo QPIC show the listed item?"},
            "colour": {
                "type": "choice",
                "instructions": "Which colour is the mug?",
                "criteria": {"blue": None, "red": None, "white": None},
            },
        },
        "assets": {"ITEM": asset(200, 150, 1), "QPIC": asset(120, 120, 2)},
        **kw,
    }


def bare_of(body):
    """The same request with its images' places written as plain words: the text of the request alone."""
    text = json.dumps(body).replace("ITEM", "item").replace("QPIC", "")
    out = json.loads(text)
    del out["assets"]
    return out


def usage(client, body):
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    return r.json()["usage"]["input_tokens"]


@pytest.mark.parametrize("budget", [70, 280, 1120])
def test_g5_usage_counts_text_and_every_image_against_the_processor(budget):
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor

    e = make_engine(budget=budget)
    client, so = make_client(e)
    body = request_body()
    ip = Gemma4UnifiedImageProcessor()

    def soft(w, h):
        out = ip(images=[Image.new("RGB", (w, h))], max_soft_tokens=budget, return_tensors="pt")
        return int(out["num_soft_tokens_per_image"][0]) + 2  # the begin and end tokens count

    assert usage(client, body) == usage(client, bare_of(body)) + soft(200, 150) + soft(120, 120)


def test_g5_a_question_image_is_counted_for_its_question_only():
    e = make_engine()
    client, so = make_client(e)
    both = request_body()
    colour_only = request_body()
    del colour_only["questions"]["matches"]
    colour_only["assets"].pop("QPIC")
    # asking the second question too adds its text and the first question's own image once, nothing else
    extra = (usage(client, both) - usage(client, colour_only)) - (
        usage(client, bare_of(both)) - usage(client, bare_of(colour_only))
    )
    assert extra == e.image_tokens(picture(120, 120))
    # and the state's image is in both counts once, whatever the number of questions
    one_q = request_body()
    del one_q["questions"]["matches"]
    one_q["assets"].pop("QPIC")
    assert usage(client, one_q) - usage(client, bare_of(one_q)) == e.image_tokens(picture(200, 150))


# ---- G6 --------------------------------------------------------------------------------------------------------------


def test_g6_a_text_request_answers_the_same_with_images_on(engine):
    text = plain_engine(engine)
    state = {"ticket": "Order 5521 arrived with a cracked screen. The customer wants it fixed before Friday."}
    qs = [
        {"kind": "choice", "instructions": "Which team handles this?", "options": ["billing", "repairs", "sales"]},
        {"kind": "noul", "instructions": "Is the device damaged?"},
    ]
    on, _ = engine.answer(state, qs)
    off, _ = text.answer(state, qs)
    for a, b in zip(on, off):
        assert np.array_equal(a, b)
    # and through the server, the paired check: the same body with and without a route header
    c1, _ = make_client(engine)
    c2, _ = make_client(text)
    body = {"state": state, "questions": {"t": {"type": "noul", "instructions": "Is the device damaged?"}}}
    assert (
        c1.post("/v1/systemone", json=body).json()["answers"] == c2.post("/v1/systemone", json=body).json()["answers"]
    )


# ---- G7 --------------------------------------------------------------------------------------------------------------


def test_g7_an_adapter_serves_images():
    e = make_engine()
    e.add_adapter("tuned", scale=0.5, seed=7)
    client, so = make_client(e, adapter_temperatures={"tuned": {"choice": 1.0, "noul": 1.0}})
    body = request_body()
    base = client.post("/v1/systemone", json=body)
    tuned = client.post("/v1/systemone", json={**body, "adapter": "tuned"})
    assert base.status_code == 200 and tuned.status_code == 200, tuned.text
    assert tuned.json()["adapter"] == "tuned" and "adapter" not in base.json()
    assert tuned.json()["answers"]["colour"]["probabilities"] != base.json()["answers"]["colour"]["probabilities"]
    assert e.forwards[-1]["images"] >= 1  # the adapter's read carried the pictures
    assert client.post("/v1/systemone", json={**body, "adapter": "nope"}).status_code == 404
    assert e.adapter_calls and e.adapter_calls[-1] == "tuned"


# ---- G8 --------------------------------------------------------------------------------------------------------------


def test_g8_the_token_types_reach_the_forward_pass(both):
    import torch

    e = both
    probs, info = e.answer(STATE, QS, images=[picture(200, 150)], tag=TAG)
    f = e.forwards[-1]
    ids, types = f["input_ids"], f["mm_token_type_ids"]
    c = marks.soft_tokens(150, 200, e.image_budget)
    assert len(types) == len(ids) and sum(types) == c
    assert all(t == (1 if i == e.image_token_id else 0) for i, t in zip(ids, types))  # begin and end tokens are text
    start = ids.index(e.boi)
    assert ids[start + 1 : start + 1 + c] == [e.image_token_id] * c and ids[start + 1 + c] == e.eoi
    assert e.model.config.get_text_config().use_bidirectional_attention == "vision"
    # the flag matters: without the types the image block is causal and the readout moves
    ids_t = torch.tensor([ids])
    pics = e._image_processor()(images=[picture(200, 150)], max_soft_tokens=e.image_budget, return_tensors="pt")
    kw = {"pixel_values": pics["pixel_values"], "image_position_ids": pics["image_position_ids"]}
    with torch.no_grad():
        mask = torch.ones_like(ids_t)
        a = e.model(
            input_ids=ids_t, attention_mask=mask, mm_token_type_ids=torch.tensor([types]), logits_to_keep=1, **kw
        ).logits[0, -1]
        z = e.model(
            input_ids=ids_t, attention_mask=mask, mm_token_type_ids=torch.zeros_like(ids_t), logits_to_keep=1, **kw
        ).logits[0, -1]
    assert (a - z).abs().max() > 1e-4


def test_g8_every_image_is_its_own_block(engine):
    state = marks.mark(1, TAG) + "\nand\n" + marks.mark(2, TAG) + "\nTitle"
    engine.answer(state, [QS[0]], images=[picture(), picture(seed=4)], tag=TAG)
    ids, types = engine.forwards[-1]["input_ids"], engine.forwards[-1]["mm_token_type_ids"]
    runs = 0
    prev = 0
    for t in types:
        runs += t == 1 and prev == 0
        prev = t
    assert runs == 2 and ids.count(engine.boi) == 2 and ids.count(engine.eoi) == 2


# ---- G9 --------------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_hub_config(monkeypatch):
    import decisio.families as fam

    monkeypatch.setattr(fam, "read_config", lambda model, revision=None: {"text_config": {"sliding_window": 1024}})


def parsed(*argv):
    ap = sv.build_parser()
    args = ap.parse_args(["--model", "x", *argv])
    args.family = GEMMA4
    return args


def test_g9_the_flags_resolve():
    assert sv.resolve_images(parsed()) is None
    assert sv.resolve_images(parsed("--images", "on")) == 560
    assert sv.resolve_images(parsed("--images", "on", "--image-budget", "560")) == 560
    assert sv.resolve_images(parsed("--images", "on", "--image-budget", "280")) == 280
    for flags, why in [
        (["--image-budget", "280"], "needs --images on"),
        (["--images", "on", "--image-budget", "300"], "one of 70, 140, 280, 560, 1120"),
        (["--images", "on", "--image-model", "q"], "--image-model"),
        (["--images", "on", "--one-engine"], "--one-engine"),
        (["--images", "on", "--backend", "hf"], "vLLM server and --backend mlx"),
    ]:
        with pytest.raises(ValueError, match=why):
            sv.resolve_images(parsed(*flags))
    q = parsed("--images", "on")
    q.family = QWEN
    with pytest.raises(ValueError, match="Gemma bases"):
        sv.resolve_images(q)
    big = parsed("--images", "on", "--model-class", "hidden-readout")
    big.family = GEMMA4_31B
    with pytest.raises(ValueError, match="no image form"):
        sv.resolve_images(big)
    ok = parsed("--images", "on", "--model-class", "hidden-readout")
    assert sv.resolve_images(ok) == 560  # the 12B's class is multimodal already
    # the 31B with no --model-class: its default (head-capable) class is text-only, so images on takes the
    # checkpoint's multimodal class instead, and the head mode that follows is "no head", not the hidden readout
    plain31 = parsed("--images", "on")
    plain31.family = GEMMA4_31B
    assert sv.resolve_images(plain31) == 280 and plain31.model_class == "text-only"
    assert sv.resolve_head_mode(plain31.model_class, None) == ("text-only", None)
    # the 12B keeps its default: its head-capable class reads images
    plain12 = parsed("--images", "on")
    assert sv.resolve_images(plain12) == 560 and plain12.model_class is None
    assert sv.resolve_head_mode(plain12.model_class, None) == ("hidden-readout", "single-engine")


def test_g9_both_gemma_bases_take_images_on_a_mac():
    assert sv.resolve_images(parsed("--images", "on", "--backend", "mlx")) == 280
    big = parsed("--images", "on", "--backend", "mlx")
    big.family = GEMMA4_31B
    assert sv.resolve_images(big) == 280  # the 31B reads through its vision tower, with the base's vision add-on


def test_g9_the_engine_refuses_a_budget_and_too_many_images():
    with pytest.raises(ValueError, match="--image-budget"):
        make_engine(budget=100)
    e = make_engine(max_images=2)
    with pytest.raises(ValueError, match="at most 2 images"):
        e.answer(STATE, QS, images=[picture()] * 3, tag=TAG)
    client, _ = make_client(e)
    body = request_body()
    body["state"] = "ITEM ITEM ITEM"
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 400 and "at most 2 images" in r.text  # System One's error shape for placeholders
    r = client.post(
        "/v1/systemone", json={"state": "x", "questions": body["questions"], "images": ["data:image/png;base64,!!"]}
    )
    assert r.status_code == 422


def test_g9_text_that_looks_like_a_mark_is_plain_text():
    look_alikes = "see \u27e6image:1\u27e7 and \u27e6image:abc:1\u27e7 and \u27e6image:" + TAG + "x:1\u27e7"
    body = request_body()
    body["state"] = {"listing": "ITEM", "note": look_alikes}
    e = make_engine()
    client, _ = make_client(e)
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    first = e.forwards[-2]
    assert first["images"] == 2  # the real images only: the state's and question A's
    assert e.tok.decode(first["input_ids"]).count("\u27e6image:") >= 3  # the look-alikes reached the model as text
    # the same text with no image at all is a text request
    r = client.post(
        "/v1/systemone", json={"state": look_alikes, "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    )
    assert r.status_code == 200
    # and through the binders: they pass it through and never refuse it
    out, blobs, _ = bind_assets(json.loads(json.dumps(body)), placed=True, tag=TAG)
    assert len(blobs) == 2 and look_alikes in out["state"]["note"]
    state, found = extract_state_images([look_alikes, "data:image/png;base64,AAAA"], True, tag=TAG)
    assert state[0] == look_alikes and len(found) == 1
    # the model's own placeholder in a text is refused where it would be mistaken for an image
    with pytest.raises(ValueError, match="stands for an image"):
        e.answer(marks.mark(1, TAG) + " <|image|>", QS, images=[picture()], tag=TAG)


def test_g9_each_request_has_its_own_tag():
    tags = {marks.new_tag() for _ in range(50)}
    assert len(tags) == 50 and all(len(t) == 16 and int(t, 16) >= 0 for t in tags)
    a = bind_assets(json.loads(json.dumps(request_body())), placed=True)[0]
    b = bind_assets(json.loads(json.dumps(request_body())), placed=True)[0]
    assert a["state"] != b["state"]
    e = make_engine()
    client, _ = make_client(e)
    client.post("/v1/systemone", json=request_body())
    one = e.tok.decode(e.forwards[-1]["input_ids"])
    client.post("/v1/systemone", json=request_body())
    assert e.tok.decode(e.forwards[-1]["input_ids"]) == one  # the tag never reaches the model


# ---- G10 -------------------------------------------------------------------------------------------------------------


def test_g10_bind_assets_keeps_where_each_image_is():
    body = {
        "state": {"frames": ["F1 ", "x F2"], "note": "none"},
        "questions": {
            "a": {"type": "noul", "instructions": "Is QA here?", "criteria": {"true": "like QA2"}},
            "b": {"type": "choice", "instructions": "Which?", "criteria": {"one": "QB", "two": {"description": "QB2"}}},
        },
        "assets": {n: asset(seed=i) for i, n in enumerate(["F1", "F2", "QA", "QA2", "QB", "QB2"])},
    }
    scopes = []
    out, blobs, _ = bind_assets(json.loads(json.dumps(body)), placed=True, scopes=scopes, tag=TAG)
    assert scopes == [None, None, "a", "a", "b", "b"] and len(blobs) == 6
    assert marks.marks_in(json.dumps(out["state"], ensure_ascii=False), TAG) == [1, 2]
    assert marks.marks_in(out["questions"]["a"]["instructions"], TAG) == [3]
    assert marks.marks_in(json.dumps(out["questions"]["b"]["criteria"], ensure_ascii=False), TAG) == [5, 6]
    # the second engine's route keeps its own text exactly
    old, blobs2, _ = bind_assets(json.loads(json.dumps(body)))
    assert old["state"]["frames"] == ["[image 1] ", "x [image 2]"] and len(blobs2) == 6


def test_g10_every_encoding_reaches_the_engine_with_its_images_in_place():
    e = make_engine()
    client, _ = make_client(e)
    qs = {"q": {"type": "noul", "instructions": "Is it a mug?"}}
    pic = picture()
    url = "data:image/png;base64," + base64.b64encode(png(96, 96)).decode()
    r1 = client.post("/v1/systemone", json={"state": {"photo": url, "title": "mug"}, "questions": qs})
    assert r1.status_code == 200, r1.text
    f1 = e.forwards[-1]
    assert f1["images"] == 1 and e.image_token_id in f1["input_ids"]
    # the list encoding puts its images first in the state; multipart parts do the same
    r2 = client.post("/v1/systemone", json={"state": "mug", "questions": qs, "images": [url]})
    assert r2.status_code == 200, r2.text
    r3 = client.post(
        "/v1/systemone",
        data={"request": json.dumps({"state": "mug", "questions": qs})},
        files={"image": ("p.png", png(96, 96), "image/png")},
    )
    assert r3.status_code == 200, r3.text
    assert r2.json()["answers"] == r3.json()["answers"]
    assert pic.size == (96, 96)


def test_g10_listed_images_open_the_state_and_a_literal_marker_stays_text():
    """On a Gemma base the `images` list is placed as it always was: every listed image opens the state, and a literal
    "[image 1]" in the text is plain text, not a place (only the Qwen `--image-model` route reads it as one). The
    request's token ids are the state with the image's mark first, then a newline, then the text as written."""
    from decisio.readout.letters import fmt_state

    e = make_engine()
    client, so = make_client(e)
    url = "data:image/png;base64," + base64.b64encode(png(96, 96)).decode()
    qs = {"q": {"type": "noul", "instructions": "Is it a mug?"}}
    e.scored_rows.clear()
    r = client.post("/v1/systemone", json={"state": "Photo: [image 1]", "questions": qs, "images": [url]})
    assert r.status_code == 200, r.text
    sent = e.scored_rows[-1][0]
    question = [{"kind": "noul", "instructions": "Is it a mug?"}]
    state = f"{marks.mark(1, TAG)}\n{fmt_state('Photo: [image 1]')}"
    want = e._prepare_images(state, [picture()], question, TAG)[0][0][0]
    assert sent == want  # the same token ids: the image first, "Photo: [image 1]" as plain words after it
    assert sent.count(e.image_token_id) == 1
    assert "Photo" not in e.tok.decode(sent[: sent.index(e.image_token_id)])  # the image comes before the text
    assert "[image 1]" in e.tok.decode(sent)


def test_g10_an_asset_in_a_question_reaches_only_that_question():
    e = make_engine()
    client, _ = make_client(e)
    r = client.post("/v1/systemone", json=request_body())
    assert r.status_code == 200, r.text
    first, second = e.forwards[-2:]
    assert (first["images"], second["images"]) == (2, 1)


# ---- G11 -------------------------------------------------------------------------------------------------------------


def test_g11_the_image_temperature_defaults_to_the_text_temperature():
    from decisio.serve.temperature import apply_temperature

    e = make_engine()
    raw, _ = e.answer(request_text_state(), [QS[1]], images=[picture(200, 150, 1)], tag=TAG)
    body = request_body()
    del body["questions"]["matches"]
    body["assets"].pop("QPIC")
    for kw, want in (({"temperature": 3.0}, 3.0), ({"temperature": 3.0, "image_temperature_scale": 2.0}, 6.0)):
        client, so = make_client(e, **kw)
        got = client.post("/v1/systemone", json=body).json()["answers"]["colour"]["probabilities"]
        expect = apply_temperature(np.asarray(raw[0], dtype=np.float64), want)
        assert np.allclose(list(got.values()), expect, atol=1e-6), kw
    # a request without images keeps the text temperature
    client, so = make_client(e, temperature=3.0, image_temperature_scale=2.0)
    assert so.image_temperature_of("choice") == 6.0 and so.temperature_of("choice") == 3.0
    assert so.image_temperature_of("choice", "tuned") == so.temperature_of("choice", "tuned")


def test_g11_the_scale_multiplies_each_question_types_text_temperature():
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "x", temperature=3.0, temperatures={"noul": 5.0, "choice": 4.0}, image_temperature_scale=1.5)
    assert so.image_temperature_of("noul") == 7.5
    assert so.image_temperature_of("choice") == 6.0
    assert so.image_temperature_of("score") == 4.5
    off = SystemOne(None, "x", temperature=3.0, temperatures={"noul": 5.0})
    assert off.image_temperature_of("noul") == 5.0 and off.image_temperature_of("score") == 3.0


def test_g11_the_31b_scales_its_text_temperatures_and_the_12b_does_not():
    from decisio.serve.systemone import SystemOne

    def served(fam):
        return SystemOne(
            None,
            "x",
            temperature=fam.temperature,
            temperatures={"choice": fam.choice_temperature, **fam.temperatures}
            if fam.choice_temperature is not None
            else dict(fam.temperatures),
            image_temperature_scale=fam.image_temperature_scale,
        )

    so = served(GEMMA4_31B)
    assert GEMMA4_31B.image_temperature_scale == 1.376
    assert so.image_temperature_of("choice") == pytest.approx(4.672 * 1.376)
    assert so.image_temperature_of("noul") == pytest.approx(5.252 * 1.376)
    for fam in (GEMMA4, GEMMA4_FP8):
        assert fam.image_temperature_scale is None
        s12 = served(fam)
        for q in ("choice", "noul", "score"):
            assert s12.image_temperature_of(q) == s12.temperature_of(q)


def test_g11_an_adapters_own_temperatures_win_over_the_scale():
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "x", temperature=3.0, image_temperature_scale=2.0)
    so.adapter_temperatures = {"tuned": {"choice": 1.25}}
    assert so.image_temperature_of("choice", "tuned") == 1.25
    assert so.image_temperature_of("choice") == 6.0


def request_text_state():
    return "listing: " + marks.mark(1, TAG) + "\ntitle: Blue ceramic mug, 350 ml"


def test_g11_the_family_field_defaults_to_none_and_is_checked():
    from dataclasses import replace

    from decisio.families import check_profile_temperatures

    assert GEMMA4.image_temperature_scale is None and GEMMA4_31B.image_temperature_scale == 1.376
    check_profile_temperatures(replace(GEMMA4, image_temperature_scale=2.5))
    for bad in (0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="image temperature scale"):
            check_profile_temperatures(replace(GEMMA4, image_temperature_scale=bad))


def test_g11_the_scale_is_not_part_of_the_exported_decision_config():
    """Like the image budget, the scale is a serving default of the base in code: a published repository's file still
    verifies against the profile, whatever the scale."""
    from decisio import hub

    for fam in (GEMMA4, GEMMA4_31B):
        t = hub.profile_of(fam)["temperatures"]
        assert set(t) == {"global", "choice"}


# ---- G12 -------------------------------------------------------------------------------------------------------------


def test_g12_health_reports_the_image_settings():
    e = make_engine(budget=560, max_images=7)
    client, _ = make_client(e)
    h = client.get("/health").json()
    assert h["images"] == {"enabled": True, "budget": 560, "max_images": 7, "route": "same engine", "box": True}
    off = plain_engine(e)
    client, _ = make_client(off)
    assert client.get("/health").json()["images"] == {"enabled": False, "box": False}
    assert sv.images_facts(off, types.SimpleNamespace(max_images=9)) == {
        "enabled": True,
        "route": "second engine (--image-model)",
        "max_images": 9,
        "prompt": "placed",
        "box": False,
    }


def test_g12_no_image_engine_says_how_to_start_one():
    client, _ = make_client(plain_engine(make_engine()))
    r = client.post("/v1/systemone", json=request_body())
    assert r.status_code == 400 and "--images on" in r.text


# ---- G13 -------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DECISIO_GEMMA_E2B_CORNERS"),
    reason="needs DECISIO_GEMMA_E2B_CORNERS=1 and the local Gemma 4 E2B",
)
def test_g13_a_square_in_each_corner_is_placed():
    from decisio.serve.hf_letters import HFGemmaImageLettersEngine

    model = os.environ.get("DECISIO_GEMMA_E2B", "google/gemma-4-E2B-it")
    e = HFGemmaImageLettersEngine(model, model=model, budget=280)
    e.fmt = FORMAT
    wrong = []
    for vertical in ("top", "bottom"):
        for horizontal in ("left", "right"):
            img = Image.new("RGB", (480, 480), "white")
            x = 20 if horizontal == "left" else 340
            y = 20 if vertical == "top" else 340
            img.paste(Image.new("RGB", (120, 120), "black"), (x, y))
            qs = [
                {
                    "kind": "choice",
                    "instructions": "Is the black square at the top or at the bottom?",
                    "options": ["top", "bottom"],
                },
                {
                    "kind": "choice",
                    "instructions": "Is the black square at the left or at the right?",
                    "options": ["left", "right"],
                },
            ]
            probs, _ = e.answer(marks.mark(1, TAG), qs, images=[img], tag=TAG)
            if probs[0][["top", "bottom"].index(vertical)] < 0.5:
                wrong.append((vertical, horizontal, "vertical", probs[0].tolist()))
            if probs[1][["left", "right"].index(horizontal)] < 0.5:
                wrong.append((vertical, horizontal, "horizontal", probs[1].tolist()))
    assert not wrong, wrong


@pytest.mark.parametrize("kind", ["unified", "tower"])
def test_warm_images_reads_a_picture_through_the_served_path(kind):
    """The warm-up a server runs at start goes through the same request path as a served picture (a fresh tag, a mark,
    the image at its place): it reads without error on both Gemma layouts."""
    engine = make_engine(kind=kind)
    engine.warm_images(rounds=1)


def test_the_vllm_image_engine_warms_its_pictures_at_start(monkeypatch):
    """The vLLM engine's start calls the shared warm-up (the one the stand-in test runs), after its image setup."""
    from decisio.serve import gemma_images
    from decisio.serve.vllm_engine import LettersEngine

    calls = []
    monkeypatch.setattr(LettersEngine, "__init__", lambda self, model, engine_kw=None, **kw: calls.append("engine"))
    monkeypatch.setattr(gemma_images.GemmaImageRequests, "_image_setup", lambda self, b, m: calls.append("setup"))
    monkeypatch.setattr(gemma_images.GemmaImageRequests, "warm_images", lambda self, rounds=2: calls.append("warm"))
    gemma_images.GemmaImageLettersEngine(GEMMA4.model, family=GEMMA4)
    assert calls == ["engine", "setup", "warm"]


# ---- budget against the sliding window ------------------------------------------------------------------------------


def _window(monkeypatch, window, encoder_layers=None):
    """The checkpoint config the server reads: a sliding window, and a vision encoder of `encoder_layers` layers (the
    31B's tower has 27) or none (the 12B's encoder-free model declares no layers)."""
    import decisio.families as fam

    cfg = {"text_config": {"sliding_window": window}} if window else {}
    cfg["vision_config"] = {"num_hidden_layers": encoder_layers} if encoder_layers else {"model_patch_size": 48}
    monkeypatch.setattr(fam, "read_config", lambda model, revision=None: cfg)


def test_budget_longer_than_the_window_is_refused_on_an_encoder_free_model(monkeypatch):
    """The 12B embeds patches straight into the language model, so a block longer than its window loses the top rows
    (measured: at 1,120 it read top-corner squares as bottom)."""
    family = GEMMA4
    _window(monkeypatch, 1024)
    for budget in ("280", "560"):
        a = parsed("--images", "on", "--image-budget", budget)
        a.family = family
        assert sv.resolve_images(a) == int(budget)
    a = parsed("--images", "on", "--image-budget", "1120")
    a.family = family
    with pytest.raises(ValueError) as e:
        sv.resolve_images(a)
    assert str(e.value) == (
        "--image-budget 1120 makes an image of up to 1,122 tokens, longer than this model's 1,024-token sliding "
        "window, so the image's later tokens could not see its top rows; use 560 or lower"
    )


@pytest.mark.parametrize("budget", ["280", "560", "1120"])
def test_a_model_with_a_vision_encoder_takes_every_budget(monkeypatch, budget):
    """The 31B reads pictures through its own 27-layer encoder, so each image token carries the whole picture and the
    language model's window does not limit the block: 1,120 is allowed (measured: the corner probe reads every corner
    at 1,120 on the 31B)."""
    _window(monkeypatch, 1024, encoder_layers=27)
    a = parsed("--images", "on", "--image-budget", budget)
    a.family = GEMMA4_31B
    assert sv.resolve_images(a) == int(budget)


def test_the_default_budget_is_per_base(monkeypatch):
    """The 12B reads images at 560 and the 31B at 280 when --image-budget is not given."""
    _window(monkeypatch, 1024)
    assert GEMMA4.image_budget == 560 and GEMMA4_31B.image_budget is None
    assert sv.resolve_images(parsed("--images", "on")) == 560
    _window(monkeypatch, 1024, encoder_layers=27)
    a = parsed("--images", "on")
    a.family = GEMMA4_31B
    assert sv.resolve_images(a) == 280
    a = parsed("--images", "on", "--image-budget", "1120")
    a.family = GEMMA4_31B
    assert sv.resolve_images(a) == 1120


def test_a_longer_window_allows_1120(monkeypatch):
    _window(monkeypatch, 2048)
    assert sv.resolve_images(parsed("--images", "on", "--image-budget", "1120")) == 1120


def test_block_length_against_the_processor_formula():
    for budget in marks.BUDGETS:
        assert marks.block_tokens(budget) == budget + 2
        biggest = max(marks.image_tokens(h, w, budget) for h, w in [(768, 768), (512, 512), (480, 480), (2000, 2000)])
        assert biggest <= marks.block_tokens(budget)
        assert marks.image_tokens(1536, 1536, budget) == marks.block_tokens(budget) - (
            budget - marks.soft_tokens(1536, 1536, budget)
        )
    assert marks.allowed_budgets(1024) == (70, 140, 280, 560)
    assert marks.allowed_budgets(1122) == marks.BUDGETS
    assert marks.allowed_budgets(None) == marks.BUDGETS
    assert marks.sliding_window_of({"text_config": {"sliding_window": 1024}}) == 1024
    assert marks.sliding_window_of({}) is None
    encoder_free = {"text_config": {"sliding_window": 1024}, "vision_config": {"model_patch_size": 48}}
    with_tower = {"text_config": {"sliding_window": 1024}, "vision_config": {"num_hidden_layers": 27}}
    assert marks.has_vision_encoder(with_tower) and not marks.has_vision_encoder(encoder_free)
    assert marks.image_window_limit(encoder_free) == 1024 and marks.image_window_limit(with_tower) is None
