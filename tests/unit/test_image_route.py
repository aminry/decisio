# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The image route of `/v1/systemone` (decisio.serve.image_engine, decisio.serve.placed_images,
decisio.serve.systemone) on the CPU stand-in.

Both engines are the CPU stand-ins (decisio.serve.hf_letters) on a small checkpoint of the served model's family, so the
checks are about the route, the prompt and the wire format, not about any number:

  I1  the image token count (`smart_resize` arithmetic) equals the model's own processor on odd and large sizes;
  I2  an image stands where its reference stands: a request with images differs from the same request without them
      by the image tokens only (system turn, tail, answer slot and labels as the text route's);
  I3  the three encodings imajev accepts (JSON `images`, multipart `request` + `image`, a data:image URI in the state)
      reach the image engine and give the same probabilities; imajev's two answer fields appear exactly when the
      request uses its extension; text requests stay on the text engine with TypeSafe's format;
  I4  the paired text check: a text request through the text route and through the image engine (`x-decisio-route:
      image`) gives bit-identical probabilities;
  I5  refusals are 422: more images than the engine's `--max-images` (set to 2 here), a bad data URL, a bad multipart
      request field, an unknown route;
  I7  every encoding of one picture gives the same probabilities: System One's `assets` (JSON `data_base64`, a multipart
      `part`), a data:image URI in the state, and the `images` list with a literal "[image 1]" in the text (that marker
      is the image's reference position); an image of the list that no marker names opens the state instead;
  I8  scope: an image of the state is in every row's shared prefix; an image of a question's own text is in that
      question's row alone, and no other question's prompt or token count carries it;
  I9  usage.input_tokens counts the text and every image, state images once and a question's own image once per
      question that carries it;
  I10 the cached prefix ends after the state's last image's end token, never inside the image's placeholder run;
  I11 `--image-prompt legacy` reproduces the pre-migration prompts byte for byte (a fixture of token ids pinned from
      main before this change);
  I12 the legacy layout's deprecation: every image response carries `Deprecation` and `Warning` headers naming the
      flag, and the server logs a warning once and then every `LEGACY_NOTICE_EVERY` requests, not per request;
  I13 a `box` question is refused with a clear 400 on this route (Box needs a Gemma base); Qwen grounding is not
      verified.

Needs transformers with the Qwen3.5 classes, torchvision, pillow, fastapi, httpx and python-multipart (the package's
dependencies and the dev extra); I6 also needs imajev's harness (`imajev_bench`) and is skipped without it:

    uv run pytest -q tests/unit/test_image_route.py
"""

import base64
import io
import json
import os
import pathlib

import numpy as np
import pytest
from PIL import Image  # noqa: E402

from decisio.serve import image_marks as marks

MODEL = os.environ.get("DECISIO_IMAGE_TEST_MODEL", "Qwen/Qwen3.5-0.8B-Base")
TAG = "0123456789abcdef"  # a request's marks carry its own tag
STATE = {"listing": {"title": "Blue ceramic mug, 350 ml", "colour": "blue"}}
QUESTIONS = {
    "matches": {"type": "noul", "instructions": "Does the photo show the listed item?"},
    "colour": {
        "type": "choice",
        "instructions": "Which colour is the mug in the photo?",
        "criteria": {"blue": None, "red": None, "white": None},
    },
}


def png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def data_url(img):
    return "data:image/png;base64," + base64.b64encode(png(img)).decode()


def picture_of(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


@pytest.fixture(scope="module")
def picture():
    return picture_of(300, 200)


@pytest.fixture(scope="module")
def served():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFImageLettersEngine, HFLettersEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    text = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    image = HFImageLettersEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(text, "decisio-test", image_engine=image)
    # multipart is the encoding imajev's image clients send; the server takes it with an X-Decisio-Client header
    return TestClient(make_app(text, so), headers={"x-decisio-client": "test"}), text, image


def probs(body):
    out = {}
    for name, a in body["answers"].items():
        out[name] = [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
    return out


def without_placeholders(ids, image):
    """The row without the image placeholders and the front padding: what the text route's row holds."""
    skip = {image.vision_start_id, image.image_token_id, image.vision_end_id}
    out = [t for t in ids if t not in skip]
    while out and out[0] == image.pad_token:
        out.pop(0)
    return out


def test_i1_token_count_matches_processor(served):
    _, _, image = served
    for w, h in [(1536, 1024), (1264, 848), (33, 17), (4000, 5000), (10, 10), (8192, 300), (1000, 999)]:
        img = Image.new("RGB", (w, h))
        g = image.processor.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0].tolist()
        assert image.pad_tokens(img) == g[0] * g[1] * g[2] // image.merge**2, (w, h)
        assert image.image_tokens(img) == image.pad_tokens(img) + 2  # the two vision delimiter tokens are billed too


def test_i2_an_image_stands_where_its_reference_stands(served, picture):
    _, text, image = served
    qs = [
        {"kind": "noul", "instructions": "Does the photo show the listed item?"},
        {"kind": "choice", "instructions": "Which colour?", "options": ["blue", "red", "white"]},
    ]
    state = "Photo of the item" + marks.mark(1, TAG) + " Title: Blue ceramic mug, 350 ml\nPrice: 12 EUR"
    rows, p_sent, p_exp, state_found, row_found, counts, end = image._prepare_images(state, [picture], qs, TAG)
    ids = rows[0][0]
    at = ids.index(image.image_token_id)
    before = image.tok.decode(ids[: at - 1])  # up to the start-of-vision token that opens the placeholder
    after = image.tok.decode(ids[at + 2 : at + 9])  # past the pad token and the end-of-vision token
    assert before.rstrip().endswith("item") and after.lstrip().startswith("Title")  # inside the state
    assert state_found == [1] and row_found == [[1], [1]]
    # a request with images and the same request without them differ by the image tokens only
    trows = text._prepare_separate(marks.without_marks(state, TAG), qs)[0]
    for (ids, lab), (tids, tlab) in zip(rows, trows):
        assert without_placeholders(ids, image) == without_placeholders(tids, image) and lab == tlab


def test_i3_encodings_agree_and_route(served, picture):
    client, _, _ = served
    req = {"state": STATE, "questions": QUESTIONS}
    r_json = client.post("/v1/systemone", json={**req, "images": [data_url(picture)]})
    r_form = client.post(
        "/v1/systemone", data={"request": json.dumps(req)}, files={"image": ("p.png", png(picture), "image/png")}
    )
    r_state = client.post(
        "/v1/systemone", json={"state": {**STATE, "photo": data_url(picture)}, "questions": QUESTIONS}
    )
    for r in (r_json, r_form, r_state):
        assert r.status_code == 200, r.text
        assert r.headers["x-decisio-route"] == "image"
        assert all(a["unknown_probability"] == 0.0 and a["abstained"] is False for a in r.json()["answers"].values())
    # neither the json nor the multipart image carries a place of its own, so both open the state the same way
    assert probs(r_json.json()) == probs(r_form.json())
    assert (
        r_json.json()["usage"]["input_tokens"] > client.post("/v1/systemone", json=req).json()["usage"]["input_tokens"]
    )
    r_text = client.post("/v1/systemone", json=req)
    assert r_text.headers["x-decisio-route"] == "text"
    assert all("unknown_probability" not in a for a in r_text.json()["answers"].values())
    r_empty = client.post("/v1/systemone", json={**req, "images": []})  # the extension used, no image: text route
    assert r_empty.headers["x-decisio-route"] == "text"
    assert all(a["abstained"] is False for a in r_empty.json()["answers"].values())


def test_i7_every_encoding_of_a_picture_reads_the_same(served, picture):
    client, _, _ = served
    asset = {"type": "image", "format": "png", "data_base64": base64.b64encode(png(picture)).decode()}
    state = {**STATE, "photo": "PHOTO"}
    uri_state = {**STATE, "photo": data_url(picture)}
    r_assets = client.post("/v1/systemone", json={"state": state, "questions": QUESTIONS, "assets": {"PHOTO": asset}})
    r_uri = client.post("/v1/systemone", json={"state": uri_state, "questions": QUESTIONS})
    req = {"state": state, "questions": QUESTIONS, "assets": {"PHOTO": {"type": "image", "part": "p"}}}
    r_part = client.post(
        "/v1/systemone", data={"request": json.dumps(req)}, files={"p": ("p.png", png(picture), "image/png")}
    )
    # the images list names the place with a literal marker in the text
    hand = {**STATE, "photo": "[image 1]"}
    r_hand = client.post("/v1/systemone", json={"state": hand, "questions": QUESTIONS, "images": [data_url(picture)]})
    for r in (r_assets, r_uri, r_part, r_hand):
        assert r.status_code == 200, r.text
        assert r.headers["x-decisio-route"] == "image"
    assert probs(r_assets.json()) == probs(r_uri.json()) == probs(r_part.json()) == probs(r_hand.json())
    assert r_assets.json()["usage"] == r_uri.json()["usage"] == r_hand.json()["usage"]
    assert all("abstained" not in a for a in r_assets.json()["answers"].values())  # System One's fields only
    # a placeholder in a question's instructions is the same as the marker written there
    q_ph = {**QUESTIONS, "colour": {**QUESTIONS["colour"], "instructions": "Which colour is the mug in PHOTO?"}}
    q_hand = {**QUESTIONS, "colour": {**QUESTIONS["colour"], "instructions": "Which colour is the mug in [image 1]?"}}
    a = client.post("/v1/systemone", json={"state": STATE, "questions": q_ph, "assets": {"PHOTO": asset}})
    b = client.post("/v1/systemone", json={"state": STATE, "questions": q_hand, "images": [data_url(picture)]})
    assert a.status_code == b.status_code == 200 and probs(a.json()) == probs(b.json())
    # an image that no marker names opens the state, so it reads differently from the same picture placed at the end
    r_list = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS, "images": [data_url(picture)]})
    assert r_list.status_code == 200 and probs(r_list.json()) != probs(r_hand.json())


def test_i7_listed_images_are_placed_by_their_markers():
    from decisio.serve.systemone import place_listed_images

    payload = {
        "state": {"a": "see [image 2] and [image 2] again", "b": ["[image 9]"]},
        "questions": {
            "q": {"type": "noul", "instructions": "Is [image 1] the same?"},
            "c": {"type": "choice", "instructions": "x", "criteria": {"k": "like [image 3]", "j": None}},
        },
    }
    out, opening = place_listed_images(payload, 3, TAG)
    m = marks.mark
    assert out["state"]["a"] == f"see {m(2, TAG)} and [image 2] again"  # the first occurrence is the place
    assert out["state"]["b"] == ["[image 9]"]  # no such image: plain text
    assert out["questions"]["q"]["instructions"] == f"Is {m(1, TAG)} the same?"
    assert out["questions"]["c"]["criteria"] == {"k": f"like {m(3, TAG)}", "j": None}
    assert opening == []
    out, opening = place_listed_images({"state": "no marker", "questions": {}}, 2, TAG)
    assert out["state"] == "no marker" and opening == [1, 2]
    assert payload["state"]["a"].startswith("see [image 2]")  # the request's own payload is not changed


def test_i4_paired_text_check(served):
    client, _, _ = served
    req = {"state": "The shipment arrived two days late and the box was damaged.", "questions": QUESTIONS}
    a = client.post("/v1/systemone", json=req)
    b = client.post("/v1/systemone", json=req, headers={"x-decisio-route": "image"})
    assert a.headers["x-decisio-route"] == "text" and b.headers["x-decisio-route"] == "image"
    assert probs(a.json()) == probs(b.json())


def test_i5_refusals(served, picture):
    client, _, image = served
    req = {"state": STATE, "questions": QUESTIONS}
    image.max_images = 2  # an engine started with --max-images 2: the third image is refused (the default is 128)
    try:
        assert client.post("/v1/systemone", json={**req, "images": [data_url(picture)] * 3}).status_code == 422
    finally:
        del image.max_images
    assert client.post("/v1/systemone", json={**req, "images": ["data:text/plain;base64,aGk="]}).status_code == 422
    assert client.post("/v1/systemone", json={**req, "images": "not a list"}).status_code == 422
    assert (
        client.post(
            "/v1/systemone", data={"request": "{not json"}, files={"image": ("p.png", png(picture), "image/png")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/systemone", json={"state": {"photo": "data:image/png;base64,QUJD="}, "questions": QUESTIONS}
        ).status_code
        == 422
    )
    assert client.post("/v1/systemone", json=req, headers={"x-decisio-route": "gpu"}).status_code == 422


def test_i6_abstain_option_decodes_in_imajevs_harness(served):
    """`--abstain-option`: the extra option's probability becomes imajev's unknown_probability; imajev's own decoder
    (imajev_bench.runner.decode_jev) accepts every answer type and rebuilds a distribution that sums to one."""
    runner = pytest.importorskip("imajev_bench.runner")
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    _, text, image = served
    so = SystemOne(text, "decisio-test", image_engine=image, abstain_option="can't tell")
    client = TestClient(make_app(text, so))
    fields = {
        "boolean": {"id": "decision", "question": "Is the mug blue?", "type": "boolean"},
        "choice": {
            "id": "decision",
            "question": "Which colour?",
            "type": "choice",
            "options": [{"value": v, "description": None} for v in ("red", "blue", "white")],
        },
        "ordinal": {
            "id": "decision",
            "question": "How full is the mug?",
            "type": "ordinal",
            "levels": [{"value": i, "description": d} for i, d in enumerate(("empty", "half", "full"))],
        },
    }
    for kind, field in fields.items():
        payload = {"request": {"state": STATE, "fields": [field]}, "images": []}
        wire = {
            "state": STATE,
            "questions": {
                "decision": {
                    "instructions": field["question"],
                    **(
                        {"type": "noul", "criteria": {"true": None, "false": None}}
                        if kind == "boolean"
                        else {"type": "choice", "criteria": {o["value"]: None for o in field["options"]}}
                        if kind == "choice"
                        else {"type": "score", "criteria": [lv["description"] for lv in field["levels"]]}
                    ),
                }
            },
            "images": [],
        }
        r = client.post("/v1/systemone", json=wire)
        assert r.status_code == 200, r.text
        a = r.json()["answers"]["decision"]
        assert 0.0 < a["unknown_probability"] < 1.0 and isinstance(a["abstained"], bool)
        decoded = runner.decode_jev(r.json(), payload)
        probs = decoded[0] if isinstance(decoded, tuple) else decoded.get("probabilities", decoded)
        assert abs(sum(probs.values()) - 1) < 1e-6 and abs(probs["__unknown__"] - a["unknown_probability"]) < 1e-12, (
            kind
        )
    # requests without imajev's extension keep TypeSafe's format exactly
    plain = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).json()
    assert all("unknown_probability" not in a for a in plain["answers"].values())


# ---- I8: scope ---------------------------------------------------------------------------------------------------


def test_i8_a_question_image_is_in_its_own_row_only(served):
    _, _, image = served
    imgs = [picture_of(300, 200, 0), picture_of(200, 120, 1)]
    state = "Photo:" + marks.mark(1, TAG) + "Title: Blue ceramic mug"
    qs = [
        {"kind": "noul", "instructions": f"Does this match {marks.mark(2, TAG)}?"},
        {"kind": "noul", "instructions": "Is the mug blue?"},
    ]
    rows, p_sent, p_exp, state_found, row_found, counts, end = image._prepare_images(state, imgs, qs, TAG)
    assert state_found == [1] and row_found == [[1, 2], [1]]
    assert rows[0][0][:p_sent] == rows[1][0][:p_sent]  # the prefix is shared
    assert rows[0][0][p_sent:].count(image.image_token_id) == 1  # question A's own part holds its image
    assert image.image_token_id not in rows[1][0][p_sent:]  # question B's own part has no image
    probs_out, info = image.answer(state, qs, images=imgs, tag=TAG)
    # the state image once, question A's own image once; question B's tokens do not count it
    assert info["image_tokens"] == [counts[0], counts[1]]
    assert info["engine_prompt_tokens"] == info["expected_prompt_tokens"]


def test_i8_an_image_in_the_state_is_shared_not_repeated(served):
    _, _, image = served
    img = picture_of(300, 200)
    state = "Photo:" + marks.mark(1, TAG) + "Title: Blue ceramic mug"
    qs = [
        {"kind": "noul", "instructions": "Is the mug blue?"},
        {"kind": "choice", "instructions": "Which colour?", "options": ["blue", "red", "white"]},
    ]
    probs_out, info = image.answer(state, qs, images=[img], tag=TAG)
    assert info["image_tokens"] == [image.image_tokens(img)]  # once for the request, however many questions read it


# ---- I9: usage ----------------------------------------------------------------------------------------------------


def test_i9_usage_counts_text_and_every_image(served):
    client, _, image = served
    img1, img2 = picture_of(300, 200, 0), picture_of(120, 120, 1)
    asset1 = {"type": "image", "format": "png", "data_base64": base64.b64encode(png(img1)).decode()}
    asset2 = {"type": "image", "format": "png", "data_base64": base64.b64encode(png(img2)).decode()}

    def usage(body):
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        return r.json()["usage"]["input_tokens"]

    both = {
        "state": {"listing": "ITEM", "title": "Blue ceramic mug, 350 ml"},
        "questions": {
            "matches": {"type": "noul", "instructions": "Does the photo QPIC show the listed item?"},
            "colour": {
                "type": "choice",
                "instructions": "Which colour is the mug?",
                "criteria": {"blue": None, "red": None, "white": None},
            },
        },
        "assets": {"ITEM": asset1, "QPIC": asset2},
    }
    bare = {
        "state": {"listing": "", "title": "Blue ceramic mug, 350 ml"},  # the text without the places of the images
        "questions": {
            "matches": {"type": "noul", "instructions": "Does the photo  show the listed item?"},
            "colour": both["questions"]["colour"],
        },
    }
    assert usage(both) == usage(bare) + image.image_tokens(img1) + image.image_tokens(img2)
    colour_only = {**both, "questions": {"colour": both["questions"]["colour"]}, "assets": {"ITEM": asset1}}
    bare_colour_only = {**bare, "questions": {"colour": bare["questions"]["colour"]}}
    # the state's image is counted once, whatever the number of questions
    assert usage(colour_only) - usage(bare_colour_only) == image.image_tokens(img1)


# ---- I10: cache boundary -------------------------------------------------------------------------------------------


def test_i10_the_prefix_ends_after_the_last_state_images_end_token(served):
    _, _, image = served
    img = picture_of(300, 200)
    qs = [
        {"kind": "noul", "instructions": "Is the mug blue?"},
        {"kind": "choice", "instructions": "Which colour?", "options": ["blue", "red", "white"]},
    ]
    unit = image.match_unit
    for state in (marks.mark(1, TAG), "Photo: " + marks.mark(1, TAG), marks.mark(1, TAG) + " A short note. " * 2):
        rows, p_sent, p_exp, state_found, row_found, counts, end = image._prepare_images(state, [img], qs, TAG)
        assert end > 0 and p_exp >= end
        assert (p_exp // unit) * unit >= end, state  # the last whole block of the prefix is after the image


def test_i10_no_prefix_image_means_no_pad_for_the_boundary(served):
    _, _, image = served
    img = picture_of(300, 200)
    q = {"kind": "noul", "instructions": f"Is this {marks.mark(1, TAG)} a mug?"}
    rows, p_sent, p_exp, state_found, *_ = image._prepare_images("A short state.", [img], [q], TAG)
    assert state_found == [] and p_exp == p_sent


# ---- I11: the legacy layout reproduces main's old prompts byte for byte -------------------------------------------

LEGACY_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "image_route_legacy.json"


@pytest.fixture(scope="module")
def legacy_engine():
    from decisio.serve.hf_letters import HFImageLettersEngine

    return HFImageLettersEngine(MODEL, pad_to="block", pad_where="front", image_prompt="legacy")


def test_i11_legacy_prompt_matches_the_fixture_pinned_from_main(legacy_engine):
    fixture = json.loads(LEGACY_FIXTURE.read_text())
    e = legacy_engine
    assert e.image_prompt == "legacy"
    cases = {
        "two_images_state_dict": (
            {"listing": {"title": "Blue ceramic mug, 350 ml", "colour": "blue"}},
            [
                {"kind": "noul", "instructions": "Does the photo show the listed item?"},
                {
                    "kind": "choice",
                    "instructions": "Which colour is the mug in [image 1]?",
                    "options": ["blue", "red", "white"],
                },
            ],
            [picture_of(300, 200, 0), picture_of(64, 96, 0)],
        ),
        "one_image_text_state": (
            "Photo: [image 1]. A mug.",
            [{"kind": "noul", "instructions": "Does the photo show the listed item?"}],
            [picture_of(300, 200, 0)],
        ),
        "one_image_one_question": (
            {"a": "[image 1]"},
            [
                {
                    "kind": "choice",
                    "instructions": "Which colour is the mug in [image 1]?",
                    "options": ["blue", "red", "white"],
                }
            ],
            [picture_of(64, 96, 0)],
        ),
    }
    for name, (state, qs, imgs) in cases.items():
        e.scored_rows.clear()
        probs_out, info = e.answer(state, qs, images=imgs)
        want = fixture[name]
        got_rows = [[list(map(int, ids)), list(map(int, lab))] for ids, lab in e.scored_rows]
        assert got_rows == want["rows"], name
        assert info["shared_prefix_tokens"] == want["shared_prefix_tokens"], name
        assert info["image_tokens"] == want["image_tokens"], name


# ---- I12: the deprecation notice ------------------------------------------------------------------------------------


def test_i12_legacy_responses_carry_deprecation_headers(served, legacy_engine, caplog):
    import logging

    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app

    _, text, placed = served
    so = SystemOne(text, "decisio-test", image_engine=legacy_engine)
    so.LEGACY_NOTICE_EVERY = 3  # the log line comes on the first request and every third after it
    client = TestClient(make_app(text, so), headers={"x-decisio-client": "test"})
    req = {"state": STATE, "questions": QUESTIONS, "images": [data_url(picture_of(300, 200))]}

    def noticed():
        return sum("image-prompt legacy" in rec.getMessage() for rec in caplog.records)

    with caplog.at_level(logging.WARNING, logger="decisio.serve.systemone"):
        r1 = client.post("/v1/systemone", json=req)
        assert r1.status_code == 200, r1.text
        assert r1.headers["Deprecation"] == "true"
        assert "image-prompt legacy" in r1.headers["Warning"] and r1.headers["Warning"].startswith('299 - "')
        assert noticed() == 1
        client.post("/v1/systemone", json=req)  # the second: no new log line
        assert noticed() == 1
        client.post("/v1/systemone", json=req)  # the third: the line again
        assert noticed() == 2
    assert so.legacy_image_requests == 3
    # a text request on the legacy server has neither header and is not counted
    r_text = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS})
    assert "Deprecation" not in r_text.headers and so.legacy_image_requests == 3

    # the default layout carries neither header
    so2 = SystemOne(text, "decisio-test", image_engine=placed)
    client2 = TestClient(make_app(text, so2), headers={"x-decisio-client": "test"})
    r2 = client2.post("/v1/systemone", json=req)
    assert r2.status_code == 200 and "Deprecation" not in r2.headers and "Warning" not in r2.headers
    assert so2.legacy_image_requests == 0


# ---- I13: Box is refused on the Qwen route -------------------------------------------------------------------------


def test_i13_a_box_question_is_refused_with_a_clear_400(served, picture):
    client, _, _ = served
    req = {
        "state": STATE,
        "questions": {"where": {"type": "box", "instructions": "Where is the mug?"}},
        "images": [data_url(picture)],
    }
    r = client.post("/v1/systemone", json=req)
    assert r.status_code == 400, r.text
    assert "not answered on the --image-model route" in r.text and "Gemma 4 base" in r.text


def test_i14_the_qwen_image_scale_is_the_temperature_itself_on_the_placed_layout(served, legacy_engine):
    """The route reads images at 1.0, not at the text temperatures: a scale of 0.8 serves 0.8 (never 0.8 times the
    text temperature), on the placed layout only."""
    from fastapi.testclient import TestClient

    from decisio.families import QWEN
    from decisio.serve.systemone import SystemOne
    from decisio.serve.temperature import apply_temperature
    from decisio.serve.vllm_engine import make_app

    client, text, placed = served
    assert QWEN.image_temperature_scale == 0.80
    req = {"state": STATE, "questions": QUESTIONS, "images": [data_url(picture_of(300, 200))]}
    plain = client.post("/v1/systemone", json=req).json()["answers"]["colour"]["probabilities"]

    def read(image_engine, **kw):
        so = SystemOne(text, "decisio-test", image_engine=image_engine, temperature=1.506, **kw)
        c = TestClient(make_app(text, so), headers={"x-decisio-client": "test"})
        return so, c.post("/v1/systemone", json=req).json()["answers"]["colour"]["probabilities"]

    so, scaled = read(placed, image_temperature_scale=0.80)
    assert so.qwen_image_temperature() == 0.80
    want = apply_temperature(np.asarray(list(plain.values()), dtype=np.float64), 0.80)
    assert np.allclose(list(scaled.values()), want, atol=1e-6)
    legacy_so = SystemOne(text, "decisio-test", image_engine=legacy_engine, image_temperature_scale=0.80)
    assert legacy_so.qwen_image_temperature() == 1.0  # the fit was for the placed layout
    assert SystemOne(text, "decisio-test", image_engine=placed).qwen_image_temperature() == 1.0  # no scale
