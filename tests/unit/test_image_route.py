# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The image route of `/v1/systemone` (decisio.serve.image_engine, decisio.serve.systemone) on the CPU stand-in.

Both engines are the CPU stand-ins (decisio.serve.hf_letters) on a small checkpoint of the served model's family, so the
checks are about the route, the prompt and the wire format, not about any number:

  I1  the image token count (`smart_resize` arithmetic) equals the model's own processor on odd and large sizes;
  I2  an image prompt is the text route's prompt with the images at the start of the user turn: the same question
      tokens after the cached prefix, one `<|image_pad|>` per image in the prefix, the expanded prefix on the block;
  I3  the three encodings imajev accepts (JSON `images`, multipart `request` + `image`, a data:image URI in the state)
      reach the image engine and give the same probabilities; imajev's two answer fields appear exactly when the
      request uses its extension; text requests stay on the text engine with TypeSafe's format;
  I4  the paired text check: a text request through the text route and through the image engine (`x-decisio-route:
      image`) gives bit-identical probabilities;
  I5  refusals are 422: three images, a bad data URL, a bad multipart request field, an unknown route.

Needs transformers with the Qwen3.5 classes, torchvision, pillow, fastapi, httpx and python-multipart (the package's
dependencies and the dev extra); I6 also needs imajev's harness (`imajev_bench`) and is skipped without it:

    uv run pytest -q tests/unit/test_image_route.py
"""
import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image  # noqa: E402

MODEL = os.environ.get("DECISIO_IMAGE_TEST_MODEL", "Qwen/Qwen3.5-0.8B-Base")
STATE = {"listing": {"title": "Blue ceramic mug, 350 ml", "colour": "blue"}}
QUESTIONS = {"matches": {"type": "noul", "instructions": "Does the photo show the listed item?"},
             "colour": {"type": "choice", "instructions": "Which colour is the mug in the photo?",
                        "criteria": {"blue": None, "red": None, "white": None}}}


def png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def data_url(img):
    return "data:image/png;base64," + base64.b64encode(png(img)).decode()


@pytest.fixture(scope="module")
def picture():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8), "RGB")


@pytest.fixture(scope="module")
def served():
    from fastapi.testclient import TestClient

    from decisio.serve.hf_letters import HFImageLettersEngine, HFLettersEngine
    from decisio.serve.systemone import SystemOne
    from decisio.serve.vllm_engine import make_app
    text = HFLettersEngine(MODEL, pad_to="block", pad_where="front")
    image = HFImageLettersEngine(MODEL, pad_to="block", pad_where="front")
    so = SystemOne(text, "decisio-test", image_engine=image)
    return TestClient(make_app(text, so)), text, image


def probs(body):
    out = {}
    for name, a in body["answers"].items():
        out[name] = [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
    return out


def test_i1_token_count_matches_processor(served):
    _, _, image = served
    for w, h in [(1536, 1024), (1264, 848), (33, 17), (4000, 5000), (10, 10), (8192, 300), (1000, 999)]:
        img = Image.new("RGB", (w, h))
        g = image.processor.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0].tolist()
        assert image.image_tokens(img) == g[0] * g[1] * g[2] // image.merge ** 2, (w, h)


def test_i2_prompt_is_the_text_prompt_with_images_first(served, picture):
    from decisio.serve.vllm_engine import USER_HEAD
    _, text, image = served
    qs = [{"kind": "noul", "instructions": "Does the photo show the listed item?"},
          {"kind": "choice", "instructions": "Which colour?", "options": ["blue", "red", "white"]}]
    rows, p_exp, p_sent, counts = image._prepare_images(STATE, [picture, picture], qs)
    trows, tp = text._prepare_separate(STATE, qs)[:2]
    assert p_exp % image.pad_unit == 0 and p_exp == p_sent + sum(c - 1 for c in counts)
    for (ids, lab), (tids, tlab) in zip(rows, trows):
        assert ids[p_sent:] == tids[tp:] and lab == tlab                # the question is read exactly as for text
        assert ids[:p_sent].count(image.image_pad_id) == 2 and image.image_pad_id not in ids[p_sent:]
    # the images open the user turn, as the model's chat template renders them
    head = image.tok.encode(USER_HEAD + "<|vision_start|><|image_pad|><|vision_end|>", add_special_tokens=False)
    ids = rows[0][0]
    k = next(i for i, t in enumerate(ids) if t != image.pad_token)          # the front padding, then the template
    assert ids[k:k + len(head)] == head


def test_i3_encodings_agree_and_route(served, picture):
    client, _, _ = served
    req = {"state": STATE, "questions": QUESTIONS}
    r_json = client.post("/v1/systemone", json={**req, "images": [data_url(picture)]})
    r_form = client.post("/v1/systemone", data={"request": json.dumps(req)},
                         files={"image": ("p.png", png(picture), "image/png")})
    r_state = client.post("/v1/systemone", json={"state": {**STATE, "photo": data_url(picture)},
                                                 "questions": QUESTIONS})
    for r in (r_json, r_form, r_state):
        assert r.status_code == 200, r.text
        assert r.headers["x-decisio-route"] == "image"
        assert all(a["unknown_probability"] == 0.0 and a["abstained"] is False for a in r.json()["answers"].values())
    # the state-embedded image leaves "[image 1]" in the state, so only the two image-list encodings are identical
    assert probs(r_json.json()) == probs(r_form.json())
    assert (r_json.json()["usage"]["input_tokens"]
            > client.post("/v1/systemone", json=req).json()["usage"]["input_tokens"])
    r_text = client.post("/v1/systemone", json=req)
    assert r_text.headers["x-decisio-route"] == "text"
    assert all("unknown_probability" not in a for a in r_text.json()["answers"].values())
    r_empty = client.post("/v1/systemone", json={**req, "images": []})     # the extension used, no image: text route
    assert r_empty.headers["x-decisio-route"] == "text"
    assert all(a["abstained"] is False for a in r_empty.json()["answers"].values())


def test_i4_paired_text_check(served):
    client, _, _ = served
    req = {"state": "The shipment arrived two days late and the box was damaged.", "questions": QUESTIONS}
    a = client.post("/v1/systemone", json=req)
    b = client.post("/v1/systemone", json=req, headers={"x-decisio-route": "image"})
    assert a.headers["x-decisio-route"] == "text" and b.headers["x-decisio-route"] == "image"
    assert probs(a.json()) == probs(b.json())


def test_i5_refusals(served, picture):
    client, _, _ = served
    req = {"state": STATE, "questions": QUESTIONS}
    assert client.post("/v1/systemone", json={**req, "images": [data_url(picture)] * 3}).status_code == 422
    assert client.post("/v1/systemone", json={**req, "images": ["data:text/plain;base64,aGk="]}).status_code == 422
    assert client.post("/v1/systemone", json={**req, "images": "not a list"}).status_code == 422
    assert client.post("/v1/systemone", data={"request": "{not json"},
                       files={"image": ("p.png", png(picture), "image/png")}).status_code == 422
    assert client.post("/v1/systemone", json={"state": {"photo": "data:image/png;base64,QUJD="}, "questions": QUESTIONS}
                       ).status_code == 422
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
    fields = {"boolean": {"id": "decision", "question": "Is the mug blue?", "type": "boolean"},
              "choice": {"id": "decision", "question": "Which colour?", "type": "choice",
                         "options": [{"value": v, "description": None} for v in ("red", "blue", "white")]},
              "ordinal": {"id": "decision", "question": "How full is the mug?", "type": "ordinal",
                          "levels": [{"value": i, "description": d} for i, d in enumerate(("empty", "half", "full"))]}}
    for kind, field in fields.items():
        payload = {"request": {"state": STATE, "fields": [field]}, "images": []}
        wire = {"state": STATE, "questions": {"decision": {"instructions": field["question"],
                **({"type": "noul", "criteria": {"true": None, "false": None}} if kind == "boolean" else
                   {"type": "choice", "criteria": {o["value"]: None for o in field["options"]}} if kind == "choice" else
                   {"type": "score", "criteria": [lv["description"] for lv in field["levels"]]})}}, "images": []}
        r = client.post("/v1/systemone", json=wire)
        assert r.status_code == 200, r.text
        a = r.json()["answers"]["decision"]
        assert 0.0 < a["unknown_probability"] < 1.0 and isinstance(a["abstained"], bool)
        decoded = runner.decode_jev(r.json(), payload)
        probs = decoded[0] if isinstance(decoded, tuple) else decoded.get("probabilities", decoded)
        assert (abs(sum(probs.values()) - 1) < 1e-6
                and abs(probs["__unknown__"] - a["unknown_probability"]) < 1e-12), kind
    # requests without imajev's extension keep TypeSafe's format exactly
    plain = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).json()
    assert all("unknown_probability" not in a for a in plain["answers"].values())
