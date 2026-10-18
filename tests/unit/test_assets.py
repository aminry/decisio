# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""System One's `assets` (placeholders bound to images) on `/v1/systemone`, with fake engines that record what they get.

A1  an image sent through `assets` reaches the image engine exactly as the same image sent through the encodings
    decisio already had: the same state, the same question texts and the same image, so the same probabilities
A2  a placeholder is found in the state (a string, inside a string, inside an object or an array), in a question's
    instructions and in its criteria descriptions (choice, noul, score), not in option keys; longest name first
A3  a placeholder used twice is two images, in reading order; more images than the image engine takes is a 400
A4  a placeholder with no asset stays text; an asset used nowhere, an empty `assets`, a nameless asset, a bad format,
    bad base64 and bytes that are not an image are 400 with System One's error shape
A5  `format` may disagree with the bytes (the bytes decide, as on System One); `type` defaults to image
A6  multipart: assets name image parts, a part wins over `data_base64`, the request part comes first, parts are named,
    unique and each used once, and a missing part is a 400
A7  the answers carry System One's fields only (no imajev fields); a score's legend shows the levels as written
A8  assets do not need System One's `TypeSafe-Danger-Zone-Images` header; `assets` with `images` is a 400
A9  an abstention task's examples may carry assets; a calibration or head task's examples may not
A10 a server without an image engine says so

  uv run pytest -q tests/unit/test_assets.py
"""

import base64
import hashlib
import io
import json
import zlib

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore


def png(color, size=(40, 30)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


RED, BLUE, GREEN = png((220, 20, 20)), png((20, 40, 220)), png((20, 200, 40))


def b64(b):
    return base64.b64encode(b).decode()


def asset(b, fmt="png"):
    return {"type": "image", "format": fmt, "data_base64": b64(b)}


def data_url(b):
    return "data:image/png;base64," + b64(b)


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class Fake:
    tok = FakeTok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters = {}

    def __init__(self):
        self.calls = []

    def facts(self):
        return {"engine": "fake", "base": "fake-base"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        # the probabilities depend on everything the engine is given, so equal probabilities mean equal input
        seen = [hashlib.sha256(i.tobytes()).hexdigest()[:8] for i in images or []]
        self.calls.append({"state": state, "questions": json.loads(json.dumps(questions)), "images": seen})
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            h = zlib.crc32(json.dumps([state, q, seen], sort_keys=True).encode()) % 1000 / 1000
            p = np.full(k, (1 - (0.5 + h / 2)) / (k - 1))
            p[0] = 0.5 + h / 2
            out.append(p)
        return out, {"image_tokens": [7] * len(images or [])}


def make(image=True):
    text, img = Fake(), Fake()
    so = SystemOne(text, "decisio-test", image_engine=img if image else None, task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), text, img


COLOR = {"type": "choice", "instructions": "What colour is the shape?", "criteria": {"red": None, "blue": None}}


def req(state, q=COLOR, assets=None, **extra):
    body = {"model": "m", "state": state, "questions": {"q": q}, **extra}
    if assets is not None:
        body["assets"] = assets
    return body


def probs(r):
    return {n: a.get("probabilities") or a["noul"] for n, a in r.json()["answers"].items()}


def test_a1_assets_reach_the_model_as_images_do():
    c, text, img = make()
    via_assets = c.post("/v1/systemone", json=req("PHOTO", assets={"PHOTO": asset(RED)}))
    via_uri = c.post("/v1/systemone", json=req(data_url(RED)))
    assert via_assets.status_code == via_uri.status_code == 200
    assert img.calls[0] == img.calls[1] and img.calls[0]["state"] == "[image 1]" and len(img.calls[0]["images"]) == 1
    assert probs(via_assets) == probs(via_uri) and text.calls == []
    assert via_assets.headers["x-decisio-route"] == "image"
    # the instructions and the criteria, against the images field with the markers written by hand
    q = {"type": "choice", "instructions": "Is PHOTO_1 red?", "criteria": {"yes": "like PHOTO_2", "no": "unlike it"}}
    a = c.post("/v1/systemone", json=req("see both", q, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    q_hand = {**q, "instructions": "Is [image 1] red?", "criteria": {"yes": "like [image 2]", "no": "unlike it"}}
    b = c.post("/v1/systemone", json=req("see both", q_hand, images=[data_url(RED), data_url(BLUE)]))
    assert a.status_code == b.status_code == 200
    assert img.calls[-2] == img.calls[-1] and probs(a) == probs(b)


def test_a2_where_placeholders_are_found():
    c, _, img = make()
    # in the state: inside a string, inside an object, inside an array; reading order is the state's order
    state = {"title": "Report", "pictures": ["PHOTO_1"], "note": {"text": "see PHOTO_2 too"}}
    r = c.post("/v1/systemone", json=req(state, COLOR, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    assert r.status_code == 200 and len(img.calls[-1]["images"]) == 2
    assert img.calls[-1]["state"] == {
        "title": "Report",
        "pictures": ["[image 1]"],
        "note": {"text": "see [image 2] too"},
    }
    r = c.post("/v1/systemone", json=req(["first PHOTO_1", "second"], COLOR, {"PHOTO_1": asset(RED)}))
    assert r.status_code == 200 and img.calls[-1]["state"] == ["first [image 1]", "second"]
    # in the instructions and the descriptions of choice, noul and score criteria
    q = {"type": "noul", "instructions": "About PHOTO_1?", "criteria": {"true": "is like PHOTO_2", "false": "no"}}
    r = c.post("/v1/systemone", json=req("text", q, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    assert r.status_code == 200 and len(img.calls[-1]["images"]) == 2
    s = {"type": "score", "instructions": "Which is closer?", "criteria": ["like PHOTO_1", "like PHOTO_2"]}
    r = c.post("/v1/systemone", json=req("text", s, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    assert r.status_code == 200 and img.calls[-1]["questions"][0]["options"] == ["like [image 1]", "like [image 2]"]
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "PHOTO_1", "b": "PHOTO_2"}}
    r = c.post("/v1/systemone", json=req("text", q, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    assert r.status_code == 200 and "[image 1]" in json.dumps(img.calls[-1]["questions"])
    # option keys are answers' names: a placeholder there is not searched, so its asset is unused
    k = {"type": "choice", "instructions": "Which?", "criteria": {"PHOTO_1": "the first", "other": None}}
    r = c.post("/v1/systemone", json=req("a red shape", k, {"PHOTO_1": asset(RED)}))
    assert r.status_code == 400 and "not used" in r.json()["detail"]["message"]
    # longest name first: PHOTO_10 is not PHOTO_1 followed by a zero
    r = c.post(
        "/v1/systemone", json=req("PHOTO_10 and PHOTO_1", COLOR, {"PHOTO_1": asset(RED), "PHOTO_10": asset(BLUE)})
    )
    assert r.status_code == 200 and img.calls[-1]["state"] == "[image 1] and [image 2]"
    pixels = [hashlib.sha256(Image.open(io.BytesIO(x)).convert("RGB").tobytes()).hexdigest()[:8] for x in (BLUE, RED)]
    assert img.calls[-1]["images"] == pixels  # PHOTO_10 (blue) first, then PHOTO_1 (red)


def test_a3_a_placeholder_used_twice_is_two_images():
    c, _, img = make()
    r = c.post("/v1/systemone", json=req("PHOTO and PHOTO", assets={"PHOTO": asset(RED)}))
    assert (
        r.status_code == 200
        and img.calls[-1]["state"] == "[image 1] and [image 2]"
        and len(img.calls[-1]["images"]) == 2
    )
    r = c.post("/v1/systemone", json=req("PHOTO PHOTO PHOTO", assets={"PHOTO": asset(RED)}))
    assert (
        r.status_code == 400
        and r.json()["detail"]["error_type"] == "api_usage_error"
        and "at most 2 images" in r.json()["detail"]["message"]
    )


def test_a4_unusable_requests_are_400_in_system_ones_shape():
    c, _, _ = make()

    def bad(assets, state="PHOTO", fragment=""):
        r = c.post("/v1/systemone", json=req(state, assets=assets))
        assert r.status_code == 400, (assets, r.text)
        d = r.json()["detail"]
        assert d["error_type"] == "api_usage_error" and fragment in d["message"], d

    r = c.post("/v1/systemone", json=req("PHOTO_1 and PHOTO_9", assets={"PHOTO_1": asset(RED)}))
    assert r.status_code == 200  # an unknown placeholder is plain text
    bad({"PHOTO": asset(RED)}, state="no placeholder here", fragment="not used")
    bad({}, fragment="non-empty")
    bad({"": asset(RED)}, fragment="nonempty placeholder name")
    bad({"PHOTO": {"type": "image", "data_base64": b64(RED)}}, fragment="format must be png, jpg or jpeg")
    bad({"PHOTO": {**asset(RED), "format": "webp"}}, fragment="png, jpg or jpeg")
    bad({"PHOTO": {"type": "video", "format": "png", "data_base64": b64(RED)}}, fragment="PHOTO")
    bad({"PHOTO": {"type": "image", "format": "png", "data_base64": "not base64 !!"}}, fragment="not valid base64")
    bad({"PHOTO": {"type": "image", "format": "png", "data_base64": ""}}, fragment="empty")
    bad({"PHOTO": asset(b"hello, not an image")}, fragment="Invalid image")
    bad({"PHOTO": {"type": "image", "format": "png"}}, fragment="empty")


def test_a5_the_bytes_decide_the_format():
    c, _, img = make()
    jpg = io.BytesIO()
    Image.new("RGB", (40, 30), (220, 20, 20)).save(jpg, "JPEG")
    for a in (
        asset(RED, "jpeg"),
        asset(RED, "jpg"),
        asset(jpg.getvalue(), "png"),
        {"format": "png", "data_base64": b64(RED)},
    ):
        assert c.post("/v1/systemone", json=req("PHOTO", assets={"PHOTO": a})).status_code == 200, a


def multipart(body, parts, order=None):
    files = [(name, (f"{name}.png", data, "image/png")) for name, data in parts]
    return c_post(body, files)


def c_post(body, files):
    client = c_post.client
    return client.post(
        "/v1/systemone", data={"request": json.dumps(body)}, files=files, headers={"x-decisio-client": "t"}
    )


def test_a6_multipart_parts():
    c, _, img = make()
    c_post.client = c
    body = req("PHOTO", assets={"PHOTO": {"type": "image", "part": "left"}})
    r = multipart(body, [("left", RED)])
    assert r.status_code == 200 and img.calls[-1]["state"] == "[image 1]"
    via_json = c.post("/v1/systemone", json=req("PHOTO", assets={"PHOTO": asset(RED)}))
    assert probs(r) == probs(via_json)
    # a part wins over data_base64
    both = req("PHOTO", assets={"PHOTO": {**asset(BLUE), "part": "left"}})
    assert multipart(both, [("left", RED)]).status_code == 200 and img.calls[-1] == img.calls[-2]
    msgs = {}
    msgs["missing"] = multipart(body, [("other", RED)])
    msgs["unused"] = multipart(body, [("left", RED), ("extra", BLUE)])
    msgs["twice"] = multipart(
        req("PHOTO_1 PHOTO_2", assets={"PHOTO_1": {"part": "left"}, "PHOTO_2": {"part": "left"}}), [("left", RED)]
    )
    for k, r in msgs.items():
        assert r.status_code == 400 and r.json()["detail"]["error_type"] == "api_usage_error", (k, r.text)
    assert "no image part named" in msgs["missing"].json()["detail"]["message"]
    assert "not used by any asset" in msgs["unused"].json()["detail"]["message"]
    assert "more than one asset" in msgs["twice"].json()["detail"]["message"]
    # the request part must come first, and parts need names
    r = c.post(
        "/v1/systemone",
        files=[("left", ("l.png", RED, "image/png")), ("request", (None, json.dumps(body)))],
        headers={"x-decisio-client": "t"},
    )
    assert r.status_code == 400 and "must start with" in r.json()["detail"]["message"]


def test_a7_system_ones_fields_only_and_the_legend_as_written():
    c, _, _ = make()
    s = {"type": "score", "instructions": "Which is closer?", "criteria": ["like PHOTO_1", "like PHOTO_2"]}
    r = c.post("/v1/systemone", json=req("text", s, {"PHOTO_1": asset(RED), "PHOTO_2": asset(BLUE)}))
    a = r.json()["answers"]["q"]
    assert (
        a["legend"] == {"0": "like PHOTO_1", "1": "like PHOTO_2"}
        and "unknown_probability" not in a
        and "abstained" not in a
    )
    plain = c.post("/v1/systemone", json=req("x", images=[data_url(RED)])).json()["answers"]["q"]
    assert plain["abstained"] is False  # the imajev encodings keep their extension fields


def test_a8_no_danger_zone_header_and_no_mixing():
    c, _, _ = make()
    r = c.post(
        "/v1/systemone", json=req("PHOTO", assets={"PHOTO": asset(RED)})
    )  # no TypeSafe-Danger-Zone-Images header
    assert r.status_code == 200
    r = c.post("/v1/systemone", json=req("PHOTO", assets={"PHOTO": asset(RED)}, images=[data_url(BLUE)]))
    assert r.status_code == 400 and "not both" in r.json()["detail"]["message"]


def test_a9_task_examples_carry_assets_only_where_images_are_served():
    c, text, img = make()
    q = {"type": "choice", "instructions": "Colour?", "criteria": {"red": None, "blue": None, "unsure": None}}
    ex = [
        {"request": req("PHOTO", q, {"PHOTO": asset(RED if i % 2 else BLUE)}), "answer": "red" if i % 2 else None}
        for i in range(12)
    ]
    body = {"id": "shapes", "option": {"key": "unsure"}, "match": "option_set", "examples": ex}
    r = c.post("/v1/abstention/tasks", json=body)
    assert r.status_code == 200, r.text
    assert img.calls and all(len(call["images"]) == 1 for call in img.calls) and text.calls == []
    # a calibration or head task is fitted on text: an example with assets is refused, not answered without its image
    r = c.post("/v1/tasks", json={"id": "t", "examples": ex[:1]})
    assert r.status_code == 422 and "text requests only" in r.text


def test_a10_no_image_engine():
    c, _, _ = make(image=False)
    r = c.post("/v1/systemone", json=req("PHOTO", assets={"PHOTO": asset(RED)}))
    assert r.status_code == 400 and "no image engine" in json.dumps(r.json())
