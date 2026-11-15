# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images on a Mac with a real checkpoint (`--backend mlx --images on`, decisio.serve.mlx_images). Apple silicon, the
`mlx-images` extra and a checkpoint with the 12B's vision weights; run under the machine's memory guard:

  DECISIO_MLX_IMAGES_MODEL=<directory or repository> uv run pytest -q -m mlx_model tests/mlx/test_mlx_images_real.py

  R1  corners: a square in each corner, "top or bottom" and "left or right" asked of each: 8 of 8 right (the image
      blocks are bidirectional; without the token types a top corner is read as "bottom")
  R2  parity: the letters of 20 image questions (DECISIO_MLX_IMAGES_PARITY: a directory with `nc_base.jsonl` and
      `S_nc_manifest.json`, the first read of the same items with mlx-vlm and the token-type fix, budget 280) agree
      within 0.02
  R3  a second question on the same image state reads from the prefix cache with the cold read's probabilities
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from decisio.families import GEMMA4
from decisio.readout.letters import PromptFormat
from decisio.serve import image_marks as marks

pytestmark = [
    pytest.mark.mlx_model,
    pytest.mark.skipif(not os.environ.get("DECISIO_MLX_IMAGES_MODEL"), reason="DECISIO_MLX_IMAGES_MODEL is not set"),
]
TAG = "0123456789abcdef"
FORMAT = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("mlx_vlm")
    from decisio.serve.mlx_images import MLXGemmaImageLettersEngine

    e = MLXGemmaImageLettersEngine(os.environ["DECISIO_MLX_IMAGES_MODEL"], family=GEMMA4, image_budget=280)
    e.fmt, e.family, e.pad_policy = FORMAT, GEMMA4, "none"
    return e


def corner_image(corner):
    im = Image.new("RGB", (640, 640), (235, 235, 235))
    x = 40 if "left" in corner else 640 - 40 - 160
    y = 40 if "top" in corner else 640 - 40 - 160
    ImageDraw.Draw(im).rectangle([x, y, x + 160, y + 160], fill=(200, 30, 30))
    return im


def test_r1_a_square_in_each_corner_is_placed(engine):
    right = 0
    for corner in ("top left", "top right", "bottom left", "bottom right"):
        vert, horiz = corner.split()
        qs = [
            {
                "kind": "choice",
                "instructions": "Is the red square in the top half or the bottom half of the picture?",
                "options": ["top", "bottom"],
            },
            {
                "kind": "choice",
                "instructions": "Is the red square in the left half or the right half of the picture?",
                "options": ["left", "right"],
            },
        ]
        probs, _ = engine.answer(f"Picture: {marks.mark(1, TAG)}", qs, images=[corner_image(corner)], tag=TAG)
        right += int(np.argmax(probs[0]) == ["top", "bottom"].index(vert))
        right += int(np.argmax(probs[1]) == ["left", "right"].index(horiz))
    assert right == 8


def test_r2_the_first_read_of_p0_agrees(engine):
    d = os.environ.get("DECISIO_MLX_IMAGES_PARITY")
    if not d:
        pytest.skip("DECISIO_MLX_IMAGES_PARITY is not set")
    items = {i["id"]: i for i in json.load(open(Path(d) / "S_nc_manifest.json"))}
    rows = [json.loads(line) for line in open(Path(d) / "nc_base.jsonl")]
    rows = [r for r in rows if r["key"].endswith("|280") and r["kind"] in ("exists", "count", "colour", "picture")][:20]
    assert len(rows) == 20
    for r in rows:
        it = items[r["item"]]
        letters = "ABCD"[: len(it["options"])]
        q = (
            it["question"]
            + "\n\nOptions:\n"
            + "\n".join(f"{k}. {o}" for k, o in zip(letters, it["options"]))
            + "\n\nAnswer with the letter of the correct option only."
        )
        imgs = [Image.open(p).convert("RGB") for p in it["images"]]
        text = engine.tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}] * len(imgs) + [{"type": "text", "text": q}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = engine.tok.encode(text, add_special_tokens=False)
        exp = engine._expand(ids, [engine._count(i) for i in imgs])
        h = engine._feed_images(exp, engine.model.make_cache(), imgs)
        z = np.array(
            engine._logits(h, [engine.tok.convert_tokens_to_ids(k) for k in letters]).astype("float32"),
            dtype=np.float64,
        )
        p = np.exp(z - z.max())
        p /= p.sum()
        assert np.allclose(p, r["p"][: len(p)] / np.sum(r["p"][: len(p)]), atol=0.02), r["key"]


def test_r3_a_second_question_reads_the_cached_state(engine):
    img = corner_image("top left")
    state = f"Picture: {marks.mark(1, TAG)}"
    q = {"kind": "noul", "instructions": "Is there a red square in the picture?"}
    a, i1 = engine.answer(state, [q], images=[img], tag=TAG)
    engine.prefix_cache.entries.clear()
    engine.prefix_cache.bytes = 0
    cold, _ = engine.answer(state, [q], images=[img], tag=TAG)
    warm, i2 = engine.answer(state, [q], images=[img], tag=TAG)
    assert i2["prefix_cache_hit"] and np.allclose(warm[0], cold[0], atol=0.005)
