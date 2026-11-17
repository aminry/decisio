# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images on a Mac with a real checkpoint (`--backend mlx --images on`, decisio.serve.mlx_images). Apple silicon, the
`mlx-images` extra and a checkpoint with the 12B's vision weights; run under the machine's memory guard:

  DECISIO_MLX_IMAGES_MODEL=<directory or repository> [DECISIO_MLX_IMAGES_VISION=<add-on file or folder>] \\
  [DECISIO_MLX_IMAGES_BASE=gemma-4-12b|gemma-4-31b] [DECISIO_MLX_IMAGES_REPORT=<json file>] \\
      uv run pytest -q -m mlx_model tests/mlx/test_mlx_images_real.py

  DECISIO_MLX_IMAGES_VISION is for a conversion without vision weights (the tuned 12B's MLX repository): the base's
  add-on (decisio.serve.mlx_vision). The corners run at budgets 280 and 560.

  R1  corners: a square in each corner, "top or bottom" and "left or right" asked of each: 8 of 8 right (the image
      blocks are bidirectional; without the token types a top corner is read as "bottom")
  (parity with decisio on vLLM, through the served request path: tests/mlx/test_mlx_images_vllm_parity.py)
  R4  memory and latency per image at 560 (written to DECISIO_MLX_IMAGES_REPORT when set)
  R3  a second question on the same image state reads from the prefix cache with the cold read's probabilities
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from decisio.families import GEMMA4, GEMMA4_31B
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

    family = GEMMA4_31B if os.environ.get("DECISIO_MLX_IMAGES_BASE") == "gemma-4-31b" else GEMMA4
    e = MLXGemmaImageLettersEngine(
        os.environ["DECISIO_MLX_IMAGES_MODEL"],
        family=family,
        image_budget=280,
        vision_weights=os.environ.get("DECISIO_MLX_IMAGES_VISION") or None,
    )
    e.fmt, e.family, e.pad_policy = FORMAT, family, "none"
    return e


def corner_image(corner):
    im = Image.new("RGB", (640, 640), (235, 235, 235))
    x = 40 if "left" in corner else 640 - 40 - 160
    y = 40 if "top" in corner else 640 - 40 - 160
    ImageDraw.Draw(im).rectangle([x, y, x + 160, y + 160], fill=(200, 30, 30))
    return im


@pytest.mark.parametrize("budget", [280, 560])
def test_r1_a_square_in_each_corner_is_placed(engine, budget):
    engine.image_budget = budget
    engine.prefix_cache.entries.clear()
    engine.prefix_cache.bytes = 0
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
    engine.image_budget = 280


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


def test_r4_memory_and_latency_per_image_at_560(engine):
    import time

    import mlx.core as mx

    from decisio.serve import mlx_memory

    engine.image_budget = 560
    state = f"Picture: {marks.mark(1, TAG)}"
    q = {"kind": "noul", "instructions": "Is there a red square in the picture?"}
    rows = []
    for i, corner in enumerate(["top left", "top right", "bottom left", "bottom right"] * 3):
        engine.prefix_cache.entries.clear()
        engine.prefix_cache.bytes = 0
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        _, info = engine.answer(state, [q], images=[corner_image(corner)], tag=TAG)
        rows.append(
            {
                "ms": (time.perf_counter() - t0) * 1000,
                "prefix_ms": info["prefix_ms"],
                "prompt_tokens": info["prompt_tokens"],
                "peak_gb": mx.get_peak_memory() / 2**30,
                "footprint_gb": mlx_memory.physical_footprint_bytes() / 2**30,
            }
        )
    warm = rows[2:]  # the first requests compile
    report = {
        "budget": 560,
        "n": len(warm),
        "ms_median": float(np.median([r["ms"] for r in warm])),
        "prefix_ms_median": float(np.median([r["prefix_ms"] for r in warm])),
        "prompt_tokens": warm[0]["prompt_tokens"],
        "peak_gb_max": max(r["peak_gb"] for r in rows),
        "footprint_gb_max": max(r["footprint_gb"] for r in rows),
        "footprint_gb_first_to_last": [rows[0]["footprint_gb"], rows[-1]["footprint_gb"]],
        "vision": engine.vision,
    }
    print("R4", json.dumps(report))
    if os.environ.get("DECISIO_MLX_IMAGES_REPORT"):
        Path(os.environ["DECISIO_MLX_IMAGES_REPORT"]).write_text(json.dumps(report, indent=1))
    assert report["footprint_gb_first_to_last"][1] < report["footprint_gb_first_to_last"][0] + 2  # memory stays flat
    engine.image_budget = 280
