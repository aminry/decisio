# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The hosted players' clips: their caption, moves without probabilities, and the closing scoreboard card."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/demos/tools"))
pytest.importorskip("PIL")

HOSTED = {
    "label": "Jev 1.13 through its public API",
    "model_id": "jev-1.13.0",
    "date": "2026-10-03",
    "route": "TypeSafe public API",
    "provider": "TypeSafe",
    "card": "Apple M5 Pro (client); the model served remotely",
}
TICKS = [
    {"decision": {"answer": {}, "latency_ms": 100.0}},
    {"decision": {"answer": {}, "latency_ms": 140.0}},
    {"decision": {"rules_fallback": True, "latency_ms": 1501.0}},
]


def test_h1_a_hosted_caption_names_the_model_date_route_and_says_it_is_model_output():
    import render_common as rc

    text = rc.caption_text(HOSTED, TICKS)
    for part in (
        "Jev 1.13 through its public API (jev-1.13.0)",
        "2026-10-03",
        "TypeSafe public API",
        "median 120 ms per decision",
        "1 of 3 requests without an answer in time",
        "from a client on an Apple M5 Pro",
        "model output",
        "AI-generated",
        "rendered by the decisio project",
    ):
        assert part in text, part
    ours = rc.caption_text(
        {"label": "Decisio", "route": "local server on the card", "card": "card", "padding": "row"}, TICKS
    )
    assert "AI-generated" not in ours and "padding row" in ours


def test_h2_moves_only_draws_no_probability():
    import numpy as np
    import render_common as rc
    from PIL import Image, ImageDraw

    def drawn(moves_only: bool, p: float):
        rc.MOVES_ONLY = moves_only
        img = Image.new("RGB", (300, 40), rc.BG)
        rc.bar(ImageDraw.Draw(img), 4, 4, 290, 26, p, "up", chosen=False)
        return np.asarray(img)

    try:
        # with probabilities shown, two different probabilities draw two different bars; moves only, the same
        assert (drawn(False, 0.2) != drawn(False, 0.9)).any()
        assert (drawn(True, 0.2) == drawn(True, 0.9)).all()
    finally:
        rc.MOVES_ONLY = False


def test_h3_the_scoreboard_card_is_deterministic_and_marks_the_clips_player():
    import render_common as rc

    rows = [["Decisio", "60%"], ["Jev 1.13 through its public API", "96%"]]
    a = rc.scoreboard("Pong", ["player", "agrees"], rows, 1280, 720, "note", "Decisio")
    b = rc.scoreboard("Pong", ["player", "agrees"], rows, 1280, 720, "note", "Decisio")
    c = rc.scoreboard("Pong", ["player", "agrees"], rows, 1280, 720, "note", None)
    assert rc.digest(a) == rc.digest(b) and rc.digest(a) != rc.digest(c)
