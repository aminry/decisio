# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The browser agent's trajectory and its renderer, on a small synthetic trajectory (no browser, no server).

U1  rendering the same trajectory twice gives identical frames
U2  the chosen element's outline is drawn where its box says
U3  the frame shown at a step's recorded time is that step's screenshot; a minimum hold is applied and captioned
U4  a trajectory reads back as it was written
U5  the measurement's writer keeps each decision's screenshot, its box in screenshot pixels and the typed text
"""

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "examples" / "demos" / "tools"
sys.path.insert(0, str(TOOLS))

import measure_ultrafast  # noqa: E402
import render_common as rc  # noqa: E402
import render_ultrafast as ru  # noqa: E402
import trajectory  # noqa: E402

COLOURS = [(200, 40, 40), (40, 170, 60), (40, 60, 200)]  # one flat screenshot per tick
FRAME = (160, 100)
BBOX = [40.0, 30.0, 60.0, 40.0]
T0 = 1_000_000.0


def decision(op, target, text=None):
    options = ["CLICK", "TYPE_TEXT", "DONE"]
    probs = {o: (0.8 if o == op else 0.1) for o in options}
    answer = {"operation": {"type": "choice", "choice": op, "confidence": 0.8, "probabilities": probs}}
    questions = {"operation": {"type": "choice", "criteria": {o: o for o in options}}}
    if target:
        name = op.lower() + "_target"
        answer[name] = {"type": "choice", "choice": target, "confidence": 0.7, "probabilities": {"1": 0.7, "2": 0.3}}
        questions[name] = {"type": "choice", "criteria": {"1": {"element": "[1] Search"}, "2": {"element": "[2] Go"}}}
    return {
        "request": {"state": {"page": {"url": "http://fixture/"}}, "questions": questions},
        "answer": answer,
        "chosen": {"operation": op, "target": target, "action": "e1" if target else op, "label": "Search"},
        "text": text,
        "latency_ms": 12,
    }


def make_trajectory(tmp_path):
    (tmp_path / "frames").mkdir()
    names = ["step_001.png", "step_002.png", "final.png"]
    for name, colour in zip(names, COLOURS):
        Image.new("RGB", FRAME, colour).save(tmp_path / "frames" / name)
    ticks = []
    for k, name in enumerate(names):
        final = k == 2
        tick = {
            "tick": k,
            "t_wall": T0 + 0.5 * k,
            "step": None if final else k + 1,
            "url": "http://fixture/",
            "title": "Fixture",
            "frame": f"frames/{name}",
            "frame_sha256": trajectory.file_sha256(tmp_path / "frames" / name),
            "frame_size": list(FRAME),
            "highlight": {"bbox": BBOX, "index": "1", "label": "Search"} if k == 0 else None,
            "decision": None if final else decision(*(("TYPE_TEXT", "1", "Lisbon") if k == 0 else ("DONE", None))),
        }
        if final:
            tick["final"] = True
        ticks.append(tick)
    head = trajectory.header(
        "ultrafast",
        trajectory.player_fields({"label": "test model", "card": "a card"}),
        {"task": "travel", "goal": "Find a stay in Lisbon.", "index": 1},
    )
    end = {"completed": True, "status": "done", "verification": {"url_end": "#x", "url_ok": True}, "elapsed_ms": 1000}
    trajectory.write(tmp_path / trajectory.NAME, head, ticks, end)
    return head, ticks, end


def test_u1_rendering_twice_gives_identical_frames(tmp_path):
    make_trajectory(tmp_path)
    runs = []
    for _ in range(2):
        clip = ru.Clip(tmp_path, min_hold=0.3)
        runs.append([rc.digest(img) for img in rc.render(clip.frame, clip.times(), clip.caption())])
    assert runs[0] == runs[1] and len(set(runs[0])) == 3


def test_u2_the_outline_is_drawn_where_the_box_says(tmp_path):
    make_trajectory(tmp_path)
    img = ru.Clip(tmp_path).frame(0.0)
    s = min(ru.SHOT_BOX[2] / FRAME[0], ru.SHOT_BOX[3] / FRAME[1])
    x, y, w, h = BBOX
    x0, y0 = ru.SHOT_BOX[0] + round(x * s), ru.SHOT_BOX[1] + round(y * s)
    x1, y1 = ru.SHOT_BOX[0] + round((x + w) * s), ru.SHOT_BOX[1] + round((y + h) * s)
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    for edge in [(x0 - 1, my), (x1 + 1, my), (mx, y0 - 1), (mx, y1 + 1)]:
        assert img.getpixel(edge) == ru.HIGHLIGHT, edge
    assert img.getpixel((mx, my)) == COLOURS[0]  # the element itself is not covered
    assert img.getpixel((x0 - 8, my)) == COLOURS[0]  # nor the page beside it


def test_u3_each_step_shows_its_own_screenshot_at_its_time(tmp_path):
    _, ticks, _ = make_trajectory(tmp_path)
    clip = ru.Clip(tmp_path)
    s = min(ru.SHOT_BOX[2] / FRAME[0], ru.SHOT_BOX[3] / FRAME[1])
    region = (ru.SHOT_BOX[0] + 4, ru.SHOT_BOX[1] + round(80 * s), ru.SHOT_BOX[0] + 30, ru.SHOT_BOX[1] + round(95 * s))
    for k, tick in enumerate(ticks):
        t = tick["t_wall"] - ticks[0]["t_wall"]
        assert clip.start_of(k) == pytest.approx(t)  # real speed: the clip clock is the recorded clock
        for when in (t, t + 0.49 if k < 2 else t + ru.END_HOLD_S - 0.01):
            crop = clip.frame(when).crop(region)
            assert [c for _, c in crop.getcolors()] == [COLOURS[k]], (k, when)
    assert clip.duration == pytest.approx(1.0 + ru.END_HOLD_S)
    assert "real speed" in clip.caption() and "held" not in clip.caption()
    held = ru.Clip(tmp_path, min_hold=0.8)
    assert held.start_of(1) == pytest.approx(0.8) and held.start_of(2) == pytest.approx(1.6)
    assert "held for readability" in held.caption()
    assert "test model" in held.caption() and "median 12 ms" in held.caption()


def test_u4_a_trajectory_reads_back_as_written(tmp_path):
    head, ticks, end = make_trajectory(tmp_path)
    assert trajectory.read(tmp_path / trajectory.NAME) == (head, ticks, end)


def png_b64(size, colour):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_u5_the_writer_keeps_screenshot_box_and_typed_text(tmp_path):
    traj = measure_ultrafast.Trajectory(tmp_path / "t", {"label": "m"}, {"task": "travel", "goal": "g", "index": 1})
    action = {"id": "e1", "kind": "fill", "label": "Search", "node": 1, "rect": {"x": 10, "y": 5, "w": 20, "h": 8}}
    page = {
        "url": "http://fixture/",
        "title": "Fixture",
        "w": 100,  # CSS pixels; the screenshot is 200 wide: a device-pixel ratio of 2
        "h": 50,
        "scroll": {"y": 0},
        "actions": [action, {"id": "wait", "kind": "wait", "label": "Wait"}],
        "screenshot": png_b64((200, 100), COLOURS[0]),
        "screenshot_ms": 40.0,
    }
    choose = {
        "choice": "e1",
        "operation": "TYPE_TEXT",
        "target": "1",
        "request": {"state": {}, "questions": {}},
        "raw_answers": {"operation": {"choice": "TYPE_TEXT"}},
        "model": "mock",
        "usage": {},
        "skipped_single_option": [],
        "latency_ms": 9,
    }
    traj.on_decision(page, choose)
    history = [{"choice": "e1", "text": "Lisbon", "text_helper": "qwen", "text_latency_ms": 300, "page_changed": True}]
    traj.after_tick(history)
    traj.on_decision(page, {**choose, "choice": "wait", "operation": "WAIT", "target": None})
    traj.after_tick(history)  # the wait was not executed (no new history row)
    traj.final = {"t_wall": 1.0, "url": "u", "title": "t", "viewport": [100, 50], "screenshot": page["screenshot"]}
    traj.write({"completed": False, "status": "blocked"})
    head, ticks, end = trajectory.read(tmp_path / "t" / trajectory.NAME)
    assert head["run"]["viewport"]["device_pixel_ratio"] == 2.0 and end["status"] == "blocked"
    first, second, final = ticks
    assert first["highlight"] == {"bbox": [20.0, 10.0, 40.0, 16.0], "index": "1", "label": "Search"}
    assert first["decision"]["text"] == "Lisbon" and first["decision"]["text_helper"]["model"] == "qwen"
    assert first["decision"]["chosen"]["operation"] == "TYPE_TEXT" and first["executed"] is True
    assert second["highlight"] is None and second["executed"] is False and second["decision"]["text"] is None
    assert final["final"] and final["decision"] is None and final["frame"] == "frames/final.png"
    for tick in ticks:
        assert trajectory.file_sha256(tmp_path / "t" / tick["frame"]) == tick["frame_sha256"]
        assert tick["frame_size"] == [200, 100]
