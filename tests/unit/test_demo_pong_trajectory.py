# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Pong trajectories and their renderer (examples/demos/tools/trajectory.py, render_pong.py).

The trajectory under tests/data was recorded by measure_pong.py against tools/mock_systemone.py (8 s, seed 20260917).
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "examples/demos/tools"
sys.path.insert(0, str(TOOLS))
TRAJ = ROOT / "tests/data/pong_mock_trajectory.jsonl.gz"

pytest.importorskip("PIL")


def test_p1_the_court_in_the_header_is_the_engine_court():
    import measure_pong

    types = (ROOT / "examples/demos/pong/lib/game/types.ts").read_text()
    const = {k: float(v) for k, v in re.findall(r"export const (\w+) = ([\d.]+);", types)}
    c = measure_pong.COURT
    assert (c["w"], c["h"], c["paddle_h"], c["ball_r"]) == (
        const["COURT_W"],
        const["COURT_H"],
        const["PADDLE_H"],
        const["BALL_R"],
    )
    assert (c["paddle_inset"], c["paddle_step"], c["points_to_win"]) == (
        const["PADDLE_INSET"],
        const["MODEL_PADDLE_STEP"],
        const["POINTS_TO_WIN"],
    )


def test_p2_a_trajectory_round_trips_and_carries_every_decision(tmp_path):
    import trajectory

    head, ticks, end = trajectory.read(TRAJ)
    assert head["format"] == trajectory.FORMAT and head["demo"] == "pong" and head["player"]["card"]
    decided = trajectory.decisions(ticks)
    assert len(decided) == end["decisions"] and end["status"] == "over"
    d = decided[0]["decision"]
    assert set(d["answer"]["move"]["probabilities"]) == {"up", "down", "stay"} and d["chosen"] in ("up", "down", "stay")
    assert d["request"]["questions"]["move"]["type"] == "choice"
    copy = tmp_path / "copy.jsonl.gz"
    trajectory.write(copy, head, ticks, end)
    assert trajectory.read(copy) == (head, ticks, end)
    # the same content gives the same bytes
    assert trajectory.write(tmp_path / "again.jsonl.gz", head, ticks, end) == trajectory.file_sha256(copy)


def test_p3_rendering_a_trajectory_twice_gives_identical_frames():
    import render_common as rc
    import render_pong

    first = [rc.digest(f) for f in render_pong.frames([render_pong.PongTrajectory(TRAJ)], 0, 3, fps=2)]
    second = [rc.digest(f) for f in render_pong.frames([render_pong.PongTrajectory(TRAJ)], 0, 3, fps=2)]
    assert first == second and len(set(first)) > 1


def test_p4_a_frame_drawn_at_a_tick_shows_that_ticks_recorded_state():
    """Find the ball and the model's paddle in the pixels of the frame drawn at each tick's time and map them back to
    court units: they must be where the trajectory says, to within half a court unit."""
    import numpy as np
    import render_pong

    lane = render_pong.PongTrajectory(TRAJ)
    s, ox, oy, _ = lane.geometry(render_pong.W, render_pong.H)
    checked = 0
    for tick in lane.ticks[::7]:
        st = tick["state"]
        img = np.asarray(lane.frame(tick["t_wall"]))
        court = img[oy : int(oy + lane.court["h"] * s), ox : int(ox + lane.court["w"] * s)]
        ys, xs = np.nonzero(np.all(court == render_pong.BALL, axis=-1))
        assert len(xs), f"no ball drawn at tick {tick['tick']}"
        assert abs(xs.mean() / s - st["ball"]["x"]) < 0.5 and abs(ys.mean() / s - st["ball"]["y"]) < 0.5
        py, _px = np.nonzero(np.all(court == render_pong.MODEL_PADDLE, axis=-1))
        assert abs((py.min() + py.max()) / 2 / s - st["rightY"]) < 0.5
        checked += 1
    assert checked >= 10
