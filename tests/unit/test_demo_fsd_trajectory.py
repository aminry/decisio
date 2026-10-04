# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The driving demo's trajectory files and their renderer (examples/demos/tools/render_fsd.py), on a synthetic drive
that needs no browser and no server.

F1  rendering the same trajectory twice gives the same frames
F2  a frame drawn at a tick's wall clock puts the ego car (and a traffic car) where the tick's state says
F3  trajectory.read gives back what trajectory.write wrote, and the same content gives the same bytes
F4  a violation flashes on the map from the moment it happened, and a rules fallback is marked in the panel
F5  between two ticks the positions are interpolated linearly, and nothing else changes before the next tick
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "demos" / "tools"))

import render_common as rc  # noqa: E402
import render_fsd  # noqa: E402
import trajectory  # noqa: E402

T0 = 1_790_000_000.0
PSI = 0.35  # the drive heads a little north of east, so the view is rotated


def answer(probs: dict) -> dict:
    choice = max(probs, key=probs.get)
    return {"type": "choice", "choice": choice, "confidence": probs[choice], "probabilities": probs}


def synthetic(path: Path) -> tuple[dict, list[dict], dict]:
    """A 3 s drive along one street through a signalled junction: a traffic car, a pedestrian, a parked car, a
    model decision, a timeout fallback and a red light run."""
    c, s = math.cos(PSI), math.sin(PSI)

    def along(d: float, lat: float = 0.0) -> list[float]:
        return [round(d * c + lat * s, 2), round(d * s - lat * c, 2)]

    edge = {
        "id": "e0",
        "from": "n0",
        "to": "n1",
        "pts": [along(-150), along(0), along(150)],
        "oneway": False,
        "lanes": 1,
        "lane_width": 3.5,
        "lane_offsets": [1.75],
        "asphalt": [-3.5, 3.5],
        "surface": [-3.8, 3.8],
        "curb": 0.22,
        "boulevard": 1.0,
        "sidewalk": 1.8,
        "control": {"type": "signal", "id": "sig0", "group": "A", "s_line": 170.0},
    }
    head = trajectory.header(
        "fsd",
        trajectory.player_fields({"label": "Decisio"}, card="test card"),
        {"suite_seed": 1, "scenario": 1, "scenario_id": "s1-1", "mode": "realtime", "map_label": "test grid"},
        map={
            "name": "test",
            "gutter": 0.3,
            "edges": [edge],
            "junctions": [{"node": "n1", "x": along(150)[0], "y": along(150)[1], "r": 4.0}],
            "stop_lines": [
                {
                    "edge": "e0",
                    "type": "signal",
                    "control": "sig0",
                    "group": "A",
                    "line": [along(20, 0.1), along(20, 3.7)],
                }
            ],
            "signals": [
                {
                    "intersection": "sig0",
                    "group": "A",
                    "edge": "e0",
                    "line": [along(20, 0.1), along(20, 3.7)],
                    "at": along(20.8, 4.5),
                }
            ],
            "buildings": [[along(-20, 12), along(10, 12), along(10, 25), along(-20, 25)]],
            "obstacles": {
                "circles": [{"kind": "trunk", "x": along(5, 6)[0], "y": along(5, 6)[1], "r": 0.3}],
                "polygons": [],
            },
            "route": [along(d, 1.75) for d in range(-10, 120)],
            "destination": along(119, 1.75),
        },
    )
    request = {
        "state": {"car": {"speed": 8}},
        "questions": {"motion": {"type": "choice", "criteria": {"drive": "", "stop": ""}}},
    }
    ticks = []
    for k in range(31):
        x, y = along(-5 + 0.8 * k, 1.75)
        tick = {
            "tick": 6 * k,
            "t_wall": round(T0 + 0.1 * k, 4),
            "t_sim": round(0.1 * k, 3),
            "ego": {
                "x": x,
                "y": y,
                "psi": PSI,
                "v": 8.0,
                "a": 0.0,
                "steer": 0.0,
                "length": 4.5,
                "width": 1.9,
                "limit": 13.9,
                "in_flight": k % 4 == 1,
                "on_road": True,
            },
            "npcs": [
                {
                    "id": "car_1",
                    "kind": "car",
                    "x": along(30 - 0.5 * k, -1.75)[0],
                    "y": along(30 - 0.5 * k, -1.75)[1],
                    "psi": PSI + math.pi,
                    "v": 5.0,
                    "length": 4.5,
                    "width": 1.9,
                }
            ],
            "pedestrians": [
                {"id": "ped_1", "x": along(12, 6.5)[0], "y": along(12, 6.5)[1], "psi": 0.0, "v": 1.2, "crossing": False}
            ],
            "parked": [
                {
                    "id": "parked_1",
                    "kind": "car",
                    "x": along(-12, 4.6)[0],
                    "y": along(-12, 4.6)[1],
                    "psi": PSI,
                    "v": 0.0,
                    "length": 4.5,
                    "width": 1.9,
                }
            ],
            "doors": [],
            "signals": {"sig0": {"A": "red" if k >= 10 else "green", "B": "green" if k >= 10 else "red"}},
            "violations": {
                "collisions": 0,
                "red_lights_run": int(k >= 20),
                "stop_signs_run": 0,
                "failed_to_yield": 0,
                "off_road_s": 0,
            },
            "decision": None,
        }
        if k == 3:
            tick["decision"] = {
                "request": request,
                "answer": {
                    "motion": answer({"drive": 0.8, "stop": 0.2}),
                    "vector": answer({"keep_lane_target": 0.6, "keep_lane_cautious": 0.3, "hard_brake": 0.1}),
                },
                "chosen": {"motion": "drive", "maneuver": "keep_lane_target"},
                "latency_ms": 42.0,
                "rules_fallback": False,
                "source": "jev",
            }
        if k == 12:
            tick["decision"] = {
                "request": request,
                "error": "timeout after 1500 ms",
                "latency_ms": 1500.4,
                "chosen": {"motion": "drive", "maneuver": "keep_lane_cautious"},
                "rules_fallback": True,
                "fallback": "rules",
                "source": "rules_fallback",
            }
        if k == 20:
            tick["events"] = [
                {"type": "red_light", "violation": True, "t_sim": tick["t_sim"], "t_wall": tick["t_wall"]}
            ]
        ticks.append(tick)
    end = {
        "pass": False,
        "failures": ["red light"],
        "arrived": True,
        "failed_at": {"red light": {"t_sim": 2.0, "t_wall": T0 + 2.0}},
    }
    trajectory.write(path, head, ticks, end)
    return head, ticks, end


@pytest.fixture
def drive(tmp_path):
    path = tmp_path / "trajectory.jsonl.gz"
    head, ticks, end = synthetic(path)
    return path, head, ticks, end


def test_f1_rendering_twice_gives_identical_frames(drive):
    path = drive[0]
    times = [T0 + x for x in (0.0, 0.33, 1.2, 2.05, 2.9)]
    first = [rc.digest(render_fsd.FsdRenderer(path).frame(t)) for t in times]
    second = [rc.digest(render_fsd.FsdRenderer(path).frame(t)) for t in times]
    assert first == second
    assert len(set(first)) == len(first)  # the picture changes as the drive goes on


def test_f2_the_ego_car_is_drawn_where_the_tick_says(drive):
    path, _, ticks, _ = drive
    r = render_fsd.FsdRenderer(path)
    for tick in (ticks[0], ticks[7], ticks[25]):
        img = r.frame(tick["t_wall"])
        assert img.size == (render_fsd.W, render_fsd.H)
        view = r.view_of(tick)
        ego = tick["ego"]
        x, y = view.px((ego["x"], ego["y"]))[0]
        assert img.getpixel((round(x), round(y))) == render_fsd.EGO
        npc = tick["npcs"][0]
        x, y = view.px((npc["x"], npc["y"]))[0]
        assert 0 <= x < render_fsd.MAP_W and 0 <= y < render_fsd.H
        assert img.getpixel((round(x), round(y))) == render_fsd.CAR
    # the transform itself: heading up, the ego car at the view's anchor, right of travel to the right
    view = render_fsd.View(10.0, 5.0, PSI)
    (ax, ay), (fx, fy), (rx, ry) = view.px(
        [(10.0, 5.0), (10 + math.cos(PSI), 5 + math.sin(PSI)), (10 + math.sin(PSI), 5 - math.cos(PSI))]
    )
    assert (ax, ay) == pytest.approx((render_fsd.MAP_W / 2, render_fsd.EGO_Y * render_fsd.H))
    assert (fx - ax, fy - ay) == pytest.approx((0, -render_fsd.SCALE))
    assert (rx - ax, ry - ay) == pytest.approx((render_fsd.SCALE, 0))


def test_f3_read_round_trips_write(drive, tmp_path):
    path, head, ticks, end = drive
    assert trajectory.read(path) == (head, ticks, end)
    again = tmp_path / "again.jsonl.gz"
    assert trajectory.write(again, head, ticks, end) == trajectory.file_sha256(path)
    assert [t["tick"] for t in trajectory.decisions(ticks)] == [18]  # the fallback is not a model decision


def test_f4_a_violation_flashes_and_a_fallback_is_marked(drive):
    path, _, ticks, _ = drive
    r = render_fsd.FsdRenderer(path)
    corner = (2, render_fsd.H // 2)
    before, at, after = (r.frame(ticks[20]["t_wall"] + dt) for dt in (-0.05, 0.0, render_fsd.FLASH_S + 0.05))
    assert at.getpixel(corner) == render_fsd.WARN
    assert before.getpixel(corner) != render_fsd.WARN and after.getpixel(corner) != render_fsd.WARN
    fallback = r.frame(ticks[12]["t_wall"])
    panel = fallback.crop((render_fsd.PANEL_X, 0, render_fsd.W, render_fsd.H))
    assert render_fsd.AMBER in {c for _, c in panel.getcolors(1 << 16)}
    model = r.frame(ticks[5]["t_wall"]).crop((render_fsd.PANEL_X, 0, render_fsd.W, render_fsd.H))
    assert render_fsd.AMBER not in {c for _, c in model.getcolors(1 << 16)}


def test_f5_positions_are_interpolated_between_ticks(drive):
    path, _, ticks, _ = drive
    r = render_fsd.FsdRenderer(path)
    a, b = ticks[4], ticks[5]
    i, state = r.state_at((a["t_wall"] + b["t_wall"]) / 2)
    assert i == 4
    assert state["ego"]["x"] == pytest.approx((a["ego"]["x"] + b["ego"]["x"]) / 2)
    assert state["npcs"][0]["y"] == pytest.approx((a["npcs"][0]["y"] + b["npcs"][0]["y"]) / 2)
    assert state["signals"] == a["signals"] and state["violations"] == a["violations"]
    assert r.state_at(ticks[-1]["t_wall"] + 5)[1] == ticks[-1]
