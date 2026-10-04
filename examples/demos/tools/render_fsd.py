# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Draw a driving clip from a trajectory file alone (`measure_fsd.py` writes one per drive).

The picture is 1280 x 720 before the caption: on the left a top-down view that follows the ego car with its heading
up (the streets, lanes, stop lines and signal heads from the header's map, each signal in the phase the tick recorded,
the traffic, the cyclists, the pedestrians, the parked cars and their open doors, the route ahead faintly, the ego car
in the accent colour); on the right the decision panel (the last applied decision's motion distribution and its top
manoeuvres as bars, the chosen option, the latency, a rules-fallback marker when the timeout replaced the answer, the
violation counters, and a short flash when a violation happens).

Frames are drawn at `render_common.FPS` over the recorded wall clock, so the clip runs at real speed: a lockstep drive,
whose simulation ran ahead of the clock between decisions, plays as fast as it ran. Between two recorded ticks the
positions are interpolated linearly; everything else shows the earlier tick.

    python render_fsd.py RUN/trajectories/run1_s1-1.jsonl.gz --out clip.mp4 [--gif clip.gif] [--start 30 --seconds 20]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import render_common as rc
import trajectory
from PIL import Image, ImageDraw

W, H = 1280, 720
MAP_W = 832  # the top-down view; the panel takes the rest
PANEL_X = MAP_W
SCALE = 7.0  # pixels per metre
SS = 2  # the view is drawn at twice the size and reduced, for smooth edges
EGO_Y = 0.66  # the ego car's height in the view, from the top: more road ahead than behind
FLASH_S = 1.6  # how long a violation stays flashed
JUMP_M = 25.0  # an object that moved further than this between two ticks was respawned: no interpolation
TOP_OPTIONS = 5

GROUND = (19, 21, 27)
BUILDING = (33, 36, 45)
BUILDING_EDGE = (45, 49, 60)
SIDEWALK = (44, 47, 57)
ASPHALT = (56, 59, 68)
DIVIDER = (128, 132, 144)
CENTRE = (196, 164, 64)
STOP_LINE = (215, 217, 222)
OBSTACLE = (40, 62, 46)
ROUTE = tuple(int(a + (b - a) * 0.38) for a, b in zip(ASPHALT, rc.ACCENT))
EGO = rc.ACCENT
EGO_EDGE = (225, 240, 255)
CAR = (170, 175, 188)
CAR_EDGE = (98, 102, 114)
BIKE = (110, 196, 150)
PARKED = (92, 97, 110)
PARKED_EDGE = (70, 74, 86)
PEDESTRIAN = (240, 196, 112)
DOOR = (226, 132, 70)
PHASE = {"green": (64, 200, 96), "yellow": (240, 200, 60), "red": (236, 72, 62)}
WARN = (236, 72, 62)
MUTED_LAMP = (90, 94, 104)
AMBER = (240, 176, 64)

VIOLATION_TEXT = {
    "collision": "COLLISION",
    "red_light": "RED LIGHT RUN",
    "stop_sign": "STOP SIGN ROLLED",
    "failed_to_yield": "FAILED TO YIELD",
    "off_road": "OFF ROAD",
}
COUNTERS = (
    ("collisions", "collisions", "collision"),
    ("red_lights_run", "red lights run", "red_light"),
    ("stop_signs_run", "stop signs rolled", "stop_sign"),
    ("failed_to_yield", "failures to yield", "failed_to_yield"),
    ("off_road_s", "seconds off road", "off_road"),
)


def offset_line(pts: np.ndarray, d: float) -> np.ndarray:
    """A polyline moved `d` metres to the right of its direction of travel (the map's convention)."""
    if len(pts) < 2:
        return pts.copy()
    tan = np.empty_like(pts)
    tan[1:-1] = pts[2:] - pts[:-2]
    tan[0] = pts[1] - pts[0]
    tan[-1] = pts[-1] - pts[-2]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    return pts + d * np.stack([tan[:, 1], -tan[:, 0]], axis=1)


def trim(pts: np.ndarray, start: float, end: float) -> np.ndarray | None:
    """The part of a polyline between `start` metres from its first point and `end` metres from its last."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    lo, hi = start, cum[-1] - end
    if hi - lo < 1.0:
        return None
    s = np.concatenate([[lo], cum[(cum > lo) & (cum < hi)], [hi]])
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def dashes(pts: np.ndarray, on: float = 3.0, off: float = 6.0) -> list[np.ndarray]:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out, s = [], 0.0
    while s < cum[-1]:
        e = min(s + on, cum[-1])
        u = np.concatenate([[s], cum[(cum > s) & (cum < e)], [e]])
        out.append(np.stack([np.interp(u, cum, pts[:, 0]), np.interp(u, cum, pts[:, 1])], axis=1))
        s += on + off
    return out


def box(x: float, y: float, psi: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    hl, hw = length / 2, width / 2
    return np.array([[x + c * a - s * b, y + s * a + c * b] for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


class Layer:
    """Shapes of one kind with their bounding boxes, so a frame draws only those in view."""

    def __init__(self, shapes: list, extra: list | None = None):
        self.shapes = shapes
        self.extra = extra if extra is not None else [None] * len(shapes)
        if shapes:
            self.bounds = np.array([[s[:, 0].min(), s[:, 1].min(), s[:, 0].max(), s[:, 1].max()] for s in shapes])
        else:
            self.bounds = np.zeros((0, 4))

    def near(self, x: float, y: float, r: float):
        b = self.bounds
        hit = (b[:, 0] < x + r) & (b[:, 2] > x - r) & (b[:, 1] < y + r) & (b[:, 3] > y - r)
        return [(self.shapes[i], self.extra[i]) for i in np.flatnonzero(hit)]


class Scene:
    """The header's map turned into drawable layers, once per trajectory."""

    def __init__(self, head: dict):
        m = head["map"]
        nodes = {j["node"]: j for j in m.get("junctions", [])}
        asphalt, walks, centre, dividers = [], [], [], []
        for e in m["edges"]:
            pts = np.array(e["pts"], dtype=float)
            if len(pts) < 2:
                continue
            lo, hi = e["surface"]
            asphalt.append(np.concatenate([offset_line(pts, lo), offset_line(pts, hi)[::-1]]))
            for side in (1, -1):
                edge = hi if side > 0 else lo
                w0 = edge + side * (e.get("curb", 0.22) + e.get("boulevard", 1.0))
                w1 = w0 + side * e.get("sidewalk", 1.8)
                walks.append(np.concatenate([offset_line(pts, w0), offset_line(pts, w1)[::-1]]))
            # markings stay out of the junctions and stop short of the stop line
            start = nodes[e["from"]]["r"] + 0.5 if e["from"] in nodes else 0.0
            end = nodes[e["to"]]["r"] + 0.5 if e["to"] in nodes else 0.0
            if e.get("control"):
                length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                end = max(end, length - e["control"]["s_line"] + 0.6)
            if not e["oneway"] and not e.get("ring"):
                line = trim(offset_line(pts, -0.12), start, end)
                if line is not None:
                    centre.append(line)
            offs = e.get("lane_offsets") or []
            for a, b in zip(offs, offs[1:]):
                line = trim(offset_line(pts, (a + b) / 2), start, end)
                if line is not None:
                    dividers += dashes(line)
        discs = [self.circle(j["x"], j["y"], j["r"]) for j in m.get("junctions", [])]
        for r in m.get("roundabouts", []):
            discs.append(self.circle(r["x"], r["y"], r["outer_r"] + m.get("gutter", 0.3)))
        self.islands = Layer([self.circle(r["x"], r["y"], r["island_r"]) for r in m.get("roundabouts", [])])
        self.asphalt = Layer(asphalt + discs)
        self.walks = Layer(walks)
        self.centre = Layer(centre)
        self.dividers = Layer(dividers)
        self.buildings = Layer([np.array(b, dtype=float) for b in m.get("buildings", []) if len(b) >= 3])
        obstacles = m.get("obstacles", {})
        circles = [self.circle(c["x"], c["y"], max(c["r"], 0.25), 8) for c in obstacles.get("circles", [])]
        polys = [np.array(p["pts"], dtype=float) for p in obstacles.get("polygons", []) if len(p["pts"]) >= 3]
        self.obstacles = Layer(circles + polys)
        signal_lines = {(s["edge"], s["intersection"]) for s in m.get("signals", [])}
        lines = [s for s in m.get("stop_lines", []) if (s["edge"], s["control"]) not in signal_lines]
        self.stop_lines = Layer([np.array(s["line"], dtype=float) for s in lines], lines)
        self.signals = Layer(
            [np.array(s["line"] + [s["at"]], dtype=float) for s in m.get("signals", [])], m.get("signals", [])
        )
        self.destination = m.get("destination")

    @staticmethod
    def circle(x: float, y: float, r: float, n: int = 20) -> np.ndarray:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)


class View:
    """World metres to view pixels: the ego car at (MAP_W / 2, EGO_Y * H), its heading pointing up."""

    def __init__(self, x: float, y: float, psi: float, ss: int = 1):
        self.x, self.y, self.c, self.s, self.k = x, y, math.cos(psi), math.sin(psi), SCALE * ss
        self.cx, self.cy = MAP_W / 2 * ss, EGO_Y * H * ss

    def px(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        dx, dy = p[:, 0] - self.x, p[:, 1] - self.y
        ahead = dx * self.c + dy * self.s
        right = dx * self.s - dy * self.c
        return np.stack([self.cx + right * self.k, self.cy - ahead * self.k], axis=1)

    def xy(self, pts) -> list[tuple[float, float]]:
        return [tuple(p) for p in self.px(pts).tolist()]


def view_radius() -> float:
    """Metres from the ego car to the furthest corner of the view, plus a margin."""
    return math.hypot(MAP_W / 2, max(EGO_Y, 1 - EGO_Y) * H) / SCALE + 10


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lerp_pose(a: dict, b: dict | None, u: float) -> dict:
    if b is None or u <= 0 or math.hypot(b["x"] - a["x"], b["y"] - a["y"]) > JUMP_M:
        return a
    return {
        **a,
        "x": a["x"] + (b["x"] - a["x"]) * u,
        "y": a["y"] + (b["y"] - a["y"]) * u,
        "psi": a["psi"] + wrap(b["psi"] - a["psi"]) * u,
    }


def lerp_list(a: list[dict], b: list[dict], u: float) -> list[dict]:
    later = {o["id"]: o for o in b}
    return [lerp_pose(o, later.get(o["id"]), u) for o in a]


class FsdRenderer:
    """A pure function of a trajectory and a wall-clock time: `frame(t_wall) -> PIL.Image` (W x H)."""

    def __init__(self, path: str | Path):
        self.head, self.ticks, self.end = trajectory.read(path)
        if not self.ticks:
            raise ValueError(f"{path}: no ticks")
        self.scene = Scene(self.head)
        self.times = [t["t_wall"] for t in self.ticks]
        self.t0, self.t1 = self.times[0], self.times[-1]
        # per tick: the last applied decision so far, the route in force, the decisions and fallbacks so far
        self.last_decision, self.route_at, self.counts = [], [], []
        routes = [np.array(self.head["map"]["route"], dtype=float)]
        last, n_model, n_fallback = None, 0, 0
        for i, t in enumerate(self.ticks):
            if t.get("route"):
                routes.append(np.array(t["route"], dtype=float))
            d = t.get("decision")
            if d:
                last = i
                n_model += "answer" in d
                n_fallback += bool(d.get("rules_fallback"))
            self.last_decision.append(last)
            self.route_at.append(len(routes) - 1)
            self.counts.append((n_model, n_fallback))
        self.routes = routes
        self.violations = [
            (e["t_wall"], e["type"], e.get("t_sim"))
            for t in self.ticks
            for e in t.get("events", [])
            if e.get("violation")
        ]
        run = self.head.get("run", {})
        self.title = " · ".join(
            str(x) for x in (run.get("scenario_id"), run.get("mode"), run.get("map_label") or run.get("map")) if x
        )

    def caption(self) -> str:
        return rc.caption_text(self.head.get("player", {}), self.ticks, self.head.get("run", {}).get("scenario_id", ""))

    def state_at(self, t: float) -> tuple[int, dict]:
        """The index of the last tick at or before `t`, and the state drawn at `t` (positions interpolated)."""
        i = rc.at_or_before(self.times, t)
        a = self.ticks[i]
        if t <= self.times[i] or i + 1 >= len(self.ticks):
            return i, a
        b = self.ticks[i + 1]
        u = (t - self.times[i]) / max(self.times[i + 1] - self.times[i], 1e-9)
        u = min(max(u, 0.0), 1.0)
        state = dict(a)
        state["ego"] = lerp_pose(a["ego"], b["ego"], u)
        for key in ("npcs", "pedestrians", "parked", "doors"):
            state[key] = lerp_list(a.get(key, []), b.get(key, []), u)
        return i, state

    def view_of(self, state: dict, ss: int = 1) -> View:
        e = state["ego"]
        return View(e["x"], e["y"], e["psi"], ss)

    # --- the top-down view ---------------------------------------------------------------------------------

    def draw_map(self, i: int, state: dict, t: float) -> Image.Image:
        img = Image.new("RGB", (MAP_W * SS, H * SS), GROUND)
        d = ImageDraw.Draw(img)
        ego = state["ego"]
        v = self.view_of(state, SS)
        r = view_radius()
        x, y = ego["x"], ego["y"]
        for shape, _ in self.scene.buildings.near(x, y, r):
            d.polygon(v.xy(shape), fill=BUILDING, outline=BUILDING_EDGE)
        for shape, _ in self.scene.walks.near(x, y, r):
            d.polygon(v.xy(shape), fill=SIDEWALK)
        for shape, _ in self.scene.obstacles.near(x, y, r):
            d.polygon(v.xy(shape), fill=OBSTACLE)
        for shape, _ in self.scene.asphalt.near(x, y, r):
            d.polygon(v.xy(shape), fill=ASPHALT)
        for shape, _ in self.scene.islands.near(x, y, r):
            d.polygon(v.xy(shape), fill=SIDEWALK)
        self.draw_route(d, v, i, ego)
        for shape, _ in self.scene.centre.near(x, y, r):
            d.line(v.xy(shape), fill=CENTRE, width=3)
        for shape, _ in self.scene.dividers.near(x, y, r):
            d.line(v.xy(shape), fill=DIVIDER, width=3)
        for shape, line in self.scene.stop_lines.near(x, y, r):
            if line["type"] == "yield":
                for dash in dashes(shape, 0.6, 0.6):
                    d.line(v.xy(dash), fill=STOP_LINE, width=int(0.5 * SCALE * SS))
            else:
                d.line(v.xy(shape), fill=STOP_LINE, width=int(0.5 * SCALE * SS))
        phases = self.ticks[i].get("signals", {})
        for shape, sig in self.scene.signals.near(x, y, r):
            colour = PHASE.get(phases.get(sig["intersection"], {}).get(sig["group"]), MUTED_LAMP)
            d.line(v.xy(shape[:2]), fill=colour, width=int(0.6 * SCALE * SS))
            (cx, cy), lamp = v.px(shape[2])[0], 1.0 * SCALE * SS
            d.ellipse([cx - lamp, cy - lamp, cx + lamp, cy + lamp], fill=colour, outline=GROUND, width=SS)
        if self.scene.destination:
            (cx, cy), rr = v.px(self.scene.destination)[0], 2.2 * SCALE * SS
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=rc.ACCENT, width=3 * SS)
            d.ellipse([cx - rr / 3, cy - rr / 3, cx + rr / 3, cy + rr / 3], fill=rc.ACCENT)
        for o in state.get("parked", []):
            d.polygon(v.xy(box(o["x"], o["y"], o["psi"], o["length"], o["width"])), fill=PARKED, outline=PARKED_EDGE)
        for o in state.get("doors", []):
            d.line(v.xy(box(o["x"], o["y"], o["psi"], o["length"], 0)[:2]), fill=DOOR, width=int(0.35 * SCALE * SS))
        for o in state.get("npcs", []):
            self.draw_vehicle(d, v, o, BIKE if o.get("kind") == "bike" else CAR, CAR_EDGE)
        pr = max(0.45 * SCALE, 3.0) * SS
        for p in state.get("pedestrians", []):
            (cx, cy) = v.px((p["x"], p["y"]))[0]
            if -pr < cx < MAP_W * SS + pr and -pr < cy < H * SS + pr:
                d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=PEDESTRIAN, outline=GROUND, width=SS)
        self.draw_vehicle(d, v, ego, EGO, EGO_EDGE)
        flash = self.flash(t)
        if flash:
            (cx, cy), rr = v.px((ego["x"], ego["y"]))[0], 5.0 * SCALE * SS
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=WARN, width=3 * SS)
        img = img.reduce(SS)
        if flash:
            self.draw_banner(img, flash)
        self.draw_scale(img)
        return img

    def draw_route(self, d: ImageDraw.ImageDraw, v: View, i: int, ego: dict) -> None:
        route = self.routes[self.route_at[i]]
        if len(route) < 2:
            return
        k = int(np.argmin(np.hypot(route[:, 0] - ego["x"], route[:, 1] - ego["y"])))
        ahead = route[k : k + int(view_radius() * 1.5)]
        if len(ahead) >= 2:
            d.line(v.xy(ahead), fill=ROUTE, width=int(1.6 * SCALE * SS), joint="curve")

    @staticmethod
    def draw_vehicle(d: ImageDraw.ImageDraw, v: View, o: dict, fill, edge) -> None:
        corners = box(o["x"], o["y"], o["psi"], o["length"], o["width"])
        d.polygon(v.xy(corners), fill=fill, outline=edge, width=SS)
        # the windscreen: a bar across the front third, so the heading reads at a glance
        c, s = math.cos(o["psi"]), math.sin(o["psi"])
        f = o["length"] * 0.18
        hw = o["width"] * 0.36
        a = (o["x"] + c * f - s * hw, o["y"] + s * f + c * hw)
        b = (o["x"] + c * f + s * hw, o["y"] + s * f - c * hw)
        d.line(v.xy([a, b]), fill=edge, width=max(SS, int(0.3 * SCALE * SS)))

    def flash(self, t: float) -> list[tuple[float, str, float]]:
        return [e for e in self.violations if 0 <= t - e[0] < FLASH_S]

    @staticmethod
    def draw_banner(img: Image.Image, flash: list) -> None:
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, MAP_W - 1, H - 1], outline=WARN, width=5)
        names = []
        for _, kind, _ in flash:
            name = VIOLATION_TEXT.get(kind, kind.upper())
            if name not in names:
                names.append(name)
        text = " + ".join(names)
        f = rc.font(22)
        w = int(f.getlength(text)) + 36
        x0 = (MAP_W - w) // 2
        d.rounded_rectangle([x0, 18, x0 + w, 58], radius=8, fill=WARN)
        d.text((MAP_W // 2, 38), text, fill=(255, 255, 255), font=f, anchor="mm")

    @staticmethod
    def draw_scale(img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        x0, y0, n = 20, H - 24, 20
        x1 = x0 + int(n * SCALE)
        d.line([(x0, y0), (x1, y0)], fill=rc.MUTED, width=2)
        for x in (x0, x1):
            d.line([(x, y0 - 5), (x, y0 + 5)], fill=rc.MUTED, width=2)
        d.text((x1 + 8, y0), f"{n} m", fill=rc.MUTED, font=rc.font(13), anchor="lm")
        # the map's licence (ODbL) asks for its attribution on anything drawn from it
        d.text((MAP_W - 12, y0), "Map data © OpenStreetMap contributors", fill=rc.MUTED, font=rc.font(12), anchor="rm")

    # --- the decision panel --------------------------------------------------------------------------------

    def draw_panel(self, img: Image.Image, i: int, state: dict, t: float) -> None:
        d = ImageDraw.Draw(img)
        x0, w = PANEL_X + 24, W - PANEL_X - 48
        x1 = x0 + w
        d.line([(PANEL_X, 0), (PANEL_X, H)], fill=rc.RULE, width=1)
        tick = self.ticks[i]
        ego = tick["ego"]
        d.text((x0, 26), self.fit(self.title, 15, w), fill=rc.MUTED, font=rc.font(15), anchor="lm")
        d.text((x0, 66), f"{abs(ego['v']) * 3.6:3.0f} km/h", fill=rc.FG, font=rc.font(30), anchor="lm")
        d.text((x1, 58), f"sim {tick['t_sim']:6.1f} s", fill=rc.MUTED, font=rc.font(15), anchor="rm")
        limit = f"limit {ego['limit'] * 3.6:.0f}" if ego.get("limit") else ""
        d.text((x1, 78), limit, fill=rc.MUTED, font=rc.font(15), anchor="rm")
        d.line([(x0, 100), (x1, 100)], fill=rc.RULE)

        k = self.last_decision[i]
        dec = self.ticks[k]["decision"] if k is not None else None
        y = 122
        d.text((x0, y), "MOTION", fill=rc.MUTED, font=rc.font(13), anchor="lm")
        if ego.get("in_flight"):  # the next request is on its way
            d.rounded_rectangle([x1 - 150, y - 10, x1, y + 10], radius=5, outline=rc.ACCENT, width=1)
            d.text((x1 - 75, y), "request in flight", fill=rc.ACCENT, font=rc.font(13), anchor="mm")
        y += 18
        y = self.draw_question(d, x0, y, w, dec, "motion", 2)
        y += 12
        options = len(((dec or {}).get("answer") or {}).get("vector", {}).get("probabilities", {}))
        head = f"MANOEUVRE  top {min(options, TOP_OPTIONS)} of {options}" if options else "MANOEUVRE"
        d.text((x0, y), head, fill=rc.MUTED, font=rc.font(13), anchor="lm")
        y += 18
        y = self.draw_question(d, x0, y, w, dec, "vector", TOP_OPTIONS)
        y += 6
        d.line([(x0, y), (x1, y)], fill=rc.RULE)
        y += 22
        if dec is None:
            d.text((x0, y), "no decision yet", fill=rc.MUTED, font=rc.font(15), anchor="lm")
        else:
            chosen = dec.get("chosen", {})
            text = f"chosen  {chosen.get('motion', '?')} / {chosen.get('maneuver') or 'none'}"
            d.text((x0, y), self.fit(text, 16, w), fill=rc.FG, font=rc.font(16), anchor="lm")
            y += 28
            age = t - self.ticks[k]["t_wall"]
            lat = dec.get("latency_ms")
            word = "latency" if "answer" in dec else "waited"  # a failed call: how long the client waited
            lat_text = f"{word} {lat:.0f} ms" if lat is not None else f"{word} n/a"
            d.text((x0, y), lat_text, fill=rc.FG, font=rc.font(16), anchor="lm")
            d.text((x1, y), f"applied {max(age, 0):.1f} s ago", fill=rc.MUTED, font=rc.font(14), anchor="rm")
            y += 30
            if dec.get("rules_fallback"):
                label = "RULES FALLBACK"
                why = dec.get("error") or ""
                d.rounded_rectangle([x0, y - 14, x1, y + 14], radius=6, fill=AMBER)
                d.text((x0 + 10, y), label, fill=rc.BG, font=rc.font(15), anchor="lm")
                d.text((x1 - 10, y), self.fit(why, 13, w - 190), fill=rc.BG, font=rc.font(13), anchor="rm")
            elif dec.get("fallback"):
                d.text(
                    (x0, y),
                    self.fit(f"manoeuvre by rules: {dec['fallback']}", 14, w),
                    fill=AMBER,
                    font=rc.font(14),
                    anchor="lm",
                )
            else:
                d.text((x0, y), "model answer", fill=rc.MUTED, font=rc.font(14), anchor="lm")
        y = 548
        d.line([(x0, y - 22), (x1, y - 22)], fill=rc.RULE)
        d.text((x0, y), "VIOLATIONS", fill=rc.MUTED, font=rc.font(13), anchor="lm")
        flashing = {kind for _, kind, _ in self.flash(t)}
        counters = tick.get("violations", {})
        for key, label, kind in COUNTERS:
            y += 24
            value = counters.get(key, 0) or 0
            hot = kind in flashing
            if hot:
                d.rectangle([x0 - 8, y - 11, x1 + 8, y + 11], fill=WARN)
            colour = (255, 255, 255) if hot else (WARN if value else rc.FG)
            d.text((x0, y), label, fill=colour, font=rc.font(15), anchor="lm")
            shown = f"{value:.1f}" if key == "off_road_s" else f"{value}"
            d.text((x1, y), shown, fill=colour, font=rc.font(15), anchor="rm")
        n_model, n_fallback = self.counts[i]
        d.text(
            (x0, H - 22),
            f"{n_model} model decisions · {n_fallback} rules fallbacks",
            fill=rc.MUTED,
            font=rc.font(13),
            anchor="lm",
        )

    def draw_question(self, d: ImageDraw.ImageDraw, x0: int, y: int, w: int, dec: dict | None, q: str, top: int) -> int:
        """The question's options as bars, most likely first; the row height is 32 px."""
        answer = ((dec or {}).get("answer") or {}).get(q)
        if not answer or not answer.get("probabilities"):
            local = ((dec or {}).get("local_answers") or {}).get(q)
            if dec and dec.get("rules_fallback"):
                text = "no answer: the rules driver decided"
            elif dec and dec.get("error"):
                text = "no answer in the response"
            elif local:
                text = f"not asked: {local.get('choice')} (only option)"
            else:
                text = "not asked" if dec else ""
            d.text((x0, y + 13), text, fill=rc.MUTED, font=rc.font(14), anchor="lm")
            return y + 32 * top
        chosen = (dec.get("chosen") or {}).get("motion" if q == "motion" else "maneuver")
        probs = sorted(answer["probabilities"].items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        for key, p in probs:
            label = self.fit(key, 14, w - 70)
            rc.bar(d, x0, y, w, 26, p, label, key == chosen, size=14)
            y += 32
        return y + 32 * (top - len(probs))

    @staticmethod
    def fit(text: str, size: int, width: int) -> str:
        f = rc.font(size)
        if f.getlength(text) <= width:
            return text
        while text and f.getlength(text + "...") > width:
            text = text[:-1]
        return text + "..."

    def frame(self, t: float) -> Image.Image:
        i, state = self.state_at(t)
        img = Image.new("RGB", (W, H), rc.BG)
        img.paste(self.draw_map(i, state, t), (0, 0))
        self.draw_panel(img, i, state, t)
        return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory")
    ap.add_argument("--out", required=True, help="the MP4 to write")
    ap.add_argument("--gif", default=None, help="also a GIF (kept small by demo_recorder.to_gif)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds after the first tick")
    ap.add_argument("--seconds", type=float, default=None, help="clip length (default: to the last tick)")
    a = ap.parse_args()
    r = FsdRenderer(a.trajectory)
    t0 = r.t0 + a.start
    t1 = r.t1 if a.seconds is None else min(r.t1, t0 + a.seconds)
    if t1 <= t0:
        raise SystemExit(f"nothing to draw: the trajectory spans {r.t1 - r.t0:.1f} s of wall clock")
    mp4 = rc.encode(rc.render(r.frame, rc.frame_times(t0, t1), r.caption()), a.out)
    print(f"{mp4}: {t1 - t0:.1f} s at {rc.FPS} fps")
    if a.gif:
        from demo_recorder import to_gif

        print(to_gif(mp4, a.gif, max_s=t1 - t0))


if __name__ == "__main__":
    main()
