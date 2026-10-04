# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Render Pong clips from trajectory files (tools/trajectory.py), at real speed, with the decision on screen.

Nothing is simulated: every frame shows a recorded tick's state. Between two ticks the ball is moved along the recorded
velocity, folded at the walls and paddle planes as the engine does, but only where that lands exactly on the next
recorded state; otherwise it is interpolated linearly between the two, and across a point (the next tick is a serve) it
is not interpolated at all. Paddles are interpolated linearly. Side-by-side lanes are drawn from separate trajectories,
each on its own clock from its first tick.

    python render_pong.py run1_decisio.jsonl.gz --out pong.mp4 [--gif pong.gif]
    python render_pong.py run1_decisio.jsonl.gz run1_cygnet.jsonl.gz --out side_by_side.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import render_common as rc
import trajectory
from PIL import Image, ImageDraw

W, H = 1280, 720
BALL = (250, 250, 250)
MODEL_PADDLE = rc.ACCENT
WALL_PADDLE = (120, 124, 136)
COURT_BG = (18, 20, 26)
MOVES = ("up", "down", "stay")


def fold(v: float, lo: float, hi: float) -> float:
    """Reflect `v` into [lo, hi], as the engine folds the ball at the walls."""
    span = hi - lo
    if span <= 0:
        return lo
    k = (v - lo) % (2 * span)
    return lo + (k if k <= span else 2 * span - k)


class PongTrajectory:
    def __init__(self, path: str | Path):
        self.head, self.ticks, self.end = trajectory.read(path)
        self.court = self.head["court"]
        self.times = [t["t_wall"] for t in self.ticks]
        self.t0, self.t1 = self.times[0], self.times[-1]
        self.decided = [i for i, t in enumerate(self.ticks) if t.get("decision") and "answer" in t["decision"]]

    def state_at(self, t: float) -> dict:
        """The drawn state at wall time `t`: the recorded tick at or before it, moved toward the next as described."""
        i = rc.at_or_before(self.times, t)
        a = self.ticks[i]["state"]
        if i + 1 >= len(self.ticks) or t <= self.times[i]:
            return a
        nxt = self.ticks[i + 1]
        b = nxt["state"]
        f = min(1.0, (t - self.times[i]) / max(1e-9, self.times[i + 1] - self.times[i]))
        if nxt["kind"] == "serve":
            return a
        c = self.court
        r, plane_l, plane_r = c["ball_r"], c["paddle_inset"], c["w"] - c["paddle_inset"]
        ball_a, ball_b = a["ball"], b["ball"]

        def moved(g: float) -> tuple[float, float]:
            x = ball_a["x"] + ball_a["vx"] * g
            if (ball_a["vx"] > 0) != (ball_b["vx"] > 0):
                x = fold(x, plane_l, plane_r)
            return x, fold(ball_a["y"] + ball_a["vy"] * g, r, c["h"] - r)

        ex, ey = moved(1.0)
        if abs(ex - ball_b["x"]) < 0.05 and abs(ey - ball_b["y"]) < 0.05:
            x, y = moved(f)
        else:
            x, y = ball_a["x"] + (ball_b["x"] - ball_a["x"]) * f, ball_a["y"] + (ball_b["y"] - ball_a["y"]) * f
        lerp = lambda p, q: p + (q - p) * f  # noqa: E731
        return {
            **a,
            "ball": {**ball_a, "x": x, "y": y},
            "leftY": lerp(a["leftY"], b["leftY"]),
            "rightY": lerp(a["rightY"], b["rightY"]),
        }

    def decision_at(self, t: float) -> dict | None:
        i = rc.at_or_before(self.times, t)
        last = [k for k in self.decided if k <= i]
        return self.ticks[last[-1]]["decision"] if last else None

    # ------------------------------------------------------------------ drawing
    def geometry(self, width: int, height: int) -> tuple[float, int, int, bool]:
        """Pixels per court unit, the court's top-left corner, and whether the decision panel sits beside the court
        (a wide picture) or under it (a lane of a side-by-side clip)."""
        c = self.court
        beside = width >= 1000
        if beside:
            s = min((width * 0.74 - 24) / c["w"], (height - 90) / c["h"])
        else:
            s = min((width - 32) / c["w"], (height - 64 - 170) / c["h"])
        return s, 16, 64, beside

    def to_px(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        s, ox, oy, _ = self.geometry(width, height)
        return ox + x * s, oy + y * s

    def frame(self, t: float, width: int = W, height: int = H) -> Image.Image:
        c = self.court
        st = self.state_at(t)
        s, ox, oy, beside = self.geometry(width, height)
        img = Image.new("RGB", (width, height), rc.BG)
        d = ImageDraw.Draw(img)
        player = self.head["player"]
        title = f"Pong · {player.get('label') or 'model'}"
        size = 20
        while size > 13 and rc.font(size).getlength(title) > width - 32:
            size -= 1
        d.text((ox, 22), title, fill=rc.FG, font=rc.font(size), anchor="lm")
        run = self.head["run"]
        variant = run.get("prompt_variant", "default")
        d.text(
            (ox, 46),
            f"seed {run.get('seed')} · {variant} prompt · the model plays the right paddle",
            fill=rc.MUTED,
            font=rc.font(13),
            anchor="lm",
        )
        cw, ch = c["w"] * s, c["h"] * s
        d.rectangle([ox, oy, ox + cw, oy + ch], fill=COURT_BG, outline=rc.RULE)
        for k in range(0, int(c["h"]), 6):  # the centre line
            d.line([ox + cw / 2, oy + k * s, ox + cw / 2, oy + (k + 3) * s], fill=rc.RULE, width=2)
        ph, inset = c["paddle_h"], c["paddle_inset"]
        for py, x0, colour in ((st["leftY"], inset - 2, WALL_PADDLE), (st["rightY"], c["w"] - inset, MODEL_PADDLE)):
            d.rectangle(
                [ox + x0 * s, oy + (py - ph / 2) * s, ox + (x0 + 2) * s - 1, oy + (py + ph / 2) * s], fill=colour
            )
        bx, by = ox + st["ball"]["x"] * s, oy + st["ball"]["y"] * s
        rr = c["ball_r"] * s
        d.ellipse([bx - rr, by - rr, bx + rr, by + rr], fill=BALL)
        score = st["score"]
        d.text((ox + cw / 2 - 18, oy + 20), str(score[0]), fill=rc.FG, font=rc.font(26), anchor="rm")
        d.text((ox + cw / 2 + 18, oy + 20), str(score[1]), fill=rc.FG, font=rc.font(26), anchor="lm")
        d.text((ox + 10, oy + ch - 12), "wall", fill=rc.MUTED, font=rc.font(12), anchor="lm")
        d.text((ox + cw - 14, oy + ch - 12), "model", fill=rc.MUTED, font=rc.font(12), anchor="rm")
        if beside:
            self.panel(d, t, int(ox + cw + 24), oy, width - int(ox + cw + 24) - 16)
        else:
            self.panel(d, t, ox, int(oy + ch + 16), int(cw), compact=True)
        return img

    def panel(self, d: ImageDraw.ImageDraw, t: float, x: int, y: int, w: int, compact: bool = False) -> None:
        dec = self.decision_at(t)
        n = sum(1 for k in self.decided if self.times[k] <= t)
        d.text(
            (x, y + 4),
            "the model's last move" if rc.MOVES_ONLY else "the model's last answer",
            fill=rc.MUTED,
            font=rc.font(14),
            anchor="lm",
        )
        if not dec:
            d.text((x, y + 34), "waiting for the first answer", fill=rc.FG, font=rc.font(14), anchor="lm")
            return
        probs = (dec["answer"].get("move") or {}).get("probabilities") or {}
        bar_w = int(w * 0.58) if compact else w
        for k, m in enumerate(MOVES):
            rc.bar(d, x, y + 24 + k * 34, bar_w, 26, float(probs.get(m, 0.0)), m, m == dec["chosen"])
        tx, ty = (x + bar_w + 20, y + 30) if compact else (x, y + 136)
        d.text((tx, ty), f"{dec['latency_ms']:.0f} ms", fill=rc.FG, font=rc.font(28), anchor="lm")
        d.text((tx, ty + 28), "this decision, round trip", fill=rc.MUTED, font=rc.font(12), anchor="lm")
        d.text((tx, ty + 60), f"{n} decisions · {t - self.t0:.1f} s", fill=rc.FG, font=rc.font(15), anchor="lm")


def lane_images(lanes: list[PongTrajectory], rel: float, width: int, height: int) -> list[Image.Image]:
    return [
        rc.with_caption(lane.frame(lane.t0 + rel, width, height), rc.caption_text(lane.head["player"], lane.ticks))
        for lane in lanes
    ]


def frames(lanes: list[PongTrajectory], start: float = 0.0, seconds: float | None = None, fps: int = rc.FPS):
    span = max(lane.t1 - lane.t0 for lane in lanes) + 1.0
    end = span if seconds is None else min(span, start + seconds)
    width = W if len(lanes) == 1 else (W - 8 * (len(lanes) - 1)) // len(lanes)
    # a lane of a side-by-side clip is as tall as its court and the panel under it
    height = H if len(lanes) == 1 else int(64 + (width - 32) / lanes[0].court["w"] * lanes[0].court["h"] + 150)
    for rel in rc.frame_times(start, end, fps):
        yield rc.side_by_side(lane_images(lanes, rel, width, height))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectories", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None)
    rc.add_clip_options(ap)
    a = ap.parse_args()
    rc.apply_clip_options(a)
    lanes = [PongTrajectory(p) for p in a.trajectories]
    mp4 = rc.encode(frames(lanes, a.start, a.seconds), a.out)
    if a.gif:
        from demo_recorder import to_gif

        to_gif(mp4, a.gif)
    print(rc.finish_clip(mp4, a))


if __name__ == "__main__":
    main()
