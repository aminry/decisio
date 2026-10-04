# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Render a browser-agent trajectory (from measure_ultrafast.py) to a clip, from the trajectory and its frames only.

Each step shows the screenshot the agent observed for its decision with the chosen element outlined and labelled, and
a panel beside it: the operation question's distribution as bars, the chosen target with its probability, the typed
text, the decision's latency, the step and, under the picture, the goal. The last picture is the page after the agent
stopped, with the run's outcome as the measurement verified it. Nothing is re-executed: the page is never loaded again.

Time runs at real speed by the recorded `t_wall`: each step's picture is held until the next step's time, and the last
one for END_HOLD_S. A fast run is over in a blink, so `--min-hold S` holds every step at least S seconds; the caption
then says "held for readability", and the clock on the picture keeps the recorded time.

    python render_ultrafast.py runs/<run>/trajectories/travel_1 --out clip.mp4 [--gif clip.gif] [--min-hold 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_common as rc  # noqa: E402
import trajectory  # noqa: E402

W, H = 1280, 720
PAD = 16
HEAD_H = 40
SHOT_BOX = (PAD, PAD + HEAD_H - 8, 862, 600)  # x, y, max width, max height of the screenshot
PANEL_X = SHOT_BOX[0] + SHOT_BOX[2] + 20
PANEL_W = W - PANEL_X - PAD
HIGHLIGHT = (255, 46, 136)
GOOD = (88, 204, 120)
BAD = (255, 110, 96)
END_HOLD_S = 2.5


def wrap(text: str, size: int, width: int, max_lines: int) -> list[str]:
    """Lines of `text` that fit `width` pixels at `size`; the last kept line ends in an ellipsis if text was cut."""
    f = rc.font(size)
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        while f.getlength(w) > width:  # a word longer than the line (a URL) is cut where it overflows
            k = max(1, int(len(w) * width / f.getlength(w)) - 1)
            lines.append(w[:k])
            w = w[k:]
        cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        last = lines[max_lines - 1]
        while last and f.getlength(last + " …") > width:
            last = last[:-1]
        lines = [*lines[: max_lines - 1], last.rstrip() + " …"]
    return lines or [""]


def count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def shot_scale(frame_size) -> float:
    return min(SHOT_BOX[2] / frame_size[0], SHOT_BOX[3] / frame_size[1])


def to_canvas(bbox, frame_size) -> tuple[int, int, int, int]:
    """A box in screenshot pixels to (x0, y0, x1, y1) on the canvas, clipped to the screenshot."""
    s = shot_scale(frame_size)
    x, y, w, h = bbox
    x0, y0 = SHOT_BOX[0] + round(x * s), SHOT_BOX[1] + round(y * s)
    x1, y1 = SHOT_BOX[0] + round((x + w) * s), SHOT_BOX[1] + round((y + h) * s)
    right, bottom = SHOT_BOX[0] + round(frame_size[0] * s) - 1, SHOT_BOX[1] + round(frame_size[1] * s) - 1
    return max(x0, SHOT_BOX[0]), max(y0, SHOT_BOX[1]), min(x1, right), min(y1, bottom)


class Clip:
    """A trajectory as a function of clip time: `frame(t)` for t in [0, duration]."""

    def __init__(self, path: str | Path, min_hold: float = 0.0):
        path = Path(path)
        self.file = path / trajectory.NAME if path.is_dir() else path
        self.dir = self.file.parent
        self.head, self.ticks, self.end = trajectory.read(self.file)
        if not self.ticks:
            raise ValueError(f"{self.file}: no ticks to render ({self.end.get('status', 'no outcome')})")
        self.min_hold = min_hold
        self.steps = sum(1 for t in self.ticks if t.get("decision"))
        self.starts, self.held = [], False
        clock = 0.0
        for k, tick in enumerate(self.ticks):
            self.starts.append(clock)
            real = self.ticks[k + 1]["t_wall"] - tick["t_wall"] if k + 1 < len(self.ticks) else END_HOLD_S
            real = max(0.0, real)
            self.held |= real < min_hold
            clock += max(real, min_hold)
        self.duration = clock
        self._cache: dict[int, Image.Image] = {}

    def start_of(self, k: int) -> float:
        """The clip time at which tick `k` is first shown."""
        return self.starts[k]

    def index_at(self, t: float) -> int:
        return rc.at_or_before(self.starts, t)

    def times(self, fps: int = rc.FPS) -> list[float]:
        return [t for t in rc.frame_times(0.0, self.duration, fps) if t < self.duration]

    def caption(self) -> str:
        if self.held:
            extra = f"held for readability: each step at least {self.min_hold:g} s"
        else:
            extra = "real speed"
        return rc.caption_text(self.head.get("player") or {}, self.ticks, extra)

    def frame(self, t: float) -> Image.Image:
        k = self.index_at(t)
        if k not in self._cache:
            self._cache[k] = self.draw(k)
        return self._cache[k]

    def draw(self, k: int) -> Image.Image:
        tick = self.ticks[k]
        img = Image.new("RGB", (W, H), rc.BG)
        d = ImageDraw.Draw(img)
        run = self.head.get("run") or {}
        elapsed = tick["t_wall"] - self.ticks[0]["t_wall"]
        if tick.get("final"):
            title = f"Browser agent · {run.get('task', '')} · stopped after {count(self.steps, 'decision')}"
        else:
            title = f"Browser agent · {run.get('task', '')} · step {tick.get('step')} of {self.steps}"
        d.text((PAD, PAD + 4), title, fill=rc.FG, font=rc.font(19), anchor="lm")
        d.text((W - PAD, PAD + 4), f"t = {elapsed:.2f} s", fill=rc.MUTED, font=rc.font(17), anchor="rm")
        bottom = self.draw_shot(img, d, tick)
        y = bottom + 10
        for line in wrap(f"Goal: {run.get('goal', '')}", 15, SHOT_BOX[2], max(1, (H - y - 4) // 19)):
            d.text((PAD, y), line, fill=rc.FG, font=rc.font(15))
            y += 19
        if tick.get("final"):
            self.draw_outcome(d, k)
        else:
            self.draw_decision(d, tick)
        return img

    def draw_shot(self, img: Image.Image, d: ImageDraw.ImageDraw, tick: dict) -> int:
        """The recorded screenshot, scaled into SHOT_BOX, with the chosen element outlined; returns its bottom."""
        x, y = SHOT_BOX[0], SHOT_BOX[1]
        if not tick.get("frame"):
            d.rectangle([x, y, x + SHOT_BOX[2], y + SHOT_BOX[3]], outline=rc.RULE)
            msg = f"no screenshot: {tick.get('error', 'not recorded')}"
            d.text((x + 16, y + 24), msg, fill=rc.MUTED, font=rc.font(15))
            return y + SHOT_BOX[3]
        with Image.open(self.dir / tick["frame"]) as shot:
            shot = shot.convert("RGB")
        size = shot.size
        s = shot_scale(size)
        scaled = shot.resize((round(size[0] * s), round(size[1] * s)), Image.Resampling.LANCZOS)
        img.paste(scaled, (x, y))
        d.rectangle([x - 1, y - 1, x + scaled.width, y + scaled.height], outline=rc.RULE)
        hl = tick.get("highlight")
        if hl:
            x0, y0, x1, y1 = to_canvas(hl["bbox"], size)
            d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=HIGHLIGHT, width=3)
            op = ((tick.get("decision") or {}).get("chosen") or {}).get("operation", "")
            label = f"[{hl.get('index')}] {op}".strip()
            f = rc.font(14)
            tw = round(f.getlength(label)) + 12
            ty = y0 - 2 - 22 if y0 - 2 - 22 >= y else y1 + 3  # above the box, or below it at the top edge
            tx = min(max(x0 - 2, x), x + scaled.width - tw)
            d.rectangle([tx, ty, tx + tw, ty + 20], fill=HIGHLIGHT)
            d.text((tx + 6, ty + 10), label, fill=(255, 255, 255), font=f, anchor="lm")
        return y + scaled.height

    def section(self, d: ImageDraw.ImageDraw, y: int, name: str) -> int:
        d.text((PANEL_X, y), name, fill=rc.MUTED, font=rc.font(13))
        return y + 20

    def bars(self, d: ImageDraw.ImageDraw, y: int, decision: dict) -> int:
        answer = (decision.get("answer") or {}).get("operation") or {}
        probs = answer.get("probabilities") or {}
        chosen = (decision.get("chosen") or {}).get("operation")
        for option, p in probs.items():
            rc.bar(d, PANEL_X, y, PANEL_W, 22, float(p), option, option == chosen)
            y += 27
        return y

    def target_bars(self, d: ImageDraw.ImageDraw, y: int, decision: dict, target: str, top: int = 3) -> int:
        """The target question's most probable options (the chosen one always among them), labelled as asked."""
        name = f"{((decision.get('chosen') or {}).get('operation') or '').lower()}_target"
        probs = ((decision.get("answer") or {}).get(name) or {}).get("probabilities") or {}
        if not probs:
            return y
        criteria = (((decision.get("request") or {}).get("questions") or {}).get(name) or {}).get("criteria") or {}
        ranked = sorted(probs, key=lambda k: -float(probs[k]))[:top]
        if target in probs and target not in ranked:
            ranked[-1] = target
        f = rc.font(13)
        for key in ranked:
            desc = criteria.get(key)
            if isinstance(desc, str):  # SYSTEMONE_STRING_DESCRIPTIONS sends each description as a JSON string
                try:
                    desc = json.loads(desc)
                except ValueError:
                    desc = {"element": desc}
            label = (desc or {}).get("element") or f"[{key}]"
            room = PANEL_W - 12 - f.getlength(f"  {float(probs[key]):.2f}")
            while f.getlength(label) > room and len(label) > 2:
                label = label[:-2] + "…"
            rc.bar(d, PANEL_X, y, PANEL_W, 20, float(probs[key]), label, key == target, size=13)
            y += 24
        if len(probs) > len(ranked):
            d.text((PANEL_X, y), f"{len(probs) - len(ranked)} more options", fill=rc.MUTED, font=rc.font(12))
            y += 16
        return y

    def draw_decision(self, d: ImageDraw.ImageDraw, tick: dict) -> None:
        dec = tick["decision"]
        chosen = dec.get("chosen") or {}
        y = self.section(d, SHOT_BOX[1], "OPERATION")
        y = self.bars(d, y, dec) + 10
        y = self.section(d, y, "TARGET")
        target = chosen.get("target")
        if target is None:
            d.text((PANEL_X, y), "none for this operation", fill=rc.MUTED, font=rc.font(16))
            y += 22
        else:
            for line in wrap(f"[{target}] {chosen.get('label') or ''}", 16, PANEL_W, 2):
                d.text((PANEL_X, y), line, fill=rc.FG, font=rc.font(16))
                y += 21
            y = self.target_bars(d, y + 4, dec, target)
        y = self.section(d, y + 10, "TYPED TEXT")
        if dec.get("text"):
            for line in wrap(f"“{dec['text']}”", 16, PANEL_W, 2):
                d.text((PANEL_X, y), line, fill=rc.FG, font=rc.font(16))
                y += 21
            helper = dec.get("text_helper") or {}
            if helper:
                note = f"by {helper.get('model')} in {helper.get('latency_ms')} ms"
                d.text((PANEL_X, y), wrap(note, 13, PANEL_W, 1)[0], fill=rc.MUTED, font=rc.font(13))
                y += 18
        else:
            d.text((PANEL_X, y), "none", fill=rc.MUTED, font=rc.font(16))
            y += 21
        y = self.section(d, y + 10, "DECISION LATENCY")
        d.text((PANEL_X, y), f"{dec.get('latency_ms', 0):.0f} ms", fill=rc.FG, font=rc.font(30))
        y = self.section(d, y + 44, "PAGE")
        for line in wrap(tick.get("title") or "", 14, PANEL_W, 1):
            d.text((PANEL_X, y), line, fill=rc.FG, font=rc.font(14))
            y += 19

    def draw_outcome(self, d: ImageDraw.ImageDraw, k: int) -> None:
        last = next((t for t in reversed(self.ticks[:k]) if t.get("decision")), None)
        y = SHOT_BOX[1]
        if last:
            y = self.section(d, y, f"LAST DECISION · STEP {last.get('step')}")
            y = self.bars(d, y, last["decision"]) + 10
        end = self.end
        y = self.section(d, y, "OUTCOME")
        done = bool(end.get("completed"))
        verdict = "task completed" if done else "task not completed"
        d.text((PANEL_X, y), verdict, fill=GOOD if done else BAD, font=rc.font(24))
        y += 34
        for line in wrap(f"agent stopped: {end.get('status', 'unknown')}", 15, PANEL_W, 4):
            d.text((PANEL_X, y), line, fill=rc.FG, font=rc.font(15))
            y += 20
        y += 2
        v = end.get("verification") or {}
        checks = []
        if v.get("url_end"):
            checks.append((f"URL ends with {v['url_end']}", v.get("url_ok")))
        if v.get("text_has"):
            checks.append((f"page shows “{v['text_has']}”", v.get("text_ok")))
        for text, ok in checks:
            mark = "yes" if ok else "no"
            for line in wrap(f"{text}: {mark}", 14, PANEL_W, 2):
                d.text((PANEL_X, y), line, fill=GOOD if ok else BAD, font=rc.font(14))
                y += 19
        y = self.section(d, y + 12, "RUN")
        facts = [
            f"{end.get('elapsed_ms', 0)} ms from first decision to stop",
            f"{count(end.get('decisions', self.steps), 'decision')}, {count(end.get('actions', 0), 'action')}",
        ]
        if end.get("text_calls"):
            facts.append(count(end["text_calls"], "text-helper call"))
        for line in facts:
            d.text((PANEL_X, y), line, fill=rc.FG, font=rc.font(14))
            y += 19


def render(clip: Clip, out: str | Path, gif: str | Path | None = None) -> Path:
    mp4 = rc.encode(rc.render(clip.frame, clip.times(), clip.caption()), out)
    if gif:
        from demo_recorder import to_gif

        to_gif(mp4, gif, None, max_s=clip.duration + 1)
    return mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", help="a trajectory directory (or its trajectory.jsonl.gz)")
    ap.add_argument("--out", required=True, help="the MP4 to write")
    ap.add_argument("--gif", default=None, help="also a GIF (under 3 MB, demo_recorder.to_gif)")
    ap.add_argument("--min-hold", type=float, default=0.0, help="show every step at least this many seconds")
    a = ap.parse_args()
    clip = Clip(a.trajectory, a.min_hold)
    mp4 = render(clip, a.out, a.gif)
    print(mp4, f"{clip.duration:.2f} s", clip.caption())


if __name__ == "__main__":
    main()
