# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Render the triage demo's stream from its trajectories (measure_triage.py writes them): a feed of routed tickets.

One trajectory gives a 1280 x 720 feed: the current ticket, the five most probable intents as bars, the chosen queue
marked right or wrong against the ticket's true intent, a running accuracy and the last few routed tickets. With
`--compare plain.jsonl.gz taught.jsonl.gz` the two arms play side by side on the same tickets, each from its own first
ticket, under a header line with the held-out lift from the run's summary.json (`--summary`). Frames follow the
recorded wall clock (a ticket appears when it was sent and its answer when it came back), so a slow model looks slow.
Every lane has the caption strip of render_common (the taught lane says "tasks registered").

Nothing is re-asked or simulated: the renderer draws only what the trajectories hold. When `--out` lies in a run
record, the record's files.json is written again so it lists the media.

    python render_triage.py runs/<run>/taught.jsonl.gz --out runs/<run>/media --stills 6
    python render_triage.py --compare runs/<run>/plain.jsonl.gz runs/<run>/taught.jsonl.gz \\
        --summary runs/<run>/summary.json --out runs/<run>/media --stills 6
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
from pathlib import Path

import render_common as rc
import trajectory
from PIL import Image, ImageDraw
from stats import write_files_json

W, H = 1280, 720
HEADER_H = 56  # the compare header band
LANE_W = (W - 8) // 2
LANE_H = H - HEADER_H
GOOD = (88, 200, 120)
BAD = (236, 94, 94)
CARD = (20, 22, 29)
LEAD_S = 0.5  # shown before the first ticket
HOLD_S = 2.5  # the last frame held after the last answer
TOP = 5
TAUGHT_EXTRA = "tasks registered"


@functools.lru_cache(maxsize=4096)
def wrap(text: str, size: int, width: int, max_lines: int) -> tuple[str, ...]:
    """Greedy word wrap to `width` pixels; the last line ends with an ellipsis when the text does not fit."""
    f = rc.font(size)
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if f.getlength(trial) <= width:
            line = trial
            continue
        if line:
            lines.append(line)
        line = word
        while f.getlength(line) > width:  # a single word wider than the line (an e-mail address on a narrow lane)
            cut = len(line)
            while cut > 1 and f.getlength(line[:cut]) > width:
                cut -= 1
            lines.append(line[:cut])
            line = line[cut:]
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = clip(lines[-1] + " …", size, width)
    return tuple(lines)


@functools.lru_cache(maxsize=4096)
def clip(text: str, size: int, width: int) -> str:
    """`text` cut to `width` pixels, with an ellipsis when cut (the longest prefix that fits, by bisection)."""
    f = rc.font(size)
    if f.getlength(text) <= width:
        return text
    lo, hi = 0, len(text)  # the longest prefix with the ellipsis that fits is text[:lo]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f.getlength(text[:mid] + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


def wrap_parts(text: str, size: int, width: int, sep: str = " · ") -> tuple[str, ...]:
    """Wrap a caption at its separators, so a line never breaks inside a part ("median 1783 ms per decision"); a part
    wider than a line is word-wrapped."""
    f = rc.font(size)
    lines: list[str] = []
    for part in text.split(sep):
        trial = f"{lines[-1]}{sep}{part}" if lines else part
        if lines and f.getlength(trial) <= width:
            lines[-1] = trial
        else:
            lines.extend(wrap(part, size, width, 99))
    return tuple(lines)


def fit_size(text: str, size: int, width: int, smallest: int = 11) -> int:
    while size > smallest and rc.font(size).getlength(text) > width:
        size -= 1
    return size


def mark(d: ImageDraw.ImageDraw, x: int, y: int, s: int, right: bool) -> None:
    """A check (right) or a cross (wrong) in a filled circle of diameter `s`, drawn as shapes so no font needs the
    glyphs."""
    d.ellipse([x, y, x + s, y + s], fill=GOOD if right else BAD)
    w = max(2, s // 7)
    if right:
        d.line(
            [(x + s * 0.27, y + s * 0.53), (x + s * 0.44, y + s * 0.7), (x + s * 0.74, y + s * 0.33)],
            fill=rc.BG,
            width=w,
            joint="curve",
        )
    else:
        a, b = s * 0.3, s * 0.7
        d.line([(x + a, y + a), (x + b, y + b)], fill=rc.BG, width=w)
        d.line([(x + a, y + b), (x + b, y + a)], fill=rc.BG, width=w)


class Feed:
    """One trajectory of the stream, indexed by wall time."""

    def __init__(self, path: str | Path):
        self.head, self.ticks, self.end = trajectory.read(path)
        if not self.ticks:
            raise ValueError(f"{path}: no tickets")
        self.player = self.head.get("player") or {}
        self.run = self.head.get("run") or {}
        self.taught = self.run.get("arm") == "taught"
        self.question = self.head.get("question", "route")
        self.sent = [t.get("t_sent", t["t_wall"]) for t in self.ticks]
        self.done = [t["t_wall"] for t in self.ticks]
        self.t0, self.t1 = self.sent[0], self.done[-1]
        self.rows = [self._row(t) for t in self.ticks]
        self.right_before = [0]  # right answers among the first n routed tickets
        for r in self.rows:
            self.right_before.append(self.right_before[-1] + int(bool(r["right"])))

    def _row(self, tick: dict) -> dict:
        d, s = tick.get("decision") or {}, tick["state"]
        answer = (d.get("answer") or {}).get(self.question) or {}
        probs = answer.get("probabilities") or {}
        top = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]
        chosen = d.get("chosen")
        return {
            "id": s.get("ticket_id", ""),
            "text": s["text"],
            "intent": s["intent"],
            "chosen": chosen,
            "right": chosen is not None and chosen == s["intent"],
            "top": top,
            "latency_ms": d.get("latency_ms"),
            "error": d.get("error"),
        }

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def lane_name(self) -> str:
        return "Tasks registered" if self.taught else "Plain question"

    def caption(self) -> str:
        return rc.caption_text(self.player, self.ticks, TAUGHT_EXTRA if self.taught else "")

    def at(self, t: float) -> tuple[int, bool, int]:
        """(index of the ticket on screen or -1 before the first, whether its answer is back, tickets routed)."""
        routed = bisect.bisect_right(self.done, t)
        if t < self.sent[0]:
            return -1, False, routed
        i = rc.at_or_before(self.sent, t)
        return i, t >= self.done[i], routed


def registration_line(run: dict) -> str:
    reg = run.get("registration") or {}
    if not reg:
        return ""
    kept = [k for k in ("head", "calibration") if (reg.get(k) or {}).get("applied")]
    kept_text = (
        ("kept: " + " and ".join("intent head" if k == "head" else k for k in kept)) if kept else "kept: nothing"
    )
    return f"registered from {reg.get('n_examples')} labelled tickets in {reg.get('seconds', 0):.0f} s, {kept_text}"


# ---- drawing -------------------------------------------------------------------------------------------------------


def draw_bars(d, x, y, w, row: dict, done: bool, size: int, bar_h: int, step: int) -> None:
    """The five most probable intents; the chosen one in the accent colour, a mark beside the chosen one (right or
    wrong) and a 'true' tag beside the true intent when it is among the five. Empty outlines while the answer is out."""
    if not done or not row["top"]:
        for k in range(TOP):
            d.rectangle([x, y + k * step, x + w, y + k * step + bar_h], outline=rc.RULE)
        return
    for k, (key, p) in enumerate(row["top"]):
        yy = y + k * step
        rc.bar(d, x, yy, w, bar_h, p, key, key == row["chosen"], size=size)
        tx = x + w + 12
        if key == row["chosen"]:
            mark(d, tx, yy + (bar_h - bar_h * 0.8) // 2, int(bar_h * 0.8), row["right"])
            tx += int(bar_h * 0.8) + 8
        if key == row["intent"]:
            d.text((tx, yy + bar_h // 2), "true", fill=GOOD, font=rc.font(size), anchor="lm")


def verdict(d, x, y, row: dict, done: bool, now_ms: float, size: int, width: int) -> None:
    if not done:
        d.text((x, y), f"routing… {now_ms:,.0f} ms", fill=rc.MUTED, font=rc.font(size), anchor="lm")
        return
    if row["chosen"] is None:
        d.text(
            (x, y),
            clip(f"no answer: {row['error'] or 'error'}", size, width),
            fill=BAD,
            font=rc.font(size),
            anchor="lm",
        )
        return
    s = int(size * 1.1)
    mark(d, x, y - s // 2, s, row["right"])
    xx = x + s + 12
    # the longest wording that fits: the true intent is never the part that gets cut
    what = "right" if row["right"] else f"wrong, true: {row['intent']}"
    latency = f" · {row['latency_ms']:,.0f} ms" if row["latency_ms"] is not None else ""
    options = [(f"routed to {row['chosen']}", what + latency), (row["chosen"], what + latency), (row["chosen"], what)]
    for text, tail in options:
        need = rc.font(size).getlength(text) + 14 + rc.font(size - 2).getlength(tail)
        if xx + need <= x + width:
            break
    d.text((xx, y), text, fill=rc.FG, font=rc.font(size), anchor="lm")
    xx += int(rc.font(size).getlength(text)) + 14
    d.text(
        (xx, y),
        clip(tail, size - 2, x + width - xx),
        fill=GOOD if row["right"] else BAD,
        font=rc.font(size - 2),
        anchor="lm",
    )


def accuracy_text(feed: Feed, routed: int) -> str:
    right = feed.right_before[routed]
    pct = f" · {100 * right / routed:.0f}%" if routed else ""
    return f"{right} / {routed} right{pct}"


def draw_single(feed: Feed, t: float) -> Image.Image:
    """The full 1280 x 720 feed of one trajectory at wall time `t`."""
    img = Image.new("RGB", (W, H), rc.BG)
    d = ImageDraw.Draw(img)
    i, done, routed = feed.at(t)
    n = len(feed.rows)
    d.text((32, 30), "Notewell support · ticket triage", fill=rc.FG, font=rc.font(24), anchor="lm")
    sub = f"Tasks {registration_line(feed.run)}" if feed.taught else "Plain question · 20 queues, no examples"
    size = fit_size(sub, 15, 760, 12)
    d.text((32, 60), clip(sub, size, 760), fill=rc.MUTED, font=rc.font(size), anchor="lm")
    d.text((W - 32, 30), accuracy_text(feed, routed), fill=rc.FG, font=rc.font(24), anchor="rm")
    d.text((W - 32, 60), f"ticket {max(i + 1, 0)} of {n}", fill=rc.MUTED, font=rc.font(15), anchor="rm")
    d.line([(0, 82), (W, 82)], fill=rc.RULE)
    d.line([(828, 82), (828, H)], fill=rc.RULE)

    if i < 0:
        d.text((32, 130), "waiting for the first ticket…", fill=rc.MUTED, font=rc.font(20), anchor="lm")
    else:
        row = feed.rows[i]
        d.text((32, 108), f"TICKET {row['id']}", fill=rc.MUTED, font=rc.font(15), anchor="lm")
        d.rounded_rectangle([24, 124, 812, 334], radius=8, fill=CARD, outline=rc.RULE)
        for k, line in enumerate(wrap(row["text"], 23, 756, 6)):
            d.text((40, 150 + k * 33), line, fill=rc.FG, font=rc.font(23), anchor="lm")
        d.text((32, 358), "TOP 5 INTENTS", fill=rc.MUTED, font=rc.font(15), anchor="lm")
        draw_bars(d, 32, 376, 560, row, done, 17, 34, 44)
        now_ms = (t - feed.sent[i]) * 1000
        verdict(d, 32, 618, row, done, now_ms, 22, 780)

    d.text((852, 108), "RECENT", fill=rc.MUTED, font=rc.font(15), anchor="lm")
    last = i - 1  # the tickets before the one on screen
    y = 128
    for j in range(last, max(-1, last - 9), -1):
        row = feed.rows[j]
        d.text((852, y + 12), clip(row["text"], 15, 380), fill=rc.FG, font=rc.font(15), anchor="lm")
        mark(d, 852, y + 30, 16, row["right"])
        line = row["chosen"] or "no answer"
        d.text((876, y + 38), line, fill=rc.FG, font=rc.font(15), anchor="lm")
        if not row["right"]:
            x = 876 + int(rc.font(15).getlength(line)) + 10
            d.text((x, y + 38), clip(f"true: {row['intent']}", 15, 1248 - x), fill=BAD, font=rc.font(15), anchor="lm")
        y += 64
    return img


def draw_lane(feed: Feed, t: float) -> Image.Image:
    """One arm in a half-width lane (LANE_W x LANE_H) for the side-by-side clip."""
    img = Image.new("RGB", (LANE_W, LANE_H), rc.BG)
    d = ImageDraw.Draw(img)
    i, done, routed = feed.at(t)
    n = len(feed.rows)
    d.text((20, 24), feed.lane_name(), fill=rc.ACCENT if feed.taught else rc.FG, font=rc.font(21), anchor="lm")
    d.text((LANE_W - 20, 24), accuracy_text(feed, routed), fill=rc.FG, font=rc.font(19), anchor="rm")
    d.line([(0, 48), (LANE_W, 48)], fill=rc.RULE)
    if i < 0:
        d.text((20, 80), "waiting for the first ticket…", fill=rc.MUTED, font=rc.font(17), anchor="lm")
    else:
        row = feed.rows[i]
        d.text((20, 66), f"TICKET {row['id']} · {i + 1} of {n}", fill=rc.MUTED, font=rc.font(13), anchor="lm")
        d.rounded_rectangle([12, 80, LANE_W - 12, 234], radius=8, fill=CARD, outline=rc.RULE)
        for k, line in enumerate(wrap(row["text"], 17, LANE_W - 52, 6)):
            d.text((26, 99 + k * 24), line, fill=rc.FG, font=rc.font(17), anchor="lm")
        draw_bars(d, 20, 248, 440, row, done, 15, 28, 36)
        verdict(d, 20, 444, row, done, (t - feed.sent[i]) * 1000, 18, LANE_W - 40)
    d.line([(0, 470), (LANE_W, 470)], fill=rc.RULE)
    d.text((20, 488), "RECENT", fill=rc.MUTED, font=rc.font(13), anchor="lm")
    last = i - 1
    y = 506
    for j in range(last, max(-1, last - 5), -1):
        row = feed.rows[j]
        mark(d, 20, y + 4, 15, row["right"])
        key = row["chosen"] or "no answer"
        d.text((44, y + 12), clip(key, 14, 196), fill=rc.FG if row["right"] else BAD, font=rc.font(14), anchor="lm")
        d.text((250, y + 12), clip(row["text"], 14, LANE_W - 270), fill=rc.MUTED, font=rc.font(14), anchor="lm")
        y += 30
    return img


def compare_header(summary: dict | None, feeds: list[Feed]) -> Image.Image:
    img = Image.new("RGB", (W, HEADER_H), rc.BG)
    d = ImageDraw.Draw(img)
    if summary:
        h = summary["heldout"]

        def pct(b):
            return f"{100 * b['est']:.1f}% [{100 * b['lo']:.1f}, {100 * b['hi']:.1f}]"

        dif = h["difference"]
        line = (
            f"Held-out accuracy, {h['n']} tickets: plain {pct(h['accuracy_plain'])} → tasks registered "
            f"{pct(h['accuracy_taught'])} · lift {100 * dif['est']:+.1f} points "
            f"[{100 * dif['lo']:+.1f}, {100 * dif['hi']:+.1f}]"
        )
    else:
        line = "Held-out accuracy: no summary given (--summary)"
    size = fit_size(line, 19, W - 40)
    d.text((20, 18), line, fill=rc.FG, font=rc.font(size), anchor="lm")
    taught = next((f for f in feeds if f.taught), None)
    sub = f"the same {len(feeds[0].rows)} stream tickets on both sides, real speed"
    if taught is not None and registration_line(taught.run):
        sub += " · " + registration_line(taught.run)
    d.text((20, 42), clip(sub, 14, W - 40), fill=rc.MUTED, font=rc.font(14), anchor="lm")
    d.line([(0, HEADER_H - 1), (W, HEADER_H - 1)], fill=rc.RULE)
    return img


def caption_layout(captions: list[str], width: int) -> tuple[int, list[list[str]]]:
    """One text size for every lane's caption, so strips side by side read alike: the largest size (18 down to 14) at
    which every caption fits on one line, else the largest (15 down to 11) at which each fits on two lines. Nothing is
    cut: the caption's end ("tasks registered") is the part a viewer most needs."""
    for size in range(18, 13, -1):
        if all(rc.font(size).getlength(c) <= width for c in captions):
            return size, [[c] for c in captions]
    for size in range(15, 10, -1):
        lines = [wrap_parts(c, size, width) for c in captions]
        if all(len(x) <= 2 for x in lines):
            return size, lines
    return 11, [wrap(c, 11, width, 2) for c in captions]


def with_caption(img: Image.Image, lines: list[str], size: int) -> Image.Image:
    """render_common's caption strip with the text laid out by caption_layout."""
    out = rc.with_caption(img, "")  # the picture, the strip and its rule
    d = ImageDraw.Draw(out)
    step = size + 5
    y0 = img.height + rc.STRIP_H // 2 - step * (len(lines) - 1) / 2
    for k, line in enumerate(lines):
        d.text((16, y0 + k * step), line, fill=rc.FG, font=rc.font(size), anchor="lm")
    return out


class Renderer:
    """frame(seconds from the start) -> captioned image; consecutive frames that show the same thing are drawn once."""

    def __init__(self, feeds: list[Feed], summary: dict | None = None):
        self.feeds = feeds
        self.compare = len(feeds) > 1
        self.header = compare_header(summary, feeds) if self.compare else None
        self.length = max(f.duration for f in feeds) + LEAD_S + HOLD_S
        width = (LANE_W if self.compare else W) - 32
        self.caption_size, self.captions = caption_layout([f.caption() for f in feeds], width)
        self._lanes: list[tuple | None] = [None] * len(feeds)  # per lane: (state, captioned image) last drawn
        self._key, self._img = None, None

    def times(self, fps: int = rc.FPS) -> list[float]:
        return rc.frame_times(0.0, self.length, fps)

    @staticmethod
    def _lane_state(f: Feed, t: float) -> tuple:
        """What a lane shows at `t`: the ticket, whether its answer is back, the count routed, the ms counter."""
        i, done, routed = f.at(t)
        return i, done, routed, None if done or i < 0 else int((t - f.sent[i]) * 1000)

    def _lane(self, k: int, dt: float) -> Image.Image:
        f = self.feeds[k]
        t = f.t0 - LEAD_S + dt
        state = self._lane_state(f, t)
        cached = self._lanes[k]
        if cached is None or cached[0] != state:
            pic = draw_lane(f, t) if self.compare else draw_single(f, t)
            cached = (state, with_caption(pic, self.captions[k], self.caption_size))
            self._lanes[k] = cached
        return cached[1]

    def frame(self, dt: float) -> Image.Image:
        key = tuple(self._lane_state(f, f.t0 - LEAD_S + dt) for f in self.feeds)
        if key == self._key:
            return self._img
        lanes = [self._lane(k, dt) for k in range(len(self.feeds))]
        if self.compare:
            body = rc.side_by_side(lanes)
            img = Image.new("RGB", (W, HEADER_H + body.height), rc.BG)
            img.paste(self.header, (0, 0))
            img.paste(body, (0, HEADER_H))
        else:
            img = lanes[0]
        self._key, self._img = key, img
        return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", nargs="?", help="one trajectory (plain.jsonl.gz or taught.jsonl.gz)")
    ap.add_argument("--compare", nargs=2, metavar=("PLAIN", "TAUGHT"), help="two trajectories side by side")
    ap.add_argument("--summary", default=None, help="the run's summary.json, for the held-out lift in the header")
    ap.add_argument("--out", required=True, help="directory for the MP4 and the stills")
    ap.add_argument("--name", default=None, help="file name stem (default triage or triage_compare)")
    ap.add_argument("--stills", type=int, default=0, help="also save N evenly spaced frames as PNG")
    ap.add_argument("--no-video", action="store_true", help="stills only")
    ap.add_argument("--fps", type=int, default=rc.FPS)
    a = ap.parse_args()
    if bool(a.trajectory) == bool(a.compare):
        ap.error("give one trajectory or --compare PLAIN TAUGHT")
    feeds = [Feed(p) for p in (a.compare or [a.trajectory])]
    summary = json.loads(Path(a.summary).read_text()) if a.summary else None
    r = Renderer(feeds, summary)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.name or ("triage_compare" if a.compare else "triage")
    times = r.times(a.fps)
    for k in range(a.stills):
        t = times[int((k + 0.5) * len(times) / a.stills)]
        path = out / f"{stem}_{k + 1:02d}.png"
        r.frame(t).save(path)
        print(path)
    if not a.no_video:
        mp4 = rc.encode((r.frame(t) for t in times), out / f"{stem}.mp4", fps=a.fps)
        print(f"{mp4}: {len(times)} frames, {r.length:.1f} s")
    record = next(
        (p for p in (out, *out.parents) if (p / "manifest.json").exists() and (p / "files.json").exists()), None
    )
    if record is not None:  # the media belong to the run record: list them with their hashes
        write_files_json(record)
        print(f"{record / 'files.json'} updated")


if __name__ == "__main__":
    main()
