# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Frames for the two games, drawn with Pillow: the live window shows them, the recorder stitches them.

One renderer for both, so what is recorded is what is seen.
Nothing here is an asset: every shape is drawn from primitives and the only font is Pillow's built-in one.
A frame is the playfield between a header (the score line) and a footer (what the player chose, with what
probability, and a caption), so a recording says who played and how fast.
"""

import math
from dataclasses import dataclass, replace

from PIL import Image, ImageDraw, ImageFont

from .hangman import ALPHABET, Hangman, HangmanView
from .mazechase import DIRS, MazeView

BG = "#0b1020"
PANEL = "#141b34"
WALL = "#2f3f7a"
WALL_EDGE = "#4a5fa8"
PELLET = "#f4e9b8"
PLAYER = "#2dd4bf"
INK = "#e6eaf7"
DIM = "#8a93b8"
GOOD = "#34d399"
BAD = "#f87171"
CHASERS = {"hunter": "#ef4444", "wanderer": "#fb923c", "drifter": "#c084fc"}

HEADER, PLAYFIELD, FOOTER = 32, 288, 88
WIDTH, HEIGHT = 480, HEADER + PLAYFIELD + FOOTER  # at scale 1, for both games
SUPERSAMPLE = 2


@dataclass
class FrameInfo:
    """What a frame says besides the game: who is playing, the last decision and its probabilities, a caption."""

    title: str = ""  # shown in the header, right-aligned
    caption: str = ""  # the footer's last lines (split on newlines): who plays, on what hardware, how fast
    elapsed_ms: float | None = None  # real time since the first move, shown at the right of the decision line
    probs: dict | None = None  # option -> probability of the last decision
    chosen: str | None = None
    latency_ms: float | None = None
    move_no: int | None = None


_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.load_default(size=size)
    return _fonts[size]


class Canvas:
    """A drawing surface in scale-1 coordinates that draws at `k` times the size."""

    def __init__(self, k):
        self.k = k
        self.img = Image.new("RGB", (WIDTH * k, HEIGHT * k), BG)
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1, radius=0):
        k = self.k
        box = (x0 * k, y0 * k, x1 * k - 1, y1 * k - 1)
        if radius:
            self.d.rounded_rectangle(box, radius * k, fill=fill, outline=outline, width=width * k)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=width * k)

    def poly(self, points, fill):
        self.d.polygon([(x * self.k, y * self.k) for x, y in points], fill=fill)

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        k = self.k
        self.d.ellipse(
            ((cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k), fill=fill, outline=outline, width=width * k
        )

    def line(self, points, fill, width=1):
        self.d.line([(x * self.k, y * self.k) for x, y in points], fill=fill, width=int(width * self.k), joint="curve")

    def text(self, x, y, s, size=13, fill=INK, anchor="la"):
        self.d.text((x * self.k, y * self.k), s, font=font(size * self.k), fill=fill, anchor=anchor)


def eyes(c, cx, cy, size, dx=0, dy=0):
    for sx in (-1, 1):
        c.circle(cx + sx * size * 0.45, cy - size * 0.1, size * 0.28, fill="white")
        c.circle(cx + sx * size * 0.45 + dx * size * 0.12, cy - size * 0.1 + dy * size * 0.12, size * 0.13, fill=BG)


def chaser_shape(c, kind, cx, cy, r):
    colour = CHASERS[kind]
    if kind == "hunter":  # a diamond
        c.poly([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], colour)
    elif kind == "wanderer":  # a triangle
        c.poly([(cx, cy - r), (cx + r, cy + r * 0.8), (cx - r, cy + r * 0.8)], colour)
    else:  # a hexagon
        c.poly(
            [(cx + r * math.cos(math.pi / 3 * i), cy + r * math.sin(math.pi / 3 * i)) for i in range(6)],
            colour,
        )
    eyes(c, cx, cy + (r * 0.15 if kind == "wanderer" else 0), r * 0.8)


def chrome(c, header_left, info, decision):
    c.rect(0, 0, WIDTH, HEADER, fill=PANEL)
    c.text(12, HEADER / 2, header_left, 15, INK, "lm")
    if info and info.title:
        c.text(WIDTH - 12, HEADER / 2, info.title, 13, DIM, "rm")
    y0 = HEIGHT - FOOTER
    c.rect(0, y0, WIDTH, HEIGHT, fill=PANEL)
    text, colour = decision
    c.text(12, y0 + 44, text, 12, colour, "lm")
    if info and info.elapsed_ms is not None:
        c.text(WIDTH - 12, y0 + 44, f"t {info.elapsed_ms / 1000:.2f} s", 12, DIM, "rm")
    for i, line in enumerate((info.caption if info else "").split("\n")[:2]):
        c.text(12, y0 + 61 + i * 15, line, 11, DIM, "lm")


def bars(c, x, y, w, items, chosen, h=12, gap=5, label_w=44):
    """Horizontal probability bars, one per (label, p); the chosen option is highlighted."""
    for i, (label, p) in enumerate(items):
        yy = y + i * (h + gap)
        on = label == chosen
        c.text(x, yy + h / 2, label, 11, INK if on else DIM, "lm")
        c.rect(x + label_w, yy, x + w, yy + h, fill="#1d2647", radius=3)
        if p > 0:
            c.rect(
                x + label_w,
                yy,
                x + label_w + max(2, (w - label_w) * p),
                yy + h,
                fill=GOOD if on else "#5b6aa8",
                radius=3,
            )
        c.text(x + w + 6, yy + h / 2, f"{p * 100:.0f}%", 11, INK if on else DIM, "lm")


def decision_line(info):
    if info is None or info.chosen is None:
        return "", DIM
    s = f"move {info.move_no}: {info.chosen}" if info.move_no else f"{info.chosen}"
    if info.probs and info.chosen in info.probs:
        s += f"  \u00b7  {info.probs[info.chosen] * 100:.0f}%"
    if info.latency_ms is not None:
        s += f"  \u00b7  {info.latency_ms:.0f} ms"
    return s, INK


def draw_maze(c, view: MazeView, info):
    rows, cols = len(view.grid), len(view.grid[0])
    cell = min(WIDTH // cols, PLAYFIELD // rows)
    ox, oy = (WIDTH - cell * cols) // 2, HEADER + (PLAYFIELD - cell * rows) // 2
    for r, row in enumerate(view.grid):
        for col, ch in enumerate(row):
            x, y = ox + col * cell, oy + r * cell
            if ch == "#":
                c.rect(x, y, x + cell, y + cell, fill=WALL)
                c.rect(x + 1, y + 1, x + cell - 1, y + cell - 1, outline=WALL_EDGE, width=1, radius=3)
            elif (r, col) in view.pellets:
                c.circle(x + cell / 2, y + cell / 2, cell * 0.11, fill=PELLET)
    px, py = ox + (view.player[1] + 0.5) * cell, oy + (view.player[0] + 0.5) * cell
    face = view.recent[-1] if view.recent else "right"
    dr, dc = DIRS[face]
    half = cell * 0.36
    c.rect(px - half, py - half, px + half, py + half, fill=PLAYER, radius=cell * 0.14)
    ux, uy, vx, vy = dc, dr, -dr, dc  # along the heading, and across it
    tip = (px + ux * half * 0.6, py + uy * half * 0.6)
    back = (px - ux * half * 0.35, py - uy * half * 0.35)
    c.line(
        [
            (back[0] + vx * half * 0.55, back[1] + vy * half * 0.55),
            tip,
            (back[0] - vx * half * 0.55, back[1] - vy * half * 0.55),
        ],
        BG,
        cell * 0.11,
    )
    for r, col, kind in view.chasers:
        chaser_shape(c, kind, ox + (col + 0.5) * cell, oy + (r + 0.5) * cell, cell * 0.4)
    chrome(
        c,
        f"MAZE CHASE    score {view.score}    pellets {len(view.pellets)}    tick {view.tick}",
        info,
        decision_line(info),
    )
    if info and info.probs:
        order = ["up", "down", "left", "right"]
        y0 = HEIGHT - FOOTER + 6
        bars(c, 12, y0, 200, [(o, info.probs.get(o, 0.0)) for o in order[:2]], info.chosen, h=10, gap=4)
        bars(c, 250, y0, 200, [(o, info.probs.get(o, 0.0)) for o in order[2:]], info.chosen, h=10, gap=4)
    if view.status != "playing":  # over the top wall row, so the maze stays in view
        text = {"won": "CLEARED", "lost": "CAUGHT", "timeout": "TIME UP"}[view.status]
        colour = GOOD if view.status == "won" else BAD
        c.rect(
            WIDTH / 2 - 70, HEADER + 4, WIDTH / 2 + 70, HEADER + cell - 4, fill=BG, outline=colour, width=2, radius=6
        )
        c.text(WIDTH / 2, HEADER + cell / 2, text, 17, colour, "mm")


def gallows(c, x, y, misses):
    """The scaffold, with one more part for each miss (head, body, two arms, two legs)."""
    ink = "#c7cdea"
    c.line([(x, y + 150), (x + 110, y + 150)], ink, 4)
    c.line([(x + 30, y + 150), (x + 30, y + 10)], ink, 4)
    c.line([(x + 30, y + 10), (x + 100, y + 10)], ink, 4)
    c.line([(x + 100, y + 10), (x + 100, y + 30)], ink, 3)
    hx, hy = x + 100, y + 44
    parts = [
        lambda: c.circle(hx, hy, 14, outline=BAD, width=3),
        lambda: c.line([(hx, hy + 14), (hx, hy + 80)], BAD, 3),
        lambda: c.line([(hx, hy + 30), (hx - 24, hy + 56)], BAD, 3),
        lambda: c.line([(hx, hy + 30), (hx + 24, hy + 56)], BAD, 3),
        lambda: c.line([(hx, hy + 80), (hx - 20, hy + 108)], BAD, 3),
        lambda: c.line([(hx, hy + 80), (hx + 20, hy + 108)], BAD, 3),
    ]
    for part in parts[: min(misses, 6)]:
        part()


def draw_hangman(c, view: HangmanView, info, secret=None, status="playing", last=None):
    gallows(c, 20, HEADER + 14, len(view.wrong))
    n = len(view.pattern)
    box = min(34, (WIDTH - 230 - 14) // n)
    x0 = 230
    for i, ch in enumerate(view.pattern):
        x = x0 + i * box
        c.rect(x + 2, HEADER + 18, x + box - 2, HEADER + 18 + box + 6, fill="#1d2647", radius=4)
        shown = ch if ch != "_" else (secret[i] if status == "lost" and secret else "")
        if shown:
            c.text(
                x + box / 2,
                HEADER + 18 + (box + 6) / 2,
                shown.upper(),
                int(box * 0.62),
                INK if ch != "_" else BAD,
                "mm",
            )
    if status != "playing":
        text, colour = ("SOLVED", GOOD) if status == "won" else ("OUT OF GUESSES", BAD)
        c.text(x0, HEADER + 78, text, 14, colour)
    elif info and info.probs:
        c.text(x0, HEADER + 78, "the model's top letters", 11, DIM)
    if info and info.probs:
        top = sorted(info.probs.items(), key=lambda kv: -kv[1])[:5]
        bars(
            c,
            x0,
            HEADER + 96,
            196,
            [(k.upper(), p) for k, p in top],
            info.chosen.upper() if info.chosen else None,
            h=11,
            gap=5,
            label_w=22,
        )
    kw, kh, ky = 34, 30, HEADER + 204
    for i, ch in enumerate(ALPHABET):
        col, row = i % 13, i // 13
        x, y = 6 + col * (kw + 2), ky + row * (kh + 4)
        if ch in view.wrong:
            fill, ink = "#5b1f2a", BAD
        elif ch in view.pattern:
            fill, ink = "#134e3a", GOOD
        else:
            fill, ink = "#1d2647", INK
        c.rect(x, y, x + kw, y + kh, fill=fill, outline=INK if ch == last else None, width=2, radius=5)
        c.text(x + kw / 2, y + kh / 2, ch.upper(), 14, ink, "mm")
    chrome(
        c,
        f"HANGMAN    {n} letters    misses {len(view.wrong)}/{view.max_wrong}",
        info,
        decision_line(info),
    )


def masked_info(game, info):
    """The info with Hangman's probabilities as the player used them: over the letters not yet guessed when it chose,
    renormalised (a server's answer also covers letters already played, which the player skips)."""
    if not info or not info.probs or not info.chosen:
        return info
    before = set(game.view().guessed) - {info.chosen}
    left = {k: p for k, p in info.probs.items() if k not in before}
    total = sum(left.values()) or 1.0
    return replace(info, probs={k: p / total for k, p in left.items()})


def frame(game, info=None, scale=1):
    """The image of a game (`MazeChase` or `Hangman`) as it stands, at `scale` times 480x408."""
    c = Canvas(SUPERSAMPLE * scale)
    if isinstance(game, Hangman):
        last = game.moves[-1] if game.moves else None
        draw_hangman(c, game.view(), masked_info(game, info), secret=game.word, status=game.status, last=last)
    else:
        draw_maze(c, game.view(), info)
    return c.img.resize((WIDTH * scale, HEIGHT * scale), Image.LANCZOS)
