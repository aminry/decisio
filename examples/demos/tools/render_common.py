# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""What every trajectory renderer shares: fonts, the caption strip, the frame clock, side-by-side lanes, encoding.

A renderer is a pure function of a trajectory and a wall-clock time: `frame(t) -> PIL.Image`. Frames are drawn at a
fixed rate over the recorded wall clock (real speed), so a slow model looks slow. Nothing is simulated: between two
ticks a renderer may only show the earlier tick's state or interpolate between the two recorded states.
"""

from __future__ import annotations

import hashlib
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 30
STRIP_H = 44
BG = (10, 11, 15)
FG = (242, 243, 245)
MUTED = (150, 154, 166)
RULE = (42, 45, 54)
ACCENT = (94, 180, 255)
FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
)
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for candidate in FONTS:
            if Path(candidate).exists():
                _font_cache[size] = ImageFont.truetype(candidate, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default(size=size)
    return _font_cache[size]


def median_latency_ms(ticks: list[dict]) -> float | None:
    lat = [t["decision"]["latency_ms"] for t in ticks if t.get("decision") and "answer" in t["decision"]]
    return statistics.median(lat) if lat else None


def latency_line(ticks: list[dict]) -> str:
    """The median over the answers that arrived, and how many requests got no usable answer in time (the driving
    demo's 1.5 s timeout, or a failed call): a median over the answers alone would hide them."""
    med = median_latency_ms(ticks)
    asked = [t["decision"] for t in ticks if t.get("decision")]
    late = sum(1 for d in asked if d.get("rules_fallback") or d.get("error") or "answer" not in d)
    line = f"median {med:.0f} ms per decision" if med is not None else "no decision"
    if late:
        line += f", {late} of {len(asked)} requests without an answer in time"
    return line


# Moves only (`--moves-only` on every renderer): an option is shown as chosen or not, with no probability, and options
# are listed in the question's own order, never ranked; where a question has many options only the chosen one is shown.
# For a hosted model whose probabilities are not to be shown (Jev), and for a chat model's one-option answer, whose
# 1.00 / 0.00 would read as a confidence it never stated.
MOVES_ONLY = False

HOSTED_NOTE = "model output · AI-generated · rendered by the decisio project"


def is_hosted(player: dict) -> bool:
    """A model reached over a hosted route (an API), not served by this project on its own card."""
    route = str(player.get("route") or "")
    return bool(route) and not route.startswith("local")


def caption_text(player: dict, ticks: list[dict], extra: str = "") -> str:
    """Model, card, median latency of the recorded run, and the padding: in every clip. A hosted model's caption names
    its exact model ID, the date, the route and the serving provider, and says the content is model output and
    AI-generated, rendered by the decisio project; its latency line is the same as ours (the client's round trip)."""
    if is_hosted(player):
        label = player.get("label") or player.get("model_id") or "model"
        mid = player.get("model_id")
        parts = [label if not mid or mid == label else f"{label} ({mid})"]
        parts.append(str(player.get("date") or "date not recorded"))
        route = str(player["route"])
        provider = player.get("provider")
        parts.append(f"{route}, served by {provider}" if provider and provider not in route else route)
        lat = latency_line(ticks)
        card = str(player.get("card") or "")
        if card:  # a hosted model's latency is the client's round trip, network included
            client = card.split(" (client)")[0]
            lat += f", from a client on an {client}" if " (client)" in card else f" ({card})"
        parts.append(lat)
        if extra:
            parts.append(extra)
        parts.append(HOSTED_NOTE)
        return " · ".join(parts)
    parts = [player.get("label") or player.get("model_id") or "model"]
    if player.get("card"):
        parts.append(str(player["card"]))
    parts.append(latency_line(ticks))
    if player.get("padding"):
        parts.append(f"padding {player['padding']}")
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def wrap_caption(text: str, width: int, size: int) -> list[str]:
    """The caption wrapped to `width` at `size` px, breaking at its " · " separators first and at spaces inside a part
    only when the part alone is too wide: it is never cut off."""
    room, f = width - 32, font(size)
    pieces = []  # the separator-delimited parts, each split at spaces only if it is wider than a line
    for part in text.split(" · "):
        if f.getlength(part) <= room:
            pieces.append(part)
            continue
        line = ""
        for w in part.split(" "):
            if line and f.getlength(f"{line} {w}") > room:
                pieces.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        pieces.append(line)
    lines = [""]
    for piece in pieces:
        trial = f"{lines[-1]} · {piece}" if lines[-1] else piece
        if f.getlength(trial) <= room or not lines[-1]:
            lines[-1] = trial
        else:
            lines.append(piece)
    return lines


def caption_lines(text: str, width: int) -> tuple[list[str], int]:
    """The caption as one line at the largest size from 18 px down to 13 px that fits `width`, or else wrapped onto at
    most five lines at the largest size from 16 px down to 11 px (wrap_caption)."""
    for size in range(18, 12, -1):
        if font(size).getlength(text) <= width - 32:
            return [text], size
    for size in range(16, 10, -1):
        lines = wrap_caption(text, width, size)
        if len(lines) <= 5:
            return lines, size
    return lines, 11


def with_caption(img: Image.Image, text: str, size: int | None = None, rows: int | None = None) -> Image.Image:
    """The picture with a caption strip under it (caption_lines). Side-by-side lanes pass one `size` and one strip
    height in `rows` (lines), so their captions match."""
    if size:
        lines = wrap_caption(text, img.width, size)
    else:
        lines, size = caption_lines(text, img.width)
    n = max(len(lines), rows or 0)
    strip = STRIP_H + (n - 1) * (size + 6)
    out = Image.new("RGB", (img.width, img.height + strip), BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.line([(0, img.height), (img.width, img.height)], fill=RULE)
    step = size + 6
    # with `rows`, the first line sits where it would in a full strip, so the lanes' captions start level
    top = img.height + strip // 2 - step * (n - 1) / 2
    for k, line in enumerate(lines):
        d.text((16, top + k * step), line, fill=FG, font=font(size), anchor="lm")
    return out


def side_by_side(images: list[Image.Image], gap: int = 8) -> Image.Image:
    """Lanes from separate trajectories, left to right, top-aligned."""
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), BG)
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def frame_times(t0: float, t1: float, fps: int = FPS) -> list[float]:
    n = max(1, int((t1 - t0) * fps) + 1)
    return [t0 + k / fps for k in range(n)]


def at_or_before(times: list[float], t: float) -> int:
    """Index of the last recorded time <= t (0 before the first)."""
    lo, hi = 0, len(times) - 1
    if t < times[0]:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid - 1
    return lo


def digest(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes() + str(img.size).encode()).hexdigest()


def encode(frames: Iterable[Image.Image], mp4: str | Path, fps: int = FPS) -> Path:
    """PNG frames to an H.264 MP4 (even dimensions, yuv420p, faststart)."""
    mp4 = Path(mp4)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for n, img in enumerate(frames, 1):
            img.save(Path(tmp) / f"{n:06d}.png")
        if n == 0:
            raise ValueError("no frames")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "%06d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(mp4),
            ],
            check=True,
        )
    return mp4


def render(frame: Callable[[float], Image.Image], times: list[float], caption: str | None) -> Iterable[Image.Image]:
    for t in times:
        img = frame(t)
        yield with_caption(img, caption) if caption else img


def bar(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, p: float, label: str, chosen: bool, size: int = 14):
    """One option's probability as a labelled bar; the chosen option in the accent colour. Under MOVES_ONLY the
    option is drawn as chosen (filled) or not (outlined), with no number."""
    d.rectangle([x, y, x + w, y + h], outline=RULE)
    if MOVES_ONLY:
        if chosen:
            d.rectangle([x, y, x + w, y + h], fill=ACCENT)
        d.text(
            (x + 6, y + h // 2),
            f"{label}  (chosen)" if chosen else label,
            fill=FG if chosen else MUTED,
            font=font(size),
            anchor="lm",
        )
        return
    d.rectangle([x, y, x + int(w * max(0.0, min(1.0, p))), y + h], fill=ACCENT if chosen else (70, 74, 86))
    d.text((x + 6, y + h // 2), f"{label}  {p:.2f}", fill=FG, font=font(size), anchor="lm")


def scoreboard(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    width: int,
    height: int,
    note: str = "",
    highlight: str | None = None,
) -> Image.Image:
    """A closing card: one row per player with that demo's totals, the row of the clip's own player marked."""
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    n = len(columns)
    hs = 15  # the column heads: wrapped onto two lines rather than shrunk

    def wrap(text: str, room: float) -> list[str]:
        words, lines = text.split(), [""]
        for w in words:
            trial = f"{lines[-1]} {w}".strip()
            if font(hs).getlength(trial) <= room or not lines[-1]:
                lines[-1] = trial
            else:
                lines.append(w)
        return lines

    size = 22
    while size > 12:
        widths = [max(font(size).getlength(str(r[k])) for r in rows) + 36 for k in range(n)]
        widths = [max(w, min(font(hs).getlength(c), 110) + 36) for w, c in zip(widths, columns)]
        if sum(widths) <= width - 80:
            break
        size -= 1
    spare = (width - 80 - sum(widths)) / max(1, n - 1)
    widths = [w + (spare if k else 0) for k, w in enumerate(widths)]
    heads = [wrap(c, w - 30) for c, w in zip(columns, widths)]
    head_h = 22 * max(len(h) for h in heads)
    step = size + 26
    block = 80 + head_h + 26 + step * len(rows)
    y = max(30, (height - block) // 2 - 20)
    ts = 30
    while ts > 16 and font(ts).getlength(title) > width - 80:
        ts -= 1
    d.text((40, y + 20), title, fill=FG, font=font(ts), anchor="lm")
    y += 70
    x = 40
    for k, lines in enumerate(heads):
        for j, line in enumerate(lines):
            d.text((x, y + 22 * j), line, fill=MUTED, font=font(hs), anchor="lt")
        x += widths[k]
    y += head_h + 8
    d.line([(40, y), (width - 40, y)], fill=RULE)
    y += step // 2 + 8
    for r in rows:
        mine = highlight is not None and r[0] == highlight
        if mine:
            d.rectangle([30, y - step // 2 + 3, width - 30, y + step // 2 - 3], fill=(22, 40, 60))
        x = 40
        for k, cell in enumerate(r):
            d.text((x, y), str(cell), fill=ACCENT if mine and k == 0 else FG, font=font(size), anchor="lm")
            x += widths[k]
        y += step
    lines = note.split("\n") if note else []
    for k, line in enumerate(lines):
        d.text((40, height - 36 - 22 * (len(lines) - 1 - k)), line, fill=MUTED, font=font(14), anchor="lm")
    return img


def append_card(mp4: str | Path, card: Image.Image, seconds: float = 5.0, fps: int = FPS) -> Path:
    """The clip with `card` held for `seconds` at its end (the card is resized to the clip's frame size)."""
    mp4 = Path(mp4)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    w, h = (int(v) for v in probe.split(","))
    with tempfile.TemporaryDirectory() as tmp:
        card_png = Path(tmp) / "card.png"
        card.resize((w, h)).save(card_png)
        tail = Path(tmp) / "tail.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{seconds}",
                "-i",
                str(card_png),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(tail),
            ],
            check=True,
        )
        out = Path(tmp) / "out.mp4"
        lst = Path(tmp) / "list.txt"
        lst.write_text(f"file '{mp4.resolve()}'\nfile '{tail}'\n")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
        out.replace(mp4)
    return mp4


def add_clip_options(ap) -> None:
    """The options every clip renderer shares: moves only, and a closing scoreboard card."""
    ap.add_argument("--moves-only", action="store_true", help="show each option as chosen or not, no probabilities")
    ap.add_argument("--end-card", default=None, help="a scoreboard JSON (title, columns, rows, note, highlight)")
    ap.add_argument("--end-card-seconds", type=float, default=5.0)


def apply_clip_options(a) -> None:
    global MOVES_ONLY
    MOVES_ONLY = bool(getattr(a, "moves_only", False))


def finish_clip(mp4: str | Path, a) -> Path:
    """Append the scoreboard card, when one is given, to a rendered clip."""
    if getattr(a, "end_card", None):
        import json

        spec = json.loads(Path(a.end_card).read_text())
        card = scoreboard(
            spec["title"], spec["columns"], spec["rows"], 1280, 720, spec.get("note", ""), spec.get("highlight")
        )
        append_card(mp4, card, a.end_card_seconds)
    return Path(mp4)
