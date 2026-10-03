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


def caption_text(player: dict, ticks: list[dict], extra: str = "") -> str:
    """Model, card, median latency of the recorded run, and the padding: in every clip."""
    med = median_latency_ms(ticks)
    parts = [player.get("label") or player.get("model_id") or "model"]
    if player.get("card"):
        parts.append(str(player["card"]))
    parts.append(f"median {med:.0f} ms per decision" if med is not None else "no decision")
    if player.get("padding"):
        parts.append(f"padding {player['padding']}")
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def with_caption(img: Image.Image, text: str) -> Image.Image:
    """The picture with a caption strip under it; the text shrinks (down to 11 px) to fit the width."""
    out = Image.new("RGB", (img.width, img.height + STRIP_H), BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    size = 18
    while size > 11 and font(size).getlength(text) > img.width - 32:
        size -= 1
    d.line([(0, img.height), (img.width, img.height)], fill=RULE)
    d.text((16, img.height + STRIP_H // 2), text, fill=FG, font=font(size), anchor="lm")
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
    """One option's probability as a labelled bar; the chosen option in the accent colour."""
    d.rectangle([x, y, x + w, y + h], outline=RULE)
    d.rectangle([x, y, x + int(w * max(0.0, min(1.0, p))), y + h], fill=ACCENT if chosen else (70, 74, 86))
    d.text((x + 6, y + h // 2), f"{label}  {p:.2f}", fill=FG, font=font(size), anchor="lm")
