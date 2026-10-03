"""MP4 and small GIF output with a caption strip under the picture, for the clips the trajectory renderers draw.

The renderers (render_pong.py, render_fsd.py, render_ultrafast.py, render_triage.py) draw every frame from a trajectory
file and encode it (render_common.encode); `to_gif` makes the small GIF for a README, keeping it under `max_gif_bytes`
by lowering the frame rate, then the length, then the width. `to_mp4` and `caption_strip` stack a caption strip under
an existing video. Nothing here records a live page.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


STRIP_H = 44
FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
)


def _width_of(video: str | Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out)


def caption_strip(text: str, width: int, path: str | Path) -> Path:
    """A PNG `width` x STRIP_H with `text` on a dark ground, to stack under the video."""
    from PIL import Image, ImageDraw, ImageFont

    def font_at(size):
        for candidate in FONTS:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default(size=size)

    # the largest size, down to 11 px, at which the caption fits the picture's width
    size = 18
    font = font_at(size)
    while size > 11 and font.getlength(text) > width - 32:
        size -= 1
        font = font_at(size)
    image = Image.new("RGB", (width, STRIP_H), (10, 11, 15))
    draw = ImageDraw.Draw(image)
    draw.line([(0, 0), (width, 0)], fill=(42, 45, 54))
    draw.text((16, STRIP_H // 2), text, fill=(242, 243, 245), font=font, anchor="lm")
    image.save(path)
    return Path(path)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _with_caption(webm, caption, tmp):
    """(inputs, filter head) that put the caption strip under the video, or none."""
    if not caption:
        return [], "[0:v]"
    png = caption_strip(caption, _width_of(webm), Path(tmp) / "caption.png")
    return ["-i", str(png)], "[0:v][1:v]vstack=inputs=2[v];[v]"


def to_mp4(webm: str | Path, mp4: str | Path, caption: str | None = None) -> Path:
    mp4 = Path(mp4)
    with tempfile.TemporaryDirectory() as tmp:
        extra, head = _with_caption(webm, caption, tmp)
        graph = head + "pad=ceil(iw/2)*2:ceil(ih/2)*2[o]"
        _ffmpeg(
            "-i",
            str(webm),
            *extra,
            "-filter_complex",
            graph,
            "-map",
            "[o]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "25",
            "-movflags",
            "+faststart",
            str(mp4),
        )
    return mp4


def to_gif(
    webm: str | Path,
    gif: str | Path,
    caption: str | None = None,
    *,
    max_gif_bytes: int = 3_000_000,
    start_s: float = 0.0,
    max_s: float = 14.0,
) -> Path:
    """GIF under `max_gif_bytes`. A GIF of a busy scene grows fast, so it tries the widest picture it can read first
    and gives up frames per second and then seconds before it gives up width: 720 px, then 640, then 560 px."""
    gif = Path(gif)
    with tempfile.TemporaryDirectory() as tmp:
        extra, head = _with_caption(webm, caption, tmp)
        for fps, width, seconds in (
            (10, 720, max_s),
            (8, 720, max_s),
            (8, 720, max_s * 0.75),
            (8, 640, max_s * 0.75),
            (6, 640, max_s * 0.6),
            (8, 560, max_s * 0.6),
            (6, 560, max_s * 0.45),
            (6, 480, max_s * 0.45),
        ):
            graph = (
                head
                + f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                + "[b][p]paletteuse=dither=bayer:bayer_scale=4[o]"
            )
            _ffmpeg(
                "-ss",
                str(start_s),
                "-t",
                f"{seconds:.1f}",
                "-i",
                str(webm),
                *extra,
                "-filter_complex",
                graph,
                "-map",
                "[o]",
                "-loop",
                "0",
                str(gif),
            )
            if gif.stat().st_size <= max_gif_bytes:
                return gif
    raise RuntimeError(f"{gif} is still {gif.stat().st_size} bytes at the smallest setting; shorten max_s")
