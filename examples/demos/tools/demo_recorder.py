"""Record a browser demo to WebM, MP4 and a small GIF, with a caption strip under the picture.

Headless Chrome driven by Playwright records the video itself, so the client and the System One server can sit on
the same machine and the latency on screen is the server's. The WebM is the raw recording. The MP4 and the GIF carry a
caption strip below the picture (model, card, median latency), drawn with Pillow and stacked under the video, so it
never covers the demo's own interface. The GIF is kept under `max_gif_bytes` by lowering the frame rate, then the width.

    from demo_recorder import record, to_mp4, to_gif
    webm = record("http://127.0.0.1:3100/?clean=1", "out/pong", seconds=40)
    caption = "Decisio | RTX PRO 6000 | median 66 ms"
    to_mp4(webm, "out/pong.mp4", caption); to_gif(webm, "out/pong.gif", caption)

Chrome: `channel="chrome"` uses the system Google Chrome (the Mac); on the box pass `channel=None` for Playwright's own
Chromium, or `executable_path` for a Chrome binary.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path


def record(
    url: str,
    out_base: str | Path,
    *,
    seconds: float,
    size: tuple[int, int] = (1280, 720),
    drive: Callable | None = None,
    channel: str | None = "chrome",
    executable_path: str | None = None,
    settle_s: float = 1.0,
    storage_state: dict | None = None,
    until_drive_done: bool = False,
    chrome_args: list[str] | None = None,
    init_script: str | None = None,
) -> Path:
    """Open `url` in headless Chrome, record `seconds` of it and return the WebM path.

    `storage_state` is a Playwright storage state (cookies, localStorage) the page starts with, for a demo that is
    opened from state a script prepared. `drive(page)` runs after the page has loaded, for demos that need clicks
    or a task started; `init_script` runs in every page before its own scripts; when it returns the
    recording continues until `seconds` have passed in total, or stops when it returns if `until_drive_done`.
    """
    from playwright.sync_api import sync_playwright

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_base.parent / f".{out_base.name}_video"
    shutil.rmtree(tmp, ignore_errors=True)
    with sync_playwright() as p:
        launch = {"headless": True}
        if chrome_args:
            launch["args"] = chrome_args
        if executable_path:
            launch["executable_path"] = executable_path
        elif channel:
            launch["channel"] = channel
        browser = p.chromium.launch(**launch)
        context = browser.new_context(
            viewport={"width": size[0], "height": size[1]},
            record_video_dir=str(tmp),
            record_video_size={"width": size[0], "height": size[1]},
            storage_state=storage_state,
        )
        if init_script:
            context.add_init_script(init_script)
        page = context.new_page()
        page.goto(url, wait_until="load")
        page.wait_for_timeout(int(settle_s * 1000))
        if drive:
            drive(page)
        remaining = seconds - settle_s
        if remaining > 0 and not until_drive_done:
            page.wait_for_timeout(int(remaining * 1000))
        video = page.video
        context.close()
        browser.close()
        src = Path(video.path())
    webm = out_base.with_suffix(".webm")
    shutil.move(str(src), webm)
    shutil.rmtree(tmp, ignore_errors=True)
    return webm


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
