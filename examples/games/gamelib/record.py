# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Recording a player's games as a GIF and an MP4, at the speed the player really played.

`Recorder.observer` goes to `players.play`; it keeps a snapshot of the game after every move. `build` then lays the
snapshots on a real-time line: a position is on screen for as long as the player took to decide the next move, and
the caption under the frame says who played, on what hardware, and the median time per move over the whole recording.
Several games can follow one another in one recording (Hangman is over in a fraction of a second at server speed).

A GIF cannot show a frame for less than 20 ms (browsers show shorter ones at 100 ms), so `min_frame_ms` floors the
timeline there; the count of frames it slowed is recorded.
"""

import copy
import json
import platform
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .render import FrameInfo, frame

GIF_LIMIT = 3 * 1024 * 1024


@dataclass
class Snapshot:
    game: object  # a copy of the game as it stood
    move: object  # the move that led here (None for the first position of a game)
    n: int
    label: str
    elapsed_ms: float  # real time since the first move of the recording
    duration_ms: float = 0.0  # how long the position was on screen: the time to decide the next move


@dataclass
class Recorder:
    kind: str
    snapshots: list = field(default_factory=list)
    label: str = ""
    elapsed: float = 0.0

    def start_game(self, label):
        self.label = label

    def observer(self, game, move, record):
        self.elapsed += move.latency_ms if move else 0.0
        self.snapshots.append(Snapshot(copy.deepcopy(game), move, len(record.moves), self.label, self.elapsed))

    def finish(self, end_hold_ms=900.0):
        """Set each position's time on screen: the next decision's latency, or `end_hold_ms` for a game's last one."""
        for s, nxt in zip(self.snapshots, [*self.snapshots[1:], None]):
            same_game = nxt is not None and nxt.label == s.label and nxt.move is not None
            s.duration_ms = nxt.move.latency_ms if same_game else end_hold_ms

    def latencies(self):
        return [s.move.latency_ms for s in self.snapshots if s.move is not None]

    def server_times(self):
        return [s.move.server_ms for s in self.snapshots if s.move is not None and s.move.server_ms is not None]

    def caption(self, who, hardware):
        lat = self.latencies()
        line2 = [hardware] if hardware else []
        if lat:
            text = f"median {statistics.median(lat):.0f} ms per move"
            server = self.server_times()
            if server:
                text += f" (server {statistics.median(server):.0f} ms)"
            line2.append(text)
        return f"{who}\n" + "  ·  ".join(line2)

    def build(self, who, hardware="", speed=1.0, min_frame_ms=20.0, scale=1):
        """([image], [duration in ms], the number of frames slowed to the floor), real time divided by `speed`."""
        caption = self.caption(who, hardware)
        images, durations, floored = [], [], 0
        for s in self.snapshots:
            move = s.move
            info = FrameInfo(
                title=s.label,
                caption=caption,
                probs=move.probs if move else None,
                chosen=move.action if move else None,
                latency_ms=move.latency_ms if move else None,
                move_no=s.n if move else None,
                elapsed_ms=s.elapsed_ms,
            )
            images.append(frame(s.game, info, scale))
            d = s.duration_ms / speed
            floored += d < min_frame_ms
            durations.append(max(min_frame_ms, d))
        return images, durations, floored


def detect_hardware():
    """A description of this machine's accelerator or chip; pass --hardware when the server runs elsewhere."""
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        if out:
            names = [line.strip() for line in out.splitlines()]
            name, mem = (x.strip() for x in names[0].split(","))
            return f"{len(names)}x {name.replace('NVIDIA ', '')} ({mem})"
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        return chip or "Apple silicon"
    return platform.processor() or platform.machine()


def merge_identical(images, durations):
    """Fold consecutive identical frames into one with their summed time (a bump into a wall changes little)."""
    out_i, out_d = [images[0]], [durations[0]]
    for img, d in zip(images[1:], durations[1:]):
        if img.tobytes() == out_i[-1].tobytes():
            out_d[-1] += d
        else:
            out_i.append(img)
            out_d.append(d)
    return out_i, out_d


def grid_times(durations, unit=10):
    """Frame times in whole units (a GIF counts in centiseconds) whose running total follows the real timeline, so
    rounding never accumulates; each is at least two units."""
    out, total, shown = [], 0.0, 0
    for d in durations:
        total += d
        target = round(total / unit) * unit
        step = max(2 * unit, target - shown)
        out.append(int(step))
        shown += step
    return out


def palette_image(images, colors):
    """One shared palette from a sample of the frames, so frames differ only where the picture does."""
    sample = images[:: max(1, len(images) // 12)][:12]
    strip = Image.new("RGB", (sample[0].width, sum(i.height for i in sample)))
    y = 0
    for img in sample:
        strip.paste(img, (0, y))
        y += img.height
    return strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def write_gif(images, durations, path, limit=GIF_LIMIT):
    """Write the GIF; reduce colours, then size, until it is under `limit` bytes. Returns what was written."""
    images, durations = merge_identical(images, durations)
    times = grid_times(durations)
    for colors, shrink in ((256, 1.0), (128, 1.0), (64, 1.0), (64, 0.75), (48, 0.6)):
        frames = images
        if shrink != 1.0:
            size = (int(images[0].width * shrink) // 2 * 2, int(images[0].height * shrink) // 2 * 2)
            frames = [im.resize(size, Image.LANCZOS) for im in images]
        pal = palette_image(frames, colors)
        pframes = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in frames]
        pframes[0].save(
            path, save_all=True, append_images=pframes[1:], duration=times, loop=0, optimize=False, disposal=1
        )
        size_bytes = Path(path).stat().st_size
        if size_bytes <= limit:
            return {"bytes": size_bytes, "colors": colors, "shrink": shrink, "frames": len(pframes), "ms": sum(times)}
    raise RuntimeError(f"{path} is {size_bytes} bytes after every reduction; record fewer or shorter games")


def write_mp4(images, durations, path, fps=50):
    """H.264 at a fixed `fps`: each output frame shows the position that was on screen at that moment."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed for the MP4 (apt install ffmpeg / brew install ffmpeg)")
    w, h = images[0].size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    step, ends, total = 1000.0 / fps, [], 0.0
    for d in durations:
        total += d
        ends.append(total)
    k, count = 0, int(total / step)
    for i in range(count):
        while k < len(ends) - 1 and i * step >= ends[k]:
            k += 1
        proc.stdin.write(images[k].tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")
    return {"frames": count, "fps": fps, "ms": round(total)}


def write_meta(path, meta):
    path.write_text(json.dumps(meta, indent=1) + "\n")
