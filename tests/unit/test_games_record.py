# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The recorder of examples/games: real-time timeline, caption, GIF size and MP4.

R1  a position stays on screen for the time the player took to decide the next move; the last one of a game is held;
    the elapsed time is the sum of the latencies so far
R2  the caption names the player, the hardware and the median latency (and the server's, when there is one)
R3  GIF timing follows the real timeline without accumulating rounding error, never below 20 ms
R4  a recorded game is a GIF under 3 MB with one frame per distinct position, and an MP4 of the same length
"""

import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "games"))

from gamelib.hangman import Hangman  # noqa: E402
from gamelib.mazechase import MazeChase  # noqa: E402
from gamelib.players import Move, TeacherPlayer, play  # noqa: E402
from gamelib.record import GIF_LIMIT, Recorder, grid_times, merge_identical, write_gif, write_mp4  # noqa: E402


class Paced(TeacherPlayer):
    """The teacher, reporting a made-up decision time: 30 ms, then 50 ms, alternately."""

    def act(self, game):
        move = super().act(game)
        self.n = getattr(self, "n", 0) + 1
        return Move(move.action, {move.action: 0.9}, 30.0 if self.n % 2 else 50.0, 25.0 if self.n % 2 else 45.0)


def record_hangman(words=("planet", "garden")):
    rec = Recorder("hangman")
    for i, w in enumerate(words, 1):
        rec.start_game(f"word {i} of {len(words)}")
        play(Hangman(w), Paced("hangman"), "hangman", w, observer=rec.observer)
    rec.finish(900)
    return rec


def test_r1_timeline():
    rec = record_hangman()
    first = [s for s in rec.snapshots if s.label == "word 1 of 2"]
    assert first[0].move is None and first[0].elapsed_ms == 0
    assert [s.duration_ms for s in first[:3]] == [30.0, 50.0, 30.0]  # the next decision's latency
    assert first[-1].duration_ms == 900  # the finished game is held
    assert [s.elapsed_ms for s in first[:3]] == [0, 30.0, 80.0]
    second = [s for s in rec.snapshots if s.label == "word 2 of 2"]
    assert second[0].elapsed_ms == first[-1].elapsed_ms  # the games follow one another


def test_r2_caption():
    rec = record_hangman()
    caption = rec.caption("decisio server: some model", "1x Some GPU (96 GB)")
    assert caption.split("\n")[0] == "decisio server: some model"
    line = caption.split("\n")[1]
    median = statistics.median(rec.latencies())
    server = statistics.median(rec.server_times())
    assert "1x Some GPU (96 GB)" in line and f"median {median:.0f} ms per move" in line
    assert f"(server {server:.0f} ms)" in line
    images, durations, floored = rec.build("who", "hw", speed=1.0)
    assert len(images) == len(durations) == len(rec.snapshots) and durations[0] == 30.0 and floored == 0
    _, durations, floored = rec.build("who", "hw", speed=4.0, min_frame_ms=20)
    assert min(durations) == 20 and floored > 0


def test_r3_gif_times():
    real = [33.0, 47.0, 21.0, 90.0, 30.0, 5.0, 28.0]
    times = grid_times(real)
    assert all(t % 10 == 0 and t >= 20 for t in times)
    assert abs(sum(times) - sum(max(20, d) for d in real)) <= 30  # no drift beyond the floor's own additions
    assert grid_times([28.0] * 100)[-1] in (20, 30) and abs(sum(grid_times([28.0] * 100)) - 2800) <= 10
    a, b = Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")
    images, durations = merge_identical([a, a.copy(), b, b.copy(), a], [10, 20, 30, 40, 50])
    assert len(images) == 3 and durations == [30, 70, 50]


def test_r4_gif_and_mp4(tmp_path):
    maze = MazeChase(3)
    rec = Recorder("maze")
    rec.start_game("game 1 of 1")
    record = play(maze, Paced("maze"), "maze", 3, observer=rec.observer)
    rec.finish(900)
    images, durations, floored = rec.build("teacher", "test machine")
    info = write_gif(images, durations, tmp_path / "m.gif")
    assert info["bytes"] <= GIF_LIMIT and (tmp_path / "m.gif").stat().st_size == info["bytes"]
    gif = Image.open(tmp_path / "m.gif")
    assert gif.n_frames == info["frames"] and gif.size == images[0].size and gif.n_frames <= len(record.moves) + 1
    total = 0
    for i in range(gif.n_frames):
        gif.seek(i)
        total += gif.info["duration"]
    assert abs(total - sum(durations)) < 30 + 10 * floored
    if shutil.which("ffmpeg"):
        out = write_mp4(images, durations, tmp_path / "m.mp4")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", tmp_path / "m.mp4"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert probe == f"{images[0].width},{images[0].height}" and out["ms"] == round(sum(durations))
    else:
        with pytest.raises(RuntimeError, match="ffmpeg"):
            write_mp4(images, durations, tmp_path / "m.mp4")
