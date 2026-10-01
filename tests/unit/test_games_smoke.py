# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A short game of each kind played headless against the CPU stand-in, through the same code a recording uses.

  S1  (fast tier) a few maze moves and a few Hangman moves are answered by a server started with --backend hf: each move
      is one /v1/systemone request with the game as text, the probabilities cover the four options or the 26 letters,
      the wall time is recorded, and the recorder lays the moves on a real-time line and writes a GIF
  S2  (fast tier) a task registered for the maze question (calibration, 12 teacher-labelled states) is found by the
      next move: x-decisio-tasks names it
  S3  (slow tier) the Hangman task registers from 5 examples per letter (the 26 options give it a head) and the next
      move is answered under it

The stand-in's play means nothing about the served model; this proves the plumbing.

    uv run pytest -q tests/unit/test_games_smoke.py
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "games"))

from gamelib.adapter import QUESTION_NAME  # noqa: E402
from gamelib.hangman import ALPHABET, Hangman  # noqa: E402
from gamelib.mazechase import ORDER, MazeChase  # noqa: E402
from gamelib.players import SystemOnePlayer, play, teacher_for  # noqa: E402
from gamelib.record import Recorder, write_gif  # noqa: E402
from gamelib.teach import hangman_teaching, register, teaching_examples, unregister  # noqa: E402

MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as r:
        return json.loads(r.read())


class Server:
    """`python -m decisio.serve.vllm_engine --backend hf` in a child process, stopped by its own PID."""

    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [sys.executable, "-m", "decisio.serve.vllm_engine", "--backend", "hf", "--model", MODEL]
        self.proc = subprocess.Popen(
            [*cmd, "--port", str(self.port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError("the stand-in server exited:\n" + self.proc.stdout.read())
            try:
                get(self.url, "/health")
                return
            except OSError:
                time.sleep(1)
        self.stop()
        raise AssertionError("the stand-in server did not come up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def server():
    s = Server()
    yield s
    s.stop()


def test_s1_a_few_moves_of_each_game(server, tmp_path):
    rec = Recorder("maze")
    rec.start_game("game 1 of 1")
    player = SystemOnePlayer("maze", server.url, name="stand-in")
    record = play(MazeChase(3), player, "maze", 3, teacher=teacher_for("maze"), observer=rec.observer, max_moves=5)
    assert len(record.moves) == 5 and all(m.action in ORDER for m in record.moves)
    assert all(set(m.probs) == set(ORDER) and abs(sum(m.probs.values()) - 1) < 1e-3 for m in record.moves)
    assert all(m.latency_ms > 0 for m in record.moves) and all(m.teacher in ORDER for m in record.moves)
    rec.finish(300)
    images, durations, _ = rec.build("stand-in", "CPU")
    info = write_gif(images, durations, tmp_path / "maze.gif")
    assert info["bytes"] < 3 * 1024 * 1024 and Image.open(tmp_path / "maze.gif").n_frames == 6
    hang = Hangman("planet")
    player = SystemOnePlayer("hangman", server.url, name="stand-in")
    record = play(hang, player, "hangman", "planet", teacher=teacher_for("hangman"), max_moves=4)
    letters = [m.action for m in record.moves]
    assert len(letters) == 4 and len(set(letters)) == 4 and all(c in ALPHABET for c in letters)  # never a repeat
    assert all(set(m.probs) == set(ALPHABET) and abs(sum(m.probs.values()) - 1) < 1e-3 for m in record.moves)
    assert QUESTION_NAME == "decision"


def test_s2_a_registered_task_is_found_by_the_next_move(server):
    player = SystemOnePlayer("maze", server.url)
    assert player.act(MazeChase(3)).task is None
    task = register(server.url, "maze", teaching_examples("maze", 12))
    try:
        assert task["n_examples"] == 12 and "applied" in task["calibration"] and task["head"]["applied"] is False
        assert player.act(MazeChase(3)).task == "maze-chase-move"
    finally:
        unregister(server.url, "maze")
    assert player.act(MazeChase(3)).task is None


@pytest.mark.slow
def test_s3_hangman_task_with_a_head(server):
    per_letter, examples = defaultdict(int), []
    for view, letter in hangman_teaching():
        if per_letter[letter] < 5:
            per_letter[letter] += 1
            examples.append((view, letter))
    assert len(examples) == 130
    task = register(server.url, "hangman", examples)
    try:
        assert task["per_option_min"] == 5 and len(task["options"]) == 26
        assert "applied" in task["head"] and "applied" in task["calibration"]
        move = SystemOnePlayer("hangman", server.url).act(Hangman("planet"))
        assert move.task == "hangman-letter" and move.action in ALPHABET
    finally:
        unregister(server.url, "hangman")
