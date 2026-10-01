# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The decisio adapter of examples/games: state text, the question, the players and the move loop, against a fake
`/v1/systemone` server so nothing needs a model.

  A1  the request: one choice question; maze has four described options, Hangman 26 letters with no descriptions;
      the question is identical on every move (so one registered task applies to the whole game)
  A2  the state text carries what a player sees and observations, and nothing it must not: no secret word, no teacher's
      move, no verdict, recommendation, advice in the question or per-option score; the question never changes
  A3  SystemOnePlayer: the choice and probabilities come back, wall time and x-decisio-server-ms are recorded, the
      request carries the model when given, Hangman masks guessed letters, a wall bump is played as answered
  A4  `play` records every move with the teacher's label, legality and latency, and calls the observer per move
  A6  headless play.py writes one frame per move
  A5  the committed teaching examples: 10 for each of the 26 letters, none from a held-out word, labelled as the solver
      labels them, and the registration body is what POST /v1/tasks takes
"""

import json
import subprocess
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

GAMES = Path(__file__).resolve().parents[2] / "examples" / "games"
sys.path.insert(0, str(GAMES))

from gamelib.adapter import (  # noqa: E402
    HANGMAN_QUESTION,
    MAZE_QUESTION,
    QUESTION_NAME,
    hangman_state,
    maze_state,
    question_for,
    request_for,
    state_text,
)
from gamelib.hangman import ALPHABET, EntropySolver, Hangman, HangmanView, heldout_words, load_words  # noqa: E402
from gamelib.mazechase import ORDER, MazeChase, neighbours  # noqa: E402
from gamelib.players import RandomPlayer, SystemOnePlayer, TeacherPlayer, play, teacher_for  # noqa: E402
from gamelib.teach import TEACHING_FILE, hangman_teaching, registration_body, teaching_examples  # noqa: E402


class Fake(BaseHTTPRequestHandler):
    """Answers every question with `answer(request)` (set on the class) and records the requests."""

    answer = None
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        probs = Fake.answer(body)
        choice = max(probs, key=probs.get)
        out = json.dumps({"answers": {QUESTION_NAME: {"type": "choice", "choice": choice, "probabilities": probs}}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-decisio-server-ms", "12.5")
        self.send_header("x-decisio-tasks", "fake-task")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def fake():
    Fake.seen = []
    server = HTTPServer(("127.0.0.1", 0), Fake)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", Fake
    server.shutdown()


def test_a1_the_question():
    assert MAZE_QUESTION["type"] == HANGMAN_QUESTION["type"] == "choice"
    assert list(MAZE_QUESTION["criteria"]) == list(ORDER)
    assert all(MAZE_QUESTION["criteria"].values())
    assert list(HANGMAN_QUESTION["criteria"]) == list(ALPHABET)
    assert all(v is None for v in HANGMAN_QUESTION["criteria"].values())
    game = Hangman("planet")
    first = request_for("hangman", game.view())
    game.step("a")
    second = request_for("hangman", game.view())
    assert first["questions"] == second["questions"] == {QUESTION_NAME: HANGMAN_QUESTION}
    assert first["state"] != second["state"]


def test_a2_state_text():
    text = hangman_state(HangmanView("_a__a_", ("e", "t"), 6))
    assert "Word length: 6" in text and "Pattern: _ a _ _ a _" in text and "Wrong guesses (2 of 6): e, t" in text
    assert "none" in hangman_state(HangmanView("____", ()))
    game = MazeChase(3)
    for _ in range(4):
        game.step(game.view().legal[0])
    text = maze_state(game.view())
    grid = text.split("\n\n")[1].split("\n")
    assert len(grid) == len(game.grid) and grid[game.player[0]][game.player[1]] == "P"
    assert sum(row.count("C") for row in grid) <= len(game.chasers)
    assert f"You: row {game.player[0]} col {game.player[1]}" in text
    assert "Open moves: " + ", ".join(game.view().legal) in text
    assert "Last moves: " in text and f"Score: {game.score}." in text and f"Pellets left: {len(game.pellets)}." in text


def test_a2_the_renders():
    view = MazeChase(3).view()
    plain, observed = state_text("maze", view), state_text("maze", view, "observed")
    assert observed.startswith(plain) and "Next cells:" in observed and "Moves:" in observed
    cells = dict(x.split(": ", 1) for x in observed.split("Next cells:\n")[1].split("\nMoves:")[0].split("\n"))
    moves = dict(x.split(": ", 1) for x in observed.split("Moves:\n")[1].split("\n"))
    assert set(cells) == set(moves) == set(ORDER)
    for d in ORDER:
        assert (cells[d] == "wall") == (moves[d] == "wall") == (d not in view.legal)
        assert cells[d] == "wall" or ", then " in cells[d]
    open_lines = [v for v in moves.values() if v != "wall"]
    assert all("nearest chaser" in v and ("eats a pellet" in v or "nearest pellet" in v) for v in open_lines)
    # only observations: none of the words a verdict, a recommendation or advice would use
    for word in ("SAFE", "RISKY", "DEADLY", "DANGER", "TRAP", "best", "should", "prefer", "recommend", "avoid"):
        assert word not in observed.replace("Which way should", ""), word
    assert state_text("maze", view, "local") == state_text("maze", view, "grid+local")
    near = MazeChase(3)
    near.chasers[0].pos = next(c for d, c in neighbours(near.grid, near.player))
    assert "chaser on that cell" in state_text("maze", near.view(), "distances")
    # the question never changes with the render, and every render of both games is one question named `move`
    for render in (None, "local", "distances", "observed"):
        assert question_for("maze", render) == MAZE_QUESTION
        assert request_for("maze", view, render)["questions"] == {QUESTION_NAME: MAZE_QUESTION}
    assert request_for("maze", view, name="decision")["questions"] == {"decision": MAZE_QUESTION}
    h = HangmanView("_a__a_", ("e", "t"), 6)
    assert state_text("hangman", h) == hangman_state(h)
    assert "Letters already guessed: a, e, t" in state_text("hangman", h, "guessed")
    count = state_text("hangman", h, "count")
    assert count.startswith(hangman_state(h)) and f"Words that still fit: {len(EntropySolver().candidates(h))}" in count
    both = state_text("hangman", h, "observed")
    assert "Letters already guessed: a, e, t" in both and "Words that still fit: " in both
    assert "%" not in both  # no per-letter shares
    for bad in ("hints", "tips", "examples", "sense", "room"):
        with pytest.raises(ValueError, match="unknown"):
            state_text("hangman", h, bad)
        with pytest.raises(ValueError, match="unknown"):
            state_text("maze", view, bad)


def test_a3_system_one_player(fake):
    url, handler = fake
    handler.answer = lambda body: {c: (0.9 if c == "e" else 0.1 / 25) for c in ALPHABET}
    player = SystemOnePlayer("hangman", url, model="some/model")
    game = Hangman("seed")
    move = player.act(game)
    assert move.action == "e" and move.probs["e"] == 0.9 and move.server_ms == 12.5 and move.task == "fake-task"
    assert move.latency_ms > 0 and move.raw is None
    body = handler.seen[-1]
    assert body["model"] == "some/model" and QUESTION_NAME in body["questions"]
    game.step("e")  # e is guessed now: the player must not repeat it, and says it was the server's first choice
    move = player.act(game)
    assert move.action != "e" and move.raw == "e"
    SystemOnePlayer("hangman", url).act(Hangman("seed"))
    assert "model" not in handler.seen[-1]
    maze = MazeChase(3)
    bump = next(d for d in ORDER if d not in maze.view().legal)
    handler.answer = lambda body: {d: (0.7 if d == bump else 0.1) for d in ORDER}
    assert SystemOnePlayer("maze", url).act(maze).action == bump  # played as answered, a bump included


def test_a3_a_server_error_is_raised(fake):
    url, handler = fake
    handler.answer = lambda body: 1 / 0  # the handler dies and the connection drops: the player says so
    with pytest.raises(Exception):  # noqa: B017
        SystemOnePlayer("hangman", url).act(Hangman("seed"))


def test_a4_play_records_every_move():
    seen = []
    record = play(
        MazeChase(1),
        RandomPlayer("maze", 0),
        "maze",
        1,
        teacher=teacher_for("maze"),
        observer=lambda g, m, r: seen.append(m),
    )
    assert seen[0] is None and len(seen) == len(record.moves) + 1
    assert record.status in {"lost", "won", "timeout"} and record.detail["ticks"] == len(record.moves)
    assert all(m.teacher in ORDER for m in record.moves)
    assert [m.n for m in record.moves] == list(range(1, len(record.moves) + 1))
    solver = EntropySolver()
    record = play(
        Hangman("planet"), TeacherPlayer("hangman", solver), "hangman", "planet", teacher=teacher_for("hangman", solver)
    )
    assert record.status in {"won", "lost"} and all(m.action == m.teacher for m in record.moves)
    assert record.detail["word"] == "planet"


def test_a5_teaching_examples_and_registration_body():
    rows = json.loads(Path(TEACHING_FILE).read_text())
    per_letter = Counter(r["letter"] for r in rows)
    assert len(rows) == 260 and set(per_letter.values()) == {10} and len(per_letter) == 26
    held, words = set(heldout_words()), set(load_words())
    assert all(r["word"] not in held and r["word"] in words for r in rows)
    solver = EntropySolver()
    for r in rows[::13]:
        view = HangmanView(r["pattern"], tuple(r["wrong"]))
        assert solver.choose(view) == r["letter"] and r["letter"] in view.remaining
        assert r["word"] in solver.candidates(view)
    body = registration_body("hangman", hangman_teaching())
    assert body["id"] == "hangman-letter" and len(body["examples"]) == 260
    one = body["examples"][0]
    assert one["answer"] in ALPHABET and list(one["request"]["questions"]) == [QUESTION_NAME]
    maze = teaching_examples("maze", 12)
    assert len(maze) == 12 and all(a in MAZE_QUESTION["criteria"] for _, a in maze)


def test_a6_headless_writes_a_frame_per_move(tmp_path):
    out = subprocess.run(
        [sys.executable, GAMES / "play.py", "maze", "--seed", "1", "--player", "teacher", "--headless",
         "--frames-dir", tmp_path],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    moves = int(out.split("after ")[1].split(" moves")[0])
    assert len(list(tmp_path.glob("frame_*.png"))) == moves + 1
