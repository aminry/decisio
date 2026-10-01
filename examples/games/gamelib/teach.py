# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Teaching examples: states labelled with what the teacher would play, and the registration of a task from them.

The model is frozen; `POST /v1/tasks` fits a per-task calibration and, for 10 or more options with at least 5 examples
each, a head on its hidden state, from labelled examples of one recurring question (docs/tasks.md).
Hangman's question is such a task: 26 letters, so the head applies, with 10 examples per letter.
"""

import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .adapter import QUESTION_NAME, QUESTIONS, request_for
from .hangman import ALPHABET, EntropySolver, Hangman, HangmanView, heldout_words, load_words
from .mazechase import MazeChase, teacher_move

TASK_IDS = {"maze": "maze-chase-move", "hangman": "hangman-letter"}
TEACHING_FILE = Path(__file__).parent / "data" / "hangman_teaching.json"


def hangman_examples(per_letter=10, seed=0, words=None, solver=None, noise=0.25, max_tries=200000):
    """[(view, letter, word)]: `per_letter` states for each letter, each labelled with the entropy teacher's choice.

    States come from teacher games on words outside the held-out 200, stopped at a random move; with probability
    `noise` a move in the lead-up is a random letter, so that rare letters (q, z, x) become the teacher's choice often
    enough to get their quota. A state is used once. The order is shuffled, so no letter's examples are adjacent."""
    words = words or load_words()
    held = set(heldout_words())
    pool = [w for w in words if w not in held]
    solver = solver or EntropySolver(words)
    rng = random.Random(f"teach:{seed}")
    need = {c: per_letter for c in ALPHABET}
    seen, out = set(), []
    for _ in range(max_tries):
        if not any(need.values()):
            break
        game = Hangman(rng.choice(pool))
        for _ in range(rng.randint(0, len(game.word) + 2)):
            if game.over:
                break
            view = game.view()
            game.step(rng.choice(view.remaining) if rng.random() < noise else solver.choose(view))
        if game.over:
            continue
        view = game.view()
        letter = solver.choose(view)
        key = (view.pattern, view.wrong)
        if need[letter] and key not in seen:
            seen.add(key)
            need[letter] -= 1
            out.append((view, letter, game.word))
    if any(need.values()):
        raise RuntimeError(f"could not find enough states for {sorted(c for c, n in need.items() if n)}")
    rng.shuffle(out)
    return out


def build_teaching_file(per_letter=10, seed=0):
    """Write data/hangman_teaching.json: the registration examples, with the word each state came from."""
    rows = [
        {"word": w, "pattern": v.pattern, "wrong": list(v.wrong), "letter": letter}
        for v, letter, w in hangman_examples(per_letter, seed)
    ]
    write_teaching_rows(rows)
    return rows


def write_teaching_rows(rows):
    TEACHING_FILE.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")


def hangman_teaching():
    """[(view, letter)] from the committed file: 10 states for each of the 26 letters, none from a held-out word."""
    rows = json.loads(TEACHING_FILE.read_text())
    return [(HangmanView(r["pattern"], tuple(r["wrong"])), r["letter"]) for r in rows]


def teaching_examples(kind, n=None, seed=0):
    """The examples to register for a game: Hangman's committed 260, or `n` maze states from the BFS teacher."""
    return hangman_teaching() if kind == "hangman" else maze_examples(n or 40, seed)


def maze_examples(n, seed=0):
    """[(view, direction)]: `n` states from games played by the BFS teacher, one every few moves, from mazes whose seeds
    start at `10_000 + seed * 1000`, far from the evaluation seeds (0 upwards)."""
    out = []
    game_seed = 10_000 + seed * 1000
    while len(out) < n:
        game = MazeChase(game_seed)
        game_seed += 1
        while not game.over and len(out) < n:
            view = game.view()
            move = teacher_move(view)
            if game.tick % 3 == 1:
                out.append((view, move))
            game.step(move)
    return out


def registration_body(kind, examples, task_id=None, render=None):
    """The body of POST /v1/tasks; the examples are rendered as the player will render the states it is asked about."""
    return {
        "id": task_id or TASK_IDS[kind],
        "examples": [{"request": request_for(kind, view, render), "answer": answer} for view, answer in examples],
    }


def call(url, path, body=None, method=None, timeout=3600):
    """(status, parsed JSON) of one HTTP call; an HTTP error is returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode(errors="replace")


def register(url, kind, examples, task_id=None, render=None):
    """Register the task; returns the server's record (what was fitted and whether it applies)."""
    status, task = call(url, "/v1/tasks", registration_body(kind, examples, task_id, render))
    if status != 200:
        raise RuntimeError(f"registration refused ({status}): {task}")
    return task


def unregister(url, kind, task_id=None):
    call(url, f"/v1/tasks/{task_id or TASK_IDS[kind]}", method="DELETE")


def registered(url, kind, task_id=None):
    """Whether the server holds this game's task now."""
    status, body = call(url, "/v1/tasks")
    return status == 200 and any(t["id"] == (task_id or TASK_IDS[kind]) for t in body.get("tasks", []))


def question(kind):
    return QUESTIONS[kind], QUESTION_NAME


if __name__ == "__main__":
    if sys.argv[1:] != ["--rebuild"]:
        sys.exit("python -m gamelib.teach --rebuild    rewrite data/hangman_teaching.json")
    print(f"{len(build_teaching_file())} examples written to {TEACHING_FILE}")
