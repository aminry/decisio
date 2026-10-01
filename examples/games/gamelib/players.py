# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Players, and the move loop that plays a game with one.

A player turns the game it is given into a `Move`: an action, the probabilities behind it when there are any, and the
wall time the decision took.
`SystemOnePlayer` asks a server over `/v1/systemone`: a decisio server, or any server that speaks that wire format;
`TeacherPlayer` and `RandomPlayer` need no server.
`play` runs a game to its end and returns a `GameRecord` that holds every move.
"""

import http.client
import json
import random
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .adapter import QUESTION_NAME, request_for
from .hangman import EntropySolver, Hangman, random_letter
from .mazechase import MazeChase, random_move, teacher_move


@dataclass
class Move:
    action: str
    probs: dict | None = None  # the option -> probability behind the action, as the player reported it
    latency_ms: float = 0.0  # wall time of the decision, request included
    server_ms: float | None = None  # what the server says it spent, when it says (decisio's x-decisio-server-ms)
    raw: str | None = None  # the option the server chose, when the player changed it (Hangman masks guessed letters)
    task: str | None = None  # x-decisio-tasks: the registered task that answered, if any


@dataclass
class MoveRecord:
    n: int
    action: str
    latency_ms: float
    server_ms: float | None
    legal: bool  # a maze move into a wall is not; every Hangman move is
    teacher: str | None  # what the teacher would have played from the same state
    raw_illegal: bool = False  # Hangman: the server's own first choice was a letter already guessed
    probs: dict | None = None
    task: str | None = None


@dataclass
class GameRecord:
    game: str
    seed: object
    player: str
    status: str = "playing"
    score: int = 0
    moves: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def latencies(self):
        return [m.latency_ms for m in self.moves]


class Player:
    name = "player"

    def describe(self):
        """What a record says about this player."""
        return {"name": self.name}

    def reset(self, game_seed=None):
        """Called before each game, with the game's seed."""

    def warm_up(self):
        """Called once before an arm is measured, so that loading a model is not counted as a move."""

    def close(self):
        """Called once after the player's arms: free what the player holds on a server."""

    def act(self, game) -> Move:
        raise NotImplementedError


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


class TeacherPlayer(Player):
    name = "teacher"

    def __init__(self, kind, solver=None):
        self.kind = kind
        self.solver = solver or (EntropySolver() if kind == "hangman" else None)

    def describe(self):
        return {"name": self.name, "kind": "teacher", "policy": "bfs" if self.kind == "maze" else "entropy"}

    def act(self, game):
        view = game.view()
        action, ms = timed(lambda: teacher_move(view) if self.kind == "maze" else self.solver.choose(view))
        return Move(action, None, ms)


class RandomPlayer(Player):
    """Uniform over the four directions (a wall bump is possible, as for a model), or over the unguessed letters.
    `legal_only` restricts the maze to open moves."""

    def __init__(self, kind, seed=0, legal_only=False):
        self.kind, self.seed, self.legal_only = kind, seed, legal_only
        self.name = "random-legal" if legal_only else "random"
        self.rng = random.Random(seed)

    def describe(self):
        return {"name": self.name, "kind": "random", "seed": self.seed, "legal_only": self.legal_only}

    def reset(self, game_seed=None):
        self.rng = random.Random(f"{self.seed}:{game_seed}")  # a stream per game, so games are not copies of each other

    def act(self, game):
        view = game.view()
        action = random_move(view, self.rng, self.legal_only) if self.kind == "maze" else random_letter(view, self.rng)
        return Move(action, None, 0.0)


class SystemOnePlayer(Player):
    """Asks a `/v1/systemone` server one question per move and plays the option it chooses.

    `model` is sent in the request when given (a decisio server ignores it). In Hangman the server's probabilities
    are masked to the letters not yet guessed and renormalised before the choice, so a repeated letter is never played
    (the move records when the server's own first choice would have been one). A maze move is played as answered."""

    def __init__(self, kind, url, model=None, name=None, timeout=600, render=None, question_name=QUESTION_NAME):
        self.kind, self.url, self.model, self.timeout = kind, url.rstrip("/"), model, timeout
        self.render, self.question_name = render, question_name
        self.name = name or model or "decisio"
        parts = urlsplit(self.url)
        self.host, self.port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        self.conn = None

    def describe(self):
        return {
            "name": self.name,
            "kind": "decisio",
            "url": self.url,
            "model": self.model,
            "render": self.render,
        }

    def _post(self, body):
        data = json.dumps(body).encode()
        for attempt in (0, 1):  # a kept-alive connection may have been closed by the server: one reconnect
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                self.conn.request("POST", "/v1/systemone", data, {"Content-Type": "application/json"})
                resp = self.conn.getresponse()
                raw = resp.read()
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw
            except (http.client.HTTPException, ConnectionError, OSError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise

    def ask(self, request):
        """(parsed answer, headers, wall ms) for one request; raises on an HTTP error."""
        body = dict(request)
        if self.model:
            body["model"] = self.model
        t0 = time.perf_counter()
        status, headers, raw = self._post(body)
        ms = (time.perf_counter() - t0) * 1000
        if status != 200:
            raise RuntimeError(f"{self.url}/v1/systemone answered {status}: {raw[:300]!r}")
        return json.loads(raw)["answers"][self.question_name], headers, ms

    def warm_up(self, requests=3):
        game = make_game(self.kind, 0) if self.kind == "maze" else make_game(self.kind, "planet")
        for _ in range(requests):
            self.ask(request_for(self.kind, game.view(), self.render, self.question_name))

    def distribution(self, view):
        """(probabilities over the options, wall ms, headers) for a state, with no game behind it."""
        answer, headers, ms = self.ask(request_for(self.kind, view, self.render, self.question_name))
        return answer["probabilities"], ms, headers

    def act(self, game):
        view = game.view()
        answer, headers, ms = self.ask(request_for(self.kind, view, self.render, self.question_name))
        probs, choice = answer["probabilities"], answer["choice"]
        raw = None
        if self.kind == "hangman":
            allowed = {k: p for k, p in probs.items() if k in view.remaining}
            top = max(allowed, key=allowed.get) if allowed and sum(allowed.values()) > 0 else view.remaining[0]
            if choice != top:
                raw = choice
            choice = top
        server_ms = headers.get("x-decisio-server-ms")
        return Move(
            choice,
            probs,
            ms,
            float(server_ms) if server_ms is not None else None,
            raw,
            headers.get("x-decisio-tasks"),
        )


DEFAULT_URLS = {"decisio": "http://127.0.0.1:8000"}
MODEL_PLAYERS = ("cygnet", "winnow", "decider", "jev-omni", "nimble")  # the comparison models (gamelib/models.py)
PLAYERS = ("teacher", "random", "random-legal", "decisio", *MODEL_PLAYERS)


def make_player(kind, which, url=None, model=None, seed=0, name=None, render=None, path=None):
    """A player by name: teacher, random, random-legal, decisio (a decisio server), or one of the comparison models
    (`url` is its server's address; `path` a local copy of the weights for the two that load in this process)."""
    if which in MODEL_PLAYERS:
        from .models import make_model_player

        return make_model_player(kind, which, url, render, path)
    if which == "teacher":
        return TeacherPlayer(kind)
    if which in ("random", "random-legal"):
        return RandomPlayer(kind, seed, legal_only=which == "random-legal")
    if which in DEFAULT_URLS:
        return SystemOnePlayer(kind, url or DEFAULT_URLS[which], model, name=name, render=render)
    raise ValueError(f"unknown player {which!r}; choose from {', '.join(PLAYERS)}")


def teacher_for(kind, solver=None):
    """A function game -> the teacher's action from the game's current view."""
    if kind == "maze":
        return lambda game: teacher_move(game.view())
    solver = solver or EntropySolver()
    return lambda game: solver.choose(game.view())


def make_game(kind, seed, words=None, config=None):
    """A maze for a seed; for Hangman `seed` is either a word or an integer that picks one from `words`."""
    if kind == "maze":
        return MazeChase(seed, config) if config else MazeChase(seed)
    return Hangman(seed) if isinstance(seed, str) else Hangman.from_seed(seed, words)


def play(game, player, kind, seed=None, teacher=None, observer=None, max_moves=1000):
    """Play `game` to its end with `player`; returns the GameRecord.

    `teacher` (a function game -> action, see `teacher_for`) labels every move with what the teacher would have played
    from the same state. `observer(game, move, record)` is called once before the first move (move None) and after each
    move, for a window or a recorder."""
    record = GameRecord(kind, seed, player.name)
    player.reset(seed)
    if observer:
        observer(game, None, record)
    while not game.over and len(record.moves) < max_moves:
        view = game.view()
        label = teacher(game) if teacher else None
        move = player.act(game)
        legal = move.action in view.legal if kind == "maze" else True
        game.step(move.action)
        record.moves.append(
            MoveRecord(
                n=len(record.moves) + 1,
                action=move.action,
                latency_ms=move.latency_ms,
                server_ms=move.server_ms,
                legal=legal,
                teacher=label,
                raw_illegal=move.raw is not None,
                probs=move.probs,
                task=move.task,
            )
        )
        if observer:
            observer(game, move, record)
    record.status = game.status
    record.score = game.score if kind == "maze" else 0
    record.detail = {"ticks": game.tick} if kind == "maze" else {"word": game.word, "wrong": len(game.wrong)}
    return record
