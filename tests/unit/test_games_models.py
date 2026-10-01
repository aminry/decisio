# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The comparison models' players of examples/games, against stand-ins (no weights, no GPU).

C1  the registry names every model with weights, a pinned revision, a recipe, a licence and a serving path
C2  a model behind its own /v1/systemone server is asked the same question under the id `decision`, with the server's
    model alias when it has one, and answers are read back as probabilities per option
C3  Jev-Omni's player shows each option as the recipe does (`key: description`, or the key) and maps the author's
    answer back to the keys; Nimble's builds the author's schema (type enum, the choices, a description per choice)
C4  an in-process player masks guessed Hangman letters, records that the model's own first choice was one, and times
    the call; `make_player` knows every model by name
"""

import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "games"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import TOLERANCE, compare, conformance_states, run  # noqa: E402
from gamelib.adapter import HANGMAN_QUESTION, MAZE_QUESTION, QUESTION_NAME  # noqa: E402
from gamelib.hangman import ALPHABET, Hangman  # noqa: E402
from gamelib.mazechase import ORDER, MazeChase  # noqa: E402
from gamelib.models import MODELS, JevOmniPlayer, NimblePlayer, option_texts  # noqa: E402
from gamelib.players import MODEL_PLAYERS, make_player  # noqa: E402
from test_games_adapter import Fake  # noqa: E402


@pytest.fixture
def fake():
    Fake.seen = []
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", Fake
    server.shutdown()


def test_c1_registry():
    assert set(MODELS) == set(MODEL_PLAYERS) == {"cygnet", "winnow", "decider", "jev-omni", "nimble"}
    for spec in MODELS.values():
        assert {"label", "weights", "revision", "recipe", "licence", "serving"} <= set(spec)
        assert len(spec["revision"]) == 40 and int(spec["revision"], 16) >= 0
    assert MODELS["decider"]["revision"].startswith("7ab294cb") and MODELS["nimble"]["revision"].startswith("bd792f4")


def test_c2_http_models(fake):
    url, handler = fake
    handler.answer = lambda body: {c: (0.5 if c == "e" else 0.5 / 25) for c in ALPHABET}
    for which in ("cygnet", "winnow", "decider"):
        player = make_player("hangman", which, url)
        assert player.describe()["revision"] == MODELS[which]["revision"] and player.name == MODELS[which]["label"]
        move = player.act(Hangman("seed"))
        body = handler.seen[-1]
        assert list(body["questions"]) == [QUESTION_NAME] and body["questions"][QUESTION_NAME] == HANGMAN_QUESTION
        assert move.action == "e" and abs(sum(move.probs.values()) - 1) < 1e-9
        assert body.get("model") == MODELS[which].get("alias")
    assert MODELS["winnow"]["alias"] == "Winnow-12B"


class StubJevOmni:
    def predict(self, *, state, question, options, **kw):
        self.last = (state, question, options)
        raw = [0.1 + i for i in range(len(options))]
        total = sum(raw)
        return {"probabilities": {o: r / total for o, r in zip(options, raw)}}


class StubNimble:
    def score(self, context, schema, score_fields=()):
        self.last = (context, schema)
        keys = schema["decision"]["choices"]
        raw = [1.0 + i for i in range(len(keys))]
        total = sum(raw)
        return {"fields": {"decision": {"probabilities": {k: r / total for k, r in zip(keys, raw)}}}}


def test_c3_in_process_players_render_the_options_as_their_recipes_do():
    maze = MAZE_QUESTION
    assert option_texts(maze)[0] == "up: Move one cell up, to the row above"
    assert option_texts(HANGMAN_QUESTION) == list(ALPHABET)
    omni = JevOmniPlayer("maze")
    omni.engine = StubJevOmni()
    probs = omni.probabilities(MazeChase(3).view())
    assert list(probs) == list(ORDER) and abs(sum(probs.values()) - 1) < 1e-9 and max(probs, key=probs.get) == "right"
    state, question, options = omni.engine.last
    assert state.startswith("Maze chase.") and question == MAZE_QUESTION["instructions"]
    assert options == option_texts(MAZE_QUESTION)
    nim = NimblePlayer("hangman")
    nim.engine = StubNimble()
    probs = nim.probabilities(Hangman("seed").view())
    context, schema = nim.engine.last
    field = schema["decision"]
    assert field["type"] == "enum" and field["choices"] == list(ALPHABET) and field["description"]
    assert field["choice_descriptions"] == {c: c for c in ALPHABET}  # no description: the key itself
    assert max(probs, key=probs.get) == "z"
    nim_maze = NimblePlayer("maze")
    nim_maze.engine = StubNimble()
    nim_maze.probabilities(MazeChase(3).view())
    assert nim_maze.engine.last[1]["decision"]["choice_descriptions"]["up"] == MAZE_QUESTION["criteria"]["up"]


def test_c4_in_process_play_masks_guessed_letters_and_times_the_call():
    nim = NimblePlayer("hangman")
    nim.engine = StubNimble()  # always prefers z
    game = Hangman("zebra")
    game.step("z")
    move = nim.act(game)
    assert move.action == "y" and move.raw == "z" and move.latency_ms >= 0 and set(move.probs) == set(ALPHABET)
    probs, ms, headers = nim.distribution(game.view())
    assert max(probs, key=probs.get) == "z" and ms >= 0 and headers == {}
    for which in MODEL_PLAYERS:
        assert (
            make_player("maze", which).which
            if which in ("jev-omni", "nimble")
            else make_player("maze", which, "http://x")
        )
    with pytest.raises(ValueError, match="unknown player"):
        make_player("maze", "gpt")


def test_c5_conformance_states_and_comparison():
    states = conformance_states()
    assert [k for k, _ in states] == ["maze"] * 5 + ["hangman"] * 5
    assert states == conformance_states()  # deterministic
    assert len({v.pattern for k, v in states if k == "hangman"}) == 5 and any(
        v.tick > 0 for k, v in states if k == "maze"
    )
    a, b = {"x": 0.7, "y": 0.3}, {"x": 0.69, "y": 0.31}
    assert compare(a, b, 0.02) == (True, pytest.approx(0.01))
    assert compare(a, {"x": 0.4, "y": 0.6}, 0.02)[0] is False

    class Ours:
        kind = "maze"

        def distribution(self, view):
            keys = list(MAZE_QUESTION["criteria"]) if self.kind == "maze" else list(ALPHABET)
            probs = {k: 1 / len(keys) for k in keys}
            probs[keys[0]] += 0.1
            return probs, 1.0, {}

    def reference(kind, view):
        keys = list(MAZE_QUESTION["criteria"]) if kind == "maze" else list(ALPHABET)
        probs = {k: 1 / len(keys) for k in keys}
        probs[keys[0]] += 0.1
        return probs

    rows = run(Ours(), reference, TOLERANCE["decider"])
    assert len(rows) == 10 and all(r["ok"] and r["max_abs_diff"] == 0 for r in rows)
    wrong = run(
        Ours(), lambda kind, view: {k: (1.0 if i == 1 else 0.0) for i, k in enumerate(reference(kind, view))}, 0.02
    )
    assert not any(r["ok"] for r in wrong)
