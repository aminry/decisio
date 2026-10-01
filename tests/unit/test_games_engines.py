# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The two game engines of examples/games: deterministic under a seed, rules as documented, teachers that play well,
and frames that are the same for the same state.

  G1  maze layouts are connected and fully walled, and the same seed gives the same game, move for move
  G2  maze rules: a wall bump passes the tick, eating scores, a caught player loses, a cleared maze wins, a timeout ends
  G3  the BFS teacher clears a majority of 40 seeded mazes and a uniformly random player none
  G4  Hangman rules, and the dictionary: 200 distinct held-out words, all in the dictionary
  G5  the entropy teacher solves nearly all of the held-out words, a random player almost none; it is given no word
  G6  frames: the same state draws the same bytes, at the documented size, for both games, including a finished one
"""

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GAMES = ROOT / "examples" / "games"
sys.path.insert(0, str(GAMES))

from gamelib.hangman import (  # noqa: E402
    ALPHABET,
    EntropySolver,
    Hangman,
    HangmanView,
    dev_words,
    heldout_words,
    load_words,
    random_letter,
)
from gamelib.mazechase import (  # noqa: E402
    DIRS,
    ORDER,
    MazeChase,
    MazeConfig,
    distances,
    random_move,
    teacher_move,
)
from gamelib.render import HEIGHT, WIDTH, FrameInfo, frame  # noqa: E402


def play_maze(seed, policy, config=MazeConfig()):
    game = MazeChase(seed, config)
    while not game.over:
        game.step(policy(game.view()))
    return game


def test_g1_layout_and_determinism():
    for seed in range(30):
        game = MazeChase(seed)
        grid = game.view().grid
        assert all(row[0] == row[-1] == "#" for row in grid) and set(grid[0]) == set(grid[-1]) == {"#"}
        reach = distances(grid, [game.player])
        open_cells = {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == " "}
        assert set(reach) == open_cells, "every open cell is reachable"
        assert len(game.chasers) == 2 and game.player not in game.pellets
        assert all(c.pos in open_cells for c in game.chasers)
    first, again = play_maze(5, teacher_move), play_maze(5, teacher_move)
    assert first.moves == again.moves and first.score == again.score and first.status == again.status
    assert MazeChase(5).view().grid != MazeChase(6).view().grid
    rng1, rng2 = random.Random(1), random.Random(1)
    g1, g2 = play_maze(9, lambda v: random_move(v, rng1)), play_maze(9, lambda v: random_move(v, rng2))
    assert g1.moves == g2.moves and [c.pos for c in g1.chasers] == [c.pos for c in g2.chasers]


def test_g2_rules():
    game = MazeChase(0, MazeConfig(chasers=1))
    walls = [d for d in ORDER if d not in game.view().legal]
    before = game.view()
    result = game.step(walls[0])
    assert result["bumped"] and game.player == before.player and game.tick == 1
    legal = game.view().legal
    pellets = len(game.pellets)
    result = game.step(legal[0])
    assert result["ate"] and game.score == 10 and len(game.pellets) == pellets - 1
    with pytest.raises(ValueError):
        game.step("sideways")
    # a chaser on the player's cell ends the game; a cleared maze wins; the tick limit ends it
    lost = MazeChase(0, MazeConfig(chasers=1))
    lost.chasers[0].pos = (lost.player[0], lost.player[1] - 1)
    while not lost.over:
        lost.step("left")
    assert lost.status == "lost"
    won = MazeChase(0, MazeConfig(chasers=1))
    nxt = won.view().legal[0]
    dr, dc = DIRS[nxt]
    won.pellets = {(won.player[0] + dr, won.player[1] + dc)}  # one pellet left, right next to the player
    won.step(nxt)
    assert won.status == "won"
    slow = MazeChase(0, MazeConfig(chasers=1, max_ticks=3))
    slow.chasers[0].pos = max(distances(slow.grid, [slow.player]).items(), key=lambda kv: kv[1])[0]
    for _ in range(3):
        slow.step(slow.view().legal[0])
    assert slow.status == "timeout"
    with pytest.raises(ValueError):
        slow.step("up")


def test_g2_pace():
    game = MazeChase(3, MazeConfig(chasers=1))
    assert game.view().hunter == "1110"
    # the hunter's pace is part of the rules and of what the teacher is told
    slow = MazeChase(3, MazeConfig(chasers=1, hunter="1010"))
    assert slow.view().hunter == "1010" and slow.pace("hunter") == "1010" and slow.pace("wanderer") == "1000"
    start = slow.chasers[0].pos
    moved = []
    for _ in range(4):
        before = slow.chasers[0].pos
        slow.step(slow.view().legal[0])
        moved.append(slow.chasers[0].pos != before)
        if slow.over:
            break
    assert moved[:2] == [True, False] or slow.over, (start, moved)  # moves on ticks 1 and 3 only
    wins = sum(play_maze(s, teacher_move, MazeConfig(chasers=1)).status == "won" for s in range(30))
    assert wins >= 24  # one chaser: the teacher clears nearly every maze


def test_g3_teacher_beats_random():
    teacher = Counter(play_maze(s, teacher_move).status for s in range(40))
    rng = random.Random(0)
    rand = Counter(play_maze(s, lambda v: random_move(v, rng)).status for s in range(40))
    assert teacher["won"] >= 20, teacher
    assert rand["won"] == 0, rand


def test_g4_hangman_rules_and_words():
    game = Hangman("seed")
    assert game.view() == HangmanView("____", (), 6)
    assert game.step("e") is True and game.view().pattern == "_ee_"
    assert game.step("z") is False and game.view().wrong == ("z",)
    with pytest.raises(ValueError):
        game.step("e")
    with pytest.raises(ValueError):
        game.step("E")
    for letter in "qjxkv":
        game.step(letter)
    assert game.status == "lost" and game.over
    with pytest.raises(ValueError):
        game.step("a")
    done = Hangman("aa")
    done.step("a")
    assert done.status == "won"
    words, held = load_words(), heldout_words()
    assert len(held) == 200 == len(set(held)) and set(held) <= set(words)
    dev = dev_words()
    assert len(dev) == 100 == len(set(dev)) and set(dev) <= set(words) and not set(dev) & set(held)
    assert all(w.isalpha() and w.islower() and 4 <= len(w) <= 12 for w in words)
    assert Hangman.from_seed(3, words).word == Hangman.from_seed(3, words).word
    assert HangmanView("_ee_", ("z",)).remaining == [c for c in ALPHABET if c not in "ez"]


def test_g5_entropy_teacher():
    words = load_words()
    solver = EntropySolver(words)
    view = HangmanView("______", ())
    scores = solver.scores(view)
    assert len(scores) == 26 and max(scores, key=scores.get) == solver.choose(view)
    assert solver.choose(view) in "eaisrt"
    cands = solver.candidates(HangmanView("_a____", ("e",)))
    assert cands and all(w[1] == "a" and "e" not in w and w.count("a") == 1 and len(w) == 6 for w in cands)
    held = heldout_words()[:60]
    wins = 0
    for w in held:
        game = Hangman(w)
        while not game.over:
            game.step(solver.choose(game.view()))
        wins += game.status == "won"
    assert wins / len(held) > 0.8
    rng = random.Random(0)
    rwins = 0
    for w in held:
        game = Hangman(w)
        while not game.over:
            game.step(random_letter(game.view(), rng))
        rwins += game.status == "won"
    assert rwins <= 3
    assert not hasattr(HangmanView, "word")  # the teacher is given a view, which holds no word


def digest(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


def test_g6_frames_are_a_function_of_the_state():
    maze = play_maze(2, teacher_move)
    info = FrameInfo(title="t", caption="c", probs={"up": 0.1, "down": 0.2, "left": 0.3, "right": 0.4}, chosen="right")
    img = frame(maze, info)
    assert img.size == (WIDTH, HEIGHT) and digest(img) == digest(frame(maze, info))
    assert digest(img) != digest(frame(MazeChase(2), info))
    hang = Hangman("absolutely")
    top = FrameInfo(probs={c: 1 / 26 for c in ALPHABET}, chosen="e")
    before = digest(frame(hang, top))
    hang.step("e")
    assert digest(frame(hang, top)) != before
    for c in "zqxjkvw":
        if not hang.over:
            hang.step(c)
    assert hang.over and frame(hang, top).size == (WIDTH, HEIGHT)
    assert frame(hang, top, scale=2).size == (WIDTH * 2, HEIGHT * 2)
