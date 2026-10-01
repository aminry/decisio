# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The decisio side of the games: a state rendered to text, and the one question asked about it.

Each move is one `/v1/systemone` request with one choice question.
The state is the game as text; the question and its options do not change from move to move, so a task registered for
them (`POST /v1/tasks`) applies to every move of the game.
Maze chase offers four options with descriptions; Hangman offers the 26 letters as bare keys with no descriptions.

How much the state says is a choice, named by a render: the base state, or `observed`, which adds observations the
game can compute and nothing else. A render is a `+`-joined list of features:

  maze     grid (the base: the maze, positions, open moves, recent moves, score)
           local      what lies in the next two cells each way
           distances  for each open move: whether it eats, and the steps from the cell it reaches to the nearest
                      pellet and to the nearest chaser
           observed   = local+distances
  hangman  base (the length, the pattern, the wrong guesses)
           guessed    every letter already played
           count      how many dictionary words still fit the pattern and the wrong guesses
           observed   = guessed+count

The state never carries a verdict, a recommendation, advice in the question, or a score per option: the model decides.
"""

from .hangman import ALPHABET, EntropySolver, HangmanView
from .mazechase import DIRS, ORDER, MazeView, distances, neighbours

QUESTION_NAME = (
    "decision"  # one id for every model: some servers read this id only, and none reads it as an instruction
)

MAZE_QUESTION = {
    "type": "choice",
    "instructions": "Which way should the player (P) move next to eat the pellets and stay clear of the chasers?",
    "criteria": {
        "up": "Move one cell up, to the row above",
        "down": "Move one cell down, to the row below",
        "left": "Move one cell left, to the column before",
        "right": "Move one cell right, to the column after",
    },
}

HANGMAN_QUESTION = {
    "type": "choice",
    "instructions": "Which letter should be guessed next? Pick a letter that has not been guessed yet.",
    "criteria": {letter: None for letter in ALPHABET},
}

QUESTIONS = {"maze": MAZE_QUESTION, "hangman": HANGMAN_QUESTION}

FEATURES = {
    "maze": {"local", "distances"},
    "hangman": {"guessed", "count"},
}
ALIASES = {"observed": {"maze": "local+distances", "hangman": "guessed+count"}}
BASE = {"maze": "grid", "hangman": "base"}


def parse_render(kind, render=None):
    """The set of features a render names; raises on a name it does not know."""
    names = [n for n in (render or BASE[kind]).split("+") if n]
    names = [m for n in names for m in (ALIASES[n][kind].split("+") if n in ALIASES else [n])]
    extra = set(names) - FEATURES[kind] - {BASE[kind]}
    if extra:
        raise ValueError(
            f"unknown {kind} render {sorted(extra)}; the base is {BASE[kind]!r}, features: {sorted(FEATURES[kind])}, "
            "or 'observed'"
        )
    return set(names) - {BASE[kind]}


def render_arg(text):
    """argparse type: a render name checked against both games' features (the game is not known yet)."""
    for kind in BASE:
        try:
            parse_render(kind, text)
            return text
        except ValueError:
            continue
    import argparse

    raise argparse.ArgumentTypeError(
        f"unknown render {text!r}; maze: {sorted(FEATURES['maze'])}, hangman: {sorted(FEATURES['hangman'])}, "
        "or observed, joined with +"
    )


def question_for(kind, render=None):
    """The question does not depend on the render: only the state says more."""
    parse_render(kind, render)
    return QUESTIONS[kind]


# ---- maze ------------------------------------------------------------------------------------------------------------


def cell_name(view: MazeView, pos) -> str:
    r, c = pos
    if view.grid[r][c] == "#":
        return "wall"
    if any((cr, cc) == pos for cr, cc, _ in view.chasers):
        return "chaser"
    return "pellet" if pos in view.pellets else "empty"


def next_cells(view: MazeView) -> str:
    """What the player would meet going each way: the next cell and the one after it (nothing beyond a wall)."""
    lines = []
    for d in ORDER:
        dr, dc = DIRS[d]
        first = (view.player[0] + dr, view.player[1] + dc)
        name = cell_name(view, first)
        if name == "wall":
            lines.append(f"{d}: wall")
        else:
            lines.append(f"{d}: {name}, then {cell_name(view, (first[0] + dr, first[1] + dc))}")
    return "Next cells:\n" + "\n".join(lines)


def steps(n):
    return f"{n} step" + ("" if n == 1 else "s")


def move_report(view: MazeView) -> str:
    """What each open move leads to: whether it eats, and the steps from the cell it reaches to the nearest pellet and
    to the nearest chaser (shortest paths through the maze). Observations only."""
    to_pellet = distances(view.grid, list(view.pellets)) if view.pellets else {}
    to_chaser = distances(view.grid, [(r, c) for r, c, _ in view.chasers])
    lines = {}
    for d, cell in neighbours(view.grid, view.player):
        near = to_chaser.get(cell)
        text = (
            "eats a pellet" if cell in view.pellets else f"no pellet here, nearest pellet {steps(to_pellet[cell])} away"
        )
        text += "; chaser on that cell" if near == 0 else f"; nearest chaser {steps(near)} away"
        lines[d] = text
    return "Moves:\n" + "\n".join(f"{d}: {lines.get(d, 'wall')}" for d in ORDER)


def maze_state(view: MazeView, render="grid") -> str:
    """The maze as the model reads it: the grid with the player (P), chasers (C), pellets (.), walls (#) and empty cells
    (-), then positions, the moves that are open, the last moves and the score; plus what the render's features add."""
    features = parse_render("maze", render)
    cells = [list(row.replace(" ", "-")) for row in view.grid]
    for r, c in view.pellets:
        cells[r][c] = "."
    for r, c, _ in view.chasers:
        cells[r][c] = "C"
    cells[view.player[0]][view.player[1]] = "P"
    grid = "\n".join("".join(row) for row in cells)
    chasers = "; ".join(f"row {r} col {c}" for r, c, _ in view.chasers)
    recent = ", ".join(view.recent) or "none yet"
    text = (
        "Maze chase. You are P. Eat every pellet (.). A chaser (C) that reaches you ends the game. "
        "# is a wall and - is an empty cell.\n"
        "Rows count down from 0 at the top, columns count right from 0.\n\n"
        f"{grid}\n\n"
        f"You: row {view.player[0]} col {view.player[1]}\n"
        f"Chasers: {chasers}\n"
        f"Open moves: {', '.join(view.legal)}\n"
        f"Last moves: {recent}\n"
        f"Score: {view.score}. Pellets left: {len(view.pellets)}. Tick {view.tick}."
    )
    if "local" in features:
        text += "\n" + next_cells(view)
    if "distances" in features:
        text += "\n" + move_report(view)
    return text


# ---- hangman ---------------------------------------------------------------------------------------------------------

_solver = None


def solver():
    global _solver
    if _solver is None:
        _solver = EntropySolver()
    return _solver


def hangman_state(view: HangmanView, render="base") -> str:
    """Hangman as the model reads it: the word length, the pattern, and the wrong guesses so far; plus what the
    render's features add."""
    features = parse_render("hangman", render)
    wrong = ", ".join(view.wrong) or "none"
    text = (
        "Hangman. Guess the hidden word one letter at a time.\n"
        f"Word length: {view.length}\n"
        f"Pattern: {' '.join(view.pattern)}\n"
        f"Wrong guesses ({len(view.wrong)} of {view.max_wrong}): {wrong}"
    )
    if "guessed" in features:
        text += "\nLetters already guessed: " + (", ".join(sorted(view.guessed)) or "none")
    if "count" in features:
        text += f"\nWords that still fit: {len(solver().candidates(view))}"
    return text


def state_text(kind, view, render=None) -> str:
    return maze_state(view, render or "grid") if kind == "maze" else hangman_state(view, render or "base")


def request_for(kind, view, render=None, name=QUESTION_NAME) -> dict:
    """The `/v1/systemone` request for one move; `name` is the question's id (a server may read one id only)."""
    return {"state": state_text(kind, view, render), "questions": {name: question_for(kind, render)}}
