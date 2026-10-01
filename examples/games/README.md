<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Games: watch a decision model play

Maze chase and Hangman, played one `/v1/systemone` question at a time, with a live window, a recorder, and measurements.
`docs/games.md` is the guide: the rules, what the model reads, how to run, record and measure, and the results with their records.
Nothing here needs a GPU except the server being played against; the CPU stand-in (`--backend hf`) serves the plumbing, not the measurements.

## Files

| File | What it is |
| --- | --- |
| `play.py` | Live in a window (pygame; a human, the teacher, random or a decisio server) or headless with a PNG per move |
| `record.py` | A player's games as a GIF (under 3 MB) and an MP4 at the speed they were played, with hardware and latency in the caption |
| `metrics_maze.py` | N seeded mazes per player: win rate, score, wall bumps, teacher agreement, time per move, bootstrap intervals |
| `metrics_hangman.py` | The 200 held-out words per player, before and after registering the Hangman task, and the letter probabilities against the solver |
| `gamelib/mazechase.py`, `gamelib/hangman.py` | The engines, deterministic under a seed, and the teachers (BFS; entropy solver) |
| `gamelib/adapter.py`, `gamelib/players.py` | The state as text, the fixed question, the players and the move loop |
| `gamelib/teach.py` | The teaching examples (`data/hangman_teaching.json`) and registration through `POST /v1/tasks` |
| `gamelib/render.py`, `gamelib/window.py`, `gamelib/record.py` | Frames drawn with Pillow from primitives (no assets), the pygame window, the encoders |
| `gamelib/metrics.py`, `gamelib/runner.py` | Intervals, calibration against the solver, the run record |
| `gamelib/data/` | The Hangman dictionary (14,094 words), its 200 held-out words, the teaching examples; `build_words.py` rebuilds the first two |

## Quick start

```
uv sync --extra dev --frozen
uv pip install pygame-ce                           # only for the window
python examples/games/play.py maze --player teacher
python examples/games/play.py hangman --player teacher --headless
```

The tests are `tests/unit/test_games_*.py`; the fast tier includes a short game of each kind against the CPU stand-in.
