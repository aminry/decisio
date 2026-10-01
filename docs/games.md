<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Watching a decision model play

Two small games, and a decisio server (or any other `/v1/systemone` server) playing them one question at a time.
They are here to make two things visible: a model answering a closed question in a few tens of milliseconds, and what teaching it from a few labelled examples does (`docs/tasks.md`).
Everything runs from `examples/games/`; the engines, the teachers and the recorder need no GPU.

Each move is one request.
The game as text is the state, the question is always the same, and the answer is a probability for every option.
Nothing is generated and nothing is parsed out of prose.

## Maze chase

A 15 by 9 maze with a pellet in every open cell and two chasers (a third is available with `--chasers 3`).
Each tick you move one cell up, down, left or right (into a wall you stay put and the tick still passes), then every chaser whose turn it is moves.
Eat every pellet and you win; a chaser that reaches your cell, or swaps places with you, ends the game; a game lasts at most 300 ticks.
The hunter goes straight for you and misses a turn one tick in four; the wanderer and the drifter move one tick in four, mostly at random and now and then toward you.
A game is a pure function of its seed and the moves played.

What the model reads (`gamelib/adapter.py`):

```
Maze chase. You are P. Eat every pellet (.). A chaser (C) that reaches you ends the game. # is a wall and - is an empty cell.
Rows count down from 0 at the top, columns count right from 0.

###############
#...#.........#
#.#.#.###.###.#
#..C...P......#
...
You: row 3 col 7
Chasers: row 3 col 3; row 1 col 9
Open moves: down, left, right
Last moves: up, up, left, left, up
Score: 40. Pellets left: 57. Tick 15.
```

The question is `choice` with four options, each with a description ("Move one cell up, to the row above" and so on).
The player plays the option the server chooses, a wall bump included; the metrics count how often that happens.

The teacher is a breadth-first search (`teacher_move`): the first step of the shortest path to the nearest pellet that no chaser can reach first, or, with none, to the safest reachable cell.
It knows the chasers' move schedules, and assumes every chaser comes straight at any cell it could reach, at its scheduled pace.
It never steps where a chaser could be first, but it can still be pinned in a corridor, which is how it loses.

## What each state contains

A one-pass model answers from what its state shows, so what a state contains is stated here exactly.
A state may carry the game's rules and observations the game can compute: positions, distances, what lies in each direction, counts.
It never carries a verdict, a recommendation, advice in the question, or a score per option; the model makes the decision.
The question and its options are the same on every move and for every state of a game.

Each game has two states, chosen with `--render` (in `play.py`, `record.py` and both metrics scripts):

| Game | State | Contents |
| --- | --- | --- |
| maze | `grid` (the default) | a short statement of the rules; the maze as a grid of `#` walls, `-` empty cells, `.` pellets, `C` chasers and `P`; your row and column; the chasers' rows and columns; the open moves; the last five moves; the score, the pellets left and the tick |
| maze | `observed` | `grid`, and then for each direction the next two cells (`down: pellet, then empty`, or `up: wall`), and for each open move whether it eats a pellet, the steps from the cell it reaches to the nearest pellet, and the steps from that cell to the nearest chaser (`left: no pellet here, nearest pellet 3 steps away; nearest chaser 5 steps away`) |
| Hangman | `base` (the default) | a short statement of the rules; the word length; the pattern with `_` for hidden letters; the wrong guesses so far and the number allowed |
| Hangman | `observed` | `base`, the letters already guessed, and how many dictionary words still fit the pattern and the wrong guesses |

`observed` has no instruction in it, no label on a move, no ranking and no count per letter.
`maze` features can be combined (`grid+local`, `grid+distances`); `local` is the next-two-cells block alone and `distances` the per-move block alone.
A task registered with a render applies to that render's states, so register and play with the same `--render`.

## Hangman

The word is one of 14,094 (ENABLE words that are among the 20,000 most frequent in Peter Norvig's counts, 4 to 12 letters, `examples/games/gamelib/data/words.txt`); six wrong guesses lose.
200 of them, drawn with a fixed seed, are held out: the evaluation plays exactly these, and no example used to teach the model comes from them (`heldout_words.txt`).

What the model reads: the word length, the revealed pattern and the wrong guesses so far.

```
Hangman. Guess the hidden word one letter at a time.
Word length: 7
Pattern: _ a _ _ m a n
Wrong guesses (2 of 6): e, t
```

The question is `choice` over the 26 letters as bare keys, with no descriptions, so the option list is fixed and a registered task applies to every move.
The player masks the letters already guessed and renormalises before it picks; the metrics count how often the server's own first choice was one of them.

The teacher is the classic entropy solver (`EntropySolver`): among the dictionary words still consistent with the state, guess the letter whose answer, the positions it occupies or none, has the highest entropy.
It is given the same state the model gets, never the word.

**Teaching.** The model is frozen.
`POST /v1/tasks` fits, from labelled examples of one recurring question, a per-task calibration and, for 10 or more options with at least 5 examples of each, a linear head on the model's hidden state; each is kept only if cross-validation on those examples shows a gain.
Hangman's 26 options make it a task the head applies to.
The 260 teaching examples (10 for each letter, `gamelib/data/hangman_teaching.json`) are states from solver games on words outside the held-out 200, each labelled with the solver's choice; rare letters are found by stopping games at random moves and letting a quarter of the lead-up moves be random.
`python -m gamelib.teach --rebuild` writes the file again.

## Running it

From `examples/games/`, with the package installed (`uv sync --extra dev --frozen`) and, for the window, `pip install pygame-ce` (it is not a dependency of decisio):

```
python play.py maze                              # you play: arrow keys or WASD, Esc quits
python play.py maze --player teacher             # the BFS teacher, in a window
python play.py hangman --player teacher          # type letters yourself without --player
python play.py maze --player decisio --url http://127.0.0.1:8000
python play.py maze --seed 3 --player teacher --headless --frames-dir /tmp/frames   # no window: a PNG per move
```

The window shows the probabilities behind each move and how long the decision took.
Players: `teacher`, `random`, `random-legal` (open moves only, the maze) and `decisio` (a decisio server).

Teaching a server from the command line:

```
python -c "import sys; sys.path.insert(0, '.'); from gamelib.teach import *; \
  print(register('http://127.0.0.1:8000', 'hangman', teaching_examples('hangman'))['head']['applied'])"
```

Registration is per server instance and in memory (`docs/tasks.md`); delete it with `DELETE /v1/tasks/hangman-letter`.

## Recording

```
python record.py maze --player decisio --url http://127.0.0.1:8000 --seeds 3 \
    --who "decisio server: Qwen3.6-35B-A3B-FP8" --hardware "1x RTX PRO 6000 Blackwell (96 GB)" \
    --gif ../../docs/media/maze.gif --mp4 ../../docs/media/maze.mp4 --meta /tmp/maze.json
python record.py hangman --player decisio --words 8 --registered --gif ../../docs/media/hangman_taught.gif
```

A recording is the player's games at the speed it really played them: a position stays on screen for as long as the player took to decide the next move, the caption gives the player, the hardware and the median time per move, and a clock shows the real time elapsed.
Run it on the machine that runs the server (or pass `--hardware`), so that the latency shown is the server's and not a network's.
A GIF cannot show a frame for less than 20 ms (browsers show shorter ones at 100 ms), so faster moves are slowed to that; `--meta` counts the frames it affected.
Hangman is over in a fraction of a second at server speed, so a recording plays several words in a row.
The GIF is reduced in colours and then in size until it is under 3 MB.
The MP4 needs `ffmpeg`.

## Measuring

```
python metrics_maze.py --out ../../runs/<date>_games-maze --games 100 --players teacher,random,random-legal,decisio \
    --url http://127.0.0.1:8000 --register 40 --hardware "1x RTX PRO 6000 Blackwell (96 GB)"
python metrics_hangman.py --out ../../runs/<date>_games-hangman --players teacher,random,decisio \
    --url http://127.0.0.1:8000 --hardware "1x RTX PRO 6000 Blackwell (96 GB)"
```

For each player the maze script plays the same N seeded mazes and reports the win rate (maze cleared), the mean score (10 per pellet), moves into walls, agreement with the teacher move by move, and the time per move.
The Hangman script plays the 200 held-out words once and reports the win rate, the mean number of wrong guesses, agreement with the solver, and the time per move.
For each decisio player it plays a second arm with the Hangman task registered, on the same words, so the difference is paired; a plain arm that was answered by a leftover task, or a registered one that was not, stops the run.
Intervals are 95% percentile bootstrap intervals over games (10,000 resamples, seed 0); paired differences resample the same games for both arms.
A run is resumable: a rerun skips the games already written.

**Calibration against the solver.** The solver plays all 200 words and its states are put to each player: the player's probabilities over the letters not yet guessed, renormalised, against the solver's choice.
The table gives how often the player's top letter is the solver's, the mean probability of the solver's letter and its log loss, the Brier score, the calibration error (ECE, ten bins) of the top letter's confidence as a predictor of agreeing with the solver, and the share of raw probability that sat on letters already guessed.
Intervals are bootstrap intervals over words.
This measures agreement with one reasonable strategy, not the probability of the letter being in the word.

A record under `runs/` holds `manifest.json` (hardware, commit, players, registration fits), `files.json` (every file's sha256), `summary.json` and `summary.md`, and the per-game rows with every move's time and probabilities.

## The comparison models

Decisio is compared with the best open decision models on JevBench v1.5.4, each served on the same GPU by its own published recipe and prompt format, one model at a time, plus random and the teacher.
Every model gets the same state and the same question; nothing is fitted or tuned for the games, and each runs at its published revision and temperature (`gamelib/models.py` holds the pins; a run's manifest repeats them).

| Model | Weights and revision | Served as | Licence |
| --- | --- | --- | --- |
| Decisio | `Qwen/Qwen3.6-35B-A3B-FP8`, the quickstart's `--model-class hidden-readout` | the Decisio server (vLLM 0.30.0) | Apache-2.0 |
| Cygnet | `google/gemma-4-12B-it` @ `707f0a3b`, frozen, letter readout, T = 3.4 | the recipe's `decision_server.py` on vLLM 0.30.0 (`github.com/blockbrain-ai/cygnet-recipe` @ `3cf591c6`) | Apache-2.0 weights, MIT shim |
| Winnow-12B | `EldanRing/Winnow-12B` @ `b6ac22b0`, `Winnow-12B-Q8_0.gguf` | the author's `winnow-inference` server (patched llama.cpp) @ `6c2b3c04` | Apache-2.0 |
| decider-4b v2 | `Mapika/decider-4b` @ `7ab294cb` | the author's `decider.serve` (decider-ai 1.4.0 or later) | Apache-2.0 |
| Jev-Omni | `akhilaaa3/Jev-Omni` @ `5addda86` | the author's `jev_omni.py` classes, in the measuring process | Apache-2.0 (declared in the card) |
| Nimble 9B | `bespokelabs/Bespoke-Nimble-9B` @ `bd792f44` on `Qwen/Qwen3.5-9B` | the author's `ParallelScorer`, in the measuring process | Apache-2.0 |

The first three servers and decider answer `/v1/systemone`, so they are asked over HTTP like Decisio; Jev-Omni and Nimble 9B publish library code and no server, so their latency is the call itself.
Before a model is measured, `conformance.py` compares its player with the model's own tooling on ten fixed game states (five maze, five Hangman): the same choice on every state and every probability within a tolerance; a model that fails is not measured.
Hangman's 26 letters are above Jev-Omni's established 20 options, above the 20 letters one Cygnet pass reads (its server reads 26 in three passes, which makes its latency about three times a single read), and in the range where decider-4b's training coverage is not documented; the tables say so beside those models.

## Results

Pending: the measurements on the GPU server are made in the next session and are not in this change yet.
They will give, with their records:

- the win rate of every model on the plain observation states (maze `grid` and `observed`, Hangman `base` and `observed`), over the same seeded mazes and the same 200 held-out words, with 95% bootstrap intervals and the time per move on the same card;
- the teaching result for Decisio: Hangman with 10 solver examples per letter registered through `POST /v1/tasks`, and the maze with the teacher's moves registered for per-task calibration, each against the same server without registration on the same games, with the paired difference.

A game where teaching does not help is reported with its numbers.
