# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Play maze chase or Hangman: live in a window, or headless with the frames written to a directory.

The player is a human at the keyboard, the teacher, a random player, or a decisio server; the window
shows the probabilities behind each move and how long the decision took.

    python examples/games/play.py maze --seed 3                          # you play: arrow keys or WASD, Esc quits
    python examples/games/play.py maze --seed 3 --player teacher         # watch the BFS teacher
    python examples/games/play.py maze --seed 3 --player decisio         # a decisio server at --url
    python examples/games/play.py hangman --seed 7 --player teacher --headless --frames-dir /tmp/frames
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gamelib.adapter import render_arg  # noqa: E402
from gamelib.hangman import heldout_words, load_words  # noqa: E402
from gamelib.mazechase import MazeConfig  # noqa: E402
from gamelib.players import PLAYERS, make_game, make_player, play  # noqa: E402
from gamelib.render import FrameInfo, frame  # noqa: E402


def game_for(args):
    if args.game == "maze":
        return make_game("maze", args.seed, config=MazeConfig(chasers=args.chasers))
    if args.word:
        return make_game("hangman", args.word)
    return make_game("hangman", args.seed, heldout_words() if args.heldout else load_words())


def player_for(args):
    if args.player == "human":
        return None
    return make_player(args.game, args.player, args.url, seed=args.seed, render=args.render)


def info_for(args, player, move, n):
    title = f"{args.game} seed {args.seed}  {player.name if player else 'you'}"
    if move is None:
        return FrameInfo(title=title)
    return FrameInfo(
        title=title,
        probs=move.probs,
        chosen=move.action,
        latency_ms=move.latency_ms,
        move_no=n,
        caption=args.caption,
    )


def headless(args):
    player, game = player_for(args), game_for(args)
    out = Path(args.frames_dir) if args.frames_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    frames = 0

    def observe(game, move, record):
        nonlocal frames
        if out:
            frame(game, info_for(args, player, move, len(record.moves))).save(out / f"frame_{frames:04d}.png")
        frames += 1

    record = play(game, player, args.game, args.seed, observer=observe)
    lat = sorted(record.latencies)
    print(
        f"{args.game} seed {args.seed}: {record.status} after {len(record.moves)} moves"
        + (f", median {lat[len(lat) // 2]:.0f} ms/move" if lat else "")
        + (f", {frames} frames in {out}" if out else "")
    )


def live(args):
    from gamelib.window import Window, hangman_letter, maze_direction

    player = player_for(args)
    game = game_for(args)
    win = Window(f"decisio games: {args.game}", args.scale)
    heading, n, quit_, info = "right", 0, False, info_for(args, player, None, 0)
    if player:
        player.reset()
    try:
        while not quit_:
            win.show(frame(game, info, args.scale))
            keys, quit_ = win.wait(args.tick_ms if not player or game.over else 0)
            if "escape" in keys:
                break
            if game.over:
                if "space" in keys:  # the next seed
                    args.seed += 1
                    game, n, heading = game_for(args), 0, "right"
                    info = info_for(args, player, None, 0)
                continue
            if player:
                move = player.act(game)
                action = move.action
            elif args.game == "maze":
                heading = maze_direction(keys) or heading
                action, move = heading, None
            else:
                action, move = hangman_letter(keys), None
                if action is None or action in game.view().guessed:
                    continue
            game.step(action)
            n += 1
            info = info_for(args, player, move, n) if move else FrameInfo(title=info.title, chosen=action, move_no=n)
            if player:  # the tick: at least tick_ms between moves, the decision's own time counts
                keys, quit_ = win.wait(max(0.0, args.tick_ms - move.latency_ms))
                if "escape" in keys:
                    break
        win.show(frame(game, info, args.scale))
    finally:
        win.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=["maze", "hangman"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--player", choices=["human", *PLAYERS], default="human")
    ap.add_argument("--url", default=None, help="the decisio server (default: http://127.0.0.1:8000)")
    ap.add_argument(
        "--render",
        type=render_arg,
        default=None,
        help="how much the state says: the base, or observed (docs/games.md)",
    )
    ap.add_argument("--caption", default="", help="the line under the frame: hardware, model")
    ap.add_argument("--chasers", type=int, default=2, choices=[1, 2, 3], help="maze: how many chasers")
    ap.add_argument("--word", default=None, help="hangman: play this word instead of the seed's")
    ap.add_argument("--heldout", action="store_true", help="hangman: draw the seed's word from the 200 held-out words")
    ap.add_argument("--scale", type=int, default=2, help="window size, in multiples of 480x408")
    ap.add_argument("--tick-ms", type=int, default=180, help="live: least time per move; a human's maze keeps going")
    ap.add_argument("--headless", action="store_true", help="no window; play to the end")
    ap.add_argument("--frames-dir", default=None, help="headless: write a PNG per move here")
    a = ap.parse_args()
    if a.player == "human" and a.headless:
        ap.error("--headless needs a player other than human")
    (headless if a.headless else live)(a)


if __name__ == "__main__":
    main()
