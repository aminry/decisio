# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Record a player's games as a GIF and an MP4 at the speed they were really played.

The caption under the frame names the player and the hardware and gives the median time per move over the recording.
Run it on the machine that runs the server (or pass --hardware), so that the latency shown is the server's and not a
network's. A recording is a few games in a row: one maze, or several Hangman words, because Hangman is over in well
under a second at server speed.

    python examples/games/record.py maze --player decisio --url http://127.0.0.1:8000 --seeds 3 \\
        --who "decisio server: Qwen3.6-35B-A3B-FP8" --hardware "1x RTX PRO 6000 Blackwell (96 GB)" \\
        --gif docs/media/maze.gif --mp4 docs/media/maze.mp4
    python examples/games/record.py hangman --player teacher --words 6 --gif /tmp/hangman.gif
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).parent))
from gamelib.adapter import render_arg  # noqa: E402
from gamelib.hangman import heldout_words  # noqa: E402
from gamelib.players import PLAYERS, make_game, make_player, play  # noqa: E402
from gamelib.record import Recorder, detect_hardware, write_gif, write_meta, write_mp4  # noqa: E402
from gamelib.teach import register, teaching_examples, unregister  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=["maze", "hangman"])
    ap.add_argument("--player", choices=PLAYERS, default="teacher")
    ap.add_argument("--url", default=None)
    ap.add_argument("--render", type=render_arg, default=None, help="how much the state says (docs/games.md)")
    ap.add_argument("--who", default=None, help="the caption's first line (default: the player's name)")
    ap.add_argument("--hardware", default=None, help="the caption's hardware (default: this machine's)")
    ap.add_argument("--seeds", default=None, help="maze: seeds, comma separated (default 3); hangman: ignored")
    ap.add_argument("--words", type=int, default=6, help="hangman: how many held-out words, from --first-word")
    ap.add_argument("--first-word", type=int, default=0, help="hangman: index of the first held-out word")
    ap.add_argument("--registered", action="store_true", help="hangman: register the teaching task before recording")
    ap.add_argument("--speed", type=float, default=1.0, help="1 is real time; 0.5 plays at half speed")
    ap.add_argument("--min-frame-ms", type=float, default=20.0, help="the shortest a frame stays (a GIF's floor)")
    ap.add_argument("--end-hold-ms", type=float, default=900.0, help="how long a finished game stays on screen")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--meta", default=None, help="write the recording's timeline and per-move records here (JSON)")
    ap.add_argument("--max-moves", type=int, default=400)
    a = ap.parse_args()
    if not (a.gif or a.mp4):
        ap.error("give --gif and/or --mp4")

    player = make_player(a.game, a.player, a.url, render=a.render)
    host = urlsplit(getattr(player, "url", "http://localhost")).hostname
    hardware = a.hardware or detect_hardware()
    if not a.hardware and host not in ("localhost", "127.0.0.1", "::1"):
        print(f"warning: the server is at {host} but the caption names this machine ({hardware}); pass --hardware")
    if a.game == "maze":
        games = [(int(s), f"seed {s}") for s in (a.seeds or "3").split(",")]
    else:
        words = heldout_words()[a.first_word : a.first_word + a.words]
        games = [(w, w) for w in words]
    if a.registered:
        task = register(player.url, "hangman", teaching_examples("hangman"))
        head, cal = task["head"].get("applied"), task["calibration"].get("applied")
        print(f"registered {task['id']}: head {head}, calibration {cal}")
    recorder, records = Recorder(a.game), []
    try:
        for i, (seed, name) in enumerate(games, 1):
            recorder.start_game(f"game {i} of {len(games)}" if a.game == "maze" else f"word {i} of {len(games)}")
            game = make_game(a.game, seed)
            records.append(play(game, player, a.game, seed, observer=recorder.observer, max_moves=a.max_moves))
            print(f"{name}: {records[-1].status} in {len(records[-1].moves)} moves")
    finally:
        if a.registered:
            unregister(player.url, "hangman")
    recorder.finish(a.end_hold_ms)
    who = a.who or player.name
    images, durations, floored = recorder.build(who, hardware, a.speed, a.min_frame_ms)
    meta = {
        "game": a.game,
        "player": player.describe(),
        "who": who,
        "hardware": hardware,
        "registered": a.registered,
        "render": a.render,
        "speed": a.speed,
        "frames": len(images),
        "frames_slowed_to_floor": floored,
        "min_frame_ms": a.min_frame_ms,
        "games": [
            {"seed": r.seed, "status": r.status, "score": r.score, "moves": len(r.moves), "detail": r.detail}
            for r in records
        ],
        "latency_ms": recorder.latencies(),
    }
    if a.gif:
        meta["gif"] = {"path": a.gif, **write_gif(images, durations, Path(a.gif))}
        print(f"{a.gif}: {meta['gif']['bytes'] / 1024:.0f} KB, {meta['gif']['frames']} frames")
    if a.mp4:
        meta["mp4"] = {"path": a.mp4, **write_mp4(images, durations, Path(a.mp4))}
        print(f"{a.mp4}: {Path(a.mp4).stat().st_size / 1024:.0f} KB")
    if a.meta:
        write_meta(Path(a.meta), meta)
    if floored:
        print(f"{floored} of {len(images)} frames were shorter than {a.min_frame_ms:.0f} ms and were slowed to it")


if __name__ == "__main__":
    main()
