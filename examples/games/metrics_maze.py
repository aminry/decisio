# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Maze chase, measured: N seeded mazes per player, the same seeds for every player.

For each player: the win rate (maze cleared) and the mean score with 95% bootstrap intervals over the mazes, moves
into walls, agreement with the BFS teacher move by move, and the time per move. Paired differences against random and
against the teacher use the same mazes. The run is a record under --out (manifest.json, files.json, per-game rows),
resumable: a rerun skips the games already written.

    python examples/games/metrics_maze.py --out runs/2026-10-02_games-maze --games 100 \\
        --players teacher,random,random-legal,decisio --url http://127.0.0.1:8000 --hardware "1x RTX PRO 6000 (96 GB)"
    python examples/games/metrics_maze.py --out /tmp/maze --games 3 --players teacher,random      # a smoke run

`--register N` adds an arm for each decisio player that has a task registered from N teacher-labelled states first
(calibration only: a four-option question gets no head, docs/tasks.md); the task is deleted afterwards.
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gamelib.adapter import render_arg  # noqa: E402
from gamelib.mazechase import ORDER, MazeConfig  # noqa: E402
from gamelib.metrics import bootstrap, fmt_ci, latency_summary, md_table, parse_players, server_info  # noqa: E402
from gamelib.players import make_game, teacher_for  # noqa: E402
from gamelib.runner import Run, check_task_use, paired, reference_pairs  # noqa: E402
from gamelib.teach import maze_examples, register, registered, unregister  # noqa: E402

CODE = {d: d[0] for d in ORDER}  # u d l r
UNCODE = {v: k for k, v in CODE.items()}


def row_for(record, game):
    moves = record.moves
    return {
        "seed": record.seed,
        "status": record.status,
        "score": record.score,
        "ticks": record.detail["ticks"],
        "pellets_left": len(game.pellets),
        "actions": "".join(CODE[m.action] for m in moves),
        "teacher": "".join(CODE[m.teacher] for m in moves),
        "legal": "".join("1" if m.legal else "0" for m in moves),
        "latency_ms": [round(m.latency_ms, 2) for m in moves],
        "server_ms": [m.server_ms for m in moves] if any(m.server_ms is not None for m in moves) else None,
        "probs": [[round(m.probs.get(d, 0.0), 4) for d in ORDER] for m in moves] if moves and moves[0].probs else None,
        "task": sum(1 for m in moves if m.task),
    }


def summarise(rows, total_pellets):
    rows = sorted(rows, key=lambda r: r["seed"])
    moves = sum(len(r["actions"]) for r in rows)
    lat = [x for r in rows for x in r["latency_ms"]]
    server = [x for r in rows if r["server_ms"] for x in r["server_ms"]]
    agree = sum(a == t for r in rows for a, t in zip(r["actions"], r["teacher"]))
    bumps = sum(c == "0" for r in rows for c in r["legal"])
    return {
        "games": len(rows),
        "win_rate": bootstrap([r["status"] == "won" for r in rows]),
        "score": bootstrap([r["score"] for r in rows]),
        "pellets_eaten_share": bootstrap([r["score"] / 10 / total_pellets[r["seed"]] for r in rows]),
        "ticks": bootstrap([r["ticks"] for r in rows]),
        "statuses": {s: sum(r["status"] == s for r in rows) for s in ("won", "lost", "timeout")},
        "moves": moves,
        "wall_bump_rate": bumps / moves if moves else None,
        "teacher_agreement": agree / moves if moves else None,
        "latency_ms": latency_summary(lat),
        "server_ms": latency_summary(server) if server else None,
        "tasks_applied_moves": sum(r["task"] for r in rows),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the run's directory (runs/<date>_<name>)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--players", default="teacher,random,random-legal")
    ap.add_argument("--url", default=None, help="the server's address (default: the player's own: decisio :8000)")
    ap.add_argument(
        "--model-path", default=None, help="a local copy of the weights, for the models that load in-process"
    )
    ap.add_argument("--register", type=int, default=0, help="also play with a task registered from N teacher states")
    ap.add_argument("--render", type=render_arg, default=None, help="how much the state says: the base, or observed")
    ap.add_argument("--chasers", type=int, default=2, choices=[1, 2, 3], help="how many chasers")
    ap.add_argument(
        "--hunter", default="1110", help="the hunter's pace over four ticks: 1110 (default), 1010 half speed"
    )
    ap.add_argument("--hardware", default="", help="what the players ran on, for the record")
    ap.add_argument("--title", default="Maze chase: seeded games per player")
    a = ap.parse_args()

    run = Run(a.out, a.title, "maze", a.hardware)
    seeds = list(range(a.first_seed, a.first_seed + a.games))
    config = MazeConfig(chasers=a.chasers, hunter=a.hunter)
    total_pellets = {s: len(make_game("maze", s, config=config).pellets) for s in seeds}
    maker = partial(make_game, "maze", config=config)
    arms, registrations, infos = {}, {}, {}
    for name, player in parse_players(a.players, "maze", a.url, a.render, a.model_path):
        infos[name] = {"player": player.describe(), "server": server_info(player)}
        args = (maker, row_for, teacher_for("maze"))
        if player.describe().get("kind") == "decisio" and registered(player.url, "maze"):
            run.log(f"{name}: a task from an earlier session was registered; removed, so that the plain arm is plain")
            unregister(player.url, "maze")
        player.warm_up()
        arms[name] = run.play_arm(name, player, seeds, *args)
        check_task_use(name, arms[name], False)
        if a.register and player.describe().get("kind") == "decisio":
            examples = maze_examples(a.register)
            t0 = time.perf_counter()
            task = register(player.url, "maze", examples, render=a.render)
            cal = task["calibration"]
            registrations[name] = {
                "n_examples": len(examples),
                "seconds": round(time.perf_counter() - t0, 1),
                "calibration": cal,
                "head": task["head"],
            }
            run.log(f"{name}: task registered from {len(examples)} states; calibration {cal.get('applied')}")
            run.log(f"{name}: calibration reason: {cal.get('reason')}")
            try:
                player.warm_up()
                arms[f"{name}+task"] = run.play_arm(f"{name}+task", player, seeds, *args)
                check_task_use(f"{name}+task", arms[f"{name}+task"], True)
            finally:
                unregister(player.url, "maze")
        player.close()

    summary = {name: summarise(rows, total_pellets) for name, rows in arms.items()}
    metrics = {"score": lambda r: r["score"], "win": lambda r: r["status"] == "won"}
    pairs = reference_pairs(list(arms), ["random", "teacher"]) + [(n, n[:-5]) for n in arms if n.endswith("+task")]
    paired_rows = paired(arms, "seed", metrics, pairs)
    result = {
        "game": "maze",
        "config": config.__dict__,
        "render": a.render,
        "seeds": [seeds[0], seeds[-1]],
        "mean_pellets": sum(total_pellets.values()) / len(total_pellets),
        "arms": summary,
        "paired": paired_rows,
        "registrations": registrations,
    }
    table = [
        [
            name,
            s["games"],
            fmt_ci(s["win_rate"], pct=True),
            fmt_ci(s["score"], digits=0),
            pct(s["teacher_agreement"]),
            pct(s["wall_bump_rate"]),
            f"{s['latency_ms']['p50']:.1f} / {s['latency_ms']['p95']:.1f}" if s["latency_ms"].get("n") else "n/a",
        ]
        for name, s in summary.items()
    ]
    md = [
        f"# {a.title}",
        "",
        f"{a.games} seeded mazes per player (seeds {seeds[0]} to {seeds[-1]}), the same for every player; "
        "95% bootstrap intervals over mazes.",
        "",
        md_table(
            ["player", "games", "win rate", "mean score", "teacher agrees", "wall bumps", "ms per move p50 / p95"],
            table,
        ),
    ]
    if paired_rows:
        points = lambda d: {k: (v * 100 if v is not None else None) for k, v in d.items()}  # noqa: E731
        rows = [[k, fmt_ci(v["score"], digits=0), fmt_ci(points(v["win"]), digits=1)] for k, v in paired_rows.items()]
        md += [
            "",
            "Paired differences over the same mazes:",
            "",
            md_table(["pair", "score", "win rate (points)"], rows),
        ]
    files = {
        "games/<player>.jsonl.gz": "one row per maze: status, score, ticks, the moves played and the teacher's moves "
        "from the same states (u d l r), per-move wall time and probabilities",
        "summary.json, summary.md": "the tables with their bootstrap intervals",
        "log.txt": "progress",
        "files.json": "sizes and sha256",
    }
    manifest = {"render": a.render, "players": infos, "registrations": registrations, "files": files}
    run.finish(result, "\n".join(md), manifest)
    print("\n".join(md))


if __name__ == "__main__":
    main()
