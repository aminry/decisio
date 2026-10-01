# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Hangman, measured: the 200 held-out words, once per player, and the letter probabilities against the solver.

For each player: the win rate and the mean number of wrong guesses with 95% bootstrap intervals over words, how often
the server's first choice was a letter already guessed, agreement with the entropy solver move by move, and the time per
move. For each decisio player a second arm is played with the task registered from the 260 teaching examples (10 per
letter, none from a held-out word): the same words, so the difference is paired.

Calibration: the solver plays all 200 words and its states (about 1,600) are put to each player; the player's
probabilities over the unguessed letters are compared with the solver's choice (agreement, probability of the solver's
letter, log loss, Brier score, and the calibration error of the top letter's confidence as a predictor of agreeing).
Everything is a record under --out and a rerun resumes.

    python examples/games/metrics_hangman.py --out runs/2026-10-02_games-hangman \\
        --players teacher,random,decisio --url http://127.0.0.1:8000 --hardware "1x RTX PRO 6000 (96 GB)"
    python examples/games/metrics_hangman.py --out /tmp/hangman --words 5 --players teacher,random     # a smoke run
"""

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gamelib.adapter import render_arg  # noqa: E402
from gamelib.hangman import ALPHABET, EntropySolver, Hangman, HangmanView, dev_words, heldout_words  # noqa: E402
from gamelib.metrics import (  # noqa: E402
    bootstrap,
    calibration_stats,
    cluster_bootstrap,
    fmt_ci,
    gunzip_file,
    gzip_file,
    latency_summary,
    masked,
    md_table,
    parse_players,
    read_jsonl,
    server_info,
)
from gamelib.players import make_game, teacher_for  # noqa: E402
from gamelib.runner import Run, check_task_use, paired, reference_pairs  # noqa: E402
from gamelib.teach import hangman_teaching, register, registered, unregister  # noqa: E402


def row_for(record, game):
    moves = record.moves
    return {
        "word": record.detail["word"],
        "status": record.status,
        "wrong": record.detail["wrong"],
        "letters": "".join(m.action for m in moves),
        "teacher": "".join(m.teacher for m in moves),
        "repeat_first_choice": sum(m.raw_illegal for m in moves),
        "latency_ms": [round(m.latency_ms, 2) for m in moves],
        "server_ms": [m.server_ms for m in moves] if any(m.server_ms is not None for m in moves) else None,
        "probs": [[round(m.probs.get(c, 0.0), 4) for c in ALPHABET] for m in moves]
        if moves and moves[0].probs
        else None,
        "task": sum(1 for m in moves if m.task),
    }


def summarise(rows):
    rows = sorted(rows, key=lambda r: r["word"])
    moves = sum(len(r["letters"]) for r in rows)
    lat = [x for r in rows for x in r["latency_ms"]]
    server = [x for r in rows if r["server_ms"] for x in r["server_ms"]]
    agree = sum(a == t for r in rows for a, t in zip(r["letters"], r["teacher"]))
    return {
        "games": len(rows),
        "win_rate": bootstrap([r["status"] == "won" for r in rows]),
        "wrong_guesses": bootstrap([r["wrong"] for r in rows]),
        "moves": moves,
        "repeat_first_choice_rate": sum(r["repeat_first_choice"] for r in rows) / moves if moves else None,
        "teacher_agreement": agree / moves if moves else None,
        "latency_ms": latency_summary(lat),
        "server_ms": latency_summary(server) if server else None,
        "tasks_applied_moves": sum(r["task"] for r in rows),
    }


def teacher_states(words, solver):
    """The states the solver meets playing each word: [{"word", "pattern", "wrong", "solver"}]."""
    out = []
    for w in words:
        game = Hangman(w)
        while not game.over:
            view = game.view()
            letter = solver.choose(view)
            out.append({"word": w, "pattern": view.pattern, "wrong": list(view.wrong), "solver": letter})
            game.step(letter)
    return out


def calibrate_arm(run, name, player, states):
    """Put every solver state to the player; resumable. Returns [{"i", "probs"}] in state order."""
    path = run.out / "calibration" / (name.replace(":", "_").replace("/", "_") + ".jsonl")
    path.parent.mkdir(exist_ok=True)
    gunzip_file(path)
    rows = read_jsonl(path)
    done = {r["i"] for r in rows}
    t0 = time.perf_counter()
    with open(path, "a") as f:
        for i, s in enumerate(states):
            if i in done:
                continue
            probs, ms, headers = player.distribution(HangmanView(s["pattern"], tuple(s["wrong"])))
            row = {
                "i": i,
                "probs": [round(probs.get(c, 0.0), 5) for c in ALPHABET],
                "ms": round(ms, 2),
                "task": 1 if headers.get("x-decisio-tasks") else 0,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if len(rows) % 200 == 0:
                run.log(f"{name}: calibration {len(rows)}/{len(states)} states, {time.perf_counter() - t0:.0f} s")
    return sorted(rows, key=lambda r: r["i"])


def calibration_of(states, rows):
    clusters = {}
    for s, r in zip(states, rows):
        view = HangmanView(s["pattern"], tuple(s["wrong"]))
        q, wasted = masked(dict(zip(ALPHABET, r["probs"])), view.remaining)
        clusters.setdefault(s["word"], []).append({"solver": s["solver"], "q": q, "wasted": wasted})
    flat = [x for items in clusters.values() for x in items]
    stats = calibration_stats(flat)
    for key in ("agreement", "nll", "ece"):
        stats[key + "_ci"] = cluster_bootstrap(clusters, lambda items, k=key: calibration_stats(items)[k], n=2000)
    return stats


def uniform_reference(states):
    """What a player with no information would score: the same probability on every unguessed letter."""
    remaining = [len(HangmanView(s["pattern"], tuple(s["wrong"])).remaining) for s in states]
    return {
        "n": len(states),
        "agreement": sum(1 / n for n in remaining) / len(remaining),
        "p_solver": sum(1 / n for n in remaining) / len(remaining),
        "nll": sum(math.log(n) for n in remaining) / len(remaining),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the run's directory (runs/<date>_<name>)")
    ap.add_argument("--words", type=int, default=200, help="how many of the held-out words (default all 200)")
    ap.add_argument("--players", default="teacher,random")
    ap.add_argument("--url", default=None, help="the server's address (default: the player's own: decisio :8000)")
    ap.add_argument(
        "--model-path", default=None, help="a local copy of the weights, for the models that load in-process"
    )
    ap.add_argument("--render", type=render_arg, default=None, help="how much the state says: the base, or observed")
    ap.add_argument(
        "--dev", action="store_true", help="the 100 development words instead of the held-out 200 (iteration)"
    )
    ap.add_argument("--no-task", action="store_true", help="do not play the registered arm for decisio players")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--hardware", default="", help="what the players ran on, for the record")
    ap.add_argument("--title", default="Hangman: the held-out words per player, before and after registration")
    a = ap.parse_args()

    run = Run(a.out, a.title, "hangman", a.hardware)
    words = (dev_words() if a.dev else heldout_words())[: a.words]
    solver = EntropySolver()
    states = teacher_states(words, solver)
    (run.out / "teacher_states.json").write_text(json.dumps(states) + "\n")
    run.log(f"{len(words)} held-out words; the solver's own games give {len(states)} states")
    args = (partial(make_game, "hangman"), row_for, teacher_for("hangman", solver))
    arms, calib, registrations, infos = {}, {}, {}, {}

    def measure(name, player):
        with_task = name.endswith("+task")
        player.warm_up()
        arms[name] = run.play_arm(name, player, words, *args)
        check_task_use(name, arms[name], with_task)
        if not a.no_calibration and hasattr(player, "distribution"):
            rows = calibrate_arm(run, name, player, states)
            check_task_use(
                name + " (calibration states)", [{"task": r["task"], "latency_ms": [0]} for r in rows], with_task
            )
            calib[name] = calibration_of(states, rows)
            run.log(f"{name}: agrees with the solver on {calib[name]['agreement'] * 100:.1f}% of {len(states)} states")

    for name, player in parse_players(a.players, "hangman", a.url, a.render, a.model_path):
        infos[name] = {"player": player.describe(), "server": server_info(player)}
        if player.describe().get("kind") == "decisio" and registered(player.url, "hangman"):
            run.log(f"{name}: a task from an earlier session was registered; removed, so that the plain arm is plain")
            unregister(player.url, "hangman")
        measure(name, player)
        if not a.no_task and player.describe().get("kind") == "decisio":
            examples = hangman_teaching()
            t0 = time.perf_counter()
            task = register(player.url, "hangman", examples, render=a.render)
            registrations[name] = {
                "n_examples": len(examples),
                "seconds": round(time.perf_counter() - t0, 1),
                "calibration": task["calibration"],
                "head": task["head"],
            }
            head, cal = task["head"].get("applied"), task["calibration"].get("applied")
            run.log(f"{name}: registered {len(examples)} examples in {registrations[name]['seconds']} s")
            run.log(f"{name}: head applied {head}, calibration applied {cal}")
            try:
                measure(f"{name}+task", player)
            finally:
                unregister(player.url, "hangman")
        player.close()

    summary = {name: summarise(rows) for name, rows in arms.items()}
    metrics = {"win": lambda r: r["status"] == "won", "wrong": lambda r: r["wrong"]}
    pairs = reference_pairs(list(arms), ["random", "teacher"]) + [(n, n[:-5]) for n in arms if n.endswith("+task")]
    paired_rows = paired(arms, "word", metrics, pairs)
    result = {
        "game": "hangman",
        "render": a.render,
        "words": len(words),
        "teacher_states": len(states),
        "arms": summary,
        "paired": paired_rows,
        "calibration": {"uniform": uniform_reference(states), **calib},
        "registrations": registrations,
    }
    table = [
        [
            name,
            s["games"],
            fmt_ci(s["win_rate"], pct=True),
            fmt_ci(s["wrong_guesses"], digits=2),
            pct(s["teacher_agreement"]),
            pct(s["repeat_first_choice_rate"]),
            f"{s['latency_ms']['p50']:.1f} / {s['latency_ms']['p95']:.1f}" if s["latency_ms"].get("n") else "n/a",
        ]
        for name, s in summary.items()
    ]
    md = [
        f"# {a.title}",
        "",
        f"{len(words)} held-out words, each played once per player; 95% bootstrap intervals over words.",
        "",
        md_table(
            [
                "player",
                "words",
                "win rate",
                "wrong guesses",
                "solver agrees",
                "repeat first choice",
                "ms per move p50 / p95",
            ],
            table,
        ),
    ]
    if paired_rows:
        points = lambda d: {k: (v * 100 if v is not None else None) for k, v in d.items()}  # noqa: E731
        rows = [[k, fmt_ci(points(v["win"]), digits=1), fmt_ci(v["wrong"], digits=2)] for k, v in paired_rows.items()]
        md += [
            "",
            "Paired differences over the same words:",
            "",
            md_table(["pair", "win rate (points)", "wrong guesses"], rows),
        ]
    if calib:
        u = result["calibration"]["uniform"]
        rows = [
            ["uniform over unguessed letters", u["n"], pct(u["agreement"]), f"{u['p_solver']:.3f}", f"{u['nll']:.2f}"]
        ]
        rows[0] += ["", "", ""]
        for name, c in calib.items():
            rows.append(
                [
                    name,
                    c["n"],
                    fmt_ci(c["agreement_ci"], pct=True),
                    f"{c['p_solver']:.3f}",
                    fmt_ci(c["nll_ci"], digits=2),
                    f"{c['brier']:.3f}",
                    fmt_ci(c["ece_ci"], digits=3),
                    pct(c["mass_on_guessed"]),
                ]
            )
        head = [
            "player",
            "states",
            "top = solver's",
            "p(solver's letter)",
            "log loss",
            "Brier",
            "ECE",
            "mass on guessed",
        ]
        md += ["", f"Letter probabilities against the solver's choice on its own {len(states)} states:", ""]
        md.append(md_table(head, rows))
    files = {
        "games/<player>.jsonl.gz": "one row per word: status, wrong guesses, the letters played and the solver's "
        "from the same states, per-move wall time and the 26 probabilities",
        "calibration/<player>.jsonl.gz": "the 26 probabilities for each of the solver's states",
        "teacher_states.json": "the solver's states: word, pattern, wrong guesses, its choice",
        "summary.json, summary.md": "the tables with their bootstrap intervals",
        "log.txt": "progress",
        "files.json": "sizes and sha256",
    }
    for f in (run.out / "calibration").glob("*.jsonl"):
        gzip_file(f)
    run.finish(
        result,
        "\n".join(md),
        {"render": a.render, "dev_words": a.dev, "players": infos, "registrations": registrations, "files": files},
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
