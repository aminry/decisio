# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure the Pong demo (examples/demos/pong): every lane of a lanes file, over repeated seeded recordings.

Each run is `pnpm record` with a different seed: the lanes play the same serve at the same instant, each against its own
server, for `--seconds`. Reported per lane: decisions per second, p50 and p95 of the per-decision latency (the System
One or chat request, timed in the recorder), and the lane's score, the rallies it kept going (returns of the ball by its
paddle) and the points it conceded. Intervals are 95% bootstrap intervals over the runs; latencies are pooled. Every run
writes one trajectory per lane (`trajectories/run<N>_<lane>.jsonl.gz`, tools/trajectory.py): the complete engine state
of every tick with its wall-clock time, and each decision's request as sent, the full answer and the latency. Clips are
rendered from those afterwards (tools/render_pong.py); nothing here records video. A lane's structured fields (model
ID, date, route, provider, padding) come from its `player` object in the lanes file.

    python measure_pong.py --lanes-file lanes.json --runs 5 --seconds 45
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import trajectory
from demo_run import DemoRun, bootstrap, card_name, interval_text, latency_summary, server_health

PONG = Path(__file__).resolve().parents[1] / "pong"
# The court the engine plays on (pong/lib/game/types.ts; tests/unit/test_demo_pong_trajectory.py checks they agree)
COURT = {"w": 160, "h": 100, "paddle_h": 20, "ball_r": 1.5, "paddle_inset": 4, "paddle_step": 12, "points_to_win": 5}
REPO = Path(__file__).resolve().parents[3]


def lane_rows(replay: dict) -> dict:
    """Per lane: the latencies, returns, misses, duration and final score of one recording."""
    rows = {}
    for lane in replay["lanes"]:
        snaps = lane["snapshots"]
        lat = [s["latencyMs"] for s in snaps if s.get("latencyMs") is not None]
        returns = misses = 0
        for prev, nxt in zip(snaps, snaps[1:]):
            pvx, nvx = prev["ball"]["vx"], nxt["ball"]["vx"]
            if pvx and nvx and (pvx > 0) != (nvx > 0):
                returns += 1
            if prev["score"] != nxt["score"]:
                misses += 1
        duration = (snaps[-1]["t"] - snaps[0]["t"]) / 1000 if len(snaps) > 1 else 0.0
        rows[lane["model"]] = {
            "label": lane["label"],
            "latency_ms": lat,
            "decisions": len(lat),
            "duration_s": duration,
            "returns": returns,
            "misses": misses,
            "final_score": snaps[-1]["score"] if snaps else None,
        }
    return rows


def serve_replay(replay_path: Path, port: int):
    """`next start` on the production build with `replay_path` as public/replay.json."""
    (PONG / "public").mkdir(exist_ok=True)
    shutil.copy(replay_path, PONG / "public" / "replay.json")
    return subprocess.Popen(
        ["pnpm", "exec", "next", "start", "-p", str(port)],
        cwd=PONG,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # pnpm starts the server as a child; stop_server ends the whole group
    )


def stop_server(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=15)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes-file", required=True)
    ap.add_argument("--only", default="", help="comma-separated lane ids to run (one model at a time on a single card)")
    ap.add_argument("--label", default="Pong lanes", help="the name of the run (the lanes carry their own labels)")
    ap.add_argument("--out", default=None, help="run directory (default runs/<date>_demos-pong)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--seed0", type=int, default=20260917)
    ap.add_argument("--port", type=int, default=3100, help="unused; kept so older command lines still parse")
    ap.add_argument("--channel", default="chrome", help="unused; kept so older command lines still parse")
    ap.add_argument("--prompt-variant", default="default", help="PONG_PROMPT_VARIANT for the System One lanes")
    ap.add_argument("--card", default="")
    a = ap.parse_args()

    lanes = json.loads(Path(a.lanes_file).read_text())
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    if only:
        lanes = [lane for lane in lanes if lane["id"] in only]
    base_url = next((x["baseUrl"] for x in lanes if x["kind"] == "systemone"), "")
    out = Path(a.out) if a.out else REPO / "runs" / f"{time.strftime('%Y-%m-%d')}_demos-pong"
    card = a.card or card_name()
    run = DemoRun(out, "Pong lanes", "pong", a.label, base_url, hardware=card)
    players = {
        lane["id"]: trajectory.player_fields(
            lane.get("player"),
            label=lane["label"],
            card=card,
            server=server_health(lane["baseUrl"]) if lane["kind"] == "systemone" else None,
        )
        for lane in lanes
    }
    per_run = []
    for i in range(a.runs):
        seed = a.seed0 + i
        rdir = (out / f"run{i + 1}").resolve()
        env = dict(
            os.environ,
            PONG_LANES_FILE=str(Path(a.lanes_file).resolve()),
            RECORD_MS=str(int(a.seconds * 1000)),
            LANES=",".join(only),
            SEED=str(seed),
            OUT_DIR=str(rdir),
            PONG_DECISION_LOG=str(run.decision_log_path),
            PONG_PROMPT_VARIANT=a.prompt_variant,
            TRAJECTORY_DIR=str(rdir / "ticks"),
        )
        run.log(f"run {i + 1}/{a.runs}: seed {seed}, {a.seconds:.0f} s, {len(lanes)} lanes")
        subprocess.run(
            ["pnpm", "record"],
            cwd=PONG,
            env=env,
            check=True,
            stdout=open(out / "record.log", "a"),
            stderr=subprocess.STDOUT,
        )
        replay = json.loads((rdir / "replay.json").read_text())
        rows = lane_rows(replay)
        per_run.append({"seed": seed, "lanes": rows})
        for lane_id, r in rows.items():
            run.log(f"  {lane_id}: {r['decisions']} decisions, {r['returns']} returns, {r['misses']} misses")
            ticks_path = rdir / "ticks" / f"{lane_id}.ticks.jsonl"
            ticks = [json.loads(x) for x in ticks_path.read_text().splitlines() if x.strip()]
            head = trajectory.header(
                "pong",
                players[lane_id],
                {"lane": lane_id, "seed": seed, "run": i + 1, "seconds": a.seconds, "prompt_variant": a.prompt_variant},
                court=COURT,
            )
            end = {k: r[k] for k in ("decisions", "returns", "misses", "final_score")}
            end["status"] = ticks[-1]["state"]["status"] if ticks else None
            trajectory.write(out / "trajectories" / f"run{i + 1}_{lane_id}.jsonl.gz", head, ticks, end)
        (rdir / "replay.json").rename(out / f"replay_run{i + 1}.json")
        shutil.rmtree(rdir, ignore_errors=True)

    summary = {}
    for lane in lanes:
        lid = lane["id"]
        rows = [r["lanes"][lid] for r in per_run if lid in r["lanes"]]
        lat = [x for r in rows for x in r["latency_ms"]]
        rate = [r["decisions"] / r["duration_s"] for r in rows if r["duration_s"]]
        summary[lid] = {
            "label": lane["label"],
            "kind": lane["kind"],
            "base_url": lane["baseUrl"],
            "model": lane.get("model"),
            "runs": len(rows),
            "decisions_per_s": bootstrap(rate),
            "latency_ms": latency_summary(lat),
            "rallies_returns_per_run": bootstrap([r["returns"] for r in rows]),
            "points_conceded_per_run": bootstrap([r["misses"] for r in rows]),
        }
    run.write_json(
        "runs.json",
        [{**r, "lanes": {k: {**v, "latency_ms": v["latency_ms"]} for k, v in r["lanes"].items()}} for r in per_run],
    )
    md = [
        f"# {a.label}",
        "",
        f"{a.runs} seeded recordings of {a.seconds:.0f} s ({card}); every lane plays the same serve at the same "
        "instant against its own server; client and servers on the same machine. 95% bootstrap intervals over runs.",
        "",
        "| lane | decisions/s | latency p50 / p95 ms | rallies kept (returns per run) | points conceded per run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lid, s in summary.items():
        lat = s["latency_ms"]
        md.append(
            f"| {s['label']} | {interval_text(s['decisions_per_s'], 2)} | "
            f"{lat.get('p50', float('nan')):.0f} / {lat.get('p95', float('nan')):.0f} | "
            f"{interval_text(s['rallies_returns_per_run'], 1)} | {interval_text(s['points_conceded_per_run'], 1)} |"
        )
    run.finish(
        summary,
        "\n".join(md),
        {
            "lanes": [{k: v for k, v in lane.items() if k != "apiKey"} for lane in lanes],
            "prompt_variant": a.prompt_variant,
            "files": {
                "runs.json": "one row per run and lane: every decision's latency, returns, misses, duration, "
                "final score",
                "replay_run<N>.json": "the recording of each run, as the page replays it",
                "summary.json, summary.md": "the tables with their bootstrap intervals",
                "record.log": "the recorder's progress",
                "trajectories/run<N>_<lane>.jsonl.gz": "each run's trajectory per lane (tools/trajectory.py)",
            },
        },
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
