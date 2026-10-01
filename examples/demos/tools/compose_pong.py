# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Put lanes recorded one model at a time on one page, and record it.

A card runs one model at a time, so each model's Pong lane is recorded in its own run (`measure_pong.py --only <id>`)
with the same seed: the same serve, the same rules, the game's randomness fixed by the seed. This merges those
recordings into one replay with one lane per model, which the page plays side by side as if they had run together, and
records the page as an MP4 and a GIF with every lane's median latency in the caption. The lanes' own latencies are
unchanged, so what is on screen is what each model measured on an idle card.

    python compose_pong.py --replay runs/a/replay_run1.json --replay runs/b/replay_run1.json --out runs/c/media
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from demo_recorder import record, to_gif, to_mp4
from demo_run import card_name
from measure_pong import PONG, serve_replay
from stats import git_state, write_files_json


def merge_replays(replays: list[dict]) -> dict:
    """One replay with every lane of every input, in order. The inputs must share a seed and have distinct lane ids."""
    if not replays:
        raise ValueError("no replays to merge")
    seeds = {r["seed"] for r in replays}
    if len(seeds) != 1:
        raise ValueError(f"the recordings must share a seed, got {sorted(seeds)}")
    lanes, seen = [], set()
    for r in replays:
        for lane in r["lanes"]:
            if lane["model"] in seen:
                raise ValueError(f"lane id {lane['model']!r} appears in two recordings")
            seen.add(lane["model"])
            lanes.append(lane)
    return {**replays[0], "lanes": lanes}


def lane_medians(replay: dict) -> list[tuple[str, float]]:
    out = []
    for lane in replay["lanes"]:
        lat = [s["latencyMs"] for s in lane["snapshots"] if s.get("latencyMs") is not None]
        out.append((lane["label"], float(np.median(lat)) if lat else float("nan")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="append", required=True, help="a replay_runN.json; repeat once per model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="pong")
    ap.add_argument("--port", type=int, default=3101)
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--card", default="")
    ap.add_argument("--size", default="1280x560")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge_replays([json.loads(Path(p).read_text()) for p in a.replay])
    (out / f"{a.name}_replay.json").write_text(json.dumps(merged))
    card = a.card or card_name()
    caption = " | ".join(f"{label} {ms:.0f} ms" for label, ms in lane_medians(merged)) + f" median | {card}"
    seconds = max(lane["snapshots"][-1]["t"] for lane in merged["lanes"]) / 1000 + 3
    width, height = (int(x) for x in a.size.split("x"))
    subprocess.run(["pnpm", "build"], cwd=PONG, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    server = serve_replay(out / f"{a.name}_replay.json", a.port)
    try:
        time.sleep(4)
        webm = record(
            f"http://127.0.0.1:{a.port}/?clean=1",
            out / a.name,
            seconds=seconds,
            size=(width, height),
            channel=a.channel or None,
        )
    finally:
        server.terminate()
        (PONG / "public" / "replay.json").unlink(missing_ok=True)
    to_mp4(webm, out / f"{a.name}.mp4", caption)
    gif = to_gif(webm, out / f"{a.name}.gif", caption, start_s=1.0)
    manifest = {
        "id": out.parent.name,
        "title": "Pong lanes recorded one model at a time, composed on one page",
        "date": time.strftime("%Y-%m-%d"),
        "submitted": False,
        "typesafe_api_calls": 0,
        "demo": "pong",
        "hardware": card,
        "code": git_state(Path(__file__).resolve().parents[3]),
        "command": " ".join(sys.argv),
        "seed": merged["seed"],
        "sources": a.replay,
        "lanes": [
            {"id": lane["model"], "label": lane["label"], "median_latency_ms": ms}
            for lane, (_label, ms) in zip(merged["lanes"], lane_medians(merged))
        ],
        "caption": caption,
        "files": {
            f"{a.name}_replay.json": "the merged replay the page played",
            f"{a.name}.webm, {a.name}.mp4, {a.name}.gif": "the recording (a caption strip on the MP4 and the GIF)",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write_files_json(out)
    print(webm, gif, gif.stat().st_size, caption)


if __name__ == "__main__":
    main()
