# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Trajectory files: the record every demo run writes, and the only input the renderer reads.

A trajectory is one gzipped JSON-lines file per run and player (`trajectory.jsonl.gz`, browser frames beside it):

    {"header": {...}}                    the first line
    {"tick": 0, "t_wall": ..., ...}      one line per tick, in order
    {"end": {...}}                       the last line: the run's outcome as the demo scored it

The header carries the format version (`FORMAT`), the demo, the demo code's commit, the player's structured fields
(label, model ID, date, route, serving provider, card, padding, and what the server said about itself at the start),
the run's seed or task, and whatever static data the renderer needs (Pong's court, the driving map).

Every tick carries `t_wall` (Unix seconds, the client's clock when the tick happened) and the complete game state as
the demo held it after the tick. A tick at which the model was asked carries `decision`:

    {"request": <the /v1/systemone body exactly as sent>, "answer": <the server's answers, every option's probability>,
     "chosen": <the option applied>, "latency_ms": <the client's round trip>}

or, when the call failed, `{"request": ..., "error": "...", "latency_ms": ...}`. Ticks without a decision (a serve,
the simulation between decisions) carry `decision: null`. Browser ticks carry the step's screenshot (`frame`, a path
relative to the trajectory file, with its sha256) and the chosen element's box (`highlight`); the page is never
re-executed to draw it.

Nothing here simulates or re-drives anything: the renderer draws only what a trajectory holds.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

FORMAT = "decisio-demo-trajectory/1"
NAME = "trajectory.jsonl.gz"
REPO = Path(__file__).resolve().parents[3]

# The player's structured fields, in this order; a field the run does not know is null, never guessed.
PLAYER_FIELDS = ("label", "model_id", "date", "route", "provider", "card", "padding", "server")


def code_commit() -> str:
    """The demo code's commit: DEMO_CODE_COMMIT if set (a shipped tree has no .git), else git, else the tree's COMMIT
    file. A working tree with changes is marked `-dirty`."""
    if os.environ.get("DEMO_CODE_COMMIT"):
        return os.environ["DEMO_CODE_COMMIT"]
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if sha:
            dirty = subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    commit_file = REPO / "COMMIT"
    return commit_file.read_text().strip() if commit_file.exists() else "unknown"


def player_fields(spec: str | Path | dict | None, **known) -> dict:
    """The player's structured fields from a JSON file or dict (`--player`), completed by what the run knows itself
    (`known`, e.g. card=..., server=...). A field given in `spec` wins over `known`."""
    given = {}
    if isinstance(spec, dict):
        given = dict(spec)
    elif spec:
        given = json.loads(Path(spec).read_text())
    out = {k: given.get(k, known.get(k)) for k in PLAYER_FIELDS}
    out.update({k: v for k, v in given.items() if k not in out})
    return out


def header(demo: str, player: dict, run: dict, **static) -> dict:
    """The first line of a trajectory. `run` names what was played (seed, task, mode, rendering); `static` holds what
    the renderer needs that does not change per tick."""
    return {
        "format": FORMAT,
        "demo": demo,
        "code_commit": code_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "player": player,
        "run": run,
        **static,
    }


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write(path: str | Path, head: dict, ticks: Iterable[dict], end: dict | None = None) -> str:
    """Write a trajectory and return its sha256. The gzip header carries no name or time, so the same content gives
    the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write((dumps({"header": head}) + "\n").encode())
        for t in ticks:
            f.write((dumps(t) + "\n").encode())
        f.write((dumps({"end": end or {}}) + "\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str | Path) -> tuple[dict, list[dict], dict]:
    """(header, ticks, end) of a trajectory; refuses a file of another format."""
    head, ticks, end = None, [], {}
    for row in iter_rows(path):
        if "header" in row:
            head = row["header"]
        elif "end" in row:
            end = row["end"]
        else:
            ticks.append(row)
    if not head or head.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return head, ticks, end


def iter_rows(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def decisions(ticks: list[dict]) -> list[dict]:
    """The ticks at which the model answered (a failed call is not a decision)."""
    return [t for t in ticks if t.get("decision") and "answer" in t["decision"]]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
