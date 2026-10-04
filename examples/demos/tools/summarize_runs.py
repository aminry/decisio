# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""One table per demo across the models' run records.

    python summarize_runs.py runs/2026-10-01_demos-decisio runs/2026-10-01_demos-decider ...

Reads `<run>/pong/summary.json`, `<run>/fsd/summary.json`, `<run>/fsd-lockstep/summary.json` and
`<run>/ultrafast/summary.json` where they exist and prints Markdown: decisions per second, p50 and p95 per-decision
latency, and each demo's own score, with the 95% bootstrap intervals the records carry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from demo_run import interval_text


def load(run: Path, demo: str):
    path = run / demo / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def lat(s: dict) -> str:
    return f"{s['p50']:.0f} / {s['p95']:.0f}" if s.get("n") else "n/a"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    out = []

    rows = []
    for run in runs:
        s = load(run, "pong")
        for lane in (s or {}).values():
            rows.append(
                [
                    lane["label"],
                    interval_text(lane["decisions_per_s"], 2),
                    lat(lane["latency_ms"]),
                    interval_text(lane["rallies_returns_per_run"], 1),
                    interval_text(lane["points_conceded_per_run"], 1),
                ]
            )
    if rows:
        out += [
            "### Pong (the lane alone, 45 s per seed)",
            "",
            table(
                [
                    "model",
                    "decisions/s",
                    "latency p50 / p95 ms",
                    "rallies kept (returns per run)",
                    "points conceded per run",
                ],
                rows,
            ),
            "",
        ]

    for demo, title in (("fsd", "Driving, realtime clock"), ("fsd-lockstep", "Driving, lockstep clock")):
        rows = []
        for run in runs:
            s = load(run, demo)
            if s:
                rows.append(
                    [
                        s["label"],
                        interval_text(s["rules_score_pass_rate"], 0, 100) + " %",
                        interval_text(s["decisions_per_s"], 2),
                        lat(s["latency_ms"]),
                        lat(s["server_ms"]),
                    ]
                )
        if rows:
            out += [
                f"### {title}",
                "",
                table(
                    [
                        "model",
                        "rules score (drives passed)",
                        "decisions/s",
                        "latency p50 / p95 ms",
                        "server ms p50 / p95",
                    ],
                    rows,
                ),
                "",
            ]

    rows = []
    for run in runs:
        s = load(run, "ultrafast")
        if not s:
            continue
        for task, t in s.items():
            c = t["completion_ms"] or {}
            rows.append(
                [
                    f"{run.name.removeprefix('2026-10-01_demos-')} / {task}",
                    f"{t['completed']}/{t['runs']}",
                    f"{c['est']:.0f} [{c['lo']:.0f}, {c['hi']:.0f}]" if c.get("est") is not None else "n/a",
                    str(t["decisions_per_second_of_task_time"]),
                    lat(t["latency_ms"]),
                    str(t["largest_option_set"]),
                ]
            )
    if rows:
        out += [
            "### Browser agent (fixture tasks)",
            "",
            table(
                [
                    "model / task",
                    "completed",
                    "completion ms",
                    "decisions/s (task time)",
                    "latency p50 / p95 ms",
                    "largest option set",
                ],
                rows,
            ),
            "",
        ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
