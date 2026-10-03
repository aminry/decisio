# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Classify every Pong decision that disagrees with the oracle (rows from pong/scripts/oracle-replay.ts).

The oracle is the perfect-information paddle policy: move toward the predicted crossing at the full step, stay inside
half a step (6 units). While the ball moves away there is no crossing and the oracle's "stay" is a convention, so those
decisions are counted apart and never scored as errors. A disagreement on approach is classified in this order:
  latency        none by construction: a Pong lane's ball advances one segment per decision, so a late answer slows
                 the game and never makes a decision wrong
  prompt effect  the move is the one the question's own wording asks for (stay within 5 units) and the oracle's
                 differs (it stays within 6): the wording and the physics disagree in that band
  near-tie       p(model's move) - p(oracle's move) < margin (0.10); needs the full distribution, joined from the
                 run's decisions.jsonl.gz
  judgement      anything else: the wrong move preferred by a clear margin
Records without a distribution (session A) report near-tie and judgement as one column. A disagreement is harmful when
playing the model's move and then the oracle's turns a return into a miss.

    python pong_error_analysis.py --rows rows.jsonl [--log decisions.jsonl.gz] [--margin 0.10] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from decision_log import read
from stats import bootstrap

CLASSES = ["latency", "prompt effect", "near-tie", "judgement", "near-tie or judgement (no distribution)"]


def state_key(s: dict) -> tuple:
    b, p = s["ball"], s["paddle"]
    return (round(b["x"], 4), round(b["y"], 4), round(b["vx"], 4), round(b["vy"], 4), round(p["y"], 4))


def distributions(log_paths: list[str]) -> dict:
    """state key -> the move question's probabilities, from decision logs (System One lanes only)."""
    out = {}
    for path in log_paths:
        for r in read(path):
            ans = ((r.get("response") or {}).get("answers") or {}).get("move") or {}
            if "one-hot" in str(ans.get("probabilities_source", "")):
                continue  # a chat answer has no distribution: its misses stay near-tie or judgement, unseparated
            probs = ans.get("probabilities")
            state = (r.get("request") or {}).get("state")
            if probs and isinstance(state, dict):
                out[(r.get("lane"), state_key(state))] = probs
    return out


def classify(r: dict, margin: float) -> str | None:
    if r["move"] == r["oracle"]:
        return None
    if r["move"] == r["question"] != r["oracle"]:
        return "prompt effect"
    dist = r.get("distribution")
    if not dist:
        return "near-tie or judgement (no distribution)"
    return "near-tie" if dist[r["move"]] - dist[r["oracle"]] < margin else "judgement"


def analyse(rows: list[dict], margin: float) -> dict:
    by_lane: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lane[r["label"]].append(r)
    out = {}
    for lane, lr in by_lane.items():
        away = [r for r in lr if r["delta"] is None]
        scored = [r for r in lr if r["delta"] is not None]
        cls = Counter(classify(r, margin) for r in scored)
        seeds = defaultdict(list)
        for r in scored:
            seeds[r["seed"]].append(r["move"] == r["oracle"])
        agree = bootstrap([sum(v) / len(v) for v in seeds.values()])
        out[lane] = {
            "decisions": len(lr),
            "ball_away": len(away),
            "ball_away_moves": sum(r["move"] != "stay" for r in away),
            "scored": len(scored),
            "agree": sum(r["move"] == r["oracle"] for r in scored),
            "agree_rate_over_games": agree,
            "games": len(seeds),
            "classes": {c: cls.get(c, 0) for c in CLASSES},
            "harmful": sum(
                r["move"] != r["oracle"] and r["outcome_model"] == "miss" and r["outcome_oracle"] == "return"
                for r in scored
            ),
            "with_distribution": sum(1 for r in scored if r.get("distribution")),
        }
    return out


def table(res: dict) -> str:
    head = (
        "| lane | decisions on approach | agree with the oracle | "
        + " | ".join(CLASSES)
        + " | harmful (return to miss) | ball moving away (moved / of) |"
    )
    lines = [head, "| " + " | ".join(["---"] * (len(CLASSES) + 5)) + " |"]
    for lane, v in res.items():
        a = v["agree_rate_over_games"]
        lines.append(
            f"| {lane} | {v['scored']} | {v['agree']} ({a['est'] * 100:.1f}% "
            f"[{a['lo'] * 100:.1f}, {a['hi'] * 100:.1f}] over {v['games']} games) | "
            + " | ".join(str(v["classes"][c]) for c in CLASSES)
            + f" | {v['harmful']} | {v['ball_away_moves']} / {v['ball_away']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--log", nargs="*", default=[])
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dists = distributions(a.log) if a.log else {}
    rows = []
    for p in a.rows:
        for line in open(p):
            r = json.loads(line)
            d = dists.get((r["lane"], state_key(r["state"])))
            if d:
                r["distribution"] = d
            rows.append(r)
    res = analyse(rows, a.margin)
    print(table(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"margin": a.margin, "lanes": res}, f, indent=1)


if __name__ == "__main__":
    main()
