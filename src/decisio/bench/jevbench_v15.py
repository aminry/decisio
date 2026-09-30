# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""JevBench v1.5 open-set Intelligence, scored privately from the harness's own per-item results
(github.com/fstandhartinger/jevbench, MIT; the frozen method: its docs/METHOD-v1.5.md sections 3.1-3.2, and the v1.5.0
headline amendment A, equal request types).

The v1.5 harness and scorer are not published (the public repository holds the v1.4.2.2 harness, the 231 published v1.2
items, typed choice / noul / score, and the frozen v1.5 method), so this implements the method's open-set rules on the
harness's results for the published items. It is a private reading, not a board number: the board's open set has 904
items (601 published, of which the 370 added in v1.5 are not in the repository; 303 never published), and half of
Intelligence is the sealed set, which only the evaluator runs.

  Choice  CC = 100 (acc - c) / (1 - c), c the mean over the tier's items of 1 / options; invalid answers wrong
  Noul    P(yes) <= 0.20 is No, >= 0.80 is Yes, anything between is an abstention, counted wrong;
          CC = 100 (acc - 0.5) / 0.5
  Score   CC = 100 (1 - mean nMAE / mean nMAE_chance); nMAE = |expected position - gold| / (K - 1); an item's
          nMAE_chance is the mean of |l - gold| / (K - 1) over its K levels; an invalid answer takes the item's
          largest possible error
  Tiers   easy 0.10, standard 0.20, judge 0.30, hard 0.40, renormalised over the tiers that have items
  Types   A (the v1.5.x headline): choice, noul, score a third each; B (the frozen method): 0.50 / 0.25 / 0.25
  Per-tier values are not clipped.

    python -m decisio.bench.jevbench_v15 --run easy=<tasks.jsonl>:<results.jsonl> --run standard=... --run hard=... \
        --out v15.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TIER_WEIGHTS = {"easy": 0.10, "standard": 0.20, "judge": 0.30, "hard": 0.40}
TYPE_WEIGHTS = {"A": {"choice": 1 / 3, "noul": 1 / 3, "score": 1 / 3},
                "B": {"choice": 0.50, "noul": 0.25, "score": 0.25}}
NOUL_NO, NOUL_YES = 0.20, 0.80
PUBLIC_FILE_TIER = {"easy": "easy", "original": "standard", "hard": "hard"}     # the published v1.2 files' tiers


def item_score(task: dict, rec: dict | None) -> dict:
    """One item under the v1.5 rules: {"type", "unit"} plus the type's fields. `rec` is the harness's result row for
    the task (None: not answered, scored as invalid)."""
    qtype = task["question"]["type"]
    probs = rec.get("probs") if rec and rec.get("valid") else None
    if qtype == "choice":
        k = len(task["labels"])
        return {"type": "choice", "correct": bool(probs is not None and rec.get("predicted") == str(task["expected"])),
                "chance": 1.0 / k, "valid": probs is not None}
    if qtype == "noul":
        if probs is None:
            return {"type": "noul", "correct": False, "abstained": False, "valid": False}
        p = float(probs["yes"])
        said = "yes" if p >= NOUL_YES else ("no" if p <= NOUL_NO else None)
        return {"type": "noul", "correct": said == str(task["expected"]), "abstained": said is None, "valid": True,
                "p_yes": p}
    k, gold = len(task["labels"]), int(task["expected"])
    chance = sum(abs(lv - gold) for lv in range(k)) / k / (k - 1)
    if probs is None:
        return {"type": "score", "nmae": max(gold, k - 1 - gold) / (k - 1), "nmae_chance": chance, "valid": False}
    pred = sum(int(lv) * float(p) for lv, p in probs.items())
    return {"type": "score", "nmae": abs(pred - gold) / (k - 1), "nmae_chance": chance, "valid": True}


def cell_cc(qtype: str, items: list[dict]) -> float:
    """Chance-corrected competence of one type x tier cell (not clipped)."""
    n = len(items)
    if qtype == "choice":
        acc, c = sum(x["correct"] for x in items) / n, sum(x["chance"] for x in items) / n
        return 100.0 * (acc - c) / (1.0 - c)
    if qtype == "noul":
        return 100.0 * (sum(x["correct"] for x in items) / n - 0.5) / 0.5
    return 100.0 * (1.0 - sum(x["nmae"] for x in items) / sum(x["nmae_chance"] for x in items))


def score_runs(runs: list[tuple[str, list[dict], list[dict]]]) -> dict:
    """runs: [(tier, tasks, harness result rows)]. Returns the per-type per-tier competence, CC per type and I_open
    under type weightings A and B, with the item counts."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for tier, tasks, rows in runs:
        if tier not in TIER_WEIGHTS:
            raise ValueError(f"unknown tier {tier!r}")
        by = {r["task_id"]: r for r in rows}
        for t in tasks:
            s = item_score(t, by.get(t["id"]))
            cells.setdefault((s["type"], tier), []).append(s)
    out = {"per_type": {}, "n_items": sum(len(v) for v in cells.values()),
           "invalid": sum(not x["valid"] for v in cells.values() for x in v)}
    for qtype in ("choice", "noul", "score"):
        tiers = {tier: cell_cc(qtype, cells[(qtype, tier)]) for tier in TIER_WEIGHTS if (qtype, tier) in cells}
        if not tiers:
            continue
        w = sum(TIER_WEIGHTS[t] for t in tiers)
        rec = {"cc": sum(TIER_WEIGHTS[t] * v for t, v in tiers.items()) / w, "tiers": tiers,
               "n": {t: len(cells[(qtype, t)]) for t in tiers}}
        if qtype == "noul":
            its = [x for t in tiers for x in cells[(qtype, t)]]
            rec["abstention_rate"] = sum(x["abstained"] for x in its) / len(its)
        out["per_type"][qtype] = rec
    for name, W in TYPE_WEIGHTS.items():
        sup = {t: W[t] for t in out["per_type"]}
        out[f"I_open_{name}"] = sum(sup[t] * out["per_type"][t]["cc"] for t in sup) / sum(sup.values())
    out["tiers_missing"] = sorted(set(TIER_WEIGHTS) - {tier for _, tier in cells})
    return out


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="<tier>=<tasks.jsonl>:<results.jsonl>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for spec in a.run:
        tier, paths = spec.split("=", 1)
        tasks, results = paths.split(":", 1)
        runs.append((tier, load_jsonl(tasks), load_jsonl(results)))
    rep = score_runs(runs)
    rep["note"] = ("private reading of JevBench v1.5 open-set Intelligence on the published v1.2 items; not a board "
                   "number (the board's open set has 904 items and half of Intelligence is sealed)")
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for qtype, r in rep["per_type"].items():
        print(f"{qtype:6s} CC {r['cc']:6.2f}  "
              + "  ".join(f"{t} {v:.1f} (n={r['n'][t]})" for t, v in r["tiers"].items())
              + (f"  abstained {r['abstention_rate']:.3f}" if qtype == "noul" else ""))
    print(f"I_open A (equal types) {rep['I_open_A']:.2f}   B (50/25/25) {rep['I_open_B']:.2f}   "
          f"items {rep['n_items']}, "
          f"invalid {rep['invalid']}, tiers without items {rep['tiers_missing']}")
    print("JEVBENCH_V15_DONE")


if __name__ == "__main__":
    main()
