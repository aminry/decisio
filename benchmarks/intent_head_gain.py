# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The intent head's paired gain over the plain readout, with a bootstrap interval, from a run's per-item rows.

For each task and draw, every test item's row (`intent_heads/<task>_d<draw>.json.gz`) carries the gold label, the
probabilities served with the head, and `hidden_lp`, the label log-probabilities of the same forward pass. In the
single-engine mode `hidden_lp` is the engine's own plain readout, so its argmax is the answer without a task. The gain
is the mean over draws of (head accuracy - plain accuracy); the interval resamples test items with replacement, the
same resample for every draw, 10,000 times with numpy's default_rng(0), and takes the 2.5th and 97.5th percentiles.

    python benchmarks/intent_head_gain.py runs/2026-09-30_plugin-verification [--out <run>/derived/...json]
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

RESAMPLES, SEED = 10_000, 0


def load(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        return json.load(f)


def task_gain(run, task):
    heads = sorted(run.glob(f"intent_heads/{task}_d[0-9].json*"))
    options = load(next(run.glob(f"intent_heads/{task}_d0_task.json*")))["options"]
    head, plain = [], []
    for path in heads:
        rows = sorted(load(path)["rows"], key=lambda r: r["i"])
        gold = np.array([options.index(r["gold"]) for r in rows])
        head.append(np.array([np.argmax(r["served"]) for r in rows]) == gold)
        plain.append(np.array([np.argmax(r["hidden_lp"]) for r in rows]) == gold)
    H, P = np.array(head, float), np.array(plain, float)
    n = H.shape[1]
    idx = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    gains = (H[:, idx].mean(axis=2) - P[:, idx].mean(axis=2)).mean(axis=0)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {
        "test_items": n,
        "draws": len(heads),
        "plain_accuracy": round(float(P.mean()), 4),
        "head_accuracy_by_draw": [round(float(x), 4) for x in H.mean(axis=1)],
        "head_accuracy": round(float(H.mean()), 4),
        "gain_points": round(100 * float(H.mean() - P.mean()), 1),
        "interval_95_points": [round(100 * float(lo), 1), round(100 * float(hi), 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    tasks = sorted({p.name.split("_d")[0] for p in a.run.glob("intent_heads/*_d0_task.json*")})
    out = {
        "method": __doc__.split("\n\n")[1].replace("\n", " "),
        "resamples": RESAMPLES,
        "seed": SEED,
        "tasks": {t: task_gain(a.run, t) for t in tasks},
    }
    text = json.dumps(out, indent=1) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
