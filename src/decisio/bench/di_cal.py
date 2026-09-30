# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Accuracy, ECE and Brier per Decision Index benchmark from the kit's own results.jsonl joined with its rows' gold
(the kit scores accuracy and macro-F1; it reports no calibration). ECE: 10 equal-mass bins on the top probability.

    python -m decisio.bench.di_cal <run dir with results.jsonl> --rows <rows.jsonl.gz> [--out di_cal.json]
"""
import argparse
import ast
import gzip
import json
from pathlib import Path

import numpy as np


def ece_equal_mass(conf, correct, bins=10):
    """Expected calibration error over `bins` bins of equal item count, sorted by confidence."""
    order = np.argsort(conf)
    e = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            e += len(chunk) / len(conf) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(e)


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gold = {}
    for line in gzip.open(a.rows, "rt"):
        r = json.loads(line)
        gold[r["id"]] = (r["family"], as_dict(r["expected"]))
    per, missing, n = {}, 0, 0
    for line in open(Path(a.run) / "results.jsonl"):
        r = json.loads(line)
        n += 1
        key = r["group_id"] if r["group_id"] in gold else next((k for k in (r["run_id"].split(":", 2)[-1],)
                                                                if k in gold), None)
        if key is None or r.get("status") != "ok":
            missing += 1
            continue
        fam, exp = gold[key]
        for qn, g in exp.items():
            ans = as_dict(r["response"])["answers"][qn]
            p = np.array(list(ans["probabilities"].values()), dtype=np.float64)
            keys = list(ans["probabilities"])
            y = keys.index(str(g))
            d = per.setdefault(fam, {"ok": [], "conf": [], "brier": [], "nll": []})
            d["ok"].append(float(ans["choice"] == str(g)))
            d["conf"].append(float(p.max()))
            d["brier"].append(float(((p - np.eye(len(p))[y]) ** 2).sum()))
            d["nll"].append(float(-np.log(max(p[y], 1e-300))))
    out = {"rows": n, "not_joined_or_failed": missing, "benchmarks": {}}
    for fam, d in sorted(per.items()):
        ok, conf = np.array(d["ok"]), np.array(d["conf"])
        out["benchmarks"][fam] = {"n": len(ok), "accuracy": float(ok.mean()), "ece": ece_equal_mass(conf, ok),
                                  "brier": float(np.mean(d["brier"])), "nll": float(np.mean(d["nll"])),
                                  "mean_confidence": float(conf.mean())}
        b = out["benchmarks"][fam]
        print(f"{fam:14s} n {b['n']:5d} accuracy {b['accuracy']:.4f} ECE {b['ece']:.4f} Brier {b['brier']:.4f} "
              f"NLL {b['nll']:.4f}")
    json.dump(out, open(a.out or Path(a.run) / "di_cal.json", "w"), indent=1)
    if missing:
        raise SystemExit(f"{missing} of {n} rows not joined or failed")


if __name__ == "__main__":
    main()
