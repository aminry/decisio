# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The Decision Index 0.2.1 values of the four benchmarks run here, computed by the kit's own index02 functions from the
kit's own `benchmark-summary.json` (its `score` writes that file first; the index step after it needs all 38 benchmarks
and stops on the ones not run, so the four values are computed here with the same functions, chance levels and
coverage rule the board uses).

    python -m decisio.bench.di_report <run dir with benchmark-summary.json and results.jsonl> --suite-dir <suite> \
        [--out ...]
GPQA Diamond is track-scored (the 0.1 panel's rule: unanswered groups score zero inside the metric), so the kit's
`score_panel` runs over the results first, exactly as its index step does.
"""

import argparse
import json
from pathlib import Path

from decision_index.scoring import index02 as X
from decision_index.scoring.index import score_panel
from decision_index.scoring.report import load_results
from decision_index.suite.io import Suite

IDS = {4: "BANKING77", 5: "CLINC150+OOS", 25: "GPQA Diamond", 57: "MMLU-Pro"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = json.load(open(Path(a.run) / "benchmark-summary.json"))
    spec = X.spec("0.2.1")
    by_id = {b["catalog_id"]: b for b in s["benchmarks"]}
    scored = score_panel(Suite(Path(a.suite_dir), "0.2.1"), load_results(Path(a.run) / "results.jsonl"))
    out = {
        "edition": "0.2.1",
        "engine": s.get("engine"),
        "latency_ms": s.get("successful_request_latency_ms"),
        "benchmarks": {},
    }
    for n, name in IDS.items():
        b = by_id.get(n, {})
        v = X.benchmark_value(n, spec, scored.get(n), b)
        out["benchmarks"][name] = {
            "catalog_id": n,
            "metric": b.get("metric"),
            "native_score": b.get("score"),
            "requests": b.get("requests"),
            "answered": b.get("answered"),
            "errors": b.get("errors"),
            "unsupported": b.get("unsupported"),
            "median_ms": b.get("median_ms"),
            "chance": X.chance_of(n, spec),
            "raw": v.get("raw"),
            "skill": v.get("skill"),
            "coverage": v.get("coverage"),
        }
        r = out["benchmarks"][name]
        print(
            f"{name:14s} {str(r['metric']):9s} native {r['native_score']} answered {r['answered']}/{r['requests']} "
            f"raw {r['raw']} skill {r['skill']} median {r['median_ms']} ms"
        )
    json.dump(out, open(a.out or Path(a.run) / "di_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
