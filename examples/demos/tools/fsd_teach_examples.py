# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Oracle-labelled examples of the driving demo's motion question, for registering a task.

The demo's rules driver drives seeded training scenarios (suite seeds the caller keeps disjoint from the evaluated ones)
in headless Chrome, lockstep, with no model. At every decision where the demo would ask the motion question, the
question exactly as it would be asked is recorded with the rules driver's motion as the answer. Then `--per` examples
per answer are drawn (seeded), at most a quarter of an answer's examples from one scenario.

The manoeuvre question cannot be registered as the demo asks it: its keys are the candidates eligible at that tick and
its descriptions carry that tick's predicted progress and outcome, so every tick is a new option list (a task is one
fixed option list, docs/handoffs/tasks.md).

    python fsd_teach_examples.py --seeds 101-110 --count 3 --per 10 --out examples.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from measure_fsd import FSD, wait_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="101-110")
    ap.add_argument("--count", type=int, default=3, help="scenarios per suite seed")
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8342)
    ap.add_argument("--channel", default="chrome")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    env = dict(os.environ, PORT=str(a.port))
    server = subprocess.Popen(
        [sys.executable, "server.py"], cwd=FSD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    pool = defaultdict(list)
    try:
        wait_for(f"http://127.0.0.1:{a.port}/api/status")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, **({"channel": a.channel} if a.channel else {}))
            for seed in range(lo, hi + 1):
                page = browser.new_page()
                page.set_default_timeout(0)
                page.goto(f"http://127.0.0.1:{a.port}/bench")
                for _ in range(240):
                    if page.evaluate(
                        "() => !!window.__bench && document.querySelector('#map-note').textContent.length > 0"
                    ):
                        break
                    page.wait_for_timeout(250)
                opts = {
                    "brain": "rules",
                    "count": a.count,
                    "seed": seed,
                    "mode": "lockstep",
                    "save": False,
                    "collectExamples": True,
                }
                result = page.evaluate("(o) => window.__bench.run(o)", opts)
                page.close()
                for k, r in enumerate(result["results"]):
                    for e in r.get("examples", []):
                        pool[e["label"]].append({**e, "seed": seed, "scenario": k + 1})
                print(f"seed {seed}: {sum(len(v) for v in pool.values())} candidate examples", flush=True)
            browser.close()
    finally:
        server.terminate()
    rng = random.Random(lo * 31 + hi)
    examples = []
    for label, cands in sorted(pool.items()):
        rng.shuffle(cands)
        per_scn = defaultdict(int)
        picked = 0
        for e in cands:
            key = (e["seed"], e["scenario"])
            if picked >= a.per or per_scn[key] >= max(1, a.per // 4):
                continue
            per_scn[key] += 1
            picked += 1
            examples.append(
                {
                    "request": {"state": e["state"], "questions": {"motion": e["question"]}},
                    "answer": label,
                    "seed": e["seed"],
                    "scenario": e["scenario"],
                    "t": e["t"],
                }
            )
    counts = {k: len(v) for k, v in pool.items()}
    Path(a.out).write_text(
        json.dumps({"question": "motion", "seeds": [lo, hi], "pool": counts, "examples": examples}) + "\n"
    )
    print("pool", counts, "->", {k: sum(e["answer"] == k for e in examples) for k in pool})


if __name__ == "__main__":
    main()
