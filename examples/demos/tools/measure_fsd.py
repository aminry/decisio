# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure the driving demo (examples/demos/fsd) against a System One server.

It starts the demo's own server pointed at `--base-url`, opens its benchmark page in headless Chrome and runs the
seeded scenario suite there with the model as the brain (`window.__bench.run`), `--runs` times with different suite
seeds. The clock is `realtime` by default: the car keeps driving while a request is in flight, so the model's latency
counts. Reported per model: the rules score (the share of drives that arrive with no collision, red light, rolled stop
sign, failure to yield or a second off the road), decisions per second, and p50 / p95 of the per-decision latency.

    python measure_fsd.py --base-url http://127.0.0.1:8100 --label "Decisio" --runs 3 --count 6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from decision_log import DecisionLog, compress
from demo_run import DemoRun, bootstrap, caption_for, card_name, interval_text, latency_summary

FSD = Path(__file__).resolve().parents[1] / "fsd"
REPO = Path(__file__).resolve().parents[3]


def wait_for(url: str, seconds: float = 60.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up in {seconds:.0f} s")


def server_text(s: dict) -> str:
    return f"{s['p50']:.0f} / {s['p95']:.0f} ms" if s.get("n") else "n/a (the server sends no x-decisio-server-ms)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--label", required=True, help="the model's name as it should read in the record and the caption")
    ap.add_argument("--model-api", default="", help="SYSTEMONE_MODEL, when the server needs a model name")
    ap.add_argument("--out", default=None, help="run directory (default runs/<date>_demos-fsd-<label>)")
    ap.add_argument("--runs", type=int, default=3, help="repeated runs, each a different suite seed")
    ap.add_argument("--count", type=int, default=6, help="scenarios per run")
    ap.add_argument("--npcs", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--mode", choices=["realtime", "lockstep"], default="realtime")
    ap.add_argument("--weather", default="dry")
    ap.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="the demo's decision timeout (default 1500 ms; raise it for lockstep)",
    )
    ap.add_argument("--port", type=int, default=8322)
    ap.add_argument("--channel", default="chrome", help="Playwright browser channel; '' for its own Chromium")
    ap.add_argument("--card", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    slug = "".join(c if c.isalnum() else "-" for c in a.label.lower()).strip("-")
    out = Path(a.out) if a.out else REPO / "runs" / f"{time.strftime('%Y-%m-%d')}_demos-fsd-{slug}"
    card = a.card or card_name()
    run = DemoRun(out, f"Driving demo against {a.label}", "fsd", a.label, a.base_url, hardware=card)

    env = dict(
        os.environ, SYSTEMONE_BASE_URL=a.base_url, PORT=str(a.port), DEMO_DECISION_LOG=str(run.decision_log_path)
    )
    if a.model_api:
        env["SYSTEMONE_MODEL"] = a.model_api
    server = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=FSD,
        env=env,
        stdout=open(out / "demo_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    rows = []
    oracle_log = DecisionLog(out / "oracle_decisions.jsonl")
    try:
        wait_for(f"http://127.0.0.1:{a.port}/api/status")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            launch = {"headless": True}
            if a.channel:
                launch["channel"] = a.channel
            browser = p.chromium.launch(**launch)
            for i in range(a.runs):
                seed = a.seed0 + i
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                page.set_default_timeout(0)
                page.goto(f"http://127.0.0.1:{a.port}/bench")
                # the page's CSP forbids string eval, which wait_for_function uses, so poll with a function
                for _ in range(240):
                    if page.evaluate(
                        "() => !!window.__bench && document.querySelector('#map-note').textContent.length > 0"
                    ):
                        break
                    page.wait_for_timeout(250)
                run.log(f"run {i + 1}/{a.runs}: suite seed {seed}, {a.count} scenarios, {a.mode}")
                t0 = time.perf_counter()
                result = page.evaluate(
                    "(o) => window.__bench.run(o)",
                    {
                        "brain": "jev",
                        "count": a.count,
                        "npcs": a.npcs,
                        "seed": seed,
                        "mode": a.mode,
                        "weather": a.weather,
                        "save": False,
                        "timeoutMs": a.timeout_ms,
                    },
                )
                wall = time.perf_counter() - t0
                page.close()
                results = result["results"]
                # every model decision with its rules oracle, one line each, kept apart from the per-run rows
                for k, r in enumerate(results):
                    for dec in r.pop("decision_log", []):
                        oracle_log.add(demo="fsd", run=i + 1, seed=seed, scenario=k + 1, scenario_id=r.get("id"), **dec)
                lat = [x for r in results for x in r.get("latencies_ms", [])]
                srv = [x for r in results for x in r.get("server_ms", [])]
                # model decisions only: the pilot also takes local decisions (a hard brake) that make no request
                decisions = len(lat)
                drive_s = sum(r.get("wall_ms", 0) for r in results) / 1000
                rows.append(
                    {
                        "seed": seed,
                        "scenarios": len(results),
                        "pass": [bool(r["pass"]) for r in results],
                        "failures": [r["failures"] for r in results],
                        "arrived": [bool(r["arrived"]) for r in results],
                        "decisions": decisions,
                        "drive_wall_s": drive_s,
                        "decisions_per_s": decisions / drive_s if drive_s else None,
                        "latency_ms": lat,
                        "server_ms": srv,
                        "avg_kmh": result["summary"].get("avg_kmh"),
                        "violations_per_km": result["summary"].get("violations_per_km"),
                        "wall_s": wall,
                    }
                )
                run.log(
                    f"  pass {sum(rows[-1]['pass'])}/{len(results)}, {decisions} decisions in {drive_s:.0f} s "
                    f"({rows[-1]['decisions_per_s'] or 0:.2f}/s), p50 {np.median(lat) if lat else float('nan'):.0f} ms"
                )
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    run.write_json("runs.json", rows)
    compress(out / "oracle_decisions.jsonl")
    passes = [x for r in rows for x in r["pass"]]
    lat_all = [x for r in rows for x in r["latency_ms"]]
    srv_all = [x for r in rows for x in r["server_ms"]]
    per_run_pass = [float(np.mean(r["pass"])) for r in rows]
    per_run_rate = [r["decisions_per_s"] for r in rows if r["decisions_per_s"] is not None]
    per_run_p50 = [float(np.median(r["latency_ms"])) for r in rows if r["latency_ms"]]
    summary = {
        "label": a.label,
        "runs": a.runs,
        "scenarios_per_run": a.count,
        "mode": a.mode,
        "rules_score_pass_rate": bootstrap(passes),
        "rules_score_pass_rate_per_run": bootstrap(per_run_pass),
        "decisions_per_s": bootstrap(per_run_rate),
        "latency_ms": latency_summary(lat_all),
        "latency_p50_over_runs": bootstrap(per_run_p50),
        "server_ms": latency_summary(srv_all),
        "options_per_question": "16 at most (15 candidates + hard_brake); see README.md",
    }
    md = [
        f"# Driving demo, {a.label}",
        "",
        f"{a.runs} runs of {a.count} seeded scenarios ({a.mode} clock, {a.npcs} traffic cars, {a.weather}); {card}; "
        "client and server on the same machine. 95% bootstrap intervals.",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| rules score (drives passed) | {interval_text(summary['rules_score_pass_rate'], 0, 100)} % "
        f"over {len(passes)} drives |",
        f"| decisions per second | {interval_text(summary['decisions_per_s'], 2)} over runs |",
        f"| per-decision latency p50 / p95 | {summary['latency_ms'].get('p50', float('nan')):.0f} / "
        f"{summary['latency_ms'].get('p95', float('nan')):.0f} ms over {summary['latency_ms'].get('n', 0)} decisions |",
        f"| server time p50 / p95 | {server_text(summary['server_ms'])} |",
    ]
    if a.note:
        md += ["", a.note]
    run.finish(
        summary,
        "\n".join(md),
        {
            "caption": caption_for(a.label, card, summary["latency_ms"].get("p50")),
            "files": {
                "oracle_decisions.jsonl.gz": "every model decision: its answers (full distributions) and the rules "
                "driver's choice on the asked snapshot and its motion on the snapshot when the answer arrived",
                "runs.json": "one row per run: pass flags, failures, decisions, drive wall time, "
                "every decision's latency",
                "summary.json, summary.md": "the tables with their bootstrap intervals",
                "demo_server.log": "the demo server's log",
            },
        },
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
