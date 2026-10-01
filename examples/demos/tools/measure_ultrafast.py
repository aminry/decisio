# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure the browser agent (examples/demos/ultrafast) on its local fixture tasks against a System One server.

It serves the fixture pages on a loopback port and opens a headless Chrome with a CDP port, points browser-harness at
it, runs each task `--runs` times and checks each outcome independently of the agent's own DONE. Reported: decisions
per second, p50 and p95 of the per-decision latency (the System One request, wall time), the options each question
asked over, the task completion time with a bootstrap interval, and the number of text-helper calls. Nothing is sent to
any hosted service: browser-harness telemetry and update checks are off.

    SYSTEMONE_BASE_URL=http://127.0.0.1:8100 TEXT_MODEL_BASE_URL=http://127.0.0.1:8200/v1 TEXT_MODEL=qwen \\
        python measure_ultrafast.py --label Decisio --runs 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo_run import DemoRun, bootstrap, caption_for, card_name, latency_summary

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1] / "ultrafast"
sys.path.insert(0, str(ROOT))

TASKS = {
    "travel": {
        # upstream's smoke goal: the first page already lists Casa Flora, so a goal without "use the filters" is met by
        # clicking it, which the verification below (the filters applied) would count as a failure
        "goal": "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
        "then open Casa Flora.",
        "url_end": "#casa-flora",
        "text_has": "Your filters: Design",
    },
    "research": {
        "goal": "Open the article about using finite choices to control browser agents.",
        "url_end": "#choices",
        "text_has": "",
    },
}
CHROMES = (
    os.environ.get("CHROME"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome"),
    shutil.which("chromium"),
    shutil.which("chrome"),
)


def chrome_path():
    for path in CHROMES:
        if path and Path(path).exists():
            return path
    sys.exit("no Chrome found: set CHROME=/path/to/chrome")


def serve_fixture(port):
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "jev_ultrafast" / "static"))
    handler.log_message = lambda *a: None
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def launch_chrome(port, headed):
    profile = tempfile.mkdtemp(prefix="uf-chrome-")
    args = [
        chrome_path(),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1120,780",
        "about:blank",
    ]
    if not headed:
        args.insert(1, "--headless=new")
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # Chrome refuses to run as root (a container) with its sandbox
        args.insert(1, "--no-sandbox")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1).read()
            return proc, profile
        except OSError:
            time.sleep(0.1)
    proc.kill()
    sys.exit("Chrome did not open its CDP port")


def run_task(name, task, base):
    from jev_ultrafast import Agent

    url = f"{base}/fixture.html?scenario={name}"
    started = time.perf_counter()
    with Agent(url, task["goal"]) as agent:
        state = None
        for state in agent.run():
            pass
        wall = time.perf_counter() - started
        verify = agent.browser.evaluate("document.body.innerText") or ""
        final_url = agent.browser.evaluate("location.href") or ""
        state = agent.snapshot()
    ok = state["status"] == "done" and final_url.endswith(task["url_end"]) and task["text_has"] in verify
    decisions = state["decisions"]
    return {
        "task": name,
        "ok": bool(ok),
        "status": state["status"],
        "elapsed_ms": state["elapsed_ms"],
        "wall_s": round(wall, 3),
        "decisions": [
            {
                "latency_ms": d["latency_ms"],
                "operation": d["operation"],
                "option_counts": d.get("option_counts", {}),
                "input_tokens": (d.get("usage") or {}).get("input_tokens"),
                "model": d.get("model"),
            }
            for d in decisions
        ],
        "text_calls": [{"latency_ms": t.get("latency_ms"), "model": t.get("model")} for t in state["text_calls"]],
        "actions": len(state["history"]),
    }


def summarise(runs):
    lat = [d["latency_ms"] for r in runs for d in r["decisions"]]
    per_question = {}
    for r in runs:
        for d in r["decisions"]:
            for q, n in d["option_counts"].items():
                per_question.setdefault(q, []).append(n)
    times = [r["elapsed_ms"] for r in runs if r["ok"]]
    decisions = sum(len(r["decisions"]) for r in runs)
    model_s = sum(sum(d["latency_ms"] for d in r["decisions"]) for r in runs) / 1000
    elapsed_s = sum(r["elapsed_ms"] for r in runs) / 1000
    return {
        "runs": len(runs),
        "completed": sum(r["ok"] for r in runs),
        "completion_rate": bootstrap([float(r["ok"]) for r in runs]),
        "decisions": decisions,
        "decisions_per_second_of_task_time": round(decisions / elapsed_s, 3) if elapsed_s else None,
        "decisions_per_second_of_model_time": round(decisions / model_s, 3) if model_s else None,
        "latency_ms": latency_summary(lat),
        "completion_ms": bootstrap(times),
        "completion_ms_p50": float(statistics.median(times)) if times else None,
        "options_per_question": {
            q: {"max": max(v), "mean": round(statistics.mean(v), 1)} for q, v in per_question.items()
        },
        "largest_option_set": max((max(v) for v in per_question.values()), default=0),
        "text_helper_calls": sum(len(r["text_calls"]) for r in runs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="the model's name as it should read in the record and the caption")
    ap.add_argument("--runs", type=int, default=5, help="runs per task")
    ap.add_argument("--tasks", default="travel,research")
    ap.add_argument("--out", default=None, help="run directory (default runs/<date>_demos-ultrafast-<label>)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--cdp-port", type=int, default=9344)
    ap.add_argument("--fixture-port", type=int, default=8766)
    ap.add_argument("--card", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    os.environ.setdefault("BH_TELEMETRY", "0")
    os.environ.setdefault("BH_UPDATE_CHECK", "0")
    os.environ["BU_CDP_URL"] = f"http://127.0.0.1:{a.cdp_port}"
    base_url = os.environ.get("SYSTEMONE_BASE_URL", "http://127.0.0.1:8100")
    slug = "".join(c if c.isalnum() else "-" for c in a.label.lower()).strip("-")
    out = Path(a.out) if a.out else REPO / "runs" / f"{time.strftime('%Y-%m-%d')}_demos-ultrafast-{slug}"
    card = a.card or card_name()
    run = DemoRun(out, f"Browser agent against {a.label}", "ultrafast", a.label, base_url, hardware=card)
    names = a.tasks.split(",")
    server = serve_fixture(a.fixture_port)
    chrome, profile = launch_chrome(a.cdp_port, a.headed)
    base = f"http://127.0.0.1:{a.fixture_port}"
    records = []
    try:
        for name in names:
            for i in range(a.runs):
                try:
                    rec = run_task(name, TASKS[name], base)
                except Exception as err:  # a failed run is a result, recorded with its reason
                    rec = {
                        "task": name,
                        "ok": False,
                        "status": f"error: {err}",
                        "elapsed_ms": 0,
                        "decisions": [],
                        "text_calls": [],
                        "actions": 0,
                        "wall_s": 0,
                    }
                records.append(rec)
                run.log(
                    f"{name} run {i + 1}/{a.runs}: ok={rec['ok']} {rec['status']} {rec['elapsed_ms']} ms, "
                    f"{len(rec['decisions'])} decisions"
                )
    finally:
        chrome.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    summary = {name: summarise([r for r in records if r["task"] == name]) for name in names}
    summary["all"] = summarise(records)
    run.write_json("runs.json", records)
    allr = summary["all"]
    md = [
        f"# Browser agent, {a.label}",
        "",
        f"{a.runs} runs of each of {len(names)} fixture tasks; {card}; client and server on the same machine; "
        "95% bootstrap intervals. The text helper is "
        f"{os.environ.get('TEXT_MODEL', '(not set)')}, a small local model that writes typed text only.",
        "",
        "| task | completed | completion time ms | decisions/s (task time) | latency p50 / p95 ms "
        "| largest option set |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in [*names, "all"]:
        s = summary[name]
        c = s["completion_ms"] or {}
        lat = s["latency_ms"]
        md.append(
            f"| {name} | {s['completed']}/{s['runs']} | "
            + (f"{c['est']:.0f} [{c['lo']:.0f}, {c['hi']:.0f}]" if c.get("est") is not None else "n/a")
            + f" | {s['decisions_per_second_of_task_time']} | "
            + (f"{lat['p50']:.0f} / {lat['p95']:.0f}" if lat.get("n") else "n/a")
            + f" | {s['largest_option_set']} |"
        )
    md += [
        "",
        "Options per question (max, mean): "
        + "; ".join(f"{q} {v['max']}, {v['mean']}" for q, v in allr["options_per_question"].items()),
    ]
    if a.note:
        md += ["", a.note]
    run.finish(
        summary,
        "\n".join(md),
        {
            "caption": caption_for(a.label, card, allr["latency_ms"].get("p50")),
            "text_model": os.environ.get("TEXT_MODEL"),
            "text_model_base_url": os.environ.get("TEXT_MODEL_BASE_URL"),
            "files": {
                "runs.json": "one row per run: outcome, completion time, every decision's latency and option counts",
                "summary.json, summary.md": "the tables with their bootstrap intervals",
            },
        },
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
