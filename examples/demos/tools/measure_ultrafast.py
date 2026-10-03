# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure the browser agent (examples/demos/ultrafast) on its local fixture tasks against a System One server.

It serves the fixture pages on a loopback port and opens a headless Chrome with a CDP port, points browser-harness at
it, runs each task `--runs` times and checks each outcome independently of the agent's own DONE. Reported: decisions
per second, p50 and p95 of the per-decision latency (the System One request, wall time), the options each question
asked over, the task completion time with a bootstrap interval, and the number of text-helper calls. Nothing is sent to
any hosted service: browser-harness telemetry and update checks are off.

Every run also writes its trajectory (`trajectory.py`) to `trajectories/<task>_<run>/`: one tick per model decision with
the screenshot the agent observed for it (`frames/step_NNN.png`, a lossless capture taken with the page's state), the
chosen element's box in screenshot pixels, the request as sent and the full answer, the typed text and the latency, then
a tick with the page after the agent stopped. `render_ultrafast.py` draws a clip from it alone. The screenshots are read
only and are not in the request, so the agent decides on the same page; they add their capture time to every step.

    SYSTEMONE_BASE_URL=http://127.0.0.1:8100 TEXT_MODEL_BASE_URL=http://127.0.0.1:8200/v1 TEXT_MODEL=qwen \\
        python measure_ultrafast.py --label Decisio --runs 10
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import trajectory
from decision_log import DecisionLog
from demo_run import DemoRun, bootstrap, caption_for, card_name, latency_summary, server_health

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


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def serve_fixture(port):
    handler = partial(QuietHandler, directory=str(ROOT / "jev_ultrafast" / "static"))
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


def decision_rows(index, task, decisions):
    """One log row per model decision: the request body the server received and the full answer."""
    return [
        {
            "demo": "ultrafast",
            "run": index,
            "task": task,
            "step": step,
            "request": d.get("request"),
            "response": {"model": d.get("model"), "answers": d.get("raw_answers"), "usage": d.get("usage")},
            "skipped_single_option": d.get("skipped_single_option", []),
            "latency_ms": d.get("latency_ms"),
        }
        for step, d in enumerate(decisions, 1)
    ]


def png_size(data: bytes) -> list[int]:
    """Width and height from a PNG's header."""
    return list(struct.unpack(">II", data[16:24]))


class Trajectory:
    """One run's trajectory: a tick per model decision, kept from the page the decision was made on (its screenshot
    taken with its state, the chosen element's box), and a tick with the page after the agent stopped. Screenshots stay
    in memory during the run and are written with the trajectory after it, so the run's clock does not include them."""

    def __init__(self, directory: Path, player: dict, run: dict):
        self.dir, self.player, self.run = Path(directory), player, run
        self.steps: list[dict] = []
        self.final: dict | None = None
        self.executed = 0

    def agent_options(self) -> dict:
        return {"screenshots": "png", "on_decision": self.on_decision}

    def on_decision(self, page, decision) -> None:
        self.steps.append({"t_wall": time.time(), "page": page, "decision": decision, "action": None})

    def after_tick(self, history) -> None:
        """Attach an executed action (its typed text) to the decision it executed: a tick runs at most one."""
        new = len(history) > self.executed
        if new and self.steps and history[-1]["choice"] == self.steps[-1]["decision"]["choice"]:
            self.steps[-1]["action"] = history[-1]
        self.executed = len(history)

    def stopped(self, browser) -> None:
        """The page after the agent stopped: a read-only capture, after the run's last step."""
        t_wall = time.time()
        try:
            shot = browser.call("Page.captureScreenshot", format="png")["data"]
            self.final = {
                "t_wall": t_wall,
                "url": browser.evaluate("location.href"),
                "title": browser.evaluate("document.title"),
                "viewport": [browser.evaluate("innerWidth"), browser.evaluate("innerHeight")],
                "screenshot": shot,
            }
        except Exception as err:  # a broken page is recorded, not hidden
            self.final = {"t_wall": t_wall, "error": str(err)}

    def _frame(self, name: str, b64: str) -> dict:
        data = base64.b64decode(b64)
        path = self.dir / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"frame": f"frames/{name}", "frame_sha256": trajectory.file_sha256(path), "frame_size": png_size(data)}

    @staticmethod
    def highlight(page, decision, frame_size) -> dict | None:
        """The chosen element's box in screenshot pixels (the page's CSS box times the device-pixel ratio); none for
        an operation without an element (DONE, BLOCKED, WAIT, SCROLL)."""
        action = next((a for a in page["actions"] if a["id"] == decision["choice"]), None)
        if not action or "rect" not in action:
            return None
        scale = frame_size[0] / page["w"]
        r = action["rect"]
        bbox = [round(r[k] * scale, 1) for k in ("x", "y", "w", "h")]
        return {"bbox": bbox, "index": decision["target"], "label": action["label"]}

    def ticks(self) -> list[dict]:
        out = []
        for k, s in enumerate(self.steps):
            page, d, act = s["page"], s["decision"], s["action"]
            frame = self._frame(f"step_{k + 1:03d}.png", page["screenshot"])
            action = next((a for a in page["actions"] if a["id"] == d["choice"]), {})
            out.append(
                {
                    "tick": k,
                    "t_wall": s["t_wall"],
                    "step": k + 1,
                    "url": page["url"],
                    "title": page["title"],
                    "viewport": [page["w"], page["h"]],
                    "scroll_y": page["scroll"]["y"],
                    "screenshot_ms": page.get("screenshot_ms"),
                    **frame,
                    "highlight": self.highlight(page, d, frame["frame_size"]),
                    "decision": {
                        "request": d["request"],
                        "answer": d["raw_answers"],
                        "model": d["model"],
                        "usage": d["usage"],
                        "skipped_single_option": d["skipped_single_option"],
                        "chosen": {
                            "operation": d["operation"],
                            "target": d["target"],
                            "action": d["choice"],
                            "kind": action.get("kind"),
                            "label": action.get("label"),
                        },
                        "text": act["text"] if act else None,
                        "text_helper": (
                            {"model": act["text_helper"], "latency_ms": act["text_latency_ms"]}
                            if act and act["text_helper"]
                            else None
                        ),
                        "latency_ms": d["latency_ms"],
                    },
                    "executed": act is not None,
                    "page_changed": act["page_changed"] if act else None,
                }
            )
        if self.final:
            f = self.final
            tick = {"tick": len(out), "t_wall": f["t_wall"], "step": None, "final": True, "decision": None}
            if "screenshot" in f:
                tick.update(url=f["url"], title=f["title"], viewport=f["viewport"])
                tick.update(self._frame("final.png", f["screenshot"]), highlight=None)
            else:
                tick["error"] = f["error"]
            out.append(tick)
        return out

    def write(self, end: dict) -> str:
        ticks = self.ticks()
        framed = next((t for t in ticks if t.get("frame")), None)
        dpr = framed["frame_size"][0] / framed["viewport"][0] if framed else None
        run = {**self.run, "viewport": {"width": 1120, "height": 780, "device_pixel_ratio": dpr}}
        head = trajectory.header("ultrafast", self.player, run)
        return trajectory.write(self.dir / trajectory.NAME, head, ticks, end)


def run_task(name, task, base, traj=None):
    """One run of a task; with `traj` (a Trajectory) the agent also captures a screenshot with every observation."""
    from jev_ultrafast import Agent

    url = f"{base}/fixture.html?scenario={name}"
    started = time.perf_counter()
    with Agent(url, task["goal"], **(traj.agent_options() if traj else {})) as agent:
        state = None
        try:
            for state in agent.run():
                if traj:
                    traj.after_tick(state["history"])
        finally:
            if traj:
                traj.stopped(agent.browser)
        wall = time.perf_counter() - started
        verify = agent.browser.evaluate("document.body.innerText") or ""
        final_url = agent.browser.evaluate("location.href") or ""
        state = agent.snapshot()
    ok = state["status"] == "done" and final_url.endswith(task["url_end"]) and task["text_has"] in verify
    decisions = state["decisions"]
    return {
        "_log": decision_rows(0, name, decisions),
        "_verification": {
            "status": state["status"],
            "final_url": final_url,
            "url_end": task["url_end"],
            "url_ok": final_url.endswith(task["url_end"]),
            "text_has": task["text_has"],
            "text_ok": task["text_has"] in verify,
            "ok": bool(ok),
        },
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
        # the trajectory's screenshot of each decision's page: its capture is in the completion time, not the latency
        "screenshot_ms": latency_summary([ms for r in runs for ms in r.get("screenshot_ms", []) if ms is not None]),
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
    ap.add_argument("--player", default=None, help="the player's fields as JSON (trajectory.PLAYER_FIELDS)")
    a = ap.parse_args()
    os.environ.setdefault("BH_TELEMETRY", "0")
    os.environ.setdefault("BH_UPDATE_CHECK", "0")
    os.environ["BU_CDP_URL"] = f"http://127.0.0.1:{a.cdp_port}"
    # browser-harness runs one daemon per BU_NAME ("default" otherwise): runs side by side must not share one
    os.environ.setdefault("BU_NAME", f"measure-{a.cdp_port}")
    base_url = os.environ.get("SYSTEMONE_BASE_URL", "http://127.0.0.1:8100")
    slug = "".join(c if c.isalnum() else "-" for c in a.label.lower()).strip("-")
    out = Path(a.out) if a.out else REPO / "runs" / f"{time.strftime('%Y-%m-%d')}_demos-ultrafast-{slug}"
    card = a.card or card_name()
    run = DemoRun(out, f"Browser agent against {a.label}", "ultrafast", a.label, base_url, hardware=card)
    names = a.tasks.split(",")
    player = trajectory.player_fields(a.player, label=a.label, card=card, server=server_health(base_url))
    # what shapes the request besides the page: the prompt variant and the request opt-ins (model.py)
    variant = {
        "prompt_variant": os.environ.get("ULTRAFAST_PROMPT_VARIANT", "default"),
        "string_descriptions": os.environ.get("SYSTEMONE_STRING_DESCRIPTIONS") == "1",
        "skip_single_option": os.environ.get("SYSTEMONE_SKIP_SINGLE_OPTION") == "1",
        "systemone_model": os.environ.get("SYSTEMONE_MODEL"),
        "text_model": os.environ.get("TEXT_MODEL"),
    }
    server = serve_fixture(a.fixture_port)
    chrome, profile = launch_chrome(a.cdp_port, a.headed)
    base = f"http://127.0.0.1:{a.fixture_port}"
    records = []
    log = DecisionLog(run.decision_log_path)
    try:
        for name in names:
            for i in range(a.runs):
                traj_run = {
                    "task": name,
                    "goal": TASKS[name]["goal"],
                    "index": i + 1,
                    "url": f"{base}/fixture.html?scenario={name}",
                    **variant,
                }
                traj = Trajectory(out / "trajectories" / f"{name}_{i + 1}", player, traj_run)
                try:
                    rec = run_task(name, TASKS[name], base, traj)
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
                for row in rec.pop("_log", []):
                    log.add(**{**row, "run": i + 1})
                traj.write(
                    {
                        "completed": rec["ok"],
                        "status": rec["status"],
                        "verification": rec.pop("_verification", None),
                        "elapsed_ms": rec["elapsed_ms"],
                        "wall_s": rec["wall_s"],
                        "decisions": len(traj.steps),
                        "actions": rec["actions"],
                        "text_calls": len(rec["text_calls"]),
                    }
                )
                rec["screenshot_ms"] = [s["page"].get("screenshot_ms") for s in traj.steps]
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
    shots = allr["screenshot_ms"]
    if shots.get("n"):
        md += [
            "",
            "Every observation also captures a lossless screenshot for the run's trajectory "
            f"(p50 {shots['p50']:.0f} ms per capture); the completion times include it, the decision latencies do not.",
        ]
    if a.note:
        md += ["", a.note]
    run.finish(
        summary,
        "\n".join(md),
        {
            "caption": caption_for(a.label, card, allr["latency_ms"].get("p50")),
            "text_model": os.environ.get("TEXT_MODEL"),
            "prompt_variant": os.environ.get("ULTRAFAST_PROMPT_VARIANT", "default"),
            "text_model_base_url": os.environ.get("TEXT_MODEL_BASE_URL"),
            "files": {
                "runs.json": "one row per run: outcome, completion time, every decision's latency and option counts",
                "summary.json, summary.md": "the tables with their bootstrap intervals",
                "trajectories/<task>_<run>/": "each run's trajectory (trajectory.jsonl.gz) and its screenshots "
                "(frames/), the only input of render_ultrafast.py",
            },
        },
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
