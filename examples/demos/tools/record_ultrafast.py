# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Record the browser agent (examples/demos/ultrafast) working through its inspector.

It starts the inspector (the demo's own server) against a System One server and a headless Chrome the agent drives
over CDP, then records the inspector page in a second headless Chrome: the page the agent sees with its indexed
elements, the ranked choices with their probabilities, the decision time, and the trail of executed actions. The
inspector's "slow motion" option (a 450 ms pause between choosing and executing) is on by default so the clip can be
read; the decision time on screen is the server's either way. The caption says so.

    python record_ultrafast.py --label Decisio --out runs/<run>/media --task travel
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from demo_recorder import record, to_gif, to_mp4
from demo_run import caption_for, card_name
from measure_ultrafast import ROOT, TASKS, launch_chrome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ultrafast")
    ap.add_argument("--task", choices=sorted(TASKS), default="travel")
    ap.add_argument(
        "--no-pace", action="store_true", help="record at full speed instead of the inspector's slow motion"
    )
    ap.add_argument("--max-seconds", type=float, default=120)
    ap.add_argument("--size", default="1440x900")
    ap.add_argument("--zoom", type=float, default=0.68, help="page zoom, so the whole inspector fits the picture")
    ap.add_argument("--summary", default=None, help="a measure_ultrafast summary.json, for the median in the caption")
    ap.add_argument("--p50", type=float, default=None)
    ap.add_argument("--card", default="")
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--cdp-port", type=int, default=9345)
    ap.add_argument("--demo-port", type=int, default=8767)
    a = ap.parse_args()

    width, height = (int(x) for x in a.size.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p50 = a.p50
    if a.summary:
        p50 = json.loads(Path(a.summary).read_text())["all"]["latency_ms"].get("p50")
    caption = caption_for(a.label, a.card or card_name(), p50)
    if not a.no_pace:
        caption += " | inspector slow motion"

    env = dict(os.environ, BH_TELEMETRY="0", BH_UPDATE_CHECK="0", BU_CDP_URL=f"http://127.0.0.1:{a.cdp_port}")
    env.setdefault("BU_NAME", f"record-{a.cdp_port}")  # one browser-harness daemon per run
    env.update(DEMO_PORT=str(a.demo_port), PYTHONPATH=str(ROOT))
    chrome, profile = launch_chrome(a.cdp_port, headed=False)
    demo = subprocess.Popen(
        [sys.executable, "-m", "jev_ultrafast.demo"],
        cwd=ROOT,
        env=env,
        stdout=open(out / "demo_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{a.demo_port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/api/state", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)

        def drive(page):
            # the inspector is taller than a screen; scale it so the decision panel and the trail are in the picture
            page.evaluate("(z) => { document.body.style.zoom = z; }", a.zoom)
            page.select_option("#scenario", a.task)
            page.fill("#goal", TASKS[a.task]["goal"])
            if a.no_pace:
                page.uncheck("#pace")
            else:
                page.check("#pace")
            page.click("#start")
            page.wait_for_selector("#auto:not([disabled])", timeout=60000)
            page.wait_for_timeout(1500)
            page.click("#auto")
            end = time.time() + a.max_seconds
            while time.time() < end:
                try:  # the inspector's server answers after a running step ends, which can take a few seconds
                    state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=30).read())
                except OSError:
                    continue
                if state.get("status") in ("done", "blocked"):
                    break
                page.wait_for_timeout(300)
            page.wait_for_timeout(3000)

        webm = record(
            base + "/",
            out / a.name,
            seconds=a.max_seconds + 10,
            size=(width, height),
            drive=drive,
            channel=a.channel or None,
            settle_s=1.0,
            until_drive_done=True,
        )
    finally:
        demo.terminate()
        chrome.terminate()
        try:
            demo.wait(timeout=10)
        except subprocess.TimeoutExpired:
            demo.kill()
        shutil.rmtree(profile, ignore_errors=True)
    to_mp4(webm, out / f"{a.name}.mp4", caption)
    gif = to_gif(webm, out / f"{a.name}.gif", caption, start_s=0.0)
    print(webm, out / f"{a.name}.mp4", gif, gif.stat().st_size, caption)


if __name__ == "__main__":
    main()
