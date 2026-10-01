# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Record the driving demo with its decision inspector open: the pilot's options with their probabilities, the
latency, p50 and decisions per minute, and the inspector's JSON (the request and the answer).

It starts the demo's server against `--base-url`, picks a seeded scenario with signals and stops (the same suite the
measurement uses), replays it in the 3D sim with the model as the pilot, and records the page in headless Chrome.
Client and server run on the same machine, so the latency on screen is the server's.

    python record_fsd.py --base-url http://127.0.0.1:8100 --label Decisio --out runs/<run>/media --seconds 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from demo_recorder import record, to_gif, to_mp4
from demo_run import caption_for, card_name
from measure_fsd import FSD, wait_for

PICK_JS = """() => {
    const last = window.__bench.last;
    let best = 0, score = -1;
    last.suite.forEach((s, i) => {
        const v = s.tags.signals + s.tags.stops;
        if (v > score) { score = v; best = i; }
    });
    const sc = last.suite[best];
    const replay = { scenario: sc, brain: 'jev', npcs: 40, weather: 'dry', bbox: last.config.bbox };
    localStorage.setItem('jev-fsd-replay', JSON.stringify(replay));
    return { id: sc.id, tags: sc.tags, bbox: last.config.bbox };
}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-api", default="")
    ap.add_argument("--out", required=True, help="directory for the WebM, MP4 and GIF")
    ap.add_argument("--name", default="fsd")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=6, help="suite size to pick the scenario from")
    ap.add_argument("--port", type=int, default=8323)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--summary", default=None, help="a measure_fsd summary.json, for the median in the caption")
    ap.add_argument("--p50", type=float, default=None)
    ap.add_argument("--card", default="")
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--quality", choices=["low", "high", "ultra"], default="high", help="the demo's graphics setting")
    ap.add_argument("--gif-start", type=float, default=6.0)
    a = ap.parse_args()

    width, height = (int(x) for x in a.size.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p50 = a.p50
    if a.summary:
        p50 = json.loads(Path(a.summary).read_text())["latency_ms"].get("p50")
    caption = caption_for(a.label, a.card or card_name(), p50)

    env = dict(os.environ, SYSTEMONE_BASE_URL=a.base_url, PORT=str(a.port))
    if a.model_api:
        env["SYSTEMONE_MODEL"] = a.model_api
    server = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=FSD,
        env=env,
        stdout=open(out / "demo_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{a.port}"
    try:
        wait_for(f"{base}/api/status")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            launch = {"headless": True}
            if a.channel:
                launch["channel"] = a.channel
            browser = p.chromium.launch(**launch)
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.set_default_timeout(0)
            page.goto(f"{base}/bench")
            for _ in range(240):
                if page.evaluate(
                    "() => !!window.__bench && document.querySelector('#map-note').textContent.length > 0"
                ):
                    break
                page.wait_for_timeout(250)
            # the suite is built with the code-only brain, in lockstep, which takes seconds;
            # the model then drives one of its scenarios in the sim
            page.evaluate(
                "(o) => window.__bench.run(o)",
                {"brain": "rules", "count": a.count, "npcs": 40, "seed": a.seed, "mode": "lockstep", "save": False},
            )
            picked = page.evaluate(PICK_JS)
            state = context.storage_state()
            browser.close()
        (out / "scenario.json").write_text(json.dumps(picked, indent=1) + "\n")

        def drive(page):
            page.evaluate("() => document.querySelector('#panel-toggle').click()")

        url = f"{base}/?replay=1&quality={a.quality}&bbox={','.join(str(x) for x in picked['bbox'])}"
        webm = record(
            url,
            out / a.name,
            seconds=a.seconds,
            size=(width, height),
            drive=drive,
            channel=a.channel or None,
            storage_state=state,
            settle_s=4.0,
            chrome_args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"],
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    to_mp4(webm, out / f"{a.name}.mp4", caption)
    gif = to_gif(webm, out / f"{a.name}.gif", caption, start_s=a.gif_start)
    print(webm, out / f"{a.name}.mp4", gif, gif.stat().st_size, caption)


if __name__ == "__main__":
    main()
