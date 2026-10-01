# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The measurement and recording helpers of examples/demos/tools, on data that needs no server and no browser.

D1  a Pong recording is turned into per-lane latencies, returns and misses
D2  a demo run record has a manifest and a files.json whose sha256 are the files', and says no hosted API was called
D3  the caption and the interval text read as the recordings print them
D4  a caption strip is stacked under the video, and a GIF of a busy picture ends up under the size limit
D5  the browser agent's summary counts options per question and completion with an interval
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "examples" / "demos" / "tools"
sys.path.insert(0, str(TOOLS))

import demo_recorder  # noqa: E402
import demo_run  # noqa: E402
import measure_pong  # noqa: E402
import measure_ultrafast  # noqa: E402

FFMPEG = shutil.which("ffmpeg") is not None


def snap(t, latency, vx, score=(0, 0)):
    return {"t": t, "latencyMs": latency, "ball": {"vx": vx, "vy": 0}, "score": list(score)}


def test_d1_lane_rows_counts_returns_misses_and_latencies():
    replay = {
        "lanes": [
            {
                "model": "sys1",
                "label": "Decisio",
                "snapshots": [snap(0, None, 1), snap(100, 80, 1), snap(200, 90, -1), snap(300, 70, 1, (0, 1))],
            }
        ]
    }
    row = measure_pong.lane_rows(replay)["sys1"]
    assert row["latency_ms"] == [80, 90, 70] and row["decisions"] == 3
    assert row["returns"] == 2 and row["misses"] == 1
    assert row["duration_s"] == pytest.approx(0.3) and row["final_score"] == [0, 1]


def test_d2_run_record_has_a_manifest_and_matching_hashes(tmp_path):
    run = demo_run.DemoRun(tmp_path / "r", "t", "pong", "Decisio", "http://127.0.0.1:1", hardware="a card")
    run.write_json("runs.json", [1, 2, 3])
    run.finish({"a": 1}, "# summary", {"files": {"runs.json": "rows"}})
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["typesafe_api_calls"] == 0 and manifest["hardware"] == "a card"
    assert manifest["client_and_server_on_the_same_machine"] is True and manifest["demo"] == "pong"
    assert "error" in manifest["server_health"]
    listed = json.loads((tmp_path / "r" / "files.json").read_text())
    names = {row["path"] for row in listed}
    assert {"runs.json", "summary.json", "summary.md", "manifest.json", "log.txt"} <= names
    for row in listed:
        assert hashlib.sha256((tmp_path / "r" / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_d3_caption_and_interval_text():
    assert demo_run.caption_for("Decisio", "RTX PRO 6000", 66.4) == "Decisio | RTX PRO 6000 | median 66 ms"
    assert demo_run.caption_for("Decisio", "card", None).endswith("median n/a")
    assert demo_run.interval_text({"est": 0.5, "lo": 0.25, "hi": 0.75}, 0, 100) == "50 [25, 75]"
    assert demo_run.interval_text({"est": None, "lo": None, "hi": None}) == "n/a"


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")
def test_d4_caption_strip_and_gif_size(tmp_path):
    webm = tmp_path / "in.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=15",
            "-t",
            "3",
            str(webm),
        ],
        check=True,
    )
    mp4 = demo_recorder.to_mp4(webm, tmp_path / "out.mp4", "Decisio | card | median 66 ms")
    size = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=height",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert int(size) == 360 + demo_recorder.STRIP_H
    gif = demo_recorder.to_gif(webm, tmp_path / "out.gif", "caption", max_gif_bytes=3_000_000, max_s=3)
    assert 0 < gif.stat().st_size <= 3_000_000


def test_d5_agent_summary_counts_options_and_completion():
    def rec(ok, ms, counts):
        decision = {"latency_ms": 100, "operation": "CLICK", "option_counts": counts}
        return {"task": "t", "ok": ok, "elapsed_ms": ms, "decisions": [decision], "text_calls": []}

    s = measure_ultrafast.summarise(
        [rec(True, 1000, {"operation": 7, "click_target": 9}), rec(False, 500, {"operation": 5})]
    )
    assert s["completed"] == 1 and s["runs"] == 2 and s["largest_option_set"] == 9
    assert s["options_per_question"]["operation"] == {"max": 7, "mean": 6.0}
    assert s["completion_ms"]["est"] == 1000 and s["latency_ms"]["p50"] == 100


def test_d6_replays_recorded_one_model_at_a_time_merge_into_lanes():
    import compose_pong

    a = {"seed": 1, "lanes": [{"model": "sys1", "label": "A", "snapshots": [{"t": 0, "latencyMs": 50}]}]}
    b = {"seed": 1, "lanes": [{"model": "cmp", "label": "B", "snapshots": [{"t": 0, "latencyMs": 90}]}]}
    merged = compose_pong.merge_replays([a, b])
    assert [lane["model"] for lane in merged["lanes"]] == ["sys1", "cmp"]
    assert compose_pong.lane_medians(merged) == [("A", 50.0), ("B", 90.0)]
    with pytest.raises(ValueError, match="share a seed"):
        compose_pong.merge_replays([a, {**b, "seed": 2}])
    with pytest.raises(ValueError, match="two recordings"):
        compose_pong.merge_replays([a, a])
