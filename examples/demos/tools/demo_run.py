# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The run record and the statistics shared by the three demo measurement scripts.

A run record is a directory under `runs/` with `manifest.json`, `files.json` (size and sha256 of every file), a summary
and the raw per-run rows, as the other runs in this repository. The intervals are 95% bootstrap intervals over the
repeated runs (or over scenarios, where a script says so); latencies are pooled over every decision of every run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from decision_log import LOG_NAME, compress
from stats import bootstrap, git_state, latency_summary, write_files_json

REPO = Path(__file__).resolve().parents[3]

__all__ = ["bootstrap", "latency_summary", "DemoRun", "server_health", "card_name", "caption_for", "interval_text"]


def server_health(base_url: str) -> dict:
    """What a System One server says about itself (/health), for the record; the error text when it cannot say."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as r:
            info = json.loads(r.read())
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    keys = ("engine", "model", "mode", "block_size", "pad_unit", "adapters", "systemone", "head_engine")
    return {k: info.get(k) for k in keys}


def card_name() -> str:
    """The GPU, from nvidia-smi, or the machine (Apple chip) when there is none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if out:
            return out.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        return chip or "Mac"
    return platform.processor() or platform.machine()


def interval_text(b: dict, digits: int = 1, scale: float = 1.0) -> str:
    if b.get("est") is None:
        return "n/a"
    return f"{b['est'] * scale:.{digits}f} [{b['lo'] * scale:.{digits}f}, {b['hi'] * scale:.{digits}f}]"


def caption_for(model: str, card: str, p50_ms: float | None) -> str:
    """The text burned into a recording: the model, the card, the median latency."""
    median = "median n/a" if p50_ms is None else f"median {p50_ms:.0f} ms"
    return f"{model} | {card} | {median}"


class DemoRun:
    """One run record: write files into `out`, then `finish` it."""

    def __init__(self, out: str | Path, title: str, demo: str, model: str, base_url: str, hardware: str = ""):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.title, self.demo, self.model, self.base_url = title, demo, model, base_url
        self.hardware = hardware or card_name()
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.logf = open(self.out / "log.txt", "a")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def write_json(self, name: str, data) -> None:
        (self.out / name).write_text(json.dumps(data, indent=1) + "\n")

    @property
    def decision_log_path(self) -> Path:
        """Where this run's decision log is written (compressed and listed by `finish`)."""
        return self.out / LOG_NAME

    def finish(self, summary: dict, markdown: str, extra: dict) -> None:
        logged = compress(self.decision_log_path)
        if logged is not None:
            extra = {
                **extra,
                "files": {
                    **extra.get("files", {}),
                    logged.name: "every model decision: the request body and the full answer",
                },
            }
        self.write_json("summary.json", summary)
        (self.out / "summary.md").write_text(markdown + "\n")
        manifest = {
            "id": self.out.name,
            "title": self.title,
            "date": time.strftime("%Y-%m-%d"),
            "started": self.started,
            "submitted": False,
            "typesafe_api_calls": 0,
            "demo": self.demo,
            "model": self.model,
            "server": self.base_url,
            "server_health": server_health(self.base_url),
            "hardware": self.hardware,
            "code": git_state(REPO),
            "command": " ".join(sys.argv),
            "client_and_server_on_the_same_machine": True,
            **extra,
        }
        self.write_json("manifest.json", manifest)
        self.logf.close()
        write_files_json(self.out)
