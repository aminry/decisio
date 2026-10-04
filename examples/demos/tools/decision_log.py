# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Request-body and distribution logging for the demo run records.

Every model decision of a measured run is one JSON line: the request body the server received and the full answer (each
question's choice, confidence and every option's probability), with the latency. The demos write the lines where the
request is made (the driving demo's server, Pong's recorder, the browser agent's measurement script); `DemoRun.finish`
compresses the file and lists it in the manifest, so every record carries them.

A row: {"demo", "ts", ...demo fields, "request": {...}, "response": {...}, "latency_ms", "server_ms"}.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

LOG_NAME = "decisions.jsonl"


class DecisionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, **row) -> None:
        line = json.dumps({"ts": round(time.time(), 3), **row}, ensure_ascii=False)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


def compress(path: str | Path) -> Path | None:
    """`decisions.jsonl` to `decisions.jsonl.gz` (the plain file removed); None when nothing was logged."""
    path = Path(path)
    if not path.exists():
        return None
    gz = Path(str(path) + ".gz")
    with open(path, "rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
        dst.writelines(src)
    path.unlink()
    return gz


def read(path: str | Path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]
