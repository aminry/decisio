# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A deterministic stand-in for a System One server, for testing the demos' plumbing (not a model).

Every choice question gets a distribution derived from a hash of the request's state and the question, so the same
request always gets the same answer; `--prefer KEY` gives an option of that key most of the mass when it is offered
(e.g. `--prefer drive` keeps a driving test moving). `POST /v1/tasks` accepts and lists tasks (nothing is fitted).
Stdlib only.

    python mock_systemone.py --port 18765 [--prefer drive] [--delay-ms 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(body: dict, prefer: str | None) -> dict:
    state = json.dumps(body.get("state"), sort_keys=True)
    out = {}
    for name, q in (body.get("questions") or {}).items():
        if q.get("type") != "choice":
            out[name] = {"type": q.get("type"), "error": "the mock answers choice questions only"}
            continue
        keys = list(q.get("criteria") or {})
        raw = [int(hashlib.sha256(f"{state}|{name}|{k}".encode()).hexdigest()[:8], 16) % 1000 + 1 for k in keys]
        if prefer in keys:
            raw[keys.index(prefer)] += 4 * sum(raw)
        total = sum(raw)
        probs = {k: r / total for k, r in zip(keys, raw)}
        choice = max(probs, key=probs.get)
        out[name] = {"type": "choice", "choice": choice, "confidence": probs[choice], "probabilities": probs}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--prefer", default=None)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    a = ap.parse_args()
    tasks: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": "mock-systemone", "engine": "mock"})
            if self.path.startswith("/v1/tasks"):
                return self._send(200, {"tasks": list(tasks.values())})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if self.path == "/v1/systemone":
                time.sleep(a.delay_ms / 1000)
                return self._send(200, {"model": "mock-systemone", "answers": answer(body, a.prefer)})
            if self.path == "/v1/tasks":
                tasks[body["id"]] = {"id": body["id"], "examples": len(body.get("examples", [])), "mock": True}
                return self._send(200, tasks[body["id"]])
            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            tid = self.path.rsplit("/", 1)[-1]
            return self._send(200, {"removed": tasks.pop(tid, {}).get("id")})

        def log_message(self, *_):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
