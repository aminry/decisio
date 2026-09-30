# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Shared by the walk-through's scripts: reading labelled tickets, the question, and talking to a decisio server.
Standard library only, so the scripts run with nothing but Python."""

import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
QUESTION_NAME = "route"  # the question's name inside each request


def load_rows(path):
    """[(text, label)] from a CSV with `text` and `label` columns, or a JSONL of {"text": ..., "label": ...}."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    return [(r["text"], r["label"]) for r in rows]


def load_question(path=None):
    return json.loads(Path(path or HERE / "question.json").read_text())


def request_for(text, question):
    """One /v1/systemone request: the ticket is the state, the recurring question is the only question."""
    return {"state": text, "questions": {QUESTION_NAME: question}}


def call(url, path, body=None, method=None, timeout=3600):
    """(status, headers, parsed JSON body) of one HTTP call; HTTP errors are returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            detail = json.loads(raw)
        except ValueError:
            detail = raw.decode(errors="replace")
        return e.code, dict(e.headers), detail


def ask(url, text, question):
    """(choice, probabilities, headers) for one ticket."""
    status, headers, body = call(url, "/v1/systemone", request_for(text, question))
    if status != 200:
        raise SystemExit(f"/v1/systemone answered {status}: {body}")
    answer = body["answers"][QUESTION_NAME]
    return answer["choice"], answer["probabilities"], {k.lower(): v for k, v in headers.items()}
