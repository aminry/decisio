# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The task-registration walk-through (examples/tasks) end to end against the CPU stand-in, as its README runs it.

  E1  the committed data is what make_tickets.py writes, and examples.json is train.csv in the request format
  E0  the smoke (fast tier): the same steps as E2 on a 3-queue version of the question, 4 tickets per queue registered
      and 1 held out, so calibration is fitted and the head is declined (fewer than 10 options) in seconds
  E2  (slow) all 12 queues, 6 tickets per queue registered and 2 held out: a server started with --backend hf takes the
      registration, evaluate.py reports paired accuracy with an interval, ask.py shows x-decisio-tasks naming the
      matched task; the task lists, exports, deletes and imports
  E3  (slow) a server started with --tasks-file on the export serves the task from its first request

The stand-in's numbers mean nothing about the served model.

    uv run pytest -q tests/unit/test_tasks_walkthrough.py
"""

import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "examples" / "tasks"
MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def subset(src, per_label, dst, labels=None):
    rows, seen = [], defaultdict(int)
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            if (labels is None or r["label"] in labels) and seen[r["label"]] < per_label:
                seen[r["label"]] += 1
                rows.append(r)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return dst


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as r:
        return json.loads(r.read())


class Server:
    """`python -m decisio.serve.vllm_engine --backend hf` in a child process, stopped by its own PID."""

    def __init__(self, *extra):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [sys.executable, "-m", "decisio.serve.vllm_engine", "--backend", "hf", "--model", MODEL]
        self.proc = subprocess.Popen(
            [*cmd, "--port", str(self.port), *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError("the stand-in server exited:\n" + self.proc.stdout.read())
            try:
                get(self.url, "/health")
                return
            except OSError:
                time.sleep(1)
        self.stop()
        raise AssertionError("the stand-in server did not come up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def test_e1_committed_data_matches_its_generators(tmp_path):
    run(EX / "make_tickets.py", "--check")
    out = tmp_path / "examples.json"
    run(EX / "register.py", "--id", "ticket-routing", "--data", EX / "train.csv", "--dry-run", "--out", out)
    assert json.loads(out.read_text()) == json.loads((EX / "examples.json").read_text())


@pytest.fixture(scope="module")
def server():
    s = Server()
    yield s
    s.stop()


SMOKE_QUEUES = ("refund", "login", "bug")


def post(url, path, body):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def delete(url, path):
    urllib.request.urlopen(urllib.request.Request(url + path, method="DELETE"), timeout=60).read()


def test_e0_smoke_register_ask_evaluate_list_export_delete_import(server, tmp_path):
    question = json.loads((EX / "question.json").read_text())
    question["criteria"] = {k: question["criteria"][k] for k in SMOKE_QUEUES}
    (tmp_path / "question.json").write_text(json.dumps(question))
    q = ["--question", tmp_path / "question.json"]
    train = subset(EX / "train.csv", 4, tmp_path / "train.csv", SMOKE_QUEUES)
    held = subset(EX / "heldout.csv", 1, tmp_path / "heldout.csv", SMOKE_QUEUES)
    out = run(EX / "evaluate.py", "--url", server.url, "--id", "smoke", "--train", train, "--heldout", held, *q)
    assert "task 'smoke': 12 examples, 3 options, at least 4 per option" in out, out
    assert "calibration:" in out and "head: not applied" in out and "95% interval" in out, out
    shown = run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1", *q)
    assert "x-decisio-tasks: smoke" in shown, shown
    export = get(server.url, "/v1/tasks?full=1")
    assert [t["id"] for t in export["tasks"]] == ["smoke"]
    delete(server.url, "/v1/tasks/smoke")
    assert get(server.url, "/v1/tasks")["tasks"] == []
    assert "x-decisio-tasks: none" in run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1", *q)
    assert post(server.url, "/v1/tasks/import", {"tasks": export["tasks"]})["loaded"] == ["smoke"]
    assert "x-decisio-tasks: smoke" in run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1", *q)
    delete(server.url, "/v1/tasks/smoke")


@pytest.mark.slow
def test_e2_register_ask_evaluate_list_export_delete_import(server, tmp_path):
    train = subset(EX / "train.csv", 6, tmp_path / "train.csv")
    held = subset(EX / "heldout.csv", 2, tmp_path / "heldout.csv")
    out = run(EX / "evaluate.py", "--url", server.url, "--id", "ticket-routing", "--train", train, "--heldout", held)
    assert "task 'ticket-routing': 72 examples, 12 options, at least 6 per option" in out, out
    assert "calibration:" in out and "head:" in out and "95% interval" in out, out
    listed = get(server.url, "/v1/tasks")["tasks"]
    assert [t["id"] for t in listed] == ["ticket-routing"]
    shown = run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1")
    assert "x-decisio-tasks: ticket-routing" in shown, shown
    export = get(server.url, "/v1/tasks?full=1")
    (tmp_path / "tasks.json").write_text(json.dumps(export))
    req = urllib.request.Request(server.url + "/v1/tasks/ticket-routing", method="DELETE")
    urllib.request.urlopen(req, timeout=60).read()
    assert get(server.url, "/v1/tasks")["tasks"] == []
    assert "x-decisio-tasks: none" in run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1")
    body = json.dumps({"tasks": export["tasks"]}).encode()
    req = urllib.request.Request(
        server.url + "/v1/tasks/import", data=body, headers={"Content-Type": "application/json"}
    )
    assert json.loads(urllib.request.urlopen(req, timeout=60).read())["loaded"] == ["ticket-routing"]
    assert "x-decisio-tasks: ticket-routing" in run(EX / "ask.py", "--url", server.url, "--file", held, "--limit", "1")
    server.exported = tmp_path / "tasks.json"


@pytest.mark.slow
def test_e3_tasks_file_at_start(server):
    exported = getattr(server, "exported", None)
    if exported is None:
        pytest.skip("needs E2's export")
    server.stop()  # one stand-in at a time
    again = Server("--tasks-file", str(exported))
    try:
        assert [t["id"] for t in get(again.url, "/v1/tasks")["tasks"]] == ["ticket-routing"]
    finally:
        again.stop()
