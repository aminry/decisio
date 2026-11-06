# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast-table image variant answers with no network: run inside the image with the network off.

    docker run --rm --network none -e HF_HUB_OFFLINE=1 -v "$PWD/docker:/check:ro" --entrypoint python3 \\
        decisio:ci-table /check/check_table_offline.py

It checks that the Hub cannot be reached, then registers a choice task with a fast table of kind static (potion-base-8M)
and one of kind static with static_model multilingual (potion-multilingual-128M) through decisio's own registration
route, on a stand-in engine (no GPU: it answers from the words of the text, 30 ms a call, as a model would take). Each
registration must fit and keep its table, read from the image's students (DECISIO_STUDENTS_DIR), and a request the
table is sure of must be answered by it. Exit 0 when every check holds.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import zlib

import numpy as np

WORDS = {
    "billing": ["invoice", "refund", "charge", "payment", "receipt", "price"],
    "shipping": ["parcel", "courier", "delivery", "tracking", "address", "package"],
    "access": ["password", "login", "account", "reset", "locked", "token"],
}
CRIT = {"billing": "billing", "shipping": "shipping", "access": "access", "other": "none of these"}
INS = "Which queue does this message belong to?"


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class StandIn:
    """Puts 0.95 on the option whose words the text has, else on `other`; each call takes 30 ms."""

    tok = Tok()
    multi_question = "sequential"
    pad_policy = "always"

    def facts(self):
        return {"engine": "stand-in", "vllm": "none", "base": "stand-in", "quantization": None, "mode": "separate"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None):
        time.sleep(0.03)
        text = str(state).lower()
        out = []
        for q in questions:
            keys = list(CRIT)
            hit = next((i for i, k in enumerate(keys) if any(w in text for w in WORDS.get(k, []))), keys.index("other"))
            p = np.full(len(keys), 0.05 / (len(keys) - 1))
            p[hit] = 0.95
            out.append(p)
        return out, {}


def examples(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gold = str(rng.choice(list(WORDS)))
        words = list(rng.choice(WORDS[gold], 3, replace=False)) + [f"word{int(i)}" for i in rng.integers(0, 200, 5)]
        rng.shuffle(words)
        state = "hello, " + " ".join(words)
        q = {"q1": {"type": "choice", "instructions": INS, "criteria": CRIT}}
        out.append({"request": {"state": state, "questions": q}, "answer": gold})
    return out


def main() -> int:
    fails = []
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        fails.append("huggingface.co can be reached: run this with the network off (--network none)")
    except OSError:
        print("the Hub cannot be reached: good")
    root = os.environ.get("DECISIO_STUDENTS_DIR")
    print("DECISIO_STUDENTS_DIR =", root, "| HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    if not root or not os.path.isdir(root):
        fails.append("DECISIO_STUDENTS_DIR is not set to a directory")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne, add_routes
    from decisio.serve.tasks import TaskStore

    for i, fast_table in enumerate(({"kind": "static"}, {"kind": "static", "static_model": "multilingual"})):
        so = SystemOne(StandIn(), "decisio-offline-check", task_store=TaskStore("fp"))
        app = FastAPI()
        add_routes(app, so)
        client = TestClient(app)
        body = {"id": f"t{i}", "examples": examples(400, i), "certify": {"alpha": 0.1, "delta": 0.1},
                "fast_table": fast_table}  # fmt: skip
        r = client.post("/v1/tasks", json=body)
        name = fast_table.get("static_model", "base")
        if r.status_code != 200:
            fails.append(f"static {name}: registration {r.status_code}: {r.text[:400]}")
            continue
        t = r.json()
        report = t.get("fast_table_report") or {}
        state = ((t.get("status") or {}).get("fast_table") or {}).get("state")
        print(f"static {name}: kept={report.get('kept')} state={state} reason={report.get('reason')}")
        if not report.get("kept") or state != "active":
            fails.append(f"static {name}: the table was not kept and active ({report})")
            continue
        q = {"q1": {"type": "choice", "instructions": INS, "criteria": CRIT}}
        a = client.post("/v1/systemone", json={"state": "hello, my invoice and the refund", "questions": q})
        by = (a.json().get("answers", {}).get("q1") or {}).get("answered_by") if a.status_code == 200 else None
        print(f"static {name}: a sure request answered by {by}")
        if by != "table":
            fails.append(f"static {name}: a request the table is sure of was answered by {by} ({a.text[:300]})")
    if fails:
        print("FAILED:\n  " + "\n  ".join(fails), file=sys.stderr)
        return 1
    print("the fast-table variant registers and serves static tables with no network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
