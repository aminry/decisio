# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""End-to-end latency of a certified task with and without its fast table, through a running decisio server.

The task is registered certified on the labelled examples once per arm, one arm after the other: first with a
fast table of each student kind in --kinds (arm `table`, the cell's `kind` naming it; default `static`, what
`"fast_table": true` fits), each timed on every cell and deleted, then without a table (arm `model`), timed the same
way and deleted. A kind is `static`, `encoder` or `auto`, and `static:multilingual` (or `auto:multilingual`) names the
static model. A request matches a choice task by its type and options, so two arms cannot be registered at once. Each
arm reads its own held-out requests, never another's, so no request is helped by a prefix another arm cached. Each
cell is an arm, a state length (`short`: the request as it is; `long`: padded with filler to about --long-chars
characters) and a concurrency (requests in flight at once).

Per request it records the client's wall time, x-decisio-server-ms, the stages of x-decisio-stages (`task` is the
certified question's read: the table's, or decisio's when the table deferred or is off) and how the answer was served
(`answered_by`: table or decisio). Per cell: the medians and the 95th percentiles, and the share the table answered.
Per table arm, the registration's `fast_table_report` (for `auto`: every kind tried and the one kept).

    python -m decisio.bench.table_latency --url http://127.0.0.1:8000 --examples train.jsonl --probe probe.jsonl \\
        --out table_latency.json [--kinds static,encoder] [--concurrency 1,16] [--lengths short,long] \\
        [--per-cell 200] [--keep-tasks]

An examples file holds one {"request": {...}, "answer": ...} per line, as `POST /v1/tasks` takes them (the request's
single question is the task's). The probe file holds requests of the same kind, not among the examples.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import time
import urllib.error
import urllib.request

FILLER = (
    " Earlier in the thread the customer had written about something else entirely, and the agent's notes repeat "
    "the account details, the device, the plan and the dates of the previous contacts."
)


def http_post(url: str, path: str, body: dict, timeout: float = 600.0) -> tuple[int, dict, dict, float]:
    """(status, headers, json body, wall ms) of a POST through urllib."""
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data, status, headers = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        data, status, headers = e.read(), e.code, dict(e.headers)
    wall = (time.perf_counter() - t) * 1000
    try:
        payload = json.loads(data or b"{}")
    except json.JSONDecodeError:
        payload = {"raw": data[:300].decode(errors="replace")}
    return status, {k.lower(): v for k, v in headers.items()}, payload, wall


def http_delete(url: str, path: str) -> int:
    req = urllib.request.Request(url.rstrip("/") + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def question_of(request: dict) -> tuple[str, dict]:
    (name, q), *rest = request["questions"].items()
    if rest:
        raise ValueError("a request of this benchmark has one question")
    return name, q


def lengthen(request: dict, chars: int) -> dict:
    """The request with its state padded to about `chars` characters: a text state gets filler after its text, an object
    state a `notes` field of filler (its own fields kept as they are)."""
    out = copy.deepcopy(request)
    state = out.get("state")
    if isinstance(state, str):
        while len(out["state"]) < chars:
            out["state"] += FILLER
        return out
    notes = ""
    while len(json.dumps(state)) + len(notes) < chars:
        notes += FILLER
    out["state"] = {**state, "notes": notes}
    return out


def stages_of(headers: dict) -> dict:
    raw = headers.get("x-decisio-stages") or ""
    out = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def one(post, url: str, request: dict) -> dict:
    status, headers, body, wall = post(url, "/v1/systemone", request)
    row = {"status": status, "wall_ms": wall}
    if status == 200:
        _, ans = next(iter(body.get("answers", {}).items()))
        sm = headers.get("x-decisio-server-ms")
        row.update(
            server_ms=float(sm) if sm is not None else None,
            stages=stages_of(headers),
            answered_by=ans.get("answered_by"),
            certified=ans.get("certified"),
        )
    else:
        row["error"] = str(body)[:300]
    return row


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))], 1)


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    task = [r["stages"].get("task") for r in ok]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "wall_ms_p50": pct([r["wall_ms"] for r in ok], 50),
        "wall_ms_p95": pct([r["wall_ms"] for r in ok], 95),
        "server_ms_p50": pct([r["server_ms"] for r in ok], 50),
        "server_ms_p95": pct([r["server_ms"] for r in ok], 95),
        "task_ms_p50": pct(task, 50),
        "task_ms_p95": pct(task, 95),
        "table_share": round(sum(r.get("answered_by") == "table" for r in ok) / len(ok), 3) if ok else None,
        "certified_share": round(sum(bool(r.get("certified")) for r in ok) / len(ok), 3) if ok else None,
    }


def kind_request(kind: str) -> dict:
    """The registration's `fast_table` for a kind spec: `static`, `encoder`, `auto`, or `static:<model>` /
    `auto:<model>` for the static model."""
    name, _, model = kind.partition(":")
    return {"kind": name, **({"static_model": model} if model else {})}


def register(post, url: str, task_id: str, examples: list[dict], kind: str | None, certify: dict) -> dict:
    """The arm's task, registered certified: with a fast table of `kind`, or without one (None)."""
    for e in examples:
        question_of(e["request"])  # one question per request
    body = {
        "id": task_id,
        "examples": examples,
        "certify": certify,
        **({"fast_table": kind_request(kind)} if kind is not None else {}),
    }
    status, _, task, wall = post(url, "/v1/tasks", body)
    if status != 200:
        raise SystemExit(f"registering {task_id} failed ({status}): {str(task)[:400]}")
    return {
        "id": task_id,
        "registration_s": round(wall / 1000, 1),
        "certified": (task.get("certificate") or {}).get("certified"),
        "fast_table": (task.get("status") or {}).get("fast_table"),
        **({"kind": kind, "fast_table_report": task.get("fast_table_report")} if kind is not None else {}),
    }


def run(
    url,
    examples,
    probe,
    concurrency=(1, 16),
    lengths=("short", "long"),
    kinds=("static",),
    per_cell=200,
    long_chars=6000,
    certify=None,
    post=http_post,
    delete=http_delete,
    keep_tasks=False,
    prefix="bench",
) -> dict:
    """Each arm in turn (a table of each kind, then the model): register its task, time every cell on the arm's own
    requests, delete the task; the report. `tasks` is keyed `table` for one kind, `table:<kind>` for several."""
    certify = certify if certify is not None else {}
    kinds = tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds):
        raise SystemExit(f"--kinds needs one or more distinct kinds, not {','.join(kinds)!r}")
    for k in kinds:
        if kind_request(k)["kind"] not in ("static", "encoder", "auto"):
            raise SystemExit(f"unknown kind {k!r}: static, encoder or auto (static:<model>, auto:<model>)")
    arms = [("table" if len(kinds) == 1 else f"table:{k}", k) for k in kinds] + [("model", None)]
    cells_needed = len(lengths) * len(concurrency) * per_cell * len(arms)
    if len(probe) < cells_needed:
        raise SystemExit(f"the probe file has {len(probe)} requests; the cells need {cells_needed}")
    tasks, cells, used = {}, [], 0
    for key, kind in arms:
        arm = "model" if kind is None else "table"
        tid = f"{prefix}-{key.replace(':', '-')}"
        tasks[key] = register(post, url, tid, examples, kind, certify)
        try:
            for length in lengths:
                for c in concurrency:
                    batch = probe[used : used + per_cell]
                    used += per_cell
                    jobs = [lengthen(r, long_chars) if length == "long" else r for r in batch]
                    with cf.ThreadPoolExecutor(max_workers=c) as pool:
                        rows = list(pool.map(lambda r: one(post, url, r), jobs))
                    cells.append({"arm": arm, "kind": kind, "length": length, "concurrency": c, **summarise(rows)})
        finally:
            # the next arm registers the same question, so this arm's task goes first; --keep-tasks keeps the last
            if not keep_tasks or kind is not None:
                delete(url, f"/v1/tasks/{tid}")
    return {"tasks": tasks, "cells": cells, "kinds": list(kinds), "per_cell": per_cell, "long_chars": long_chars}


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--examples", required=True, help="labelled examples, one {request, answer} per line")
    ap.add_argument("--probe", required=True, help="held-out requests, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="static", help="the table arms' student kinds, e.g. static,encoder,auto")
    ap.add_argument("--concurrency", default="1,16")
    ap.add_argument("--lengths", default="short,long")
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--long-chars", type=int, default=6000)
    ap.add_argument("--keep-tasks", action="store_true")
    a = ap.parse_args(argv)
    report = run(
        a.url,
        read_jsonl(a.examples),
        [r.get("request", r) for r in read_jsonl(a.probe)],
        concurrency=tuple(int(x) for x in a.concurrency.split(",")),
        lengths=tuple(a.lengths.split(",")),
        kinds=tuple(a.kinds.split(",")),
        per_cell=a.per_cell,
        long_chars=a.long_chars,
        keep_tasks=a.keep_tasks,
    )
    report["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    for c in report["cells"]:
        print(
            f"{c['arm']:5s} {c['kind'] or '-':8s} {c['length']:5s} c={c['concurrency']:<3d} wall p50 {c['wall_ms_p50']}"
            f" p95 {c['wall_ms_p95']}  server p50 {c['server_ms_p50']}  task p50 {c['task_ms_p50']}"
            f"  table {c['table_share']}"
            f"  errors {c['errors']}"
        )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
