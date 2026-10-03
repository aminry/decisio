# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure the support-ticket triage demo (examples/demos/triage): one question, 20 queues, before and after teaching.

Against one System One server, in this order:

1. the plain arm: every held-out ticket, then every stream ticket, one `/v1/systemone` call each with the same question
   ("Which support queue should handle this ticket?", the 20 intents of taxonomy.json as criteria with descriptions);
2. registration: `POST /v1/tasks` with the 200 training tickets as labelled examples, timed, with what the server kept
   (calibration, the intent head) and why;
3. the taught arm: the same held-out and stream tickets again;
4. `DELETE /v1/tasks/{id}`, also when a step fails, so the server is left as it was found.

The run refuses to start if a task of that id is already registered, and stops before registering if the plain arm's
answers say a task already applies to the question (the `x-decisio-tasks` header). Held-out tickets run unpaced, one
at a time; stream tickets keep at least `--pace-ms` between their starts so a viewer can read them.

Reported: held-out accuracy before and after with 95% bootstrap intervals over tickets, and the paired difference with
its interval; macro-F1; per-intent accuracy; the registration's wall time; decisions per second and latency p50/p95
(unpaced held-out calls). The record (runs/<date>_demos-triage) carries manifest.json, files.json, summary.json and
summary.md, the per-ticket rows, the registration response, every decision's request and full answer, and two
trajectories of the stream, plain.jsonl.gz and taught.jsonl.gz, that render_triage.py draws.

    python measure_triage.py --url http://127.0.0.1:8000 --label Decisio --pace-ms 1500
    python measure_triage.py --url http://127.0.0.1:18773 --label "Qwen3-0.6B-Base (CPU stand-in)" --limit-heldout 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import trajectory
from decision_log import DecisionLog
from demo_run import DemoRun, bootstrap, card_name, interval_text, latency_summary, server_health

REPO = Path(__file__).resolve().parents[3]
TRIAGE = REPO / "examples" / "demos" / "triage"
QUESTION = "route"
TASK_ID = "notewell-triage"
TAUGHT_EXTRA = "tasks registered"


def load_taxonomy(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or TRIAGE / "taxonomy.json").read_text())


def question_for(taxonomy: dict) -> dict:
    """The one question every ticket is asked: the 20 intents in taxonomy order, each with its description."""
    return {
        "type": "choice",
        "instructions": taxonomy["instructions"],
        "criteria": {i["key"]: i["description"] for i in taxonomy["intents"]},
    }


def request_for(text: str, question: dict) -> dict:
    return {"state": text, "questions": {QUESTION: question}}


def load_split(name: str, data: str | Path | None = None) -> list[dict]:
    return json.loads((Path(data or TRIAGE / "data") / f"{name}.json").read_text())["tickets"]


def limit_stratified(tickets: list[dict], n: int | None) -> list[dict]:
    """The first `n` tickets taken in turn from each intent (file order within an intent), so a short check still
    covers every intent evenly; all of them when `n` is None."""
    if n is None or n >= len(tickets):
        return tickets
    by: dict[str, list[dict]] = {}
    for t in tickets:
        by.setdefault(t["intent"], []).append(t)
    out, k = [], 0
    while len(out) < n:
        for rows in by.values():
            if k < len(rows) and len(out) < n:
                out.append(rows[k])
        k += 1
    return out


def call(url: str, path: str, body=None, method: str | None = None, timeout: float = 3600):
    """(status, lower-cased headers, parsed body) of one HTTP call; HTTP errors are returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            detail = json.loads(raw)
        except ValueError:
            detail = raw.decode(errors="replace")
        return e.code, {k.lower(): v for k, v in e.headers.items()}, detail


def route(url: str, ticket: dict, question: dict, timeout: float) -> dict:
    """One ticket routed: the request as sent, the full answer or the error, the chosen intent, timings, the tasks the
    server says it applied."""
    req = request_for(ticket["text"], question)
    t_sent = time.time()
    t0 = time.perf_counter()
    try:
        status, headers, body = call(url, "/v1/systemone", req, timeout=timeout)
    except (OSError, ValueError) as e:
        status, headers, body = None, {}, str(e)
    latency_ms = (time.perf_counter() - t0) * 1000
    out = {"request": req, "t_sent": t_sent, "t_wall": time.time(), "latency_ms": latency_ms}
    answer = body.get("answers", {}).get(QUESTION) if status == 200 and isinstance(body, dict) else None
    if answer and "choice" in answer:
        out.update(answer=body["answers"], chosen=answer["choice"], response=body)
    else:
        out["error"] = f"status {status}: {body}"[:500]
    out["tasks"] = [t for t in headers.get("x-decisio-tasks", "").split(",") if t.strip()]
    out["server_ms"] = float(headers["x-decisio-server-ms"]) if "x-decisio-server-ms" in headers else None
    return out


# ---- scoring -------------------------------------------------------------------------------------------------------


def correct(gold: list[str], chosen: list[str | None]) -> np.ndarray:
    """1 where the chosen intent is the true one; a failed call (None) counts as wrong."""
    return np.array([int(c is not None and c == g) for g, c in zip(gold, chosen)], dtype=float)


def macro_f1(gold: list[str], chosen: list[str | None], labels: list[str]) -> float:
    """The mean over `labels` of each intent's F1 (an intent never predicted and never true is skipped)."""
    scores = []
    for k in labels:
        tp = sum(g == k and c == k for g, c in zip(gold, chosen))
        fp = sum(g != k and c == k for g, c in zip(gold, chosen))
        fn = sum(g == k and c != k for g, c in zip(gold, chosen))
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def compare_arms(gold: list[str], plain: list[str | None], taught: list[str | None], labels: list[str], seed=0) -> dict:
    """Accuracy of each arm with its 95% bootstrap interval over tickets, the paired difference (taught minus plain,
    the same tickets resampled for both), macro-F1, per-intent accuracy, and the changed answers."""
    before, after = correct(gold, plain), correct(gold, taught)
    per_intent = {}
    for k in labels:
        idx = [i for i, g in enumerate(gold) if g == k]
        if idx:
            per_intent[k] = {
                "n": len(idx),
                "plain": float(before[idx].mean()),
                "taught": float(after[idx].mean()),
            }
    return {
        "n": len(gold),
        "accuracy_plain": bootstrap(before, seed=seed),
        "accuracy_taught": bootstrap(after, seed=seed),
        "difference": bootstrap(after - before, seed=seed),
        "macro_f1_plain": macro_f1(gold, plain, labels),
        "macro_f1_taught": macro_f1(gold, taught, labels),
        "fixed": int(((after == 1) & (before == 0)).sum()),
        "broken": int(((after == 0) & (before == 1)).sum()),
        "per_intent": per_intent,
    }


def confusions(gold: list[str], chosen: list[str | None], top: int = 10) -> list[dict]:
    pairs = Counter((g, c) for g, c in zip(gold, chosen) if c != g)
    return [{"true": g, "chosen": c, "n": n} for (g, c), n in pairs.most_common(top)]


def registration_summary(task: dict, seconds: float, n_examples: int) -> dict:
    """What the server kept from the registration, in the fields a reader (and the renderer) needs."""
    cal, head = task.get("calibration") or {}, task.get("head") or {}
    out = {
        "seconds": round(seconds, 2),
        "n_examples": task.get("n_examples", n_examples),
        "per_option_min": task.get("per_option_min"),
        "calibration": {"applied": bool(cal.get("applied")), "reason": cal.get("reason")},
        "head": {"applied": bool(head.get("applied")), "reason": head.get("reason")},
    }
    for k in ("cv_acc_plain", "cv_acc_fitted", "cv_logloss_plain", "cv_logloss_fitted"):
        if k in cal:
            out["calibration"][k] = cal[k]
    if head.get("applied"):
        out["head"]["lambda"] = head.get("lambda")
    if head.get("cv_logloss"):
        out["head"]["cv_logloss"] = head["cv_logloss"]
    if task.get("mock"):
        out["mock"] = True  # tools/mock_systemone.py accepts a task and fits nothing
    return out


def registration_text(reg: dict | None) -> str:
    if not reg:
        return "no registration"
    kept = [k for k in ("head", "calibration") if reg.get(k, {}).get("applied")]
    what = " and ".join(kept) if kept else "nothing kept (served as before)"
    return f"{reg.get('n_examples')} examples registered in {reg['seconds']:.0f} s; kept: {what}"


# ---- the run -------------------------------------------------------------------------------------------------------


def run_split(url, tickets, question, arm, split, log: DecisionLog, run: DemoRun, pace_ms=0.0, timeout=600.0):
    """Route every ticket of a split in order; returns the per-ticket results and the split's wall time."""
    results = []
    t_start = time.perf_counter()
    last_start = None
    for n, t in enumerate(tickets, 1):
        if pace_ms and last_start is not None:
            wait = last_start + pace_ms / 1000 - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        last_start = time.perf_counter()
        r = route(url, t, question, timeout)
        results.append(r)
        log.add(
            demo="triage",
            arm=arm,
            split=split,
            ticket=t["id"],
            intent=t["intent"],
            request=r["request"],
            response=r.get("response"),
            error=r.get("error"),
            latency_ms=round(r["latency_ms"], 2),
            server_ms=r["server_ms"],
        )
        if n % 50 == 0 or n == len(tickets):
            right = sum(x.get("chosen") == tk["intent"] for x, tk in zip(results, tickets))
            run.log(f"  {arm} {split}: {n}/{len(tickets)} routed, {right} right")
    return results, time.perf_counter() - t_start


def stream_trajectory(path, tickets, results, player, run_info, taxonomy) -> str:
    ticks = []
    for i, (t, r) in enumerate(zip(tickets, results)):
        decision = {"request": r["request"], "latency_ms": round(r["latency_ms"], 2)}
        if "answer" in r:
            decision.update(answer=r["answer"], chosen=r["chosen"])
        else:
            decision["error"] = r["error"]
        decision["tasks_applied"] = r["tasks"]
        ticks.append(
            {
                "tick": i,
                "t_wall": r["t_wall"],
                "t_sent": r["t_sent"],
                "state": {"ticket_id": t["id"], "text": t["text"], "intent": t["intent"]},
                "decision": decision,
            }
        )
    right = sum(r.get("chosen") == t["intent"] for t, r in zip(tickets, results))
    head = trajectory.header(
        "triage",
        player,
        run_info,
        question=QUESTION,
        instructions=taxonomy["instructions"],
        intents=taxonomy["intents"],
    )
    end = {"tickets": len(tickets), "right": right, "accuracy": right / len(tickets) if tickets else None}
    return trajectory.write(path, head, ticks, end)


def latency_block(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if "answer" in r]
    return {
        "latency_ms": latency_summary([r["latency_ms"] for r in ok]),
        "server_ms": latency_summary([r["server_ms"] for r in ok if r["server_ms"] is not None]),
        "decisions_per_s": len(ok) / wall_s if wall_s else None,
        "wall_s": wall_s,
        "errors": len(results) - len(ok),
        "tasks_header": dict(Counter(",".join(r["tasks"]) or "none" for r in results)),
    }


def markdown(s: dict) -> str:
    h, reg = s["heldout"], s["registration"]
    pct = 100.0

    def acc(b):
        return interval_text(b, 1, pct)

    lines = [
        f"# Support-ticket triage: {s['label']}",
        "",
        f"{h['n']} held-out tickets over {len(h['per_intent'])} intents, asked one at a time before and after "
        f"registering {reg.get('n_examples')} labelled training tickets ({s['card']}); "
        "client and server on the same machine. Intervals: 95% bootstrap over tickets; the difference is paired.",
        "",
        "| | plain | tasks registered | difference (points) |",
        "| --- | --- | --- | --- |",
        f"| accuracy (%) | {acc(h['accuracy_plain'])} | {acc(h['accuracy_taught'])} | {acc(h['difference'])} |",
        f"| macro-F1 | {h['macro_f1_plain']:.3f} | {h['macro_f1_taught']:.3f} | |",
        f"| changed answers | | {h['fixed']} fixed, {h['broken']} broken | |",
    ]
    for arm in ("plain", "taught"):
        lat = s["latency"][arm]["heldout"]
        lines.append(
            f"| {arm}: decisions/s, latency p50 / p95 ms | {lat['decisions_per_s']:.2f}, "
            f"{lat['latency_ms'].get('p50', float('nan')):.0f} / {lat['latency_ms'].get('p95', float('nan')):.0f} | | |"
        )
    lines += [
        "",
        f"Registration: {registration_text(reg)}; the taught arm's answers named the task on "
        f"{s['heldout_matched_task']} of {h['n']} held-out tickets (`x-decisio-tasks`).",
        f"- calibration: {'applied' if reg['calibration']['applied'] else 'not applied'} "
        f"({reg['calibration'].get('reason')})",
        f"- intent head: {'applied' if reg['head']['applied'] else 'not applied'} ({reg['head'].get('reason')})",
        "",
        f"Stream ({s['stream']['n']} tickets, paced at {s['pace_ms']:.0f} ms): "
        f"plain {s['stream']['right_plain']} right, tasks registered {s['stream']['right_taught']} right.",
        "",
        "| intent | n | plain | tasks registered |",
        "| --- | --- | --- | --- |",
    ]
    for k, v in h["per_intent"].items():
        lines.append(f"| {k} | {v['n']} | {v['plain']:.2f} | {v['taught']:.2f} |")
    for arm in ("plain", "taught"):
        lines += ["", f"Most frequent confusions, {arm} (true -> chosen: count):", ""]
        lines += [f"- {c['true']} -> {c['chosen']}: {c['n']}" for c in s["confusions"][arm]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the System One server")
    ap.add_argument("--label", default="Decisio", help="the player's name in the record and on screen")
    ap.add_argument("--player", default=None, help="JSON file of the player's structured fields (trajectory.py)")
    ap.add_argument("--out", default=None, help="run directory (default runs/<date>_demos-triage)")
    ap.add_argument("--task-id", default=TASK_ID)
    ap.add_argument("--data", default=str(TRIAGE / "data"), help="directory of train.json, heldout.json, stream.json")
    ap.add_argument("--taxonomy", default=str(TRIAGE / "taxonomy.json"))
    ap.add_argument("--pace-ms", type=float, default=0.0, help="least time between two stream tickets' starts")
    ap.add_argument("--limit-heldout", type=int, default=None, help="route only N held-out tickets, N/20 per intent")
    ap.add_argument("--limit-stream", type=int, default=None, help="route only the first N stream tickets")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-call timeout in seconds")
    ap.add_argument("--card", default="", help="the card that runs the server (default: this machine's)")
    a = ap.parse_args()

    taxonomy = load_taxonomy(a.taxonomy)
    labels = [i["key"] for i in taxonomy["intents"]]
    question = question_for(taxonomy)
    train = load_split("train", a.data)
    heldout = limit_stratified(load_split("heldout", a.data), a.limit_heldout)
    stream = load_split("stream", a.data)[: a.limit_stream]
    if {t["text"] for t in train} & {t["text"] for t in heldout + stream}:
        sys.exit("a training ticket is also a held-out or stream ticket; the splits must be disjoint")

    health = server_health(a.url)
    if "error" in health:
        sys.exit(f"{a.url}/health: {health['error']}")
    card = a.card or card_name()
    player = trajectory.player_fields(a.player, label=a.label, model_id=health.get("model"), card=card, server=health)
    out = Path(a.out) if a.out else REPO / "runs" / f"{time.strftime('%Y-%m-%d')}_demos-triage"
    run = DemoRun(out, "Support-ticket triage, before and after teaching", "triage", a.label, a.url, hardware=card)
    log = DecisionLog(run.decision_log_path)

    status, _, listing = call(a.url, "/v1/tasks")
    if status == 200 and any(t.get("id") == a.task_id for t in (listing or {}).get("tasks", [])):
        sys.exit(f"a task {a.task_id!r} is already registered on {a.url}; this run would replace it (use --task-id)")

    results: dict[str, dict[str, list]] = {"plain": {}, "taught": {}}
    walls: dict[str, dict[str, float]] = {"plain": {}, "taught": {}}
    reg = None
    registered = False
    try:
        run.log(f"plain arm: {len(heldout)} held-out tickets, then {len(stream)} stream tickets")
        results["plain"]["heldout"], walls["plain"]["heldout"] = run_split(
            a.url, heldout, question, "plain", "heldout", log, run, timeout=a.timeout
        )
        applied = {t for r in results["plain"]["heldout"] for t in r["tasks"]}
        if applied:
            sys.exit(f"tasks {sorted(applied)} already apply to this question on {a.url}; the plain arm is not plain")
        results["plain"]["stream"], walls["plain"]["stream"] = run_split(
            a.url, stream, question, "plain", "stream", log, run, a.pace_ms, a.timeout
        )

        run.log(f"registering {len(train)} training tickets as task {a.task_id!r}")
        body = {
            "id": a.task_id,
            "examples": [{"request": request_for(t["text"], question), "answer": t["intent"]} for t in train],
        }
        t0 = time.perf_counter()
        registered = True  # from here on the task may exist on the server (even if this call times out)
        status, _, task = call(a.url, "/v1/tasks", body, timeout=max(a.timeout, 7200))
        seconds = time.perf_counter() - t0
        if status != 200:
            sys.exit(f"registration refused ({status}): {task}")
        reg = registration_summary(task, seconds, len(train))
        run.write_json("registration.json", {"seconds": seconds, "response": task})
        run.log(registration_text(reg))

        run.log(f"taught arm: {len(heldout)} held-out tickets, then {len(stream)} stream tickets")
        results["taught"]["heldout"], walls["taught"]["heldout"] = run_split(
            a.url, heldout, question, "taught", "heldout", log, run, timeout=a.timeout
        )
        results["taught"]["stream"], walls["taught"]["stream"] = run_split(
            a.url, stream, question, "taught", "stream", log, run, a.pace_ms, a.timeout
        )
    finally:
        if registered:
            status, _, gone = call(a.url, f"/v1/tasks/{a.task_id}", method="DELETE")
            run.log(f"task {a.task_id!r} deleted ({status}: {gone})")

    gold = [t["intent"] for t in heldout]
    chosen = {arm: [r.get("chosen") for r in results[arm]["heldout"]] for arm in results}
    held = compare_arms(gold, chosen["plain"], chosen["taught"], labels)
    stream_gold = [t["intent"] for t in stream]
    stream_chosen = {arm: [r.get("chosen") for r in results[arm]["stream"]] for arm in results}
    matched = sum(a.task_id in r["tasks"] for r in results["taught"]["heldout"])
    summary = {
        "label": a.label,
        "card": card,
        "url": a.url,
        "task_id": a.task_id,
        "pace_ms": a.pace_ms,
        "heldout": held,
        "heldout_matched_task": matched,
        "registration": reg,
        "stream": {
            "n": len(stream),
            "right_plain": int(correct(stream_gold, stream_chosen["plain"]).sum()),
            "right_taught": int(correct(stream_gold, stream_chosen["taught"]).sum()),
        },
        "latency": {
            arm: {split: latency_block(results[arm][split], walls[arm][split]) for split in ("heldout", "stream")}
            for arm in results
        },
        "confusions": {arm: confusions(gold, chosen[arm]) for arm in results},
    }
    rows = []
    for i, t in enumerate(heldout):
        row = {"id": t["id"], "intent": t["intent"]}
        for arm in results:
            r = results[arm]["heldout"][i]
            probs = (r.get("answer") or {}).get(QUESTION, {}).get("probabilities", {})
            row[arm] = {
                "chosen": r.get("chosen"),
                "p_true": probs.get(t["intent"]),
                "latency_ms": round(r["latency_ms"], 2),
                "tasks": r["tasks"],
                **({"error": r["error"]} if "error" in r else {}),
            }
        rows.append(row)
    run.write_json("heldout_rows.json", rows)

    traj_files = {}
    for arm in results:
        run_info = {
            "arm": arm,
            "split": "stream",
            "task_id": a.task_id if arm == "taught" else None,
            "registration": reg if arm == "taught" else None,
            "pace_ms": a.pace_ms,
            "tickets": len(stream),
        }
        sha = stream_trajectory(out / f"{arm}.jsonl.gz", stream, results[arm]["stream"], player, run_info, taxonomy)
        traj_files[f"{arm}.jsonl.gz"] = {"sha256": sha}
        run.log(
            f"{arm}.jsonl.gz: {sum(r.get('chosen') == g for r, g in zip(results[arm]['stream'], stream_gold))}"
            f"/{len(stream)} stream tickets right"
        )
    md = markdown(summary)
    run.finish(
        summary,
        md,
        {
            "player": player,
            "task_id": a.task_id,
            "pace_ms": a.pace_ms,
            "limit_heldout": a.limit_heldout,
            "limit_stream": a.limit_stream,
            "data": {
                name: {"n": len(rows_), "sha256": trajectory.file_sha256(Path(a.data) / f"{name}.json")}
                for name, rows_ in (("train", train), ("heldout", heldout), ("stream", stream))
            },
            "files": {
                "summary.json, summary.md": "held-out accuracy before and after with intervals, macro-F1, per-intent "
                "accuracy, registration, latency",
                "heldout_rows.json": "per held-out ticket and arm: the chosen intent, the true intent's probability, "
                "latency, the tasks the server applied",
                "registration.json": "the registration's wall time and the server's response",
                "plain.jsonl.gz, taught.jsonl.gz": "the stream as trajectories (trajectory.py), the renderer's input",
                "log.txt": "the run's progress",
            },
            "trajectories": traj_files,
        },
    )
    print(md)


if __name__ == "__main__":
    main()
