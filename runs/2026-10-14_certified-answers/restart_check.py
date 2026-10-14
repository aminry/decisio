"""Does a certificate hold across fresh restarts of the server? (decisio.readout.certify, POST /v1/tasks with a
certify block)

Starts the server as a new process several times on one backend and model, and each time either (A) registers
the task with
certification again from the same labelled examples, or (B) loads the task exported from the first start. Then
it asks the same
questions and compares what the restarts say: the certificate's threshold and serving-stack digest, and, for
every question,
the choice, the probability and whether the answer was certified.

    python restart_check.py --backend hf --model Qwen/Qwen3-0.6B-Base --out results/cpu_stand_in [--restarts 3]
    python restart_check.py --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit --out
    results/mac_mlx
    python restart_check.py --backend vllm --base tachara-ai/decisio-gemma-4-12b@<revision> --out
    results/card_12b   (on a card)

The task is a three-queue support-ticket question on tickets the script writes itself from templates (no data file, no
download; `--mixed` makes a share of them mention a second queue after the first, which the label ignores): by
default 400
labelled examples for registration (half held out of the fit by the server) and 200 questions to ask. Nothing here is a
measurement of a served model's quality: the stand-in's numbers mean nothing about the models decisio serves.
Standard library only.
"""

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUEUES = {
    "billing": "Payments, invoices, refunds and charges",
    "shipping": "Delivery, tracking, lost or late parcels",
    "access": "Login, passwords and account lockouts",
}
OPENERS = ["Hello,", "Hi team,", "Good morning,", "Urgent:", "Quick question:", "Please help:", "Dear support,", "Hey,"]
BODIES = {
    "billing": [
        "I was charged twice for my last invoice",
        "please refund the payment from last week",
        "my card was billed the wrong amount",
        "I need a copy of the invoice for my subscription",
        "there is an unknown charge on my statement",
        "the refund has not reached my bank account",
        "can I change the payment method on my plan",
        "why did the price on my bill go up",
    ],
    "shipping": [
        "my parcel has not arrived yet",
        "the tracking page shows no update for days",
        "the courier left the package at the wrong address",
        "my order is late and I need it this week",
        "the delivery was marked as done but I got nothing",
        "I want to change the delivery address of my order",
        "the box arrived damaged and open",
        "when will my shipment leave the warehouse",
    ],
    "access": [
        "I cannot log in to my account",
        "I forgot my password and the reset email never comes",
        "my account is locked after too many attempts",
        "the two factor code does not work any more",
        "I cannot sign in on my phone",
        "my username is not recognised by the login page",
        "please unlock my profile",
        "the login keeps sending me back to the start",
    ],
}
TAILS = [
    "",
    " Thanks.",
    " This is the second time I write.",
    " It is for my company account.",
    " Can you check today?",
    " Best regards.",
]


def tickets(n, seed, mixed=0.0):
    """n distinct tickets with their queue. With `mixed` above 0 that share of tickets also carries a sentence
    from another
    queue after the first one ("Also: ..."); the label is the queue of the first sentence, so a model that weighs both
    topics will sometimes be wrong, which is what a task with a real error rate looks like."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        queue = rng.choice(list(QUEUES))
        extra = ""
        if rng.random() < mixed:
            other = rng.choice([q for q in QUEUES if q != queue])
            extra = f" Also: {rng.choice(BODIES[other])}."
        text = (
            f"{rng.choice(OPENERS)} {rng.choice(BODIES[queue])}.{extra}{rng.choice(TAILS)} "
            f"(ref {rng.randrange(10**5):05d})"
        )
        if text not in seen:
            seen.add(text)
            out.append((text, queue))
    return out


def question(text):
    return {
        "state": {"ticket": text},
        "model": "decisio",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which support queue should handle this ticket?",
                "criteria": dict(QUEUES),
            }
        },
    }


def call(url, path, body=None, method=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method or ("POST" if body is not None else "GET"))
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def start(a, out, tag, port):
    cmd = [sys.executable, "-m", "decisio.serve.vllm_engine", "--backend", a.backend, "--port", str(port)]
    if a.model:
        cmd += ["--model", a.model]
    if a.base:
        cmd += ["--base", a.base]
    log = open(out / f"server_{tag}.log", "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    )
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 1500:
        if proc.poll() is not None:
            raise SystemExit(f"the server ended while starting (code {proc.returncode}); see {out}/server_{tag}.log")
        try:
            if call(url, "/health", timeout=5)[0] == 200:
                return proc, url, round(time.time() - t0, 1)
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise SystemExit("the server did not become healthy in 25 minutes")


def stop(proc):
    """End the server we started, by its own handle: ask, wait, then insist (an MLX server can take a while to
    let go)."""
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def answers_of(url, asked):
    out = []
    for text, gold in asked:
        status, body = call(url, "/v1/systemone", question(text))
        assert status == 200, body
        a = body["answers"]["q"]
        out.append(
            {
                "gold": gold,
                "choice": a["choice"],
                "top": max(a["probabilities"].values()),
                "certified": a["certified"],
                "reason": a.get("certified_reason"),
                "certificate": a["certificate"],
            }
        )
    return out


def certificate_facts(task):
    c = task["certificate"]
    keys = (
        "id",
        "alpha",
        "delta",
        "n",
        "threshold",
        "certified_share",
        "certified_n",
        "errors_at_threshold",
        "stack_sha256",
    )
    return {k: c.get(k) for k in keys}


def commit_of_this_checkout():
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "src"], capture_output=True, text=True
            ).stdout.strip()
        )
        return {"decisio_commit": sha, "src_modified_since": dirty}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model")
    ap.add_argument("--base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--examples", type=int, default=400)
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--mixed", type=float, default=0.0, help="the share of tickets that also mention another queue")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = tickets(a.examples + a.questions, seed=20261014, mixed=a.mixed)
    reg, asked = pool[: a.examples], pool[a.examples :]
    body = {
        "id": "ticket-queue",
        "examples": [{"request": question(t), "answer": g} for t, g in reg],
        "certify": {"alpha": a.alpha, "delta": a.delta},
    }
    runs = {"A": [], "B": []}
    exported = None
    for i in range(1, a.restarts + 1):  # (A) register with certification again after every restart
        proc, url, up = start(a, out, f"A{i}", free_port())
        try:
            t0 = time.time()
            status, task = call(url, "/v1/tasks", body)
            assert status == 200, task
            took = round(time.time() - t0, 1)
            answers = answers_of(url, asked)
            if exported is None:
                exported = call(url, "/v1/tasks?full=1")[1]["tasks"]
                (out / "first_start_task_response.json").write_text(json.dumps(task, indent=1))
                samples = []
                for want in (True, False):  # one certified and one uncertified answer as the server returned them
                    k = next((n for n, x in enumerate(answers) if x["certified"] is want), None)
                    if k is not None:
                        samples.append(
                            {
                                "ticket": asked[k][0],
                                "answer": call(url, "/v1/systemone", question(asked[k][0]))[1]["answers"]["q"],
                            }
                        )
                (out / "first_start_sample_answers.json").write_text(json.dumps(samples, indent=1))
            runs["A"].append(
                {
                    "restart": i,
                    "start_seconds": up,
                    "register_seconds": took,
                    "certificate": certificate_facts(task),
                    "status": task["status"]["state"],
                    "answers": answers,
                }
            )
        finally:
            stop(proc)
    for i in range(1, a.restarts + 1):  # (B) a fresh process takes the first start's export
        proc, url, up = start(a, out, f"B{i}", free_port())
        try:
            status, loaded = call(url, "/v1/tasks/import", {"tasks": exported})
            assert status == 200, loaded
            task = call(url, "/v1/tasks/ticket-queue")[1]
            answers = answers_of(url, asked)
            runs["B"].append(
                {
                    "restart": i,
                    "start_seconds": up,
                    "certificate": certificate_facts(task),
                    "status": task["status"]["state"],
                    "answers": answers,
                }
            )
        finally:
            stop(proc)
    ref = runs["A"][0]

    def diff(x, y):
        return {
            "certified_flags_that_differ": sum(
                p["certified"] != q["certified"] for p, q in zip(x["answers"], y["answers"])
            ),
            "choices_that_differ": sum(p["choice"] != q["choice"] for p, q in zip(x["answers"], y["answers"])),
            "largest_top_probability_difference": max(
                abs(p["top"] - q["top"]) for p, q in zip(x["answers"], y["answers"])
            ),
            "threshold_equal": x["certificate"]["threshold"] == y["certificate"]["threshold"],
            "certificate_id_equal": x["certificate"]["id"] == y["certificate"]["id"],
            "stack_digest_equal": x["certificate"]["stack_sha256"] == y["certificate"]["stack_sha256"],
        }

    result = {
        "backend": a.backend,
        "model": a.model,
        "base": a.base,
        "alpha": a.alpha,
        "delta": a.delta,
        "examples": a.examples,
        "questions": a.questions,
        "mixed": a.mixed,
        **commit_of_this_checkout(),
        "certificate_of_the_first_start": ref["certificate"],
        "certified_share_of_the_questions": sum(x["certified"] for x in ref["answers"]) / len(ref["answers"]),
        "errors_among_certified_questions": sum(x["certified"] and x["choice"] != x["gold"] for x in ref["answers"]),
        "accuracy_of_the_questions": sum(x["choice"] == x["gold"] for x in ref["answers"]) / len(ref["answers"]),
        "A_register_again_vs_the_first_start": [diff(ref, r) for r in runs["A"][1:]],
        "B_import_the_first_export_vs_the_first_start": [diff(ref, r) for r in runs["B"]],
        "states_after_import": [r["status"] for r in runs["B"]],
        "seconds": {
            "A": [(r["start_seconds"], r["register_seconds"]) for r in runs["A"]],
            "B": [r["start_seconds"] for r in runs["B"]],
        },
        "answers_sha256": hashlib.sha256(json.dumps(ref["answers"], sort_keys=True).encode()).hexdigest(),
    }
    (out / "restart_result.json").write_text(json.dumps(result, indent=1))
    (out / "first_start_answers.json").write_text(json.dumps(ref["answers"], indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
