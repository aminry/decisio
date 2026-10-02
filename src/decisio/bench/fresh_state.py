# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-request latency on fresh states, stage by stage, through a running decisio server.

For each state length (tokens) and question count, every request carries a state the server has never seen (a new
support ticket each repeat), so nothing comes from the prefix cache. Per request it records the client's wall time,
the server's total (x-decisio-server-ms) and the engine's stages (x-decisio-stages: prepare, warm, questions,
readout, and the engine's total as `engine`). The time splits into HTTP and JSON (client minus server), the route's
own work (server minus engine), tokenising and padding (prepare), the warm-up prefill, the questions' engine call
and the readout.

    python -m decisio.bench.fresh_state --url http://127.0.0.1:8000 --tokenizer $DECISIO_MODEL --out fresh.json \\
        [--tokens 300,1000,3000] [--questions 1,4] [--repeats 20]

--engine-floor runs no server: it starts vLLM in this process with the served engine's settings and times single
one-token generate calls on fresh, unpadded prompts of each length (no chat template, no letters prompt), which is
what the model's forward costs on this card with nothing of ours around it. Stop the server first (one engine per card).

    python -m decisio.bench.fresh_state --engine-floor --model $DECISIO_MODEL --out floor.json [--tokens 300,1056,3000]
"""

import argparse
import json
import statistics
import time
import urllib.request

QUESTIONS = [
    {"type": "noul", "instructions": "Does this ticket need a response within the hour?"},
    {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Invoices, payments",
            "access": "Login, passwords",
            "bug": "Wrong behaviour",
            "other": None,
        },
    },
    {"type": "noul", "instructions": "Is the customer at risk of leaving?"},
    {
        "type": "score",
        "instructions": "Rate the business impact.",
        "criteria": ["None", "One user, with a workaround", "Many users blocked", "Data loss or legal exposure"],
    },
]
TOPICS = ["login", "an invoice", "an API outage", "a refund", "data export", "a slow dashboard", "a webhook"]


def state_text(tok, n_tokens, seed):
    """A fresh support ticket of exactly n_tokens tokens; the seed makes it differ from every other request's."""
    para = (
        f"Ticket {100000 + seed}. Customer writes about {TOPICS[seed % len(TOPICS)]}: since {seed % 12 + 1} o'clock "
        f"none of our {seed % 90 + 10} staff can work normally, and payroll is due today. "
    )
    note = "Agent note {k}: checked logs for request {r}, saw error code {c}, asked for a screenshot. "
    text, k = para, 0
    while len(tok.encode(text, add_special_tokens=False)) < n_tokens:
        text += note.format(k=k, r=seed * 97 + k, c=400 + (seed + k) % 200)
        k += 1
    return tok.decode(tok.encode(text, add_special_tokens=False)[:n_tokens])


def ask(url, state, n_questions):
    body = {"state": state, "questions": {f"q{i}": QUESTIONS[i % len(QUESTIONS)] for i in range(n_questions)}}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
        headers = {k.lower(): v for k, v in r.headers.items()}
    wall = (time.perf_counter() - t) * 1000
    stages = dict(kv.split("=") for kv in headers.get("x-decisio-stages", "").split(";") if "=" in kv)
    return {
        "wall_ms": wall,
        "server_ms": float(headers["x-decisio-server-ms"]),
        **{k: float(v) for k, v in stages.items()},
    }


def summary(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: round(statistics.median(r[k] for r in rows), 2) for k in keys}


def engine_floor(a):
    """Median wall time of llm.generate for one fresh prompt of n tokens, one output token, logprobs over two ids."""
    import os
    import random

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from decisio.serve.vllm_engine import SERVED_ENGINE, deep_gemm_guard, engine_kwargs

    deep_gemm_guard("vllm", os.environ)
    kw = engine_kwargs(argparse.Namespace(engine=json.dumps(SERVED_ENGINE), model_class=a.model_class))
    # as LettersEngine builds it: its defaults, then the served engine's keyword arguments on top
    llm = LLM(
        **{
            "model": a.model,
            "max_model_len": 32768,
            "enable_prefix_caching": True,
            "max_logprobs": 256,
            "logprobs_mode": "processed_logprobs",
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            **kw,
        }
    )
    rng = random.Random(0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[9454, 2152], detokenize=False)
    cells = []
    for n in (int(x) for x in a.tokens.split(",")):
        times = []
        for i in range(a.warmups + a.repeats):
            ids = [rng.randrange(1000, 150000) for _ in range(n)]  # never seen before: nothing from the cache
            t = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            if i >= a.warmups:
                times.append((time.perf_counter() - t) * 1000)
        cells.append({"prompt_tokens": n, "median_ms": round(statistics.median(times), 2), "ms": times})
        print(f"{n:5d} tokens: engine floor {cells[-1]['median_ms']:.1f} ms", flush=True)
    return {"label": a.label, "engine_floor": True, "model": a.model, "model_class": a.model_class, "cells": cells}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokenizer", default=None, help="the served model's tokenizer (a path or hub id)")
    ap.add_argument("--tokens", default="300,1000,3000")
    ap.add_argument("--questions", default="1,4")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-floor", action="store_true", help="time vLLM's forward alone, in this process")
    ap.add_argument("--model", default=None, help="--engine-floor: the checkpoint")
    ap.add_argument("--model-class", default="hidden-readout", help="--engine-floor: as the server's --model-class")
    a = ap.parse_args()
    if a.engine_floor:
        out = engine_floor(a)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cells, seed = [], 0
    for n_tok in (int(x) for x in a.tokens.split(",")):
        for n_q in (int(x) for x in a.questions.split(",")):
            for _ in range(a.warmups):  # the shapes' first requests, not recorded
                seed += 1
                ask(a.url, state_text(tok, n_tok, seed), n_q)
            rows = []
            for _ in range(a.repeats):
                seed += 1
                rows.append(ask(a.url, state_text(tok, n_tok, seed), n_q))
            cell = {
                "state_tokens": n_tok,
                "questions": n_q,
                "repeats": a.repeats,
                "median": summary(rows),
                "rows": rows,
            }
            cells.append(cell)
            m = cell["median"]
            print(
                f"{n_tok:5d} tokens {n_q} q: wall {m['wall_ms']:.1f}  server {m['server_ms']:.1f}  "
                + "  ".join(f"{k} {v:.1f}" for k, v in m.items() if k not in ("wall_ms", "server_ms")),
                flush=True,
            )
    with open(a.out, "w") as f:
        json.dump({"label": a.label, "url": a.url, "cells": cells}, f, indent=1)


if __name__ == "__main__":
    main()
