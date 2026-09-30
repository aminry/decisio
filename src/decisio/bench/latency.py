# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A latency and throughput grid through the serving engine (vllm_engine.LettersEngine), in either mode.

Two numbers per cell, because they answer different questions:

  latency     one request alone (state + N distinct questions), p50 over repeats, each repeat with a
              fresh state. What a caller waits; comparable to a hosted API's p50 server time.
  throughput  R such requests at once (answer_many: all states prefilled together, then all
              questions), R chosen so a cell holds at least --load questions. What a loaded server
              sustains, and so what a token costs: dollars per billion input tokens =
              price per hour / (input tokens per second x 3600) x 1e9.

Every request's state is fresh: seeded by repeat, question count, state length and kind, so no cell reads
another cell's cached states.

Input tokens are counted as a per-token API bills them: the state once per request, plus each question's
own text (instruction and option listing), in this model's tokenizer. Recomputed or padding tokens are
our cost, not the customer's, so they are not counted as billable.

The 77-option cells use BANKING77's intent names (banking77_labels.json; PolyAI, CC-BY-4.0) as options.

    python -m decisio.bench.latency --model $DECISIO_VIEW --out bench.json --price 1.05 \
        [--pad-to block]
    python -m decisio.bench.latency --model ... --mode packed --pack 64 --out bench_packed.json --price 1.05
    python -m decisio.bench.latency ... --url http://localhost:8000     # latency through the HTTP endpoint instead
"""

import argparse
import json
import math
import time
from pathlib import Path

LABELS = Path(__file__).with_name("banking77_labels.json")
TOPICS = [
    "billing",
    "a refund",
    "a login problem",
    "an API outage",
    "shipping",
    "a password reset",
    "data export",
    "an invoice",
    "account closure",
    "a duplicate charge",
    "latency",
    "a security concern",
    "onboarding",
]
ASPECTS = [
    "the first sentence",
    "the last paragraph",
    "the mention of orders",
    "the timing",
    "the tone",
    "the request for escalation",
    "the mention of errors",
    "the account details",
]
COUNTS = {
    "noul": [1, 10, 100, 1000],
    "choice4": [1, 10, 100, 1000],
    "choice77": [50, 100, 1000],
    "score": [1, 10, 100, 1000],
}
KIND_SEED = {"noul": 0, "choice4": 10**7, "choice77": 2 * 10**7, "score": 3 * 10**7}


def state_ids(tok, n_tokens, rep):
    """A support-ticket state of exactly n_tokens tokens, distinct per `rep`."""
    para = (
        f"Ticket {8800 + rep}. Customer writes: our API integration started returning 500 errors on every "
        "request about 20 minutes ago, and we cannot process customer orders until this is fixed. We are on "
        "the enterprise plan and this is the second outage this month. Please escalate to an engineer. "
    )
    extra = (
        "Support agent notes: checked the status page, no incident posted; asked the customer for request "
        "ids; the customer's account shows {k} failed webhooks since 09:{m:02d} and a plan renewal due next "
        "week. "
    )
    text, k = para, 0
    while len(tok.encode(text, add_special_tokens=False)) < n_tokens:
        text += extra.format(k=k * 7 + rep, m=k % 60)
        k += 1
    return tok.encode(text, add_special_tokens=False)[:n_tokens]


LETTER_KINDS = ("noul", "choice4", "choice77", "score")


def questions(kind, n, labels, seed):
    out = []
    for i in range(n):
        j = i + seed  # distinct within a request, and across repeats
        if kind == "noul":
            out.append(
                {"kind": "noul", "instructions": f"Question {j}: Is this message about {TOPICS[j % len(TOPICS)]}?"}
            )
        elif kind == "choice4":
            out.append(
                {
                    "kind": "choice",
                    "options": ["billing", "technical", "sales", "account"],
                    "instructions": f"Question {j}: Considering {ASPECTS[j % len(ASPECTS)]}, "
                    "which team should handle this?",
                }
            )
        elif kind == "choice77":
            out.append(
                {
                    "kind": "choice",
                    "options": labels,
                    "instructions": f"Question {j}: Considering {ASPECTS[j % len(ASPECTS)]}, "
                    "what is the customer's intent?",
                }
            )
        else:
            out.append(
                {
                    "kind": "score",
                    "options": ["not urgent", "somewhat", "urgent", "critical"],
                    "instructions": f"Question {j}: Considering {ASPECTS[j % len(ASPECTS)]}, how urgent is this?",
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="separate", choices=["separate", "packed"])
    ap.add_argument("--pad-to", default=None)
    ap.add_argument("--pad-where", default="between")
    ap.add_argument("--pack", type=int, default=64)
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--states", default="500,2000,8000")
    ap.add_argument("--kinds", default=",".join(LETTER_KINDS))
    ap.add_argument("--counts", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--load", type=int, default=2000, help="questions per throughput cell (at least)")
    ap.add_argument("--max-concurrent", type=int, default=64)
    ap.add_argument("--price", type=float, required=True, help="GPU dollars per hour actually paid")
    ap.add_argument("--hardware", default=None, help="label for the table, default the GPU name")
    ap.add_argument("--url", default=None, help="measure latency through a running decisio server's /v1/answer")
    ap.add_argument(
        "--options-in-state",
        action="store_true",
        help="choice kinds: the option listing goes in the (cached) state once, each question is its "
        "instruction alone (shared option listing); billed as if options were sent per question",
    )
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from decisio.serve import vllm_engine as sv

    labels = json.load(open(args.labels))["banking77"]
    eng = sv.LettersEngine(
        args.model,
        mode=args.mode,
        pad_to=args.pad_to,
        pad_where=args.pad_where,
        pack=args.pack,
        engine_kw=json.loads(args.engine),
    )
    facts = eng.facts()
    print("ENGINE", json.dumps(facts), flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    client = None
    if args.url:
        from decisio.serve.client import Client

        client = Client(args.url)

    def billable(state, qs):
        # the per-token API basis: the state once, every question with its own option listing, whatever we compute
        plain = [{k: v for k, v in q.items() if k != "options_in_state"} for q in qs]
        return ntok(state) + sum(ntok(sv.question_text(tok, q)[0]) for q in plain)

    def state_text(n, seed):
        return tok.decode(state_ids(tok, n, seed))

    def shared(kind, state, qs):
        """--options-in-state: the listing moves into the state, once; questions keep their options field."""
        if not args.options_in_state or kind not in ("choice4", "choice77", "score"):
            return state, qs
        listing = sv.options_listing(tok, qs[0]["options"])
        assert all(q["options"] == qs[0]["options"] for q in qs)
        return f"{state}\n\n{listing}", [dict(q, options_in_state=True) for q in qs]

    # warm-up: compile kernels and capture graphs for every state length
    for n in (int(x) for x in args.states.split(",")):
        eng.answer(state_text(n, 7), questions("noul", 4, labels, 10**6))
    cells, results = [], []
    for k in args.kinds.split(","):
        for n in [int(x) for x in args.counts.split(",")] if args.counts else COUNTS[k]:
            cells.append((k, n))
    for n_tok in (int(x) for x in args.states.split(",")):
        for kind, n_q in cells:
            lat, server = [], []
            for rep in range(args.repeats):
                seed = 1000 * rep + 17 * n_q + n_tok + KIND_SEED[kind]  # a state never another cell's
                state, qs = state_text(n_tok, seed), questions(kind, n_q, labels, seed)
                state, qs = shared(kind, state, qs)
                t = time.perf_counter()
                if client:
                    _, info = client.answer(state, qs)
                else:
                    _, info = eng.answer(state, qs)
                lat.append((time.perf_counter() - t) * 1000)
                server.append(info["server_ms"])
            R = max(1, min(args.max_concurrent, math.ceil(args.load / n_q)))
            raw = [
                (
                    state_text(n_tok, 10**5 + r * 31 + n_q + KIND_SEED[kind]),
                    questions(kind, n_q, labels, 10**5 + r * n_q),
                )
                for r in range(R)
            ]
            bill = sum(billable(s, q) for s, q in raw)
            reqs = [shared(kind, s, q) for s, q in raw]
            t = time.perf_counter()
            _, info = eng.answer_many(reqs)
            wall = time.perf_counter() - t
            tps = bill / wall
            med = sorted(lat)[len(lat) // 2]
            r = {
                "hardware": args.hardware or facts["gpu"],
                "engine": f"vLLM {facts['vllm']}",
                "mode": args.mode,
                "pack": args.pack if args.mode == "packed" else None,
                "pad_unit": facts["pad_unit"],
                "options_in_state": bool(args.options_in_state and kind != "noul"),
                "quantization": facts["quantization"],
                "state_tokens": n_tok,
                "kind": kind,
                "questions": n_q,
                "distinct_questions": True,
                "via": "http" if client else "engine",
                "latency_ms_p50": round(med, 1),
                "latency_ms_per_question": round(med / n_q, 3),
                "server_ms_p50": round(sorted(server)[len(server) // 2], 1),
                "throughput_requests": R,
                "throughput_questions_per_s": round(R * n_q / wall, 1),
                "throughput_ms_per_question": round(wall * 1000 / (R * n_q), 3),
                "billable_tokens_per_s": round(tps),
                "processed_tokens_per_s": round(info["prompt_tokens"] / wall),
                "price_per_hour": args.price,
                "dollars_per_billion_tokens": round(args.price / (tps * 3600) * 1e9, 2),
                "dollars_per_million_questions": round(args.price / 3600 * wall / (R * n_q) * 1e6, 2),
            }
            results.append(r)
            print(
                f"{n_tok:5d}t {kind:8s} {n_q:5d}q  latency {med:8.1f} ms ({r['latency_ms_per_question']:.3f}/q)  "
                f"load {r['throughput_ms_per_question']:.3f} ms/q  ${r['dollars_per_billion_tokens']}/B tok",
                flush=True,
            )
            json.dump({"engine": facts, "args": vars(args), "cells": results}, open(args.out, "w"), indent=1)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    from decisio.serve.vllm_engine import run_or_die

    run_or_die(main)
