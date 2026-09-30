# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Conformance of `/v1/systemone` (decisio.serve.systemone) to TypeSafe's wire format: gates C2-C4, run before every
benchmark measurement in runs/.

  C2  TypeSafe's own client (typesafe-sdk, `TypeSafeClient`) parses our responses: 12 requests covering noul with and
      without criteria, choice with and without descriptions (2 to 151 options), score, string/object/array states and
      several questions per request; `models.list()` parses; an invalid request gets 422 in the SDK's
      `HTTPValidationError` shape. Every answer is also checked against the adapter's rules (keys, sums, confidence).
  C3  prompt identity, tokenizer only: for each item the engine's own request builder (`_prepare_separate`) gives the
      same token rows and label tokens for the `/v1/systemone` form as for the `/v1/answer` form.
  C4  paired probabilities over HTTP: each item's state is cached by a first `/v1/answer` call, then `/v1/answer` and
      `/v1/systemone` are asked in turn; their probabilities must be bit-identical (max abs delta p = 0).

    python -m decisio.serve.systemone_conformance --url http://127.0.0.1:8000 --items conformance_items.json \
        --tokenizer /models/Qwen3.6-35B-A3B-FP8 --block-size 1056 --out conformance.json

--items is a JSON list, one object per item: {"task", "i", "label", "answer": a `/v1/answer` body with one question,
"systemone": the same item as a `/v1/systemone` body with one question named "q"}. The records in runs/ keep each item's
task, index, label and both routes' probabilities, not its text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from decisio.serve.systemone import SystemOneRequest, choice_confidence, score_confidence, to_engine_question, top_index
from decisio.serve.temperature import apply_temperature

CLINC_LIKE = [f"intent_{i:03d}" for i in range(150)] + ["oos"]
LONG = " ".join(f"Line {i}: the customer reports that the invoice total differs from the order total by {i} euros."
                for i in range(120))


def c2_requests():
    from typesafe_sdk import Choice, Noul, Score
    return [
        ("noul, no criteria, string state", "The shipment arrived two days late.",
         {"late": Noul(instructions="Was the shipment late?")}),
        ("noul, both criteria, object state", {"ticket": "Refund not received", "priority": "high"},
         {"refund": Noul(instructions="Is this about a refund?",
                         criteria={"true": "The customer asks about money back.", "false": "Anything else."})}),
        ("noul, one criterion, JSON instructions", "Buy cheap watches now!!!",
         {"spam": Noul(instructions={"task": "Decide whether the message is unsolicited advertising."},
                       criteria={"true": "Advertising"})}),
        ("choice, two options, no descriptions", "I love this phone.",
         {"sentiment": Choice(criteria={"positive": None, "negative": None})}),
        ("choice, four options with descriptions", "I was charged twice for one order.",
         {"team": Choice(instructions="Which team should handle this?",
                         criteria={"billing": "Payments and invoices", "technical": "Bugs and outages",
                                   "sales": "New purchases", "account": "Login and profile"})}),
        ("choice, 151 options", "What is the weather tomorrow in Berlin?",
         {"intent": Choice(instructions="What is the user's intent?", criteria={k: None for k in CLINC_LIKE})}),
        ("choice, mixed and JSON descriptions", "Server returns 500 on every call.",
         {"severity": Choice(criteria={"low": None, "medium": "Degraded but usable",
                                       "high": {"examples": ["outage", "data loss"]}})}),
        ("score, three levels", "Please fix this today, production is down.",
         {"urgency": Score(instructions="How urgent is this message?", criteria=["Can wait", "This week", "Today"])}),
        ("score, five JSON levels", "A decent read, some slow chapters.",
         {"rating": Score(instructions="Rate the review.", criteria=[{"stars": i} for i in range(1, 6)])}),
        ("three questions in one request", "Order #123 arrived broken; I want my money back.",
         {"refund": Noul(instructions="Does the customer want a refund?"),
          "team": Choice(criteria={"billing": None, "shipping": None, "support": None}),
          "anger": Score(criteria=["calm", "annoyed", "angry"])}),
        ("array state", ["first message: hello", "second message: my card was declined"],
         {"card": Noul(instructions="Is the problem about a card?")}),
        ("long state, three questions", LONG,
         {"invoice": Noul(instructions="Is this about invoices?"),
          "count": Score(instructions="How many discrepancies are mentioned?",
                         criteria=["none", "one", "several", "many"]),
          "topic": Choice(criteria={"billing": None, "delivery": None, "product quality": None, "other": None})}),
    ]


def check_answer(name, q, a):
    """The adapter's rules, recomputed on what our server returned."""
    if q.type == "noul":
        assert a.type == "noul" and 0.0 <= a.noul <= 1.0, f"{name}: noul out of range"
        return
    keys = list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria))]
    # the SDK's typed score view keys probabilities and legend by int level; the wire keys are "0".."K-1"
    got = [str(k) for k in a.probabilities]
    assert got == keys, f"{name}: probability keys {got[:5]} != {keys[:5]}"
    p = np.array(list(a.probabilities.values()))
    assert abs(p.sum() - 1.0) < 1e-6 and (p >= 0).all(), f"{name}: not a distribution (sum {p.sum()})"
    if q.type == "choice":
        assert a.choice == keys[top_index(keys, p)], (f"{name}: choice is not the most probable option "
                                                      "(exact ties: the key that sorts first)")
        assert abs(a.confidence - choice_confidence(p)) < 1e-9, f"{name}: choice confidence rule"
    else:
        assert abs(a.score - float((np.arange(len(p)) * p).sum())) < 1e-9, f"{name}: score is not the expectation"
        assert abs(a.confidence - score_confidence(p)) < 1e-9, f"{name}: score confidence rule"
        assert [str(k) for k in a.legend] == keys, f"{name}: legend keys"


def gate_c2(url):
    import httpx
    from typesafe_sdk import SystemOneResponse, TypeSafeClient
    out, ok = [], True
    with TypeSafeClient(api_key="local-no-auth", base_url=url, timeout=600) as client:
        for label, state, questions in c2_requests():
            t = time.perf_counter()
            try:
                r = client.system_one(state=state, questions=questions, model="rlcd-served-default")
                assert isinstance(r, SystemOneResponse)
                assert set(r.answers) == set(questions), "answer names differ from question names"
                assert r.usage.input_tokens > 0 and r.usage.output_tokens == len(questions)
                wire = SystemOneRequest.model_validate({"state": state, "model": "m",
                                                        "questions": {k: v.model_dump(exclude_none=True)
                                                                      for k, v in questions.items()}})
                for name, q in wire.questions.items():
                    check_answer(name, q, r.answers[name].root if hasattr(r.answers[name], "root") else r.answers[name])
                out.append({"request": label, "ok": True, "ms": round((time.perf_counter() - t) * 1000, 1),
                            "answers": {k: v.model_dump() if hasattr(v, "model_dump") else v
                                        for k, v in r.answers.items()}})
            except Exception as e:                                  # noqa: BLE001 - every failure is reported
                ok = False
                out.append({"request": label, "ok": False, "error": f"{type(e).__name__}: {e}"})
        try:
            models = client.models.list()
            out.append({"request": "models.list", "ok": True, "models": [m.name for m in models.models]})
        except Exception as e:                                      # noqa: BLE001
            ok = False
            out.append({"request": "models.list", "ok": False, "error": f"{type(e).__name__}: {e}"})
    r = httpx.post(f"{url}/v1/systemone", json={"model": "m", "questions": {"q": {"type": "noul"}}}, timeout=60)
    body = r.json()
    shaped = (r.status_code == 422 and isinstance(body.get("detail"), list)
              and all({"loc", "msg", "type"} <= set(d) for d in body["detail"]))
    ok &= shaped
    out.append({"request": "invalid (no state) -> 422 HTTPValidationError", "ok": shaped, "status": r.status_code,
                "detail": body.get("detail", [])[:2]})
    return ok, out


def tokenizer_engine(tokenizer, block_size, pad_where="front"):
    """The engine's request builder without an engine: every attribute `_prepare_separate` reads, and nothing else."""
    import threading

    from transformers import AutoTokenizer

    from decisio.serve.vllm_engine import PAD_TOKEN, LettersEngine
    eng = LettersEngine.__new__(LettersEngine)
    eng.tok = AutoTokenizer.from_pretrained(tokenizer)
    eng.mode, eng.pad_token, eng.pad_where = "separate", PAD_TOKEN, pad_where
    eng.adapters = {}
    eng.block_size = eng.match_unit = block_size
    eng.pad_unit = block_size
    eng._lock = threading.Lock()
    return eng


def gate_c3(items, tokenizer, block_size):
    eng = tokenizer_engine(tokenizer, block_size)
    bad = []
    for it in items:
        a_rows, a_p = eng._prepare_separate(it["answer"]["state"], it["answer"]["questions"])
        req = SystemOneRequest.model_validate(it["systemone"])
        s_q = [to_engine_question(q)[0] for q in req.questions.values()]
        s_rows, s_p = eng._prepare_separate(req.state, s_q)
        if a_rows != s_rows or a_p != s_p:
            bad.append(f"{it['task']}:{it['i']}")
    return not bad, {"items": len(items), "differ": bad[:20], "n_differ": len(bad)}


def gate_c4(url, items):
    import httpx
    diffs, flips, rows = [], 0, []
    with httpx.Client(timeout=600) as c:
        # /v1/systemone serves its plain readout under the global temperature (decisio.serve.temperature); /v1/answer is
        # the raw readout, so the same function is applied to it before comparing
        T = float((c.get(f"{url}/health").json().get("systemone") or {}).get("temperature", 1.0))
        for it in items:
            c.post(f"{url}/v1/answer", json=it["answer"]).raise_for_status()               # caches the state
            a = c.post(f"{url}/v1/answer", json=it["answer"]).json()["answers"][0]
            s = c.post(f"{url}/v1/systemone", json=it["systemone"]).json()["answers"]["q"]
            # compare what each route transmits: P(yes) for yes/no (the wire carries no P(no)), every probability for
            # choice
            if s["type"] == "noul":
                pa, ps = apply_temperature(a["probs"], T)[:1], np.array([s["noul"]])
            else:
                pa, ps = apply_temperature(a["probs"], T), np.array([s["probabilities"][o] for o in a["options"]])
            d = float(np.abs(pa - ps).max())
            diffs.append(d)
            flips += int((pa > 0.5).any() != (ps > 0.5).any()) if len(pa) == 1 else int(pa.argmax() != ps.argmax())
            rows.append({"task": it["task"], "i": it["i"], "label": it["label"], "answer": pa.tolist(),
                         "systemone": ps.tolist(), "max_abs": d})
    m = max(diffs) if diffs else float("nan")
    return m == 0.0 and flips == 0, {"items": len(items), "temperature": T, "max_abs_delta_p": m,
                                     "mean_abs_delta_p": float(np.mean(diffs)),
                                     "top_answer_flips": flips, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.items))[: a.n]
    res, ok = {"url": a.url, "items": len(items)}, True
    for name, fn in (("C2", lambda: gate_c2(a.url)), ("C3", lambda: gate_c3(items, a.tokenizer, a.block_size)),
                     ("C4", lambda: gate_c4(a.url, items))):
        g_ok, detail = fn()
        res[name] = {"pass": g_ok, "detail": detail}
        ok &= g_ok
        summary = {k: v for k, v in detail.items() if k != "rows"} if isinstance(detail, dict) else \
            [f"{x['request']}: {'ok' if x['ok'] else 'FAIL ' + x.get('error', '')}" for x in detail]
        print(f"{'ok  ' if g_ok else 'FAIL'} {name} {json.dumps(summary)[:1500]}", flush=True)
    res["pass"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    print("CONFORMANCE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
