# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Conformance of `/v1/systemone` (decisio.serve.systemone) to TypeSafe's wire format: gates C2-C4, run before every
benchmark measurement in runs/.

  C2  TypeSafe's own client (typesafe-sdk, `TypeSafeClient`) parses our responses: 12 requests covering noul with and
      without criteria, choice with and without descriptions (2 to 151 options), score, string/object/array states and
      several questions per request; `models.list()` parses; an invalid request gets 422 in the SDK's
      `HTTPValidationError` shape. Every answer is also checked against the adapter's rules (keys, sums, confidence).
  C3  prompt identity, tokenizer only: for each item the engine's own request builder (`_prepare_separate`) gives the
      same token rows and label tokens for the `/v1/systemone` form, rendered with the server's own flags (from
      /health), as for the `/v1/answer` form.
  C4  paired probabilities over HTTP: each item's state is cached by a first `/v1/answer` call, then `/v1/answer` and
      `/v1/systemone` are asked in turn; their probabilities must be bit-identical (max abs delta p = 0).
  C5  the wire against System One's published OpenAPI document (version 0.2.0, kept as a snapshot of structural facts in
      `benchmarks/system_one_wire_0.2.0.json`): request fields, required fields, question types and the limits of their
      criteria, and the response and answer fields. Offline. Every difference is named in DOCUMENTED_DIFFERENCES (and
      in `docs/api.md`); a difference that is not named fails, and so does a named one that has gone away.

    python -m decisio.serve.systemone_conformance --url http://127.0.0.1:8000 --items conformance_items.json \
        --tokenizer $DECISIO_MODEL --block-size 1056 --out conformance.json

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
LONG = " ".join(
    f"Line {i}: the customer reports that the invoice total differs from the order total by {i} euros."
    for i in range(120)
)


def c2_requests():
    from typesafe_sdk import Choice, Noul, Score

    return [
        (
            "noul, no criteria, string state",
            "The shipment arrived two days late.",
            {"late": Noul(instructions="Was the shipment late?")},
        ),
        (
            "noul, both criteria, object state",
            {"ticket": "Refund not received", "priority": "high"},
            {
                "refund": Noul(
                    instructions="Is this about a refund?",
                    criteria={"true": "The customer asks about money back.", "false": "Anything else."},
                )
            },
        ),
        (
            "noul, one criterion, JSON instructions",
            "Buy cheap watches now!!!",
            {
                "spam": Noul(
                    instructions={"task": "Decide whether the message is unsolicited advertising."},
                    criteria={"true": "Advertising"},
                )
            },
        ),
        (
            "choice, two options, no descriptions",
            "I love this phone.",
            {"sentiment": Choice(criteria={"positive": None, "negative": None})},
        ),
        (
            "choice, four options with descriptions",
            "I was charged twice for one order.",
            {
                "team": Choice(
                    instructions="Which team should handle this?",
                    criteria={
                        "billing": "Payments and invoices",
                        "technical": "Bugs and outages",
                        "sales": "New purchases",
                        "account": "Login and profile",
                    },
                )
            },
        ),
        (
            "choice, 151 options",
            "What is the weather tomorrow in Berlin?",
            {"intent": Choice(instructions="What is the user's intent?", criteria={k: None for k in CLINC_LIKE})},
        ),
        (
            "choice, mixed and JSON descriptions",
            "Server returns 500 on every call.",
            {
                "severity": Choice(
                    criteria={
                        "low": None,
                        "medium": "Degraded but usable",
                        "high": {"examples": ["outage", "data loss"]},
                    }
                )
            },
        ),
        (
            "score, three levels",
            "Please fix this today, production is down.",
            {"urgency": Score(instructions="How urgent is this message?", criteria=["Can wait", "This week", "Today"])},
        ),
        (
            "score, five JSON levels",
            "A decent read, some slow chapters.",
            {"rating": Score(instructions="Rate the review.", criteria=[{"stars": i} for i in range(1, 6)])},
        ),
        (
            "three questions in one request",
            "Order #123 arrived broken; I want my money back.",
            {
                "refund": Noul(instructions="Does the customer want a refund?"),
                "team": Choice(criteria={"billing": None, "shipping": None, "support": None}),
                "anger": Score(criteria=["calm", "annoyed", "angry"]),
            },
        ),
        (
            "array state",
            ["first message: hello", "second message: my card was declined"],
            {"card": Noul(instructions="Is the problem about a card?")},
        ),
        (
            "long state, three questions",
            LONG,
            {
                "invoice": Noul(instructions="Is this about invoices?"),
                "count": Score(
                    instructions="How many discrepancies are mentioned?", criteria=["none", "one", "several", "many"]
                ),
                "topic": Choice(criteria={"billing": None, "delivery": None, "product quality": None, "other": None}),
            },
        ),
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
        assert a.choice == keys[top_index(keys, p)], (
            f"{name}: choice is not the most probable option (exact ties: the key that sorts first)"
        )
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
                r = client.system_one(state=state, questions=questions, model="decisio-served-default")
                assert isinstance(r, SystemOneResponse)
                assert set(r.answers) == set(questions), "answer names differ from question names"
                assert r.usage.input_tokens > 0 and r.usage.output_tokens == len(questions)
                wire = SystemOneRequest.model_validate(
                    {
                        "state": state,
                        "model": "m",
                        "questions": {k: v.model_dump(exclude_none=True) for k, v in questions.items()},
                    }
                )
                for name, q in wire.questions.items():
                    check_answer(name, q, r.answers[name].root if hasattr(r.answers[name], "root") else r.answers[name])
                out.append(
                    {
                        "request": label,
                        "ok": True,
                        "ms": round((time.perf_counter() - t) * 1000, 1),
                        "answers": {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in r.answers.items()},
                    }
                )
            except Exception as e:  # noqa: BLE001 - every failure is reported
                ok = False
                out.append({"request": label, "ok": False, "error": f"{type(e).__name__}: {e}"})
        try:
            models = client.models.list()
            out.append({"request": "models.list", "ok": True, "models": [m.name for m in models.models]})
        except Exception as e:  # noqa: BLE001
            ok = False
            out.append({"request": "models.list", "ok": False, "error": f"{type(e).__name__}: {e}"})
    r = httpx.post(f"{url}/v1/systemone", json={"model": "m", "questions": {"q": {"type": "noul"}}}, timeout=60)
    body = r.json()
    shaped = (
        r.status_code == 422
        and isinstance(body.get("detail"), list)
        and all({"loc", "msg", "type"} <= set(d) for d in body["detail"])
    )
    ok &= shaped
    out.append(
        {
            "request": "invalid (no state) -> 422 HTTPValidationError",
            "ok": shaped,
            "status": r.status_code,
            "detail": body.get("detail", [])[:2],
        }
    )
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
    eng.block_size = eng.match_unit = block_size or 16
    eng.pad_unit = block_size or None  # --block-size 0: a base served without padding (Gemma 4)
    eng._lock = threading.Lock()
    return eng


def server_rendering(url):
    """The rendering flags the server reports in /health (to_engine_question's keyword arguments)."""
    import httpx

    s = httpx.get(f"{url}/health", timeout=60).json().get("systemone") or {}
    return {k: s[k] for k in ("hide_index_keys", "desnake_labels", "describe_options") if k in s}


def server_format(url):
    """(PromptFormat, noul rendering) the server reports in /health; the defaults when it reports none."""
    import httpx

    from decisio.readout.letters import PromptFormat

    h = httpx.get(f"{url}/health", timeout=60).json()
    return PromptFormat(**(h.get("prompt_format") or {})), (h.get("systemone") or {}).get("noul_rendering", "words")


def served_temperature(health: dict, qtype: str) -> float:
    """The temperature `/v1/systemone` serves a question type at, from /health's systemone block (as
    SystemOne.temperature_of: the type's own temperature, else the global one)."""
    return float((health.get("temperatures") or {}).get(qtype, health.get("temperature", 1.0)))


def answer_counterpart(it, noul="words"):
    """The `/v1/answer` body that asks exactly what `/v1/systemone` asks for this item, and the index of P(yes) in its
    answer for a yes/no item (None for choice and score). Under --noul-rendering letters or letters-keys a yes/no
    question is a two-option choice, false first (systemone.noul_as_letters), so its counterpart is that choice and
    P(yes) is its second probability; otherwise the item's own `answer` body, P(yes) first."""
    from decisio.serve.systemone import noul_as_letters

    body = it["answer"]
    q = next(iter(SystemOneRequest.model_validate(it["systemone"]).questions.values()))
    if q.type != "noul":
        return body, None
    if noul != "words":
        got = noul_as_letters(q, keys_shown=noul == "letters-keys")
        if got is not None:
            eq = got[0]
            return {**body, "questions": [{k: eq[k] for k in ("kind", "instructions", "options")}]}, 1
    return body, 0


def gate_c3(items, tokenizer, block_size, rendering=None, fmt=None, noul="words"):
    from decisio.serve.systemone import noul_as_letters

    eng = tokenizer_engine(tokenizer, block_size)
    if fmt is not None:
        eng.fmt = fmt
    rendering = rendering or {}
    bad = []
    for it in items:
        body, _ = answer_counterpart(it, noul)
        a_rows, a_p = eng._prepare_separate(body["state"], body["questions"])
        req = SystemOneRequest.model_validate(it["systemone"])
        s_q = [
            (noul_as_letters(q, keys_shown=noul == "letters-keys") if q.type == "noul" and noul != "words" else None)
            or to_engine_question(q, **rendering)
            for q in req.questions.values()
        ]
        s_q = [x[0] for x in s_q]
        s_rows, s_p = eng._prepare_separate(req.state, s_q)
        if a_rows != s_rows or a_p != s_p:
            bad.append(f"{it['task']}:{it['i']}")
    return not bad, {"items": len(items), "differ": bad[:20], "n_differ": len(bad)}


def gate_c4(url, items):
    import httpx

    diffs, flips, rows = [], 0, []
    with httpx.Client(timeout=600) as c:
        # /v1/systemone serves its plain readout under each question type's temperature (decisio.serve.temperature:
        # the global one, or the type's own, as choice questions on the Qwen base); /v1/answer is the raw readout, so
        # the same function is applied to it before comparing
        health = c.get(f"{url}/health").json().get("systemone") or {}
        noul = health.get("noul_rendering", "words")
        used = {}
        for it in items:
            body, yes_at = answer_counterpart(it, noul)
            c.post(f"{url}/v1/answer", json=body).raise_for_status()  # caches the state
            a = c.post(f"{url}/v1/answer", json=body).json()["answers"][0]
            s = c.post(f"{url}/v1/systemone", json=it["systemone"]).json()["answers"]["q"]
            T = used.setdefault(s["type"], served_temperature(health, s["type"]))
            # compare what each route transmits: P(yes) for yes/no (the wire carries no P(no)), every probability for
            # choice
            if s["type"] == "noul":
                pa, ps = apply_temperature(a["probs"], T)[yes_at : yes_at + 1], np.array([s["noul"]])
            else:
                pa, ps = apply_temperature(a["probs"], T), np.array([s["probabilities"][o] for o in a["options"]])
            d = float(np.abs(pa - ps).max())
            diffs.append(d)
            flips += int((pa > 0.5).any() != (ps > 0.5).any()) if len(pa) == 1 else int(pa.argmax() != ps.argmax())
            rows.append(
                {
                    "task": it["task"],
                    "i": it["i"],
                    "label": it["label"],
                    "answer": pa.tolist(),
                    "systemone": ps.tolist(),
                    "max_abs": d,
                }
            )
    m = max(diffs) if diffs else float("nan")
    return m == 0.0 and flips == 0, {
        "items": len(items),
        "temperatures": used,
        "max_abs_delta_p": m,
        "mean_abs_delta_p": float(np.mean(diffs)),
        "top_answer_flips": flips,
        "rows": rows,
    }


# ---- C5: the wire against System One's published OpenAPI document ---------------------------------------------------

LIMIT_KEYS = ("minProperties", "maxProperties", "minItems", "maxItems")

# What differs from System One's wire on purpose, in the words `c5_differences` uses; each is in docs/api.md.
DOCUMENTED_DIFFERENCES = (
    "request field `assets`: System One has it, decisio does not read it",
    "request field `model`: required by System One, optional here",
    "question type `box`: not supported",
    "score criteria maxItems: System One 10, decisio none",
)


def _ref(doc_schemas: dict, ref: str) -> dict:
    return doc_schemas[ref.rsplit("/", 1)[1]]


def _facts_of(schema: dict) -> dict:
    return {"properties": sorted(schema.get("properties", {})), "required": sorted(schema.get("required", []))}


def _criteria_limits(schema: dict) -> dict:
    crit = schema.get("properties", {}).get("criteria", {})
    return {k: crit[k] for k in LIMIT_KEYS if k in crit}


def wire_facts(openapi: dict, source: str = "https://api.typesafe.ai/openapi.json", fetched: str = "") -> dict:
    """The structural facts of System One's request and response from its OpenAPI document (no descriptions)."""
    sc = openapi["components"]["schemas"]
    qmap = sc["Question"]["discriminator"]["mapping"]
    amap = sc["Answer"]["discriminator"]["mapping"]
    return {
        "source": source,
        "version": openapi["info"]["version"],
        "fetched": fetched,
        "request": {
            **_facts_of(sc["SystemOneRequest"]),
            "questions": {
                k: sc["SystemOneRequest"]["properties"]["questions"][k]
                for k in LIMIT_KEYS
                if k in sc["SystemOneRequest"]["properties"]["questions"]
            },
        },
        "question_types": {
            t: {**_facts_of(_ref(sc, r)), "criteria": _criteria_limits(_ref(sc, r))} for t, r in qmap.items()
        },
        "noul_criteria": _facts_of(sc["NoulCriteria"]),
        "response": _facts_of(sc["SystemOneResponse"]),
        "usage": _facts_of(sc["Usage"]),
        "answers": {t: _facts_of(_ref(sc, r)) for t, r in amap.items()},
    }


def decisio_wire_facts() -> dict:
    """The same facts from decisio's own request models."""
    sch = SystemOneRequest.model_json_schema()
    defs = sch["$defs"]
    qmap = sch["properties"]["questions"]["additionalProperties"]["discriminator"]["mapping"]
    return {
        "request": {
            **_facts_of(sch),
            "questions": {
                k: sch["properties"]["questions"][k] for k in LIMIT_KEYS if k in sch["properties"]["questions"]
            },
        },
        "question_types": {
            t: {**_facts_of(_ref(defs, r)), "criteria": _criteria_limits(_ref(defs, r))} for t, r in qmap.items()
        },
        "noul_criteria": _facts_of(defs["NoulCriteria"]),
    }


def c5_differences(snapshot: dict) -> list[str]:
    """Where decisio's request models differ from System One's, as sentences."""
    ours, theirs = decisio_wire_facts(), snapshot
    out = []
    for f in sorted(set(theirs["request"]["properties"]) - set(ours["request"]["properties"])):
        out.append(f"request field `{f}`: System One has it, decisio does not read it")
    for f in sorted(set(ours["request"]["properties"]) - set(theirs["request"]["properties"])):
        out.append(f"request field `{f}`: decisio has it, System One does not")
    for f in sorted(set(theirs["request"]["required"]) - set(ours["request"]["required"])):
        out.append(f"request field `{f}`: required by System One, optional here")
    for f in sorted(set(ours["request"]["required"]) - set(theirs["request"]["required"])):
        out.append(f"request field `{f}`: required here, not by System One")
    if ours["request"]["questions"] != theirs["request"]["questions"]:
        out.append(
            f"questions limits: System One {theirs['request']['questions']}, decisio {ours['request']['questions']}"
        )
    for t in sorted(set(theirs["question_types"]) - set(ours["question_types"])):
        out.append(f"question type `{t}`: not supported")
    for t in sorted(set(ours["question_types"]) - set(theirs["question_types"])):
        out.append(f"question type `{t}`: decisio has it, System One does not")
    for t in sorted(set(theirs["question_types"]) & set(ours["question_types"])):
        a, b = theirs["question_types"][t], ours["question_types"][t]
        for key in ("properties", "required"):
            if a[key] != b[key]:
                out.append(f"{t} question {key}: System One {a[key]}, decisio {b[key]}")
        for k in LIMIT_KEYS:
            if a["criteria"].get(k) != b["criteria"].get(k):
                mine, theirs_k = b["criteria"].get(k, "none"), a["criteria"].get(k, "none")
                out.append(f"{t} criteria {k}: System One {theirs_k}, decisio {mine}")
    if ours["noul_criteria"] != theirs["noul_criteria"]:
        out.append(f"noul criteria fields: System One {theirs['noul_criteria']}, decisio {ours['noul_criteria']}")
    return out


def response_differences(snapshot: dict, response: dict) -> list[str]:
    """What a `/v1/systemone` response lacks against System One's response, usage and answer fields (extra fields are
    decisio's additions and are not differences). The response holds one answer per type it wants checked."""
    out = []
    for f in snapshot["response"]["required"]:
        if f not in response:
            out.append(f"response field `{f}` missing")
    for f in snapshot["usage"]["required"]:
        if f not in response.get("usage", {}):
            out.append(f"usage field `{f}` missing")
    for name, ans in response.get("answers", {}).items():
        want = snapshot["answers"].get(ans.get("type"))
        if want is None:
            out.append(f"answer `{name}`: type {ans.get('type')!r} is not a System One answer type")
            continue
        for f in want["required"]:
            if f not in ans:
                out.append(f"{ans['type']} answer `{name}`: field `{f}` missing")
    return out


def gate_c5(snapshot: dict, responses: list[dict] | None = None):
    """(pass, detail): every request difference is documented, every documented one still exists, and the responses
    given carry System One's fields."""
    diffs = c5_differences(snapshot)
    undocumented = [d for d in diffs if d not in DOCUMENTED_DIFFERENCES]
    stale = [d for d in DOCUMENTED_DIFFERENCES if d not in diffs]
    resp = [d for r in (responses or []) for d in response_differences(snapshot, r)]
    return not (undocumented or stale or resp), {
        "system_one_version": snapshot["version"],
        "differences": diffs,
        "undocumented": undocumented,
        "documented_but_gone": stale,
        "response_missing": resp,
    }


def c5_main(argv):
    ap = argparse.ArgumentParser(prog="systemone_conformance c5")
    ap.add_argument("--snapshot", required=True, help="benchmarks/system_one_wire_0.2.0.json")
    ap.add_argument("--url", help="a running server: one noul, one choice and one score question are asked and checked")
    ap.add_argument("--refresh", help="a downloaded System One openapi.json: print the snapshot facts for it and exit")
    a = ap.parse_args(argv)
    if a.refresh:
        print(json.dumps(wire_facts(json.load(open(a.refresh))), indent=1))
        return 0
    responses = []
    if a.url:
        import urllib.request

        body = {
            "state": "The shipment arrived two days late.",
            "questions": {
                "n": {"type": "noul", "instructions": "Was it late?"},
                "c": {"type": "choice", "criteria": {"late": None, "on_time": None}},
                "s": {"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad", "terrible"]},
            },
        }
        req = urllib.request.Request(
            a.url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        responses.append(json.load(urllib.request.urlopen(req, timeout=120)))
    ok, detail = gate_c5(json.load(open(a.snapshot)), responses)
    print(json.dumps({"C5": {"pass": ok, **detail}}, indent=1))
    return 0 if ok else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "c5":
        sys.exit(c5_main(sys.argv[2:]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--block-size", type=int, required=True, help="the server's padding unit; 0 for no padding")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.items))[: a.n]
    res, ok = {"url": a.url, "items": len(items)}, True
    for name, fn in (
        ("C2", lambda: gate_c2(a.url)),
        ("C3", lambda: gate_c3(items, a.tokenizer, a.block_size, server_rendering(a.url), *server_format(a.url))),
        ("C4", lambda: gate_c4(a.url, items)),
    ):
        g_ok, detail = fn()
        res[name] = {"pass": g_ok, "detail": detail}
        ok &= g_ok
        summary = (
            {k: v for k, v in detail.items() if k != "rows"}
            if isinstance(detail, dict)
            else [f"{x['request']}: {'ok' if x['ok'] else 'FAIL ' + x.get('error', '')}" for x in detail]
        )
        print(f"{'ok  ' if g_ok else 'FAIL'} {name} {json.dumps(summary)[:1500]}", flush=True)
    res["pass"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    print("CONFORMANCE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
